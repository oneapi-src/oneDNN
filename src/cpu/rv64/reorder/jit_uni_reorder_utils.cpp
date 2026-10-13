/*******************************************************************************
* Copyright 2018 Intel Corporation
* Copyright 2020-2023 FUJITSU LIMITED
* Copyright 2022-2025 Arm Ltd. and affiliates
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <cassert>
#include <climits>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/primitive_attr.hpp"
#include "common/reorder_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

#include "cpu/platform.hpp"

#include "cpu/rv64/reorder/jit_uni_reorder_utils.hpp"

#if defined(DNNL_DEV_MODE)
#define DEBUg(...) \
    do { \
        if (get_verbose(verbose_t::debuginfo) >= 5) { __VA_ARGS__ } \
    } while (0)
#else
#define DEBUg(...)
#endif
#define DEBUG(...) DEBUg(__VA_ARGS__)

using namespace dnnl::impl::types;
using namespace dnnl::impl::status;

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

namespace tr {

bool prb_has_small_strides(const prb_t &prb) {
    constexpr ptrdiff_t max_stride = (1LL << 31) - 1;
    for (int d = 0; d < prb.ndims; ++d) {
        const ptrdiff_t cms = max_stride / prb.nodes[d].n;
        const bool small_strides = true
                && prb.nodes[d].is < cms / (int)data_type_size(prb.itype)
                && prb.nodes[d].os < cms / (int)data_type_size(prb.otype);
        if (!small_strides) return false;
    }
    return true;
}

bool prb_has_huge_prime_number(const prb_t &prb) {
    for (int d = 0; d < prb.ndims; ++d) {
        auto n = prb.nodes[d].n;
        if (n >= INT_MAX && math::is_prime(n)) return true;
    }
    return false;
}

bool prb_is_f32_default_plain_blocked_reorder(
        const prb_t &p, plain_blocked_reorder_desc_t *desc) {
    if (p.ndims < 2 || p.itype != data_type::f32 || p.otype != data_type::f32
            || p.src_scale_type != scale_type_t::NONE
            || p.dst_scale_type != scale_type_t::NONE || p.ioff != 0
            || p.ooff != 0 || p.beta != 0.f || p.req_src_zp || p.req_dst_zp
            || p.req_s8s8_comp || p.req_asymmetric_comp
            || !prb_has_small_strides(p))
        return false;

    auto is_supported_block
            = [](size_t n) { return utils::one_of(n, 4ul, 8ul, 16ul, 32ul); };

    plain_blocked_reorder_desc_t candidates[2];
    int ncandidates = 0;

    auto try_nodes = [&](int block_node_id, int inner_node_id,
                             plain_blocked_reorder_desc_t &candidate) {
        const auto &block_node = p.nodes[block_node_id];
        const auto &inner_node = p.nodes[inner_node_id];
        if (!is_supported_block(block_node.n)) return false;
        if (block_node.tail_size > block_node.n) return false;

        const bool plain_to_blocked = block_node.os == 1 && inner_node.is == 1
                && block_node.is == static_cast<ptrdiff_t>(inner_node.n)
                && inner_node.os == static_cast<ptrdiff_t>(block_node.n);
        const bool blocked_to_plain = block_node.is == 1 && inner_node.os == 1
                && block_node.os == static_cast<ptrdiff_t>(inner_node.n)
                && inner_node.is == static_cast<ptrdiff_t>(block_node.n);
        if (!plain_to_blocked && !blocked_to_plain) return false;

        for (int d = 0; d < p.ndims; ++d) {
            if (p.nodes[d].tail_size > 0 && d != block_node_id) return false;
        }

        if (!block_node.is_dim_id_empty()) {
            int same_dim_nodes = 0;
            for (int d = 0; d < p.ndims; ++d)
                if (p.nodes[d].dim_id == block_node.dim_id) ++same_dim_nodes;
            if (same_dim_nodes > 2) return false;
        }

        candidate.block_node_id = block_node_id;
        candidate.inner_node_id = inner_node_id;
        candidate.block = block_node.n;
        candidate.plain_to_blocked = plain_to_blocked;
        return true;
    };

    plain_blocked_reorder_desc_t candidate;
    if (try_nodes(0, 1, candidate)) candidates[ncandidates++] = candidate;
    if (try_nodes(1, 0, candidate)) candidates[ncandidates++] = candidate;
    if (ncandidates == 0) return false;

    int selected = -1;
    for (int i = 0; i < ncandidates; ++i) {
        if (p.nodes[candidates[i].block_node_id].dim_id == 1) {
            selected = i;
            break;
        }
    }
    if (selected == -1 && ncandidates == 1
            && p.nodes[candidates[0].block_node_id].is_dim_id_empty())
        selected = 0;
    if (selected == -1) return false;

    if (desc) *desc = candidates[selected];
    return true;
}

/** ad-hoc structure to describe blocked memory layout */
struct layout_desc_t {
    layout_desc_t()
        : dt(dnnl_data_type_undef)
        , ndims(0)
        , id {-1}
        , dims {0}
        , tails {0}
        , is_blk {false}
        , strides {0} {}
    data_type_t dt;
    int ndims;
    dims_t id;
    dims_t dims;
    dims_t tails;
    bool is_blk[DNNL_MAX_NDIMS];
    strides_t strides;
};

status_t cvt_mem_desc_to_layout_desc(const memory_desc_t &md_,
        layout_desc_t &ld, const dims_t &blocks, const dims_t &external_padding,
        const dims_t &tails) {
    static constexpr bool it_is_blk = true;

    const auto md = memory_desc_wrapper(md_);

    if (!md.is_blocking_desc()) return invalid_arguments;

    const auto &bd = md.blocking_desc();

    ld.ndims = 0;
    ld.dt = md.data_type();

    auto add_dim = [&ld](int id, dim_t dim, dim_t tail, bool is_blk,
                           ptrdiff_t stride) {
        assert((size_t)ld.ndims < sizeof(ld.dims) / sizeof(ld.dims[0]));
        ld.id[ld.ndims] = id;
        ld.dims[ld.ndims] = dim;
        ld.strides[ld.ndims] = stride;
        ld.tails[ld.ndims] = tail;
        ld.is_blk[ld.ndims] = is_blk;
        ++ld.ndims;
    };

    for (int d = 0; d < md.ndims(); ++d) {
        const int ld_ndims_start = ld.ndims;
        if (blocks[d] != 1) {
            stride_t stride = 1;
            dim_t tail = tails[d];
            for (int iblk = bd.inner_nblks - 1; iblk >= 0; --iblk) {
                if (bd.inner_idxs[iblk] == d) {
                    const dim_t inner_tail = tail % bd.inner_blks[iblk];
                    add_dim(d, bd.inner_blks[iblk], inner_tail, it_is_blk,
                            stride);
                    tail = utils::div_up(tail, bd.inner_blks[iblk]);
                }
                stride *= bd.inner_blks[iblk];
            }
        }

        const dim_t dim_with_external_padding
                = (md.padded_dims()[d] + external_padding[d]) / blocks[d];
        const dim_t padded_dim = md.padded_dims()[d] / blocks[d];
        const dim_t tail = dim_with_external_padding != padded_dim
                ? dim_with_external_padding
                        - (dim_with_external_padding - padded_dim)
                : 0;

        add_dim(d, dim_with_external_padding, tail, !it_is_blk, bd.strides[d]);

        // reverse the order of dims
        for (int ld_d = 0; ld_d < (ld.ndims - ld_ndims_start) / 2; ++ld_d) {
            const int idx0 = ld_ndims_start + ld_d;
            const int idx1 = ld.ndims - 1 - ld_d;
            nstl::swap(ld.dims[idx0], ld.dims[idx1]);
            nstl::swap(ld.strides[idx0], ld.strides[idx1]);
            nstl::swap(ld.tails[idx0], ld.tails[idx1]);
            nstl::swap(ld.is_blk[idx0], ld.is_blk[idx1]);
        }
    }

    return success;
}

static bool is_with_groups(const memory_desc_t &dst_md) {
    using namespace memory_extra_flags;
    auto dst_d = memory_desc_wrapper(dst_md);
    const int grp_bit = 1 << 1;
    auto check_flag_and_mask = [&](int flag, int mask) {
        return (dst_d.extra().flags & flag) && (mask & grp_bit);
    };

    return check_flag_and_mask(
                   compensation_conv_s8s8, dst_d.extra().compensation_mask)
            || check_flag_and_mask(compensation_conv_asymmetric_src,
                    dst_d.extra().asymm_compensation_mask);
}

static inline int get_next_parent_node(node_t *nodes, int ndims, int cur_node) {
    const int cur_id = nodes[cur_node].dim_id;
    for (int d = cur_node + 1; d < ndims; ++d) {
        if (nodes[d].dim_id == cur_id) return d;
    }
    return -1;
}

static void prb_set_compensation_strides(prb_t &p) {

    auto require_n_stride = [&](int cur_node) -> bool {
        const int parent = get_next_parent_node(p.nodes, p.ndims, cur_node);
        if (parent < 0) return false;

        const size_t p_n = p.nodes[parent].n;

        // if 'parent_node.n' is larger than 1, then cur_node stride
        // is 'cur_node.n'
        return p_n > size_t(1);
    };

    const auto compensation_needed = p.req_s8s8_comp || p.req_asymmetric_comp;
    if (!compensation_needed) return;
    int mask = p.compensation_mask;
    ptrdiff_t cs = 1;
    for (int d = 0; d < p.ndims; ++d) {
        if (mask & (1 << p.nodes[d].dim_id)) {

            // correct cases when 'cs' exceeds output stride
            if (cs > p.nodes[d].os) cs = p.nodes[d].os;

            p.nodes[d].cs = cs;
            const bool n_stride = require_n_stride(d);
            if (p.nodes[d].tail_size > 0 && (!p.nodes[d].is_zero_pad_needed)
                    && (!n_stride))
                cs *= p.nodes[d].tail_size;
            else
                cs *= p.nodes[d].n;
        }
    }
}

status_t prb_init(prb_t &p, const memory_desc_t &imd, const memory_desc_t &omd,
        const primitive_attr_t *attr) {
    auto im_d = memory_desc_wrapper(imd);
    auto om_d = memory_desc_wrapper(omd);

    auto check_post_ops = [](const primitive_attr_t *attr) {
        const auto &po = attr->post_ops_;
        return po.len() == 0 || (po.len() == 1 && po.entry_[0].is_sum(false));
    };

    VDISPATCH_REORDER_IC(
            im_d.is_blocking_desc(), VERBOSE_UNSUPPORTED_FORMAT_KIND);
    VDISPATCH_REORDER_IC(
            om_d.is_blocking_desc(), VERBOSE_UNSUPPORTED_FORMAT_KIND);
    VDISPATCH_REORDER_IC(!im_d.has_zero_dim(), VERBOSE_EMPTY_TENSOR, "src");
    VDISPATCH_REORDER_IC(!om_d.has_zero_dim(), VERBOSE_EMPTY_TENSOR, "dst");
    VDISPATCH_REORDER_IC(!im_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_REORDER_IC(!om_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    using smask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER_IC(attr->has_default_values(smask_t::scales
                                 | smask_t::zero_points | smask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_REORDER_IC(check_post_ops(attr), VERBOSE_UNSUPPORTED_POSTOP);

    bool is_tail_present = false;
    dims_t iblocks, oblocks, i_tails, o_tails, i_paddings, o_paddings;
    im_d.compute_blocks(iblocks);
    om_d.compute_blocks(oblocks);

    for (int d = 0; d < om_d.ndims(); ++d) {
        const auto dim = om_d.dims()[d];
        const auto pdim = om_d.padded_dims()[d];
        const auto cblock = oblocks[d];
        // do not allow excess pdim other than required for rounding-up of dim.
        VDISPATCH_REORDER_IC(utils::rnd_up(dim, cblock) == pdim,
                VERBOSE_UNSUPPORTED_PAD_FEATURE, "dst");
    }

    utils::array_set(i_tails, 0, im_d.ndims());
    utils::array_set(o_tails, 0, om_d.ndims());
    utils::array_set(i_paddings, 0, im_d.ndims());
    utils::array_set(o_paddings, 0, om_d.ndims());

    for (int d = 0; d < im_d.ndims(); ++d) {
        const dim_t i_dim = im_d.dims()[d];
        const dim_t o_dim = om_d.dims()[d];
        const dim_t i_tail = i_dim % iblocks[d];
        const dim_t o_tail = o_dim % oblocks[d];

        if (o_tail > 0) {
            is_tail_present = true;
            o_tails[d] = o_tail;
            o_paddings[d] = oblocks[d] - o_tail;
        }

        if (i_tail > 0) {
            is_tail_present = true;
            i_tails[d] = i_tail;
            i_paddings[d] = iblocks[d] - i_tail;
        }
    }

    // To compute input layout description we need to pass output paddings
    // which will be used to compute input dims rounded up to multiple of
    // output dims. Analogous applies to output layout description.
    // This is demanded by the algorithm of nodes creation.
    // Example:
    // input:
    //  format: abc
    //  size: 77, 15, 3
    //  o_padding: 3, 17, 0
    //  returns ild: 80, 32, 3
    // output:
    //  format: ABc16b16a2b
    //  size: 77, 15, 3
    //  i_padding: 0, 0, 0
    //  returns old: 5, 16, 1, 16, 2, 3
    layout_desc_t ild, old;
    CHECK(cvt_mem_desc_to_layout_desc(imd, ild, iblocks, o_paddings, i_tails));
    CHECK(cvt_mem_desc_to_layout_desc(omd, old, oblocks, i_paddings, o_tails));

    p.itype = ild.dt;
    p.otype = old.dt;
    p.is_tail_present = is_tail_present;
    p.req_src_zp = !attr->zero_points_.has_default_values(DNNL_ARG_SRC);
    p.req_dst_zp = !attr->zero_points_.has_default_values(DNNL_ARG_DST);

    p.src_scale_type = scale_type_t::NONE;
    int src_mask = 0;
    if (!attr->scales_.has_default_values(DNNL_ARG_SRC)) {
        src_mask = attr->scales_.get_mask(DNNL_ARG_SRC);
        p.src_scale_type
                = src_mask == 0 ? scale_type_t::COMMON : scale_type_t::MANY;
    }

    p.dst_scale_type = scale_type_t::NONE;
    int dst_mask = 0;
    if (!attr->scales_.has_default_values(DNNL_ARG_DST)) {
        dst_mask = attr->scales_.get_mask(DNNL_ARG_DST);
        p.dst_scale_type
                = dst_mask == 0 ? scale_type_t::COMMON : scale_type_t::MANY;
    }

    VDISPATCH_REORDER_IC(
            IMPLICATION(p.src_scale_type != scale_type_t::NONE
                            && p.dst_scale_type != scale_type_t::NONE,
                    src_mask == dst_mask),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    // Reject destinations that carry extra flags this kernel cannot produce
    // (e.g. RNN compensation buffers); otherwise they would be silently
    // ignored and leave the appended buffer uninitialized.
    const uint64_t supported_extra_flags
            = memory_extra_flags::compensation_conv_s8s8
            | memory_extra_flags::compensation_conv_asymmetric_src
            | memory_extra_flags::scale_adjust;
    VDISPATCH_REORDER_IC(!(om_d.extra().flags & ~supported_extra_flags),
            VERBOSE_UNSUPPORTED_MD_FLAG, "dst");

    p.scale_adjust = (om_d.extra().flags & memory_extra_flags::scale_adjust)
            ? om_d.extra().scale_adjust
            : 1.f;
    p.req_s8s8_comp
            = om_d.extra().flags & memory_extra_flags::compensation_conv_s8s8;
    p.req_asymmetric_comp = om_d.extra().flags
            & memory_extra_flags::compensation_conv_asymmetric_src;

    const bool with_groups = is_with_groups(omd);

    auto mask_ok = [&](bool check, int mask) {
        return IMPLICATION(check, mask == (with_groups ? 0x3 : 0x1));
    };

    VDISPATCH_REORDER_IC(
            mask_ok(p.req_s8s8_comp, om_d.extra().compensation_mask),
            VERBOSE_UNSUPPORTED_MD_FLAG, "dst");
    VDISPATCH_REORDER_IC(mask_ok(p.req_asymmetric_comp,
                                 om_d.extra().asymm_compensation_mask),
            VERBOSE_UNSUPPORTED_MD_FLAG, "dst");

    ptrdiff_t ss[max_ndims] = {0}; // scales strides
    if (p.src_scale_type == scale_type_t::MANY
            || p.dst_scale_type == scale_type_t::MANY) {
        const int mask = nstl::max(src_mask, dst_mask);
        ptrdiff_t dense_stride = 1;
        ptrdiff_t last_stride = 1;
        for (int d = old.ndims - 1; d >= 0; --d) {
            assert((d == 0 || old.id[d - 1] <= old.id[d])
                    && "logical dimensions should be in ascending order");
            if (mask & (1 << old.id[d])) {
                if ((d + 1) < old.ndims && old.id[d + 1] != old.id[d]
                        && (mask & (1 << old.id[d + 1]))) {
                    dense_stride = dense_stride * imd.dims[old.id[d + 1]];
                    last_stride = dense_stride;
                }
                ss[d] = last_stride;
                last_stride *= old.dims[d];
            }
        }
    }

    const auto compensation_needed = p.req_s8s8_comp || p.req_asymmetric_comp;
    if (compensation_needed) {
        p.compensation_mask = p.req_s8s8_comp
                ? om_d.extra().compensation_mask
                : (p.req_asymmetric_comp ? om_d.extra().asymm_compensation_mask
                                         : tr::prb_t::invalid_comp_mask);

        if (p.compensation_mask == tr::prb_t::asymmetric_comp_mask)
            return unimplemented;

        assert(p.compensation_mask == tr::prb_t::standard_comp_mask
                || p.compensation_mask == tr::prb_t::comp_mask_with_groups);
    }

    int ndims = 0;

    int i_pos = 0; /* state for input  -- current dimension */
    int o_pos = 0; /* state for output -- current dimension */

    while (i_pos < ild.ndims && o_pos < old.ndims) {
        assert(ild.id[i_pos] == old.id[o_pos]);

        assert(ndims < max_ndims);
        if (ndims == max_ndims) return runtime_error;

        if (ild.dims[i_pos] == old.dims[o_pos]) {
            p.nodes[ndims].n = ild.dims[i_pos];
            p.nodes[ndims].dim_id = old.id[o_pos];
            p.nodes[ndims].tail_size = old.tails[o_pos];
            p.nodes[ndims].is_zero_pad_needed
                    = old.is_blk[o_pos] && old.tails[o_pos] > 0;
            p.nodes[ndims].is = ild.strides[i_pos];
            p.nodes[ndims].os = old.strides[o_pos];
            p.nodes[ndims].ss = ss[o_pos];
            ++ndims;
            ++i_pos;
            ++o_pos;
        } else if (ild.dims[i_pos] < old.dims[o_pos]) {
            // old must be divisible by ild or we will not be
            // able to create valid nodes. The problem appears
            // when stag=Acdb48a and dtag=Acdb32a for example.
            if (ild.dims[i_pos] == 0 || old.dims[o_pos] % ild.dims[i_pos] != 0)
                return status::unimplemented;

            dim_t factor = old.dims[o_pos] / ild.dims[i_pos];

            const size_t tail_of_upper_dim
                    = utils::div_up(old.tails[o_pos], factor) == ild.dims[i_pos]
                    ? 0
                    : utils::div_up(old.tails[o_pos], factor);
            const size_t tail_of_lower_dim = old.tails[o_pos] % factor;

            p.nodes[ndims].n = ild.dims[i_pos];
            p.nodes[ndims].dim_id = old.id[o_pos];
            p.nodes[ndims].tail_size = tail_of_upper_dim;
            p.nodes[ndims].is_zero_pad_needed
                    = old.is_blk[o_pos] && tail_of_upper_dim > 0;
            p.nodes[ndims].is = ild.strides[i_pos];
            p.nodes[ndims].os = old.strides[o_pos] * factor;
            p.nodes[ndims].ss = ss[o_pos] * factor;
            ++ndims;
            ++i_pos;
            old.dims[o_pos] = factor;
            old.tails[o_pos] = tail_of_lower_dim;
        } else if (ild.dims[i_pos] > old.dims[o_pos]) {
            // ild must be divisible by old or we will not be
            // able to create valid nodes. The problem appears
            // when stag=Acdb32a and dtag=Acdb48a for example.
            if (old.dims[o_pos] == 0 || ild.dims[i_pos] % old.dims[o_pos] != 0)
                return status::unimplemented;

            dim_t factor = ild.dims[i_pos] / old.dims[o_pos];
            p.nodes[ndims].n = old.dims[o_pos];
            p.nodes[ndims].dim_id = old.id[o_pos];
            p.nodes[ndims].tail_size = old.tails[o_pos];
            p.nodes[ndims].is_zero_pad_needed
                    = old.is_blk[o_pos] && old.tails[o_pos] > 0;
            p.nodes[ndims].is = ild.strides[i_pos] * factor;
            p.nodes[ndims].os = old.strides[o_pos];
            p.nodes[ndims].ss = ss[o_pos];
            ++ndims;
            ++o_pos;
            ild.dims[i_pos] = factor;
        }
    }

    p.ndims = ndims;
    p.full_ndims = ndims;

    p.ioff = memory_desc_wrapper(imd).offset0();
    p.ooff = memory_desc_wrapper(omd).offset0();

    const int sum_idx = attr->post_ops_.find(primitive_kind::sum);
    p.beta = sum_idx == -1 ? 0.f : attr->post_ops_.entry_[sum_idx].sum.scale;

    DEBUG({
        verbose_printf(
                verbose_t::debuginfo, "init : %s\n", prb_dump(p).c_str());
    });

    prb_normalize(p);
    DEBUG({
        verbose_printf(
                verbose_t::debuginfo, "norm : %s\n", prb_dump(p).c_str());
    });

    // compensation strides require prb_normalized
    prb_set_compensation_strides(p);

    prb_simplify(p);
    DEBUG({
        verbose_printf(
                verbose_t::debuginfo, "smpl : %s\n", prb_dump(p).c_str());
    });

    return success;
}

// Sorts the prb array in increasing sizes of the output stride.
void prb_normalize(prb_t &p) {
    for (int d = 0; d < p.ndims; ++d) {
        int min_pos = d;
        for (int j = d + 1; j < p.ndims; ++j) {
            bool new_min = p.nodes[j].os < p.nodes[min_pos].os
                    || (p.nodes[j].os == p.nodes[min_pos].os
                            && p.nodes[j].n < p.nodes[min_pos].n);
            if (new_min) min_pos = j;
        }
        if (min_pos != d) { nstl::swap(p.nodes[d], p.nodes[min_pos]); }
    }
}

void prb_node_dependency(prb_t &prb) {
    for (int i = 0; i < prb.ndims; i++) {
        tr::node_t &node = prb.nodes[i];
        node.parent_node_id = node_t::empty_field;
        for (int j = i + 1; j < prb.ndims; j++) {
            const tr::node_t &potential_parent_node = prb.nodes[j];
            if (!potential_parent_node.is_dim_id_empty()
                    && potential_parent_node.dim_id == node.dim_id) {
                node.parent_node_id = j;
                break;
            }
        }
    }
}

// Combines variables which appear together on both sides of the reorder.
void prb_simplify(prb_t &p) {
#if defined(__GNUC__) && __GNUC__ >= 4
/* GCC produces bogus array subscript is above array bounds warning for
 * the `p.nodes[j - 1] = p.nodes[j]` line below, so disable it for now. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

    const auto skip_dim_combining = [&p](const int node_id) -> bool {
        return (p.is_tail_in_one_of_child_nodes(node_id)
                       && p.nodes[node_id].n > 1)
                || p.nodes[node_id].tail_size > 0;
    };

    if (p.is_tail_present) prb_node_dependency(p);

    for (int d = 0; d < p.ndims - 1; ++d) {
        auto &this_node = p.nodes[d + 0];
        auto &next_node = p.nodes[d + 1];
        const bool skip_dims_combining
                = skip_dim_combining(d) || skip_dim_combining(d + 1);
        if (skip_dims_combining) continue;

        const bool trivial_fold = next_node.n == static_cast<size_t>(1);
        const bool real_fold = next_node.is
                        == static_cast<ptrdiff_t>(this_node.n * this_node.is)
                && next_node.os
                        == static_cast<ptrdiff_t>(this_node.n * this_node.os)
                && next_node.ss
                        == static_cast<ptrdiff_t>(this_node.n * this_node.ss)
                && next_node.cs
                        == static_cast<ptrdiff_t>(this_node.n * this_node.cs);
        if (trivial_fold || real_fold) {
            this_node.n *= next_node.n;
            this_node.dim_id = node_t::empty_field;
            this_node.is_zero_pad_needed = false;
            for (int j = d + 2; j < p.ndims; ++j)
                p.nodes[j - 1] = p.nodes[j];
            --p.ndims;
            --p.full_ndims;
            --d; // make another try
            if (p.is_tail_present) prb_node_dependency(p);
        }
    }
#if defined(__GNUC__) && __GNUC__ >= 4
#pragma GCC diagnostic pop
#endif
}

void prb_node_split(prb_t &p, int dim, size_t new_node_size) {
    assert(dim < p.ndims);
    assert(p.ndims < max_ndims);
    assert(p.nodes[dim].n % new_node_size == 0);

    p.ndims += 1;
    p.full_ndims += 1;

    for (int d = p.ndims; d > dim + 1; --d)
        p.nodes[d] = p.nodes[d - 1];

    const size_t upper_node_size = p.nodes[dim].n / new_node_size;
    const size_t lower_node_size = new_node_size;
    p.nodes[dim + 1].n = upper_node_size;
    p.nodes[dim].n = lower_node_size;

    const bool is_tail = p.nodes[dim].tail_size > 0;
    const size_t upper_node_tail
            = utils::div_up(p.nodes[dim].tail_size, lower_node_size)
                    == upper_node_size
            ? 0
            : utils::div_up(p.nodes[dim].tail_size, lower_node_size);
    const size_t lower_node_tail = p.nodes[dim].tail_size % lower_node_size;
    p.nodes[dim].tail_size = is_tail ? lower_node_tail : 0;
    p.nodes[dim + 1].tail_size = is_tail ? upper_node_tail : 0;

    p.nodes[dim + 1].is_zero_pad_needed
            = p.nodes[dim].is_zero_pad_needed && p.nodes[dim + 1].tail_size > 0;
    p.nodes[dim].is_zero_pad_needed
            = p.nodes[dim].is_zero_pad_needed && p.nodes[dim].tail_size > 0;

    p.nodes[dim + 1].dim_id = p.nodes[dim].dim_id;
    p.nodes[dim + 1].is = p.nodes[dim].is * lower_node_size;
    p.nodes[dim + 1].os = p.nodes[dim].os * lower_node_size;
    p.nodes[dim + 1].ss = p.nodes[dim].ss * lower_node_size;
    p.nodes[dim + 1].cs = p.nodes[dim].cs * lower_node_size;
}

void prb_node_swap(prb_t &p, int d0, int d1) {
    assert(d0 < p.ndims);
    assert(d1 < p.ndims);
    assert(p.ndims < max_ndims);

    if (d0 == d1) return;

    nstl::swap(p.nodes[d0], p.nodes[d1]);
}

void prb_node_move(prb_t &p, int d0, int d1) {
    assert(d0 < p.ndims);
    assert(d1 < p.ndims);
    assert(p.ndims < max_ndims);

    if (d0 == d1) return;

    node_t node = p.nodes[d0];

    if (d0 < d1)
        for (int d = d0; d < d1; ++d)
            p.nodes[d] = p.nodes[d + 1];
    else
        for (int d = d0; d > d1; --d)
            p.nodes[d] = p.nodes[d - 1];

    p.nodes[d1] = node;
}

std::string prb_dump(const prb_t &p) {
    stringstream_t ss;
    ss << "@@@ type:" << dnnl_dt2str(p.itype) << ':' << dnnl_dt2str(p.otype)
       << " ndims:" << p.ndims;
    for (int d = 0; d < p.ndims; ++d) {
        if (d != 0) ss << 'x';
        const auto &node = p.nodes[d];
        ss << '[' << node.n << ':' << node.tail_size << ':' << node.dim_id
           << ':' << node.parent_node_id << ':'
           << (node.is_zero_pad_needed ? "true" : "false") << ':' << node.is
           << ':' << node.os << ':' << node.ss << ':' << node.cs << ']';
    }
    ss << " off:" << p.ioff << ':' << p.ooff;
    return ss.str();
}

void prb_block_for_cache(prb_t &prb) {
    /* If strides for 0th and 1st nodes are cache friendly
     * then one can altogether do away with blocking ! */
    static constexpr int num_elems_thr = 16;
    const bool stride_cache_friendly
            = ((prb.nodes[0].is % 64 == 0 && prb.nodes[0].n > num_elems_thr)
                      || (prb.ndims > 1 && prb.nodes[1].is % num_elems_thr == 0
                              && prb.nodes[1].n > num_elems_thr))
            && !prb.is_tail_present;

    // performance improvement for shapes with large inner-most dimension
    const size_t L1_cache_sz
            = size_t(3) * platform::get_per_core_cache_size(1) / 4;
    const size_t itype_sz_ = data_type_size(prb.itype);
    const size_t inner_block_sz = prb.nodes[0].n * itype_sz_;
    const bool requires_inner_blocking = inner_block_sz > L1_cache_sz
            // 'is_tail_present' is not supported for cache_blocking when
            // asymmetric_comp is executed.
            && IMPLICATION(prb.req_asymmetric_comp, !prb.is_tail_present);

    const bool cache_blocking_needed
            = stride_cache_friendly || requires_inner_blocking;
    if (!cache_blocking_needed) return;

    int unit_input_stride_idx = -1;
    for (auto idx = 0; idx < prb.ndims; ++idx) {
        if (prb.nodes[idx].is == 1) unit_input_stride_idx = idx;
    }

    /* Re-prioritize the sequential read over sequential write:
     *                             /-> [n0:is0:1][16n1:1:osk]...
     * [n0:is0:1]...[nk:1:osk] -->     or
     *                             \-> [16n1:1:osk][n0:is0:1]... */
    if (unit_input_stride_idx != -1) {
        const auto output_stride = prb.nodes[unit_input_stride_idx].os;
        const auto num_elems = prb.nodes[unit_input_stride_idx].n;

        const bool split_needed = (num_elems > num_elems_thr)
                && (num_elems % num_elems_thr == 0);
        const int move_location = (output_stride % 4 != 0) ? 0 : 1;
        if (split_needed)
            prb_node_split(prb, unit_input_stride_idx, num_elems_thr);

        /* Because of cache-unfriendly nature of unit-output stride node, let
         * us move unit-input stride node on or near front! */
        if (unit_input_stride_idx != move_location)
            prb_node_move(prb, unit_input_stride_idx, move_location);
    }

    /* Potentially, split the node with os=1 in two and pull in the node with
     * is=1 between them for better cache reuse:
     * [n0:is0:1][n1:1:os1] --> [16n0:is0:1][n1:1:os1][n0/16:is0*16:16] */
    if (prb.ndims >= 2 && prb.nodes[0].os == 1 && prb.nodes[1].is == 1) {
        const auto num_elems = prb.nodes[0].n;

        const bool split_needed = (num_elems > num_elems_thr)
                && (num_elems % num_elems_thr == 0);
        if (split_needed) {
            prb_node_split(prb, 0, num_elems_thr);
            prb_node_move(prb, 1, 2);

            // Update node information
            prb_node_dependency(prb);

            // heuristics - looping over the unrolled dims should maximize reuse
            // of the already cached data; observation is choosing the smallest
            // dim from the remaining (from 2 up to ndims) gives good results
            constexpr int new_position = 2;
            const auto dim_beg_it = std::begin(prb.nodes);
            const auto dim_two_it = dim_beg_it + new_position;
            const auto dim_last_it = dim_beg_it + prb.ndims;
            const auto min_n_node_it = std::min_element(dim_two_it, dim_last_it,
                    [](const tr::node_t &lhs, const tr::node_t &rhs) {
                return lhs.n < rhs.n;
            });
            const auto min_idx = std::distance(dim_beg_it, min_n_node_it);
            // check if min_idx node is parent of node with tail processing which
            // is currently unsupported (i.e. tail processing can only be handled
            // at the inner-most dimension)
            bool inner_block_has_tail = false;
            for (int idx = min_idx - 1; idx >= new_position; idx--) {
                if (prb.nodes[idx].parent_node_id == min_idx) {
                    inner_block_has_tail = true;
                    break;
                }
            }

            if (min_idx > new_position && (!inner_block_has_tail))
                prb_node_move(prb, min_idx, new_position);
        }
    }
}

void prb_thread_kernel_balance(prb_t &prb, int &ndims_ker_max, int nthr) {
    size_t size_total = 1;
    for (int d = 0; d < prb.ndims; ++d)
        size_total *= prb.nodes[d].n;

    // The general expression for size_drv_thr can be written as
    // size_drv_min = C0 + FC * (nthr > 1 ? 1 : 0) + VC * (nthr - 1)
    // where FC and VC are fixed and variable costs respectively.
    // Though for now, the below heuristic seems to be good enough
    const size_t size_drv_thr = (nthr > 1) ? 16 * nthr : 1;

    /* size_drv_min is the minimal size for the parallel
     * driver required for good parallelization */
    const size_t size_drv_min
            = nstl::min<size_t>(size_drv_thr, utils::div_up(size_total, 1024));

    /* kdims -- # of dimensions processed by a kernel
     * size_ker_cur -- product of the dimension processed by a kernel
     * size_drv_cur -- product of the dimension processed by a driver */

    int kdims = prb.ndims;
    size_t size_drv_cur = 1;
    for (; kdims > 1 && size_drv_cur < size_drv_min; --kdims)
        size_drv_cur *= prb.nodes[kdims - 1].n;

    size_t size_ker_cur = 1;
    for (int d = 0; d < kdims; ++d)
        size_ker_cur *= prb.nodes[d].n;

    /* Initially kdims is chosen so that size_drv_cur >= size_drv_min.
     *
     * It might happen that for chosen kdims the size_ker_cur is too small
     * (less than tr::ker_prb_size_min). In that case try to split the
     * innermost driver dimension into two, to increase size_ker_cur. */
    const bool want_borrow_ker_from_drv = kdims < prb.ndims
            && size_ker_cur < tr::ker_prb_size_min
            && size_drv_cur > size_drv_min;
    if (want_borrow_ker_from_drv) {
        /* size_want_borrow is the minimal size, so that:
         *  o) size_ker_cur * size_want_borrow >= tr::ker_prb_size_min
         *  o) current innermost driver dimension is divisible by
         *     size_want_borrow (so that we can evenly split that
         *     dimension into two)
         *
         *  In the worst case the minimal size_want_borrow is equal
         *  to the innermost driver dimension itself. In that case
         *  we will sacrifice it in favor of kernel (is it fine?). */
        size_t size_want_borrow
                = utils::div_up(tr::ker_prb_size_min, size_ker_cur);
        for (; prb.nodes[kdims].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims].n)
            prb_node_split(prb, kdims, size_want_borrow);
        kdims += 1;
    }

    /* On the other hand it might happen that for chosen kdims
     * the size_drv_cur is too small (less than size_drv_min). In that case
     * try to split the outermost kernel dimension into two, to increase
     * size_drv_cur. */
    const bool want_borrow_drv_from_ker = size_ker_cur > tr::ker_prb_size_min
            && size_drv_cur < size_drv_min;

    if (want_borrow_drv_from_ker) {
        size_t size_want_borrow = utils::div_up(size_drv_min, size_drv_cur);
        for (; prb.nodes[kdims - 1].n % size_want_borrow; ++size_want_borrow)
            ;

        if (size_want_borrow != prb.nodes[kdims - 1].n)
            prb_node_split(
                    prb, kdims - 1, prb.nodes[kdims - 1].n / size_want_borrow);
    }

    ndims_ker_max = kdims;

    if (want_borrow_ker_from_drv || want_borrow_drv_from_ker) {
        DEBUG({
            verbose_printf(
                    verbose_t::debuginfo, "split: %s\n", prb_dump(prb).c_str());
            verbose_printf(verbose_t::debuginfo, "ndims_ker_max = %d\n",
                    ndims_ker_max);
        });
    }
}

} // namespace tr

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

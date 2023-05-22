/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#include <cassert>
#include <set>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

#include "cpu/x64/jit_uni_reorder.hpp"

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
namespace x64 {

namespace tr {

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
            int tail = tails[d];
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

        // TODO: NOW: revisit, do we need a reverse?
        // TODO: NOW: consider using strides instead of block sizes in md
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

    bool ok = im_d.is_blocking_desc() && om_d.is_blocking_desc()
            && !im_d.has_runtime_dims_or_strides() && !im_d.has_zero_dim()
            && !om_d.has_runtime_dims_or_strides() && !om_d.has_zero_dim()
            && attr->has_default_values(
                    primitive_attr_t::skip_mask_t::scales_runtime
                    | primitive_attr_t::skip_mask_t::zero_points_runtime
                    | primitive_attr_t::skip_mask_t::post_ops)
            && check_post_ops(attr);
    if (!ok) return unimplemented;

    bool is_tail_present = false;
    dims_t iblocks, oblocks, i_tails, o_tails, i_paddings, o_paddings;
    im_d.compute_blocks(iblocks);
    om_d.compute_blocks(oblocks);

    for (int d = 0; d < om_d.ndims(); ++d) {
        const auto dim = om_d.dims()[d];
        const auto pdim = om_d.padded_dims()[d];
        const auto cblock = oblocks[d];
        // do not allow excess pdim other than required for rounding-up of dim.
        if (utils::rnd_up(dim, cblock) != pdim) return unimplemented;
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
    bool is_src_set = false;
    CHECK(attr->scales_.get(DNNL_ARG_SRC, &src_mask, &is_src_set));
    if (is_src_set) {
        p.src_scale_type
                = src_mask == 0 ? scale_type_t::COMMON : scale_type_t::MANY;
    }

    p.dst_scale_type = scale_type_t::NONE;
    int dst_mask = 0;
    bool is_dst_set = false;
    CHECK(attr->scales_.get(DNNL_ARG_DST, &dst_mask, &is_dst_set));
    if (is_dst_set) {
        p.dst_scale_type
                = dst_mask == 0 ? scale_type_t::COMMON : scale_type_t::MANY;
    }

    if (is_src_set && is_dst_set && src_mask != dst_mask)
        return status::unimplemented;

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

    if (!mask_ok(p.req_s8s8_comp, om_d.extra().compensation_mask)
            || !mask_ok(p.req_asymmetric_comp,
                    om_d.extra().asymm_compensation_mask))
        return status::unimplemented;

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
        printf("init : ");
        prb_dump(p);
    });

    prb_normalize(p);
    DEBUG({
        printf("norm : ");
        prb_dump(p);
    });

    // compensation strides require prb_normalized
    prb_set_compensation_strides(p);

    prb_simplify(p);
    DEBUG({
        printf("smpl : ");
        prb_dump(p);
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

void prb_dump(const prb_t &p) {
    printf("@@@ type:%s:%s ndims:%d ", dnnl_dt2str(p.itype),
            dnnl_dt2str(p.otype), p.ndims);
    for (int d = 0; d < p.ndims; ++d) {
        if (d != 0) printf("x");
        printf("[%zu:%zu:%d:%d:%s:%td:%td:%td:%td]", p.nodes[d].n,
                p.nodes[d].tail_size, p.nodes[d].dim_id,
                p.nodes[d].parent_node_id,
                p.nodes[d].is_zero_pad_needed ? "true" : "false", p.nodes[d].is,
                p.nodes[d].os, p.nodes[d].ss, p.nodes[d].cs);
    }
    printf(" off:%zu:%zu\n", p.ioff, p.ooff);
}

} // namespace tr

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

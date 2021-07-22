/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

using namespace dnnl::impl::types;
using namespace dnnl::impl::status;

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace tr {

/** ad-hoc structure to describe blocked memory layout */
struct layout_desc_t {
    data_type_t dt;
    int ndims;
    dims_t id;
    dims_t dims;
    strides_t strides;
};

static status_t compute_blk_and_tail(
        const memory_desc_t &md_, const int idx, int &blk, int &tail) {
    const auto md = memory_desc_wrapper(md_);
    const auto &bd = md.blocking_desc();
    if (tail == 0) return status::success;

    // find dims that have multiple inner blocks
    const std::set<dim_t> unique_inner_idxs(
            bd.inner_idxs, bd.inner_idxs + bd.inner_nblks);
    std::set<dim_t> dims_with_multiple_blks;
    for (dim_t dim : unique_inner_idxs) {
        if (std::count(bd.inner_idxs, bd.inner_idxs + bd.inner_nblks, dim) > 1)
            dims_with_multiple_blks.insert(dim);
    }

    // Dims that have a tail and have multiple blocks are not supported by the jit kernel yet.
    // For example:
    // src_tag = abcd
    // dst_tag = ABcd16b16a4b
    // 16x15x3x3
    // In this case, 'b' dim has two blocks and has a tail. It is not a supported case.
    if (dims_with_multiple_blks.find(idx) != dims_with_multiple_blks.end())
        return status::unimplemented;

    // Only supports inconsistent padding in single and double blocks
    // and the total block size <= 256
    for (int iblk = bd.inner_nblks - 1; iblk > 0; --iblk) {
        if (bd.inner_idxs[iblk] == idx) break;
        blk *= bd.inner_blks[iblk];
        tail *= bd.inner_blks[iblk];
    }
    if (unique_inner_idxs.size() > 2 || blk > 256) return status::unimplemented;

    return status::success;
}

static status_t compute_chunk_idx(const prb_t &p, const memory_desc_t &imd_,
        const memory_desc_t &omd_, const int blk_idx, int &chunk_idx) {
    const auto imd = memory_desc_wrapper(imd_);
    const auto omd = memory_desc_wrapper(omd_);
    const auto &ibd = imd.blocking_desc();
    const auto &obd = omd.blocking_desc();
    if (p.ip_tail == 0 && p.op_tail == 0) return status::success;

    const ptrdiff_t is
            = ibd.strides[blk_idx] * obd.inner_blks[obd.inner_idxs[blk_idx]];
    const ptrdiff_t os = obd.strides[blk_idx];

    for (int i = blk_idx; i < omd.ndims(); ++i) {
        if (p.nodes[i].os == os && p.nodes[i].is == is) {
            chunk_idx = i;
            return status::success;
        }
    }

    return status::invalid_arguments;
}

status_t cvt_mem_desc_to_layout_desc(const memory_desc_t &md_,
        layout_desc_t &ld, const dims_t &blocks, const dims_t &ext_padding) {
    const auto md = memory_desc_wrapper(md_);

    if (!md.is_blocking_desc()) return invalid_arguments;

    const auto &bd = md.blocking_desc();

    ld.ndims = 0;
    ld.dt = md.data_type();

    auto P = [&ld](int id, int dim, ptrdiff_t stride) {
        assert((size_t)ld.ndims < sizeof(ld.dims) / sizeof(ld.dims[0]));
        ld.id[ld.ndims] = id;
        ld.dims[ld.ndims] = dim;
        ld.strides[ld.ndims] = stride;
        ++ld.ndims;
    };

    for (int d = 0; d < md.ndims(); ++d) {
        const int ld_ndims_start = ld.ndims;
        if (blocks[d] != 1) {
            stride_t stride = 1;
            for (int iblk = bd.inner_nblks - 1; iblk >= 0; --iblk) {
                if (bd.inner_idxs[iblk] == d) P(d, bd.inner_blks[iblk], stride);
                stride *= bd.inner_blks[iblk];
            }
        }
        P(d, (md.padded_dims()[d] + ext_padding[d]) / blocks[d], bd.strides[d]);

        // TODO: NOW: revisit, do we need a reverse?
        // TODO: NOW: consider using strides instead of block sizes in md
        // reverse the order of dims
        for (int ld_d = 0; ld_d < (ld.ndims - ld_ndims_start) / 2; ++ld_d) {
            const int idx0 = ld_ndims_start + ld_d;
            const int idx1 = ld.ndims - 1 - ld_d;
            nstl::swap(ld.dims[idx0], ld.dims[idx1]);
            nstl::swap(ld.strides[idx0], ld.strides[idx1]);
        }
    }

    return success;
}

status_t prb_init(prb_t &p, const memory_desc_t &imd, const memory_desc_t &omd,
        const primitive_attr_t *attr, bool with_groups) {
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
                    primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::zero_points_runtime
                    | primitive_attr_t::skip_mask_t::post_ops)
            && check_post_ops(attr);
    if (!ok) return unimplemented;

    dims_t iblocks, oblocks, ip_padding, op_padding;
    im_d.compute_blocks(iblocks);
    om_d.compute_blocks(oblocks);
    utils::array_set(ip_padding, 0, im_d.ndims());
    utils::array_set(op_padding, 0, om_d.ndims());

    /* padding_dim consistency check
     * only supports inconsitent padding for src
     * TODO: Add inconsistent padding support for dst */
    int ip_tail = 0;
    int op_tail = 0;
    int iblk_w_tail = 1;
    int oblk_w_tail = 1;
    int blk_idx = 0;

    for (int d = 0; d < im_d.ndims(); ++d) {
        const int ip_tmp_dim = im_d.padded_dims()[d];
        const int op_tmp_dim = om_d.padded_dims()[d];
        const int ip_tmp_tail = ip_tmp_dim % oblocks[d];
        const int op_tmp_tail = op_tmp_dim % iblocks[d];

        const bool pdim_consistent = ip_tmp_dim == op_tmp_dim
                && ip_tmp_tail == 0 && op_tmp_tail == 0;
        const bool pdim_tail = ip_tmp_tail > 0
                && (ip_tmp_dim + oblocks[d] - ip_tmp_tail) == op_tmp_dim
                && op_tmp_tail == 0 && ip_tail == 0;
        if (!pdim_consistent && !pdim_tail) return status::unimplemented;
        if (pdim_tail) {
            blk_idx = d;
            ip_tail = ip_tmp_tail;
            op_tail = op_tmp_tail;
            iblk_w_tail = iblocks[d];
            oblk_w_tail = oblocks[d];
            ip_padding[d] = oblocks[d] - ip_tmp_tail;
            op_padding[d] = iblocks[d] - op_tmp_tail;
        }
    }
    CHECK(compute_blk_and_tail(omd, blk_idx, oblk_w_tail, ip_tail));

    layout_desc_t ild, old;
    status_t status
            = cvt_mem_desc_to_layout_desc(imd, ild, iblocks, ip_padding);
    if (status != success) return status;
    status = cvt_mem_desc_to_layout_desc(omd, old, oblocks, op_padding);
    if (status != success) return status;

    p.itype = ild.dt;
    p.otype = old.dt;
    p.ip_tail = ip_tail;
    p.op_tail = op_tail;
    p.iblock = iblk_w_tail;
    p.oblock = oblk_w_tail;
    p.req_src_zp = *attr->zero_points_.get(DNNL_ARG_SRC);
    p.req_dst_zp = *attr->zero_points_.get(DNNL_ARG_DST);
    p.scale_type = attr->output_scales_.has_default_values()
            ? scale_type_t::NONE
            : (attr->output_scales_.mask_ == 0 ? scale_type_t::COMMON
                                               : scale_type_t::MANY);
    p.scale_adjust = (om_d.extra().flags & memory_extra_flags::scale_adjust)
            ? om_d.extra().scale_adjust
            : 1.f;
    p.req_s8s8_comp
            = om_d.extra().flags & memory_extra_flags::compensation_conv_s8s8;
    p.req_asymmetric_comp = om_d.extra().flags
            & memory_extra_flags::compensation_conv_asymmetric_src;
    const auto compensation_needed = p.req_s8s8_comp || p.req_asymmetric_comp;

    auto mask_ok = [&](bool check, int mask) {
        return IMPLICATION(check, mask == (with_groups ? 0x3 : 0x1));
    };

    if (!mask_ok(p.req_s8s8_comp, om_d.extra().compensation_mask)
            || !mask_ok(p.req_asymmetric_comp,
                    om_d.extra().asymm_compensation_mask))
        return status::unimplemented;

    const auto compute_strides
            = [&](ptrdiff_t *strides, const int mask) {
                  ptrdiff_t last_stride = 1;
                  for (int d = old.ndims - 1; d >= 0; --d) {
                      assert((d == 0 || old.id[d - 1] <= old.id[d])
                    && "logical dimensions should be in ascending order");
                      if (mask & (1 << old.id[d])) {
                          strides[d] = last_stride;
                          last_stride *= old.dims[d];
                      }
                  }
              };

    ptrdiff_t ss[max_ndims] = {0}; // scales strides
    if (p.scale_type == scale_type_t::MANY)
        compute_strides(ss, attr->output_scales_.mask_);

    ptrdiff_t cs_[max_ndims] = {0}; // compensation strides
    ptrdiff_t *cs = cs_;
    if (compensation_needed) {
        const int compensation_mask = with_groups ? 0x3 : 0x1;
        if (p.scale_type == scale_type_t::MANY
                && attr->output_scales_.mask_ == compensation_mask)
            cs = ss;
        else {
            compute_strides(cs_, compensation_mask);
            cs = cs_;
        }
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
            p.nodes[ndims].is = ild.strides[i_pos];
            p.nodes[ndims].os = old.strides[o_pos];
            p.nodes[ndims].ss = ss[o_pos];
            p.nodes[ndims].cs = cs[o_pos];
            ++ndims;
            ++i_pos;
            ++o_pos;
        } else if (ild.dims[i_pos] < old.dims[o_pos]) {
            assert(old.dims[o_pos] % ild.dims[i_pos] == 0);
            int factor = old.dims[o_pos] / ild.dims[i_pos];
            p.nodes[ndims].n = ild.dims[i_pos];
            p.nodes[ndims].is = ild.strides[i_pos];
            p.nodes[ndims].os = old.strides[o_pos] * factor;
            p.nodes[ndims].ss = ss[o_pos] * factor;
            p.nodes[ndims].cs = cs[o_pos] * factor;
            ++ndims;
            ++i_pos;
            old.dims[o_pos] = factor;
        } else if (ild.dims[i_pos] > old.dims[o_pos]) {
            assert(ild.dims[i_pos] % old.dims[o_pos] == 0);
            int factor = ild.dims[i_pos] / old.dims[o_pos];
            p.nodes[ndims].n = old.dims[o_pos];
            p.nodes[ndims].is = ild.strides[i_pos] * factor;
            p.nodes[ndims].os = old.strides[o_pos];
            p.nodes[ndims].ss = ss[o_pos];
            p.nodes[ndims].cs = cs[o_pos];
            ++ndims;
            ++o_pos;
            ild.dims[i_pos] = factor;
        }
    }
    int blk_chunk_idx = ndims - 1;
    CHECK(compute_chunk_idx(p, imd, omd, blk_idx, blk_chunk_idx));

    p.ndims = ndims;
    p.full_ndims = ndims;
    p.blk_chunk_idx = blk_chunk_idx;

    p.ioff = memory_desc_wrapper(imd).offset0();
    p.ooff = memory_desc_wrapper(omd).offset0();

    const int sum_idx = attr->post_ops_.find(primitive_kind::sum);
    p.beta = sum_idx == -1 ? 0.f : attr->post_ops_.entry_[sum_idx].sum.scale;

    if (!attr->zero_points_.has_default_values()) {
        int new_ip_tail = 0;
        int d = 0;
        for (; d < im_d.ndims(); ++d) {
            const auto curr_ip_tail = im_d.dims()[d] % oblocks[d];
            if (curr_ip_tail > 0) {
                new_ip_tail = curr_ip_tail;
                break;
            }
        }

        if (new_ip_tail > 0) {
            p.ip_tail = new_ip_tail;
            p.iblock = iblocks[d];
            p.oblock = oblocks[d];
        }
    }

    return success;
}

status_t prb_check_blk(prb_t &p, const memory_desc_t &md_) {
    const auto md = memory_desc_wrapper(md_);
    const auto &bd = md.blocking_desc();
    if (p.ip_tail == 0) return status::success;

    // Check if the inner blocks and p.nodes[blk].n in the firsti nblks
    // is equivalent in reverse order when has tail in block layout.
    const int nblk = bd.inner_nblks;
    for (int iblk = 0; iblk < nblk; ++iblk) {
        if (bd.inner_blks[nblk - iblk - 1]
                != static_cast<ptrdiff_t>(p.nodes[iblk].n))
            return status::unimplemented;
    }
    return status::success;
}

void prb_normalize(prb_t &p) {
    for (int d = 0; d < p.ndims; ++d) {
        int min_pos = d;
        for (int j = d + 1; j < p.ndims; ++j) {
            bool new_min = false || p.nodes[j].os < p.nodes[min_pos].os
                    || (true && p.nodes[j].os == p.nodes[min_pos].os
                            && p.nodes[j].n < p.nodes[min_pos].n);
            if (new_min) min_pos = j;
        }
        if (min_pos != d) {
            nstl::swap(p.nodes[d], p.nodes[min_pos]);
            if (p.blk_chunk_idx == min_pos || p.blk_chunk_idx == d)
                p.blk_chunk_idx = p.blk_chunk_idx == min_pos ? d : min_pos;
        }
    }
}

void prb_simplify(prb_t &p) {
#if defined(__GNUC__) && __GNUC__ >= 4
/* GCC produces bogus array subscript is above array bounds warning for
 * the `p.nodes[j - 1] = p.nodes[j]` line below, so disable it for now. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    for (int d = 0; d < p.ndims - 1; ++d) {
        auto &this_node = p.nodes[d + 0];
        auto &next_node = p.nodes[d + 1];
        const bool skip_blk_idx = (p.ip_tail > 0 || p.op_tail > 0)
                && (p.blk_chunk_idx == d || p.blk_chunk_idx == d + 1);
        const bool fold = false
                || (next_node.n == static_cast<size_t>(1)
                        && !skip_blk_idx) // trivial case, just drop next node
                || (true // or real folding if possible
                        && !skip_blk_idx
                        && next_node.is
                                == static_cast<ptrdiff_t>(
                                        this_node.n * this_node.is)
                        && next_node.os
                                == static_cast<ptrdiff_t>(
                                        this_node.n * this_node.os)
                        && next_node.ss
                                == static_cast<ptrdiff_t>(
                                        this_node.n * this_node.ss)
                        && next_node.cs
                                == static_cast<ptrdiff_t>(
                                        this_node.n * this_node.cs));
        if (fold) {
            this_node.n *= next_node.n;
            for (int j = d + 2; j < p.ndims; ++j)
                p.nodes[j - 1] = p.nodes[j];
            if (d < p.blk_chunk_idx) --p.blk_chunk_idx;
            --p.ndims;
            --p.full_ndims;
            --d; // make another try
        }
    }
#if defined(__GNUC__) && __GNUC__ >= 4
#pragma GCC diagnostic pop
#endif
}

void prb_node_split(prb_t &p, int dim, size_t n1) {
    assert(dim < p.ndims);
    assert(p.ndims < max_ndims);
    assert(p.nodes[dim].n % n1 == 0);

    p.ndims += 1;
    p.full_ndims += 1;
    if (dim < p.blk_chunk_idx) p.blk_chunk_idx += 1;

    for (int d = p.ndims; d > dim + 1; --d)
        p.nodes[d] = p.nodes[d - 1];

    p.nodes[dim + 1].n = p.nodes[dim].n / n1;
    p.nodes[dim + 1].is = p.nodes[dim].is * n1;
    p.nodes[dim + 1].os = p.nodes[dim].os * n1;
    p.nodes[dim + 1].ss = p.nodes[dim].ss * n1;
    p.nodes[dim + 1].cs = p.nodes[dim].cs * n1;

    p.nodes[dim].n = n1;
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
    for (int d = 0; d < p.ndims; ++d)
        printf("[%zu:%td:%td:%td:%td]", p.nodes[d].n, p.nodes[d].is,
                p.nodes[d].os, p.nodes[d].ss, p.nodes[d].cs);
    printf(" off:%zu:%zu\n", p.ioff, p.ooff);
}

} // namespace tr

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

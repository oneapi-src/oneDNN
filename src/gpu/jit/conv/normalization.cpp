/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "gpu/jit/conv/normalization.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Adds size one spatial dimensions according to input parameters. Spatial
// dimensions are assumed to be the last dimensions.
layout_t normalize_conv_spatial(
        const layout_t &layout, int old_sp_ndims, int reduced_dim) {
    int old_ndims = layout.ndims();
    int new_ndims = old_ndims - old_sp_ndims + 3;

    dim_assignment_t to_3d(old_ndims, new_ndims);
    for (int i = 0; i < old_ndims; i++) {
        if (i < old_ndims - old_sp_ndims) {
            // Non-spatial dimensions.
            to_3d.assign(i, i);
        } else {
            // Spatial dimensions.
            int sp_idx = 3 - (old_ndims - i);
            if (reduced_dim == 3) {
                sp_idx = 2;
            } else if (sp_idx < reduced_dim) {
                sp_idx += 1;
            }
            to_3d.assign(i, new_ndims - (3 - sp_idx));
        }
    }
    return to_3d.map(layout);
}

layout_t insert_dimension(const layout_t &layout, int dim_idx) {
    auto new_blocks = layout.blocks();
    for (auto &b : new_blocks) {
        if (b.dim_idx >= dim_idx) b.dim_idx++;
    }
    return layout_t(layout.type(), layout.ndims() + 1, layout.offset(),
            new_blocks,
            /*do_normalize=*/false);
}

layout_t remove_size_1_dimension(const layout_t &layout, int dim_idx) {
    ir_assert(0 <= dim_idx && dim_idx < layout.ndims());
    ir_assert(layout.dim(dim_idx) == 1);
    dim_assignment_t a(layout.ndims(), layout.ndims() - 1);
    for (int i = 0; i < layout.ndims(); i++) {
        if (i == dim_idx) continue;
        a.assign(i, i < dim_idx ? i : i - 1);
    }
    return a.map(layout);
}

layout_t split_dimension(
        const layout_t &_layout, int dim_idx, int outer_block) {
    int rem_inner_block
            = ir_utils::safe_divide(_layout.dim(dim_idx), outer_block);
    auto layout = insert_dimension(_layout, dim_idx);
    std::vector<block_t> new_blocks;
    for (auto &eb : layout.enumerated_blocks()) {
        auto &b = eb.second;
        if (b.dim_idx != dim_idx + 1) {
            new_blocks.push_back(b);
            continue;
        }
        if (b.block % rem_inner_block == 0) {
            new_blocks.emplace_back(dim_idx + 1, rem_inner_block, b.stride);
            new_blocks.emplace_back(dim_idx, b.block / rem_inner_block,
                    dim_t(b.stride) * rem_inner_block);
            rem_inner_block = 1;
        } else {
            new_blocks.push_back(b);
            rem_inner_block = ir_utils::safe_divide(rem_inner_block, b.block);
        }
    }

    // Remove inner blocks with size one.
    std::vector<block_t> _new_blocks;
    std::vector<bool> seen(layout.ndims());
    for (auto it = new_blocks.rbegin(); it != new_blocks.rend(); ++it) {
        if (it->block == 1 && seen[it->dim_idx]) continue;
        _new_blocks.push_back(*it);
        seen[it->dim_idx] = true;
    }
    std::reverse(_new_blocks.begin(), _new_blocks.end());
    return layout_t(layout.type(), layout.ndims(), layout.offset(), _new_blocks,
            /*do_normalize=*/false);
}

layout_t normalize_conv_groups(const layout_t &layout, bool with_groups,
        int groups, bool is_dw, bool add_groups, bool is_wei) {
    if (with_groups == add_groups) return layout;
    if (is_wei) {
        ir_assert(groups == 1)
                << "Adding/removing groups can be done only for single group.";
        if (add_groups) return insert_dimension(layout, 0);
        return remove_size_1_dimension(layout, 0);
    }

    ir_assert(!with_groups) << "Unexpected groups in source/destination.";
    if (is_dw) groups = layout.dim(1);
    if (layout.dim(1) == 1) groups = 1;
    return split_dimension(layout, /*dim_idx=*/1, groups);
}

layout_t normalize_conv_layout(const layout_t &_layout, bool with_groups,
        int groups, bool is_dw, int reduced_dim, bool add_groups, bool is_wei) {
    int old_sp_ndims = _layout.ndims() - (with_groups ? 3 : 2);

    layout_t layout = _layout;
    layout = normalize_conv_spatial(layout, old_sp_ndims, reduced_dim);
    layout = normalize_conv_groups(
            layout, with_groups, groups, is_dw, add_groups, is_wei);

    return layout;
}

std::vector<dim_t> normalize_conv_dims(std::vector<dim_t> &dims,
        bool with_groups, int groups, bool is_dw, int reduced_dim,
        bool add_groups, bool is_wei) {
    layout_t dummy_layout(type_t::u8(), 0, dims);
    return normalize_conv_layout(dummy_layout, with_groups, groups, is_dw,
            reduced_dim, add_groups, is_wei)
            .dims();
}

void normalize_conv_layouts(layout_t &src_layout, layout_t &wei_layout,
        layout_t &dst_layout, layout_t &bia_layout, bool with_groups, int g,
        int ic, int oc, bool is_dw, int reduced_dim, bool add_groups) {
    src_layout = normalize_conv_layout(src_layout, /*with_groups=*/false,
            g > 1 ? src_layout.dim(1) / ic : 1, is_dw, reduced_dim, add_groups,
            /*is_wei=*/false);
    wei_layout = normalize_conv_layout(wei_layout, with_groups, g, is_dw,
            reduced_dim, add_groups, /*is_wei=*/true);
    dst_layout = normalize_conv_layout(dst_layout, /*with_groups=*/false,
            g > 1 ? dst_layout.dim(1) / oc : 1, is_dw, reduced_dim, add_groups,
            /*is_wei=*/false);
    if (add_groups && !bia_layout.is_empty()) {
        ir_assert(bia_layout.ndims() == 1) << bia_layout;
        bia_layout = split_dimension(bia_layout, 0, g);
    }
}

uint32_t conv_post_op_view_mapper_t::normalize_mask(uint32_t orig_mask) const {
    int cp_ndims = cp_view().nvdims();
    ir_assert(cp_ndims >= 3);
    // Add groups to match ngcdhw layout.
    bool add_groups = (cp_view().vvars()[1].as<var_t>().name == "g");
    // Number of dimensions before normalization.
    int orig_ndims = 2 + prb_.ndims;
    std::vector<dim_t> dummy_dims(orig_ndims, 1);
    dim_t mask_set_value = 2;
    for (int i = 0; i < orig_ndims; i++) {
        if ((orig_mask & (1 << i)) != 0) dummy_dims[i] = mask_set_value;
    }
    auto cvt_dims = normalize_conv_dims(dummy_dims, /*with_groups=*/false,
            prb_.g, prb_.is_dw, prb_.reduced_dim,
            /*add_groups=*/false, /*is_wei=*/false);
    // Split channels into groups and channels to match ngcdhw layout.
    if (add_groups) cvt_dims.insert(cvt_dims.begin() + 1, cvt_dims[1]);
    ir_assert(int(cvt_dims.size()) == cp_ndims);

    uint32_t mask = 0;
    for (int i = 0; i < cp_ndims; i++) {
        if (cvt_dims[i] == mask_set_value) mask = mask | (1 << i);
    }
    return mask;
}

void maybe_reshape_dims(int ndims, layout_t &layout, std::vector<dim_t> &dims,
        std::vector<dim_t> &padded_dims) {
    ir_assert(layout.ndims() == int(dims.size()));
    if (layout.ndims() < ndims) {
        layout = layout_t(layout.type(), ndims, layout.offset(),
                layout.blocks(), /*do_normalize=*/false);
        dims.resize(ndims, 1);
        padded_dims.resize(ndims, 1);
    }
}

view_t conv_post_op_view_mapper_t::create_view(const memory_desc_t &md) const {
    int cp_ndims = cp_view().nvdims();
    ir_assert(cp_ndims >= 3);
    // Add groups to match ngcdhw layout.
    bool add_groups = (cp_view().vvars()[1].as<var_t>().name == "g");
    layout_t layout(md, /*do_normalize=*/false);
    std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
    std::vector<dim_t> padded_dims(md.padded_dims, md.padded_dims + md.ndims);
    maybe_reshape_dims(prb_.ndims, layout, dims, padded_dims);
    layout = normalize_conv_layout(layout, /*with_groups=*/false, prb_.g,
            prb_.is_dw, prb_.reduced_dim, add_groups,
            /*is_wei=*/false);
    dims = normalize_conv_dims(dims, /*with_groups=*/false, prb_.g, prb_.is_dw,
            prb_.reduced_dim, add_groups, /*is_wei=*/false);
    padded_dims = normalize_conv_dims(padded_dims, /*with_groups=*/false,
            prb_.g, prb_.is_dw, prb_.reduced_dim, add_groups,
            /*is_wei=*/false);
    ir_assert(layout.ndims() == cp_ndims) << "Incompatible dimensions.";
    uint32_t bound_check_mask = 0;
    for (int i = 0; i < cp_ndims; i++) {
        if (dims[i] == 1) continue; // Broadcast, no bound check needed.
        if (padded_dims[i] != cp_view().tlayout().dim(i)) {
            bound_check_mask |= (1 << i);
        } else if (cp_view().has_tmask(i)) {
            bound_check_mask |= (1 << i);
        }
    }
    return view_t(layout, cp_view().vvars(), dims, bound_check_mask);
}

view_t conv_post_op_view_mapper_t::try_create_bias_view(uint32_t mask) const {
    if ((prb_.is_fwd || prb_.is_bwd_d) && prb_.with_bias)
        return create_view(prb_.conv_pd->invariant_bia_md()->data_type, mask);
    return {};
}

bool conv_post_op_view_mapper_t::is_spurious_spatial(int dim_idx) const {
    auto &var = cp_view().vvars()[dim_idx].as<var_t>();

    int sp_idx = -1;
    if (utils::one_of(var.name, "od", "id")) {
        sp_idx = 0;
    } else if (utils::one_of(var.name, "oh", "ih")) {
        sp_idx = 1;
    } else if (utils::one_of(var.name, "ow", "iw")) {
        sp_idx = 2;
    } else {
        return false;
    }

    int p = utils::pick(sp_idx, prb_.pd, prb_.ph, prb_.pw);
    int s = utils::pick(sp_idx, prb_.sd, prb_.sh, prb_.sw);
    int k = utils::pick(sp_idx, prb_.kd, prb_.kh, prb_.kw);
    int d = utils::pick(sp_idx, prb_.dd, prb_.dh, prb_.dw);

    if (prb_.is_fwd) {
        int o_value = utils::pick(sp_idx, prb_.od, prb_.oh, prb_.ow);
        int o_bound = schedule_.var_bound(var);
        int i = utils::pick(sp_idx, prb_.id, prb_.ih, prb_.iw);

        for (int o = o_value; o < o_bound; o++) {
            int i_min = o * s - p;
            if (i_min < i) return true;
        }
        return false;
    }

    if (prb_.is_bwd_d) {
        int i_value = utils::pick(sp_idx, prb_.id, prb_.ih, prb_.iw);
        int i_bound = schedule_.var_bound(var);
        int o = utils::pick(sp_idx, prb_.od, prb_.oh, prb_.ow);

        for (int i = i_value; i < i_bound; i++) {
            int os_min = i - (k - 1) * (d + 1) + p;
            if (os_min < o * s) return true;
        }
        return false;
    }

    return false;
}

bool conv_post_op_view_mapper_t::need_to_restore_zero_padding() const {
    return prb_.with_bias;
}

bool conv_post_op_view_mapper_t::use_dst_in_sum_post_op() const {
    return prb_.is_fwd;
}

bool conv_post_op_view_mapper_t::can_use_scales() const {
    return prb_.is_fwd || prb_.is_bwd_d;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

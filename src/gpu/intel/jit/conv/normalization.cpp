/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "gpu/intel/jit/conv/normalization.hpp"
#include "gpu/intel/jit/conv/config.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

layout_t insert_dimension(const layout_t &layout, dim_idx_t dim_idx) {
    auto new_blocks = layout.blocks();
    for (auto &b : new_blocks) {
        if (b.dim_idx >= dim_idx) b.dim_idx++;
    }
    return layout_t(layout.type(), layout.ndims() + 1, layout.offset(),
            new_blocks,
            /*do_normalize=*/false);
}

layout_t remove_size_1_dimension(const layout_t &layout, dim_idx_t dim_idx) {
    gpu_assert(dim_idx != dim_idx::invalid && dim_idx < layout.ndims());
    gpu_assert(layout.dim(dim_idx) == 1);
    dim_assignment_t a(layout.ndims(), layout.ndims() - 1);
    for (dim_idx_t i = 0; i < layout.ndims(); i++) {
        if (i == dim_idx) continue;
        a.assign(i, i < dim_idx ? i : i - 1);
    }
    return a.map(layout);
}

layout_t split_dimension(
        const layout_t &_layout, dim_idx_t dim_idx, dim_t outer_block) {
    dim_t rem_inner_block
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
        dim_t groups, bool is_dw, bool add_groups, bool is_wei) {
    if (with_groups == add_groups) return layout;
    if (is_wei) {
        gpu_assert(groups == 1)
                << "Adding/removing groups can be done only for single group.";
        if (add_groups) return insert_dimension(layout, 0);
        return remove_size_1_dimension(layout, 0);
    }

    gpu_assert(!with_groups) << "Unexpected groups in source/destination.";
    if (is_dw) groups = layout.dim(1);
    if (layout.dim(1) == 1) groups = 1;
    return split_dimension(layout, /*dim_idx=*/1, groups);
}

layout_t normalize_conv_layout(const layout_t &_layout, bool with_groups,
        dim_t groups, bool is_dw, const std::array<int, 3> &dhw_map,
        bool add_groups, bool is_wei) {
    layout_t layout = _layout;
    layout = spatials_to_3d(layout, with_groups, dhw_map);
    layout = normalize_conv_groups(
            layout, with_groups, groups, is_dw, add_groups, is_wei);

    return layout;
}

std::vector<dim_t> normalize_conv_dims(std::vector<dim_t> &dims,
        bool with_groups, dim_t groups, bool is_dw,
        const std::array<int, 3> &dhw_map, bool add_groups, bool is_wei) {
    layout_t dummy_layout(type_t::u8(), 0, dims);
    return normalize_conv_layout(dummy_layout, with_groups, groups, is_dw,
            dhw_map, add_groups, is_wei)
            .dims();
}

void normalize_conv_layouts(layout_t &src_layout, layout_t &wei_layout,
        layout_t &dst_layout, layout_t &bia_layout, bool with_groups, dim_t g,
        dim_t ic, dim_t oc, bool is_dw, const std::array<int, 3> &dhw_map,
        bool add_groups) {
    src_layout = normalize_conv_layout(src_layout, /*with_groups=*/false,
            g > 1 ? src_layout.dim(1) / ic : 1, is_dw, dhw_map, add_groups,
            /*is_wei=*/false);
    wei_layout = normalize_conv_layout(wei_layout, with_groups, g, is_dw,
            dhw_map, add_groups, /*is_wei=*/true);
    dst_layout = normalize_conv_layout(dst_layout, /*with_groups=*/false,
            g > 1 ? dst_layout.dim(1) / oc : 1, is_dw, dhw_map, add_groups,
            /*is_wei=*/false);
    if (add_groups && !bia_layout.is_empty()) {
        gpu_assert(bia_layout.ndims() == 1) << bia_layout;
        bia_layout = split_dimension(bia_layout, 0, g);
    }
}

uint32_t conv_post_op_view_mapper_t::normalize_mask(uint32_t orig_mask) const {
    dim_idx_t cp_ndims = cp_view().nvdims();
    gpu_assert(cp_ndims >= 3);
    // Add groups to match ngcdhw layout.
    bool add_groups = (cp_view().vvars()[1].as<var_t>().name == "g");
    // Number of dimensions before normalization.
    int orig_ndims = prb_.ndims;
    std::vector<dim_t> dummy_dims(orig_ndims, 1);
    dim_t mask_set_value = 2;
    for (int i = 0; i < orig_ndims; i++) {
        if ((orig_mask & (1 << i)) != 0) dummy_dims[i] = mask_set_value;
    }
    auto cvt_dims = normalize_conv_dims(dummy_dims, /*with_groups=*/false,
            prb_.g, prb_.is_dw, prb_.dhw_map,
            /*add_groups=*/false, /*is_wei=*/false);
    // Split channels into groups and channels to match ngcdhw layout.
    if (add_groups) cvt_dims.insert(cvt_dims.begin() + 1, cvt_dims[1]);
    gpu_assert(cvt_dims.size() == cp_ndims);

    uint32_t mask = 0;
    for (dim_idx_t i = 0; i < cp_ndims; i++) {
        if (cvt_dims[i] == mask_set_value) mask = mask | (1 << i);
    }
    return mask;
}

void maybe_reshape_dims(dim_idx_t ndims, layout_t &layout,
        std::vector<dim_t> &dims, std::vector<dim_t> &padded_dims) {
    gpu_assert(layout.ndims() == dim_idx_t(dims.size()));
    if (layout.ndims() < ndims) {
        layout = layout_t(layout.type(), ndims, layout.offset(),
                layout.blocks(), /*do_normalize=*/false);
        dims.resize(ndims, 1);
        padded_dims.resize(ndims, 1);
    }
}

// this method only gets called when ZP precompute is in order;
// in all other cases ZPs are applied ad-hoc, without a post-op
view_t conv_post_op_view_mapper_t::create_src_zp_view(uint32_t mask) const {
    auto map_o2k = [this](view_t &v, dim_idx_t idx, dim_t O, dim_t I, dim_t K,
                           dim_t D, dim_t P, dim_t S) {
        const auto KD = (K - 1) * (D + 1) + 1;
        const auto KDP = (KD > 1) ? KD - P : 0;
        const bool needs_right_bound = (O - 1) * S + KDP >= I;
        expr_t o = v.vvars()[idx];
        if (KD >= I) {
            o = o * S;
        } else {
            expr_t l, r;
            dim_t off = P;
            if (P > 0) l = binary_op_t::make(op_kind_t::_min, o * S - P, 0);
            if (needs_right_bound) {
                if (schedule_.var_bound(o) > O) {
                    auto q = binary_op_t::make(
                            op_kind_t::_min, o * S + KDP, (O - 1) * S + KDP);
                    r = binary_op_t::make(op_kind_t::_max, q, I);
                } else {
                    r = binary_op_t::make(op_kind_t::_max, o * S + KDP, I);
                }
                off -= I;
            }
            o = (!l.is_empty()) ? l : o;
            o = (!r.is_empty()) ? (!l.is_empty()) ? l + r : r : o;
            o = (off != 0) ? o + off : o;
        }
        const auto &x = view_t::placeholder_var();
        dim_t L = ir_utils::max_unique_pad_states(O, I, KD, P, S, true);
        bool mask = L < ir_utils::max_unique_pad_states(O, I, KD, P, S, false);
        v.set_vdim(v.vvars()[idx], (needs_right_bound) ? O : 1);
        v.set_tdim(idx, o, (mask) ? x < L : expr_t());
    };

    const auto &vars = cp_view().vvars();
    auto dst = normalize_conv_layout(zp_dst_, /*with_groups=*/false, prb_.g,
            prb_.is_dw, prb_.dhw_map, /*add_groups=*/true, /*is_wei=*/false);
    gpu_assert((vars.size() == 6) && (dst.ndims() == 6));

    bool non_1_spatials[3] = {false, false, false};
    std::vector<block_t> new_blk;
    for (auto &b : dst.blocks()) {
        if ((b.dim_idx >= 3) && (b.dim_idx <= 5)) {
            if (b.block > 1)
                non_1_spatials[b.dim_idx - 3] = true;
            else if (non_1_spatials[b.dim_idx - 3])
                continue;
        }
        new_blk.emplace_back(b);
    }
    dst = layout_t(dst.type(), dst.ndims(), dst.offset(), new_blk, false);

    view_t view(vars, 6);
    view.set_vdim(vars[0], 1); // mb
    view.set_vdim(vars[1], prb_.g);
    view.set_vdim(vars[2], prb_.oc);
    view.set_tdim(0, vars[0]);
    view.set_tdim(1, vars[1]);
    view.set_tdim(2, vars[2]);
    map_o2k(view, 3, prb_.od, prb_.id, prb_.kd, prb_.dd, prb_.pd, prb_.sd);
    map_o2k(view, 4, prb_.oh, prb_.ih, prb_.kh, prb_.dh, prb_.ph, prb_.sh);
    map_o2k(view, 5, prb_.ow, prb_.iw, prb_.kw, prb_.dw, prb_.pw, prb_.sw);
    view.set_tlayout(dst);
    return view;
}

view_t conv_post_op_view_mapper_t::create_view(const memory_desc_t &md) const {
    dim_idx_t cp_ndims = cp_view().nvdims();
    gpu_assert(cp_ndims >= 3);
    // Add groups to match ngcdhw layout.
    bool add_groups = (cp_view().vvars()[1].as<var_t>().name == "g");
    layout_t layout(md, /*do_normalize=*/false);
    std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
    std::vector<dim_t> padded_dims(md.padded_dims, md.padded_dims + md.ndims);
    maybe_reshape_dims(prb_.ndims, layout, dims, padded_dims);
    layout = normalize_conv_layout(layout, /*with_groups=*/false, prb_.g,
            prb_.is_dw, prb_.dhw_map, add_groups,
            /*is_wei=*/false);
    dims = normalize_conv_dims(dims, /*with_groups=*/false, prb_.g, prb_.is_dw,
            prb_.dhw_map, add_groups, /*is_wei=*/false);
    padded_dims = normalize_conv_dims(padded_dims, /*with_groups=*/false,
            prb_.g, prb_.is_dw, prb_.dhw_map, add_groups,
            /*is_wei=*/false);
    gpu_assert(layout.ndims() == cp_ndims) << "Incompatible dimensions.";
    uint32_t bound_check_mask = 0;
    for (dim_idx_t i = 0; i < cp_ndims; i++) {
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

bool conv_post_op_view_mapper_t::is_spurious_spatial(dim_idx_t dim_idx) const {
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

    dim_t p = utils::pick(sp_idx, prb_.pd, prb_.ph, prb_.pw);
    dim_t s = utils::pick(sp_idx, prb_.sd, prb_.sh, prb_.sw);
    dim_t k = utils::pick(sp_idx, prb_.kd, prb_.kh, prb_.kw);
    dim_t d = utils::pick(sp_idx, prb_.dd, prb_.dh, prb_.dw);

    if (prb_.is_fwd) {
        dim_t o_value = utils::pick(sp_idx, prb_.od, prb_.oh, prb_.ow);
        dim_t o_bound = schedule_.var_bound(var);
        dim_t i = utils::pick(sp_idx, prb_.id, prb_.ih, prb_.iw);

        for (dim_t o = o_value; o < o_bound; o++) {
            dim_t i_min = o * s - p;
            if (i_min < i) return true;
        }
        return false;
    }

    if (prb_.is_bwd_d) {
        dim_t i_value = utils::pick(sp_idx, prb_.id, prb_.ih, prb_.iw);
        dim_t i_bound = schedule_.var_bound(var);
        dim_t o = utils::pick(sp_idx, prb_.od, prb_.oh, prb_.ow);

        for (dim_t i = i_value; i < i_bound; i++) {
            dim_t os_min = i - (k - 1) * (d + 1) + p;
            if (os_min < o * s) return true;
        }
        return false;
    }

    return false;
}

bool conv_post_op_view_mapper_t::need_to_restore_zero_padding() const {
    auto &zp = prb_.conv_pd->attr()->zero_points_;
    return prb_.with_bias || !zp.has_default_values(DNNL_ARG_WEIGHTS);
}

bool conv_post_op_view_mapper_t::use_dst_in_sum_post_op() const {
    return prb_.is_fwd;
}

bool conv_post_op_view_mapper_t::can_use_scales() const {
    return prb_.is_fwd || prb_.is_bwd_d;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GPU_JIT_IR_POST_OPS_HPP
#define GPU_JIT_IR_POST_OPS_HPP

#include <string>
#include <vector>

#include "common/eltwise_pd.hpp"
#include "gpu/jit/ir/eltwise.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/normalization.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Specific to int8
struct zero_points_config_t {
    bool do_src_compensation = false;
    bool do_dst_compensation = false;
    bool is_runtime_src_zero_points = false;
    bool is_runtime_dst_zero_points = false;
    bool is_common_src_zero_point = false;
    bool is_common_dst_zero_point = false;
    int common_src_zero_point = 0;
    int common_dst_zero_point = 0;

    zero_points_config_t(const primitive_desc_t *pd = nullptr)
        : do_src_compensation(pd
                && !pd->attr()->zero_points_.has_default_values(DNNL_ARG_SRC))
        , do_dst_compensation(pd
                  && !pd->attr()->zero_points_.has_default_values(DNNL_ARG_DST))
        , is_runtime_src_zero_points(
                  pd && !pd->attr()->zero_points_.defined(DNNL_ARG_SRC))
        , is_runtime_dst_zero_points(
                  pd && !pd->attr()->zero_points_.defined(DNNL_ARG_DST))
        , is_common_src_zero_point(
                  pd && pd->attr()->zero_points_.common(DNNL_ARG_SRC))
        , is_common_dst_zero_point(
                  pd && pd->attr()->zero_points_.common(DNNL_ARG_DST))
        , common_src_zero_point(0)
        , common_dst_zero_point(0) {}

    bool with_zero_points() const {
        if (do_src_compensation) return true;
        if (do_dst_compensation) return true;
        if (is_runtime_src_zero_points) return true;
        if (is_runtime_dst_zero_points) return true;
        if (is_common_src_zero_point && common_src_zero_point != 0) return true;
        if (is_common_dst_zero_point && common_dst_zero_point != 0) return true;
        return false;
    }
};

enum class post_load_op_kind_t {
    none,
    inv,
};

class post_op_tensor_info_t {
public:
    post_op_tensor_info_t() = default;

    post_op_tensor_info_t(bool is_input, bool is_output, const view_t &view,
            const expr_t &buf, uint32_t mask, const expr_t &op_var, float scale,
            post_load_op_kind_t post_load_op)
        : is_input_(is_input)
        , is_output_(is_output)
        , view_(view)
        , buf_(buf)
        , mask_(mask)
        , op_var_(op_var)
        , scale_(scale)
        , post_load_op_(post_load_op) {
        if (op_var_.is_empty())
            op_var_ = var_t::make(type_t::f32(), make_op_var_name(buf));
        if (scale != 1)
            ir_assert(is_output_)
                    << "Scale is supported with output tensors only.";
    }

    bool is_input() const { return is_input_; }

    bool is_output() const { return is_output_; }

    bool needs_masked_update() const { return needs_masked_update_; }

    const view_t &view() const { return view_; }

    const expr_t &buf() const { return buf_; }

    const uint32_t &mask() const { return mask_; }

    const expr_t &op_var() const { return op_var_; }

    float scale() const { return scale_; }

    post_load_op_kind_t post_load_op() const { return post_load_op_; }

    post_op_tensor_info_t create_sub_tensor(const tensor_t &tile) const {
        auto ret = *this;
        ret.view_ = ret.view_.create_sub_view(tile);
        return ret;
    }

    void require_masked_update() { needs_masked_update_ = true; }

private:
    static std::string make_op_var_name(const expr_t &buf) {
        auto *var = buf.as_ptr<var_t>();
        if (var) return var->name;

        auto *ptr = buf.as_ptr<ptr_t>();
        if (ptr) {
            auto prefix = make_op_var_name(ptr->base);
            ir_assert(is_const(ptr->off));
            int off = to_cpp<int>(ptr->off);
            return prefix + "_" + std::to_string(off);
        }

        ir_error_not_expected() << "Can't generate op var name: " << buf;
        return "unknown";
    }
    bool is_input_;
    bool is_output_;
    bool needs_masked_update_ = false;
    view_t view_;
    expr_t buf_;
    uint32_t mask_;
    expr_t op_var_;
    float scale_;
    post_load_op_kind_t post_load_op_ = post_load_op_kind_t::none;
};

// There are two types of post-ops:
// - Eltwise:          lhs = eltwise(rhs) and rhs must be equal lhs
//   Eltwise is supported via special IR function eltwise_t
// - Generic post-op:  lhs = rhs
// Left-hand side (lhs) represents a single post-op tensor. Right-hand side
// tensor (rhs) is an IR expression over post-op tensors and constants.
//
// Post-op tensors support broadcast (when used from rhs) and reduction (when
// used from lhs) semantics.
//
// If lhs is (a x 1) tensor and rhs is (a x b) tensor then rhs is reduced:
//     lhs(i, 0) = sum over j rhs(i, j)
//
// If lhs is (a x b) tensor and rhs is (a x 1) tensor then rhs is broadcasted:
//     lhs(i, j) = rhs(i, 0)
class post_op_t {
public:
    post_op_t() = default;

    post_op_t(const expr_t &lhs, const expr_t &rhs,
            const func_t &eltwise = func_t())
        : lhs_(lhs), rhs_(simplify_rewrite(rhs)), eltwise_(eltwise) {}

    const expr_t &lhs() const { return lhs_; }

    const expr_t &rhs() const { return rhs_; }

    const func_t &eltwise() const { return eltwise_; }

    bool uses(const expr_t &op_var) const {
        if (contains_object(lhs_, op_var)) return true;
        if (contains_object(rhs_, op_var)) return true;
        return false;
    }

private:
    expr_t lhs_;
    expr_t rhs_;
    func_t eltwise_;
};

inline op_kind_t alg_kind_to_op_kind(alg_kind_t alg) {
    switch (alg) {
        case alg_kind::binary_add: return op_kind_t::_add;
        case alg_kind::binary_sub: return op_kind_t::_sub;
        case alg_kind::binary_mul: return op_kind_t::_mul;
        case alg_kind::binary_div: return op_kind_t::_div;
        case alg_kind::binary_min: return op_kind_t::_min;
        case alg_kind::binary_max: return op_kind_t::_max;
        case alg_kind::binary_ge: return op_kind_t::_ge;
        case alg_kind::binary_gt: return op_kind_t::_gt;
        case alg_kind::binary_le: return op_kind_t::_le;
        case alg_kind::binary_lt: return op_kind_t::_lt;
        case alg_kind::binary_eq: return op_kind_t::_eq;
        case alg_kind::binary_ne: return op_kind_t::_ne;
        default: ir_error_not_expected();
    }
    return op_kind_t::undef;
}

class conv_problem_t;
class post_op_context_t {
public:
    post_op_context_t() = delete;

    post_op_context_t(const primitive_attr_t &attr,
            const zero_points_config_t &zp_cfg, bool fuse_spatial,
            const view_t &cp_view, const gemm_schedule_t &schedule,
            const kernel_info_t &kernel_info, const memory_desc_t &dst_md,
            const memory_desc_t &out_md, const conv_problem_t *prb);

    const view_t &cp_view() const { return cp_view_; }

    const std::vector<post_op_t> &post_ops() const { return post_ops_; }

    const std::vector<post_op_tensor_info_t> &post_op_tensor_infos() const {
        return tensor_infos_;
    }

    bool need_to_restore_zero_padding() const {
        return need_to_restore_zero_padding_;
    }

private:
    static bool has_padding(const memory_desc_t &md) {
        const auto &dims = md.dims;
        const auto &padded_dims = md.padded_dims;
        for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
            if (dims[i] != padded_dims[i]) return true;
        }
        return false;
    }

    bool init_need_to_restore_zero_padding(const primitive_attr_t &attr,
            const memory_desc_t &dst_md, const memory_desc_t &out_md) const;

    // Checks if convolution computes output elements that are out of bound in
    // the output tensor. This can happen due to spatial padding.
    //
    // For example for forward convolution OW is padded to OW_PADDED. Then if
    // ow >= OW (out of bounds) and iw = ow * SW - PW + kw * (DW + 1) < IW (in
    // bounds) convolution computes an out-of-bound element which is not
    // generally zero. This requires special handling if there are post-ops
    // followed the convolution.
    bool is_spurious_spatial(const gemm_schedule_t &schedule, int dim_idx,
            const conv_problem_t *prb) const;

    int cp_ndims() const { return cp_view_.nvdims(); }

    dim_t cp_dim(int idx) const { return cp_view_.vdims()[idx]; }

    dim_t cp_padded_dim(int idx) const { return cp_view_.tlayout().dim(idx); }

    bool has_cp_mask(int idx) const { return cp_view_.has_tmask(idx); }

    bool is_cp_dim_zero_padded(int idx) const {
        return cp_view_.is_masked_vdim(idx);
    }

    const expr_t &add_input_tensor(const view_t &view, const expr_t &buf,
            post_load_op_kind_t post_load_op = post_load_op_kind_t::none) {
        return add_tensor(/*is_input=*/true, /*is_output=*/false, view, buf,
                expr_t(), /*scale=*/1.0f, post_load_op);
    }

    const expr_t &add_output_tensor(
            const view_t &view, const expr_t &buf, float scale = 1.0f) {
        return add_tensor(/*is_input=*/false, /*is_output=*/true, view, buf,
                expr_t(), scale);
    }

    const expr_t &add_tensor(bool is_input, bool is_output, const view_t &view,
            const expr_t &buf, const expr_t &op_var, float scale = 1.0f,
            post_load_op_kind_t post_load_op = post_load_op_kind_t::none) {
        ir_assert(view.nvdims() == cp_view_.nvdims());
        uint32_t mask
                = (buf.is_empty() ? ~(1u << cp_ndims()) : compute_mask(view));
        tensor_infos_.emplace_back(is_input, is_output, view, buf, mask, op_var,
                scale, post_load_op);
        return tensor_infos_.back().op_var();
    }

    uint32_t compute_mask(const view_t &view) const {
        ir_assert(cp_view_.nvdims() == view.nvdims());
        uint32_t mask = 0;
        for (int i = 0; i < view.nvdims(); i++) {
            if (view.vdims()[i] != 1) mask |= (1 << i);
        }
        return mask;
    }

    // rhs tensor has plain layout.
    view_t create_view(const type_t &type, uint32_t rhs_mask) const {
        std::vector<dim_t> rhs_dims = cp_view_.vdims();
        uint32_t bound_check_mask = 0;
        for (int i = 0; i < cp_ndims(); i++) {
            if ((rhs_mask & (1 << i)) == 0) {
                // Broadcast dimension.
                rhs_dims[i] = 1;
            } else if (is_cp_dim_zero_padded(i)) {
                bound_check_mask |= (1 << i);
            }
        }
        return view_t(layout_t(type, 0, rhs_dims, /*do_normalize=*/false),
                cp_view_.vvars(), bound_check_mask);
    }

    // rhs tensor layout is defined by md memory descriptor.
    view_t create_view(const memory_desc_t &md) const {
        ir_assert(cp_ndims() >= 3);
        // Add groups to match ngcdhw layout.
        bool add_groups = (cp_view_.vvars()[1].as<var_t>().name == "g");
        layout_t layout(md, /*do_normalize=*/false);
        std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
        std::vector<dim_t> padded_dims(
                md.padded_dims, md.padded_dims + md.ndims);
        maybe_reshape_dims(ndims_, layout, dims, padded_dims);
        layout = normalize_conv_layout(layout, /*with_groups=*/false, g_,
                is_dw_, reduced_dim_, fuse_spatial_, add_groups,
                /*is_wei=*/false);
        dims = normalize_conv_dims(dims, /*with_groups=*/false, g_, is_dw_,
                reduced_dim_, fuse_spatial_, add_groups, /*is_wei=*/false);
        padded_dims = normalize_conv_dims(padded_dims,
                /*with_groups=*/false, g_, is_dw_, reduced_dim_, fuse_spatial_,
                add_groups, /*is_wei=*/false);
        ir_assert(layout.ndims() == cp_ndims()) << "Incompatible dimensions.";
        uint32_t bound_check_mask = 0;
        for (int i = 0; i < cp_ndims(); i++) {
            if (dims[i] == 1) continue; // Broadcast, no bound check needed.
            if (padded_dims[i] != cp_padded_dim(i)) {
                bound_check_mask |= (1 << i);
            } else if (has_cp_mask(i)) {
                bound_check_mask |= (1 << i);
            }
        }
        return view_t(layout, cp_view_.vvars(), dims, bound_check_mask);
    }

    static void maybe_reshape_dims(int ndims, layout_t &layout,
            std::vector<dim_t> &dims, std::vector<dim_t> &padded_dims) {
        ir_assert(layout.ndims() == int(dims.size()));
        if (layout.ndims() < ndims) {
            layout = layout_t(layout.type(), ndims, layout.offset(),
                    layout.blocks(), /*do_normalize=*/false);
            dims.resize(ndims, 1);
            padded_dims.resize(ndims, 1);
        }
    }

    uint32_t normalize_mask(uint32_t orig_mask) const {
        ir_assert(cp_ndims() >= 3);
        // Add groups to match ngcdhw layout.
        bool add_groups = (cp_view_.vvars()[1].as<var_t>().name == "g");
        // Number of dimensions before normalization.
        int orig_ndims = 2 + ndims_;
        std::vector<dim_t> dummy_dims(orig_ndims, 1);
        dim_t mask_set_value = 2;
        for (int i = 0; i < orig_ndims; i++) {
            if ((orig_mask & (1 << i)) != 0) dummy_dims[i] = mask_set_value;
        }
        auto cvt_dims = normalize_conv_dims(dummy_dims, /*with_groups=*/false,
                g_, is_dw_, reduced_dim_, fuse_spatial_,
                /*add_groups=*/false,
                /*is_wei=*/false);
        // Split channels into groups and channels to match ngcdhw layout.
        if (add_groups) cvt_dims.insert(cvt_dims.begin() + 1, cvt_dims[1]);
        ir_assert(int(cvt_dims.size()) == cp_ndims());

        uint32_t mask = 0;
        for (int i = 0; i < cp_ndims(); i++) {
            if (cvt_dims[i] == mask_set_value) mask = mask | (1 << i);
        }
        return mask;
    }

    bool fuse_spatial_ = false;
    bool need_to_restore_zero_padding_ = false;
    bool is_dw_ = false;

    int ndims_;
    int reduced_dim_;
    int g_;

    const zero_points_config_t &zp_cfg_;

    view_t cp_view_;
    std::vector<post_op_t> post_ops_;
    std::vector<post_op_tensor_info_t> tensor_infos_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

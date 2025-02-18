/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_POST_OPS_HPP
#define GPU_INTEL_JIT_IR_POST_OPS_HPP

#include <string>
#include <vector>

#include "common/eltwise_pd.hpp"
#include "gpu/intel/jit/ir/eltwise.hpp"
#include "gpu/intel/jit/ir/gemm_schedule.hpp"
#include "gpu/intel/jit/ir/ir.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Specific to int8
struct zero_points_config_t {
public:
    bool do_src_compensation = false;
    bool do_wei_compensation = false;
    bool do_dst_compensation = false;
    bool is_runtime_src_zero_points = false;
    bool is_runtime_wei_zero_points = false;
    bool is_runtime_dst_zero_points = false;
    bool is_common_src_zero_point = false;
    bool is_common_wei_zero_point = false;
    bool is_common_dst_zero_point = false;
    bool needs_src_reorder_precalc = false;
    bool needs_src_conv_precalc = false;
    int common_src_zero_point = 0;
    int common_wei_zero_point = 0;
    int common_dst_zero_point = 0;
    type_t src_zp_type = type_t::s32();
    type_t wei_zp_type = type_t::s32();
    type_t dst_zp_type = type_t::s32();

    zero_points_config_t(const primitive_desc_t *pd = nullptr)
        : do_src_compensation(pd
                && !pd->attr()->zero_points_.has_default_values(DNNL_ARG_SRC))
        , do_wei_compensation(pd
                  && !pd->attr()->zero_points_.has_default_values(
                          DNNL_ARG_WEIGHTS))
        , do_dst_compensation(pd
                  && !pd->attr()->zero_points_.has_default_values(DNNL_ARG_DST))
        , is_runtime_src_zero_points(pd
                  && !pd->attr()->zero_points_.has_default_values(DNNL_ARG_SRC))
        , is_runtime_wei_zero_points(pd
                  && !pd->attr()->zero_points_.has_default_values(
                          DNNL_ARG_WEIGHTS))
        , is_runtime_dst_zero_points(pd
                  && !pd->attr()->zero_points_.has_default_values(DNNL_ARG_DST))
        , is_common_src_zero_point(
                  pd && pd->attr()->zero_points_.get_mask(DNNL_ARG_SRC) == 0)
        , is_common_wei_zero_point(pd
                  && pd->attr()->zero_points_.get_mask(DNNL_ARG_WEIGHTS) == 0)
        , is_common_dst_zero_point(
                  pd && pd->attr()->zero_points_.get_mask(DNNL_ARG_DST) == 0)
        , needs_src_reorder_precalc(
                  pd && do_src_compensation && can_use_src_reorder_precalc(pd))
        , needs_src_conv_precalc(pd && do_src_compensation
                  && !needs_src_reorder_precalc && can_use_src_conv_precalc(pd))
        , common_src_zero_point(0)
        , common_wei_zero_point(0)
        , common_dst_zero_point(0) {
        if (pd) {
            auto &zp = pd->attr()->zero_points_;
            src_zp_type = zp.get_data_type(DNNL_ARG_SRC);
            wei_zp_type = zp.get_data_type(DNNL_ARG_WEIGHTS);
            dst_zp_type = zp.get_data_type(DNNL_ARG_DST);
        }
    }

    bool with_zero_points() const {
        if (do_src_compensation) return true;
        if (do_wei_compensation) return true;
        if (do_dst_compensation) return true;
        if (is_runtime_src_zero_points) return true;
        if (is_runtime_wei_zero_points) return true;
        if (is_runtime_dst_zero_points) return true;
        if (is_common_src_zero_point && common_src_zero_point != 0) return true;
        if (is_common_wei_zero_point && common_wei_zero_point != 0) return true;
        if (is_common_dst_zero_point && common_dst_zero_point != 0) return true;
        return false;
    }

private:
    bool can_use_src_reorder_precalc(const primitive_desc_t *pd) {
        if (pd->kind() != primitive_kind_t::dnnl_convolution) return false;
        // Reorder-based precomputed ZPs are only available if the user did not
        // specify the weights mem desc so the convolution can choose it freely
        // and set a mem desc flag asking a reorder to precompute the values.
        return (pd->invariant_wei_md()->format_kind == format_kind::any)
                && pd->attr()->zero_points_.get_mask(DNNL_ARG_SRC) == 0
                && pd->attr()->zero_points_.has_default_values(
                        DNNL_ARG_WEIGHTS);
    }
    bool can_use_src_conv_precalc(const primitive_desc_t *pd) {
        if (pd->kind() != primitive_kind_t::dnnl_convolution) return false;
        // In general, conv-based precomputed ZPs are slower than the regular
        // ZPs up to a point where a nested convolution that does the precalc
        // takes less time than the in-situ compensations; that usually happens
        // around MB = 64, but the exact number is just a heuristic.
        // TODO: a finer-grained estimate
        return (pd->invariant_src_md()->dims[0] >= 64)
                && pd->attr()->zero_points_.has_default_values(
                        DNNL_ARG_WEIGHTS);
    }
};

class post_op_tensor_info_t {
public:
    post_op_tensor_info_t() = default;

    post_op_tensor_info_t(bool is_input, bool is_output, const view_t &view,
            const expr_t &buf, uint32_t mask, const expr_t &op_var,
            const expr_t &compute_expr, const bool do_convert = true)
        : is_input_(is_input)
        , is_output_(is_output)
        , view_(view)
        , buf_(buf)
        , mask_(mask)
        , op_var_(op_var)
        , compute_expr_(compute_expr)
        , do_convert_(do_convert) {
        if (op_var_.is_empty())
            op_var_ = var_t::make(type_t::f32(), make_op_var_name(buf));
    }

    bool is_input() const { return is_input_; }

    bool is_output() const { return is_output_; }

    bool needs_masked_update() const { return needs_masked_update_; }

    const view_t &view() const { return view_; }

    const expr_t &buf() const { return buf_; }

    const uint32_t &mask() const { return mask_; }

    const expr_t &op_var() const { return op_var_; }

    const expr_t &compute_expr() const { return compute_expr_; }

    bool needs_compute() const { return !compute_expr().is_empty(); }

    bool do_convert() const { return do_convert_; }

    post_op_tensor_info_t create_sub_tensor(const tensor_t &tile) const {
        auto ret = *this;
        ret.view_ = ret.view_.create_sub_view(tile);
        return ret;
    }

    void retype(const type_t &new_type) { view_ = view_.retype(new_type); }

    void require_masked_update() { needs_masked_update_ = true; }

private:
    static std::string make_op_var_name(const expr_t &buf) {
        auto *var = buf.as_ptr<var_t>();
        if (var) return var->name;

        auto *ptr = buf.as_ptr<ptr_t>();
        if (ptr) {
            auto prefix = make_op_var_name(ptr->base);
            gpu_assert(is_const(ptr->off));
            dim_t off = to_cpp<dim_t>(ptr->off);
            return prefix + "_" + std::to_string(off);
        }

        gpu_error_not_expected() << "Can't generate op var name: " << buf;
        return "unknown";
    }
    bool is_input_;
    bool is_output_;
    bool needs_masked_update_ = false;
    view_t view_;
    expr_t buf_;
    uint32_t mask_;
    expr_t op_var_;
    expr_t compute_expr_;
    bool do_convert_ = true;
};

class post_op_view_mapper_t {
public:
    post_op_view_mapper_t() = delete;
    post_op_view_mapper_t(const view_t &cp_view) : cp_view_(cp_view) {}
    virtual ~post_op_view_mapper_t() = default;

    const view_t &cp_view() const { return cp_view_; };

    virtual view_t create_view(const type_t &type, uint32_t rhs_mask) const {
        std::vector<dim_t> rhs_dims = cp_view_.vdims();
        uint32_t bound_check_mask = 0;
        for (int i = 0; i < int(rhs_dims.size()); i++) {
            if ((rhs_mask & (1 << i)) == 0) {
                // Broadcast dimension.
                rhs_dims[i] = 1;
            } else if (cp_view_.is_masked_vdim(i)) {
                bound_check_mask |= (1 << i);
            }
        }
        return view_t(layout_t(type, 0, rhs_dims, /*do_normalize=*/false),
                cp_view_.vvars(), bound_check_mask);
    }

    virtual view_t create_view(const memory_desc_t &md) const {
        return cp_view().retype(md.data_type);
    }

    virtual view_t create_src_zp_view(uint32_t mask) const {
        return create_view(type_t::s32(), mask);
    }

    virtual view_t try_create_bias_view(uint32_t mask) const { return {}; }

    virtual bool is_spurious_spatial(dim_idx_t dim_idx) const { return false; };
    virtual bool need_to_restore_zero_padding() const { return false; }
    virtual bool use_dst_in_sum_post_op() const { return true; }
    virtual bool can_use_scales() const { return true; }
    virtual bool can_use_simple_src_zps() const { return true; }

private:
    const view_t &cp_view_;
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
        if (eltwise_.is<eltwise_t>()) {
            auto &eltwise_func = eltwise_.as<eltwise_t>();
            if (eltwise_func.alg_kind == alg_kind::eltwise_stochastic_round)
                if (contains_object(eltwise_func.seed, op_var)) return true;
        }
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
        default: gpu_error_not_expected();
    }
    return op_kind_t::undef;
}

class post_op_context_t {
public:
    post_op_context_t() = delete;

    post_op_context_t(const primitive_attr_t &attr,
            const zero_points_config_t &zp_cfg, const gemm_schedule_t &schedule,
            const kernel_info_t &kernel_info, const memory_desc_t &dst_md,
            const memory_desc_t &out_md, const post_op_view_mapper_t &po_vm);

    const view_t &cp_view() const { return po_vm_.cp_view(); }

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
            const memory_desc_t &dst_md, const memory_desc_t &out_md,
            const zero_points_config_t &zp_cfg) const;

    dim_idx_t cp_ndims() const { return cp_view().nvdims(); }

    bool is_cp_dim_zero_padded(dim_idx_t idx) const {
        return cp_view().is_masked_vdim(idx);
    }

    const expr_t &add_input_tensor(const view_t &view, const expr_t &buf,
            const bool do_convert = true,
            const expr_t &compute_expr = expr_t()) {
        return add_tensor(/*is_input=*/true, /*is_output=*/false, view, buf,
                expr_t(), compute_expr, do_convert);
    }

    const expr_t &add_tensor(bool is_input, bool is_output, const view_t &view,
            const expr_t &buf, const expr_t &op_var,
            const expr_t &compute_expr = expr_t(),
            const bool do_convert = true) {
        gpu_assert(cp_ndims() == view.nvdims());
        uint32_t mask = (buf.is_empty() && compute_expr.is_empty()
                        ? ~(1u << cp_ndims())
                        : compute_mask(view));
        tensor_infos_.emplace_back(is_input, is_output, view, buf, mask, op_var,
                compute_expr, do_convert);
        return tensor_infos_.back().op_var();
    }

    uint32_t compute_mask(const view_t &view) const {
        gpu_assert(cp_ndims() == view.nvdims());
        uint32_t mask = 0;
        for (dim_idx_t i = 0; i < cp_ndims(); i++) {
            if (view.vdims()[i] != 1) mask |= (1 << i);
        }
        return mask;
    }

    bool need_to_restore_zero_padding_ = false;
    const post_op_view_mapper_t &po_vm_;
    std::vector<post_op_t> post_ops_;
    std::vector<post_op_tensor_info_t> tensor_infos_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

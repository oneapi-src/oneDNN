/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef GPU_JIT_CONV_POST_OP_SUPPORT_HPP
#define GPU_JIT_CONV_POST_OP_SUPPORT_HPP

#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/eltwise_pd.hpp"
#include "common/primitive_attr.hpp"
#include "common/utils.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/gemm_schedule.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/kernel_info.hpp"
#include "gpu/jit/conv/tensor.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class eltwise_t : public func_impl_t {
public:
    IR_DECL_DERIVED_TYPE_ID(eltwise_t, func_impl_t)

    static func_t make(
            alg_kind_t alg_kind, float scale, float alpha, float beta) {
        return func_t(new eltwise_t(alg_kind, scale, alpha, beta));
    }

    bool is_equal(const object_impl_t &obj) const override {
        if (!obj.is<self_type>()) return false;
        auto &other = obj.as<self_type>();

        return (alg_kind == other.alg_kind) && (scale == other.scale)
                && (alpha == other.alpha) && (beta == other.beta);
    }

    size_t get_hash() const override {
        return ir_utils::get_hash(alg_kind, scale, alpha, beta);
    }

    std::string str() const override {
        switch (alg_kind) {
            case alg_kind::eltwise_relu: return "relu";
            case alg_kind::eltwise_tanh: return "tanh";
            case alg_kind::eltwise_elu: return "elu";
            case alg_kind::eltwise_square: return "square";
            case alg_kind::eltwise_abs: return "abs";
            case alg_kind::eltwise_sqrt: return "sqrt";
            case alg_kind::eltwise_swish: return "swish";
            case alg_kind::eltwise_linear: return "linear";
            case alg_kind::eltwise_bounded_relu: return "bounded_relu";
            case alg_kind::eltwise_soft_relu: return "soft_relu";
            case alg_kind::eltwise_logistic: return "logistic";
            case alg_kind::eltwise_logsigmoid: return "logsigmoid";
            case alg_kind::eltwise_mish: return "mish";
            case alg_kind::eltwise_exp: return "exp";
            case alg_kind::eltwise_log: return "log";
            case alg_kind::eltwise_clip: return "clip";
            case alg_kind::eltwise_clip_v2: return "clip_v2";
            case alg_kind::eltwise_pow: return "pow";
            case alg_kind::eltwise_gelu_tanh: return "gelu_tanh";
            case alg_kind::eltwise_gelu_erf: return "gelu_erf";
            case alg_kind::eltwise_hardswish: return "hardswish";
            case alg_kind::eltwise_relu_use_dst_for_bwd:
                return "relu_use_dst_for_bwd";
            case alg_kind::eltwise_tanh_use_dst_for_bwd:
                return "tanh_use_dst_for_bwd";
            case alg_kind::eltwise_elu_use_dst_for_bwd:
                return "elu_use_dst_for_bwd";
            case alg_kind::eltwise_sqrt_use_dst_for_bwd:
                return "sqrt_use_dst_for_bwd";
            case alg_kind::eltwise_logistic_use_dst_for_bwd:
                return "logistic_use_dst_for_bwd";
            case alg_kind::eltwise_exp_use_dst_for_bwd:
                return "exp_use_dst_for_bwd";
            case alg_kind::eltwise_clip_v2_use_dst_for_bwd:
                return "clip_v2_use_dst_for_bwd";
            case alg_kind::eltwise_round: return "round";
            default: ir_error_not_expected();
        }
        return "unknown";
    }

    IR_DEFINE_ARG_GET(elems, 0)
    IR_DEFINE_ARG_GET(data, 1)

    alg_kind_t alg_kind;
    float scale;
    float alpha;
    float beta;

private:
    eltwise_t(alg_kind_t alg_kind, float scale, float alpha, float beta)
        : alg_kind(alg_kind), scale(scale), alpha(alpha), beta(beta) {}
};

class post_op_tensor_info_t {
public:
    post_op_tensor_info_t() = default;

    post_op_tensor_info_t(bool is_input, bool is_output, const view_t &view,
            const expr_t &buf, uint32_t mask, const expr_t &op_var, float scale)
        : is_input_(is_input)
        , is_output_(is_output)
        , view_(view)
        , buf_(buf)
        , mask_(mask)
        , op_var_(op_var)
        , scale_(scale) {
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

class post_op_context_t {
public:
    post_op_context_t() = default;

    post_op_context_t(const convolution_pd_t *pd, const conv_config_t &cfg,
            const gemm_schedule_t &gemm_schedule,
            const kernel_info_t &kernel_info)
        : pd_(pd), cfg_(&cfg), cp_view_(gemm_schedule.c_view()) {

        auto c = add_tensor(/*is_input=*/false, /*is_output=*/false, cp_view_,
                expr_t(), var_t::make(type_t::f32(), "c"));

        // Handle bias.
        if ((pd->is_fwd() || pd->is_bwd_d()) && pd->with_bias()) {
            uint32_t mask = normalize_mask(1 << 1); // Per-channel mask.
            auto view = create_view(pd->invariant_bia_md()->data_type, mask);
            auto buf = kernel_info.find_arg("bia");
            auto bia = add_input_tensor(view, buf);
            post_ops_.emplace_back(c, c + bia);
        }

        auto *attr = pd->attr();

        // Handle output scales.
        bool with_oscales = !attr->output_scales_.has_default_values();
        if (with_oscales) {
            uint32_t mask = normalize_mask(attr->output_scales_.mask_);
            auto view = create_view(type_t::f32(), mask);
            auto buf = kernel_info.find_arg("oscales");
            auto oscales = add_input_tensor(view, buf);
            post_ops_.emplace_back(c, c * oscales);
        }

        // Handle post-ops.
        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_eltwise()) {
                auto func = eltwise_t::make(po.eltwise.alg, po.eltwise.scale,
                        po.eltwise.alpha, po.eltwise.beta);
                post_ops_.emplace_back(c, c, func);
            } else if (po.is_sum(/*require_scale_one=*/false,
                               /*require_zp_zero=*/false)) {
                float scale = po.sum.scale;
                int32_t zp = po.sum.zero_point;
                if (pd->is_bwd_w()) {
                    ir_assert(scale == 1) << "BWD_W doesn't support "
                                             "non-default scale for sum.";
                    ir_assert(cfg.do_atomic_update)
                            << "Sum post-op with BWD_W requires atomic "
                               "updates.";
                    continue;
                }
                auto view = cp_view_;
                if (po.sum.dt != data_type::undef)
                    view = view.retype(po.sum.dt);
                auto buf = kernel_info.find_arg(pd->is_fwd() ? "dst" : "src");
                auto c_old = add_input_tensor(view, buf);
                post_ops_.emplace_back(c, c + scale * (c_old - zp));
            } else if (po.is_prelu()) {
                uint32_t rhs_mask = normalize_mask(po.prelu.mask);
                auto rhs_view = create_view(type_t::f32(), rhs_mask);
                auto buf_name = "prelu_rhs_" + std::to_string(i);
                auto rhs_buf = kernel_info.find_arg(buf_name);
                auto rhs = add_input_tensor(rhs_view, rhs_buf);
                post_ops_.emplace_back(
                        c, binary_op_t::make(op_kind_t::_prelu, c, rhs));
            } else if (po.is_binary()) {
                auto buf_name = "binary_rhs_" + std::to_string(i);
                auto view = create_view(po.binary.src1_desc);
                auto buf = kernel_info.find_arg(buf_name);
                auto rhs = add_input_tensor(view, buf);
                auto op_kind = alg_kind_to_op_kind(po.binary.alg);
                post_ops_.emplace_back(c, binary_op_t::make(op_kind, c, rhs));
            } else {
                ir_error_not_expected();
            }
        }

        // Handle dst zero points.
        if (cfg.zp_cfg.do_dst_compensation) {
            if (cfg.zp_cfg.is_runtime_dst_zero_points) {
                uint32_t mask = normalize_mask(
                        cfg.zp_cfg.is_common_dst_zero_point ? 0 : 1 << 1);
                auto view = create_view(type_t::s32(), mask);
                auto buf = kernel_info.find_arg("dst_zero_points");
                auto in = add_input_tensor(view, buf);
                post_ops_.emplace_back(c, c + in);
            } else {
                auto func = eltwise_t::make(alg_kind::eltwise_linear,
                        /*scale=*/1.f,
                        /*alpha=*/1.f,
                        /*beta=*/float(cfg.zp_cfg.common_dst_zero_point));
                post_ops_.emplace_back(c, c, func);
            }
        }

        need_to_restore_zero_padding_ = init_need_to_restore_zero_padding(pd);

        // Require masked updates when needed.
        for (auto &info : tensor_infos_) {
            if (!info.is_output()) continue;

            if (need_to_restore_zero_padding_) {
                info.require_masked_update();
                continue;
            }

            for (int i = 0; i < cp_ndims(); i++) {
                if ((info.mask() & (1 << i)) != 0) continue;
                if (is_spurious_spatial(gemm_schedule, i)) {
                    info.require_masked_update();
                    break;
                }
            }
        }
    }

    const view_t &cp_view() const { return cp_view_; }

    const std::vector<post_op_t> &post_ops() const { return post_ops_; }

    const std::vector<post_op_tensor_info_t> &post_op_tensor_infos() const {
        return tensor_infos_;
    }

    bool need_to_restore_zero_padding() const {
        return need_to_restore_zero_padding_;
    }

private:
    bool init_need_to_restore_zero_padding(const convolution_pd_t *pd) const {
        auto *attr = pd_->attr();
        if (cfg_->with_bias) return true;
        for (int i = 0; i < attr->post_ops_.len(); i++) {
            auto &po = attr->post_ops_.entry_[i];
            if (po.is_eltwise()) {
                if (!eltwise_fwd_pd_t::eltwise_preserves_zero(po.eltwise))
                    return true;
            } else if (po.is_sum(/*require_scale_one=*/false,
                               /*require_zp_zero=*/false)) {
                if (po.sum.zero_point != 0) return true;
                for (int j = 0; j < cp_ndims(); j++) {
                    if (!is_cp_dim_zero_padded(j)) continue;
                    // Size one dimensions are treated as broadcast which does
                    // not preserve zero padding with block updates.
                    if (cp_view_.vdims()[j] == 1) return true;
                }
            } else if (po.is_binary()) {
                for (int j = 0; j < cp_ndims(); j++) {
                    if (!is_cp_dim_zero_padded(j)) continue;
                    // Check if binary preserves zeros: (0 op X == 0) or (0 op 0 == 0).
                    bool zero_op_x_ok = (po.binary.alg == alg_kind::binary_mul);
                    bool zero_op_zero_ok = zero_op_x_ok
                            || utils::one_of(po.binary.alg,
                                    alg_kind::binary_add, alg_kind::binary_sub,
                                    alg_kind::binary_min, alg_kind::binary_max,
                                    alg_kind::binary_gt, alg_kind::binary_lt,
                                    alg_kind::binary_ne);

                    uint32_t rhs_mask
                            = utils::get_dims_mask(cp_view_.vdims().data(),
                                    po.binary.src1_desc.dims, cp_ndims());
                    if ((rhs_mask & (1 << j)) == 0 && !zero_op_x_ok)
                        return true;
                    if (!zero_op_zero_ok) return true;
                }
            } else if (po.is_prelu()) {
                return false;
            } else {
                ir_error_not_expected();
            }
        }
        if (cfg_->zp_cfg.do_src_compensation
                && pd_->dst_md()->dims[0] != pd_->dst_md()->padded_dims[0])
            return true;
        if (cfg_->zp_cfg.do_dst_compensation
                && cfg_->zp_cfg.is_common_dst_zero_point
                && pd_->dst_md()->dims[1] != pd_->dst_md()->padded_dims[1])
            return true;
        return false;
    }

    // Checks if convolution computes output elements that are out of bound in
    // the output tensor. This can happen due to spatial padding.
    //
    // For example for forward convolution OW is padded to OW_PADDED. Then if
    // ow >= OW (out of bounds) and iw = ow * SW - PW + kw * (DW + 1) < IW (in
    // bounds) convolution computes an out-of-bound element which is not
    // generally zero. This requires special handling if there are post-ops
    // followed the convolution.
    bool is_spurious_spatial(
            const gemm_schedule_t &gemm_schedule, int dim_idx) const {
        auto &var = cp_view_.vvars()[dim_idx].as<var_t>();

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

        int p = utils::pick(sp_idx, cfg_->pd, cfg_->ph, cfg_->pw);
        int s = utils::pick(sp_idx, cfg_->sd, cfg_->sh, cfg_->sw);
        int k = utils::pick(sp_idx, cfg_->kd, cfg_->kh, cfg_->kw);
        int d = utils::pick(sp_idx, cfg_->dd, cfg_->dh, cfg_->dw);

        if (cfg_->is_fwd) {
            int o_value = utils::pick(sp_idx, cfg_->od, cfg_->oh, cfg_->ow);
            int o_bound = gemm_schedule.var_bound(var);
            int i = utils::pick(sp_idx, cfg_->id, cfg_->ih, cfg_->iw);

            for (int o = o_value; o < o_bound; o++) {
                int i_min = o * s - p;
                if (i_min < i) return true;
            }
            return false;
        }

        if (cfg_->is_bwd_d) {
            int i_value = utils::pick(sp_idx, cfg_->id, cfg_->ih, cfg_->iw);
            int i_bound = gemm_schedule.var_bound(var);
            int o = utils::pick(sp_idx, cfg_->od, cfg_->oh, cfg_->ow);

            for (int i = i_value; i < i_bound; i++) {
                int os_min = i - (k - 1) * (d + 1) + p;
                if (os_min < o * s) return true;
            }
            return false;
        }

        return false;
    }

    int cp_ndims() const { return cp_view_.nvdims(); }

    dim_t cp_dim(int idx) const { return cp_view_.vdims()[idx]; }

    dim_t cp_padded_dim(int idx) const { return cp_view_.tlayout().dim(idx); }

    bool has_cp_mask(int idx) const { return cp_view_.has_tmask(idx); }

    bool is_cp_dim_zero_padded(int idx) const {
        return cp_view_.is_masked_vdim(idx);
    }

    const expr_t &add_input_tensor(const view_t &view, const expr_t &buf) {
        return add_tensor(/*is_input=*/true, /*is_output=*/false, view, buf);
    }

    const expr_t &add_output_tensor(
            const view_t &view, const expr_t &buf, float scale = 1.0f) {
        return add_tensor(/*is_input=*/false, /*is_output=*/true, view, buf,
                expr_t(), scale);
    }

    const expr_t &add_tensor(bool is_input, bool is_output, const view_t &view,
            const expr_t &buf, const expr_t &op_var = expr_t(),
            float scale = 1.0f) {
        ir_assert(view.nvdims() == cp_view_.nvdims());
        uint32_t mask
                = (buf.is_empty() ? ~(1u << cp_ndims()) : compute_mask(view));
        tensor_infos_.emplace_back(
                is_input, is_output, view, buf, mask, op_var, scale);
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
        maybe_reshape_dims(cfg_->ndims, layout, dims, padded_dims);
        layout = normalize_conv_layout(layout, /*with_groups=*/false, cfg_->g,
                cfg_->is_dw, cfg_->reduced_dim, cfg_->fuse_spatial, add_groups,
                /*is_wei=*/false);
        dims = normalize_conv_dims(dims, /*with_groups=*/false, cfg_->g,
                cfg_->is_dw, cfg_->reduced_dim, cfg_->fuse_spatial, add_groups,
                /*is_wei=*/false);
        padded_dims = normalize_conv_dims(padded_dims,
                /*with_groups=*/false, cfg_->g, cfg_->is_dw, cfg_->reduced_dim,
                cfg_->fuse_spatial, add_groups, /*is_wei=*/false);
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
        int orig_ndims = 2 + cfg_->ndims;
        std::vector<dim_t> dummy_dims(orig_ndims, 1);
        dim_t mask_set_value = 2;
        for (int i = 0; i < orig_ndims; i++) {
            if ((orig_mask & (1 << i)) != 0) dummy_dims[i] = mask_set_value;
        }
        auto cvt_dims = normalize_conv_dims(dummy_dims, /*with_groups=*/false,
                cfg_->g, cfg_->is_dw, cfg_->reduced_dim, cfg_->fuse_spatial,
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

    const convolution_pd_t *pd_;
    const conv_config_t *cfg_;

    bool need_to_restore_zero_padding_ = false;

    view_t cp_view_;
    std::vector<post_op_t> post_ops_;
    std::vector<post_op_tensor_info_t> tensor_infos_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

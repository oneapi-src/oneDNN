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

#ifndef GPU_JIT_CONV_POST_OPS_HPP
#define GPU_JIT_CONV_POST_OPS_HPP

#include <string>
#include <vector>

#include "common/convolution_pd.hpp"
#include "common/eltwise_pd.hpp"
#include "gpu/jit/ir/eltwise.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/kernel_info.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/utils/utils.hpp"

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class post_op_context_t {
public:
    post_op_context_t() = default;

    post_op_context_t(const conv_config_t &cfg,
            const gemm_schedule_t &gemm_schedule,
            const kernel_info_t &kernel_info)
        : prb_(&cfg.prb()), cfg_(&cfg), cp_view_(gemm_schedule.c_view()) {

        auto *pd = prb_->conv_pd;
        auto *attr = prb_->attr;

        auto c = add_tensor(/*is_input=*/false, /*is_output=*/false, cp_view_,
                expr_t(), var_t::make(type_t::f32(), "c"));

        // Prepare src/weights/dst scale expressions.
        std::vector<expr_t> scales(3, expr_t(1.0f));
        auto &src_scales = scales[0];
        auto &wei_scales = scales[1];
        auto &dst_scales = scales[2];
        if ((prb_->is_fwd || prb_->is_bwd_d)
                && !attr->scales_.has_default_values()) {
            const char *names[] = {"src_scales", "wei_scales", "dst_scales"};
            expr_t c_scaled = c;
            for (int i = 0; i < 3; i++) {
                auto buf = kernel_info.find_arg(names[i], /*allow_empty=*/true);
                if (buf.is_empty()) continue;
                int key = kernel_info.key(names[i]) & ~DNNL_ARG_ATTR_SCALES;
                int mask = attr->scales_.get(key).mask_;
                if (i == 1) {
                    // Convert o/i weights mask to src/dst.
                    // XXX: per_oc for BWD_D is treated as per_ic assuming it's called from
                    // deconvolution.
                    ir_assert(
                            utils::one_of(mask, 0, prb_->with_groups ? 3 : 1));
                    if (mask != 0) mask = (1 << 1);
                } else {
                    ir_assert(mask == 0);
                }
                auto view = create_view(type_t::f32(), normalize_mask(mask));
                scales[i] = add_input_tensor(view, buf);
            }
        }

        // Handle input and weights scales.
        if (!is_one(src_scales) || !is_one(wei_scales)) {
            auto c_scaled = c * src_scales * wei_scales;
            post_ops_.emplace_back(c, c_scaled);
        }

        // Handle bias.
        if ((pd->is_fwd() || pd->is_bwd_d()) && pd->with_bias()) {
            uint32_t mask = normalize_mask(1 << 1); // Per-channel mask.
            auto view = create_view(pd->invariant_bia_md()->data_type, mask);
            auto buf = kernel_info.find_arg("bia");
            auto bia = add_input_tensor(view, buf);
            post_ops_.emplace_back(c, c + bia);
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

        // Handle dst scale.
        if (!is_one(dst_scales)) {
            auto c_scaled = c / dst_scales;
            post_ops_.emplace_back(c, c_scaled);
        }

        // Handle dst zero points.
        auto &zp_cfg = prb_->zp_cfg;
        if (zp_cfg.do_dst_compensation) {
            if (zp_cfg.is_runtime_dst_zero_points) {
                uint32_t mask = normalize_mask(
                        zp_cfg.is_common_dst_zero_point ? 0 : 1 << 1);
                auto view = create_view(type_t::s32(), mask);
                auto buf = kernel_info.find_arg("dst_zero_points");
                auto in = add_input_tensor(view, buf);
                post_ops_.emplace_back(c, c + in);
            } else {
                auto func = eltwise_t::make(alg_kind::eltwise_linear,
                        /*scale=*/1.f,
                        /*alpha=*/1.f,
                        /*beta=*/float(zp_cfg.common_dst_zero_point));
                post_ops_.emplace_back(c, c, func);
            }
        }

        need_to_restore_zero_padding_ = init_need_to_restore_zero_padding();

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
    static bool has_padding(const memory_desc_t &md) {
        const auto &dims = md.dims;
        const auto &padded_dims = md.padded_dims;
        for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
            if (dims[i] != padded_dims[i]) return true;
        }
        return false;
    }

    bool init_need_to_restore_zero_padding() const {
        auto *pd = prb_->conv_pd;
        auto *attr = prb_->attr;

        if (!has_padding(prb_->c_md())) return false;

        if (prb_->with_bias) return true;
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
        if (prb_->zp_cfg.do_src_compensation
                && pd->dst_md()->dims[0] != pd->dst_md()->padded_dims[0])
            return true;
        if (prb_->zp_cfg.do_dst_compensation
                && prb_->zp_cfg.is_common_dst_zero_point
                && output_md(pd)->dims[1] != output_md(pd)->padded_dims[1])
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

        int p = utils::pick(sp_idx, prb_->pd, prb_->ph, prb_->pw);
        int s = utils::pick(sp_idx, prb_->sd, prb_->sh, prb_->sw);
        int k = utils::pick(sp_idx, prb_->kd, prb_->kh, prb_->kw);
        int d = utils::pick(sp_idx, prb_->dd, prb_->dh, prb_->dw);

        if (prb_->is_fwd) {
            int o_value = utils::pick(sp_idx, prb_->od, prb_->oh, prb_->ow);
            int o_bound = gemm_schedule.var_bound(var);
            int i = utils::pick(sp_idx, prb_->id, prb_->ih, prb_->iw);

            for (int o = o_value; o < o_bound; o++) {
                int i_min = o * s - p;
                if (i_min < i) return true;
            }
            return false;
        }

        if (prb_->is_bwd_d) {
            int i_value = utils::pick(sp_idx, prb_->id, prb_->ih, prb_->iw);
            int i_bound = gemm_schedule.var_bound(var);
            int o = utils::pick(sp_idx, prb_->od, prb_->oh, prb_->ow);

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
        maybe_reshape_dims(prb_->ndims, layout, dims, padded_dims);
        layout = normalize_conv_layout(layout, /*with_groups=*/false, prb_->g,
                prb_->is_dw, prb_->reduced_dim, cfg_->fuse_spatial(),
                add_groups,
                /*is_wei=*/false);
        dims = normalize_conv_dims(dims, /*with_groups=*/false, prb_->g,
                prb_->is_dw, prb_->reduced_dim, cfg_->fuse_spatial(),
                add_groups,
                /*is_wei=*/false);
        padded_dims = normalize_conv_dims(padded_dims,
                /*with_groups=*/false, prb_->g, prb_->is_dw, prb_->reduced_dim,
                cfg_->fuse_spatial(), add_groups, /*is_wei=*/false);
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
        int orig_ndims = 2 + prb_->ndims;
        std::vector<dim_t> dummy_dims(orig_ndims, 1);
        dim_t mask_set_value = 2;
        for (int i = 0; i < orig_ndims; i++) {
            if ((orig_mask & (1 << i)) != 0) dummy_dims[i] = mask_set_value;
        }
        auto cvt_dims = normalize_conv_dims(dummy_dims, /*with_groups=*/false,
                prb_->g, prb_->is_dw, prb_->reduced_dim, cfg_->fuse_spatial(),
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

    const conv_problem_t *prb_;
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

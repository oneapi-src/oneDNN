/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/jit/conv/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

post_op_context_t::post_op_context_t(const primitive_attr_t &attr,
        const zero_points_config_t &zp_cfg, bool fuse_spatial,
        const view_t &cp_view, const gemm_schedule_t &schedule,
        const kernel_info_t &kernel_info, const memory_desc_t &dst_md,
        const memory_desc_t &out_md, const conv_problem_t *prb)
    : fuse_spatial_(fuse_spatial)
    , is_dw_((prb) ? prb->is_dw : false)
    , ndims_((prb) ? prb->ndims : dst_md.ndims)
    , reduced_dim_((prb) ? prb->reduced_dim : -1)
    , g_((prb) ? prb->g : 1)
    , zp_cfg_(zp_cfg)
    , cp_view_(cp_view) {

    auto c = add_tensor(/*is_input=*/false, /*is_output=*/false, cp_view_,
            expr_t(), var_t::make(type_t::f32(), "c"));

    const auto *conv_pd = (prb) ? prb->conv_pd : nullptr;

    // Prepare src/weights/dst scale expressions.
    expr_t src_scales(1.0f);
    expr_t wei_scales(1.0f);
    expr_t dst_scales(1.0f);
    if ((!conv_pd || (conv_pd->is_fwd() || conv_pd->is_bwd_d()))
            && !attr.scales_.has_default_values()) {
        auto scale_args = get_scale_args();
        for (int i = 0; i < (int)scale_args.size(); i++) {
            auto buf = kernel_info.find_arg(
                    scale_args[i].first, /*allow_empty=*/true);
            if (buf.is_empty()) continue;
            int key = kernel_info.key(scale_args[i].first)
                    & ~DNNL_ARG_ATTR_SCALES;
            int mask = attr.scales_.get(key).mask_;
            view_t view;
            switch (key) {
                case DNNL_ARG_SRC:
                    ir_assert(mask == 0);
                    view = create_view(type_t::f32(), mask);
                    src_scales = add_input_tensor(
                            view, buf, post_load_op_kind_t::none);
                    break;
                case DNNL_ARG_WEIGHTS:
                    // Convert o/i weights mask to src/dst.
                    // XXX: per_oc for BWD_D is treated as per_ic assuming it's
                    // called from deconvolution.
                    ir_assert(utils::one_of(mask, 0,
                            conv_pd && conv_pd->with_groups() ? 3 : 1));
                    if (mask != 0) { mask = normalize_mask(1 << 1); }
                    view = create_view(type_t::f32(), mask);
                    wei_scales = add_input_tensor(
                            view, buf, post_load_op_kind_t::none);
                    break;
                case DNNL_ARG_DST: // Invert dst scales right after load.
                    ir_assert(mask == 0);
                    view = create_view(type_t::f32(), mask);
                    dst_scales = add_input_tensor(
                            view, buf, post_load_op_kind_t::inv);
                    break;
            }
        }
    }

    // Handle src zero points for non-convolutions.
    if (!conv_pd && zp_cfg_.do_src_compensation) {
        if (zp_cfg_.is_runtime_src_zero_points) {
            uint32_t mask = (!zp_cfg_.is_common_src_zero_point) ? 1 << 1 : 0;
            auto view = create_view(type_t::s32(), mask);
            auto buf = kernel_info.find_arg("src_zero_points");
            auto in = add_input_tensor(view, buf);
            post_ops_.emplace_back(c, c - in);
        } else {
            auto func = eltwise_t::make(alg_kind::eltwise_linear,
                    /*scale=*/1.f,
                    /*alpha=*/1.f,
                    /*beta=*/-float(zp_cfg_.common_src_zero_point));
            post_ops_.emplace_back(c, c, func);
        }
    }

    // Handle input and weights scales.
    if (!is_one(src_scales) || !is_one(wei_scales)) {
        auto c_scaled = c * src_scales * wei_scales;
        post_ops_.emplace_back(c, c_scaled);
    }

    // Handle bias.
    if (conv_pd && (conv_pd->is_fwd() || conv_pd->is_bwd_d())
            && conv_pd->with_bias()) {
        uint32_t mask = normalize_mask(1 << 1); // Per-channel mask.
        auto view = create_view(conv_pd->invariant_bia_md()->data_type, mask);
        auto buf = kernel_info.find_arg("bia");
        auto bia = add_input_tensor(view, buf);
        post_ops_.emplace_back(c, c + bia);
    }

    // Handle post-ops.
    for (int i = 0; i < attr.post_ops_.len(); i++) {
        auto &po = attr.post_ops_.entry_[i];
        if (po.is_eltwise()) {
            auto func = eltwise_t::make(po.eltwise.alg, po.eltwise.scale,
                    po.eltwise.alpha, po.eltwise.beta);
            post_ops_.emplace_back(c, c, func);
        } else if (po.is_sum(/*require_scale_one=*/false,
                           /*require_zp_zero=*/false)) {
            float scale = po.sum.scale;
            int32_t zp = po.sum.zero_point;
            if (conv_pd && conv_pd->is_bwd_w()) {
                ir_assert(scale == 1) << "BWD_W doesn't support "
                                         "non-default scale for sum.";
                continue;
            }
            auto view = cp_view_;
            if (po.sum.dt != data_type::undef) view = view.retype(po.sum.dt);
            auto buf = kernel_info.find_arg(
                    !conv_pd || conv_pd->is_fwd() ? "dst" : "src");
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
        auto c_scaled = c * dst_scales;
        post_ops_.emplace_back(c, c_scaled);
    }

    // Handle dst zero points.
    if (zp_cfg_.do_dst_compensation) {
        if (zp_cfg_.is_runtime_dst_zero_points) {
            uint32_t mask = (!zp_cfg_.is_common_dst_zero_point) ? 1 << 1 : 0;
            if (conv_pd) mask = normalize_mask(mask);
            auto view = create_view(type_t::s32(), mask);
            auto buf = kernel_info.find_arg("dst_zero_points");
            auto in = add_input_tensor(view, buf);
            post_ops_.emplace_back(c, c + in);
        } else {
            auto func = eltwise_t::make(alg_kind::eltwise_linear,
                    /*scale=*/1.f,
                    /*alpha=*/1.f,
                    /*beta=*/float(zp_cfg_.common_dst_zero_point));
            post_ops_.emplace_back(c, c, func);
        }
    }

    need_to_restore_zero_padding_ = has_padding(out_md)
            && ((conv_pd && conv_pd->with_bias())
                    || init_need_to_restore_zero_padding(attr, dst_md, out_md));

    if (!prb) return;

    // Require masked updates when needed.
    for (auto &info : tensor_infos_) {
        if (!info.is_output()) continue;

        if (need_to_restore_zero_padding_) {
            info.require_masked_update();
            continue;
        }

        for (int i = 0; i < cp_ndims(); i++) {
            if ((info.mask() & (1 << i)) != 0) continue;
            if (is_spurious_spatial(schedule, i, prb)) {
                info.require_masked_update();
                break;
            }
        }
    }
}

bool post_op_context_t::init_need_to_restore_zero_padding(
        const primitive_attr_t &attr, const memory_desc_t &dst_md,
        const memory_desc_t &out_md) const {
    for (int i = 0; i < attr.post_ops_.len(); i++) {
        auto &po = attr.post_ops_.entry_[i];
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
                        || utils::one_of(po.binary.alg, alg_kind::binary_add,
                                alg_kind::binary_sub, alg_kind::binary_min,
                                alg_kind::binary_max, alg_kind::binary_gt,
                                alg_kind::binary_lt, alg_kind::binary_ne);

                uint32_t rhs_mask
                        = utils::get_dims_mask(cp_view_.vdims().data(),
                                po.binary.src1_desc.dims, cp_ndims());
                if ((rhs_mask & (1 << j)) == 0 && !zero_op_x_ok) return true;
                if (!zero_op_zero_ok) return true;
            }
        } else if (po.is_prelu()) {
            return false;
        } else {
            ir_error_not_expected();
        }
    }
    if (zp_cfg_.do_src_compensation && dst_md.dims[0] != dst_md.padded_dims[0])
        return true;
    if (zp_cfg_.do_dst_compensation && zp_cfg_.is_common_dst_zero_point
            && out_md.dims[1] != out_md.padded_dims[1])
        return true;
    return false;
}

bool post_op_context_t::is_spurious_spatial(const gemm_schedule_t &schedule,
        int dim_idx, const conv_problem_t *prb) const {
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

    int p = utils::pick(sp_idx, prb->pd, prb->ph, prb->pw);
    int s = utils::pick(sp_idx, prb->sd, prb->sh, prb->sw);
    int k = utils::pick(sp_idx, prb->kd, prb->kh, prb->kw);
    int d = utils::pick(sp_idx, prb->dd, prb->dh, prb->dw);

    if (prb->is_fwd) {
        int o_value = utils::pick(sp_idx, prb->od, prb->oh, prb->ow);
        int o_bound = schedule.var_bound(var);
        int i = utils::pick(sp_idx, prb->id, prb->ih, prb->iw);

        for (int o = o_value; o < o_bound; o++) {
            int i_min = o * s - p;
            if (i_min < i) return true;
        }
        return false;
    }

    if (prb->is_bwd_d) {
        int i_value = utils::pick(sp_idx, prb->id, prb->ih, prb->iw);
        int i_bound = schedule.var_bound(var);
        int o = utils::pick(sp_idx, prb->od, prb->oh, prb->ow);

        for (int i = i_value; i < i_bound; i++) {
            int os_min = i - (k - 1) * (d + 1) + p;
            if (os_min < o * s) return true;
        }
        return false;
    }

    return false;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

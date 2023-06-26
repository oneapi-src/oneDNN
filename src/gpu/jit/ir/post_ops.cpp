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

#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/tensor_config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

post_op_context_t::post_op_context_t(const primitive_attr_t &attr,
        const zero_points_config_t &zp_cfg, const gemm_schedule_t &schedule,
        const kernel_info_t &kernel_info, const memory_desc_t &dst_md,
        const memory_desc_t &out_md, const post_op_view_mapper_t &po_vm)
    : po_vm_(po_vm) {

    auto c = add_tensor(/*is_input=*/false, /*is_output=*/false, cp_view(),
            expr_t(), var_t::make(type_t::f32(), "c"));

    // Prepare src/weights/dst scale expressions.
    expr_t src_scales(1.0f);
    expr_t wei_scales(1.0f);
    expr_t dst_scales(1.0f);
    expr_t src_wei_scales(1.0f);
    expr_t inv_dst_scales(1.0f);
    if (po_vm_.can_use_scales() && !attr.scales_.has_default_values()) {
        auto scale_args = get_scale_args();
        int src_scales_mask = 0;
        int wei_scales_mask = 0;
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
                    view = po_vm_.create_view(type_t::f32(), mask);
                    src_scales = add_input_tensor(view, buf);
                    src_scales_mask = mask;
                    break;
                case DNNL_ARG_WEIGHTS:
                    // Convert o/i weights mask to src/dst.
                    // XXX: per_oc for BWD_D is treated as per_ic assuming it's
                    // called from deconvolution.
                    ir_assert(utils::one_of(mask, 0, 1, 3));
                    view = po_vm_.create_view(
                            type_t::f32(), (mask) ? 1 << 1 : 0);
                    wei_scales = add_input_tensor(view, buf);
                    wei_scales_mask = mask;
                    break;
                case DNNL_ARG_DST: // Invert dst scales right after load.
                    ir_assert(mask == 0);
                    view = po_vm_.create_view(type_t::f32(), mask);
                    dst_scales = add_input_tensor(view, buf);
                    break;
            }
        }
        // Use virtual tensors for scalar scales airthmetic:
        // - For src/wei scales: pre-multiply them to avoid extra multiplications
        // - For dst scales: compute inverse right after load
        if ((!is_one(src_scales) || !is_one(wei_scales))
                && utils::everyone_is(0, src_scales_mask, wei_scales_mask)) {
            src_wei_scales = add_tensor(/*is_input=*/false,
                    /*is_output=*/false, po_vm_.create_view(type_t::f32(), 0),
                    expr_t(), var_t::make(type_t::f32(), "src_wei_scales"),
                    src_scales * wei_scales);
            src_scales = expr_t(1.0f);
            wei_scales = expr_t(1.0f);
        }
        if (!is_one(dst_scales)) {
            inv_dst_scales = add_tensor(/*is_input=*/false,
                    /*is_output=*/false, po_vm_.create_view(type_t::f32(), 0),
                    expr_t(), var_t::make(type_t::f32(), "inv_dst_scales"),
                    expr_t(1.0f) / dst_scales);
            dst_scales = expr_t(1.0f);
        }
    }

    if (po_vm_.can_use_simple_src_zps() && zp_cfg.do_src_compensation) {
        if (zp_cfg.is_runtime_src_zero_points) {
            uint32_t mask = (!zp_cfg.is_common_src_zero_point) ? 1 << 1 : 0;
            auto view = po_vm_.create_view(type_t::s32(), mask);
            auto buf = kernel_info.find_arg("src_zero_points");
            auto in = add_input_tensor(view, buf);
            post_ops_.emplace_back(c, c - in);
        } else {
            auto func = eltwise_t::make(alg_kind::eltwise_linear,
                    /*scale=*/1.f,
                    /*alpha=*/1.f,
                    /*beta=*/-float(zp_cfg.common_src_zero_point));
            post_ops_.emplace_back(c, c, func);
        }
    }

    // Handle input and weights scales.
    if (!is_one(src_wei_scales)) {
        auto c_scaled = c * src_wei_scales;
        post_ops_.emplace_back(c, c_scaled);
    } else if (!is_one(src_scales) || !is_one(wei_scales)) {
        auto c_scaled = c * src_scales * wei_scales;
        post_ops_.emplace_back(c, c_scaled);
    }

    // Handle bias.
    auto bias_view = po_vm_.try_create_bias_view(1 << 1);
    if (!bias_view.is_empty()) {
        auto buf = kernel_info.find_arg("bia");
        auto bia = add_input_tensor(bias_view, buf);
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
            auto view = cp_view();
            if (po.sum.dt != data_type::undef) view = view.retype(po.sum.dt);
            auto buf = kernel_info.find_arg(
                    (po_vm_.use_dst_in_sum_post_op()) ? "dst" : "src");
            auto c_old = add_input_tensor(view, buf);
            post_ops_.emplace_back(c, c + scale * (c_old - zp));
        } else if (po.is_prelu()) {
            auto rhs_view = po_vm_.create_view(type_t::f32(), po.prelu.mask);
            auto buf_name = "prelu_rhs_" + std::to_string(i);
            auto rhs_buf = kernel_info.find_arg(buf_name);
            auto rhs = add_input_tensor(rhs_view, rhs_buf);
            post_ops_.emplace_back(
                    c, binary_op_t::make(op_kind_t::_prelu, c, rhs));
        } else if (po.is_binary()) {
            auto buf_name = "binary_rhs_" + std::to_string(i);
            auto view = po_vm_.create_view(po.binary.src1_desc);
            auto buf = kernel_info.find_arg(buf_name);
            auto rhs = add_input_tensor(view, buf);
            auto op_kind = alg_kind_to_op_kind(po.binary.alg);
            post_ops_.emplace_back(c, binary_op_t::make(op_kind, c, rhs));
        } else {
            ir_error_not_expected();
        }
    }

    // Handle dst scale.
    if (!is_one(inv_dst_scales)) {
        auto c_scaled = c * inv_dst_scales;
        post_ops_.emplace_back(c, c_scaled);
    }

    // Handle dst zero points.
    if (zp_cfg.do_dst_compensation) {
        if (zp_cfg.is_runtime_dst_zero_points) {
            uint32_t mask = (!zp_cfg.is_common_dst_zero_point) ? 1 << 1 : 0;
            auto view = po_vm_.create_view(type_t::s32(), mask);
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

    need_to_restore_zero_padding_ = has_padding(out_md)
            && (po_vm_.need_to_restore_zero_padding()
                    || init_need_to_restore_zero_padding(
                            attr, dst_md, out_md, zp_cfg));

    // Require masked updates when needed.
    for (auto &info : tensor_infos_) {
        if (!info.is_output()) continue;

        if (need_to_restore_zero_padding_) {
            info.require_masked_update();
            continue;
        }

        for (int i = 0; i < cp_ndims(); i++) {
            if (!(info.mask() & (1 << i)) && po_vm_.is_spurious_spatial(i)) {
                info.require_masked_update();
                break;
            }
        }
    }
}

bool post_op_context_t::init_need_to_restore_zero_padding(
        const primitive_attr_t &attr, const memory_desc_t &dst_md,
        const memory_desc_t &out_md, const zero_points_config_t &zp_cfg) const {
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
                if (cp_view().vdims()[j] == 1) return true;
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
                        = utils::get_dims_mask(cp_view().vdims().data(),
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
    if (zp_cfg.do_src_compensation && dst_md.dims[0] != dst_md.padded_dims[0])
        return true;
    if (zp_cfg.do_dst_compensation && zp_cfg.is_common_dst_zero_point
            && out_md.dims[1] != out_md.padded_dims[1])
        return true;
    return false;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

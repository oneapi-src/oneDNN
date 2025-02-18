/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
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

#include "cpu/aarch64/matmul/acl_lowp_matmul_sq.hpp"

#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEQuantizationLayer.h"

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {
status_t acl_lowp_matmul_sq_resource_t::configure(
        const acl_lowp_matmul_sq_conf_t &almc) {
    if (!acl_obj_) return status::out_of_memory;
    acl_obj_->src_tensor.allocator()->init(almc.src_tensor_info);
    acl_obj_->wei_tensor.allocator()->init(almc.wei_tensor_info);
    if (almc.with_bias) {
        acl_obj_->bia_tensor.allocator()->init(almc.bia_tensor_info);
    }
    acl_obj_->dst_tensor.allocator()->init(almc.dst_tensor_info);
    arm_compute::QuantizationInfo qi {1.0, 0, true};
    acl_obj_->src_tensor.info()->set_quantization_info(qi);
    acl_obj_->wei_tensor.info()->set_quantization_info(qi);
    acl_obj_->dst_tensor.info()->set_quantization_info(qi);
    acl_obj_->gemm.configure(&acl_obj_->src_tensor, &acl_obj_->wei_tensor,
            almc.with_bias ? &acl_obj_->bia_tensor : nullptr,
            &acl_obj_->dst_tensor, almc.gemm_info);
    return status::success;
}
status_t acl_lowp_matmul_sq_t::pd_t::init(engine_t *engine) {
    VDISPATCH_MATMUL(set_default_formats(), "failed to set default formats");
    using smask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_MATMUL(
            attr()->has_default_values(smask_t::scales_runtime
                    | smask_t::zero_points_runtime | smask_t::post_ops),
            "only scale, zero point and post-ops attrs supported");

    static const std::vector<int> supported_args {
            DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
    for (int arg : supported_args) {
        if (attr()->scales_.has_default_values(arg)) continue;

        VDISPATCH_MATMUL(attr()->scales_.get_mask(arg) == 0,
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    for (int arg : supported_args) {
        if (attr()->zero_points_.has_default_values(arg)) continue;

        VDISPATCH_MATMUL(attr()->zero_points_.get_mask(arg) == 0,
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    const memory_desc_wrapper src_d(src_md_);
    const memory_desc_wrapper wei_d(weights_md_);
    const memory_desc_wrapper bia_d(bias_md_);
    const memory_desc_wrapper dst_d(dst_md_);
    using namespace data_type;
    VDISPATCH_MATMUL(utils::one_of(src_d.data_type(), s8, u8)
                            && wei_d.data_type() == s8
                            && src_d.data_type() == s8
                    ? dst_d.data_type() == s8
                    : dst_d.data_type() == u8,
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(utils::one_of(bia_d.data_type(), f32, undef),
            VERBOSE_UNSUPPORTED_DT_CFG);
    // reject in case the op is running in a Neoverse-N1.
    VDISPATCH_MATMUL(arm_compute::CPUInfo::get().has_i8mm(),
            "Neoverse-N1 not supported");
    VDISPATCH_MATMUL(src_d.matches_tag(format_tag::ab)
                    && wei_d.matches_tag(format_tag::ab)
                    && dst_d.matches_tag(format_tag::ab),
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL_SC(
            memory_desc_init_by_tag(bias_md_, bias_md_.ndims, bias_md_.dims,
                    bias_md_.data_type, format_tag::ab),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    // We set the QuantizationInfo to be dynamic because it is re-set in run()
    almc_.src_tensor_info
            = arm_compute::TensorInfo(arm_compute::TensorShape(K(), M()), 1,
                    acl_utils::get_acl_data_t(src_d.data_type(), true),
                    arm_compute::QuantizationInfo(1.0, 0, true));
    almc_.src_tensor_info.set_are_values_constant(false);
    almc_.wei_tensor_info
            = arm_compute::TensorInfo(arm_compute::TensorShape(N(), K()), 1,
                    acl_utils::get_acl_data_t(wei_d.data_type(), true),
                    arm_compute::QuantizationInfo(1.0, 0, true));
    almc_.wei_tensor_info.set_are_values_constant(false);
    almc_.dst_tensor_info
            = arm_compute::TensorInfo(arm_compute::TensorShape(N(), M()), 1,
                    acl_utils::get_acl_data_t(dst_d.data_type(), true),
                    arm_compute::QuantizationInfo(1.0, 0, true));
    almc_.bia_tensor_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(), 1, arm_compute::DataType::S32);
    almc_.with_bias = bia_d.format_kind() != format_kind::undef;
    if (almc_.with_bias) {
        // This is not currently guarded in ACL
        VDISPATCH_MATMUL(bia_d.ndims() == 2 && bia_d.dims()[0] == 1
                        && bia_d.dims()[1] == N(),
                "Only 1xN bias is supported");
        almc_.bia_tensor_info.set_tensor_shape(
                arm_compute::TensorShape(bia_d.dims()[1], bia_d.dims()[0]));
    }
    arm_compute::GEMMLowpOutputStageInfo info;
    info.type = arm_compute::GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    info.gemmlowp_multiplier = 1073741824;
    info.gemmlowp_shift = -1;
    info.gemmlowp_offset = 0;
    info.gemmlowp_min_bound = -128;
    info.gemmlowp_max_bound = 127;
    info.output_data_type = almc_.dst_tensor_info.data_type();
    almc_.gemm_info.set_gemmlowp_output_stage(info);
    auto scratchpad = scratchpad_registry().registrar();
    const dnnl::impl::memory_desc_t dst_md_ {desc_.dst_desc};
    arm_compute::ActivationLayerInfo act_info;
    CHECK(init_scratchpad(engine, scratchpad, acl_post_ops, attr_.post_ops_,
            act_info, dst_md_));
    almc_.gemm_info.set_activation_info(act_info);
    ACL_CHECK_VALID(arm_compute::NEGEMMLowpMatrixMultiplyCore::validate(
            &almc_.src_tensor_info, &almc_.wei_tensor_info,
            almc_.with_bias ? &almc_.bia_tensor_info : nullptr,
            &almc_.dst_tensor_info, almc_.gemm_info));
    return status::success;
}
status_t acl_lowp_matmul_sq_t::pd_t::init_scratchpad(engine_t *engine,
        memory_tracking::registrar_t &scratchpad, acl_post_ops_t &post_ops,
        dnnl::impl::post_ops_t &attr_post_ops,
        arm_compute::ActivationLayerInfo &act_info,
        const dnnl::impl::memory_desc_t &dst_md) {
    CHECK(post_ops.init(engine, attr_post_ops, dst_md, act_info));
    // ACL only accepts s32 bias for quantization and since
    // the current bias vector is f32 we need to convert.
    if (almc_.with_bias) {
        const memory_desc_wrapper bias_d(&bias_md_);
        scratchpad.book(memory_tracking::names::key_conv_bias_s32_convert,
                bias_d.nelems(), bias_d.data_type_size());
    }
    return status::success;
}
status_t acl_lowp_matmul_sq_t::create_resource(
        engine_t *engine, resource_mapper_t &mapper) const {
    if (mapper.has_resource(this)) return status::success;
    auto r = utils::make_unique<acl_lowp_matmul_sq_resource_t>();
    if (!r) return status::out_of_memory;
    CHECK(r->configure(pd()->almc_));
    mapper.add(this, std::move(r));
    return status::success;
}
status_t acl_lowp_matmul_sq_t::execute(const exec_ctx_t &ctx) const {
    std::lock_guard<std::mutex> _lock {this->mtx_};
    bool with_bias = pd()->almc_.with_bias;
    acl_lowp_matmul_sq_obj_t &acl_obj
            = ctx.get_resource_mapper()
                      ->get<acl_lowp_matmul_sq_resource_t>(this)
                      ->get_acl_obj();
    auto src = CTX_IN_MEM(const int8_t *, DNNL_ARG_SRC);
    auto wei = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(const int8_t *, DNNL_ARG_DST);
    acl_obj.src_tensor.allocator()->import_memory(const_cast<int8_t *>(src));
    acl_obj.wei_tensor.allocator()->import_memory(const_cast<int8_t *>(wei));
    acl_obj.dst_tensor.allocator()->import_memory(const_cast<int8_t *>(dst));
    DEFINE_ARG_SCALES_BUFFER(src_scale, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scale, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(wei_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scale, DNNL_ARG_DST);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, DNNL_ARG_DST);
    if (with_bias) {
        const auto scratchpad = ctx.get_scratchpad_grantor();
        auto bia_s32_base = scratchpad.get<uint32_t>(
                memory_tracking::names::key_conv_bias_s32_convert);
        auto bia_f32_base = CTX_IN_MEM(const float32_t *, DNNL_ARG_BIAS);
        const float bias_scale = 1 / (*src_scale * (*wei_scale));
        const int num_elements
                = acl_obj.bia_tensor.info()->total_size() / sizeof(float32_t);
        parallel_nd(num_elements, [&](dim_t e) {
            const auto b = int32_t(std::round(bia_f32_base[e] * bias_scale));
            bia_s32_base[e] = b;
        });
        acl_obj.bia_tensor.allocator()->init(*acl_obj.bia_tensor.info());
        acl_obj.bia_tensor.allocator()->import_memory(bia_s32_base);
    }
    acl_obj.src_tensor.info()->set_quantization_info(
            arm_compute::QuantizationInfo(*src_scale, -src_zero_point, true));
    acl_obj.wei_tensor.info()->set_quantization_info(
            arm_compute::QuantizationInfo(*wei_scale, -wei_zero_point, true));
    // for efficiency reasons, oneDNN saves the inverse of the destination
    acl_obj.dst_tensor.info()->set_quantization_info(
            arm_compute::QuantizationInfo(
                    1.0 / (*dst_scale), dst_zero_point, true));
    // The two calls below are stateful and, therefore, not fully thread-safe.
    // This issue is being addressed, and the lock will be removed when the
    // matmul stateless work is finished.
    acl_obj.gemm.update_quantization_parameters();
    acl_obj.gemm.run();
    // free() here tells ACL it can no longer use it, it does not deallocate
    acl_obj.src_tensor.allocator()->free();
    acl_obj.wei_tensor.allocator()->free();
    if (with_bias) { acl_obj.bia_tensor.allocator()->free(); }
    acl_obj.dst_tensor.allocator()->free();
    return status::success;
};
} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

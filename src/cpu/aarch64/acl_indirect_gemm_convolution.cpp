/*******************************************************************************
* Copyright 2021-2022, 2024 Arm Ltd. and affiliates
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

#include "acl_indirect_gemm_convolution.hpp"
#include "acl_convolution_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_indirect_gemm_convolution_fwd_t::init(engine_t *engine) {
    auto acp_ = pd()->acp_;
    acl_obj_->conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info,
            arm_compute::Conv2dInfo(acp_.padstride_info, acp_.dilation_info,
                    acp_.act_info, acp_.fast_math, 1, acp_.weights_info));
    acl_obj_->aux_mem_req = acl_obj_->conv.workspace();
    return status::success;
}

status_t acl_indirect_gemm_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    using src_data_t = data_t;
    using wei_data_t = data_t;
    using dst_data_t = data_t;
    using bia_data_t = data_t;

    auto src_base = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);

    // import_memory() and free() methods do not allocate/free any additional
    // memory, only acquire/release pointers.
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor = nullptr;
    arm_compute::Tensor dst_tensor;

    auto const pd = this->pd();
    auto const acp = pd->acp_;

    src_tensor.allocator()->init(acp.src_tensor_info);
    wei_tensor.allocator()->init(acp.wei_tensor_info);
    dst_tensor.allocator()->init(acp.dst_tensor_info);

    src_tensor.allocator()->import_memory(const_cast<src_data_t *>(src_base));
    wei_tensor.allocator()->import_memory(const_cast<wei_data_t *>(wei_base));

    const auto scratchpad = ctx.get_scratchpad_grantor();

    // If we have an unfused sum post op, put the result in a scratchpad tensor.
    // Result will be summed to the dst during acl_post_ops.execute
    auto dst_base = acp.use_dst_acc_for_sum
            ? scratchpad.get<void>(memory_tracking::names::key_generic_acc)
            : CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    dst_tensor.allocator()->import_memory(dst_base);

    if (acp.with_bias) {
        auto bia_base = CTX_IN_MEM(const bia_data_t *, DNNL_ARG_BIAS);
        bia_tensor.allocator()->init(acp.bia_tensor_info);
        bia_tensor.allocator()->import_memory(
                const_cast<bia_data_t *>(bia_base));
    }

    arm_compute::ITensorPack pack
            = {{arm_compute::TensorType::ACL_SRC_0, &src_tensor},
                    {arm_compute::TensorType::ACL_SRC_1, &wei_tensor},
                    {arm_compute::TensorType::ACL_SRC_2, &bia_tensor},
                    {arm_compute::TensorType::ACL_DST, &dst_tensor}};

    // Temp workspaces.
    const auto aux_mem = acl_obj_->aux_mem_req;
    arm_compute::Tensor gemm_tmp_buffer_tensor;
    arm_compute::Tensor conv_permuted_weights_tensor;
    arm_compute::Tensor gemm_pretranspose_tensor;

    if (aux_mem[0].size > 0) {
        gemm_tmp_buffer_tensor.allocator()->init(
                acp.gemm_tmp_buffer_info, aux_mem[0].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_tmp_buffer);
        gemm_tmp_buffer_tensor.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[0].slot, &gemm_tmp_buffer_tensor);
    }
    if (aux_mem[2].size > 0) {
        gemm_pretranspose_tensor.allocator()->init(
                acp.gemm_pretranspose_info, aux_mem[2].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_pretranspose);
        gemm_tmp_buffer_tensor.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[2].slot, &gemm_pretranspose_tensor);
    }
    if (aux_mem[3].size > 0) {
        conv_permuted_weights_tensor.allocator()->init(
                acp.conv_permuted_weights_info, aux_mem[3].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_conv_permuted_weights);
        conv_permuted_weights_tensor.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[3].slot, &conv_permuted_weights_tensor);
    }

    acl_obj_->conv.run(pack);

    void *dst = dst_tensor.buffer();

    pd->post_ops.execute(ctx, dst);

    return status::success;
}

status_t acl_indirect_gemm_convolution_fwd_t::create_resource(
        engine_t *engine, resource_mapper_t &mapper) const {

    CHECK(pd()->post_ops.create_resource(engine, mapper));
    return status::success;
}

status_t acl_indirect_gemm_convolution_fwd_t::pd_t::init_conf() {
    if (weights_md_.ndims != 4) return status::unimplemented;

    // Indirect is slower for small convolution kernels, except when src, weight and dst are BF16
    if (weights_md_.dims[2] == 1 && weights_md_.dims[3] == 1
            && !dnnl::impl::utils::everyone_is(data_type::bf16,
                    src_md_.data_type, weights_md_.data_type,
                    dst_md_.data_type))
        return status::unimplemented;

    CHECK(acl_convolution_utils::acl_init_conf(
            acp_, src_md_, weights_md_, dst_md_, bias_md_, *desc(), *attr()));

    // If we do not need to pad input channels for fast math mode then it would
    // be faster to run convolution with im2row instead of using indirect kernel
    int block_by = arm_compute::block_by(acp_.weights_info.weight_format());
    int ic = src_md_.dims[1];
    if (acp_.fast_math && ic % block_by == 0) return status::unimplemented;

    // clang-format off
    // NOTE: indirect convolution method supports only nhwc layout.
    ACL_CHECK_VALID(Op::validate(
        &acp_.src_tensor_info,
        &acp_.wei_tensor_info,
        acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
        &acp_.dst_tensor_info,
        arm_compute::Conv2dInfo(acp_.padstride_info,
                                acp_.dilation_info,
                                acp_.act_info,
                                acp_.fast_math,
                                1, acp_.weights_info)));
    // clang-format on

    return status::success;
}

void acl_indirect_gemm_convolution_fwd_t::pd_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const std::vector<arm_compute::experimental::MemoryInfo> &aux_mem) {
    if (aux_mem[0].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer,
                aux_mem[0].size, 1, aux_mem[0].alignment, aux_mem[0].alignment);
        acp_.gemm_tmp_buffer_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[0].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[2].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_pretranspose,
                aux_mem[2].size, 1, aux_mem[2].alignment, aux_mem[2].alignment);
        acp_.gemm_pretranspose_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[2].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[3].size > 0) {
        scratchpad.book(memory_tracking::names::key_conv_permuted_weights,
                aux_mem[3].size, 1, aux_mem[3].alignment, aux_mem[3].alignment);
        acp_.conv_permuted_weights_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[3].size), 1,
                arm_compute::DataType::U8);
    }
}

status_t acl_indirect_gemm_convolution_fwd_t::pd_t::init(engine_t *engine) {
    using namespace data_type;
    using smask_t = primitive_attr_t::skip_mask_t;

    const bool is_fp16_ok = expect_data_types(f16, f16, f16, f16, undef)
            && attr()->has_default_values(smask_t::post_ops, f16);
    const bool is_fp32_ok = expect_data_types(f32, f32, f32, f32, undef)
            && attr()->has_default_values(
                    smask_t::post_ops | smask_t::fpmath_mode, f32);
    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && utils::one_of(true, is_fp16_ok, is_fp32_ok)
            && !has_zero_dim_memory();
    if (!ok) return status::unimplemented;

    CHECK(init_conf());

    // Book memory.
    Op conv;
    conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info,
            arm_compute::Conv2dInfo(acp_.padstride_info, acp_.dilation_info,
                    acp_.act_info, acp_.fast_math, 1, acp_.weights_info));
    const auto aux_mem_req = conv.workspace();
    auto scratchpad = scratchpad_registry().registrar();
    init_scratchpad(scratchpad, aux_mem_req);

    CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_, acp_.act_info));
    acp_.use_dst_acc_for_sum = post_ops.has_sum();

    if (acp_.use_dst_acc_for_sum) {
        const memory_desc_wrapper dst_d(&dst_md_);
        scratchpad.book(memory_tracking::names::key_generic_acc, dst_d.nelems(),
                dst_d.data_type_size());
    }

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

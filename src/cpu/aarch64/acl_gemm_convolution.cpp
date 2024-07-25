/*******************************************************************************
* Copyright 2020-2024 Arm Ltd. and affiliates
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

#include "acl_gemm_convolution.hpp"
#include "acl_convolution_utils.hpp"
#include "common/memory_tracking.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
status_t
acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t, bia_t>::create_resource(
        engine_t *engine, resource_mapper_t &mapper) const {
    CHECK(pd()->post_ops.create_resource(engine, mapper));
    return status::success;
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
bool acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t,
        bia_t>::pd_t::zero_points_ok() const {
    using namespace data_type;
    // TODO: add support for asymmetric quantization
    return attr()->zero_points_.has_default_values();
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
bool acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t,
        bia_t>::pd_t::output_scales_mask_ok() const {
    using namespace data_type;
    const auto &mask = attr()->output_scales_.mask_;
    return IMPLICATION(!utils::one_of(src_t, s8, u8),
                   attr()->output_scales_.has_default_values())
            // TODO: add support for per_channel quantization
            && mask == 0;
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
void acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t,
        bia_t>::pd_t::aux_mem_booker(memory_tracking::registrar_t &scratchpad,
        const std::vector<arm_compute::experimental::MemoryInfo> &aux_mem) {
    if (aux_mem[0].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_asm_tmp_buffer,
                aux_mem[0].size, 1, aux_mem[0].alignment, aux_mem[0].alignment);
        acp_.gemm_tmp_asm_buffer_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[0].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[1].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_pretranspose_b,
                aux_mem[1].size, 1, aux_mem[1].alignment, aux_mem[1].alignment);
        acp_.gemm_pretranspose_b_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[1].size), 1,
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
        scratchpad.book(memory_tracking::names::key_gemm_interleaved_lhs,
                aux_mem[3].size, 1, aux_mem[3].alignment, aux_mem[3].alignment);
        acp_.gemm_interleaved_lhs_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[3].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[4].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_pretransposed_rhs,
                aux_mem[4].size, 1, aux_mem[4].alignment, aux_mem[4].alignment);
        acp_.gemm_pretransposed_rhs_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[4].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[5].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_transposed_1xwrhs,
                aux_mem[5].size, 1, aux_mem[5].alignment, aux_mem[5].alignment);
        acp_.gemm_transposed_1xwrhs_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[5].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[6].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer,
                aux_mem[6].size, 1, aux_mem[6].alignment, aux_mem[6].alignment);
        acp_.gemm_tmp_buffer_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[6].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[7].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_mm_result_s32,
                aux_mem[7].size, 1, aux_mem[7].alignment, aux_mem[7].alignment);
        acp_.gemm_mm_result_s32_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[7].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[8].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_mm_signed_a,
                aux_mem[8].size, 1, aux_mem[8].alignment, aux_mem[8].alignment);
        acp_.gemm_mm_signed_a_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[8].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[9].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_mm_signed_output,
                aux_mem[9].size, 1, aux_mem[9].alignment, aux_mem[9].alignment);
        acp_.gemm_mm_signed_output_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[8].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[10].size > 0) {
        scratchpad.book(memory_tracking::names::key_conv_gemm_col,
                aux_mem[10].size, 1, aux_mem[10].alignment,
                aux_mem[10].alignment);
        acp_.conv_im2col_output_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[10].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[11].size > 0) {
        scratchpad.book(memory_tracking::names::key_conv_permuted_weights,
                aux_mem[11].size, 1, aux_mem[11].alignment,
                aux_mem[11].alignment);
        acp_.conv_permuted_weights_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[11].size), 1,
                arm_compute::DataType::U8);
    }
    if (aux_mem[12].size > 0) {
        scratchpad.book(memory_tracking::names::key_gemm_output,
                aux_mem[12].size, 1, aux_mem[12].alignment,
                aux_mem[12].alignment);
        acp_.conv_gemm_output_info = arm_compute::TensorInfo(
                arm_compute::TensorShape(aux_mem[12].size), 1,
                arm_compute::DataType::U8);
    }
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
status_t acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t, bia_t>::pd_t::init(
        engine_t *engine) {
    using namespace data_type;
    using smask_t = primitive_attr_t::skip_mask_t;

    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && expect_data_types(src_t, wei_t, bia_t, dst_t, undef)
            && !has_zero_dim_memory()
            && attr()->has_default_values(
                    smask_t::post_ops | smask_t::fpmath_mode, dst_t)
            && output_scales_mask_ok() && zero_points_ok();
    if (!ok) return status::unimplemented;

    CHECK(init_conf());

    Op conv;
    conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info, acp_.weights_info,
            acp_.dilation_info, acp_.act_info, acp_.fast_math);

    auto scratchpad = scratchpad_registry().registrar();
    const auto mem_req = conv.workspace();
    aux_mem_booker(scratchpad, mem_req);

    CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_, acp_.act_info));
    acp_.use_dst_acc_for_sum = post_ops.has_sum();

    if (acp_.use_dst_acc_for_sum) {
        const memory_desc_wrapper dst_d(&dst_md_);

        scratchpad.book(memory_tracking::names::key_generic_acc, dst_d.nelems(),
                dst_d.data_type_size());
    }

    return status::success;
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
status_t acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t, bia_t>::init(
        engine_t *engine) {
    auto acp_ = pd()->acp_;
    acl_obj_->conv.configure(&acp_.src_tensor_info, &acp_.wei_tensor_info,
            acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
            &acp_.dst_tensor_info, acp_.padstride_info, acp_.weights_info,
            acp_.dilation_info, acp_.act_info, acp_.fast_math);
    acl_obj_->aux_mem_req = acl_obj_->conv.workspace();
    return status::success;
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
status_t
acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t, bia_t>::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src_base = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);

    // import_memory() and free() methods do not allocate/free any additional
    // memory, only acquire/release pointers.
    arm_compute::Tensor src_tensor, wei_tensor, bia_tensor = nullptr,
                                                dst_tensor;
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

    // AsmGemm Kernel.

    arm_compute::Tensor key_gemm_asm_tmp_buffer;
    if (aux_mem[0].size > 0) {
        key_gemm_asm_tmp_buffer.allocator()->init(
                acp.gemm_tmp_asm_buffer_info, aux_mem[0].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_asm_tmp_buffer);
        key_gemm_asm_tmp_buffer.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[0].slot, &key_gemm_asm_tmp_buffer);
    }
    arm_compute::Tensor gemm_pretranspose_b_tensor;
    if (aux_mem[1].size > 0) {
        gemm_pretranspose_b_tensor.allocator()->init(
                acp.gemm_pretranspose_b_info, aux_mem[1].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_pretranspose_b);
        gemm_pretranspose_b_tensor.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[1].slot, &gemm_pretranspose_b_tensor);
    }
    arm_compute::Tensor gemm_pretranspose_tensor;
    if (aux_mem[2].size > 0) {
        gemm_pretranspose_tensor.allocator()->init(
                acp.gemm_pretranspose_info, aux_mem[2].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_pretranspose);
        gemm_pretranspose_tensor.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[2].slot, &gemm_pretranspose_tensor);
    }
    arm_compute::Tensor gemm_interleaved_lhs;
    if (aux_mem[3].size > 0) {
        gemm_interleaved_lhs.allocator()->init(
                acp.gemm_interleaved_lhs_info, aux_mem[3].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_interleaved_lhs);
        gemm_interleaved_lhs.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[3].slot, &gemm_interleaved_lhs);
    }
    arm_compute::Tensor gemm_pretransposed_rhs;
    if (aux_mem[4].size > 0) {
        gemm_pretransposed_rhs.allocator()->init(
                acp.gemm_pretransposed_rhs_info, aux_mem[4].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_pretransposed_rhs);
        gemm_pretransposed_rhs.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[4].slot, &gemm_pretransposed_rhs);
    }
    arm_compute::Tensor gemm_transposed_1xwrhs;
    if (aux_mem[5].size > 0) {
        gemm_transposed_1xwrhs.allocator()->init(
                acp.gemm_transposed_1xwrhs_info, aux_mem[5].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_transposed_1xwrhs);
        gemm_transposed_1xwrhs.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[5].slot, &gemm_transposed_1xwrhs);
    }
    arm_compute::Tensor gemm_tmp_buffer;
    if (aux_mem[6].size > 0) {
        gemm_tmp_buffer.allocator()->init(
                acp.gemm_tmp_buffer_info, aux_mem[6].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_tmp_buffer);
        gemm_tmp_buffer.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[6].slot, &gemm_tmp_buffer);
    }

    arm_compute::Tensor gemm_mm_result_s32;
    if (aux_mem[7].size > 0) {
        gemm_mm_result_s32.allocator()->init(
                acp.gemm_mm_result_s32_info, aux_mem[7].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_mm_result_s32);
        gemm_mm_result_s32.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[7].slot, &gemm_mm_result_s32);
    }

    arm_compute::Tensor gemm_mm_signed_a;
    if (aux_mem[8].size > 0) {
        gemm_mm_signed_a.allocator()->init(
                acp.gemm_mm_signed_a_info, aux_mem[8].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_mm_signed_a);
        gemm_mm_signed_a.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[8].slot, &gemm_mm_signed_a);
    }

    arm_compute::Tensor gemm_mm_signed_output;
    if (aux_mem[9].size > 0) {
        gemm_mm_signed_output.allocator()->init(
                acp.gemm_mm_signed_output_info, aux_mem[9].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_gemm_mm_signed_output);
        gemm_mm_signed_output.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[9].slot, &gemm_mm_signed_output);
    }

    arm_compute::Tensor conv_im2col_output;
    if (aux_mem[10].size > 0) {
        conv_im2col_output.allocator()->init(
                acp.conv_im2col_output_info, aux_mem[10].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_conv_gemm_col);
        conv_im2col_output.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[10].slot, &conv_im2col_output);
    }

    arm_compute::Tensor conv_permuted_weights;
    if (aux_mem[11].size > 0) {
        conv_permuted_weights.allocator()->init(
                acp.conv_permuted_weights_info, aux_mem[11].alignment);
        auto buffer = scratchpad.get<void>(
                memory_tracking::names::key_conv_permuted_weights);
        conv_permuted_weights.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[11].slot, &conv_permuted_weights);
    }

    arm_compute::Tensor conv_gemm_output;
    if (aux_mem[12].size > 0) {
        conv_gemm_output.allocator()->init(
                acp.conv_gemm_output_info, aux_mem[12].alignment);
        auto buffer
                = scratchpad.get<void>(memory_tracking::names::key_gemm_output);
        conv_gemm_output.allocator()->import_memory(buffer);
        pack.add_tensor(aux_mem[12].slot, &conv_gemm_output);
    }

    acl_obj_->conv.run(pack);

    void *dst = dst_tensor.buffer();

    pd->post_ops.execute(ctx, dst);

    return status::success;
}

template <data_type_t src_t, data_type_t wei_t, data_type_t dst_t,
        data_type_t bia_t>
status_t
acl_gemm_convolution_fwd_t<src_t, wei_t, dst_t, bia_t>::pd_t::init_conf() {
    if (weights_md_.ndims != 4) return status::unimplemented;

    // General Compute Library checks, memory tags are also set there
    CHECK(acl_convolution_utils::acl_init_conf(
            acp_, src_md_, weights_md_, dst_md_, bias_md_, *desc(), *attr()));

    // clang-format off
    // Validate convolution manually to check for return status
    ACL_CHECK_VALID(arm_compute::NEGEMMConvolutionLayer::validate(
        &acp_.src_tensor_info,
        &acp_.wei_tensor_info,
        acp_.with_bias ? &acp_.bia_tensor_info : nullptr,
        &acp_.dst_tensor_info,
        acp_.padstride_info,
        acp_.weights_info,
        acp_.dilation_info,
        acp_.act_info,
        acp_.fast_math));
    // clang-format on

    return status::success;
}

using namespace data_type;
template struct acl_gemm_convolution_fwd_t<f32>;
template struct acl_gemm_convolution_fwd_t<f16>;
template struct acl_gemm_convolution_fwd_t<s8, s8, s8, s32>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

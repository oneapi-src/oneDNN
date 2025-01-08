/*******************************************************************************
* Copyright 2020-2025 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_CONVOLUTION_UTILS_HPP
#define CPU_AARCH64_ACL_CONVOLUTION_UTILS_HPP

#include <map>
#include "acl_post_ops.hpp"
#include "acl_utils.hpp"
#include "arm_compute/runtime/experimental/operators/CpuDepthwiseConv2d.h"
#include "cpu/cpu_convolution_pd.hpp"
#include <type_traits>
namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <typename ConvOp>
struct acl_obj_t {
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
    ConvOp conv;
    arm_compute::experimental::MemoryRequirements aux_mem_req;
};

struct acl_conv_conf_t {
    bool with_bias;
    bool fast_math;
    // If this is true, the result of the convolution goes into a temporarily
    // allocated ACL tensor to be accumulated into the oneDNN dst during postops
    bool use_dst_acc_for_sum;
    // Tells that the selected algorithm is Winograd. This is needed because the
    // algorithm can be set to algorithm::convolution_auto and later on we need to
    // skip fixed-format protocol as ACL Winograd does not support it.
    bool alg_winograd;
    arm_compute::TensorInfo src_tensor_info;
    arm_compute::TensorInfo wei_tensor_info;
    arm_compute::TensorInfo bia_tensor_info;
    arm_compute::TensorInfo dst_tensor_info;

    arm_compute::PadStrideInfo padstride_info;
    arm_compute::Size2D dilation_info;
    // Additional information about the weights not included in wei_tensor_info
    arm_compute::WeightsInfo weights_info;
    // Note: this will default to not enabled, and will do nothing
    arm_compute::ActivationLayerInfo act_info;
};

namespace acl_convolution_utils {

status_t acl_init_conf(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr);

status_t init_conf_wino(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr);

} // namespace acl_convolution_utils

// Keys are anonymous with local linkage. So deduce the type automagically.
using conv_key_t = decltype(memory_tracking::names::key_gemm_tmp_buffer);

template <typename op_t, typename post_ops_t>
status_t init_scratchpad(op_t &conv, memory_tracking::registrar_t &scratchpad,
        const std::map<int, conv_key_t> &conv_keys, engine_t *engine,
        post_ops_t &post_ops, dnnl::impl::post_ops_t &attr_post_ops,
        arm_compute::ActivationLayerInfo &act_info, bool &use_dst_acc_for_sum,
        const dnnl::impl::memory_desc_t &dst_md) {

    // Book temp mem.
    const auto aux_mem_req = conv.workspace();
    for (const auto &key : conv_keys) {
        const auto id = key.first;
        if (aux_mem_req[id].size > 0) {
            scratchpad.book(key.second, aux_mem_req[id].size, 1,
                    aux_mem_req[id].alignment, aux_mem_req[id].alignment);
        }
    }

    CHECK(post_ops.init(engine, attr_post_ops, dst_md, act_info));
    use_dst_acc_for_sum = post_ops.has_sum();

    if (use_dst_acc_for_sum) {
        const memory_desc_wrapper dst_d(&dst_md);
        scratchpad.book(memory_tracking::names::key_generic_acc, dst_d.nelems(),
                dst_d.data_type_size());
    }

    return status::success;
}

template <typename conv_obj_t, typename conv_pd_t, typename src_data_t,
        typename wei_data_t = src_data_t, typename dst_data_t = src_data_t,
        typename bia_data_t = src_data_t>
status_t execute_forward_conv_acl(const exec_ctx_t &ctx,
        conv_obj_t *acl_conv_obj, const conv_pd_t *pd,
        const std::map<int, conv_key_t> &conv_keys) {

    auto src_base = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);

    // import_memory() and free() methods do not allocate/free any additional
    // memory, only acquire/release pointers.
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor = nullptr;
    arm_compute::Tensor dst_tensor;

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

    // Constness of the weight tensor matters for depthwise conv in ACL.
    // Otherwise, it will package the weights more often than needed, as
    // it will expect the weights to change within the duration of the run
    // func.
    arm_compute::ITensorPack pack;
    pack.add_tensor(arm_compute::TensorType::ACL_SRC_0, &src_tensor);
    pack.add_const_tensor(arm_compute::TensorType::ACL_SRC_1, &wei_tensor);
    pack.add_const_tensor(arm_compute::TensorType::ACL_SRC_2, &bia_tensor);
    pack.add_tensor(arm_compute::TensorType::ACL_DST, &dst_tensor);

    // Get temp workspaces.
    const auto aux_mem = acl_conv_obj->aux_mem_req;

    // Hold onto tmp tensors while we need pack.
    std::vector<arm_compute::Tensor> tmp_tensors(aux_mem.size());
    for (const auto &key : conv_keys) {
        const auto id = key.first;
        if (aux_mem[id].size > 0) {
            const auto info = arm_compute::TensorInfo(
                    arm_compute::TensorShape(aux_mem[id].size), 1,
                    arm_compute::DataType::U8);
            auto buffer = scratchpad.get<void>(key.second);
            tmp_tensors[id].allocator()->init(info, aux_mem[id].alignment);
            tmp_tensors[id].allocator()->import_memory(buffer);
            pack.add_tensor(aux_mem[id].slot, &tmp_tensors[id]);
        }
    }

    acl_conv_obj->conv.run(pack);

    void *dst = dst_tensor.buffer();
    pd->post_ops.execute(ctx, dst);

    return status::success;
}

template <typename conv_obj_t, typename conv_pd_t, typename src_data_t,
        typename wei_data_t = src_data_t, typename dst_data_t = src_data_t,
        typename bia_data_t = src_data_t>
status_t execute_forward_conv_acl(
        const exec_ctx_t &ctx, conv_obj_t &acl_conv_obj, const conv_pd_t *pd) {
    bool with_bias = pd->acp_.with_bias;
    bool use_dst_acc_for_sum = pd->acp_.use_dst_acc_for_sum;

    auto src_base = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);

    // import_memory() and free() methods do not allocate/free any additional
    // memory, only acquire/release pointers.
    acl_conv_obj.src_tensor.allocator()->import_memory(
            const_cast<src_data_t *>(src_base));
    acl_conv_obj.wei_tensor.allocator()->import_memory(
            const_cast<wei_data_t *>(wei_base));

    const auto scratchpad = ctx.get_scratchpad_grantor();

    // If we have an unfused sum post op, put the result in a scratchpad tensor.
    // Result will be summed to the dst during acl_post_ops.execute
    auto dst_base = use_dst_acc_for_sum
            ? scratchpad.get<void>(memory_tracking::names::key_generic_acc)
            : CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    acl_conv_obj.dst_tensor.allocator()->import_memory(dst_base);

    if (with_bias) {
        auto bia_base = CTX_IN_MEM(const bia_data_t *, DNNL_ARG_BIAS);
        acl_conv_obj.bia_tensor.allocator()->import_memory(
                const_cast<bia_data_t *>(bia_base));
    }

    acl_conv_obj.conv.run();

    acl_conv_obj.src_tensor.allocator()->free();
    acl_conv_obj.wei_tensor.allocator()->free();
    if (with_bias) { acl_conv_obj.bia_tensor.allocator()->free(); }

    void *dst = acl_conv_obj.dst_tensor.buffer();
    pd->post_ops.execute(ctx, dst);

    acl_conv_obj.dst_tensor.allocator()->free();

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_CONVOLUTION_UTILS_HPP

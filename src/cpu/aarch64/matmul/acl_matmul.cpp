/*******************************************************************************
* Copyright 2021-2024 Arm Ltd. and affiliates
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

#include "cpu/aarch64/matmul/acl_matmul.hpp"
#include <mutex>

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace data_type;

namespace {
using data_t = prec_traits_t<data_type::f32>::type;
} // namespace

status_t acl_matmul_t::init(engine_t *engine) {
    auto amp_ = pd()->amp_;
    // Configure transpose kernel for src and wei
    if (amp_.is_transA && !amp_.do_transC) {
        acl_obj_->transA.configure(&amp_.src_acc_info, &amp_.src_tensor_info);
    }
    if (amp_.is_transB && !amp_.do_transC) {
        acl_obj_->transB.configure(&amp_.wei_acc_info, &amp_.wei_tensor_info);
    }
    if (amp_.do_transC) {
        acl_obj_->transC.configure(&amp_.dst_acc_info, &amp_.dst_tensor_info);
    }
    // Configure GEMM
    if (amp_.do_transC) {
        acl_obj_->asm_gemm.configure(&amp_.wei_tensor_info,
                &amp_.src_tensor_info, nullptr, &amp_.dst_acc_info,
                amp_.gemm_info);
    } else {
        acl_obj_->asm_gemm.configure(&amp_.src_tensor_info,
                &amp_.wei_tensor_info, nullptr, &amp_.dst_tensor_info,
                amp_.gemm_info);
    }
    acl_obj_->aux_mem_req = acl_obj_->asm_gemm.workspace();
    if (amp_.do_act) {
        auto dst_info_to_use
                = amp_.do_transC ? &amp_.dst_acc_info : &amp_.dst_tensor_info;
        acl_obj_->act.configure(dst_info_to_use, dst_info_to_use,
                amp_.gemm_info.activation_info());
    }

    return status::success;
}

status_t acl_matmul_t::pd_t::init(engine_t *engine) {
    using smask_t = primitive_attr_t::skip_mask_t;
    const bool is_fp32_ok
            = utils::everyone_is(data_type::f32, src_md()->data_type,
                      weights_md()->data_type, dst_md()->data_type,
                      desc()->accum_data_type)
            && platform::has_data_type_support(data_type::f32);
    const bool is_fp16_ok
            = utils::everyone_is(data_type::f16, src_md()->data_type,
                      weights_md()->data_type, dst_md()->data_type)
            && platform::has_data_type_support(data_type::f16);
    const bool is_bf16_ok
            = utils::everyone_is(data_type::bf16, src_md()->data_type,
                      weights_md()->data_type, dst_md()->data_type)
            && platform::has_data_type_support(data_type::bf16);
    const bool is_bf16f32_ok
            = utils::everyone_is(data_type::bf16, src_md()->data_type,
                      weights_md()->data_type)
            && utils::everyone_is(data_type::f32, dst_md()->data_type)
            && platform::has_data_type_support(data_type::bf16);

    // we need to save this state as it can change inside set_default_formats()
    weights_format_kind_ = weights_md_.format_kind;

    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_MATMUL(utils::one_of(true, is_fp32_ok, is_fp16_ok, is_bf16_ok,
                             is_bf16f32_ok),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(attr()->has_default_values(
                             smask_t::post_ops | smask_t::fpmath_mode),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_MATMUL(
            !has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    if (weights_format_kind_ == format_kind::any) {
        CHECK(acl_matmul_utils::init_conf_matmul<true>(
                amp_, src_md_, weights_md_, dst_md_, *desc(), *attr()));
    } else {
        CHECK(acl_matmul_utils::init_conf_matmul<false>(
                amp_, src_md_, weights_md_, dst_md_, *desc(), *attr()));
    }

    // We can only fuse sum if it is the first post op and we aren't
    // transposing dst after
    if (attr_.post_ops_.contain(primitive_kind::sum, 0) && !amp_.do_transC) {
        // Check there isn't another sum after the first
        VDISPATCH_MATMUL(attr_.post_ops_.find(primitive_kind::sum, 1, -1) < 0,
                "cannot contain multiple sum post-ops");
        VDISPATCH_MATMUL(attr_.post_ops_.entry_[0].sum.scale == 1.0f,
                "sum post op scale must be 1 (no scale)");
        VDISPATCH_MATMUL(attr_.post_ops_.entry_[0].sum.zero_point == 0,
                "sum post op zero point must be 0 (no shift)");
        amp_.gemm_info.set_accumulate(true);
    }

    amp_.do_act = false;
    arm_compute::ActivationLayerInfo act_info;
    CHECK(acl_post_ops.init(engine, attr_.post_ops_, dst_md_, act_info,
            amp_.gemm_info.accumulate() ? 1 : 0));
    amp_.gemm_info.set_activation_info(act_info);
    if (act_info.enabled()
            && !arm_compute::experimental::op::ll::CpuGemmAssemblyDispatch::
                       is_activation_supported(act_info)) {
        auto dst_info_to_use
                = amp_.do_transC ? &amp_.dst_acc_info : &amp_.dst_tensor_info;
        ACL_CHECK_VALID(arm_compute::experimental::op::CpuActivation::validate(
                dst_info_to_use, dst_info_to_use, act_info));
        amp_.do_act = true;
    }
    amp_.use_dst_acc_for_sum = acl_post_ops.has_sum();

    // Validate ACL GEMM
    if (amp_.do_transC) {
        ACL_CHECK_VALID(
                arm_compute::experimental::op::ll::CpuGemmAssemblyDispatch::
                        validate(&amp_.wei_tensor_info, &amp_.src_tensor_info,
                                nullptr, &amp_.dst_acc_info, amp_.gemm_info));
    } else {
        ACL_CHECK_VALID(arm_compute::experimental::op::ll::
                        CpuGemmAssemblyDispatch::validate(&amp_.src_tensor_info,
                                &amp_.wei_tensor_info, nullptr,
                                &amp_.dst_tensor_info, amp_.gemm_info));
    }

    auto scratchpad = scratchpad_registry().registrar();
    arm_compute::experimental::MemoryRequirements aux_mem_req;

    // Query buffer memory requirement, if not using fixed-format kernel
    if (weights_format_kind_ != format_kind::any) {
        arm_compute::experimental::op::ll::CpuGemmAssemblyDispatch asm_gemm;
        if (amp_.do_transC) {
            asm_gemm.configure(&amp_.wei_tensor_info, &amp_.src_tensor_info,
                    nullptr, &amp_.dst_acc_info, amp_.gemm_info);
        } else {
            asm_gemm.configure(&amp_.src_tensor_info, &amp_.wei_tensor_info,
                    nullptr, &amp_.dst_tensor_info, amp_.gemm_info);
        }
        aux_mem_req = asm_gemm.workspace();
    }
    CHECK(acl_matmul_utils::init_scratchpad(
            scratchpad, amp_, src_md_, weights_md_, dst_md_, aux_mem_req));

    return status::success;
}

template <bool IsFixedFormat>
status_t acl_matmul_t::execute_forward(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src_base = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);

    const auto &amp = pd()->amp_;

    std::unique_lock<std::mutex> locker {mtx_, std::defer_lock};

    // Some of the underlying kernels used by ACL still require some state and
    // are not safe to be called in parallel with different execution contexts.
    // Eventually when all kernels are truly stateless, this guard can be
    // removed.
    if (!acl_obj_->asm_gemm.has_stateless_impl()) { locker.lock(); }

    bool is_transA = amp.is_transA;
    bool is_transB = amp.is_transB;
    bool do_transC = amp.do_transC;
    bool do_act = amp.do_act;
    bool use_dst_acc_for_sum = amp.use_dst_acc_for_sum;

    const auto scratchpad = ctx.get_scratchpad_grantor();

    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor = nullptr;
    arm_compute::Tensor dst_tensor;
    arm_compute::Tensor dst_acc_tensor;
    src_tensor.allocator()->init(amp.src_tensor_info);
    wei_tensor.allocator()->init(amp.wei_tensor_info);
    dst_tensor.allocator()->init(amp.dst_tensor_info);

    // If we have an unfused sum post op, put the result in a scratchpad tensor.
    // Result will be summed to the dst during acl_post_ops.execute
    auto dst_base = use_dst_acc_for_sum ? scratchpad.get<void>(
                            memory_tracking::names::key_matmul_dst_in_acc_dt)
                                        : CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    dst_tensor.allocator()->import_memory(dst_base);

    // Run transpose kernel
    if (is_transA && !is_transB) {
        arm_compute::Tensor src_acc_tensor;
        src_acc_tensor.allocator()->init(amp.src_acc_info);
        src_acc_tensor.allocator()->import_memory(
                const_cast<data_t *>(src_base));
        auto transA_scratch = scratchpad.get<void>(
                memory_tracking::names::key_matmul_src_trans);
        src_tensor.allocator()->import_memory(transA_scratch);
        arm_compute::ITensorPack transpose_pack;
        transpose_pack.add_tensor(
                arm_compute::TensorType::ACL_SRC, &src_acc_tensor);
        transpose_pack.add_tensor(
                arm_compute::TensorType::ACL_DST, &src_tensor);
        acl_obj_->transA.run(transpose_pack);
        wei_tensor.allocator()->import_memory(const_cast<data_t *>(wei_base));
        src_acc_tensor.allocator()->free();
    } else if (is_transB && !is_transA) {
        arm_compute::Tensor wei_acc_tensor;
        wei_acc_tensor.allocator()->init(amp.wei_acc_info);
        wei_acc_tensor.allocator()->import_memory(
                const_cast<data_t *>(wei_base));
        auto transB_scratch = scratchpad.get<void>(
                memory_tracking::names::key_matmul_wei_trans);
        wei_tensor.allocator()->import_memory(transB_scratch);
        arm_compute::ITensorPack transpose_pack;
        transpose_pack.add_tensor(
                arm_compute::TensorType::ACL_SRC, &wei_acc_tensor);
        transpose_pack.add_tensor(
                arm_compute::TensorType::ACL_DST, &wei_tensor);
        acl_obj_->transB.run(transpose_pack);
        src_tensor.allocator()->import_memory(const_cast<data_t *>(src_base));
        wei_acc_tensor.allocator()->free();
    } else if (is_transA && is_transB && !do_transC) {
        arm_compute::Tensor src_acc_tensor;
        arm_compute::Tensor wei_acc_tensor;
        src_acc_tensor.allocator()->init(amp.src_acc_info);
        src_acc_tensor.allocator()->import_memory(
                const_cast<data_t *>(src_base));
        wei_acc_tensor.allocator()->init(amp.wei_acc_info);
        wei_acc_tensor.allocator()->import_memory(
                const_cast<data_t *>(wei_base));
        auto transA_scratch = scratchpad.get<void>(
                memory_tracking::names::key_matmul_src_trans);
        auto transB_scratch = scratchpad.get<void>(
                memory_tracking::names::key_matmul_wei_trans);
        src_tensor.allocator()->import_memory(transA_scratch);
        wei_tensor.allocator()->import_memory(transB_scratch);
        arm_compute::ITensorPack transpose_packA;
        transpose_packA.add_tensor(
                arm_compute::TensorType::ACL_SRC, &src_acc_tensor);
        transpose_packA.add_tensor(
                arm_compute::TensorType::ACL_DST, &src_tensor);
        arm_compute::ITensorPack transpose_packB;
        transpose_packB.add_tensor(
                arm_compute::TensorType::ACL_SRC, &wei_acc_tensor);
        transpose_packB.add_tensor(
                arm_compute::TensorType::ACL_DST, &wei_tensor);
        acl_obj_->transA.run(transpose_packA);
        acl_obj_->transB.run(transpose_packB);
        src_acc_tensor.allocator()->free();
        wei_acc_tensor.allocator()->free();
    } else {
        src_tensor.allocator()->import_memory(const_cast<data_t *>(src_base));
        wei_tensor.allocator()->import_memory(const_cast<data_t *>(wei_base));
        if (do_transC) {
            auto transC_scratch = scratchpad.get<void>(
                    memory_tracking::names::key_matmul_dst_trans);
            dst_acc_tensor.allocator()->init(amp.dst_acc_info);
            dst_acc_tensor.allocator()->import_memory(transC_scratch);
        }
    }

    arm_compute::ITensorPack matmul_pack;
    if (do_transC) {
        matmul_pack.add_const_tensor(
                arm_compute::TensorType::ACL_SRC_0, &wei_tensor);
        matmul_pack.add_const_tensor(
                arm_compute::TensorType::ACL_SRC_1, &src_tensor);
        matmul_pack.add_tensor(arm_compute::TensorType::ACL_SRC_2, &bia_tensor);
        matmul_pack.add_tensor(
                arm_compute::TensorType::ACL_DST, &dst_acc_tensor);
    } else {
        matmul_pack.add_const_tensor(
                arm_compute::TensorType::ACL_SRC_0, &src_tensor);
        matmul_pack.add_const_tensor(
                arm_compute::TensorType::ACL_SRC_1, &wei_tensor);
        matmul_pack.add_tensor(arm_compute::TensorType::ACL_SRC_2, &bia_tensor);
        matmul_pack.add_tensor(arm_compute::TensorType::ACL_DST, &dst_tensor);
    }

    // Get pointer to scratchpad memory and create a workspace tensor for
    // CpuGemm. Fixed-format kernel does not need this workspace tensor.
    std::vector<arm_compute::Tensor> tmp_tensors(acl_obj_->aux_mem_req.size());
    if (!IsFixedFormat) {
        for (const auto &key : matmul_keys) {
            const auto id = key.first;
            if (acl_obj_->aux_mem_req[id].size > 0) {
                const auto info = arm_compute::TensorInfo(
                        arm_compute::TensorShape(
                                acl_obj_->aux_mem_req[id].size),
                        1, arm_compute::DataType::U8);
                auto buffer = scratchpad.get<void>(key.second);
                tmp_tensors[id].allocator()->init(
                        info, acl_obj_->aux_mem_req[id].alignment);
                tmp_tensors[id].allocator()->import_memory(buffer);
                matmul_pack.add_tensor(
                        acl_obj_->aux_mem_req[id].slot, &tmp_tensors[id]);
            }
        }
    }

    acl_obj_->asm_gemm.run(matmul_pack);
    if (do_act) {
        auto dst_to_use = do_transC ? &dst_acc_tensor : &dst_tensor;
        arm_compute::ITensorPack act_pack;
        act_pack.add_tensor(arm_compute::TensorType::ACL_SRC, dst_to_use);
        act_pack.add_tensor(arm_compute::TensorType::ACL_DST, dst_to_use);
        acl_obj_->act.run(act_pack);
    }

    if (do_transC) {
        arm_compute::ITensorPack transpose_packC;
        transpose_packC.add_tensor(
                arm_compute::TensorType::ACL_SRC, &dst_acc_tensor);
        transpose_packC.add_tensor(
                arm_compute::TensorType::ACL_DST, &dst_tensor);
        acl_obj_->transC.run(transpose_packC);
    }

    void *dst = dst_tensor.buffer();
    pd()->acl_post_ops.execute(ctx, dst);

    return status;
}

template status_t acl_matmul_t::execute_forward<true>(
        const exec_ctx_t &ctx) const;
template status_t acl_matmul_t::execute_forward<false>(
        const exec_ctx_t &ctx) const;

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

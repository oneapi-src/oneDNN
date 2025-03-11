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

#include "acl_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_eltwise_fwd_t::execute(exec_ctx_t &ctx) const {
    return execute_forward(ctx);
}

status_t acl_eltwise_fwd_t::init(engine_t *engine) {
    auto aep = pd()->aep;

    act_->configure(&aep.data_info, &aep.data_info, aep.act_info);

    return status::success;
}

const acl_eltwise_fwd_t::pd_t *acl_eltwise_fwd_t::pd() const {
    return static_cast<const pd_t *>(primitive_t::pd().get());
}

status_t acl_eltwise_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    const void *src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    return execute_forward(ctx, src, dst);
}

status_t acl_eltwise_fwd_t::execute_forward(
        const exec_ctx_t &ctx, const void *src, void *dst) const {

    auto aep = pd()->aep;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;

    src_tensor.allocator()->init(aep.data_info);
    src_tensor.allocator()->import_memory(const_cast<void *>(src));
    dst_tensor.allocator()->init(aep.data_info);
    dst_tensor.allocator()->import_memory(dst);

    arm_compute::ITensorPack act_pack;
    act_pack.add_tensor(arm_compute::TensorType::ACL_SRC, &src_tensor);
    act_pack.add_tensor(arm_compute::TensorType::ACL_DST, &dst_tensor);

    act_->run(act_pack);

    return status::success;
}

status_t acl_eltwise_fwd_t::pd_t::init(engine_t *engine) {
    using namespace utils;
    using namespace data_type;
    const memory_desc_wrapper src_d(src_md());

    bool ok = is_fwd() && one_of(src_d.data_type(), f32, f16, s32, s8)
            && !has_zero_dim_memory() && attr()->has_default_values()
            && set_default_formats_common() && src_d.is_dense()
            && src_d == memory_desc_wrapper(dst_md());
    if (!ok) return status::unimplemented;

    // Workaround for the inaccuracies caused by
    // logistic/soft_relu/elu/gelu_erf of ACL for fp16.
    // TODO: Relax the error bounds in eltwise checks, or rework these
    // fp16 operations in ACL for better accuracy.
    using namespace dnnl::impl::alg_kind;
    if (src_d.data_type() == f16
            && utils::one_of(desc_.alg_kind, eltwise_logistic,
                    eltwise_soft_relu, eltwise_elu, eltwise_gelu_erf)) {
        return status::unimplemented;
    }

    auto acl_data_t = acl_utils::get_acl_data_t(src_d.data_type());

    // Operator acts elementwise, so we only require that the product of
    // all the dimensions equals the total number of elements. We are
    // free to swap/combine dimensions. ACL performs SIMD parallelism
    // over the first dimension and thread parallelism over the second.
    // We pick a single dimension to thread over (taking the max of 2 to
    // reduce the chance of it being 1), with the remaining dimensions
    // to SIMD over.
    dim_t thread_dim = std::max(W(), ndims() >= 2 ? C() : 1);
    auto shape
            = arm_compute::TensorShape(src_d.nelems() / thread_dim, thread_dim);
    aep.data_info = arm_compute::TensorInfo(shape, 1, acl_data_t);

    CHECK(acl_utils::convert_to_acl_act(desc_, aep.act_info));

    ACL_CHECK_VALID(Op::validate(&aep.data_info, &aep.data_info, aep.act_info));

    return status::success;
}
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

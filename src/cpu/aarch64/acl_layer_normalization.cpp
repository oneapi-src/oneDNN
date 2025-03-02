/*******************************************************************************
* Copyright 2023, 2025 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_layer_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_layer_normalization_fwd_t::pd_t::init(engine_t *engine) {

    // dir and flags
    ACL_CHECK_SUPPORT(!is_fwd(), "ACL lnorm supports forward propagation only");
    ACL_CHECK_SUPPORT(is_training(), "ACL supports inference only for lnorm");
    ACL_CHECK_SUPPORT(
            use_global_stats(), "ACL does not support global stats with lnorm");
    ACL_CHECK_SUPPORT(use_scale() || use_shift(),
            "ACL does not support lnorm scale and shift");

    // attr-scales
    ACL_CHECK_SUPPORT(!attr()->has_default_values(),
            "ACL does not support scales attribute");

    // tag and stat_tag
    ACL_CHECK_SUPPORT(src_md()->ndims < 2 || src_md()->ndims > 5,
            "src tensor must have between 2 and 5 (inclusive) "
            "dimensions");

    // msdNorm only supports lnorm for src in a channels last format.
    // So if channels aren't last (ie. if they aren't dense),
    // then reorder into a channels last format
    std::string ref_implementation_guess = "simple:any";
    if (src_md()->format_desc.blocking.strides[ndims() - 1] != 1) {
        CHECK(memory_desc_init_by_tag(
                src_md_, get_channels_last_format(src_md_.ndims)));
        ref_implementation_guess = "ref:any";
    }
    if (dst_md_ != src_md_)
        // Make sure dst and src share a format
        CHECK(memory_desc_init_by_md_and_dt(
                dst_md_, src_md_, src_md()->data_type));
    if (!set_default_stat_md_format(src_md_)) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md_);
    const memory_desc_wrapper dst_d(dst_md_);

    ACL_CHECK_SUPPORT(src_d.has_zero_dim() || dst_d.has_zero_dim(),
            "data tensor(s) must not have a zero dimension");

    // data type
    ACL_CHECK_SUPPORT(
            src_d.data_type() != data_type::f32, "ACL Lnorm only supports F32");
    ACL_CHECK_SUPPORT(dst_d.data_type() != src_d.data_type(),
            "src and dst must share data types");

    // Problem shape
    int C = norm_axis(); // Channel dim size
    int X = src_d.nelems() / C; // Non-channel dims size

    ACL_CHECK_SUPPORT(!use_acl_heuristic(X, C, dnnl_get_max_threads(),
                              is_training(), ref_implementation_guess),
            "ACL is unoptimal in this case");

    anp.data_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(C, X), 1, arm_compute::DataType::F32);

    ACL_CHECK_VALID(arm_compute::NEMeanStdDevNormalizationLayer::validate(
            &anp.data_info, &anp.data_info, desc()->layer_norm_epsilon));

    return status::success;
}

status_t acl_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);

    arm_compute::Tensor data_tensor;

    auto const acp = pd()->anp;

    data_tensor.allocator()->init(acp.data_info);

    data_tensor.allocator()->import_memory(const_cast<float *>(src));

    arm_compute::ITensorPack pack;
    pack.add_tensor(arm_compute::TensorType::ACL_SRC_0, &data_tensor);

    acl_obj.get()->msdNorm.run(pack);

    data_tensor.allocator()->free();

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
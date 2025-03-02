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

acl_layer_normalization_fwd_t::acl_layer_normalization_fwd_t(const pd_t *apd)
    : primitive_t(apd)
    , acl_obj_(std::make_unique<
              arm_compute::experimental::op::CpuMeanStdDevNormalization>()) {}

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

    anp_data_info = arm_compute::TensorInfo(
            arm_compute::TensorShape(C, X), 1, arm_compute::DataType::F32);

    ACL_CHECK_VALID(
            arm_compute::experimental::op::CpuMeanStdDevNormalization::validate(
                    &anp_data_info, &anp_data_info,
                    desc()->layer_norm_epsilon));

    return status::success;
}

format_tag_t acl_layer_normalization_fwd_t::pd_t::get_channels_last_format(
        size_t ndim) {
    assert(ndim > 1 && ndim < 6);
    switch (ndim) {
        case 2: return format_tag::nc;
        case 3: return format_tag::tnc;
        case 4: return format_tag::ldnc;
        case 5: return format_tag::abcde;
        default: return format_tag::undef;
    }
}

bool acl_layer_normalization_fwd_t::pd_t::use_acl_heuristic(int X, int C,
        int threads, bool ref_has_stats,
        std::string &ref_implementation_guess) {
    // Above a certain C, acl is always better, and below a certain C,
    // acl is always worse. for C in between these two, whether acl is
    // better can be approximated with the workload (X*C) per thread.
    // The values here were derived empirically and all depend on
    // threads, whether ref can use provided stats, and which reference
    // implementation acl is competing with.

    int acl_competitive_C = C;
    int acl_better_C = C;
    int acl_better_XC_per_thread = X * C;

    if (ref_implementation_guess == "simple:any") {
        acl_competitive_C = 64;
        if (ref_has_stats) {
            acl_better_C = 4096;
            acl_better_XC_per_thread = threads == 1 ? 4096 : 8192;
        } else {
            acl_better_C = threads <= 2 ? 1024 : 4096;
            acl_better_XC_per_thread = threads == 1 ? 1024 : 4096;
        }
    } else if (ref_implementation_guess == "ref:any") {
        acl_competitive_C = 0;
        if (ref_has_stats) {
            if (threads == 1) {
                acl_better_C = 64;
            } else if (threads == 2) {
                acl_better_C = 256;
            } else {
                acl_better_C = 1024;
            }

            if (threads == 1) {
                acl_better_XC_per_thread = 256;
            } else if (threads <= 16) {
                acl_better_XC_per_thread = 512;
            } else {
                acl_better_XC_per_thread = 1024;
            }
        } else {
            if (threads == 1) {
                acl_better_C = 64;
                acl_better_XC_per_thread = 128;
            } else if (threads <= 32) {
                acl_better_C = 256;
                acl_better_XC_per_thread = 256;
            } else {
                acl_better_C = 1024;
                acl_better_XC_per_thread = 512;
            }
        }
    }

    return C > acl_competitive_C
            && (C > acl_better_C || X * C > acl_better_XC_per_thread * threads);
}

const acl_layer_normalization_fwd_t::pd_t *
acl_layer_normalization_fwd_t::pd() const {
    return (const pd_t *)primitive_t::pd().get();
}

status_t acl_layer_normalization_fwd_t::init(engine_t *engine) {
    //auto aep = pd()->anp_data_info;
    acl_obj_->configure(
            const_cast<arm_compute::TensorInfo *>(&pd()->anp_data_info),
            const_cast<arm_compute::TensorInfo *>(&pd()->anp_data_info),
            pd()->desc()->layer_norm_epsilon);
    return status::success;
}

status_t acl_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    const auto *src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto *dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;

    src_tensor.allocator()->init(pd()->anp_data_info);
    src_tensor.allocator()->import_memory(const_cast<float *>(src));
    dst_tensor.allocator()->init(pd()->anp_data_info);
    dst_tensor.allocator()->import_memory(dst);

    arm_compute::ITensorPack act_pack;
    act_pack.add_tensor(arm_compute::TensorType::ACL_SRC, &src_tensor);
    act_pack.add_tensor(arm_compute::TensorType::ACL_DST, &dst_tensor);
    acl_obj_->run(act_pack);

    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

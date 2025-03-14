/*******************************************************************************
* Copyright 2022-2023, 2025 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_POOLING_HPP
#define CPU_AARCH64_ACL_POOLING_HPP

#include "cpu/cpu_pooling_pd.hpp"

#include "cpu/aarch64/acl_utils.hpp"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/runtime/IOperator.h"
#include "arm_compute/runtime/experimental/operators/CpuPooling.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_pooling_conf_t {
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::PoolingLayerInfo pool_info;
    arm_compute::TensorInfo ws_info;
    bool use_ws;
};

struct acl_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;
        DECLARE_COMMON_PD_T("acl", acl_pooling_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            auto scratchpad = scratchpad_registry().registrar();
            CHECK(init_scratchpad(scratchpad));

            // ACL supports forward propagation only
            bool ok = set_default_params() == status::success && is_fwd()
                    && utils::everyone_is(
                            src_md()->data_type, dst_md()->data_type)
                    && utils::one_of(
                            src_md()->data_type, data_type::f32, data_type::f16)
                    && attr()->has_default_values()
                    && attr_.set_default_formats(dst_md(0)) == status::success
                    && !is_dilated() && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            const pooling_desc_t *pod = desc();

            // Choose the pooling type
            const alg_kind_t alg = pod->alg_kind;
            const bool is_max_pool = (alg == alg_kind::pooling_max);
            asp_.pool_info.pool_type = is_max_pool
                    ? arm_compute::PoolingType::MAX
                    : arm_compute::PoolingType::AVG;

            // Check if workspace Tensor is needed
            const bool ws_init = (is_max_pool
                    && pod->prop_kind == prop_kind::forward_training);
            asp_.use_ws = ws_init;

            ACL_CHECK_SUPPORT(ws_init && src_md()->data_type != data_type::f32,
                    "ACL Max pooling forward training only supports f32");

            if (ws_init)
                // ACL only supports U32/S32 no U8
                init_default_ws(data_type::s32);
            auto src_tag = memory_desc_matches_one_of_tag(
                    *src_md(), format_tag::nhwc, format_tag::nchw);
            auto dst_tag = memory_desc_matches_one_of_tag(
                    *dst_md(), format_tag::nhwc, format_tag::nchw);

            ACL_CHECK_SUPPORT(
                    utils::one_of(format_tag::undef, src_tag, dst_tag),
                    "src or dst is not format nhwc or nchw");
            ACL_CHECK_SUPPORT(src_tag != dst_tag,
                    "src and dst have different memory formats");

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            const int ndims = src_d.ndims();
            ACL_CHECK_SUPPORT(ndims != 4, "Tensor is not 4d");

            // Pooling window
            asp_.pool_info.pool_size = arm_compute::Size2D(KW(), KH());
            // Choose the data layout
            bool is_nhwc = src_tag == format_tag::nhwc;
            const auto acl_layout = is_nhwc ? arm_compute::DataLayout::NHWC
                                            : arm_compute::DataLayout::NCHW;
            asp_.pool_info.data_layout = acl_layout;
            const auto acl_data_t
                    = acl_utils::get_acl_data_t(src_d.data_type());

            bool use_square_acl_kernel = !is_nhwc && KH() == KW()
                    && (KH() == 2 || KH() == 3 || KH() == 7);
            if (is_max_pool) {
                ACL_CHECK_SUPPORT(
                        !use_acl_max_pool_heuristic(
                                MB() * IC() * OH() * OW() * KH() * KW(),
                                dnnl_get_max_threads(), is_nhwc,
                                use_square_acl_kernel,
                                pod->prop_kind == prop_kind::forward_training),
                        "ACL not used as profiling suggests that native oneDNN "
                        "kernels are faster for this problem");
            } else {
                ACL_CHECK_SUPPORT(!use_acl_avg_pool_heuristic(MB() * IC() * OH()
                                                  * OW() * KH() * KW(),
                                          dnnl_get_max_threads(), is_nhwc,
                                          use_square_acl_kernel),
                        "ACL not used as profiling suggests that native oneDNN "
                        "kernels are faster for this problem");
            }

            asp_.pool_info.exclude_padding
                    = (alg == alg_kind::pooling_avg_exclude_padding);

            asp_.pool_info.pad_stride_info = arm_compute::PadStrideInfo(KSW(),
                    KSH(), padL(), padR(), padT(), padB(),
                    arm_compute::DimensionRoundingType::FLOOR);

            asp_.src_info = arm_compute::TensorInfo(is_nhwc
                            ? arm_compute::TensorShape(IC(), IW(), IH(), MB())
                            : arm_compute::TensorShape(IW(), IH(), IC(), MB()),
                    1, acl_data_t, acl_layout);
            asp_.dst_info = arm_compute::TensorInfo(is_nhwc
                            ? arm_compute::TensorShape(OC(), OW(), OH(), MB())
                            : arm_compute::TensorShape(OW(), OH(), OC(), MB()),
                    1, acl_data_t, acl_layout);

            // Use datatype lowest property instead of using -INF
            asp_.pool_info.use_inf_as_limit = false;

            if (ws_init) {
                asp_.ws_info = arm_compute::TensorInfo(is_nhwc
                                ? arm_compute::TensorShape(
                                        OC(), OW(), OH(), MB())
                                : arm_compute::TensorShape(
                                        OW(), OH(), OC(), MB()),
                        1, arm_compute::DataType::U32, acl_layout);

                // Return kernel indices instead of source indices.
                asp_.pool_info.use_kernel_indices = true;
                ACL_CHECK_VALID(
                        arm_compute::experimental::op::CpuPooling::validate(
                                &asp_.src_info, &asp_.dst_info, asp_.pool_info,
                                &asp_.ws_info));
            } else {
                asp_.pool_info.use_kernel_indices = false;
                ACL_CHECK_VALID(
                        arm_compute::experimental::op::CpuPooling::validate(
                                &asp_.src_info, &asp_.dst_info,
                                asp_.pool_info));
            }

            return status::success;
        }

        // Generally, ACL is faster above a per thread problem size 'cutoff'.
        // The value of this cutoff has been found empirically,
        // through profiling on a Neoverse-N1 cpu.
        // Note: This rule is approximate, Not all problems follow this rule.
        //
        // Parameters used in the heuristics:
        // - problem_size: defined as mb * ic * oh * ow * kh * kw
        // - thread_count
        // - is_nhwc (as opposed to nchw)
        // - use_square_acl_kernels: For nchw pooling, acl has faster kernels
        //   for pooling window (kernel) sizes of 2x2, 3x3, or 7x7.
        // - is_training: Max pooling training cases require a workspace tensor,
        //   so these cases have been implemented in a seperate kernel in ACL.
        bool use_acl_avg_pool_heuristic(int problem_size, int thread_count,
                bool is_nhwc, bool use_square_acl_kernel) {
            int cutoff;
            if (is_nhwc) {
                if (thread_count == 1)
                    cutoff = 200;
                else
                    cutoff = 4096;
            } else {
                if (use_square_acl_kernel) {
                    if (thread_count == 1)
                        cutoff = 100;
                    else if (thread_count > 32)
                        return false;
                    else
                        cutoff = 2048;
                } else
                    return false;
            }
            return problem_size > cutoff * thread_count;
        }
        bool use_acl_max_pool_heuristic(int problem_size, int thread_count,
                bool is_nhwc, bool use_square_acl_kernel, bool is_training) {
            int cutoff;
            if (is_nhwc) {
                if (thread_count == 1)
                    cutoff = 200;
                else {
                    if (is_training)
                        cutoff = 2048;
                    else
                        cutoff = 4096;
                }
            } else {
                if (use_square_acl_kernel) {
                    if (thread_count == 1)
                        cutoff = 100;
                    else
                        cutoff = 1024;
                } else {
                    if (thread_count == 1)
                        return true;
                    else if (thread_count > 16)
                        return false;
                    else
                        cutoff = 25000;
                }
            }
            return problem_size > cutoff * thread_count;
        }

        acl_pooling_conf_t asp_;

        status_t init_scratchpad(memory_tracking::registrar_t &scratchpad) {
            const memory_desc_wrapper dst_d(&dst_md_);
            scratchpad.book(memory_tracking::names::key_pool_reduction,
                    dst_d.nelems(), sizeof(float));
            if (asp_.use_ws) {
                scratchpad.book(
                        memory_tracking::names::key_pool_ind_plain2blocked_cvt,
                        dst_d.nelems(), sizeof(uint32_t));
            }
            return status::success;
        }

    }; // pd_t

    // constructor
    acl_pooling_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t init(engine_t *engine) override;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<arm_compute::experimental::op::CpuPooling> pooling_op_;
}; // acl_pooling_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_POOLING_HPP

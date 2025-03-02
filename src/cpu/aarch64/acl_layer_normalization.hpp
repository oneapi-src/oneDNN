/*******************************************************************************
* Copyright 2023-2025 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_LAYER_NORMALIZATION_HPP
#define CPU_AARCH64_ACL_LAYER_NORMALIZATION_HPP

#include "arm_compute/runtime/experimental/operators/CpuMeanStdDevNormalization.h"
#include "cpu/aarch64/acl_utils.hpp"
#include "cpu/cpu_layer_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_msdnorm_obj_t {
    arm_compute::experimental::op::CpuMeanStdDevNormalization msdNorm;
};

struct acl_msdnorm_conf_t {
    arm_compute::TensorInfo data_info; // src and dst tensors
};

struct acl_layer_normalization_fwd_t : public primitive_t {
    using Op = arm_compute::experimental::op::CpuMeanStdDevNormalization;
    struct pd_t : public cpu_layer_normalization_fwd_pd_t {
        using cpu_layer_normalization_fwd_pd_t::
                cpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_layer_normalization_fwd_t);

        status_t init(engine_t *engine);

        format_tag_t get_channels_last_format(size_t ndim) {
            assert(ndim > 1 && ndim < 6);
            switch (ndim) {
                case 2: return format_tag::nc;
                case 3: return format_tag::tnc;
                case 4: return format_tag::ldnc;
                case 5: return format_tag::abcde;
                default: return format_tag::undef;
            }
        }

        bool use_acl_heuristic(int X, int C, int threads, bool ref_has_stats,
                std::string ref_implementation_guess) {
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
                    } else if (threads <= 32) {
                        acl_better_C = 256;
                    } else {
                        acl_better_C = 1024;
                    }

                    if (threads == 1) {
                        acl_better_XC_per_thread = 128;
                    } else if (threads <= 32) {
                        acl_better_XC_per_thread = 256;
                    } else {
                        acl_better_XC_per_thread = 512;
                    }
                }
            }

            return C > acl_competitive_C
                    && (C > acl_better_C
                            || X * C > acl_better_XC_per_thread * threads);
        }

        acl_msdnorm_conf_t anp = utils::zero<decltype(anp)>();

    }; // pd_t

    acl_layer_normalization_fwd_t(const pd_t *apd)
        : primitive_t(apd), acl_obj(std::make_unique<acl_msdnorm_obj_t>()) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<acl_msdnorm_obj_t> acl_obj;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_LAYER_NORMALIZATION_HPP
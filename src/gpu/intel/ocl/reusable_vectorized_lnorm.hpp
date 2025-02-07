/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_OCL_REUSABLE_VECTORIZED_LNORM_HPP
#define GPU_OCL_REUSABLE_VECTORIZED_LNORM_HPP

#include "common/c_types_map.hpp"
#include "common/layer_normalization_pd.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_layer_normalization_pd.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

//************* Common Reusable structs *************//

struct reusable_vectorized_lnorm_params_t
    : trivially_serializable_t<reusable_vectorized_lnorm_params_t> {

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names
                = {"lnorm_reusable_vectorized"};
        return kernel_names;
    }

    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), get_kernel_ctx());
        return status;
    }

    compute::kernel_ctx_t get_kernel_ctx() const;

    compute::dispatch_compile_params_t gws_params;
    /// Number of work items in a sub-group
    int sg_size;

    /// Number of elements to process in each work-item
    int vector_size;

    /// The number of times the loops need to unroll
    int unroll;

    data_type_t input_dt = data_type::undef;
    data_type_t output_dt = data_type::undef;
    data_type_t ss_dt = data_type::undef;

    bool use_scale = false;
    bool use_shift = false;

    /// If true calculate the mean and variance values. Otherwise reads from global memory
    bool calculate_stats = false;

    /// Saves the mean and variance to memory
    bool save_stats = false;

    uint8_t padding[4] = {false};
};

struct reusable_vectorized_lnorm_runtime_params_t {
    compute::dispatch_runtime_params_t gws_params;
};

//************* FWD implementation *************//

struct reusable_vectorized_layer_normalization_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_layer_normalization_fwd_pd_t {
        using gpu_layer_normalization_fwd_pd_t::
                gpu_layer_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:reusable:vectorized",
                reusable_vectorized_layer_normalization_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            data_type_t src_dt = src_md()->data_type;
            data_type_t dst_dt = dst_md()->data_type;

            const bool uses_f16 = utils::one_of(f16, src_dt, dst_dt);
            const bool uses_f64 = utils::one_of(f64, src_dt, dst_dt);

            /// TODO(umar): Can be implemented if we have a compatible WEI_TO_ACC macro in kernel
            if (uses_f64) return status::unimplemented;

            const bool f16_ok = IMPLICATION(uses_f16,
                    compute_engine->mayiuse(compute::device_ext_t::khr_fp16));
            const bool f64_ok = IMPLICATION(uses_f64,
                    compute_engine->mayiuse(compute::device_ext_t::khr_fp64));

            VDISPATCH_LNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM(f16_ok, VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "fp16");
            VDISPATCH_LNORM(f64_ok, VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "fp64");
            VDISPATCH_LNORM(check_scale_shift_data_type({f32, bf16, f16}),
                    VERBOSE_UNSUPPORTED_DT);

            using skip_mask_t = primitive_attr_t::skip_mask_t;
            VDISPATCH_LNORM(
                    attr()->has_default_values(skip_mask_t::scales_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_LNORM_SC(init_conf(engine), "Failed init_conf");

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);

        reusable_vectorized_lnorm_params_t conf;
        reusable_vectorized_lnorm_runtime_params_t rt_conf;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        return create_kernel(engine, calculate_lnorm_kernel_,
                pd()->conf.get_kernel_names()[0], pd()->conf);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t calculate_lnorm_kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

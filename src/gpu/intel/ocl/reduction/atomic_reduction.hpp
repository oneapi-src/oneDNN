/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REDUCTION_ATOMIC_REDUCTION_HPP
#define GPU_INTEL_OCL_REDUCTION_ATOMIC_REDUCTION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_reduction_pd.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/gpu_primitive_attr.hpp"
#include "gpu/intel/ocl/reduction/reduction_utils.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct atomic_reduction_key_params_t
    : trivially_serializable_t<atomic_reduction_key_params_t> {
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        compute::kernel_ctx_t kernel_ctx;
        CHECK(get_kernel_ctx(kernel_ctx));
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
        return status;
    }

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names = {"atomic_reduce"};
        return kernel_names;
    }

    status_t get_kernel_ctx(compute::kernel_ctx_t &) const;

    // Basic reduction parameters
    reduction_alg_kind_t alg, secondary_alg;
    data_type_t src_type, dst_type;

    // Implementation-specific parameters
    int32_t threads_per_eu;
    int32_t subgroup_size;
    int32_t vect_size;
    int32_t full_unroll_factor;
    int32_t tail_unroll_factor;
    int32_t global_acc;
    dim_t local_acc;

    compute::dispatch_compile_params_t params;
};
assert_trivially_serializable(atomic_reduction_key_params_t);

struct atomic_reduction_conf_t : public reduction_subproblem_t {
    atomic_reduction_conf_t(const reduction_subproblem_t &subprb,
            reduction_alg_kind_t alg, reduction_alg_kind_t secondary_alg,
            data_type_t src_type, data_type_t dst_type,
            const compute::device_info_t &device_info,
            gpu_primitive_attr_t *gpu_attr);
    status_t init_dispatcher(const compute::compute_engine_t *engine,
            const gpu_primitive_attr_t *gpu_attr);

    atomic_reduction_key_params_t conf;
    compute::dispatch_runtime_params_t rt_conf;
};

struct atomic_reduction_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_reduction_pd_t {
        using gpu_reduction_pd_t::gpu_reduction_pd_t;

        DECLARE_COMMON_PD_T("ocl:atomic", atomic_reduction_t);

        status_t init(impl::engine_t *engine) {
            using smask_t = primitive_attr_t::skip_mask_t;
            const auto attr_skip_mask = smask_t::gpu_attr;
            VDISPATCH_REDUCTION_SC(
                    set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_REDUCTION(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REDUCTION(!memory_desc_ndims_ok(src_md(), dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src", "dst");
            VDISPATCH_REDUCTION_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_REDUCTION(
                    !attr()->deterministic_, VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_REDUCTION_SC(init_conf(engine), "init_conf");
            init_scratchpad();

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_finalization_pd(impl::engine_t *engine);
        void init_scratchpad();

        int div = 0;
        float eps = 1, power = 1;
        std::vector<atomic_reduction_conf_t> phases;
        bool needs_finalization = false;
        std::shared_ptr<primitive_desc_t> eltwise_pd_;
    };

    status_t init(impl::engine_t *engine) override {
        auto &phases = pd()->phases;

        for (auto &phase : phases) {
            compute::kernel_t kernel;
            CHECK(create_kernel(engine, kernel, "atomic_reduce", phase.conf));
            kernels_.push_back(kernel);
        }

        if (pd()->needs_finalization) {
            CHECK(create_nested_primitive(
                    eltwise_p_, pd()->eltwise_pd_, engine));
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_atomic(ctx);
    }

private:
    status_t execute_atomic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const {
        return reinterpret_cast<const pd_t *>(primitive_t::pd().get());
    }

    std::vector<compute::kernel_t> kernels_;
    std::shared_ptr<impl::primitive_t> eltwise_p_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

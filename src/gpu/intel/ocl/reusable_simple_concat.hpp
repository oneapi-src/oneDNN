/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REUSABLE_SIMPLE_CONCAT_HPP
#define GPU_INTEL_OCL_REUSABLE_SIMPLE_CONCAT_HPP

#include "common/engine.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_concat_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct reusable_simple_concat_params_t
    : trivially_serializable_t<reusable_simple_concat_params_t> {

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names
                = {"reusable_simple_concat"};
        return kernel_names;
    }

    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), get_kernel_ctx());
        return status;
    }

    compute::kernel_ctx_t get_kernel_ctx() const;

    dim_t n_blocks;
    dim_t blocks[6];
    dim_t strides[6];

    dim_t read_block;
    dim_t write_block;
    int n;
    int simd;
    int data_type_size;
    bool use_large_index = true;
    uint8_t padding[3] = {0};
};

struct reusable_simple_concat_runtime_params_t {
    dim_t dst_extern_dim_size;
    dim_t dst_offset0;
    dim_t dst_ext_offset;
    dim_t src_extern_dim_sizes[64];
    dim_t offset[64];
    dim_t padded_offset[64];

    dim_t dst_concat_axis;
    dim_t dst_padded_concat_axis;
    dim_t read_overlap;
    dim_t gws0_block;
    dim_t inner_axis;

    compute::range_t gws_d = compute::range_t::one();
    compute::range_t lws_d;
};

struct reusable_simple_concat_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_concat_pd_t {
        using gpu_concat_pd_t::gpu_concat_pd_t;

        DECLARE_CONCAT_PD_T("simple:reusable", reusable_simple_concat_t);

        status_t init(impl::engine_t *engine) {
            VDISPATCH_CONCAT(n_inputs() <= 64, VERBOSE_BAD_PARAM, "n_inputs");
            VDISPATCH_CONCAT(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONCAT_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_CONCAT_SC(init_conf(engine),
                    VERBOSE_PRIMITIVE_CREATION_FAIL,
                    "reusable_concat init_conf");

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);

        reusable_simple_concat_params_t conf;
        reusable_simple_concat_runtime_params_t rt_conf;
    };

    status_t init(impl::engine_t *engine) override {
        CHECK(create_kernel(
                engine, kernel_, pd()->conf.get_kernel_names()[0], pd()->conf));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_concat(ctx);
    }

private:
    status_t execute_concat(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

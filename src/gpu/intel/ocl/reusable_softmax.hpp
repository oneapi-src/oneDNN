/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REUSABLE_SOFTMAX_HPP
#define GPU_INTEL_OCL_REUSABLE_SOFTMAX_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_softmax_pd.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/ocl_stream.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct reusable_softmax_params_t {
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), get_kernel_ctx());
        return status;
    }

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names
                = {"reusable_softmax_fwd_generic"};
        return kernel_names;
    }

#if __cplusplus >= 202002L
    bool operator==(const reusable_softmax_params_t &) const = default;
#endif
    serialized_t serialize() const {
        assert_trivially_serializable(reusable_softmax_params_t);
        return serialized_t(*this);
    }

    static reusable_softmax_params_t deserialize(const serialized_t &s) {
        return deserializer_t(s).pop<reusable_softmax_params_t>();
    }

    compute::kernel_ctx_t get_kernel_ctx() const;

    data_type_t src_data_type;
    data_type_t dst_data_type;
    bool is_logsoftmax;

    uint8_t padding[7] = {0};

    compute::dispatch_compile_params_t gws_params;
};

struct reusable_softmax_runtime_params_t {
    dim_t softmax_axis_stride;
    dim_t softmax_axis_size;
    compute::dispatch_runtime_params_t gws_params;
};

struct reusable_softmax_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_softmax_fwd_pd_t {
        using gpu_softmax_fwd_pd_t::gpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:reusable", reusable_softmax_fwd_t);

        status_t init(impl::engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            const auto src_dt = src_d.data_type();
            const auto dst_dt = dst_d.data_type();

            using namespace data_type;
            VDISPATCH_SOFTMAX(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_SOFTMAX(
                    utils::one_of(src_dt, f64, f32, f16, bf16, u8, s8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    utils::one_of(dst_dt, f32, f16, f64, bf16, u8, s8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(IMPLICATION(utils::one_of(f16, src_dt, dst_dt),
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SOFTMAX(
                    IMPLICATION(utils::one_of(data_type::f64, dst_dt, src_dt),
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::scales_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_SOFTMAX_SC(
                    set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);

            const memory_desc_wrapper src_mdw(src_md());
            int64_t ndims = static_cast<int64_t>(src_mdw.ndims());

            std::vector<compute::dim_id_t> src_dims(ndims),
                    dispatch_dims(ndims - 1);

            size_t dim_idx = 0;
            for (int64_t i = 0; i < ndims; i++) {
                src_dims[i] = i;
                if (i != desc()->softmax_axis) dispatch_dims[dim_idx++] = i;
            }

            // runtime parameters: reduction dimension size
            const auto &dims = src_mdw.dims();
            rt_conf.softmax_axis_size = dims[desc()->softmax_axis];

            // softmax stride from matching block (only one supported)
            block_layout_t layout(src_mdw);
            size_t num_matching_blocks = 0;
            for (const auto &block : layout) {
                if (block.dim_idx == desc()->softmax_axis) {
                    num_matching_blocks++;
                    rt_conf.softmax_axis_stride = block.stride;
                }
            }
            if (num_matching_blocks > 1) return status::unimplemented;

            conf.is_logsoftmax = is_logsoftmax();
            conf.src_data_type = src_dt;
            conf.dst_data_type = dst_dt;

            compute::named_buffer_t src_buf("SRC", *src_mdw.md_, src_dims);
            compute::named_buffer_t dst_buf("DST", src_buf);

            compute::reusable_dispatch_config_t dispatch_config(
                    compute_engine, dispatch_dims);
            CHECK(dispatch_config.register_buffer(src_buf));
            CHECK(dispatch_config.register_buffer(dst_buf));

            const auto *gpu_attr = utils::downcast<gpu_primitive_attr_t *>(
                    attr()->gpu_attr_.get());

            compute::reusable_dispatch_t dispatch;
            CHECK(dispatch_config.generate(dispatch,
                    compute::default_lws_strategy_t(compute_engine, gpu_attr)));
            conf.gws_params = dispatch.get_compile_params();
            rt_conf.gws_params = dispatch.get_runtime_params();

            return status::success;
        }

        reusable_softmax_params_t conf;
        reusable_softmax_runtime_params_t rt_conf;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;

        std::vector<compute::kernel_t> kernels;
        CHECK(create_kernels(
                engine, kernels, pd()->conf.get_kernel_names(), pd()->conf));
        kernel_ = kernels[0];

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_generic(ctx);
    }

protected:
    status_t execute_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

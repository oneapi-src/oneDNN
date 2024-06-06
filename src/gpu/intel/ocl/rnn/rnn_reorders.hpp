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

#ifndef GPU_INTEL_OCL_RNN_RNN_REORDERS_HPP
#define GPU_INTEL_OCL_RNN_RNN_REORDERS_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/ocl_utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct rnn_weights_reorder_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public reorder_pd_t {
        using reorder_pd_t::reorder_pd_t;

        DECLARE_COMMON_PD_T("cross_engine::rnn", rnn_weights_reorder_t);

        status_t init(impl::engine_t *engine, impl::engine_t *src_engine,
                impl::engine_t *dst_engine) {
            // Note: currently rnn_u8s8_compensation and rnn_s8s8_compensation
            // have common bit so we have to perform additional checks to
            // separate these two cases
            VDISPATCH_REORDER(
                    !IMPLICATION(dst_md()->extra.flags
                                    & memory_extra_flags::rnn_u8s8_compensation,
                            types::extra_flag_rnn_s8s8_compensation_is_set(
                                    dst_md()->extra.flags)),
                    VERBOSE_BAD_FLAGS);

            VDISPATCH_REORDER(utils::one_of(src_engine->kind(),
                                      engine_kind::gpu, engine_kind::cpu),
                    VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_REORDER(dst_engine->kind() == engine_kind::gpu,
                    VERBOSE_BAD_ENGINE_KIND);

            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(dst_engine);

            VDISPATCH_REORDER(compute_engine->mayiuse(
                                      compute::device_ext_t::intel_subgroups),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");
            VDISPATCH_REORDER(
                    IMPLICATION(
                            utils::one_of(data_type::f16, src_md()->data_type,
                                    dst_md()->data_type),
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short)),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_REORDER_SC(init_conf(engine), "init_conf()");
            init_scratchpad();
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        rnn_reorder_conf_t conf;

    private:
        DECLARE_GPU_REORDER_CREATE();

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();

            if (conf.do_reorder) {
                size_t sz = conf.nelems;
                scratchpad.book(memory_tracking::names::key_reorder_rnn_space,
                        sz, sizeof(float), OCL_BUFFER_ALIGNMENT);
            }
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        CHECK(create_kernel(engine, &kernel_, "wei_reorder", kernel_ctx));
        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    status_t init_res_storage(
            impl::engine_t *engine, gpu_resource_t *r) const override {
        if (!pd()->conf.do_reorder) return status::success;
        memory_storage_t *tmp_mem_storage_ptr = nullptr;
        size_t size = pd()->conf.scales_count * sizeof(float);
        CHECK(engine->create_memory_storage(&tmp_mem_storage_ptr, size));

        void *scales_ptr = nullptr;
        std::unique_ptr<memory_storage_t> tmp_mem_storage(tmp_mem_storage_ptr);
        CHECK(tmp_mem_storage->map_data(
                &scales_ptr, nullptr, sizeof(float) * pd()->conf.scales_count));
        utils::array_copy((float *)scales_ptr,
                pd()->attr()->rnn_weights_qparams_.scales_,
                pd()->conf.scales_count);
        CHECK(tmp_mem_storage->unmap_data(scales_ptr, nullptr));
        r->add_memory_storage(SCALES_, std::move(tmp_mem_storage));
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
    enum { SCALES_ = 0 };
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

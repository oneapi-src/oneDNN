/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_GEN9_SUM_HPP
#define GPU_INTEL_OCL_GEN9_SUM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/gpu_sum_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct gen9_sum_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;

        DECLARE_SUM_PD_T("ocl:gen9:any", gen9_sum_t);

        status_t init(impl::engine_t *engine) {
            const int n = n_inputs();

            VDISPATCH_SUM(n <= max_num_arrs, "too many inputs for primitive");

            const memory_desc_wrapper o_d(dst_md());

            // for IO bytes less than 1MB fall back into many_inputs_sum kernel for better performance.
            size_t io_bytes = (n + 1) * o_d.data_type_size() * o_d.nelems(true);
            if (io_bytes < 1024 * 1024) return status::unimplemented;

            VDISPATCH_SUM_SC(
                    gpu_sum_pd_t::init(engine), VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_SUM(!memory_desc_ndims_ok(dst_md()), VERBOSE_BAD_NDIMS,
                    "dst", dst_md()->ndims);

            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(src_md(i));
                VDISPATCH_SUM(i_d == o_d, VERBOSE_INCONSISTENT_DIM, "i_d", i,
                        "o_d", i);
            }

            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        const memory_desc_wrapper data_d(pd()->dst_md());
        const memory_desc_wrapper data_s(pd()->src_md());

        kernel_ctx.set_data_type(data_s.data_type());
        size_t io_bytes = (pd()->n_inputs() + 1) * data_d.data_type_size()
                * data_d.nelems(true);
        // Heuristics: for IO bytes smaller than 10MB reduce vector size for better perf.
        if (io_bytes < 10 * 1024 * 1024) { vector_size /= 2; }
        kernel_ctx.define_int("VECT_DT_N", vector_size);
        kernel_ctx.define_int("N_INPUTS", pd()->n_inputs());
        kernel_ctx.define_int("N_ELEMS", data_d.nelems(true));

        def_memory_desc_info(
                kernel_ctx, memory_desc_info_t::create(data_d), "SRC");
        def_memory_desc_info(
                kernel_ctx, memory_desc_info_t::create(data_s), "DST");

        CHECK(create_kernel(engine, &kernel_, "gen9_sum", kernel_ctx));

        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    status_t init_res_storage(
            impl::engine_t *engine, gpu_resource_t *r) const override {
        const dim_t count = pd()->n_inputs();
        const float *s_data = pd()->scales();

        const size_t size = count * sizeof(float);
        std::unique_ptr<memory_storage_t> scales;
        memory_storage_t *scale = nullptr;
        auto s = engine->create_memory_storage(&scale, size);
        if (s != status::success) return s;
        float *mapped_mem_storage = nullptr;
        s = scale->map_data((void **)&mapped_mem_storage, nullptr, size);
        if (s != status::success) return s;
        utils::array_copy(mapped_mem_storage, s_data, count);
        s = scale->unmap_data((void *)mapped_mem_storage, nullptr);
        if (s != status::success) return s;
        scales.reset(scale);
        r->add_memory_storage(SCALES_, std::move(scales));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    enum { max_num_arrs = 16 };
    int vector_size = 8;
    enum { SCALES_ = 0 };
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_SIMPLE_SUM_HPP
#define GPU_INTEL_OCL_SIMPLE_SUM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_sum_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

template <data_type_t data_type>
struct simple_sum_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_sum_pd_t {
        using gpu_sum_pd_t::gpu_sum_pd_t;

        DECLARE_SUM_PD_T("ocl:simple:any", simple_sum_t);

        status_t init(impl::engine_t *engine) {
            const int n = n_inputs();

            VDISPATCH_SUM_SC(
                    gpu_sum_pd_t::init(engine), VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_SUM(n <= max_num_arrs, "too many inputs for primitive");

            const memory_desc_wrapper o_d(dst_md());
            VDISPATCH_SUM(
                    o_d.data_type() == data_type, VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_SUM(o_d.is_dense(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

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
        CHECK(create_kernel(engine, &kernel_, "simple_sum", kernel_ctx));
        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

    enum { max_num_arrs = 16 };
    using data_t = typename prec_traits_t<data_type>::type;

private:
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

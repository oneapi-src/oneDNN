/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_OCL_SIMPLE_SUM_HPP
#define GPU_OCL_SIMPLE_SUM_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/jit_primitive_conf.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_sum_pd.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

template <data_type_t data_type>
struct simple_sum_t : public primitive_impl_t {
    struct pd_t : public ocl_sum_pd_t {
        using ocl_sum_pd_t::ocl_sum_pd_t;

        DECLARE_SUM_PD_T("ocl:simple:any", simple_sum_t);

        status_t init() {
            const int n = n_inputs();

            bool ok = true && ocl_sum_pd_t::init() == status::success
                    && n <= max_num_arrs;
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper o_d(dst_md());
            ok = ok && o_d.data_type() == data_type && o_d.is_dense();
            if (!ok) return status::unimplemented;

            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(src_md(i));
                if (i_d != o_d) return status::unimplemented;
            }

            return status::success;
        }
        jit_simple_sum_conf_t jss_;
    };

    simple_sum_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        compute_engine->create_kernel(&kernel_, "simple_sum", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override;

    enum { max_num_arrs = 16 };
    typedef typename prec_traits<data_type>::type data_t;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

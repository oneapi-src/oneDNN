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

#ifndef GPU_OCL_JIT_SIMPLE_SUM_KERNEL_HPP
#define GPU_OCL_JIT_SIMPLE_SUM_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct jit_simple_sum_kernel {

    jit_simple_sum_kernel(const jit_simple_sum_conf_t &ajss) : jss(ajss) {}

    ~jit_simple_sum_kernel() {}

    static status_t init_conf(jit_simple_sum_conf_t &jss, const sum_pd_t *pd) {
        UNUSED(jss);
        UNUSED(pd);
        return status::success;
    };

    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            const jit_simple_sum_conf_t &jss) {
        UNUSED(kernel_ctx);
        UNUSED(jss);
        return status::success;
    }

    jit_simple_sum_conf_t jss;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

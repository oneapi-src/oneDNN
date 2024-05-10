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

#ifndef GPU_INTEL_JIT_JIT_GENERATOR_BASE_HPP
#define GPU_INTEL_JIT_JIT_GENERATOR_BASE_HPP

#include <vector>
#include <CL/cl.h>

#include "xpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct jit_generator_base {
    virtual ~jit_generator_base() = default;
    virtual const char *kernel_name() const = 0;
    virtual xpu::binary_t get_binary(cl_context context, cl_device_id device)
            = 0;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_JIT_JIT_GENERATOR_BASE_HPP

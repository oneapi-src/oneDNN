/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_JIT_IR_KERNEL_DESC_HPP
#define GPU_JIT_IR_KERNEL_DESC_HPP

#include "gpu/compute/compute_engine.hpp"
#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/ir/hw.hpp"
#include "gpu/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_primitive_t;

namespace compute {
class kernel_t;
}

namespace jit {

class kernel_info_t;

class kernel_desc_base_t {
public:
    virtual ~kernel_desc_base_t() = default;
    virtual std::string kernel_name() const = 0;
    virtual exec_config_t exec_cfg() const = 0;
    virtual bool with_dpas() const = 0;
    virtual status_t init_kernel_info(kernel_info_t &kernel_info) const = 0;
    virtual status_t create_kernel(compute::kernel_t &kernel,
            gpu_primitive_t *primitive, engine_t *engine) const = 0;
    virtual serialized_t serialize() const = 0;
    hw_t hw() const { return exec_cfg().hw(); }
};

class kernel_params_base_t {
public:
    virtual ~kernel_params_base_t() = default;
    virtual status_t init_dispatch_kernel_info(kernel_info_t &kernel_info,
            const kernel_desc_base_t &desc) const = 0;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

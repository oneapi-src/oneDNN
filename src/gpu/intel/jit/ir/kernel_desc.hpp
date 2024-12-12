/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_KERNEL_DESC_HPP
#define GPU_INTEL_JIT_IR_KERNEL_DESC_HPP

#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/jit/ir/fma.hpp"
#include "gpu/intel/jit/ir/hw.hpp"
#include "gpu/intel/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

struct gpu_primitive_t;

namespace compute {
class kernel_t;
}

namespace jit {

class kernel_iface_t;
class kernel_info_t;
class kernel_params_base_t;

class kernel_desc_base_t {
public:
    virtual ~kernel_desc_base_t() = default;
    virtual std::string kernel_name() const = 0;
    virtual exec_config_t exec_cfg(const impl::engine_t *engine) const = 0;
    virtual bool with_dpas() const = 0;
    virtual compute::range_t local_range() const = 0;
    virtual void init_kernel_iface(kernel_iface_t &kernel_iface) const = 0;
    virtual void init_kernel_info(kernel_info_t &kernel_info,
            const kernel_params_base_t &params,
            const impl::engine_t *engine) const = 0;
    virtual status_t create_kernel(compute::kernel_t &kernel,
            gpu_primitive_t *primitive, impl::engine_t *engine) const = 0;
    virtual serialized_t serialize() const = 0;
};

class kernel_params_base_t {
public:
    virtual ~kernel_params_base_t() = default;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

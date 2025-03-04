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

#ifndef GPU_INTEL_OCL_DEVICE_INFO_HPP
#define GPU_INTEL_OCL_DEVICE_INFO_HPP

#include <string>
#include <vector>
#include <CL/cl.h>

#include "gpu/intel/compute/device_info.hpp"
#include "gpu/intel/ocl/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

class device_info_t : public compute::device_info_t {
public:
    std::string get_cl_ext_options() const;

protected:
    status_t init_device_name(impl::engine_t *engine) override;
    status_t init_arch(impl::engine_t *engine) override;
    status_t init_runtime_version(impl::engine_t *engine) override;
    status_t init_extensions(impl::engine_t *engine) override;
    status_t init_attributes(impl::engine_t *engine) override;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_INTEL_OCL_DEVICE_INFO_HPP

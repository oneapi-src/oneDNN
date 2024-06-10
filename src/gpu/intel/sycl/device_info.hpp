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

#ifndef SYCL_DEVICE_INFO_HPP
#define SYCL_DEVICE_INFO_HPP

#include "gpu/intel/compute/device_info.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

class device_info_t : public gpu::intel::compute::device_info_t {
protected:
    status_t init_device_name(impl::engine_t *engine) override;
    status_t init_arch(impl::engine_t *engine) override;
    status_t init_runtime_version(impl::engine_t *engine) override;
    status_t init_extensions(impl::engine_t *engine) override;
    status_t init_attributes(impl::engine_t *engine) override;
};

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // SYCL_DEVICE_INFO_HPP

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

#ifndef SYCL_DEVICE_INFO_HPP
#define SYCL_DEVICE_INFO_HPP

#include <vector>
#include <CL/sycl.hpp>

#include "ocl/ocl_device_info.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace sycl {

class sycl_device_info_t : public ocl::ocl_device_info_t {
public:
    sycl_device_info_t(const cl::sycl::device &dev)
        : ocl::ocl_device_info_t(ocl::ocl_utils::make_ocl_wrapper(dev.get())) {}
};

} // namespace sycl
} // namespace impl
} // namespace mkldnn

#endif // SYCL_DEVICE_INFO_HPP

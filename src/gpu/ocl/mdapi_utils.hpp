/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GPU_OCL_MDAPI_UTILS_HPP
#define GPU_OCL_MDAPI_UTILS_HPP

#include <memory>
#include <CL/cl.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

class mdapi_helper_impl_t;

class mdapi_helper_t {
public:
    mdapi_helper_t();
    cl_command_queue create_queue(
            cl_context cl_ctx, cl_device_id dev, cl_int *err) const;
    double get_freq(cl_event event) const;

private:
    std::shared_ptr<mdapi_helper_impl_t> impl_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

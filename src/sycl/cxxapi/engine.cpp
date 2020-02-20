/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "dnnl.hpp"
#include <CL/sycl.hpp>

#include "sycl/capi.hpp"

namespace dnnl {

engine::engine(
        kind akind, const cl::sycl::device &dev, const cl::sycl::context &ctx) {
    dnnl_engine_t aengine;
    error::wrap_c_api(dnnl_engine_create_sycl(&aengine, convert_to_c(akind),
                              static_cast<const void *>(&dev),
                              static_cast<const void *>(&ctx)),
            "could not create an engine");
    reset(aengine);
}

cl::sycl::context engine::get_sycl_context() const {
    void *ctx_ptr;
    error::wrap_c_api(dnnl_engine_get_sycl_context(get(), &ctx_ptr),
            "could not get a context handle");
    auto ctx = *static_cast<cl::sycl::context *>(ctx_ptr);
    return ctx;
}

cl::sycl::device engine::get_sycl_device() const {
    void *dev_ptr;
    error::wrap_c_api(dnnl_engine_get_sycl_device(get(), &dev_ptr),
            "could not get a device handle");
    auto dev = *static_cast<cl::sycl::device *>(dev_ptr);
    return dev;
}

} // namespace dnnl

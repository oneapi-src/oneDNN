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

#ifndef CL_ENGINE_HPP
#define CL_ENGINE_HPP

#include <CL/cl.h>

#include "common/engine.hpp"
#include "ocl/cl_device_info.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

// High-level interface to OpenCL-compatible engines.
//
// The interface supports obtaining OpenCL device/context objects and allows
// to make queries to determine if an OpenCL extension is supported.
class cl_engine_t : public engine_t
{
public:
    cl_engine_t(engine_kind_t kind, backend_kind_t backend_kind,
            const cl_device_info_t &device_info)
        : engine_t(kind, backend_kind), device_info_(device_info) {}

    status_t init() {
        OCL_CHECK(device_info_.init());
        return status::success;
    }

    virtual cl_device_id ocl_device() const = 0;
    virtual cl_context ocl_context() const = 0;

    bool mayiuse(cl_device_ext_t ext) const { return device_info_.has(ext); }

    int get_eu_count() const { return device_info_.eu_count(); }
    int get_hw_threads() const { return device_info_.hw_threads(); }

    const runtime_version_t get_runtime_version() const {
        return device_info_.runtime_version();
    }

private:
    cl_device_info_t device_info_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

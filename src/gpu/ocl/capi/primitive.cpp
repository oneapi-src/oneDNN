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

#include <CL/cl.h>

#include "oneapi/dnnl/dnnl_ocl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory.hpp"
#include "common/utils.hpp"

#include "gpu/ocl/ocl_c_types_map.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"

using namespace dnnl::impl;

status_t dnnl_ocl_interop_primitive_execute(
        const primitive_iface_t *primitive_iface, stream_t *stream, int nargs,
        const dnnl_exec_arg_t *args, const cl_event *deps, int ndeps,
        cl_event *return_event_) {

    return status::unimplemented;
}

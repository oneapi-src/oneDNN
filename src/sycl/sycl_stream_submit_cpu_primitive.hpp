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

#ifndef SYCL_STREAM_SUBMIT_CPU_DISPATCH_HPP
#define SYCL_STREAM_SUBMIT_CPU_DISPATCH_HPP

#include "common/c_types_map.hpp"
#include "sycl/sycl_stream_cpu_thunk.hpp"
#include "xpu/sycl/utils.hpp"

#include <vector>

namespace dnnl {
namespace impl {
namespace sycl {

void submit_cpu_primitive(stream_t *stream, const primitive_iface_t *prim_iface,
        const exec_ctx_t &exec_ctx, ::sycl::handler &cgh);

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif

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

#include <CL/sycl.hpp>

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

void register_profiling_event(const ::sycl::event &event);
status_t get_profiling_time(uint64_t *nsec);
status_t reset_profiling();

} // namespace sycl
} // namespace impl
} // namespace dnnl

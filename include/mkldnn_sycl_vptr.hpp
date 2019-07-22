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

#ifndef MKLDNN_SYCL_VPTR_HPP
#define MKLDNN_SYCL_VPTR_HPP

#include <CL/sycl.hpp>

#include "mkldnn.h"

namespace mkldnn {

void MKLDNN_API *sycl_malloc(size_t size);

void MKLDNN_API sycl_free(void *ptr);

bool MKLDNN_API is_sycl_vptr(void *ptr);

cl::sycl::buffer<uint8_t, 1> MKLDNN_API get_sycl_buffer(void *ptr);

size_t MKLDNN_API get_sycl_offset(void *ptr);

} // namespace mkldnn

#endif

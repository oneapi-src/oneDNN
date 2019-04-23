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

#include "mkldnn_sycl_vptr.hpp"

#include "sycl/virtual_ptr.hpp"

namespace mkldnn {
static cl::sycl::codeplay::PointerMapper &get_pointer_mapper_instance() {
    static cl::sycl::codeplay::PointerMapper pointer_mapper;
    return pointer_mapper;
}

void *sycl_malloc(size_t size) {
    auto &p_map = get_pointer_mapper_instance();
    return cl::sycl::codeplay::SYCLmalloc(size, p_map);
}

void sycl_free(void *ptr) {
    auto &p_map = get_pointer_mapper_instance();
    return cl::sycl::codeplay::SYCLfree(ptr, p_map);
}

bool is_sycl_vptr(void *ptr) {
    auto &p_map = get_pointer_mapper_instance();
    return p_map.is_vptr(ptr);
}

cl::sycl::buffer<uint8_t, 1> get_sycl_buffer(void *ptr) {
    auto &p_map = get_pointer_mapper_instance();
    return p_map.get_buffer<uint8_t>(ptr);
}

size_t get_sycl_offset(void *ptr) {
    auto &p_map = get_pointer_mapper_instance();
    return p_map.get_offset(ptr);
}

} // namespace mkldnn

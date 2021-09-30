/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef EXAMPLE_UTILS_HPP
#define EXAMPLE_UTILS_HPP

#include <numeric>

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

inline int64_t product(const std::vector<int64_t> &dims) {
    return dims.empty() ? 0
                        : std::accumulate(dims.begin(), dims.end(), (int64_t)1,
                                std::multiplies<int64_t>());
}

#ifdef DNNL_GRAPH_WITH_SYCL
template <typename dtype>
void fill_buffer(
        cl::sycl::queue &q, void *usm_buffer, size_t length, dtype value) {
    dtype *usm_buffer_casted = static_cast<dtype *>(usm_buffer);
    q.parallel_for(cl::sycl::range<1>(length), [=](cl::sycl::id<1> i) {
         int idx = (int)i[0];
         usm_buffer_casted[idx] = value;
     }).wait();
}

void *sycl_malloc_wrapper(size_t n, const void *dev, const void *ctx,
        dnnl::graph::allocator::attribute attr) {
    return malloc_device(n, *static_cast<const cl::sycl::device *>(dev),
            *static_cast<const cl::sycl::context *>(ctx));
}

void sycl_free_wrapper(void *ptr, const void *context) {
    free(ptr, *static_cast<const cl::sycl::context *>(context));
}
#endif

#endif

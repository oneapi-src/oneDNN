/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "dnnl.hpp"

namespace dnnl {
namespace sycl {

engine engine_create(engine::kind kind, const cl::sycl::device &dev,
        const cl::sycl::context &ctx);
cl::sycl::device engine_get_device(const engine &e);
cl::sycl::context engine_get_context(const engine &e);

stream stream_create(const engine &e, cl::sycl::queue &queue);
cl::sycl::queue stream_get_queue(const stream &s);

template <typename T, int ndims>
memory memory_create(cl::sycl::buffer<T, ndims> buf, stream &s);
template <typename T, int ndims>
void memory_set_data_handle(memory &m, cl::sycl::buffer<T, ndims> b, stream &s);
template <typename T, int ndims>
cl::sycl::buffer<T, ndims> memory_get_data_handle(const memory &m);

cl::sycl::event execute(const primitive &p, const stream &s,
        const std::unordered_map<int, memory> &args,
        const std::vector<cl::sycl::event> &dependencies = {});

} // namespace sycl
} // namespace dnnl

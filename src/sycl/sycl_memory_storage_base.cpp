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

#include "sycl/sycl_memory_storage_base.hpp"
#include "sycl/sycl_stream.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

namespace {
template <::sycl::access_mode mode>
gpu::sycl::sycl_memory_arg_t<mode> get_empty_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) {
    using arg_type = gpu::sycl::sycl_memory_arg_t<mode>;
    auto *sycl_stream = utils::downcast<sycl_stream_t *>(stream);
    return arg_type::create_empty(sycl_stream->get_dummy_accessor<mode>(cgh));
}
} // namespace

gpu::sycl::sycl_in_memory_arg_t sycl_memory_storage_base_t::empty_in_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) {
    return get_empty_memory_arg<::sycl::access::mode::read>(stream, cgh);
}

gpu::sycl::sycl_out_memory_arg_t
sycl_memory_storage_base_t::empty_out_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) {
    return get_empty_memory_arg<::sycl::access::mode::write>(stream, cgh);
}

gpu::sycl::sycl_inout_memory_arg_t
sycl_memory_storage_base_t::empty_inout_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) {
    return get_empty_memory_arg<::sycl::access::mode::read_write>(stream, cgh);
}

} // namespace sycl
} // namespace impl
} // namespace dnnl

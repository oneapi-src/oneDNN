/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#include "common/stream.hpp"

#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/stream_impl.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

namespace {
template <::sycl::access_mode mode>
memory_arg_t<mode> get_empty_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) {
    using arg_type = memory_arg_t<mode>;
    auto *sycl_stream_impl
            = utils::downcast<xpu::sycl::stream_impl_t *>(stream->impl());
    return arg_type::create_empty(
            sycl_stream_impl->get_dummy_accessor<mode>(cgh));
}
} // namespace

in_memory_arg_t memory_storage_base_t::empty_in_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) {
    return get_empty_memory_arg<::sycl::access::mode::read>(stream, cgh);
}

out_memory_arg_t memory_storage_base_t::empty_out_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) {
    return get_empty_memory_arg<::sycl::access::mode::write>(stream, cgh);
}

inout_memory_arg_t memory_storage_base_t::empty_inout_memory_arg(
        stream_t *stream, ::sycl::handler &cgh) {
    return get_empty_memory_arg<::sycl::access::mode::read_write>(stream, cgh);
}
} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl

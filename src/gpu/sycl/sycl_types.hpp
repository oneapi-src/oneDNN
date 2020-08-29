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

#ifndef GPU_SYCL_SYCL_TYPES_HPP
#define GPU_SYCL_SYCL_TYPES_HPP

#include "sycl/sycl_compat.hpp"
#include "sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

template <::sycl::access_mode mode>
struct sycl_memory_arg_t {
    using acc_t = ::sycl::accessor<uint8_t, 1, mode>;

    sycl_memory_arg_t(void *usm, const acc_t &dummy_acc)
        : usm_(usm), acc_(dummy_acc) {}
    sycl_memory_arg_t(const acc_t &acc) : usm_(nullptr), acc_(acc) {}
    // This method must be called only from inside a kernel.
    void *get_pointer() { return usm_ ? usm_ : acc_.get_pointer().get(); }

private:
    void *usm_;
    acc_t acc_;
};

// TODO: come up with better names?
using sycl_in_memory_arg_t = sycl_memory_arg_t<::sycl::access::mode::read>;
using sycl_out_memory_arg_t = sycl_memory_arg_t<::sycl::access::mode::write>;
using sycl_inout_memory_arg_t
        = sycl_memory_arg_t<::sycl::access::mode::read_write>;

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif

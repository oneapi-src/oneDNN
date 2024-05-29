/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef XPU_OCL_STREAM_IMPL_HPP
#define XPU_OCL_STREAM_IMPL_HPP

#include "common/engine.hpp"
#include "common/stream_impl.hpp"
#include "common/utils.hpp"

#include "xpu/ocl/engine_impl.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

class stream_impl_t : public impl::stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags)
        : impl::stream_impl_t(flags), queue_(nullptr) {}
    stream_impl_t(cl_command_queue queue, unsigned flags)
        : impl::stream_impl_t(flags), queue_(queue) {
        assert(queue_);
    }

    ~stream_impl_t() override = default;

    status_t set_queue(cl_command_queue queue) {
        queue_ = queue;
        return status::success;
    }

    cl_command_queue queue() { return queue_; }

private:
    xpu::ocl::wrapper_t<cl_command_queue> queue_;
};

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif

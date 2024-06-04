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

#ifndef COMMON_STREAM_IMPL_HPP
#define COMMON_STREAM_IMPL_HPP

#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

class stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags) : flags_(flags) {}
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    stream_impl_t(threadpool_interop::threadpool_iface *threadpool)
        : flags_(stream_flags::in_order), threadpool_(threadpool) {}
#endif

    virtual ~stream_impl_t() = default;

    unsigned flags() const { return flags_; }

    bool is_profiling_enabled() const {
        return (flags() & dnnl::impl::stream_flags::profiling);
    }

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    status_t get_threadpool(
            threadpool_interop::threadpool_iface **threadpool) const {
        *threadpool = threadpool_;
        return status::success;
    }
#endif

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(stream_impl_t)

    unsigned flags_;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    threadpool_interop::threadpool_iface *threadpool_ = nullptr;
#endif
};

} // namespace impl
} // namespace dnnl

#endif

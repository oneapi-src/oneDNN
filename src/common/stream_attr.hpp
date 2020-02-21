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

#ifndef STREAM_ATTR_HPP
#define STREAM_ATTR_HPP

#include <cassert>
#include "dnnl.h"
#include "dnnl_threadpool_iface.hpp"

#include "c_types_map.hpp"
#include "nstl.hpp"

struct dnnl_stream_attr : public dnnl::impl::c_compatible {
    dnnl_stream_attr(dnnl::impl::engine_kind_t kind) : kind_(kind) {}

    dnnl::impl::status_t set_threadpool(dnnl::threadpool_iface *threadpool) {
        using namespace dnnl::impl;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        if (kind_ != engine_kind::cpu) return status::invalid_arguments;
        threadpool_ = threadpool;
        return status::success;
#else
        return status::invalid_arguments;
#endif
    }

    dnnl::impl::status_t get_threadpool(
            dnnl::threadpool_iface **threadpool) const {
        using namespace dnnl::impl;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        if (kind_ != engine_kind::cpu) return status::invalid_arguments;
        *threadpool = threadpool_;
        return status::success;
#else
        return status::invalid_arguments;
#endif
    }

    dnnl::impl::engine_kind_t get_engine_kind() { return kind_; }

private:
    dnnl::impl::engine_kind_t kind_;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    dnnl::threadpool_iface *threadpool_ = nullptr;
#endif
};

#endif

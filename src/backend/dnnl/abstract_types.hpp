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

#ifndef LLGA_BACKEND_DNNL_ABSTRACT_TYPES_HPP
#define LLGA_BACKEND_DNNL_ABSTRACT_TYPES_HPP

#include <cstdlib>
#include <cstring>
#include <dnnl.h>
#include <dnnl.hpp>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

#include "interface/allocator.hpp"
#include "interface/engine.hpp"

#if DNNL_GRAPH_WITH_SYCL
#include "dnnl_sycl.hpp"
#endif

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

using error = dnnl::error;
using memory = dnnl::memory;
using format_tag = memory::format_tag;
using tag = memory::format_tag;
using data_type = typename memory::data_type;
using dims = typename memory::dims;
using dim = memory::dim;
using query = dnnl::query;
using kind = dnnl::primitive::kind;
using prop_kind = dnnl::prop_kind;
using algorithm = dnnl::algorithm;
using normalization_flag = dnnl::normalization_flags;
using query = dnnl::query;
using scale_t = std::vector<float>;
using exec_args = std::unordered_map<int, memory>;

#ifndef NDEBUG
#define BACKEND_DNNL_ENFORCE(condition, message) \
    do { \
        error::wrap_c_api((condition) ? dnnl_success : dnnl_invalid_arguments, \
                (message)); \
    } while (false)
#else
#define BACKEND_DNNL_ENFORCE(condition, message)
#endif

const size_t DNNL_CPU_MEMALIGNMENT = 4096;

#if DNNL_GRAPH_WITH_SYCL
const size_t DNNL_SYCL_MEMALIGNMENT = 16;
#endif

inline dnnl::engine make_dnnl_engine(const impl::engine_t &eng) {
#if DNNL_GRAPH_WITH_SYCL
    return dnnl::sycl_interop::make_engine(
            eng.sycl_device(), eng.sycl_context());
#else
    return dnnl::engine(static_cast<dnnl::engine::kind>(eng.kind()),
            static_cast<size_t>(eng.device_id()));
#endif
}

struct allocator {
    static void *malloc(size_t size, const dnnl::engine &eng,
            const impl::allocator_t *alc) {
#if DNNL_GRAPH_WITH_SYCL
        return alc->allocate(size, dnnl::sycl_interop::get_device(eng),
                dnnl::sycl_interop::get_context(eng),
                {impl::allocator_lifetime::persistent, DNNL_SYCL_MEMALIGNMENT});
#else
        return eng.get_kind() == dnnl::engine::kind::cpu ? alc->allocate(size,
                       {impl::allocator_lifetime::persistent,
                               DNNL_CPU_MEMALIGNMENT})
                                                         : nullptr;
#endif
    }

    static void free(
            void *p, const dnnl::engine &eng, const impl::allocator_t *alc) {
#if DNNL_GRAPH_WITH_SYCL
        return alc->deallocate(p, dnnl::sycl_interop::get_context(eng));
#else
        if (eng.get_kind() == dnnl::engine::kind::cpu)
            return alc->deallocate(p);
        else
            return;
#endif
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

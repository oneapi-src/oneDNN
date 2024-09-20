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

#ifndef COMMON_ENGINE_IMPL_HPP
#define COMMON_ENGINE_IMPL_HPP

#include "common/c_types_map.hpp"
#include "common/engine_id.hpp"
#include "common/stream_impl.hpp"
#include "common/utils.hpp"

#ifdef ONEDNN_BUILD_GRAPH
#include "graph/interface/allocator.hpp"
#endif

#define VERROR_ENGINE_IMPL(cond, stat, msg, ...) \
    do { \
        if (!(cond)) { \
            VERROR(common, runtime, msg, ##__VA_ARGS__); \
            return stat; \
        } \
    } while (0)

namespace dnnl {
namespace impl {

class engine_impl_t {
public:
    engine_impl_t() = delete;
    engine_impl_t(engine_kind_t kind, runtime_kind_t runtime_kind, size_t index)
        : kind_(kind), runtime_kind_(runtime_kind), index_(index) {}

    virtual ~engine_impl_t() = default;

    engine_kind_t kind() const { return kind_; }
    runtime_kind_t runtime_kind() const { return runtime_kind_; }
    size_t index() const { return index_; }

    virtual engine_id_t engine_id() const {
        // Used for non-sycl CPU engine only that doesn't have device and
        // context.
        return {};
    }

#ifdef ONEDNN_BUILD_GRAPH
    void *get_allocator() const { return (void *)(&allocator_); };
    void set_allocator(graph::allocator_t *alloc) { allocator_ = *alloc; }
#endif

    virtual status_t init() { return status::success; }

    virtual status_t create_memory_storage(memory_storage_t **storage,
            engine_t *engine, unsigned flags, size_t size, void *handle) const {
        assert(!"unexpected");
        return status::runtime_error;
    }

    virtual status_t create_stream_impl(
            impl::stream_impl_t **stream_impl, unsigned flags) const {
        auto *si = new impl::stream_impl_t(flags);
        if (!si) return status::out_of_memory;
        *stream_impl = si;
        return status::success;
    }

    virtual int get_buffer_alignment() const {
        assert(!"unexpected");
        return -1;
    }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_impl_t)

    engine_kind_t kind_;
    runtime_kind_t runtime_kind_;
    size_t index_;

#ifdef ONEDNN_BUILD_GRAPH
    graph::allocator_t allocator_;
#endif
};

} // namespace impl
} // namespace dnnl

#endif

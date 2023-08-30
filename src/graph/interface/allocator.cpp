/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "oneapi/dnnl/dnnl_graph.h"
#include "oneapi/dnnl/dnnl_graph_sycl.h"

#include "common/engine.hpp"
#include "common/rw_mutex.hpp"

#include "graph/interface/allocator.hpp"
#include "graph/interface/c_types_map.hpp"

#include "graph/utils/utils.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.h"
#endif

using namespace dnnl::impl::graph;

status_t DNNL_API dnnl_graph_allocator_create(allocator_t **allocator,
        host_allocate_f host_malloc, host_deallocate_f host_free) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    UNUSED(allocator);
    UNUSED(host_malloc);
    UNUSED(host_free);
    return status::invalid_arguments;
#else
    if (utils::any_null(host_malloc, host_free)) {
        *allocator = new dnnl_graph_allocator();
    } else {
        *allocator = new dnnl_graph_allocator(host_malloc, host_free);
    }
    return status::success;
#endif
}

status_t DNNL_API dnnl_graph_sycl_interop_allocator_create(
        allocator_t **allocator, sycl_allocate_f sycl_malloc,
        sycl_deallocate_f sycl_free) {
#ifdef DNNL_WITH_SYCL
    if (utils::any_null(sycl_malloc, sycl_free)) {
        *allocator = new dnnl_graph_allocator();
    } else {
        *allocator = new dnnl_graph_allocator(sycl_malloc, sycl_free);
    }
    return status::success;
#else
    UNUSED(allocator);
    UNUSED(sycl_malloc);
    UNUSED(sycl_free);
    return status::unimplemented;
#endif
}

status_t DNNL_API dnnl_graph_allocator_destroy(allocator_t *allocator) {
    if (allocator == nullptr) return status::invalid_arguments;
    delete allocator;
    return status::success;
}

status_t DNNL_API dnnl_graph_make_engine_with_allocator(engine_t **engine,
        engine_kind_t kind, size_t index, const allocator_t *alloc) {
    auto ret = dnnl_engine_create(engine, kind, index);
    if (ret != status::success) return ret;

    (*engine)->set_allocator(const_cast<allocator_t *>(alloc));
    return status::success;
}

status_t DNNL_API dnnl_graph_sycl_interop_make_engine_with_allocator(
        engine_t **engine, const void *device, const void *context,
        const allocator_t *alloc) {
#ifdef DNNL_WITH_SYCL
    auto ret = dnnl_sycl_interop_engine_create(engine, device, context);
    if (ret != status::success) return ret;

    (*engine)->set_allocator(const_cast<allocator_t *>(alloc));
    return status::success;
#else
    UNUSED(engine);
    UNUSED(device);
    UNUSED(context);
    UNUSED(alloc);
    return status::unimplemented;
#endif
}

void dnnl_graph_allocator::monitor_t::record_allocate(
        const void *buf, size_t size, dnnl_graph_allocator::mem_type_t type) {
    const auto persistent = dnnl_graph_allocator::mem_type_t::persistent;
    const auto temp = dnnl_graph_allocator::mem_type_t::temp;
    if (type == persistent) {
        persist_mem_ += size;
        persist_mem_infos_.emplace(buf, mem_info_t {size, persistent});
    } else if (type == temp) {
        auto tid = std::this_thread::get_id();
        temp_mem_[tid] += size;
        if (peak_temp_mem_[tid] < temp_mem_[tid])
            peak_temp_mem_[tid] = temp_mem_[tid];
        temp_mem_infos_[tid].emplace(buf, mem_info_t {size, temp});
    } else {
        // we didn't use output type buffer now.
        assertm(0, "we didn't use output type buffer now");
    }
}

void dnnl_graph_allocator::monitor_t::record_deallocate(const void *buf) {
    bool is_persist = persist_mem_infos_.find(buf) != persist_mem_infos_.end();
    if (is_persist) {
        auto persist_pos = persist_mem_infos_.find(buf);
        persist_mem_ -= persist_pos->second.size_;
        persist_mem_infos_.erase(persist_pos);
    } else {
        auto tid = std::this_thread::get_id();
        auto temp_pos = temp_mem_infos_[tid].find(buf);
        if (temp_pos != temp_mem_infos_[tid].end()) {
            temp_mem_[tid] -= temp_pos->second.size_;
        }
    }
}

void dnnl_graph_allocator::monitor_t::reset_peak_temp_memory() {
    auto tid = std::this_thread::get_id();
    rw_mutex_.lock_write();
    peak_temp_mem_[tid] = 0;
    rw_mutex_.unlock_write();
}

size_t dnnl_graph_allocator::monitor_t::get_peak_temp_memory() {
    auto tid = std::this_thread::get_id();
    rw_mutex_.lock_read();
    size_t ret = peak_temp_mem_.at(tid);
    rw_mutex_.unlock_read();
    return ret;
}

size_t dnnl_graph_allocator::monitor_t::get_total_persist_memory() {
    rw_mutex_.lock_read();
    size_t size = persist_mem_;
    rw_mutex_.unlock_read();
    return size;
}

void dnnl_graph_allocator::monitor_t::lock_write() {
    rw_mutex_.lock_write();
}

void dnnl_graph_allocator::monitor_t::unlock_write() {
    rw_mutex_.unlock_write();
}

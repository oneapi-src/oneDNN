/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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
#include "dnnl_sycl.h"
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
        *allocator = allocator_t::create();
    } else {
        *allocator = allocator_t::create(host_malloc, host_free);
    }
    return status::success;
#endif
}

status_t DNNL_API dnnl_graph_sycl_interop_allocator_create(
        allocator_t **allocator, sycl_allocate_f sycl_malloc,
        sycl_deallocate_f sycl_free) {
#ifdef DNNL_WITH_SYCL
    if (utils::any_null(sycl_malloc, sycl_free)) {
        *allocator = allocator_t::create();
    } else {
        *allocator = allocator_t::create(sycl_malloc, sycl_free);
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
    allocator->release();
    return status::success;
}

status_t DNNL_API dnnl_graph_make_engine_with_allocator(engine_t **engine,
        engine_kind_t kind, size_t index, const allocator_t *alloc) {
    auto ret = dnnl_engine_create(engine, kind, index);
    if (ret != status::success) return status::invalid_arguments;

    reinterpret_cast<engine_t *>(*engine)->set_allocator(
            const_cast<allocator_t *>(alloc));
    return status::success;
}

status_t DNNL_API dnnl_graph_sycl_interop_make_engine_with_allocator(
        engine_t **engine, const void *device, const void *context,
        const allocator_t *alloc) {
#ifdef DNNL_WITH_SYCL
    auto ret = dnnl_sycl_interop_engine_create(engine, device, context);
    if (ret != status::success) return status::invalid_arguments;

    reinterpret_cast<engine_t *>(*engine)->set_allocator(
            const_cast<allocator_t *>(alloc));
    return status::success;
#else
    UNUSED(engine);
    UNUSED(device);
    UNUSED(context);
    UNUSED(alloc);
    return status::unimplemented;
#endif
}

std::unordered_map<const dnnl_graph_allocator *, size_t>
        dnnl_graph_allocator::monitor_t::persist_mem_;
std::unordered_map<const dnnl_graph_allocator *,
        std::unordered_map<const void *, dnnl_graph_allocator::mem_info_t>>
        dnnl_graph_allocator::monitor_t::persist_mem_infos_;

std::unordered_map<std::thread::id,
        std::unordered_map<const dnnl_graph_allocator *, size_t>>
        dnnl_graph_allocator::monitor_t::temp_mem_;
std::unordered_map<std::thread::id,
        std::unordered_map<const dnnl_graph_allocator *, size_t>>
        dnnl_graph_allocator::monitor_t::peak_temp_mem_;
std::unordered_map<std::thread::id,
        std::unordered_map<const dnnl_graph_allocator *,
                std::unordered_map<const void *,
                        dnnl_graph_allocator::mem_info_t>>>
        dnnl_graph_allocator::monitor_t::temp_mem_infos_;

dnnl::impl::utils::rw_mutex_t dnnl_graph_allocator::monitor_t::rw_mutex_;

void dnnl_graph_allocator::monitor_t::record_allocate(
        const dnnl_graph_allocator *alloc, const void *buf, size_t size,
        dnnl_graph_allocator::mem_type_t type) {
    const auto persistent = dnnl_graph_allocator::mem_type_t::persistent;
    const auto temp = dnnl_graph_allocator::mem_type_t::temp;
    if (type == persistent) {
        persist_mem_[alloc] += size;
        persist_mem_infos_[alloc].emplace(buf, mem_info_t {size, persistent});
    } else if (type == temp) {
        auto tid = std::this_thread::get_id();
        temp_mem_[tid][alloc] += size;
        if (peak_temp_mem_[tid][alloc] < temp_mem_[tid][alloc])
            peak_temp_mem_[tid][alloc] = temp_mem_[tid][alloc];
        temp_mem_infos_[tid][alloc].emplace(buf, mem_info_t {size, temp});
    } else {
        // we didn't use output type buffer now.
        assertm(0, "we didn't use output type buffer now");
    }
}

void dnnl_graph_allocator::monitor_t::record_deallocate(
        const dnnl_graph_allocator *alloc, const void *buf) {
    bool is_persist = persist_mem_infos_.find(alloc) != persist_mem_infos_.end()
            && persist_mem_infos_.at(alloc).find(buf)
                    != persist_mem_infos_.at(alloc).end();
    if (is_persist) {
        auto persist_pos = persist_mem_infos_.at(alloc).find(buf);
        persist_mem_[alloc] -= persist_pos->second.size_;
        persist_mem_infos_[alloc].erase(persist_pos);
    } else {
        auto tid = std::this_thread::get_id();
        auto temp_pos = temp_mem_infos_[tid][alloc].find(buf);
        temp_mem_[tid][alloc] -= temp_pos->second.size_;
    }
}

void dnnl_graph_allocator::monitor_t::reset_peak_temp_memory(
        const dnnl_graph_allocator *alloc) {
    auto tid = std::this_thread::get_id();
    rw_mutex_.lock_write();
    peak_temp_mem_[tid][alloc] = 0;
    rw_mutex_.unlock_write();
}

size_t dnnl_graph_allocator::monitor_t::get_peak_temp_memory(
        const dnnl_graph_allocator *alloc) {
    auto tid = std::this_thread::get_id();
    rw_mutex_.lock_read();
    size_t ret = peak_temp_mem_.at(tid).at(alloc);
    rw_mutex_.unlock_read();
    return ret;
}

size_t dnnl_graph_allocator::monitor_t::get_total_persist_memory(
        const dnnl_graph_allocator *alloc) {
    rw_mutex_.lock_read();
    size_t size = persist_mem_.find(alloc) != persist_mem_.end()
            ? persist_mem_.at(alloc)
            : 0;
    rw_mutex_.unlock_read();
    return size;
}

void dnnl_graph_allocator::monitor_t::lock_write() {
    rw_mutex_.lock_write();
}

void dnnl_graph_allocator::monitor_t::unlock_write() {
    rw_mutex_.unlock_write();
}

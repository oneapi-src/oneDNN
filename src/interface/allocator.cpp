/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "interface/allocator.hpp"
#include "interface/c_types_map.hpp"

#include "utils/rw_mutex.hpp"
#include "utils/utils.hpp"

using namespace dnnl::graph::impl;

status_t DNNL_GRAPH_API dnnl_graph_allocator_create(
        allocator_t **created_allocator, cpu_allocate_f cpu_malloc,
        cpu_deallocate_f cpu_free) {
    if (utils::any_null(cpu_malloc, cpu_free)) {
        *created_allocator = allocator_t::create();
    } else {
        *created_allocator = allocator_t::create(cpu_malloc, cpu_free);
    }
    return status::success;
}

status_t DNNL_GRAPH_API dnnl_graph_sycl_interop_allocator_create(
        allocator_t **created_allocator, sycl_allocate_f sycl_malloc,
        sycl_deallocate_f sycl_free) {
#if DNNL_GRAPH_WITH_SYCL
    if (utils::any_null(sycl_malloc, sycl_free)) {
        *created_allocator = allocator_t::create();
    } else {
        *created_allocator = allocator_t::create(sycl_malloc, sycl_free);
    }
    return status::success;
#else
    UNUSED(created_allocator);
    UNUSED(sycl_malloc);
    UNUSED(sycl_free);
    return status::unsupported;
#endif
}

status_t DNNL_GRAPH_API dnnl_graph_allocator_destroy(allocator_t *allocator) {
    if (allocator == nullptr) return status::invalid_argument;
    allocator->release();
    return status::success;
}

std::unordered_map<const dnnl_graph_allocator *, size_t>
        dnnl_graph_allocator::monitor_t::persist_mem_;
std::unordered_map<const dnnl_graph_allocator *,
        std::unordered_map<const void *, dnnl_graph_allocator::mem_info_t>>
        dnnl_graph_allocator::monitor_t::persist_mem_infos_;

thread_local std::unordered_map<const dnnl_graph_allocator *, size_t>
        dnnl_graph_allocator::monitor_t::temp_mem_;
thread_local std::unordered_map<const dnnl_graph_allocator *, size_t>
        dnnl_graph_allocator::monitor_t::peak_temp_mem_;
thread_local std::unordered_map<const dnnl_graph_allocator *,
        std::unordered_map<const void *, dnnl_graph_allocator::mem_info_t>>
        dnnl_graph_allocator::monitor_t::temp_mem_infos_;

utils::rw_mutex_t dnnl_graph_allocator::monitor_t::rw_mutex_;

void dnnl_graph_allocator::monitor_t::record_allocate(
        const dnnl_graph_allocator *alloc, const void *buf, size_t size,
        const dnnl_graph_allocator::attribute_t &attr) {
    if (attr.data.type == allocator_lifetime::persistent) {
        rw_mutex_.lock_write();
        persist_mem_[alloc] += size;
        persist_mem_infos_[alloc].emplace(
                buf, mem_info_t {size, allocator_lifetime::persistent});
        rw_mutex_.unlock_write();
    } else if (attr.data.type == allocator_lifetime::temp) {
        temp_mem_[alloc] += size;
        if (peak_temp_mem_[alloc] < temp_mem_[alloc])
            peak_temp_mem_[alloc] = temp_mem_[alloc];
        temp_mem_infos_[alloc].emplace(
                buf, mem_info_t {size, allocator_lifetime::temp});
    } else {
        // we didn't use output type buffer now.
        assertm(0, "we didn't use output type buffer now");
    }
}

void dnnl_graph_allocator::monitor_t::record_deallocate(
        const dnnl_graph_allocator *alloc, const void *buf) {
    rw_mutex_.lock_read();
    bool is_persist = persist_mem_infos_.find(alloc) != persist_mem_infos_.end()
            && persist_mem_infos_.at(alloc).find(buf)
                    != persist_mem_infos_.at(alloc).end();
    rw_mutex_.unlock_read();

    if (is_persist) {
        rw_mutex_.lock_write();
        auto persist_pos = persist_mem_infos_.at(alloc).find(buf);
        persist_mem_[alloc] -= persist_pos->second.size_;
        persist_mem_infos_[alloc].erase(persist_pos);
        rw_mutex_.unlock_write();
    } else {
        auto temp_pos = temp_mem_infos_[alloc].find(buf);
        temp_mem_[alloc] -= temp_pos->second.size_;
        temp_mem_infos_[alloc].erase(temp_pos);
    }
}

void dnnl_graph_allocator::monitor_t::reset_peak_temp_memory(
        const dnnl_graph_allocator *alloc) {
    peak_temp_mem_[alloc] = 0;
}

size_t dnnl_graph_allocator::monitor_t::get_peak_temp_memory(
        const dnnl_graph_allocator *alloc) {
    return peak_temp_mem_[alloc];
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

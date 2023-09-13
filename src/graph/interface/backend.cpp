/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "graph/interface/backend.hpp"

#include "graph/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {

std::pair<size_t, size_t> backend_registry_t::decode_layout_id(
        size_t layout_id) {
    size_t backend_id = layout_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1);
    size_t layout_idx = layout_id >> BACKEND_ID_LENGTH;
    return {layout_idx, backend_id};
}

size_t backend_registry_t::encode_layout_id(
        size_t layout_idx, size_t backend_id) {
    size_t layout_id = (layout_idx << BACKEND_ID_LENGTH)
            | (backend_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1));
    return layout_id;
}

size_t backend_registry_t::extract_layout_id(size_t layout_id) {
    return layout_id >> BACKEND_ID_LENGTH;
}

size_t backend_registry_t::extract_backend_id(size_t layout_id) {
    return layout_id & (size_t)((1 << BACKEND_ID_LENGTH) - 1);
}

// Backend API used by each backend to check the constant tensor cache enabling
// status
bool is_constant_cache_enabled() {
    int result = 0;
    dnnl_graph_get_constant_tensor_cache(&result);
    return result;
}

class constant_cache_flag_t {
    std::atomic<bool> constant_cache_enabled_;

    // We specialize the constructor so that we can initialize the flag
    // according to the env var. Because, with the new constant tensor cache
    // control API, the cache is disabled by default. If we want to run examples
    // with caching, we need to change the code to call
    // set_constant_tensor_cache(1) explicitly and rebuild it, which makes
    // testing both two configurations in pre-CI inconvenient. So we add the
    // internal env var _ONEDNN_CONSTANT_CACHE. If it's set by users, the
    // initial status will equal to the env var value.
    constant_cache_flag_t() {
        // If env var is set, use it. Otherwise, use flag=0 by default.
        int flag = utils::getenv_int_internal("CONSTANT_CACHE", 0);
        store(flag);
    }

    constant_cache_flag_t(const constant_cache_flag_t &) = delete;
    constant_cache_flag_t(constant_cache_flag_t &&) = delete;
    constant_cache_flag_t &operator=(const constant_cache_flag_t &) = delete;
    constant_cache_flag_t &operator=(constant_cache_flag_t &&) = delete;

public:
    static constant_cache_flag_t &get_singleton() {
        static constant_cache_flag_t ins;
        return ins;
    }

    int load() const {
        return static_cast<int>(
                constant_cache_enabled_.load(std::memory_order_relaxed));
    }
    void store(int flag) {
        constant_cache_enabled_.store(
                static_cast<bool>(flag), std::memory_order_relaxed);
    }
};

} // namespace graph
} // namespace impl
} // namespace dnnl

// Constant cache control API
dnnl::impl::graph::status_t dnnl_graph_set_constant_tensor_cache(int flag) {
    if (flag < 0) return dnnl::impl::graph::status::invalid_arguments;
    dnnl::impl::graph::constant_cache_flag_t::get_singleton().store(flag);
    return dnnl::impl::graph::status::success;
}

dnnl::impl::graph::status_t dnnl_graph_get_constant_tensor_cache(int *flag) {
    if (flag == nullptr) return dnnl::impl::graph::status::invalid_arguments;
    *flag = dnnl::impl::graph::constant_cache_flag_t::get_singleton().load();
    return dnnl::impl::graph::status::success;
}

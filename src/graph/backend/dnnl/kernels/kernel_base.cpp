/*******************************************************************************
 * Copyright 2024-2025 Intel Corporation
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

#include "graph/backend/dnnl/kernels/kernel_base.hpp"
#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t kernel_base_t::compile(const dnnl_partition_impl_t *part,
        const engine_t *aengine, const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    auto ret = compile_impl(part, aengine, inputs, outputs);
    if (ret != status::success) return ret;
    return prepare_inplace_pairs_impl();
}

status_t kernel_base_t::execute(const stream_t *astream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs) {
    return execute_impl(astream, inputs, outputs);
}

bool kernel_base_t::enabled_constant_cache() const {
    if (!p_engine_.get(true)) { return false; }

    const bool enabled = is_constant_cache_enabled(p_engine_);
    return enabled;
}

size_t kernel_base_t::encode_constant_cache_key(
        const std::vector<tensor_t> &inputs, size_t cache_key) const {
    // Encode the constant memory address into cache key for differentiation
    size_t encoded_cache_key = cache_key;
    for (const auto &in : inputs) {
        if (logical_tensor_wrapper_t(in.get_logical_tensor()).is_constant()) {
            encoded_cache_key = hash_combine(encoded_cache_key,
                    reinterpret_cast<uintptr_t>(in.get_data_handle()));
        }
    }
    return encoded_cache_key;
}

const std::vector<inplace_pair_t> &kernel_base_t::get_inplace_pairs() const {
    return inplace_pairs_;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

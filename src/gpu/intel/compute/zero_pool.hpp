/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_GEMM_ZERO_POOL_HPP
#define GPU_INTEL_JIT_GEMM_ZERO_POOL_HPP

#include <array>
#include <mutex>

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/compute_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

class zero_pool_t {
public:
    zero_pool_t(compute::compute_engine_t *engine, size_t chunk_size,
            bool stream_private = false, bool in_order = false);

    status_t init();

    status_t claim(compute::compute_stream_t *stream, size_t len,
            std::unique_ptr<memory_storage_t> &out_mem, int *out_token);
    void async_release(int token, const xpu::event_t &ev);

    void attach_client();
    int detach_client();
    int clients() const { return clients_; }

    std::mutex &mutex() { return mutex_; }

private:
    compute::compute_engine_t *engine_ = nullptr;

    static constexpr int max_chunks = 64;

    std::unique_ptr<memory_storage_t> mem_;
    size_t chunk_size_ = 0;
    int chunk_count_ = 0;
    bool stream_private_ = false;
    bool in_order_ = false;
    int clients_ = 0;
    int next_slot_ = 0;
    bool inited_ = false;

    std::array<bool, max_chunks> event_pending_ = {false};
    std::array<std::unique_ptr<xpu::event_t>, max_chunks> events_ = {nullptr};
    std::mutex mutex_;

    status_t claim_unpooled(compute::compute_stream_t *stream, size_t len,
            std::unique_ptr<memory_storage_t> &out_mem);
};

status_t lookup_zero_pool(compute::compute_engine_t *engine,
        compute::compute_stream_t *stream, size_t chunk_size,
        zero_pool_t **out_pool);
void release_zero_pool(zero_pool_t *pool);

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

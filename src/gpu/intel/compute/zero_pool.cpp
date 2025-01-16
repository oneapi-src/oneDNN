/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include <mutex>
#include <unordered_map>

#include "gpu/intel/compute/zero_pool.hpp"

#ifdef DNNL_WITH_SYCL
#include "gpu/intel/sycl/stream.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

static std::unordered_map<engine_id_t, zero_pool_t *> zero_pool_cache;
static std::mutex zero_pool_cache_mutex;

#ifdef DNNL_WITH_SYCL
// Unfortunately, weak_ptrs cannot be hashed, so unordered_map not possible here.
// SYCL is currently missing owner_less for command graphs, so define it ourselves.
struct weak_graph_owner_less {
    bool operator()(const sycl::stream_t::weak_graph_t &lhs,
            const sycl::stream_t::weak_graph_t &rhs) const noexcept {
        return lhs.owner_before(rhs);
    }
};
static std::map<sycl::stream_t::weak_graph_t, zero_pool_t *,
        weak_graph_owner_less>
        recorded_zero_pool_cache;
#endif

struct cleanup_sentinel_t {
    cleanup_sentinel_t(bool *ptr) : ptr_(ptr) {}
    cleanup_sentinel_t(const cleanup_sentinel_t &) = delete;
    cleanup_sentinel_t(cleanup_sentinel_t &&other) = delete;
    cleanup_sentinel_t &operator=(const cleanup_sentinel_t &) = delete;
    cleanup_sentinel_t &operator=(cleanup_sentinel_t &&other) = delete;
    ~cleanup_sentinel_t() { *ptr_ = true; }

private:
    bool *ptr_;
};

static bool in_cleanup() {
    static bool destroyed = false;
    static cleanup_sentinel_t sentinel {&destroyed};

    return destroyed;
}

status_t lookup_zero_pool(compute::compute_engine_t *engine,
        compute::compute_stream_t *stream, size_t chunk_size,
        zero_pool_t **out_pool) {
    status_t status = status::success;

    (void)in_cleanup();

#ifdef DNNL_WITH_SYCL
    // If recording, get a per-graph zero pool.
    const auto *sycl_stream
            = utils::downcast<const gpu::intel::sycl::stream_t *>(stream);
    if (sycl_stream->recording()) {
        {
            std::lock_guard<std::mutex> lock(zero_pool_cache_mutex);
            auto &pool = recorded_zero_pool_cache
                    [sycl_stream->get_current_graph_weak()];
            if (!pool) {
                pool = new zero_pool_t(engine, chunk_size, true,
                        stream->flags() & stream_flags::in_order);
                status = pool->init();

                // Short-term hack: intentionally leak pool as
                //  we cannot know when graph is no longer in use.
                pool->attach_client();
            }
            *out_pool = pool;
        }
        (*out_pool)->attach_client();
        return status;
    }
#endif

    // In regular mode, find a per-engine zero pool.
    auto engine_id = engine->engine_id();

    {
        std::lock_guard<std::mutex> lock(zero_pool_cache_mutex);

        auto &pool = zero_pool_cache[engine_id];
        if (!pool) {
            pool = new zero_pool_t(engine, chunk_size);
            status = pool->init();
        }
        *out_pool = pool;
    }

    (*out_pool)->attach_client();
    return status;
}

void release_zero_pool(zero_pool_t *pool) {
    int clients = pool->detach_client();

    /* Skip zero pool cleanup if static objects are being destroyed,
       because USM free may not work. */
    if (clients == 0 && !in_cleanup()) {
        std::lock_guard<std::mutex> lock(zero_pool_cache_mutex);
        if (pool->clients() == 0) {
            for (auto iter = zero_pool_cache.begin();
                    iter != zero_pool_cache.end(); iter++) {
                if (iter->second == pool) {
                    zero_pool_cache.erase(iter);
                    break;
                }
            }
            delete pool;
        }
    }
}

zero_pool_t::zero_pool_t(compute::compute_engine_t *engine, size_t chunk_size,
        bool stream_private, bool in_order)
    : engine_(engine)
    , chunk_size_(chunk_size)
    , chunk_count_(stream_private ? 1 : 16)
    , stream_private_(stream_private)
    , in_order_(in_order) {

    assert(chunk_count_ <= max_chunks);
}

status_t zero_pool_t::init() {
    memory_storage_t *mem = nullptr;
    auto status = engine_->create_memory_storage(&mem,
            memory_flags_t::alloc | memory_flags_t::prefer_device_usm,
            chunk_count_ * chunk_size_, nullptr);

    /* NULL out engine_ to ensure we don't try to rely on its value later,
       when the engine may no longer be valid. This is because zero_pool_t is
       tied to an engine_id_t rather than a specific engine. */
    engine_ = nullptr;

    if (status == status::success) mem_.reset(mem);
    return status;
}

void zero_pool_t::attach_client() {
    std::lock_guard<std::mutex> lock(mutex_);
    clients_++;
}

int zero_pool_t::detach_client() {
    std::lock_guard<std::mutex> lock(mutex_);
    return --clients_;
}

status_t zero_pool_t::claim_unpooled(compute::compute_stream_t *stream,
        size_t len, std::unique_ptr<memory_storage_t> &out_mem) {

    memory_storage_t *new_mem = nullptr;
    auto status = stream->engine()->create_memory_storage(&new_mem,
            memory_flags_t::alloc | memory_flags_t::prefer_device_usm, len,
            nullptr);

    if (status == status::success) {
        stream->fill(*new_mem, 0, len, stream->ctx().get_deps(),
                stream->ctx().get_deps());
        out_mem.reset(new_mem);
    }

    return status;
}

status_t zero_pool_t::claim(compute::compute_stream_t *stream, size_t len,
        std::unique_ptr<memory_storage_t> &out_mem, int *out_token) {
    out_mem.reset();
    *out_token = -1;
    if (len == 0) return status::success;

    if (len > chunk_size_ || !mem_) return claim_unpooled(stream, len, out_mem);

    std::lock_guard<std::mutex> lock(mutex_);

    if (!inited_) {
        // One-time zero initialization before first use.
        // Use immediate mode to ensure zero initialization
        //   occurs now and is not recorded.
        stream->enter_immediate_mode();
        stream->fill(*mem_, 0, chunk_count_ * chunk_size_,
                stream->ctx().get_deps(), stream->ctx().get_deps());
        stream->exit_immediate_mode();
        inited_ = true;
    }

    auto slot = next_slot_++;
    if (next_slot_ >= chunk_count_) next_slot_ = 0;

    if (!stream_private_ && event_pending_[slot]) {
        // Rare case: another thread claimed this slot but has not yet registered a completion event for it.
        // No choice but to create a temporary allocation (or yield and wait for the completion event).
        return claim_unpooled(stream, len, out_mem);
    }

    if (stream_private_) {
        // Per-stream zero pool. No event synchronization needed.
        if (!in_order_) CHECK(stream->barrier());
    } else if (events_[slot]) {
        // Slot is claimed and has an outstanding event. Wait on it and delete it.
        stream->ctx().append_deps(*events_[slot]);
        events_[slot].reset();
    }

    *out_token = slot;
    out_mem = mem_->get_sub_storage(slot * chunk_size_, chunk_size_);
    event_pending_[slot] = true;

    return status::success;
}

void zero_pool_t::async_release(int token, const xpu::event_t &ev) {
    if (token >= 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        int slot = token;
        if (!stream_private_) events_[slot] = ev.clone();
        event_pending_[slot] = false;
    }
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

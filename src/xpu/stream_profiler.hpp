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

#ifndef XPU_STREAM_PROFILER_HPP
#define XPU_STREAM_PROFILER_HPP

#include <cassert>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include "common/c_types_map.hpp"

#include "xpu/context.hpp"

namespace dnnl {
namespace impl {
namespace xpu {

struct stream_profiler_t {
    stream_profiler_t(const stream_t *stream, int stamp = 0)
        : stamp_(stamp), stream_(stream) {}
    virtual ~stream_profiler_t() = default;

    struct entry_t {
        uint64_t min_nsec = std::numeric_limits<uint64_t>::max();
        uint64_t max_nsec = 0;
        double freq = 0;
        int kernel_count = 0;

        uint64_t get_nsec() const { return max_nsec - min_nsec; }
    };

    struct registered_event_t {
        registered_event_t(
                std::unique_ptr<xpu::event_t> &&event, uint64_t stamp)
            : event(std::move(event)), stamp(stamp) {}

        std::unique_ptr<xpu::event_t> event;
        uint64_t stamp;
    };

    virtual status_t get_info(profiling_data_kind_t data_kind, int *num_entries,
            uint64_t *data) const = 0;

    uint64_t stamp() const { return stamp_; }

    void register_event(std::unique_ptr<xpu::event_t> &&event) {
        events_.emplace_back(std::move(event), stamp_);
    }

    void reset() {
        events_.clear();
        m_.lock();
        stamp_ = 0;
        m_.unlock();
    }

    // The contract is profiler interfaces are called only in between
    // `start_profiling` and `stop_profiling`, which provide a secure
    // multi-threaded access because of the lock. It allows to strip the lock
    // from all other calls, e.g., `stamp, or `register_event` (except `reset`)
    // to reduce the overhead for profiling.
    void start_profiling() {
        m_.lock();
        stamp_++;
    }
    void stop_profiling() { m_.unlock(); }

    void set_callback(void (*callback)(uint64_t, uint64_t)) {
        callback_ = callback;
    }

    status_t notify_profiling_complete() const {
        if (callback_) callback_(0, std::numeric_limits<uint64_t>::max());
        return status::success;
    }

protected:
    status_t get_info_impl(const std::map<uint64_t, entry_t> &stamp2entry,
            profiling_data_kind_t data_kind, uint64_t *data) const {
        int idx = 0;
        for (auto &kv : stamp2entry) {
            auto &e = kv.second;
            switch ((int)data_kind) {
                case profiling_data_kind::time: data[idx] = e.get_nsec(); break;
                case profiling_data_kind::cycles: {
                    double freq = e.freq / e.kernel_count;
                    data[idx] = static_cast<uint64_t>(
                            freq * static_cast<double>(e.get_nsec()) / 1e9);
                    if (callback_) callback_(kv.first, e.get_nsec());
                    break;
                }
                default: assert(!"unexpected data kind");
            }
            idx++;
        }
        return status::success;
    }

    std::recursive_mutex m_;
    std::vector<registered_event_t> events_;
    uint64_t stamp_;
    const stream_t *stream_;
    void (*callback_)(uint64_t, uint64_t) = nullptr;
};

} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif

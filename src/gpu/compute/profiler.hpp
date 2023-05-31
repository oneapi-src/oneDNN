/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_COMPUTE_PROFILER_HPP
#define GPU_COMPUTE_PROFILER_HPP

#include <unordered_map>

#include "common/c_types_map.hpp"

#include "gpu/compute/context.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

struct profiler_t {
    profiler_t(const stream_t *stream, int stamp = 0)
        : stamp_(stamp), stream_(stream) {}
    virtual ~profiler_t() = default;

    struct entry_t {
        uint64_t nsec = 0;
        double freq = 0;
        int kernel_count = 0;
    };

    struct registered_event_t {
        registered_event_t(std::unique_ptr<event_t> &&event, uint64_t stamp)
            : event(std::move(event)), stamp(stamp) {}

        std::unique_ptr<event_t> event;
        uint64_t stamp;
    };

    virtual status_t get_info(profiling_data_kind_t data_kind, int *num_entries,
            uint64_t *data) const = 0;

    void register_event(std::unique_ptr<event_t> &&event) {
        events_.emplace_back(std::move(event), stamp_);
    }

    void reset() {
        events_.clear();
        m_.lock();
        stamp_ = 0;
        m_.unlock();
    }

    void start_profiling() {
        m_.lock();
        stamp_++;
    }
    void stop_profiling() { m_.unlock(); }

protected:
    static status_t get_info_impl(
            const std::unordered_map<uint64_t, entry_t> &stamp2entry,
            profiling_data_kind_t data_kind, uint64_t *data) {
        int idx = 0;
        for (auto &kv : stamp2entry) {
            auto &e = kv.second;
            switch ((int)data_kind) {
                case profiling_data_kind::time: data[idx] = e.nsec; break;
                case profiling_data_kind::cycles: {
                    double freq = e.freq / e.kernel_count;
                    data[idx] = freq * e.nsec / 1e9;
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
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

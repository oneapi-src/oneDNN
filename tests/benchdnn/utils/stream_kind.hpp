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

#ifndef UTILS_STREAM_KIND_HPP
#define UTILS_STREAM_KIND_HPP

#include "oneapi/dnnl/dnnl.hpp"

#include "common.hpp"
#include "utils/engine.hpp"

#include <future>
#include <sstream>

enum class stream_kind_t {
    // The library-defined stream kind.
    def = 0x0,
    in_order = 0x1,
    out_of_order = 0x2,
};

extern stream_kind_t stream_kind; // user stream kind
extern stream_kind_t default_stream_kind; // the default stream kind

dnnl_stream_flags_t stream_kind2stream_flags(
        stream_kind_t stream_kind, bool use_profiling);

stream_kind_t str2stream_kind(const char *str);

std::ostream &operator<<(std::ostream &s, stream_kind_t stream_kind);

struct stream_t {
    stream_t() = default;
    // `enable_profiling=false` forces the stream to be created without the
    // profiling flag even in performance mode. It's used by the shared
    // data-preparation stream (see `get_test_stream`) which is never queried
    // for profiling data.
    stream_t(const engine_t &engine, void *interop_obj = nullptr,
            bool enable_profiling = true);
    operator dnnl_stream_t() const {
        return stream_.get(/* allow_empty = */ true);
    }
    operator dnnl::stream &() { return stream_; }
    // Wrapper over dnnl::stream::wait() to avoid explicit casts to dnnl::stream
    // at graph driver call sites.
    void wait() { stream_.wait(); }
    stream_t &operator=(stream_t &&rhs) = default;

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(stream_t);
    dnnl::stream stream_;
};

struct stream_staller_t {
    // Enqueue tasks to stall a primitive execution tasks for asynchronous
    // threadpool runtime. For rest runtimes does nothing.
    stream_staller_t(stream_t &stream);

    // A signal the submission has completed and ready for execution.
    void release();

private:
    std::promise<void> prom_;
};

// RAII handle over the shared data-preparation stream returned by
// `get_test_stream()`. It injects a `dnnl_stream_wait` on destruction, so a
// caller's submitted work is drained when the guard leaves scope. This upholds
// the serialization contract as long as each caller keeps a single short-lived
// guard for one operation and lets it die before the next `get_test_stream()`
// acquisition - which is how all current call sites use it.
//
// Caveat: the guarantee is tied to destruction, not acquisition. Two live
// guards in the same scope both reference the same static stream, and
// construction does not wait, so work submitted through the first is not
// drained before work is submitted through the second; both are only waited on
// at scope exit. Enforcing "drained before reuse" unconditionally would require
// a wait() in the constructor too, whose cost could negate the whole benefit of
// reusing a static stream over recreating one per operation.
struct stream_guard_t {
    explicit stream_guard_t(const stream_t &stream) : stream_(&stream) {}
    stream_guard_t(stream_guard_t &&rhs) : stream_(rhs.stream_) {
        rhs.stream_ = nullptr;
    }
    ~stream_guard_t() {
        if (stream_) DNN_SAFE_V(dnnl_stream_wait(*stream_));
    }
    operator dnnl_stream_t() const { return *stream_; }

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(stream_guard_t);
    const stream_t *stream_;
};

// A process-wide stream over the test engine (`get_test_engine()`) dedicated to
// sequential, fully synchronized data-preparation tasks such as device memory
// filling. Stream creation is costly on GPU runtimes, so reusing a single
// stream across the many per-memory fill operations avoids that overhead.
//
// Reuse is safe only because these tasks are submitted serially from the main
// thread and each waits on the stream before returning; the returned
// `stream_guard_t` injects that wait automatically on scope exit. The stream is
// created without the profiling flag so it never interferes with performance
// measurement, which uses its own dedicated streams.
stream_guard_t get_test_stream();

#endif

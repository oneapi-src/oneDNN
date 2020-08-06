/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_STREAM_HPP
#define LLGA_STREAM_HPP

#include <functional>
#include "engine.hpp"
#include "llga_api_detail.hpp"
#include "partition.hpp"
#include "tensor.hpp"

namespace llga {

class thread_pool {
public:
    /// A default constructor
    thread_pool() = default;

    /// A default destructor
    virtual ~thread_pool() = default;

    /// Get total number of threads in the pool
    ///
    /// @returns Number of threads in the pool
    virtual int num_threads() = 0;

    /// Get the current thread id in the pool
    ///
    /// @returns Thread id
    virtual int thread_id() = 0;

    // Used by the thread in the pool to know if it is running
    // in a parallel region
    /// Return a flag that indicates if the current thread is running
    /// in a parallel region
    ///
    /// @returns @c true The cuurent thread is running in a parallel region;
    ///     @c false The current thread is not running in a parallel region
    virtual bool in_parallel() = 0;

    /// Spawn threads and split tasks among threads
    ///
    /// @param begin Begin of the range
    /// @param end End of the range
    /// @param grain_size Smallest unit of a task
    /// @param fn Closure running each task ranging from task_begin
    //   and range_end
    virtual void parallel_for(int begin, int end, int grain_size,
            const std::function<void(int, int)> &fn)
            = 0;

    /// Spawn threads
    ///
    /// @param fn Closure run by each thread given task id per thread.
    virtual void parallel(const std::function<void(int, int)> &fn) = 0;
};

namespace api {

class thread_pool {
public:
    /// Constructs a thread_pool object
    ///
    /// @param num_threads Number of threads in the thread pool
    thread_pool(int32_t num_threads);
};

class stream_attr {
public:
    /// Constructs stream attributes
    ///
    /// @param pool A thread pool bound to this stream attribute
    stream_attr(thread_pool &pool);

    /// Get the thread pool
    ///
    /// @returns A pointer to the thread pool bound to this stream attribute
    thread_pool *get_thread_pool(;
};

class stream {
public:
    /// Constructs a stream for the specified engine
    ///
    /// @param handle A stream handle derived from framework
    /// @param engine Engine to create stream on
    /// @param attr A stream attribute, defaults to nullptr
    stream(void *handle, engine &engine, const stream_attr *attr = nullptr);

    /// Get the current handle of the stream, either sycl::stream or
    /// opaque stream handle
    ///
    /// @returns A handle to the stream
    void *get_stream_handle() const;

    /// Execute a compiled partition
    ///
    /// @param acompiled_partition Compiled partition to run
    /// @param inputs A list of input tensors in the partition
    /// @param outputs A list of output tensors in the partition
    void submit(compiled_partition &acompiled_partition,
            const std::vector<tensor> &inputs, std::vector<tensor> &outputs);
};

} // namespace api
} // namespace llga

#endif

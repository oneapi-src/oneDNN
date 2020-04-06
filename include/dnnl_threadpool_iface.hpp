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

#ifndef DNNL_THREADPOOL_IFACE_HPP
#define DNNL_THREADPOOL_IFACE_HPP

#include <functional>

namespace dnnl {

/// Abstract threadpool interface. The users are expected to subclass this
/// interface and pass an object to the library during CPU stream creation or
/// directly in case of BLAS functions.
struct threadpool_iface {
    /// Returns the number of worker threads.
    virtual int get_num_threads() const = 0;

    /// Returns true if the calling thread belongs to this threadpool.
    virtual bool get_in_parallel() const = 0;

    /// Submits n instances of a closure for execution in parallel:
    ///
    /// for (int i = 0; i < n; i++) fn(i, n);
    ///
    virtual void parallel_for(int n, const std::function<void(int, int)> &fn)
            = 0;

    /// Returns threadpool behavior flags bit mask (see below).
    virtual uint64_t get_flags() const = 0;

    /// If set, parallel_for() returns immediately and oneDNN needs implement
    /// waiting for the submitted closures to finish execution on its own.
    static constexpr uint64_t ASYNCHRONOUS = 1;

    virtual ~threadpool_iface() {}
};

} // namespace dnnl

#endif

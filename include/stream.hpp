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
namespace api {

class stream {
public:
    /// Constructs a stream for the specified engine
    ///
    /// @param engine Engine to create stream on
    /// @param attr A stream attribute, defaults to nullptr
    stream(engine &engine, const stream_attr *attr = nullptr);

    /// Constructs a stream for the specified engine and SYCL queue
    ///
    /// @param engine Engine to create stream on
    /// @param queue SYCL queue to create stream on
    /// @param attr A stream attribute, defaults to nullptr
    stream(engine &engine, const cl::sycl::queue &queue,
            const stream_attr *attr = nullptr);
};

} // namespace api
} // namespace llga

#endif

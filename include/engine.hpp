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

#ifndef LLGA_ENGINE_HPP
#define LLGA_ENGINE_HPP

#include "llga_api_detail.hpp"

namespace llga {
namespace api {

class engine {
public:
    /// engine kind
    enum class kind {
        /// An unspecified engine
        any = llga_any_engine,
        /// CPU engine
        cpu = llga_cpu,
        /// GPU engine
        gpu = llga_gpu,
    };

    /// Constructs an engine with specified kind and device_id
    ///
    /// @param akind The kind of engine to construct
    /// @param device_id Specify which device to be used
    engine(kind akind, int device_id);

    /// Create an engine from SYCL device and context
    /// @param akind The kind of engine to construct
    /// @param dev The SYCL device that this engine will encapsulate
    /// @param ctx The SYCL context that this engine will use
    engine(kind akind, const cl::sycl::device &dev,
            const cl::sycl::context &ctx);

    /// Returns device handle of the current engine
    ///
    /// @returns Device handle
    void *get_device_handle() const;

    /// Returns device id of the current engine
    ///
    /// @returns Device id
    int get_device_id() const;

    /// Returns concrete kind of the current engine
    ///
    ///@returns Kind of engine
    kind get_kind() const;
};

} // namespace api
} // namespace llga

#endif

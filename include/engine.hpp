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
class allocator {
public:
    /// Constructs an allocator with specified allcation/deallcation function
    ///
    /// @param fw_allocator A framework allocator passed to backend
    /// @param allocate_persistent Allocation function for persistent buffer
    /// @param deallocate_persistent Deallocation function for persistent buffer
    /// @param allocate_output Allocation function for output buffer
    /// @param allocate_temp Allocation function for temporary buffer
    allocator(void *fw_allocator, llga_allocate_persistent allocate_persistent,
            llga_deallocate_persistent deallocate_persistent,
            llga_allocate_output allocate_output,
            llga_allocate_temp allocate_temp);

    /// Constructs an empty allocator
    ///
    allocator() : allocator(nullptr, nullptr, nullptr, nullptr, nullptr) {}
};

class engine {
public:
    /// engine kind
    enum class kind {
        cpu = dnnl_graph_cpu,
        dpcpp = dnnl_graph_dpcpp,
        accelerator = dnnl_graph_accelerator,
        xpu_dpcpp = dnnl_graph_xpu_dpcpp
    };

    /// Constructs an engine with a device handle passed by framework
    ///
    /// @param akind Engine kind
    /// @param device_id Specify which device to be used
    /// @param device_handle A handle of the specified device
    /// @param alloc The memory allocator bound with engine
    engine(kind akind, int device_id, void *device_handle, allocator &alloc);

    /// Constructs an engine with specified kind and device id
    ///
    /// @param akind Engine kind
    /// @param device_id Specify which device to be used
    engine(kind akind, int device_id);

    /// Constructs an engine with proviced allocator
    ///
    /// @param akind Engine kind
    /// @param device_id Specify which device to be used
    /// @param alloc The memory allocator bound with engine
    engine(kind akind, int device_id, allocator &alloc);

    /// Set allocator to an engine
    ///
    /// @param alloc The memory allocator bound with engine
    void set_allocator(allocator &alloc);

    /// Get device handle of the current engine
    ///
    /// @returns Device handle
    void *get_device_handle() const;

    /// Get device id of the current engine
    ///
    /// @returns Device id
    int get_device_id() const;

    /// Get concrete kind of the current engine
    ///
    ///@returns Kind of engine
    llga_engine_kind_t get_engine_kind() const;
};

} // namespace api
} // namespace llga

#endif

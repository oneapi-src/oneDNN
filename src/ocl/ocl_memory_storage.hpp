/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef OCL_MEMORY_STORAGE_HPP
#define OCL_MEMORY_STORAGE_HPP

#include <CL/cl.h>

#include "common/c_types_map.hpp"
#include "common/memory_storage.hpp"
#include "common/utils.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

class ocl_memory_storage_t : public memory_storage_impl_t {
public:
    ocl_memory_storage_t(engine_t *engine, unsigned flags, size_t size,
            size_t alignment, void *handle);

    virtual status_t get_data_handle(void **handle) const override {
        *handle = static_cast<void *>(mem_object_.get());
        return status::success;
    }

    virtual status_t set_data_handle(void *handle) override {
        mem_object_ = ocl_utils::ocl_wrapper_t<cl_mem>(
                static_cast<cl_mem>(handle), true);
        return status::success;
    }

    virtual status_t map_data(void **mapped_ptr) const override;
    virtual status_t unmap_data(void *mapped_ptr) const override;

    cl_mem mem_object() const { return mem_object_.get(); }

    virtual uintptr_t base_offset() const override { return 0; }

private:
    ocl_utils::ocl_wrapper_t<cl_mem> mem_object_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

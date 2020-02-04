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

#ifndef GPU_OCL_OCL_RESOURCE_HPP
#define GPU_OCL_OCL_RESOURCE_HPP

#include "dnnl.h"

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ocl_resource_t : public resource_t {
    using key_kernel_t = compute::binary_t::id_t;
    using mapped_kernel_t = compute::kernel_t;

    using key_memory_t = int;
    using mapped_memory_t = std::unique_ptr<memory_storage_t>;

    ocl_resource_t() = default;

    status_t create_kernels_and_add(
            engine_t *engine, const std::vector<compute::binary_t> &binaries) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        std::vector<compute::kernel_t> kernels;
        CHECK(compute_engine->create_kernels(&kernels, binaries));
        for (size_t i = 0; i < binaries.size(); i++) {
            if (!binaries[i]) continue;
            assert(binary_id_to_kernel_.count(binaries[i].get_id()) == 0);
            binary_id_to_kernel_.emplace(binaries[i].get_id(), kernels[i]);
        }
        return status::success;
    }

    status_t create_kernel_and_add(
            engine_t *engine, const compute::binary_t &binary) {
        if (!binary) return status::success;
        assert(binary_id_to_kernel_.count(binary.get_id()) == 0);
        return create_kernels_and_add(engine, {binary});
    }

    const compute::kernel_t &get_kernel(key_kernel_t id) const {
        assert(binary_id_to_kernel_.count(id));
        const auto &kernel = binary_id_to_kernel_.at(id);
        assert(kernel);
        return kernel;
    }

    void add_memory_storage(key_memory_t idx, mapped_memory_t &&m) {
        assert(idx_to_memory_storage_.count(idx) == 0);
        if (!m) return;
        idx_to_memory_storage_.emplace(idx, std::move(m));
    }

    const memory_storage_t *get_memory_storage(int idx) const {
        assert(idx_to_memory_storage_.count(idx) != 0);
        return idx_to_memory_storage_.at(idx).get();
    }

private:
    std::unordered_map<key_kernel_t, mapped_kernel_t> binary_id_to_kernel_;
    std::unordered_map<key_memory_t, mapped_memory_t> idx_to_memory_storage_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

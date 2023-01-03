/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GPU_GPU_RESOURCE_HPP
#define GPU_GPU_RESOURCE_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/resource.hpp"
#include "gpu/compute/compute.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_resource_t : public resource_t {
    using key_memory_t = int;
    using mapped_memory_t = std::unique_ptr<memory_storage_t>;

    gpu_resource_t() = default;

    void add_memory_storage(key_memory_t idx, mapped_memory_t &&m) {
        assert(idx_to_memory_storage_.count(idx) == 0);
        if (!m) return;
        idx_to_memory_storage_.emplace(idx, std::move(m));
    }

    const memory_storage_t *get_memory_storage(int idx) const {
        assert(idx_to_memory_storage_.count(idx) != 0);
        return idx_to_memory_storage_.at(idx).get();
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(gpu_resource_t);

private:
    std::unordered_map<key_memory_t, mapped_memory_t> idx_to_memory_storage_;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

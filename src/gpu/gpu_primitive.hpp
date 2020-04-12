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

#ifndef GPU_GPU_PRIMITIVE_HPP
#define GPU_GPU_PRIMITIVE_HPP

#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

struct gpu_primitive_t : public primitive_t {
    using primitive_t::primitive_t;

    status_t create_binaries(engine_t *engine,
            std::vector<compute::binary_t> *binaries,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        CHECK(compute_engine->create_binaries(
                binaries, kernel_names, kernel_ctx));
        register_binaries(*binaries);
        return status::success;
    }

    status_t create_binary(engine_t *engine, compute::binary_t *binary,
            const char *kernel_name, const compute::kernel_ctx_t &kernel_ctx) {

        std::vector<compute::binary_t> binaries(1);
        auto status
                = create_binaries(engine, &binaries, {kernel_name}, kernel_ctx);
        if (status == status::success) *binary = binaries[0];
        return status;
    }

private:
    void register_binaries(const std::vector<compute::binary_t> &binaries) {
        for (const auto &b : binaries) {
            registered_binaries_.push_back(b);
        }
    }

    // TODO: introduce compute::kernel_t with a binary state instead
    std::vector<compute::binary_t> registered_binaries_;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

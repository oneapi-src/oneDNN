/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_COMPUTE_COMPUTE_ENGINE_HPP
#define GPU_COMPUTE_COMPUTE_ENGINE_HPP

#include <cassert>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "gpu/compute/device_info.hpp"
#include "gpu/compute/dispatch.hpp"
#include "gpu/compute/kernel.hpp"
#include "gpu/compute/kernel_ctx.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace compute {

class compute_engine_t : public engine_t {
public:
    compute_engine_t(engine_kind_t kind, runtime_kind_t runtime_kind,
            device_info_t *device_info)
        : engine_t(kind, runtime_kind), device_info_(device_info) {}

    status_t init() { return device_info_->init(); }

    const device_info_t *device_info() const { return device_info_.get(); }

    status_t create_kernel(kernel_t *kernel, const binary_t &binary) const {

        std::vector<kernel_t> kernels(1);
        auto status = create_kernels(&kernels, {binary});
        if (status == status::success) *kernel = kernels[0];

        return status;
    }

    status_t create_binary(binary_t *binary, const char *kernel_name,
            const kernel_ctx_t &kernel_ctx) const {

        std::vector<binary_t> binaries(1);
        auto status = create_binaries(&binaries, {kernel_name}, kernel_ctx);
        if (status == status::success) *binary = binaries[0];
        return status;
    }

    virtual status_t create_kernels(std::vector<kernel_t> *kernels,
            const std::vector<binary_t> &binaries) const = 0;

    virtual status_t create_binaries(std::vector<compute::binary_t> *binaries,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) const = 0;

    bool mayiuse(device_ext_t ext) const { return device_info_->has(ext); }

    dispatch_t create_dispatch(const memory_desc_t *md = nullptr) const {
        return dispatch_t(this, md);
    }

private:
    std::unique_ptr<device_info_t> device_info_;
};

} // namespace compute
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

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

#ifndef COMPUTE_ENGINE_HPP
#define COMPUTE_ENGINE_HPP

#include <cassert>
#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "compute/device_info.hpp"
#include "compute/dispatch.hpp"
#include "compute/kernel.hpp"
#include "compute/kernel_ctx.hpp"

namespace dnnl {
namespace impl {
namespace compute {

class compute_engine_t : public engine_t {
public:
    compute_engine_t(engine_kind_t kind, runtime_kind_t runtime_kind,
            device_info_t *device_info)
        : engine_t(kind, runtime_kind), device_info_(device_info) {}

    status_t init() { return device_info_->init(); }

    const device_info_t *device_info() const { return device_info_.get(); }

    status_t create_kernel(kernel_t *kernel, const char *kernel_name,
            const kernel_ctx_t &kernel_ctx) const {

        std::vector<kernel_t> kernels(1);
        auto status = create_kernels(&kernels, {kernel_name}, kernel_ctx);
        if (status == status::success) *kernel = kernels[0];

        return status;
    }

    virtual status_t create_kernels(std::vector<kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const kernel_ctx_t &kernel_ctx) const = 0;

    bool mayiuse(device_ext_t ext) const { return device_info_->has(ext); }

    dispatch_t create_dispatch(const memory_desc_t *md = nullptr) const {
        return dispatch_t(this, md);
    }

private:
    std::unique_ptr<device_info_t> device_info_;
};

} // namespace compute
} // namespace impl
} // namespace dnnl

#endif

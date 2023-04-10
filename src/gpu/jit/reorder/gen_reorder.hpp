/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_JIT_REORDER_GEN_REORDER_HPP
#define GPU_JIT_REORDER_GEN_REORDER_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class reorder_config_t;
class kernel_info_t;

class gen_reorder_t : public gpu_primitive_t {
public:
    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("jit:ir", gen_reorder_t);

        status_t init(engine_t *, engine_t *, engine_t *);
        status_t init_kernel_info();

        std::shared_ptr<reorder_config_t> cfg;
        std::shared_ptr<kernel_info_t> kernel_info;

    private:
        DECLARE_GPU_REORDER_CREATE();
    };

    using gpu_primitive_t::gpu_primitive_t;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

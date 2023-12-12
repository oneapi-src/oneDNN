/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_JIT_POOLING_GEN_POOLING_HPP
#define GPU_JIT_POOLING_GEN_POOLING_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_pooling_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/jit/pooling/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class pooling_config_t;
class kernel_info_t;

class gen_pooling_fwd_t : public gpu_primitive_t {
public:
    struct pd_t : public gpu_pooling_fwd_pd_t {
        pd_t(const pooling_desc_t *adesc, const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : gpu_pooling_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("jit:ir", gen_pooling_fwd_t);

        status_t init(engine_t *);

        std::shared_ptr<pool_conf_t> pool_conf;
        std::shared_ptr<exec_config_t> exec_cfg;
        std::shared_ptr<layout_t> src;
        std::shared_ptr<layout_t> dst;
    };

    using gpu_primitive_t::gpu_primitive_t;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    pooling_config_t cfg_;
    kernel_info_t kernel_info_;
    compute::kernel_t kernel_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

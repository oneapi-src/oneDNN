/*******************************************************************************
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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
#ifndef CPU_RV64_RVV_LAYER_NORMALIZATION_BWD_HPP
#define CPU_RV64_RVV_LAYER_NORMALIZATION_BWD_HPP

#include "common/primitive.hpp"
#include "cpu/cpu_layer_normalization_pd.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct jit_rvv_layernorm_bwd_kernel_t;

struct jit_uni_layer_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_layer_normalization_bwd_pd_t {
        using cpu_layer_normalization_bwd_pd_t::
                cpu_layer_normalization_bwd_pd_t;
        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", v, ""),
                jit_uni_layer_normalization_bwd_t);
        status_t init(const engine_t *engine);
        int nthr_ = 1;
        bool diff_weights_ = false;
    };

    explicit jit_uni_layer_normalization_bwd_t(const pd_t *apd);
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
    const jit_rvv_layernorm_bwd_kernel_t *kernel_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

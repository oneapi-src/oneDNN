/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_X64_ZEN64_INNER_PRODUCT_ZEN_INNER_PRODUCT_HPP
#define CPU_X64_ZEN64_INNER_PRODUCT_ZEN_INNER_PRODUCT_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"

#include "cpu/cpu_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace inner_product {

// Zen inner product = a thin 2D adapter mapping the IP problem onto a nested
// zen_matmul (no duplication of its validation / post-ops / launcher). The
// weights path is chosen per prop_kind:
//   - forward_inference w/ open weights: advertise the opaque zen_packed format
//     so the framework prepacks [OC,IC] weights once into [K=IC, N=OC]; matmul
//     reads [IC,OC]. Constant weights => packed only once.
//   - otherwise (training, or pre-resolved plain weights): hand matmul the plain
//     [OC,IC] weights as an [IC,OC] view (same bytes); zen_matmul packs each call.
struct zen_inner_product_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public ::dnnl::impl::cpu::cpu_inner_product_fwd_pd_t {
        using ::dnnl::impl::cpu::cpu_inner_product_fwd_pd_t::
                cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T("zen:ip:f32|bf16|f16:amd", zen_inner_product_fwd_t);

        status_t init(const engine_t *engine);

        std::shared_ptr<primitive_desc_t> matmul_pd_;

    private:
        // Reshape the IP problem to 2D, advertise zen_packed weights, and
        // create the nested zen_matmul pd (mirrors matmul_inner_product).
        status_t init_matmul_params(const engine_t *engine);

        // Resolve open descriptors to plain (backward-suitable) layouts for
        // training
        status_t set_training_formats();

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    matmul_pd_->scratchpad_registry());
        }
    };

    status_t init(engine_t *engine) override {
        CHECK(pd()->matmul_pd_->create_primitive(matmul_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> matmul_;
};

} // namespace inner_product
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

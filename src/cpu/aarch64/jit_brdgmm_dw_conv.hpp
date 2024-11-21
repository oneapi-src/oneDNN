/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_BRDGMM_DW_CONV_HPP
#define CPU_AARCH64_JIT_BRDGMM_DW_CONV_HPP

#include "common/primitive.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/aarch64/brgemm/brgemm.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa>
struct brdgmm_dw_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("brdgmm_dw:", jcp_.isa, ""),
                brdgmm_dw_convolution_fwd_t);

        status_t init(engine_t *engine);
        jit_brdgmm_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();
        std::vector<brgemm_t> bcps_;
        std::vector<brgemm_batch_element_t> batches_;
        std::vector<int> bs_;
        size_t buffer_size_ = 0;

    private:
        status_t init_brdgmm_conf();
        status_t init_scratchpad();
        void init_batch_elements();
    };

    brdgmm_dw_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    std::vector<std::unique_ptr<brgemm_kernel_t>> brdgmm_kernels_;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

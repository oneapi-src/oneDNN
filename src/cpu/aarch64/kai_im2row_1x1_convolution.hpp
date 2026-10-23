/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_KAI_IM2ROW_1X1_CONVOLUTION_HPP
#define CPU_AARCH64_KAI_IM2ROW_1X1_CONVOLUTION_HPP

#include "cpu/aarch64/kai_convolution_base.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct kai_im2row_1x1_convolution_fwd_t : public kai_convolution_fwd_base_t {
    struct pd_t : public kai_convolution_fwd_base_t::pd_t {
        using kai_convolution_fwd_base_t::pd_t::pd_t;

        DECLARE_COMMON_PD_T("im2row_1x1:kai", kai_im2row_1x1_convolution_fwd_t,
                USE_GLOBAL_SCRATCHPAD);

        status_t init(const engine_t *engine) {
            return kai_convolution_fwd_base_t::pd_t::init(engine);
        }

    private:
        const char *impl_base_name() const override { return "im2row_1x1:kai"; }
        status_t init_datapath(const engine_t *engine) override;
        void book_datapath_scratchpad(memory_tracking::registrar_t &scratchpad,
                size_t src_dt_size) const override;
    };

    kai_im2row_1x1_convolution_fwd_t(const pd_t *apd)
        : kai_convolution_fwd_base_t(apd) {}

private:
    void copy_im2row(const kernel_call_args_t &args, char *im2row) const;
    status_t setup_kernel_arrays(const kernel_call_args_t &args) const override;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

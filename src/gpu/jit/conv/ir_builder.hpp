/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GPU_JIT_CONV_IR_BUILDER_HPP
#define GPU_JIT_CONV_IR_BUILDER_HPP

#include <array>

#include "common/convolution_pd.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/ir_builder.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/tensor.hpp"

#include "gpu/jit/conv/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class conv_ir_builder_t : public ir_builder_t {
public:
    conv_ir_builder_t(
            const conv_config_t &cfg, const kernel_info_t &kernel_info)
        : ir_builder_t(kernel_info), prb_(cfg.prb()), cfg_(cfg) {
        build();
    }

private:
    void build() override;
    void init_fwd(gemm_schedule_t &gemm_schedule, view_t &src_view,
            view_t &wei_view, view_t &dst_view, expr_t &src_buf,
            expr_t &wei_buf, expr_t &dst_buf);
    void init_bwd_d(gemm_schedule_t &gemm_schedule, view_t &dst_view,
            view_t &wei_view, view_t &src_view, expr_t &dst_buf,
            expr_t &wei_buf, expr_t &src_buf);
    void init_bwd_w(gemm_schedule_t &gemm_schedule, view_t &src_view,
            view_t &dst_view, view_t &wei_view, view_t &bia_view,
            expr_t &src_buf, expr_t &dst_buf, expr_t &wei_buf, expr_t &bia_buf,
            expr_t &bia_reduction_condition);

    const conv_problem_t &prb_;
    const conv_config_t &cfg_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

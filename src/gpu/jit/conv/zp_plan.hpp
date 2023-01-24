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

#ifndef GPU_JIT_CONV_ZP_PLAN_HPP
#define GPU_JIT_CONV_ZP_PLAN_HPP

#include <string>

#include "common/utils.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/plan_utils.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/send_plan.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct zp_plan_impl_t;

struct zp_plan_t : public base_plan_t {
    zp_plan_t(ngen::HW hw);
    ~zp_plan_t();
    void init(const conv_config_t &cfg, const gemm_schedule_t &gemm_schedule,
            const view_t &zp_view, const layout_t &src_layout,
            const layout_t &wei_layout, const layout_t &dst_layout);

    explicit operator bool() const;
    int load_reg_buf_size() const;
    int mask_reg_buf_size() const;
    int comp_reg_buf_size() const;
    stmt_t load_create_stmt(const expr_t &mem_buf, const expr_t &reg_buf,
            int subtile_idx) const;
    stmt_t comp_init_create_stmt(buffer_manager_t &buf_mgr,
            const expr_t &zp_buf, const expr_t &wei_buf, const expr_t &comp_buf,
            int subtile_idx) const;
    stmt_t mask_init_create_stmt(const expr_t &mask_buf, int subtile_idx) const;
    stmt_t comp_apply_create_stmt(const expr_t &comp_buf,
            const expr_t &mask_buf, const expr_t &c_buf, int subtile_idx) const;
    bool can_split(abc_kind_t abc, int factor) const;
    void set_split(abc_kind_t abc, int factor);
    int estimate_regs() const;
    std::string str() const;

    IR_DEFINE_DUMP()

    std::unique_ptr<zp_plan_impl_t> impl;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

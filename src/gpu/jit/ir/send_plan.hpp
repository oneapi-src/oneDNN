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

#ifndef GPU_JIT_IR_SEND_PLAN_HPP
#define GPU_JIT_IR_SEND_PLAN_HPP

#include <memory>

#include "gpu/jit/ir/hw.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class send_plan_impl_t;
struct send_params_t;

class send_plan_t {
public:
    send_plan_t();
    send_plan_t(send_plan_t &&other);
    send_plan_t(std::unique_ptr<send_plan_impl_t> impl);
    ~send_plan_t();
    send_plan_t &operator=(send_plan_t &&other);

    operator bool() const { return (bool)impl_; }

    const send_params_t &send_params() const;
    bool is_2d() const;
    bool is_scattered() const;
    const layout_t &reg_layout() const;
    int reg_buf_size() const;
    void set_reg_buf_size(int size);

    stmt_t create_stmt(const expr_t &mem_buf, const expr_t &reg_buf,
            int subtile_idx = 0) const;

    int estimate_regs(bool with_buffer = true, bool with_headers = true,
            bool reuse_headers = false) const;
    bool can_split(int factor) const;
    void set_split(int factor);
    int split_factor() const;

    std::string str(const std::string &tag = "send_plan") const;

    IR_DEFINE_DUMP()

private:
    std::unique_ptr<send_plan_impl_t> impl_;
};

bool can_use_send_plan(const view_t &view);

send_plan_t create_send_plan(const exec_config_t &exec_cfg, const view_t &view,
        const send_params_t &send_params, bool zero_out = true);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

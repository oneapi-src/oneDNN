/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gpu/jit/ir/hw_config.hpp"
#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class send_plan_impl_t;
struct send_hint_t;

class send_plan_t {
public:
    send_plan_t();
    send_plan_t(send_plan_t &&other);
    send_plan_t(std::unique_ptr<send_plan_impl_t> impl);
    ~send_plan_t();

    operator bool() const { return (bool)impl_; }

    bool is_2d() const;
    const layout_t &reg_layout() const;
    int reg_buf_size() const;

    stmt_t create_stmt(const send_hint_t &hint, const expr_t &mem_buf,
            const expr_t &reg_buf) const;

private:
    std::unique_ptr<send_plan_impl_t> impl_;
};

send_plan_t create_send_plan(
        const hw_config_t &hw_cfg, const view_t &view, const send_hint_t &hint);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

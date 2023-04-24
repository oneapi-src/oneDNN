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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_XBYAK_PRINTER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_XBYAK_PRINTER_HPP

#include <memory>
#include <sstream>
#include <vector>

#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/viewer.hpp>
#include <compiler/jit/xbyak/ir/reg_allocation/virtual_slot.hpp>
#include <util/pos_track_stream.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

class xbyak_printer_t : public ir_viewer_t {
public:
    xbyak_printer_t(std::ostream &os, const_ir_module_ptr &ir_mod,
            x86_64::target_profile_t &profile);

private:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;

    func_c dispatch(func_c e) override;
    stmt_c dispatch(stmt_c e) override;
    expr_c dispatch(expr_c e) override;

    void view(assign_c v) override;
    void view(stmts_c v) override;
    void view(if_else_c v) override;
    void view(evaluate_c v) override;
    void view(for_loop_c v) override;
    void view(returns_c v) override;
    void view(define_c v) override;

    void print_index_indents(int64_t index);
    void print_padding_indents();

    ostream &print_expr_info(ostream &os, const expr &arg);
    ostream &print_expr_vec(ostream &os, const std::vector<expr_c> &args);

    x86_64::target_profile_t &profile_;
    std::shared_ptr<virtual_slots_map_t> virtual_slots_map_;

    constexpr static int index_width_ = 6;
    int indent_ = 0;

    track_pos_stream_t ss_;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

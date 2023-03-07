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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_TRANSFORM_REGISTER_ALLOCATION_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_TRANSFORM_REGISTER_ALLOCATION_HPP

#include <compiler/ir/function_pass.hpp>
#include <compiler/jit/xbyak/x86_64/target_profile.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

namespace attr_keys {
// attr for func, contains all global spilled exprs inside func
// data type: std::vector<expr_c>
constexpr const char *global_spilled = "global_spilled";
// attr for func, contains all utilized registers inside func
// data type: std::set<virt_reg_index_t>
constexpr const char *register_usage = "register_usage";
// attr for for_loop, contains load assign for spilled iter_begin_
// data type: stmt
constexpr const char *load_loop_begin = "load_loop_begin";
// attr for for_loop, contains load assign for spilled iter_end_
// data type: stmt
constexpr const char *load_loop_end = "load_loop_end";
// attr for for_loop, contains load assign for spilled step_
// data type: stmt
constexpr const char *load_loop_step = "load_loop_step";
} // namespace attr_keys

/* *
 * Register allocation uses virtual_slot to represent physcial register, each
 * expr have its own virtual_reg which contains its reg_type and live_range.
 * Pre-allocation pass put all virtual_regs in a priority_queue.
 * For each virtual_reg, allocator use virtual_slots_map to aquire corresponding
 * virtual_slot index range in a virtual_slots_array for assignment and try to
 * find a available slot that does not interfer with other virtual_regs.
 * Virtual_slot uses non-overlapping balanced intervel tree to store live_range
 * and check interference, if no virtual_slot is available, virtual_regs is
 * spilled and spill will be resolved using address mode checking.
 *
 *        <------------------Live Range------------------>
 *        _______________________________________________
 * Slot 0|__[--virt_reg_0--)___[-virt_reg_2-)____________|
 * Slot 1|________[------virt_reg_1------)_______________|
 * Slot 2|_________________[-------virt_reg_3-------)____|
 * Slot 3|_______________________________________________|
 * Slot ...
 * */
class register_allocation_t : public function_pass_t {
public:
    register_allocation_t(const x86_64::target_profile_t &profile);
    func_c operator()(func_c v) override;

private:
    const x86_64::target_profile_t &profile_;
};

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

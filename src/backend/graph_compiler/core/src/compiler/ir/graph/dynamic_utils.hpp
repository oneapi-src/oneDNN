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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_UTILS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_UTILS_HPP
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <compiler/dimensions.hpp>
#include <runtime/dynamic_dispatch/op_dispatch_tables.hpp>
#include <util/def.hpp>
namespace sc {
class sc_op;
struct op_dispatch_key_base_t;
// op tables used in ir_module
struct op_dispatch_tables_t {
    // format table:
    // for tunable op: input format keys => out formats value
    // for reorder op: the field is meaningless.
    // for other ops: input format keys with only one known format => all in/out
    // formats value
    std::unordered_map<std::vector<runtime::dispatch_key>,
            std::vector<runtime::dispatch_key>>
            format_table_;
    // kernel table: in/out format keys => function symbol
    std::unordered_map<std::vector<runtime::dispatch_key>, std::string>
            kernel_table_;
};

using op_dispatch_tables_ptr = std::shared_ptr<op_dispatch_tables_t>;
using dispatch_table_map_t
        = std::unordered_map<std::string, op_dispatch_tables_ptr>;

void initialize_format_table_with_op(
        const std::shared_ptr<sc_op> &op, op_dispatch_tables_ptr &tb);
void add_dispatch_symbol_to_kernel_table(op_dispatch_tables_ptr &tb,
        const op_dispatch_key_base_t *keys, const std::string &func_name);
bool can_op_be_dispatched(const std::shared_ptr<sc_op> &op);
namespace runtime {
struct dynamic_tensor_t;
}
struct graph_tensor;
SC_API runtime::dynamic_tensor_t convert_graph_tensor_to_dynamic_tensor(
        const std::shared_ptr<graph_tensor> &in, void *data_ptr = nullptr,
        sc_dim *shape_ptr = nullptr);
} // namespace sc
#endif

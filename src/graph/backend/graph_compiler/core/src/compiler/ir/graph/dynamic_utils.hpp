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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_UTILS_HPP
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include <compiler/dimensions.hpp>
#include <compiler/ir/sc_expr.hpp>
#include <runtime/dynamic_dispatch/op_dispatch_tables.hpp>
#include <runtime/threadpool_mode.hpp>
#include <util/def.hpp>
#include <util/reflection.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
class sc_op;
class sc_graph_t;
struct graph_tensor;
struct op_dispatch_key_base_t;
struct combined_op_dispatch_key_t;
struct dispatch_key_set_base_t;
struct sc_data_format_t;
class ir_module_t;
struct context_t;
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
    /**
     * the struct to hold info of a partial specialization instance of a dynamic
     * shaped op.
     * If graph_ is null, the field `name_or_postfix_` is the only
     * valid field of `op_func_info`. `name_or_postfix_` indicates the name of
     * TIR function in the IR module for the partial specialization.
     * If graph_ is not null, the partial specialization has not been compiled
     * when `lower_graph()` is called. The field `op_` contains the dynamic
     * shaped op, whose container is stored in `graph_`. The `name_or_postfix_`
     * is the postfix for the TIR function name of the partial specialization
     */
    struct op_func_info {
        std::shared_ptr<sc_graph_t> graph_;
        std::shared_ptr<sc_op> op_;
        std::string name_or_postfix_;
        std::shared_ptr<context_t> ctx_;
        std::shared_ptr<const thread_pool_mode_t> use_managed_tp_;
        bool internal_;
        // only valid when `!already_compiled()`. Lower the partial
        // specialization to TIR
        std::shared_ptr<ir_module_t> lower();
        /**
         * op_func_info which indicates that the partial specialization is not
         * compiled or lowered yet, internal means the internal functions of op.
         */
        op_func_info(const std::shared_ptr<sc_graph_t> &graph,
                const std::shared_ptr<sc_op> &op,
                const std::string &name_or_postfix,
                const std::shared_ptr<context_t> &ctx,
                const std::shared_ptr<const thread_pool_mode_t> &use_managed_tp,
                bool internal = false)
            : graph_(graph)
            , op_(op)
            , name_or_postfix_(name_or_postfix)
            , ctx_(ctx)
            , use_managed_tp_(use_managed_tp)
            , internal_(internal) {}
        /**
         * op_func_info which indicates that the partial specialization is
         * already lowered to TIR in the current IR module
         */
        op_func_info(const std::string &name_or_postfix)
            : name_or_postfix_(name_or_postfix), internal_(false) {}
        bool already_compiled() { return graph_ == nullptr; }
    };
    // config table: configs of tunable_op => impl kind
    std::unordered_map<std::vector<uint64_t>, int> impl_kind_table_;
    // kernel table: in/out format keys => function symbol
    std::unordered_map<std::vector<runtime::dispatch_key>, op_func_info>
            kernel_table_;
    reflection::shared_general_object_t op_info_;
};

using op_dispatch_tables_ptr = std::shared_ptr<op_dispatch_tables_t>;
using dispatch_table_map_t
        = std::unordered_map<std::string, op_dispatch_tables_ptr>;

void initialize_format_table_with_op(
        const std::shared_ptr<sc_op> &op, op_dispatch_tables_ptr &tb);
void initialize_impl_kind_table_with_op(const std::shared_ptr<context_t> &ctx,
        const std::shared_ptr<sc_op> &op, op_dispatch_tables_ptr &tb);
void initialize_dispatch_table_with_op(const std::shared_ptr<context_t> &ctx,
        const std::shared_ptr<sc_op> &op, op_dispatch_tables_ptr &tb);
void add_dispatch_symbol_to_kernel_table(op_dispatch_tables_ptr &tb,
        const op_dispatch_key_base_t *keys,
        op_dispatch_tables_t::op_func_info &&func_module);
bool is_internal_op(const std::shared_ptr<sc_op> &op);
bool can_op_be_dispatched(const std::shared_ptr<sc_op> &op);
bool can_op_query_output(const std::shared_ptr<sc_op> &op);
bool can_op_be_queried(const std::shared_ptr<sc_op> &op);
std::vector<std::shared_ptr<dispatch_key_set_base_t>>
get_dispatch_set_vec_from_ops(const std::vector<std::shared_ptr<sc_op>> &ops);

// find the parent node who could be dispatched related to current op, parent
// node could be tuanble op/reorder op/input op.
std::shared_ptr<sc_op> find_parent_dispatch_node(
        const std::shared_ptr<graph_tensor> &in);

constexpr const int no_link_idx = -1;
using op_layout_link_vec_t = std::vector<std::vector<std::pair<int, int>>>;
// The first dim is the op index who has effective dispatch keys inside fused
// op, the second dim is op's in/out format index, the value is linked op index
// and linked op in/out format index pair.
op_layout_link_vec_t get_op_layout_link_relationships(
        const std::vector<std::shared_ptr<sc_op>> &ops,
        const std::vector<std::shared_ptr<dispatch_key_set_base_t>>
                &dispatch_keys,
        const std::shared_ptr<sc_op> &modified_inp);
// Query function order, if we meet reorder, query its next tunable op first.
void lower_query_function(std::vector<bool> &visited,
        const std::shared_ptr<sc_op> &node,
        const std::function<void(const std::shared_ptr<sc_op> &)> &callback);

void visit_fused_graph_by_query_order(sc_graph_t &graph,
        const std::function<void(const std::shared_ptr<sc_op> &)> &callback);
// Judge whether the two input layout could be linked. The linked means the
// graph is in the valid status of layout. For example, pattern like "reorder +
// matmul", the output layout of reorder should be equal to the input layout of
// matmul. And blocking factor =1 is prepared for binary elementwise op with
// broadcast semantic. For example, MKmk(32, 16) and MKmk(1, 16) are linked,
// while MKmk(32,16) and MKmk(1, 32) not.
bool is_linked_layout(
        const sc_data_format_t &layout1, const sc_data_format_t &layout2);
std::vector<std::shared_ptr<sc_op>> get_graph_inner_dispatch_ops(
        sc_graph_t &graph, int *total_num_key);
void update_graph_format_by_key(const std::shared_ptr<context_t> &ctx,
        const std::shared_ptr<sc_op> &fused_op, sc_graph_t &graph,
        const combined_op_dispatch_key_t &key, int &key_idx,
        size_t node_input_offset, size_t graph_input_offset,
        const std::shared_ptr<sc_op> &modified_inp = nullptr);
expr call_op_dynamic_query_function(
        const std::shared_ptr<sc_op> &op, const std::vector<expr> &args);
void create_internal_dispatch_funcs_by_node(
        const std::shared_ptr<context_t> &ctx,
        std::shared_ptr<ir_module_t> &ret_mod, const std::string &table_name,
        const std::shared_ptr<sc_op> &node,
        const std::shared_ptr<const thread_pool_mode_t> &use_mtp);
void create_dispatch_funcs_by_keys(const std::shared_ptr<context_t> &ctx,
        std::shared_ptr<ir_module_t> &ret_mod, const std::string &table_name,
        const std::shared_ptr<sc_op> &node, const op_dispatch_key_base_t *key,
        expr &op_dispatch_kernel, int &dyn_idx,
        const std::shared_ptr<const thread_pool_mode_t> &use_mtp,
        bool internal);
int get_num_of_internal_funcs(const std::shared_ptr<sc_op> &node);
int get_num_of_internal_funcs(const sc_graph_t &graph);
int count_dynamic_dims(const sc_dims &in);

namespace runtime {
struct dynamic_tensor_t;
}
SC_API runtime::dynamic_tensor_t convert_graph_tensor_to_dynamic_tensor(
        const std::shared_ptr<graph_tensor> &in, void *data_ptr = nullptr,
        sc_dim *shape_ptr = nullptr);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif

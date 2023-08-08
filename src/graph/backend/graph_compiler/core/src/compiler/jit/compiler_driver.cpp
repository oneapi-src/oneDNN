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

#include <functional>
#include "jit.hpp"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/graph_code_cache.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_API std::shared_ptr<jit_function_t> compiler_driver(const context_ptr &ctx,
        sc_graph_t &graph,
        const std::function<std::vector<sc_op_ptr>(sc_graph_t &graph)>
                &get_args,
        const dnnl::impl::graph::gc::graph_config *in_cfg,
        ir_module_ptr *out_ir_module) {
    graph_driver(graph, ctx, in_cfg, nullptr, 0, 0, 0, nullptr, nullptr,
            nullptr, /*allow_cache*/ true);
    if (auto code = graph.attrs_.get_or_else<std::shared_ptr<jit_module_code>>(
                "graph_code_cache", nullptr)) {
        auto static_tbl = prepare_static_table_for_cached_code(
                *code->graph_cache_handle_, graph);

        return std::make_shared<jit_module>(std::move(static_tbl), code)
                ->get_function(code->entry_func_name_);
    }

    auto irm = lower_graph(ctx, graph, get_args(graph));
    if (out_ir_module) { *out_ir_module = irm; }
    auto ret = jit_engine_t::make(ctx)->get_entry_func(irm, true);
    auto jitm = ret->get_module();

    if (auto graph_key = graph.attrs_.get_or_else<
                         std::shared_ptr<prehashed_graph_for_code_share_t>>(
                "graph_code_cache_key", nullptr)) {
        register_code_in_graph_cache(*jitm, std::move(graph_key));
    }
    return ret;
}

SC_API std::shared_ptr<jit_function_t> compiler_driver(const context_ptr &ctx,
        sc_graph_t &graph, const std::vector<sc_op_ptr> &args,
        const dnnl::impl::graph::gc::graph_config *in_cfg,
        ir_module_ptr *out_ir_module) {
    return compiler_driver(
            ctx, graph, [&args](sc_graph_t &) { return args; }, in_cfg,
            out_ir_module);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

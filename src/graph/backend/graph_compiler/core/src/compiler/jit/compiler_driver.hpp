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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_COMPILER_DRIVER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_COMPILER_DRIVER_HPP

#include <functional>
#include <memory>
#include <vector>
#include "jit.hpp"
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/graph_config.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
/**
 * The top-level frontend of the compiler. It runs the Graph IR passes, lowers
 * the GIR to Tensor IR, runs TIR passes and finally JIT-compiles the code. It
 * may reuse the privious compilation result, if the graph is the same and the
 * previous compiled JIT-module is still alive.
 * @param ctx compiler context
 * @param graph the graph
 * @param get_args the callback to be called after the graph passes and before
 * lowering the graph. It should accept the graph after the graph passes and
 * return the arguments Ops of the graph. The order of the returned arguments
 * will be the order of the arguments of JIT'd main-entry function
 * @param in_cfg optional graph config
 * @param out_ir_module optionally returning the generated TIR module
 * @returns the JIT'd executable main entry function of the graph. It may share
 * the code and buffer with previously compiled graph.
 */
SC_API std::shared_ptr<jit_function_t> compiler_driver(const context_ptr &ctx,
        sc_graph_t &graph,
        const std::function<std::vector<sc_op_ptr>(sc_graph_t &graph)>
                &get_args,
        const dnnl::impl::graph::gc::graph_config *in_cfg = nullptr,
        ir_module_ptr *out_ir_module = nullptr);

/**
 * @see compiler_driver above. The simplifed interface of compiler_driver. The
 * args are given in vectors.
 */
SC_API std::shared_ptr<jit_function_t> compiler_driver(const context_ptr &ctx,
        sc_graph_t &graph, const std::vector<sc_op_ptr> &args,
        const dnnl::impl::graph::gc::graph_config *in_cfg = nullptr,
        ir_module_ptr *out_ir_module = nullptr);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

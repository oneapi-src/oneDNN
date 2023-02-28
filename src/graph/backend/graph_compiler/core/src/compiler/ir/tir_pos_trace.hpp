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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TIR_POS_TRACE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TIR_POS_TRACE_HPP

#include <string>
#include <compiler/ir/sc_function.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// the tracer to track the current TIR location. Useful for pretty-printing the
// TIR error with detailed context.
struct tir_pos_tracer {
    const node_base *cur_func_ = nullptr;
    const node_base *cur_node_ = nullptr;
    // returns the string representation of current function and the source
    // position of the current node
    std::string to_string() const;
};

// the RAII holder to set and auto-reset the tir_pos_tracer. Use TIR_ERROR_TRACE
// in IR visitors
struct tir_pos_trace {
    const node_base *const old_value_;
    const node_base **const target_value_;
    tir_pos_trace(tir_pos_tracer *tracer, const func_base *v)
        : old_value_(tracer->cur_func_), target_value_(&tracer->cur_func_) {
        tracer->cur_func_ = v;
    }
    tir_pos_trace(tir_pos_tracer *tracer, const stmt_base_t *v)
        : old_value_(tracer->cur_node_), target_value_(&tracer->cur_node_) {
        tracer->cur_node_ = v;
    }
    tir_pos_trace(tir_pos_tracer *tracer, const expr_base *v)
        : old_value_(tracer->cur_node_), target_value_(&tracer->cur_node_) {
        // var and tensors are used multiple times and are hard to trace the
        // location of the use
        if (v->node_type_ != sc_expr_type::var
                && v->node_type_ != sc_expr_type::tensor) {
            tracer->cur_node_ = v;
        }
    }
    ~tir_pos_trace() { *target_value_ = old_value_; }
};

#define TIR_ERROR_TRACE(v) \
    tir_pos_trace __sc_tir_error_trace {&this->pass_error_tracer_, v.get()};

#define COMPILE_ASSERT_POS(cond, ...) \
    COMPILE_ASSERT(cond, this->pass_error_tracer_.to_string() << __VA_ARGS__)

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

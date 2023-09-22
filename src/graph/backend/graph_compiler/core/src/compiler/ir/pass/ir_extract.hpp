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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_IR_EXTRACT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_IR_EXTRACT_HPP

#include <vector>
#include "../viewer.hpp"
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Extract vars from given IR. Will NOT change the given IR
 *
 * @param vars_ In and out parameter. Holds the extracted vars
 * */
class ir_var_extractor_t : public ir_viewer_t {
protected:
    std::vector<expr_c> &vars_;
    std::unordered_set<expr_c> dedup_;

public:
    ir_var_extractor_t(std::vector<expr_c> &vars) : vars_(vars) {}

    void view(var_c v) override;
    void operator()(const expr_c &v);
    void operator()(const stmt_c &v);
    void operator()(func_c v);
};

/**
 * Capture vars from given IR prior to a statement (i.e. terminator). Will NOT
 * change the given IR
 *
 * @param vars_ Out parameter. Holds the captured vars
 * @param terminator_ In parameter. Holds the stmt where extraction terminates
 * */
class ir_var_capturer_t : public ir_viewer_t {
protected:
    std::vector<expr_c> &vars_;
    stmt_c terminator_;
    bool terminated_;

public:
    ir_var_capturer_t(std::vector<expr_c> &vars, stmt_c t = stmt_c())
        : vars_(vars), terminator_(t) {}

    void view(define_c v) override;
    void view(stmts_c v) override;
    void view(for_loop_c v) override;
    void view(if_else_c v) override;
    void operator()(const stmt_c &v);
    void operator()(const func_c &v);
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

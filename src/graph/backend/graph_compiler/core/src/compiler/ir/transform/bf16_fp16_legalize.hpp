/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BF16_FP16_LEGALIZE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BF16_FP16_LEGALIZE_HPP

#include <tuple>
#include <utility>
#include "../function_pass.hpp"
#include "../sc_function.hpp"
#include "../viewer.hpp"
#include "../visitor.hpp"
#include <compiler/config/context.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class bf16_fp16_promote_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    context_ptr ctx_;
    bf16_fp16_promote_impl_t(context_ptr ctx = get_default_context())
        : ctx_(std::move(ctx)) {}
    std::tuple<expr_c, expr_c> docast(
            const expr &orig_a, const expr &orig_b, bool *is_low_precision_fp);
    expr_c visit(binary_c v) final;
    expr_c visit(cmp_c v) final;
    expr_c visit(select_c v) final;
    expr_c visit(intrin_call_c v) final;
};

// An analyzer viewer runs before elimination to count the valid usage number of
// bf16 / fp16 vars, to decide whether they need to be promoted to f32.
class bf16_fp16_elimination_analyzer_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    context_ptr ctx_;
    std::unordered_map<expr_c, int> var_use_cnt_;
    bf16_fp16_elimination_analyzer_t(context_ptr ctx) : ctx_(std::move(ctx)) {}
    void view(var_c v) override;
    void view(assign_c v) override;
    void view(define_c v) override;
    void view(intrin_call_c v) override;
};

class bf16_fp16_cast_elimination_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    context_ptr ctx_;
    // need to convert bf16 / fp16 var to f32
    std::unordered_map<expr_c, expr_c> cvt_map_;
    // inherit from analyzer
    std::unordered_map<expr_c, int> &var_use_cnt_;
    expr_c visit(cast_c v) final;
    expr_c visit(var_c v) final;
    stmt_c visit(define_c v) final;
    stmt_c visit(assign_c v) final;
    stmt_c visit(returns_c v) final;
    bf16_fp16_cast_elimination_impl_t(
            context_ptr ctx, std::unordered_map<expr_c, int> &var_use_cnt)
        : ctx_(ctx), var_use_cnt_(var_use_cnt) {}
};

/**
 * bfloat16 and floating point16 legalize pass.
 *
 * It will do the following (a, b as bfloat16 input, c as bfloat16 output, "+"
 * as example):
 * c = a + b => c = bf16(float(a)+float(b))
 * c = a + neg(b) => c = bf16(float(a), neg(float(b)))
 * */
class bf16_fp16_legalizer_t : public function_pass_t {
public:
    bf16_fp16_legalizer_t(context_ptr ctx = get_default_context())
        : ctx_(std::move(ctx)) {}
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c f);
    expr_c operator()(expr_c f);
    SC_DECL_PASS_INFO_FUNC();

private:
    context_ptr ctx_;
};

/**
 * bfloat16 and floating point16 elimination pass.
 *
 * The pass should be evaluated after bf16_fp16_legalize and recommended after
 * index2var.
 * It will do the two elimination:
 * 1. Eliminate consecutive bf16 transformations, e.g. f32(bf16(f32(a))) =>
 * f32(a)
 * 2. Promote consecutive bf16 calculation stmt of var to f32(only for var).
 */
class bf16_fp16_eliminator_t : public function_pass_t {
public:
    bf16_fp16_eliminator_t(context_ptr ctx) : ctx_(std::move(ctx)) {}
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c f);
    expr_c operator()(expr_c f);
    SC_DECL_PASS_INFO_FUNC();

private:
    context_ptr ctx_;
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

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
#ifndef BACKEND_GRAPH_COMPILER_PATTERNS_NORM_PATTERN_HPP
#define BACKEND_GRAPH_COMPILER_PATTERNS_NORM_PATTERN_HPP

#include <memory>

#include "backend/graph_compiler/patterns/fusions.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace compiler_impl {
namespace pass {

namespace pm = impl::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = impl::utils::pm::pb_graph_t;
using FCreatePattern = impl::pass::FCreatePattern;

/* instance_norm
       [IN0]     [IN1]          [IN2]
         |         |              |
         |         |          ReduceMean(1)
         |         |        /           |
         |  SquareDifference            |
         |         |                    |
         |    ReduceMean(2)             |
         |         |                    |
         |       Add1 - [IN3]           |
         |         |                    |
         |     Rsqrt  [IN4]             | 
         |         |   /                |
         |   Multiply(1) --            /
          \      /         \          /
          Multiply(2)       Multiply(3)
              |                 |
              |              Subtract - [IN5]
               \              /      
                     Add(2)
                       |
                     [OUT0]
*/

std::shared_ptr<pb_graph_t> instance_norm_subgraph(
        const pm::decision_function &type_check_fn) {
    // create instance norm subgraph
    auto instance_norm_subgraph
            = std::make_shared<pb_graph_t>("instance_norm_subgraph");

    auto reduce_mean_1 = instance_norm_subgraph->append_op(
            impl::op_kind::ReduceMean, "reduce_mean_1");
    reduce_mean_1->append_decision_function(check_reduce_attrs);
    reduce_mean_1->append_decision_function(type_check_fn);

    auto sqr_diff = instance_norm_subgraph->append_op(
            impl::op_kind::SquaredDifference, {in_edge(1, reduce_mean_1, 0)},
            "squared_difference");
    sqr_diff->append_decision_function(type_check_fn);

    auto reduce_mean_2
            = instance_norm_subgraph->append_op(impl::op_kind::ReduceMean,
                    {in_edge(0, sqr_diff, 0)}, "reduce_mean_2");
    reduce_mean_2->append_decision_function(check_reduce_attrs);

    auto add_1 = instance_norm_subgraph->append_op(
            impl::op_kind::Add, {in_edge(0, reduce_mean_2, 0)}, "add_1");
    add_1->append_decision_function(type_check_fn);

    auto rsqrt = instance_norm_subgraph->append_op(
            impl::op_kind::Rsqrt, {in_edge(0, add_1, 0)}, "rsqrt");

    auto mul_1 = instance_norm_subgraph->append_op(
            impl::op_kind::Multiply, {in_edge(0, rsqrt, 0)}, "mul_1");
    mul_1->append_decision_function(type_check_fn);
    auto mul_2 = instance_norm_subgraph->append_op(
            impl::op_kind::Multiply, {in_edge(1, mul_1, 0)}, "mul_2");
    mul_2->append_decision_function(type_check_fn);
    auto mul_3 = instance_norm_subgraph->append_op(impl::op_kind::Multiply,
            {in_edge(0, reduce_mean_1, 0), in_edge(1, mul_1, 0)}, "mul_3");

    auto sub = instance_norm_subgraph->append_op(
            impl::op_kind::Subtract, {in_edge(1, mul_3, 0)}, "sub");
    sub->append_decision_function(type_check_fn);

    auto add_2 = instance_norm_subgraph->append_op(impl::op_kind::Add,
            {in_edge(0, mul_2, 0), in_edge(1, sub, 0)}, "add_2");

    instance_norm_subgraph->create_input_port(0, mul_2, 0);
    instance_norm_subgraph->create_input_port(1, sqr_diff, 0);
    instance_norm_subgraph->create_input_port(2, reduce_mean_1, 0);
    instance_norm_subgraph->create_input_port(3, add_1, 1);
    instance_norm_subgraph->create_input_port(4, mul_1, 1);
    instance_norm_subgraph->create_input_port(5, sub, 0);
    instance_norm_subgraph->create_output_port(0, add_2, 0);

    return instance_norm_subgraph;
};

std::shared_ptr<pb_graph_t> relu_alternation_subgraph(
        const pm::decision_function &type_check_fn) {
    /* Append a subgraph of relu alternation
     *         |
     *  [Relu/LeakyRelu]
     *         |
     * */
    auto relu_option = std::make_shared<pb_graph_t>("relu_option");
    auto relu_op = relu_option->append_op(impl::op_kind::ReLU, "relu");
    relu_op->append_decision_function(type_check_fn);
    relu_option->create_input_port(0, relu_op, 0);
    relu_option->create_output_port(0, relu_op, 0);

    auto leaky_relu_option = std::make_shared<pb_graph_t>("leaky_relu_option");
    auto leaky_relu_op = leaky_relu_option->append_op(
            impl::op_kind::LeakyReLU, "leaky_relu");
    leaky_relu_op->append_decision_function(type_check_fn);
    leaky_relu_option->create_input_port(0, leaky_relu_op, 0);
    leaky_relu_option->create_output_port(0, leaky_relu_op, 0);

    auto relu_alternation_unit
            = std::make_shared<pb_graph_t>("relu_alternation_unit");
    auto relu_alter = relu_alternation_unit->append_alternation(
            {relu_option, leaky_relu_option}, "relu_alternation_op");
    relu_alternation_unit->create_input_port(0, relu_alter, 0);
    relu_alternation_unit->create_output_port(0, relu_alter, 0);

    return relu_alternation_unit;
};

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(fp32_norm_pattern)

/* f32_instance_norm[_[leaky_]relu]
    (f32)[IN0] (f32)[IN1] (f32)[IN2]
         |         |          |
         |         |      ReduceMean(1)
         |         |        /   \
         |  SquareDifference     |
         |         |             |
         |    ReduceMean(2)      |
         |         |             |
         |      Add1 -[IN3](f32) |
         |         |             |
         |     Rsqrt (f32)[IN4]  |
         |         |   /         |
         |      Multiply(1)      |
          \      /    \         /
         Multiply(2)   Multiply(3)
              |            |
              |       Subtract - [IN5](f32)
               \          /
                  Add(2)
                   |
             [Relu/LeakyRelu]
                   |
               [OUT0](f32)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, fp32_instance_norm_pattern)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    std::shared_ptr<pb_graph_t> ins_norm_subgraph
                            = instance_norm_subgraph(
                                    check_input_dtype<impl::data_type::f32>);
                    // use [1, 2) repetition to add subgraph
                    auto instance_norm_op
                            = pgraph->append_repetition(ins_norm_subgraph,
                                    {0, 0}, 1, 2, "instance_norm_subgraph");

                    auto relu_alternation = relu_alternation_subgraph(
                            check_input_dtype<impl::data_type::f32>);
                    pgraph->append_optional(relu_alternation,
                            {in_edge(0, instance_norm_op, 0)},
                            "optional_relu_alternation");
                });

COMPILER_BACKEND_REGISTER_PASSES_DEF_END

COMPILER_BACKEND_REGISTER_PASSES_DEF_BEGIN(bf16_norm_pattern)

/* bf16_instance_norm[_[leaky_]relu]
    (bf16)[IN0] (bf16)[IN1]  (bf16)[IN2]
         |         |              |
         |         |          ReduceMean(1)
         |         |        /             \
         |  SquareDifference               |
         |         |                       |
         |    ReduceMean(2)                |
         |         |                       |
         |       Add1 -[IN3](f32)          |
         |         | (optional)            |
         |     Rsqrt TypeCast(1) - [IN4]   | 
         |         |   /                   |
         |   Multiply(1) --            ---/
          \      /         \          /
          Multiply(2)       Multiply(3)
              |                 |
              |              Subtract - TypeCast(2) - [IN5]
               \              /         (optional)
                     Add(2)
                       |
                 ReLU/LeakyReLU
                       |
                  [OUT0](bf16)
*/
COMPILER_BACKEND_REGISTER_TRANSFORMATION_PASS(
        compiler, bf16_instance_norm_pattern)
        .set_priority(5.0f)
        .set_kind(impl::partition_kind::misc_post_ops)
        .set_attr<FCreatePattern>("FCreatePattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto cast_subgraph
                            = std::make_shared<pb_graph_t>("cast_subgraph");
                    auto cast_op = cast_subgraph->append_op(
                            impl::op_kind::TypeCast, "optional_cast");
                    cast_subgraph->create_input_port(0, cast_op, 0);
                    cast_subgraph->create_output_port(0, cast_op, 0);

                    auto cast_1
                            = pgraph->append_optional(cast_subgraph, "cast_1");
                    auto cast_2
                            = pgraph->append_optional(cast_subgraph, "cast_2");

                    std::shared_ptr<pb_graph_t> ins_norm_subgraph
                            = instance_norm_subgraph(
                                    check_input_dtype<impl::data_type::bf16>);
                    // use [1, 2) repetition to add subgraph
                    auto instance_norm_op = pgraph->append_repetition(
                            ins_norm_subgraph, {0, 0}, 1, 2,
                            {in_edge(4, cast_1, 0), in_edge(5, cast_2, 0)},
                            "instance_norm_subgraph");

                    auto relu_alternation = relu_alternation_subgraph(
                            check_input_dtype<impl::data_type::bf16>);
                    pgraph->append_optional(relu_alternation,
                            {in_edge(0, instance_norm_op, 0)},
                            "optional_relu_alternation");
                });

COMPILER_BACKEND_REGISTER_PASSES_DEF_END

} // namespace pass
} // namespace compiler_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

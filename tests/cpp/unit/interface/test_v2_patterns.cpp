/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <memory>

#include "interface/graph.hpp"
#include "utils/pm/nested_matcher.hpp"
#include "gtest/gtest.h"

#include "cpp/unit/utils.hpp"

using namespace dnnl::graph::impl;
using namespace dnnl::graph::impl::op_kind;
using namespace dnnl::graph::impl::utils::pm;
using namespace dnnl::graph::tests::unit::utils;

const iport_t IN0 = 0;
const iport_t IN1 = 1;
const iport_t IN2 = 2;
const iport_t IN3 = 3;
const oport_t OUT0 = 0;
const oport_t OUT1 = 1;
const oport_t OUT2 = 2;

#define MUTE(x) (void)(x)

//
// All pattern starts with a "pb_graph"
//
TEST(PatternMatcherV2, Graph) {
    auto pgraph = make_shared<pb_graph_t>("pgraph");

    ASSERT_NE(pgraph, nullptr);
}

//
// Pattern is grown by appending pattern ops ("pb_op", "alternation" and
// "repetition") to a "pb_graph" with pb_graph.append_op(),
// append_alternation(), append_optional() and append_repeition().
// Pattern can be a nested graph since "alteration" and "repetition"
// embeds "pb_graph".
// Pattern graph has the following properties.
// - During matching, aggegrate pattern nodes (pb_graph, alternation,
// repetition) will be unpacked recursively until all nodes are expanded
// to just "pb_op"s
// - Any inner "pb_graph" embedded inside "alternation" or "repetition" needs
// to provide a mapping from the "pb_graph"'s in/out port to it's inner node's
// in/out port to enable unpacking. This is done by calling create_input_port()
// and create_output_port().
// - "alternation" and "repetition"'s in/out ports are mapped to the same
// numberred in/out ports of embedded "pb_graph"(s)
// - One graph op is matched with one "pb_op". And expanded pattern graph's
// "pb_op" are not aliased. So graph ops matched with different "pb_op"s cannot
// be aliased.
// - Graph op attribute checking for is done by "decision_function"s of a
// "pb_op". Every "pb_op" needs to provide at least one "decision_function".
// One "decision_function" needs to be passed as an arugument to append_op()
// Some variants of append_op() provides a quick way to setup common
// "decision_function"s.
// Use pb_op.append_decision_function() to add additional attribute checkers.
// - Pattern matcher matches graph op edges with pb_op edges. Graph ops can
// have more edges than constrained by the pattern graph. Those are marked as
// unhandled edges during matching. Unhandled edges are two types. One is
// connected to a graph op matched by this pattern and called an internal edge.
// The other is called an external edge.
// - Matcher has two different modes of handling unhandled edges. First mode
// assumes all unhandled inputs as external input and assumes unhandled outputs
// from ops matched with non root pb_op (side outputs) are not allowed.
// This mode is useful for backends backed by fixed kernels such as oneDNN
// primitives. To allow side outputs, pb_op.allow_external_output() is provided
// to override this behavior. The second mode auto exports unhandled external
// inputs and outputs.
// - "pb_op" inputs are not commutative. Use set_commutative_pair() to manually
// allow commutative input pairs.
// Pattern matcher has two different mode/way of handling unmatched graph op
// edges.
// - Order of external inputs and outputs returned by matcher is implementation
// dependent. (Port numbers provided by create_input_port() and
// create_output_port() may be used to enforce ordering for fixed patterns from
// a flat pattern graph. But the idea is not practical in general. For example,
// nested patterns may have variable number of side inputs so fixed ordering
// cannot be enforced.)
// - In case a match has multiple aliased external inputs, they are not merged
// and matcher reports them as separate inputs.
//

//
// Leaf pattern ops can be created by passing dnnl_graph op_kind.
// External inputs and outputs of a match will be ordered and
// exposed as part of the match. The order depends on matcher
// implementation.
//
TEST(PatternMatcherV2, GraphAppendLeafOp) {
    auto graphp = make_shared<pb_graph_t>("pgraph");
    // Grow internal graph
    // Leaf pattern op "Add"
    auto op0 = graphp->append_op(Add, "padd");
    ASSERT_NE(graphp, nullptr);
    ASSERT_NE(op0, nullptr);
}

//
// Convolution + BiasAdd
// A vector of all in coming edges to the new op can passed to
// append_op for non leaf pattern ops
//
TEST(PatternMatcherV2, GraphAppendNonLeafOp) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    auto graphp = make_shared<pb_graph_t>("conv_bias");
    // Grow internal graph
    // Convolution -> BiasAdd
    // Leaf pattern op
    auto op0 = graphp->append_op(Convolution, "pconv");
    // Non leaf pattern op "BiasAdd" with only one of the inputs constrained
    // input 0 is constrained to output 0 of "Abs" op
    // unconstrained input is like matching "Any"
    // input 1 is free to match any op
    auto op1 = graphp->append_op(BiasAdd, {in_edge(IN0, op0, OUT0)}, "pbias");
    // Make sure that input1 to "BiasAdd" node does not come from within
    // the matched pattern
    ASSERT_NE(op1->get_producer(IN0), nullptr);
    ASSERT_EQ(op1->get_producer(IN0)->first, op0);
    ASSERT_EQ(op1->get_producer(IN0)->second, OUT0);
    ASSERT_EQ(op0->get_consumers(OUT0)->at(0)->first, op1);
    ASSERT_EQ(op0->get_consumers(OUT0)->at(0)->second, IN0);

    graph_t gr;
    op_t *a = gr.create_op(Convolution, "conv");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(BiasAdd, "bias");
    b->fill_and_connect_input(0, *a, 0);
    a->add_input(lt_vec[2]);
    op_t *c = gr.create_op(ReLU, "relu");
    c->fill_and_connect_input(0, *b, 0);
    c->add_output(lt_vec[3]);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(m1.inputs.size(), 3);
    ASSERT_EQ(m1.outputs.size(), 1);
    match_t m2;
    EXPECT_FALSE(match_pattern(b, graphp, m2));
    match_t m3;
    EXPECT_TRUE(match_pattern(b, graphp, m3, false, false));
    ASSERT_EQ(m3.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(m3.inputs.size(), 3);
    ASSERT_EQ(m3.outputs.size(), 1);

    // mark matched dnnl_graph_op with pattern name
    for (auto p : m3.op_pb_op_pairs) {
        p.first->set_attr<string>("matched_pattern", graphp->get_name());
    }
    // dnnl_graph_ops with "match_pattern" attr set will not match.
    match_t m4;
    EXPECT_FALSE(match_pattern(a, graphp, m4));
}

TEST(PatternMatcherV2, GraphNoAllowSideOutput) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    auto op0 = graphp->append_op(Convolution, "pconv");
    auto op1 = graphp->append_op(BiasAdd, {in_edge(IN0, op0, OUT0)}, "pbias");
    MUTE(op1);

    graph_t gr;
    op_t *a = gr.create_op(Convolution, "conv");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(BiasAdd, "bias");
    b->fill_and_connect_input(0, *a, 0);
    a->add_input(lt_vec[2]);
    op_t *c = gr.create_op(ReLU, "relu");
    c->fill_and_connect_input(0, *b, 0);
    c->add_output(lt_vec[3]);
    // create a side output from "conv" to "add"
    op_t *d = gr.create_op(Add, "add");
    d->fill_and_connect_input(0, *a, 0);
    d->add_input(lt_vec[4]);

    match_t m1;
    EXPECT_FALSE(match_pattern(a, graphp, m1));
}

TEST(PatternMatcherV2, GraphAutoAllowSideOutput) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    auto op0 = graphp->append_op(Convolution, "pconv");
    auto op1 = graphp->append_op(BiasAdd, {in_edge(IN0, op0, OUT0)}, "pbias");
    MUTE(op1);

    graph_t gr;
    op_t *a = gr.create_op(Convolution, "conv");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(BiasAdd, "bias");
    b->fill_and_connect_input(0, *a, 0);
    a->add_input(lt_vec[2]);
    op_t *c = gr.create_op(ReLU, "relu");
    c->fill_and_connect_input(0, *b, 0);
    c->add_output(lt_vec[3]);
    // create a side output from "conv" to "add"
    op_t *d = gr.create_op(Add, "add");
    d->fill_and_connect_input(0, *a, 0);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1, true));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(m1.inputs.size(), 3);
    ASSERT_EQ(m1.outputs.size(), 2);
}

TEST(PatternMatcherV2, GraphManualAllowSideOutput) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    auto op0 = graphp->append_op(Convolution, "pconv");
    op0->allow_external_output(0);
    auto op1 = graphp->append_op(BiasAdd, {in_edge(IN0, op0, OUT0)}, "pbias");
    MUTE(op1);

    graph_t gr;
    op_t *a = gr.create_op(Convolution, "conv");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(BiasAdd, "bias");
    b->fill_and_connect_input(0, *a, 0);
    a->add_input(lt_vec[2]);
    op_t *c = gr.create_op(ReLU, "relu");
    c->fill_and_connect_input(0, *b, 0);
    c->add_output(lt_vec[3]);
    // create a side output from "conv" to "add"
    op_t *d = gr.create_op(Add, "add");
    d->fill_and_connect_input(0, *a, 0);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(m1.inputs.size(), 3);
    ASSERT_EQ(m1.outputs.size(), 2);
}

TEST(PatternMatcherV2, ConvAddFusion) {
    // conv + add fusion
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    shared_ptr<pb_graph_t> pattern_graph = make_shared<pb_graph_t>("pgraph");

    auto conv = pattern_graph->append_op(Convolution, "pconv");
    auto add = pattern_graph->append_op(
            Add, {in_edge(IN0, conv, OUT0), in_edge(IN1, conv, OUT0)}, "padd");
    MUTE(add);

    graph_t gr;
    op_t *c = gr.create_op(Convolution, "conv");
    c->add_input(lt_vec[0]);
    c->add_input(lt_vec[1]);
    op_t *a = gr.create_op(Add, "add");
    a->fill_and_connect_input(0, *c, 0);
    a->fill_and_connect_input(1, *c, 0);
    a->add_output(lt_vec[2]);

    match_t a_matcher;
    EXPECT_TRUE(match_pattern(c, pattern_graph, a_matcher));
    ASSERT_EQ(a_matcher.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(a_matcher.inputs.size(), 2);
    ASSERT_EQ(a_matcher.outputs.size(), 1);
}

TEST(PatternMatcherV2, ConvAddFusionCase2) {
    // (Jihui)  conv + add fusion
    // when dnnl backend want to use this pattern, it always has such
    // requirement: one of add's inputs comes from conv, and the other one can
    // from anywhere(both internal and external), but we want to ensure that
    // only add can use conv's output for match_pattern, the default mode just
    // allow unhandled inputs come from external and another mode will enable
    // external outputs which don't meet our expectation
    // I think there are two options to handle such case:
    // the first one is add a function like allow_internal_inputs()
    // the second one is allow internal inputs by default
    // I'm not sure if there has any case which need inputs must come from
    // external?
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    shared_ptr<pb_graph_t> pattern_graph = make_shared<pb_graph_t>();

    auto conv = pattern_graph->append_op(Convolution, "pconv");
    auto add
            = pattern_graph->append_op(Add, {in_edge(IN0, conv, OUT0)}, "padd");
    add->allow_internal_input(IN1);
    add->set_commutative_pair({IN0, IN1});

    graph_t gr;
    op_t *c0 = gr.create_op(Convolution, "conv0");
    c0->add_input(lt_vec[0]);
    c0->add_input(lt_vec[1]);
    op_t *c1 = gr.create_op(Convolution, "conv1");
    op_t *a0 = gr.create_op(Add, "add0");
    c1->fill_and_connect_input(0, *c0, 0);
    c1->add_input(lt_vec[2]);
    a0->fill_and_connect_input(0, *c1, 0);
    a0->fill_and_connect_input(1, *c0, 0);
    a0->add_output(lt_vec[3]);

    match_t a_matcher;
    // we want matcher can match c1 and can't match c0, because
    // c0's output is also used by c1
    EXPECT_FALSE(match_pattern(c0, pattern_graph, a_matcher));
    EXPECT_TRUE(match_pattern(c1, pattern_graph, a_matcher));

    graph_t gr1;
    op_t *c2 = gr.create_op(Convolution, "conv2");
    c2->add_input(lt_vec[4]);
    c2->add_input(lt_vec[5]);
    op_t *a1 = gr.create_op(Add, "add1");
    a1->fill_and_connect_input(0, *c2, 0);
    a1->fill_and_connect_input(1, *c2, 0);

    match_t b_matcher;
    // and we want this pattern can also match c2
    EXPECT_TRUE(match_pattern(c2, pattern_graph, b_matcher));
}

TEST(PatternMatcherV2, NoAllowUnmatchedEdgeFromInternal) {
    // conv + add fusion
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(3);
    shared_ptr<pb_graph_t> pattern_graph = make_shared<pb_graph_t>("pgraph");

    auto conv = pattern_graph->append_op(Convolution, "pconv");
    auto add
            = pattern_graph->append_op(Add, {in_edge(IN0, conv, OUT0)}, "padd");
    MUTE(add);

    graph_t gr;
    op_t *c = gr.create_op(Convolution, "conv");
    c->add_input(lt_vec[0]);
    c->add_input(lt_vec[1]);
    op_t *a = gr.create_op(Add, "add");
    a->fill_and_connect_input(0, *c, 0);
    a->fill_and_connect_input(1, *c, 0);
    a->add_output(lt_vec[2]);

    match_t a_matcher;
    EXPECT_FALSE(match_pattern(c, pattern_graph, a_matcher));
}

TEST(PatternMatcherV2, AutoAllowUnmatchedEdgeFromInternal) {
    // conv + add fusion
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(3);
    shared_ptr<pb_graph_t> pattern_graph = make_shared<pb_graph_t>("pgraph");

    auto conv = pattern_graph->append_op(Convolution, "pconv");
    auto add
            = pattern_graph->append_op(Add, {in_edge(IN0, conv, OUT0)}, "padd");
    MUTE(add);

    graph_t gr;
    op_t *c = gr.create_op(Convolution, "conv");
    c->add_input(lt_vec[0]);
    c->add_input(lt_vec[1]);
    op_t *a = gr.create_op(Add, "add");
    a->fill_and_connect_input(0, *c, 0);
    a->fill_and_connect_input(1, *c, 0);
    a->add_output(lt_vec[2]);

    match_t a_matcher;
    EXPECT_TRUE(match_pattern(c, pattern_graph, a_matcher, true));
    ASSERT_EQ(a_matcher.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(a_matcher.inputs.size(), 2);
    ASSERT_EQ(a_matcher.outputs.size(), 1);
}

TEST(PatternMatcherV2, CommutativeInputBothConstrained) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    shared_ptr<pb_graph_t> pattern_graph = make_shared<pb_graph_t>("pgraph");

    auto conv = pattern_graph->append_op(Convolution, "pconv");
    auto elu
            = pattern_graph->append_op(Elu, {in_edge(IN0, conv, OUT0)}, "pelu");
    auto absnode
            = pattern_graph->append_op(Abs, {in_edge(IN0, conv, OUT0)}, "pabs");
    auto add = pattern_graph->append_op(Add,
            {in_edge(IN0, elu, OUT0), in_edge(IN1, absnode, OUT0)}, "padd");
    add->set_commutative_pair({IN0, IN1});

    graph_t gr1;
    op_t *c1 = gr1.create_op(Convolution, "conv");
    c1->add_input(lt_vec[0]);
    c1->add_input(lt_vec[1]);
    op_t *e1 = gr1.create_op(Elu, "elu");
    e1->fill_and_connect_input(0, *c1, 0);
    op_t *a1 = gr1.create_op(Abs, "abs");
    a1->fill_and_connect_input(0, *c1, 0);
    op_t *s1 = gr1.create_op(Add, "add");
    // set the commutative inputs in pattern order
    s1->fill_and_connect_input(0, *e1, 0);
    s1->fill_and_connect_input(1, *a1, 0);
    s1->add_output(lt_vec[2]);

    match_t m1;
    EXPECT_TRUE(match_pattern(c1, pattern_graph, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 4);
    ASSERT_EQ(m1.inputs.size(), 2);
    ASSERT_EQ(m1.outputs.size(), 1);

    graph_t gr2;
    op_t *c2 = gr2.create_op(Convolution, "conv");
    c2->add_input(lt_vec[3]);
    c2->add_input(lt_vec[4]);
    op_t *e2 = gr2.create_op(Elu, "elu");
    e2->fill_and_connect_input(0, *c2, 0);
    op_t *a2 = gr2.create_op(Abs, "abs");
    a2->fill_and_connect_input(0, *c2, 0);
    op_t *s2 = gr2.create_op(Add, "add");
    // set the commutative inputs in reverse pattern order
    s2->fill_and_connect_input(1, *e2, 0);
    s2->fill_and_connect_input(0, *a2, 0);
    s2->add_output(lt_vec[5]);

    match_t m2;
    EXPECT_TRUE(match_pattern(c2, pattern_graph, m2));
    ASSERT_EQ(m2.op_pb_op_pairs.size(), 4);
    ASSERT_EQ(m2.inputs.size(), 2);
    ASSERT_EQ(m2.outputs.size(), 1);
}

TEST(PatternMatcherV2, CommutativeInput) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    shared_ptr<pb_graph_t> pattern_graph = make_shared<pb_graph_t>("pgraph");
    auto conv0 = pattern_graph->append_op(Convolution, "pconv0");
    auto conv1 = pattern_graph->append_op(Convolution, "pconv1");
    auto relu0 = pattern_graph->append_op(
            ReLU, {in_edge(IN0, conv0, OUT0)}, "prelu0");
    relu0->append_decision_function([](op_t *o) -> bool {
        if (o->num_inputs() == 0) { return false; }
        auto v = o->get_input_value(0);
        if (v == nullptr) { return false; }
        op_t &c = v->get_producer();
        return c.has_attr("with_bias") ? c.get_attr<bool>("with_bias") : false;
    });
    auto relu1 = pattern_graph->append_op(
            ReLU, {in_edge(IN0, conv1, OUT0)}, "prelu1");
    auto add = pattern_graph->append_op(Add,
            {in_edge(IN0, relu0, OUT0), in_edge(IN1, relu1, OUT0)}, "padd");
    add->set_commutative_pair({IN0, IN1});

    graph_t gr;
    op_t *c0 = gr.create_op(Convolution, "conv0");
    c0->add_input(lt_vec[0]);
    c0->add_input(lt_vec[1]);
    op_t *c1 = gr.create_op(Convolution, "conv1");
    c1->add_input(lt_vec[2]);
    c1->add_input(lt_vec[3]);
    c1->add_input(lt_vec[4]);
    c1->set_attr<bool>("with_bias", true);
    op_t *r0 = gr.create_op(ReLU, "relu0");
    op_t *r1 = gr.create_op(ReLU, "relu1");
    op_t *a = gr.create_op(Add, "add");
    r0->fill_and_connect_input(0, *c0, 0);
    r1->fill_and_connect_input(0, *c1, 0);
    a->fill_and_connect_input(0, *r0, 0);
    a->fill_and_connect_input(1, *r1, 0);
    a->add_output(lt_vec[5]);

    match_t a_matcher;
    EXPECT_TRUE(match_pattern(a, pattern_graph, a_matcher, false, false));
    ASSERT_EQ(a_matcher.op_pb_op_pairs.size(), 5);
}

//
// Convolution + BiasAdd + Elu
// Convolution + BiasAdd + Sigmoid
// Convolution + BiasAdd + ReLU
// Convolution + BiasAdd + HardTanh
// Convolution + BiasAdd + Square
// Convolution + BiasAdd + Tanh
// Convolution + BiasAdd + Sqrt
//
TEST(PatternMatcherV2, ConvBiasActivationFusion) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    auto conv = graphp->append_op(Convolution, "pconv");
    auto bias = graphp->append_op(BiasAdd, {in_edge(IN0, conv, OUT0)}, "pbias");
    auto act = graphp->append_alternation(
            {Elu, Sigmoid, ReLU, HardTanh, Square, Tanh, Sqrt},
            {in_edge(IN0, bias, OUT0)}, "pactivation");
    MUTE(act);

    graph_t gr;
    op_t *a = gr.create_op(Convolution, "conv");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(BiasAdd, "bias");
    b->fill_and_connect_input(0, *a, 0);
    b->add_input(lt_vec[2]);
    op_t *c = gr.create_op(ReLU, "relu");
    c->fill_and_connect_input(0, *b, 0);
    c->add_output(lt_vec[3]);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 3);
    ASSERT_EQ(m1.inputs.size(), 3);
    ASSERT_EQ(m1.outputs.size(), 1);
    match_t m2;
    EXPECT_TRUE(match_pattern(c, graphp, m2, false, false));
    ASSERT_EQ(m2.op_pb_op_pairs.size(), 3);
    ASSERT_EQ(m2.inputs.size(), 3);
    ASSERT_EQ(m2.outputs.size(), 1);
    match_t m3;
    EXPECT_FALSE(match_pattern(a, graphp, m3, false, false));
}

//
// Convolution + BiasAdd + Add + ReLU
// Convolution + BiasAdd + Add + ELU
//
TEST(PatternMatcherV2, ConvBiasSumActivationFusion) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(7);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    auto conv = graphp->append_op(Convolution, "pconv");
    auto bias = graphp->append_op(BiasAdd, {in_edge(IN0, conv, OUT0)}, "pbias");
    auto add = graphp->append_op(Add, {in_edge(IN0, bias, OUT0)}, "padd");
    add->set_commutative_pair({IN0, IN1});
    auto act = graphp->append_alternation(
            {Elu, ReLU}, {in_edge(IN0, add, OUT0)}, "pactivation");
    MUTE(act);

    graph_t gr;
    op_t *a = gr.create_op(Convolution, "conv");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(BiasAdd, "bias");
    b->fill_and_connect_input(0, *a, 0);
    b->add_input(lt_vec[2]);
    op_t *c = gr.create_op(Add, "add1");
    op_t *e = gr.create_op(Add, "add2");
    e->add_input(lt_vec[3]);
    e->add_input(lt_vec[4]);
    e->add_output(lt_vec[5]);
    // Force check commutative input
    c->fill_and_connect_input(0, *e, 0);
    c->fill_and_connect_input(1, *b, 0);
    op_t *d = gr.create_op(Elu, "elu");
    d->fill_and_connect_input(0, *c, 0);
    d->add_output(lt_vec[6]);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 4);
    ASSERT_EQ(m1.inputs.size(), 4);
    ASSERT_EQ(m1.outputs.size(), 1);
    match_t m2;
    EXPECT_TRUE(match_pattern(d, graphp, m2, false, false));
    ASSERT_EQ(m2.op_pb_op_pairs.size(), 4);
    ASSERT_EQ(m2.inputs.size(), 4);
    ASSERT_EQ(m2.outputs.size(), 1);
}

//
// MatMul + BiasAdd + Add
//
TEST(PatternMatcherV2, MatmulBiasSumFusion) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    auto matmul = graphp->append_op(MatMul, "pmatmul");
    auto bias
            = graphp->append_op(BiasAdd, {in_edge(IN0, matmul, OUT0)}, "pbias");
    auto add = graphp->append_op(Add, {in_edge(IN0, bias, OUT0)}, "padd");
    add->set_commutative_pair({IN0, IN1});

    graph_t gr;
    op_t *a = gr.create_op(MatMul, "matmul");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(BiasAdd, "bias");
    b->fill_and_connect_input(0, *a, 0);
    b->add_input(lt_vec[2]);
    op_t *c = gr.create_op(Add, "add");
    c->fill_and_connect_input(0, *b, 0);
    c->add_input(lt_vec[3]);
    op_t *d = gr.create_op(Elu, "elu");
    d->fill_and_connect_input(0, *c, 0);
    d->add_output(lt_vec[4]);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 3);
    ASSERT_EQ(m1.inputs.size(), 4);
    ASSERT_EQ(m1.outputs.size(), 1);
    match_t m2;
    EXPECT_FALSE(match_pattern(d, graphp, m2, false, false));
    match_t m3;
    EXPECT_TRUE(match_pattern(c, graphp, m3, false, false));
    ASSERT_EQ(m3.op_pb_op_pairs.size(), 3);
    ASSERT_EQ(m3.inputs.size(), 4);
    ASSERT_EQ(m3.outputs.size(), 1);
}

//
// MatMul + ReLU
// MatMul + Elu
// MatMul + GELU
// MatMul + Sigmoid
// MatMul + HardTanh
//
TEST(PatternMatcherV2, MatmulActivationFusion) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    auto mat = graphp->append_op(MatMul, "pmatmul");
    auto act = graphp->append_alternation(
            {ReLU, Elu, GELU, Sigmoid, HardTanh}, {in_edge(IN0, mat, OUT0)});
    MUTE(act);

    graph_t gr;
    op_t *a = gr.create_op(MatMul, "matmul");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(Sigmoid, "sigmoid");
    b->fill_and_connect_input(0, *a, 0);
    op_t *c = gr.create_op(Add, "add");
    c->fill_and_connect_input(0, *b, 0);
    c->add_input(lt_vec[2]);
    c->add_output(lt_vec[3]);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(m1.inputs.size(), 2);
    ASSERT_EQ(m1.outputs.size(), 1);
    match_t m2;
    EXPECT_FALSE(match_pattern(c, graphp, m2));
    match_t m3;
    EXPECT_TRUE(match_pattern(b, graphp, m3, false, false));
    ASSERT_EQ(m3.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(m3.inputs.size(), 2);
    ASSERT_EQ(m3.outputs.size(), 1);
}

TEST(PatternMatcherV2, ConvSwishFusion) {
    // conv_swish pass
    //   conv
    //   |   |
    //   | sigmoid
    //   |   |
    // multiply

    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(3);
    shared_ptr<pb_graph_t> pattern_graph = make_shared<pb_graph_t>("pgraph");
    auto conv = pattern_graph->append_op(Convolution, "pconv");
    auto sigmoid = pattern_graph->append_op(
            Sigmoid, {in_edge(IN0, conv, OUT0)}, "psigmoid");
    in_edges_t mul_edges
            = {in_edge(IN0, conv, OUT0), in_edge(IN1, sigmoid, OUT0)};
    auto mul = pattern_graph->append_op(Multiply, mul_edges, "pmul");
    mul->set_commutative_pair({IN0, IN1});

    graph_t gr;
    op_t *c = gr.create_op(Convolution, "conv");
    c->add_input(lt_vec[0]);
    c->add_input(lt_vec[1]);
    op_t *s = gr.create_op(Sigmoid, "sigmoid");
    s->fill_and_connect_input(0, *c, 0);
    op_t *m = gr.create_op(Multiply, "mul");
    m->fill_and_connect_input(0, *c, 0);
    m->fill_and_connect_input(1, *s, 0);
    m->add_output(lt_vec[2]);

    match_t a_matcher;
    EXPECT_TRUE(match_pattern(c, pattern_graph, a_matcher));
    ASSERT_EQ(a_matcher.op_pb_op_pairs.size(), 3);
    ASSERT_EQ(a_matcher.inputs.size(), 2);
    ASSERT_EQ(a_matcher.outputs.size(), 1);
    match_t b_matcher;
    EXPECT_TRUE(match_pattern(m, pattern_graph, b_matcher, false, false));
    ASSERT_EQ(b_matcher.op_pb_op_pairs.size(), 3);
    ASSERT_EQ(b_matcher.inputs.size(), 2);
    ASSERT_EQ(b_matcher.outputs.size(), 1);
}

TEST(PatternMatcherV2, ConvSumEltwiseFusion) {
    // conv + sum + (Relu / Elu / HardTanh / Square / Tanh / Abs / Sqrt)
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    shared_ptr<pb_graph_t> pattern_graph = make_shared<pb_graph_t>("pgraph");
    auto conv = pattern_graph->append_op(Convolution, "pconv");
    auto add
            = pattern_graph->append_op(Add, {in_edge(IN0, conv, OUT0)}, "padd");
    add->set_commutative_pair({IN0, IN1});

    shared_ptr<pb_graph_t> optional_act
            = make_shared<pb_graph_t>("poptionalbody");
    auto act = optional_act->append_alternation(
            {Elu, ReLU, Square, Tanh, Abs, Sqrt, HardTanh}, "pactivation");
    optional_act->create_input_port(IN0, act, IN0);
    optional_act->create_output_port(OUT0, act, OUT0);
    pattern_graph->append_optional(
            optional_act, {in_edge(IN0, add, OUT0)}, "poptional");

    graph_t gr;
    op_t *c = gr.create_op(Convolution, "conv");
    c->add_input(lt_vec[0]);
    c->add_input(lt_vec[1]);
    op_t *m = gr.create_op(MatMul, "matmul");
    m->add_input(lt_vec[2]);
    m->add_input(lt_vec[3]);
    m->add_output(lt_vec[4]);
    op_t *a = gr.create_op(Add, "add");
    a->fill_and_connect_input(0, *c, 0);
    a->fill_and_connect_input(1, *m, 0);
    a->add_output(lt_vec[5]);

    match_t a_matcher;
    EXPECT_TRUE(match_pattern(c, pattern_graph, a_matcher));
    ASSERT_EQ(a_matcher.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(a_matcher.inputs.size(), 3);
    ASSERT_EQ(a_matcher.outputs.size(), 1);
}

//
// Alternation, Repetition, Optional are nested pattern nodes
// that has a body(s) of graph.
// Input and Output ports of those nested patterns get mapped to
// the corresponding port (same index) of the body.
// If you need to change that mapping, wrap the body in a graph
// and use create_input_port/create_output_port to change the
// mapping.
//

//
// Alternation node wraps two or more alternatives and
// constructed with append_alternation.
// Input or Output "n" of the alternation node connects to
// Input of Output "n" of the alternative.
//
TEST(PatternMatcherV2, Alternation) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(4);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    // MatMul -> (Add | Multiply)
    auto matmul = graphp->append_op(MatMul, "pmatmul");

    // Prepare the alternative graphs
    auto addgraph = make_shared<pb_graph_t>("paddgraph");
    auto add = addgraph->append_op(Add, "padd");
    add->set_commutative_pair({IN0, IN1});
    addgraph->create_input_port(IN0, add, IN0);
    addgraph->create_input_port(IN1, add, IN1);
    addgraph->create_output_port(OUT0, add, OUT0);
    auto mulgraph = make_shared<pb_graph_t>("pmulgraph");
    auto mul = mulgraph->append_op(Multiply, "pmul");
    mul->set_commutative_pair({IN0, IN1});
    mulgraph->create_input_port(IN0, mul, IN0);
    mulgraph->create_input_port(IN1, mul, IN1);
    mulgraph->create_output_port(OUT0, mul, OUT0);
    // We can add a helper function like
    // single_op_graph(op_kind);
    // that create a new graph add a single node and sets
    // inner consumer and producers.

    auto alt = graphp->append_alternation(
            {addgraph, mulgraph}, {in_edge(IN0, matmul, OUT0)}, "palternation");
    MUTE(alt);

    graph_t gr;
    op_t *a = gr.create_op(MatMul, "matmul");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(Add, "add");
    b->fill_and_connect_input(0, *a, 0);
    b->add_input(lt_vec[2]);
    op_t *c = gr.create_op(ReLU, "relu");
    c->fill_and_connect_input(0, *b, 0);
    c->add_output(lt_vec[3]);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(m1.inputs.size(), 3);
    ASSERT_EQ(m1.outputs.size(), 1);
    match_t m2;
    EXPECT_FALSE(match_pattern(c, graphp, m2, false, false));
    match_t m3;
    EXPECT_TRUE(match_pattern(b, graphp, m3, false, false));
    ASSERT_EQ(m3.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(m3.inputs.size(), 3);
    ASSERT_EQ(m3.outputs.size(), 1);
}

//
// Repetition node wraps body that gets repeated a
// number of times specified by a range and constructed with
// append_repetition.
// The body repeats inself by connecting edges through an
// output port to input port mapping.
// The mapping has to be given as an argument to append_repetition.
//
TEST(PatternMatcherV2, Repetition) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(5);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    // Pattern that captures
    // MatMul -> (Add | Multiply) -> ReLU
    // MatMul -> (Add | Multiply) -> (Add | Multiply) -> ReLU
    auto matmul = graphp->append_op(MatMul, "pmatmul");
    auto repbody = make_shared<pb_graph_t>("prepetitionbody");
    auto addormul = repbody->append_alternation({Add, Multiply}, "paddormul");
    addormul->set_commutative_pair({IN0, IN1});
    repbody->create_input_port(IN0, addormul, IN0);
    // No need to create IN1 for the body since it is not connected to
    // an outer pattern.
    // repbody->create_input_port(IN1, addormul, IN1);
    repbody->create_output_port(OUT0, addormul, OUT0);

    // Repeat 1 or 2 times [1, 3) by mapping OUT0 back to IN0
    auto rep = graphp->append_repetition(repbody, {{OUT0, IN0}}, 1, 3,
            {in_edge(IN0, matmul, OUT0)}, "prepetition");
    auto relu = graphp->append_op(ReLU, {in_edge(IN0, rep, OUT0)}, "prelu");
    MUTE(relu);

    graph_t gr;
    op_t *a = gr.create_op(MatMul, "matmul");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(Add, "add");
    b->fill_and_connect_input(0, *a, 0);
    b->add_input(lt_vec[2]);
    op_t *c = gr.create_op(Multiply, "mul");
    c->fill_and_connect_input(0, *b, 0);
    c->add_input(lt_vec[3]);
    op_t *d = gr.create_op(ReLU, "relu");
    d->fill_and_connect_input(0, *c, 0);
    c->add_output(lt_vec[4]);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 4);
    ASSERT_EQ(m1.inputs.size(), 4);
    ASSERT_EQ(m1.outputs.size(), 1);
}

//
// "Optional" is a special case of repetition that repeats one or zero times
// and constructed with append_optional.
// output to input port mapping isn't needed since the body does not repeat
// more than once.
//
TEST(PatternMatcherV2, Optional) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(6);
    auto graphp = make_shared<pb_graph_t>("pgraph");
    // Pattern that captures
    // MatMul -> ReLU
    // MatMul -> (Add | Multiply) -> ReLU
    auto matmul = graphp->append_op(MatMul, "pmatmul");
    auto repbody = make_shared<pb_graph_t>("poptionalbody");
    auto addormul = repbody->append_alternation({Add, Multiply}, "paddormul");
    addormul->set_commutative_pair({IN0, IN1});
    repbody->create_input_port(IN0, addormul, IN0);
    repbody->create_output_port(OUT0, addormul, OUT0);
    auto rep = graphp->append_optional(
            repbody, {in_edge(IN0, matmul, OUT0)}, "poptional");
    auto relu = graphp->append_op(ReLU, {in_edge(IN0, rep, OUT0)}, "prelu");
    MUTE(relu);

    graph_t gr;
    op_t *a = gr.create_op(MatMul, "matmul");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(ReLU, "relu");
    b->fill_and_connect_input(0, *a, 0);
    b->add_output(lt_vec[2]);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 2);
    ASSERT_EQ(m1.inputs.size(), 2);
    ASSERT_EQ(m1.outputs.size(), 1);

    graph_t gr2;
    op_t *a2 = gr2.create_op(MatMul, "matmul");
    a2->add_input(lt_vec[3]);
    a2->add_input(lt_vec[4]);
    op_t *b2 = gr2.create_op(Add, "add");
    b2->fill_and_connect_input(0, *a2, 0);
    b2->add_input(zero_logical_tensor());
    op_t *c2 = gr2.create_op(ReLU, "relu");
    c2->fill_and_connect_input(0, *b2, 0);
    c2->add_output(lt_vec[5]);

    match_t m2;
    EXPECT_TRUE(match_pattern(a2, graphp, m2));
    ASSERT_EQ(m2.op_pb_op_pairs.size(), 3);
    ASSERT_EQ(m2.inputs.size(), 3);
    ASSERT_EQ(m2.outputs.size(), 1);

    match_t m3;
    EXPECT_TRUE(match_pattern(c2, graphp, m3, false, false));
    ASSERT_EQ(m3.op_pb_op_pairs.size(), 3);
    ASSERT_EQ(m3.inputs.size(), 3);
    ASSERT_EQ(m3.outputs.size(), 1);
}

//
// ?: means optional
// ^: means repetition
// Conv+(BN)?+ReLU
// Conv+(BN)?+ReLU+Add
// Conv+(BN)?+ReLU+Conv+(BN)?+ReLU+Conv+(BN)?+ReLU+Add
//
// Conv+(BN)?+ReLU+(((Conv+(BN)?+ReLU)^2)?+Add)?
//
// Note that each "()" requires an addition pb_graph.
// So for this example, we need 1 + 5 = 6 pb_graphs.
//
// Since this example is not a fixed pattern and has
// variable number of side inputs, we cannot use
// create_input_port to setup globbal ordering for inputs.
//
// create_input_port/create_output_port is still needed for
// setting up the contact interface for nested patterns.
//
TEST(PatternMatcherV2, ComplexRepetition) {
    std::vector<logical_tensor_t> lt_vec = create_logical_tensors(20);
    auto graphp = make_shared<pb_graph_t>("pmaingraph");
    // Basic building block
    // Convolution + (BatchNormInference)? + ReLU

    // Conv
    auto conv = graphp->append_op(Convolution, "pconv1");
    // Optional BN
    auto body = make_shared<pb_graph_t>("poptional1body");
    auto bn = body->append_op(BatchNormInference, "pbn1");
    // Interface for body
    body->create_input_port(IN0, bn, IN0);
    body->create_output_port(OUT0, bn, OUT0);
    auto opt = graphp->append_optional(
            body, {in_edge(IN0, conv, OUT0)}, "poptional1");
    // ReLU
    auto relu = graphp->append_op(ReLU, {in_edge(IN0, opt, OUT0)}, "prelu1");
    // Create same block to use as repetition body
    auto graphp2 = make_shared<pb_graph_t>("prepetitionbody");
    auto conv2 = graphp2->append_op(Convolution, "pconv2");
    auto body2 = make_shared<pb_graph_t>("poptional2body");
    auto bn2 = body2->append_op(BatchNormInference, "pbn2");
    // Interface for body2
    body2->create_input_port(IN0, bn2, IN0);
    body2->create_output_port(OUT0, bn2, OUT0);
    auto opt2 = graphp2->append_optional(
            body2, {in_edge(IN0, conv2, OUT0)}, "poptional2");
    auto relu2 = graphp2->append_op(ReLU, {in_edge(IN0, opt2, OUT0)}, "prelu2");
    // Interface for graphp2
    graphp2->create_input_port(IN0, conv2, IN0);
    graphp2->create_output_port(OUT0, relu2, OUT0);

    // repeat body exactly two times
    auto graphp3 = make_shared<pb_graph_t>("poptional3");
    auto rep = graphp3->append_repetition(
            graphp2, {{OUT0, IN0}}, 2, 3, "prepetition");
    // Interface for graphp3
    graphp3->create_input_port(IN0, rep, IN0);
    graphp3->create_output_port(OUT0, rep, OUT0);

    // optional repeated body followed by an "Add"
    auto graphp4 = make_shared<pb_graph_t>("poptional4body");
    auto opt3 = graphp4->append_optional(graphp3, "poptional3");
    auto add = graphp4->append_op(Add, {in_edge(IN0, opt3, OUT0)}, "padd");
    add->set_commutative_pair({IN0, IN1});
    // Interface for graphp4
    graphp4->create_input_port(IN0, opt3, IN0);
    graphp4->create_output_port(OUT0, add, OUT0);

    // Append the complex pattern to relu
    auto opt4 = graphp->append_optional(
            graphp4, {in_edge(IN0, relu, OUT0)}, "poptional4");
    MUTE(opt4);

    graph_t gr;
    op_t *a = gr.create_op(Convolution, "conv1");
    a->add_input(lt_vec[0]);
    a->add_input(lt_vec[1]);
    op_t *b = gr.create_op(ReLU, "relu1");
    b->fill_and_connect_input(0, *a, 0);
    op_t *c = gr.create_op(Convolution, "conv2");
    c->fill_and_connect_input(0, *b, 0);
    c->add_input(lt_vec[2]);
    op_t *d = gr.create_op(ReLU, "relu2");
    d->fill_and_connect_input(0, *c, 0);
    op_t *e = gr.create_op(Convolution, "conv3");
    e->fill_and_connect_input(0, *d, 0);
    e->add_input(lt_vec[3]);
    op_t *f = gr.create_op(ReLU, "relu3");
    f->fill_and_connect_input(0, *e, 0);
    op_t *g = gr.create_op(Add, "add");
    g->fill_and_connect_input(0, *f, 0);
    g->add_input(lt_vec[4]);
    g->add_output(lt_vec[5]);

    match_t m1;
    EXPECT_TRUE(match_pattern(a, graphp, m1));
    ASSERT_EQ(m1.op_pb_op_pairs.size(), 7);
    ASSERT_EQ(m1.inputs.size(), 5);
    ASSERT_EQ(m1.outputs.size(), 1);

    graph_t gr2;
    op_t *a2 = gr2.create_op(Convolution, "conv1");
    a2->add_input(lt_vec[6]);
    a2->add_input(lt_vec[7]);
    op_t *b2 = gr2.create_op(BatchNormInference, "bn1");
    b2->fill_and_connect_input(0, *a2, 0);
    b2->add_input(lt_vec[8]);
    b2->add_input(lt_vec[9]);
    b2->add_input(lt_vec[10]);
    b2->add_input(lt_vec[11]);
    b2->add_output(lt_vec[12]);
    match_t m3;
    EXPECT_FALSE(match_pattern(a2, graphp, m3));

    graph_t gr4;
    op_t *a4 = gr4.create_op(Convolution, "conv1");
    a4->add_input(lt_vec[16]);
    a4->add_input(lt_vec[17]);
    op_t *b4 = gr4.create_op(ReLU, "relu1");
    b4->fill_and_connect_input(0, *a4, 0);
    op_t *c4 = gr4.create_op(Add, "add");
    c4->fill_and_connect_input(0, *b4, 0);
    c4->add_input(lt_vec[18]);
    c4->add_output(lt_vec[19]);
    match_t m4;
    EXPECT_TRUE(match_pattern(a4, graphp, m4));
}

//
// Shared input can be expressed by using create_input_port()
// to forward a single graph input to multiple node inputs.
//
TEST(PatternMatcherV2, SharedInput) {
    auto graphp = make_shared<pb_graph_t>("pgraph");
    // Pattern that captures shared input to three MatMuls
    //        |--> MatMul
    //   Any ----> MatMul
    //        |--> MatMul
    auto mm1 = graphp->append_op(MatMul, "pmatmul1");
    auto mm2 = graphp->append_op(MatMul, "pmatmul2");
    auto mm3 = graphp->append_op(MatMul, "pmatmul3");
    // Map the shared input to inputs of the three MatMuls
    graphp->create_input_port(IN0, mm1, IN0);
    graphp->create_input_port(IN0, mm2, IN0);
    graphp->create_input_port(IN0, mm3, IN0);

    ASSERT_EQ(graphp->get_inner_consumer(IN0)->size(), 3);

    graph_t gr;
    op_t *a = gr.create_op(MatMul, "matmul1");
    a->add_input(zero_logical_tensor());
    a->add_input(zero_logical_tensor());
    op_t *b = gr.create_op(MatMul, "matmul2");
    b->add_input(zero_logical_tensor());
    b->add_input(zero_logical_tensor());
    op_t *c = gr.create_op(MatMul, "matmul3");
    c->add_input(zero_logical_tensor());
    c->add_input(zero_logical_tensor());
    op_t *d = gr.create_op(Concat, "concat");
    d->fill_and_connect_input(0, *a, 0);
    d->fill_and_connect_input(1, *b, 0);
    d->fill_and_connect_input(2, *c, 0);
    d->add_output(zero_logical_tensor());

    // Limitation: Matcher does not support matching this test case yet.
}

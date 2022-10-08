/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "gtest/gtest.h"

#include "cpp/unit/backend/dnnl/dnnl_test_common.hpp"
#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(Execute, Typecast) {
    impl::engine_t &engine = get_engine();

    test::vector<float> f32_val {
            12.5234537f, 0.f, -32.6735142f, -1.f, -2.8765223f, 66.66f};
    test::vector<uint16_t> bf16_val(f32_val.size(), 10);

    impl::op_t typecast_op(impl::op_kind::TypeCast);

    // prepare input/output logical tensor
    // shape (2, 3), stride (3, 1), plain format
    impl::logical_tensor_t f32_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    impl::logical_tensor_t bf16_lt = utils::logical_tensor_init(
            1, {2, 3}, {3, 1}, impl::data_type::bf16);

    typecast_op.add_input(f32_lt);
    typecast_op.add_output(bf16_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&typecast_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("typecast_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&f32_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&bf16_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine), impl::status::success);

    impl::stream_t &stream = get_stream();
    impl::tensor_t f32_ts(f32_lt, &engine, f32_val.data());
    impl::tensor_t bf16_ts(bf16_lt, &engine, bf16_val.data());

    // f32 --> bf16
    ASSERT_EQ(cp.execute(&stream, {f32_ts}, {bf16_ts}), impl::status::success);
    stream.wait();
}

TEST(Compile, TypecastNegativeInput) {
    impl::engine_t &engine = get_engine();

    impl::op_t typecast_op(impl::op_kind::TypeCast);
    impl::logical_tensor_t src_lt = utils::logical_tensor_init(
            0, {2, 3}, {3, 1}, impl::data_type::f32);
    // failure case : different shape (dims)
    impl::logical_tensor_t dst_lt = utils::logical_tensor_init(
            1, {1, 2, 3}, {1, 1, 2}, impl::data_type::bf16);
    typecast_op.add_input(src_lt);
    typecast_op.add_output(dst_lt);

    impl::graph_t g(engine.kind());
    g.add_op(&typecast_op);
    g.build_graph();

    impl::pass::pass_base_ptr apass = get_pass("typecast_pass");
    apass->run(g);
    ASSERT_EQ(g.get_num_partitions(), 1U);
    auto part = g.get_partitions()[0];

    // compile
    impl::partition_t p;
    p.init(part);
    impl::compiled_partition_t cp(p);

    std::vector<const impl::logical_tensor_t *> inputs {&src_lt};
    std::vector<const impl::logical_tensor_t *> outputs {&dst_lt};

    ASSERT_EQ(p.compile(&cp, inputs, outputs, &engine),
            impl::status::invalid_shape);
}

/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <array>
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "interface/op.hpp"

/**
 * 1. Create a dnnl::graph::op object
 * 2. Validate if dnnl::graph::op has expected contents
 * 3. Validate if the backend object llga::implementation::op is consistent with dnnl::graph::op
 */
TEST(op_test, create_simple) {
    struct param {
        size_t id;
        dnnl::graph::logical_tensor::data_type dtype;
        int ndims;
        dnnl::graph::logical_tensor::layout_type ltype;
    };
    std::array<int64_t, 5> proto_dims = {1, 2, 3, 4, 5};
    std::vector<param> proto_inputs {
            {0, dnnl::graph::logical_tensor::data_type::s8, 1,
                    dnnl::graph::logical_tensor::layout_type::undef},
            {2, dnnl::graph::logical_tensor::data_type::f32, 2,
                    dnnl::graph::logical_tensor::layout_type::undef}};
    std::vector<param> proto_outputs {
            {3, dnnl::graph::logical_tensor::data_type::s8, 3,
                    dnnl::graph::logical_tensor::layout_type::undef},
            {4, dnnl::graph::logical_tensor::data_type::f32, 4,
                    dnnl::graph::logical_tensor::layout_type::undef},
            {5, dnnl::graph::logical_tensor::data_type::undef, 5,
                    dnnl::graph::logical_tensor::layout_type::undef}};

    dnnl::graph::op bridge_op(0, dnnl::graph::op::kind::Add, "kAdd");

    std::vector<std::shared_ptr<dnnl::graph::logical_tensor>> inputs;
    for (auto &proto : proto_inputs) {
        inputs.emplace_back(std::make_shared<dnnl::graph::logical_tensor>(
                proto.id, proto.dtype,
                std::vector<int64_t> {begin(proto_dims),
                        std::next(begin(proto_dims), proto.ndims)},
                proto.ltype));
        bridge_op.add_input(*inputs.back());
    }

    std::vector<std::shared_ptr<dnnl::graph::logical_tensor>> outputs;
    for (auto &proto : proto_outputs) {
        outputs.emplace_back(std::make_shared<dnnl::graph::logical_tensor>(
                proto.id, proto.dtype,
                std::vector<int64_t> {begin(proto_dims),
                        std::next(begin(proto_dims), proto.ndims)},
                proto.ltype));
        bridge_op.add_output(*outputs.back());
    }

    // Validate internal representation
    auto backend_op = bridge_op.get();
    ASSERT_EQ(backend_op->kind(), kAdd);
    ASSERT_STREQ(backend_op->debug().c_str(), "kAdd");

    auto compare2 = [&](const std::vector<llga::impl::logical_tensor_t> &actual,
                            const std::vector<param> &expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            ASSERT_EQ(actual[i].ndims, expected[i].ndims);
            for (size_t j = 0; j < actual[i].ndims; j++) {
                ASSERT_EQ(actual[i].dims[j], proto_dims[j]);
            }
        }
    };

    compare2(backend_op->inputs(), proto_inputs);
    compare2(backend_op->outputs(), proto_outputs);
}

/**
 * 1. Create a dnnl::graph::op object
 * 2. Attempt to add input with dims = -1
 * 3. Validate backend's structure integrity - expect ndims==-1 (undefined)
 */
TEST(op_test, input_dims_n1) {
    dnnl::graph::op bridge_op(0, dnnl::graph::op::kind::Add, "kAdd");
    dnnl::graph::logical_tensor input {0,
            dnnl::graph::logical_tensor::data_type::undef,
            dnnl::graph::logical_tensor::layout_type::undef};
    bridge_op.add_input(input);

    auto backend_op = bridge_op.get();
    ASSERT_FALSE(backend_op->inputs().empty());
    ASSERT_EQ(backend_op->inputs().front().ndims, -1);
}

/**
 * 1. Create a dnnl::graph::op object
 * 2. Attempt to add input with dims = 0
 * 3. Validate backend's structure integrity - expect ndims != -1 (undefined)
 */
TEST(op_test, input_dims_0) {
    dnnl::graph::op bridge_op(0, dnnl::graph::op::kind::Add, "kAdd");
    dnnl::graph::logical_tensor input(0,
            dnnl::graph::logical_tensor::data_type::undef, 0,
            dnnl::graph::logical_tensor::layout_type::undef);
    bridge_op.add_input(input);

    auto backend_op = bridge_op.get();
    ASSERT_FALSE(backend_op->inputs().empty());
    ASSERT_NE(backend_op->inputs().front().ndims, -1);
}

TEST(op_test, add_inputs_outputs) {
    dnnl::graph::op add_op {
            0, dnnl::graph::op::kind::Convolution, std::string("convolution")};

    dnnl::graph::logical_tensor lt_1 {0,
            dnnl::graph::logical_tensor::data_type::f32, 0,
            dnnl::graph::logical_tensor::layout_type::any};
    dnnl::graph::logical_tensor lt_2 {1,
            dnnl::graph::logical_tensor::data_type::f32, 1,
            dnnl::graph::logical_tensor::layout_type::any};

    add_op.add_input(lt_1);
    add_op.add_output(lt_2);

    auto backend_op = add_op.get();
    ASSERT_EQ(backend_op->inputs().back().data_type, dnnl_graph_f32);
    ASSERT_EQ(backend_op->inputs().size(), 1);
    ASSERT_EQ(backend_op->inputs().back().id, 0);
    ASSERT_EQ(backend_op->outputs().back().id, 1);
}

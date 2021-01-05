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

#include <gtest/gtest.h>

#include "interface/ir.hpp"
#include "interface/partition.hpp"

TEST(partition_test, create_simple) {
    dnnl::graph::impl::partition p;
    ASSERT_EQ(p.num_ops(), 0);
}

TEST(partition_test, add_ops) {
    dnnl::graph::impl::partition p;
    size_t id = 100;
    p.add_op(id);
    ASSERT_EQ(p.num_ops(), 1);

    std::vector<size_t> ids {101, 102};
    p.add_op(ids);
    ASSERT_EQ(p.num_ops(), 3);
}

TEST(partition_test, get_ops) {
    dnnl::graph::impl::partition p;
    size_t id = 100;
    p.add_op(id);
    auto ops = p.get_ops();
    ASSERT_EQ(ops.size(), 1);
    ASSERT_EQ(ops.count(id), 1);
}

TEST(partition_test, init) {
    // (todo)xinyu: improve engine test
    dnnl::graph::impl::engine_t eng {};
    dnnl::graph::impl::partition p;
    dnnl::graph::impl::node_t n(dnnl::graph::impl::op_kind::Convolution);
    n.set_attr<int>("groups", 0);
    p.init(&n, eng.kind());
    ASSERT_TRUE(p.is_initialized());
    ASSERT_EQ(p.node()->get_op_kind(), dnnl::graph::impl::op_kind::Convolution);
    ASSERT_TRUE(p.node()->has_attr("groups"));
    ASSERT_EQ(p.node()->get_attr<int>("groups"), 0);
}

TEST(partition_test, copy) {
    dnnl::graph::impl::engine_t eng {};
    dnnl::graph::impl::partition p;
    dnnl::graph::impl::node_t n(dnnl::graph::impl::op_kind::Convolution);
    n.set_attr<int>("groups", 0);
    p.init(&n, eng.kind());
    ASSERT_TRUE(p.is_initialized());
    ASSERT_EQ(p.node()->get_op_kind(), dnnl::graph::impl::op_kind::Convolution);
    ASSERT_TRUE(p.node()->has_attr("groups"));
    ASSERT_EQ(p.node()->get_attr<int>("groups"), 0);

    // copy the partition
    dnnl::graph::impl::partition p_copy(p);
    dnnl::graph::impl::node_t *p_node
            = const_cast<dnnl::graph::impl::node_t *>(p_copy.node());
    p_node->set_attr<int>("groups", 1);
    ASSERT_EQ(p_copy.node()->get_attr<int>("groups"), 1);
    ASSERT_NE(p_copy.node()->get_attr<int>("groups"),
            p.node()->get_attr<int>("groups"));
}

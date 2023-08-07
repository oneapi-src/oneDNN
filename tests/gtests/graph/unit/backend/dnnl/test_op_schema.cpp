/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "interface/graph.hpp"
#include "interface/op_schema.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/dnnl_shape_infer.hpp"
#include "backend/dnnl/internal_attrs.hpp"
#include "backend/dnnl/internal_ops.hpp"

#include "graph/unit/utils.hpp"

using namespace dnnl::impl::graph;
using namespace dnnl::graph::tests::unit::utils;

TEST(OpSchema, InferSqueezeOutputShape) {
    auto &be = graph::dnnl_impl::dnnl_backend::get_singleton();
    EXPECT_EQ(be.get_name(), "dnnl_backend");
    const op_kind_t kind = dnnl_impl::op_kind::dnnl_squeeze;
    const op_schema_t *op_schema_ = op_schema_registry_t::get_op_schema(kind);
    std::vector<std::vector<int64_t>> axes_list {{1}, {1, 2}, {-1}, {-1, -2}};
    std::vector<std::vector<int64_t>> src_shapes {
            {3, 1, 4, 5}, {3, 1, 1, 4, 5}, {3, 4, 5, 1}, {3, 4, 5, 1, 1}};
    std::vector<std::vector<int64_t>> dst_shapes {
            {3, 4, 5}, {3, 4, 5}, {3, 4, 5}, {3, 4, 5}};
    for (size_t i = 0; i < axes_list.size(); ++i) {
        op_t op {kind, "squeeze"};
        op.set_attr<std::vector<int64_t>>(op_attr::axes, axes_list[i]);

        logical_tensor_t lt_in = logical_tensor_init(
                0, src_shapes[i], data_type::f32, layout_type::strided);
        logical_tensor_t lt_out
                = logical_tensor_init(1, data_type::f32, layout_type::strided);

        std::vector<logical_tensor_t *> in {&lt_in};
        std::vector<logical_tensor_t *> out {&lt_out};

        status_t ret = op_schema_->shape_infer(&op, in, out);
        EXPECT_EQ(ret, status::success);

        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        const std::vector<int64_t> expected_out_shape = dst_shapes[i];
        EXPECT_EQ(inferred_out_shape, expected_out_shape);
    }
}

TEST(OpSchema, InferUnsqueezeOutputShape) {
    auto &be = graph::dnnl_impl::dnnl_backend::get_singleton();
    EXPECT_EQ(be.get_name(), "dnnl_backend");
    const op_kind_t kind = dnnl_impl::op_kind::dnnl_unsqueeze;
    const op_schema_t *op_schema_ = op_schema_registry_t::get_op_schema(kind);

    const std::vector<int64_t> src_shape {4};

    const std::vector<std::vector<int64_t>> dst_shapes {
            {1, 1, 1, 4}, {1, 1, 4, 1}};
    const std::vector<std::vector<int64_t>> axes {{0, 1, 2}, {0, 1, -1}};

    for (size_t i = 0; i < dst_shapes.size(); ++i) {
        op_t op {kind, "unsqueeze"};
        op.set_attr<std::vector<int64_t>>(dnnl_impl::op_attr::axes, axes[i]);

        logical_tensor_t lt_in = logical_tensor_init(
                0, src_shape, data_type::f32, layout_type::strided);
        logical_tensor_t lt_out
                = logical_tensor_init(1, data_type::f32, layout_type::strided);

        std::vector<logical_tensor_t *> in {&lt_in};
        std::vector<logical_tensor_t *> out {&lt_out};

        status_t ret = op_schema_->shape_infer(&op, in, out);
        EXPECT_EQ(ret, status::success);

        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        const std::vector<int64_t> &expected_out_shape = dst_shapes[i];
        EXPECT_EQ(inferred_out_shape, expected_out_shape);
    }
}

TEST(OpSchema, InferUnsqueezeOutputShapeBasedOnAxes) {
    auto &be = graph::dnnl_impl::dnnl_backend::get_singleton();
    EXPECT_EQ(be.get_name(), "dnnl_backend");
    const op_kind_t kind = dnnl_impl::op_kind::dnnl_unsqueeze;
    const op_schema_t *op_schema_ = op_schema_registry_t::get_op_schema(kind);

    const std::vector<std::vector<int64_t>> axes_list {
            {1}, {1, 2}, {-1}, {-1, -2}};
    const std::vector<std::vector<int64_t>> src_shapes {
            {3, 4, 5}, {3, 4, 5}, {3, 4, 5}, {3, 4, 5}};
    const std::vector<std::vector<int64_t>> dst_shapes {
            {3, 1, 4, 5}, {3, 1, 1, 4, 5}, {3, 4, 5, 1}, {3, 4, 5, 1, 1}};

    for (size_t i = 0; i < axes_list.size(); ++i) {
        op_t op {kind, "unsqueeze"};
        op.set_attr<std::vector<int64_t>>(op_attr::axes, axes_list[i]);

        logical_tensor_t lt_in = logical_tensor_init(
                0, src_shapes[i], data_type::f32, layout_type::strided);
        logical_tensor_t lt_out
                = logical_tensor_init(1, data_type::f32, layout_type::strided);

        std::vector<logical_tensor_t *> in {&lt_in};
        std::vector<logical_tensor_t *> out {&lt_out};

        status_t ret = op_schema_->shape_infer(&op, in, out);
        EXPECT_EQ(ret, status::success);

        const std::vector<int64_t> inferred_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        const std::vector<int64_t> &expected_out_shape = dst_shapes[i];
        EXPECT_EQ(inferred_out_shape, expected_out_shape);
    }
}

TEST(OpSchema, DnnlBinary) {
    op_kind_t op_kind = dnnl_impl::op_kind::dnnl_binary;
    const size_t expected_in_size_lower = 2;
    const size_t expected_out_size = 2;
    const size_t expected_attr_size = 7;
    const std::map<op_attr_t, bool> attrs_data
            = {{op_attr::auto_broadcast, false},
                    {dnnl_impl::op_attr::alg_kind, true}};
    verify_op_schema(op_kind, expected_in_size_lower, expected_out_size,
            expected_attr_size, attrs_data);

    const size_t expected_in_size_upper = 32;
    verify_op_schema(op_kind, expected_in_size_upper, expected_out_size,
            expected_attr_size, attrs_data);
}

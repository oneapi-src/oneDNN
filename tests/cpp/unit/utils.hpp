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
#ifndef UTILS_HPP
#define UTILS_HPP

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/op.hpp"
#include "interface/op_schema.hpp"
#include "interface/value.hpp"

namespace dnnl {
namespace graph {
namespace tests {
namespace unit {
namespace utils {

#define EXPECT_SUCCESS(expression) \
    EXPECT_EQ((expression), dnnl::graph::impl::status::success)

#define SKIP_IF(cond, msg) \
    do { \
        if (cond) { \
            std::cout << "[  SKIPPED ] " << (msg) << std::endl; \
            GTEST_SKIP(); \
        } \
    } while (0)

static inline dnnl::graph::impl::logical_tensor_t logical_tensor_init(size_t id,
        dnnl::graph::impl::data_type_t dtype,
        dnnl::graph::impl::layout_type_t ltype
        = dnnl::graph::impl::layout_type::undef) {
    dnnl::graph::impl::logical_tensor_t val;
    val.id = id;
    val.data_type = dtype;
    val.layout_type = ltype;
    val.ndims = -1;
    // initialize dims and layout field to avoid dirty data
    val.dims[0] = -1;
    val.layout.strides[0] = -1;
    val.property = dnnl::graph::impl::property_type::undef;

    return val;
}

static inline dnnl::graph::impl::logical_tensor_t logical_tensor_init(size_t id,
        std::vector<dnnl::graph::impl::dim_t> dims,
        dnnl::graph::impl::data_type_t dtype,
        dnnl::graph::impl::layout_type_t ltype
        = dnnl::graph::impl::layout_type::strided) {
    if (dims.size() == 0) { return logical_tensor_init(id, dtype); }
    dnnl::graph::impl::logical_tensor_t val;
    val.id = id;
    val.data_type = dtype;
    val.ndims = static_cast<int>(dims.size());
    val.property = dnnl::graph::impl::property_type::undef;

    // dims
    for (size_t d = 0; d < dims.size(); ++d) {
        val.dims[d] = dims[d];
    }

    // strides
    val.layout_type = ltype;
    if (ltype == dnnl::graph::impl::layout_type::strided) {
        val.layout.strides[val.ndims - 1] = 1;
        for (int s = val.ndims - 2; s >= 0; --s) {
            size_t si = static_cast<size_t>(s);
            val.layout.strides[si] = dims[si + 1] * val.layout.strides[si + 1];
        }
    } else {
        // initialize layout field to avoid dirty data
        val.layout.strides[0] = -1;
    }

    return val;
}

static inline dnnl::graph::impl::logical_tensor_t logical_tensor_init(size_t id,
        std::vector<dnnl::graph::impl::dim_t> dims,
        std::vector<dnnl::graph::impl::dim_t> strides,
        dnnl::graph::impl::data_type_t dtype) {
    dnnl::graph::impl::logical_tensor_t val;
    val.id = id;
    val.data_type = dtype;
    val.ndims = static_cast<int>(dims.size());

    // dims and strides
    for (size_t d = 0; d < dims.size(); ++d) {
        val.dims[d] = dims[d];
        val.layout.strides[d] = strides[d];
    }

    val.layout_type = dnnl::graph::impl::layout_type::strided;
    val.property = dnnl::graph::impl::property_type::undef;

    return val;
}

static inline std::vector<int64_t> compute_dense_strides(
        const std::vector<int64_t> &output_dims) {
    std::vector<int64_t> output_strides(output_dims.size());
    for (auto it = output_dims.begin(); it < output_dims.end(); ++it) {
        const auto val = std::accumulate(std::next(it), output_dims.end(), 1,
                std::multiplies<int64_t>());
        const auto dist = std::distance(output_dims.begin(), it);
        output_strides[static_cast<size_t>(dist)] = val;
    }
    return output_strides;
}

static inline std::vector<dnnl::graph::impl::logical_tensor_t>
create_logical_tensors(size_t num_lt) {
    size_t count = 0;
    std::vector<dnnl::graph::impl::logical_tensor_t> lt_vec;
    lt_vec.reserve(num_lt);
    while (count < num_lt) {
        lt_vec.emplace_back(logical_tensor_init(count, impl::data_type::f32));
        count++;
    }
    return lt_vec;
}

/**
 * This function verifies op schema. Should be used as a test helper.
 * attrs_data argument should contain all attributes (as keys) associated with op_kind,
 * along with information (as value) whether they are required or not.
 * Please treat the op_schema_test.Convolution as an example.
 */
static inline void verify_op_schema(const dnnl::graph::impl::op_kind_t op_kind_,
        const size_t expected_in_size, const size_t expected_out_size,
        const size_t expected_attr_size,
        const std::map<dnnl::graph::impl::op_attr_t, bool> &attrs_data) {
    using namespace dnnl::graph::impl;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    EXPECT_TRUE(nullptr != op_schema_);

    const std::set<size_t> input_size = op_schema_->get_num_inputs();
    EXPECT_TRUE(input_size.find(expected_in_size) != input_size.end());

    const std::set<size_t> output_size = op_schema_->get_num_outputs();
    EXPECT_TRUE(output_size.find(expected_out_size) != output_size.end());

    size_t attr_size = op_schema_->get_attrs().size();
    EXPECT_EQ(attr_size, expected_attr_size);

    for (const auto &attr_data : attrs_data) {
        const auto &attr_name = attr_data.first;
        const auto is_required = attr_data.second;
        EXPECT_EQ(op_schema_->get_attrs().count(attr_name), 1);
        EXPECT_EQ(op_schema_->get_attrs().at(attr_name).required_, is_required);
    }
}

static inline void verify_shape_infer_for_arithmetic_op_no_broadcast(
        const dnnl::graph::impl::op_kind_t op_kind_) {
    using namespace dnnl::graph::impl;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};
    const std::string no_broadcast_attr_val = "none";
    op_.set_attr(op_attr::auto_broadcast, no_broadcast_attr_val);

    // In valid situation, the inputs layout should only be strided or opaque,
    // so we need to test both of these two input layout type
    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    for (const auto &ltype : layout_types) {
        logical_tensor_t lt_in1 = logical_tensor_init(
                0, {1, 3, 416, 416}, data_type::f32, ltype);
        logical_tensor_t lt_in2 = logical_tensor_init(
                1, {1, 3, 416, 416}, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in {&lt_in1, &lt_in2};
        logical_tensor_t lt_out
                = logical_tensor_init(2, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> out {&lt_out};

        status_t ret = op_schema_->shape_infer(&op_, in, out);
        EXPECT_EQ(ret, status::success);
        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        const std::vector<int64_t> expected_out_shape = {1, 3, 416, 416};
        EXPECT_EQ(infered_out_shape, expected_out_shape);

        const std::vector<int64_t> infered_out_strides
                = logical_tensor_wrapper_t(lt_out).vstrides();
        const std::vector<int64_t> expected_out_strides
                = compute_dense_strides(expected_out_shape);
        EXPECT_EQ(infered_out_strides, expected_out_strides);

        // negative case - non-matching input dims
        logical_tensor_t lt_in2_neg
                = logical_tensor_init(1, {1, 3, 32, 32}, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in_neg {&lt_in1, &lt_in2_neg};
        logical_tensor_t lt_out_neg
                = logical_tensor_init(2, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> out_neg {&lt_out_neg};
        ret = op_schema_->shape_infer(&op_, in_neg, out_neg);
        EXPECT_EQ(ret, status::invalid_shape);
    }
}

#define for_ for
static inline void verify_shape_infer_for_arithmetic_op_with_broadcast(
        const dnnl::graph::impl::op_kind_t op_kind_) {
    using namespace dnnl::graph::impl;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};
    const std::vector<std::vector<int64_t>> in1_shapes
            = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    const std::vector<std::vector<int64_t>> in2_shapes = {{3, 1, 64}, {1}};
    const std::vector<std::vector<int64_t>> expected_out_shapes
            = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    for_(const auto &in1_shape : in1_shapes)
    for_(const auto &in2_shape : in2_shapes)
    for_(const auto &expected_out_shape : expected_out_shapes)
    for (const auto &ltype : layout_types) {
        logical_tensor_t lt_in1
                = logical_tensor_init(0, in1_shape, data_type::f32, ltype);
        logical_tensor_t lt_in2
                = logical_tensor_init(1, in2_shape, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in {&lt_in1, &lt_in2};
        logical_tensor_t lt_out
                = logical_tensor_init(2, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> out {&lt_out};

        // shape inference without explicitly setting auto_broadcast
        // should be enabled by default
        op_schema_->shape_infer(&op_, in, out);
        const std::vector<int64_t> infered_out_shape
                = logical_tensor_wrapper_t(lt_out).vdims();
        EXPECT_EQ(infered_out_shape, expected_out_shape);

        const std::vector<int64_t> infered_out_strides
                = logical_tensor_wrapper_t(lt_out).vstrides();
        const std::vector<int64_t> expected_out_strides
                = compute_dense_strides(expected_out_shape);
        EXPECT_EQ(infered_out_strides, expected_out_strides);

        // explicitly setting auto_broadcast
        const std::string with_broadcast_attr_val = "numpy";
        op_.set_attr(op_attr::auto_broadcast, with_broadcast_attr_val);
        logical_tensor_t lt_out_expl
                = logical_tensor_init(3, data_type::f32, layout_type::strided);
        std::vector<logical_tensor_t *> out_expl {&lt_out_expl};

        op_schema_->shape_infer(&op_, in, out_expl);
        const std::vector<int64_t> infered_out_shape_expl
                = logical_tensor_wrapper_t(lt_out_expl).vdims();
        EXPECT_EQ(infered_out_shape_expl, expected_out_shape);

        const std::vector<int64_t> infered_out_strides2
                = logical_tensor_wrapper_t(lt_out).vstrides();
        EXPECT_EQ(infered_out_strides2, expected_out_strides);
    }
}
#undef for_

static inline void set_conv_common_attr(impl::op_t &conv,
        std::vector<int64_t> strides = {1, 1},
        std::vector<int64_t> pads_begin = {0, 0},
        std::vector<int64_t> pads_end = {0, 0},
        std::vector<int64_t> dilations = {1, 1}, std::string auto_pad = "None",
        std::string data_format = "NXC", std::string filter_format = "XIO",
        int64_t groups = 1) {
    conv.set_attr(impl::op_attr::strides, strides);
    conv.set_attr(impl::op_attr::pads_begin, pads_begin);
    conv.set_attr(impl::op_attr::pads_end, pads_end);
    conv.set_attr(impl::op_attr::dilations, dilations);
    conv.set_attr(impl::op_attr::auto_pad, auto_pad);
    conv.set_attr(impl::op_attr::data_format, data_format);
    conv.set_attr(impl::op_attr::filter_format, filter_format);
    conv.set_attr(impl::op_attr::groups, groups);
}

static inline void set_conv_dw_base_op_attr(impl::op_t &conv) {
    std::vector<int64_t> conv_strides {1, 1};
    std::vector<int64_t> conv_pads_begin {0, 0};
    std::vector<int64_t> conv_pads_end {0, 0};
    std::vector<int64_t> conv_dilations {1, 1};
    std::string conv_auto_pad = "None";
    std::string conv_data_format = "NCX";
    std::string conv_filter_format = "OIX";
    int64_t conv_groups = 1;
    set_conv_common_attr(conv, conv_strides, conv_pads_begin, conv_pads_end,
            conv_dilations, conv_auto_pad, conv_data_format, conv_filter_format,
            conv_groups);
}

static inline void set_conv_dw_post_op_attr(
        impl::op_t &dw, const std::string &dw_type) {
    std::vector<int64_t> dw_strides = ("k3s1p1" == dw_type)
            ? std::vector<int64_t> {1, 1}
            : std::vector<int64_t> {2, 2};
    std::vector<int64_t> dw_pads_begin {1, 1};
    std::vector<int64_t> dw_pads_end {1, 1};
    std::vector<int64_t> dw_dilations {1, 1};
    std::string dw_auto_pad = "None";
    std::string dw_data_format = "NCX";
    std::string dw_filter_format = "OIX";
    int64_t dw_groups = 4;
    set_conv_common_attr(dw, dw_strides, dw_pads_begin, dw_pads_end,
            dw_dilations, dw_auto_pad, dw_data_format, dw_filter_format,
            dw_groups);
}

static inline void set_convtranspose_common_attr(
        dnnl::graph::impl::op_t &convtranspose,
        std::vector<int64_t> strides = {1, 1},
        std::vector<int64_t> pads_begin = {0, 0},
        std::vector<int64_t> pads_end = {0, 0},
        std::vector<int64_t> dilations = {1, 1}, std::string auto_pad = "None",
        std::string data_format = "NXC", std::string filter_format = "XIO",
        int64_t groups = 1, std::vector<int64_t> output_padding = {0, 0}) {
    set_conv_common_attr(convtranspose, strides, pads_begin, pads_end,
            dilations, auto_pad, data_format, filter_format, groups);
    convtranspose.set_attr(impl::op_attr::output_padding, output_padding);
}

static inline void infer_conv_shape(dnnl::graph::impl::op_kind_t kind) {
    using namespace dnnl::graph::impl;
    const op_schema_t *conv_op_schema
            = op_schema_registry_t::get_op_schema(kind);

    op_t conv_op {kind, op_t::kind2str(kind)};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> dilations = {1, 1};
    std::string auto_pad = "VALID";
    std::string data_format = "NXC";
    std::string filter_format = "OIX";
    int64_t groups = 1;

    set_conv_common_attr(conv_op, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    // data shape {N, H, W, IC}
    logical_tensor_t lt_data_0
            = logical_tensor_init(0, {1, 224, 224, 3}, data_type::f32);
    // weight shape {OC, IC, KH, KW}
    logical_tensor_t lt_weight
            = logical_tensor_init(1, {16, 3, 3, 3}, data_type::f32);
    // bias shape {OC}
    logical_tensor_t lt_bias = logical_tensor_init(2, {16}, data_type::f32);

    // add input
    logical_tensor_t lt_data_1
            = logical_tensor_init(3, {1, 16, 111, 111}, data_type::f32);

    std::vector<logical_tensor_t *> lt_in {
            &lt_data_0, &lt_weight, &lt_bias, &lt_data_1};
    logical_tensor_t lt_o
            = logical_tensor_init(4, {-1, -1, -1, -1}, data_type::f32);
    std::vector<logical_tensor_t *> lt_out {&lt_o};

    conv_op_schema->shape_infer(&conv_op, lt_in, lt_out);
    auto infered_pads_begin
            = conv_op.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
    auto infered_pads_end
            = conv_op.get_attr<std::vector<int64_t>>(op_attr::pads_end);
    const std::vector<int64_t> expected_pads = {0, 0};
    EXPECT_EQ(infered_pads_begin, expected_pads);
    EXPECT_EQ(infered_pads_end, expected_pads);

    const std::vector<int64_t> expect_output_shape = {1, 111, 111, 16};
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_o).vdims();
    EXPECT_EQ(infered_out_shape, expect_output_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_o).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expect_output_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

static inline void verify_shape_infer_for_conv(
        const dnnl::graph::impl::op_kind_t op_kind_, std::string data_format,
        std::string filter_format, int64_t groups,
        const std::vector<int64_t> &in_data,
        const std::vector<int64_t> &in_weight,
        const std::vector<int64_t> &expected_out_shape) {
    using namespace dnnl::graph::impl;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    std::vector<int64_t> strides;
    std::vector<int64_t> pads_begin;
    std::vector<int64_t> pads_end;
    std::vector<int64_t> dilations;
    strides.assign(in_data.size() - 2, 2);
    pads_begin.assign(in_data.size() - 2, 1);
    pads_end.assign(in_data.size() - 2, 2);
    dilations.assign(in_data.size() - 2, 1);
    std::string auto_pad = "VALID";

    set_conv_common_attr(op_, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    logical_tensor_t lt_data = logical_tensor_init(0, in_data, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, in_weight, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_data, &lt_weight};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    // shape inference without explicitly setting auto_broadcast
    // should be enabled by default
    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    auto infered_pads_begin
            = op_.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
    auto infered_pads_end
            = op_.get_attr<std::vector<int64_t>>(op_attr::pads_end);
    std::vector<int64_t> expected_pads;
    expected_pads.assign(in_data.size() - 2, 0);
    EXPECT_EQ(infered_pads_begin, expected_pads);
    EXPECT_EQ(infered_pads_end, expected_pads);
}

static inline void verify_shape_infer_for_convtranspose_bprop_data(
        const dnnl::graph::impl::op_kind_t op_kind_,
        const std::string &data_format, const std::string &filter_format,
        const int64_t groups, const int64_t stride, const int64_t pad,
        const int64_t dilation, const std::vector<int64_t> &diff_dst_shape,
        const std::vector<int64_t> &wei_shape,
        const std::vector<int64_t> &expected_diff_src_shape) {
    using namespace dnnl::graph::impl;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    const size_t spatial_dims = diff_dst_shape.size() - 2;
    const std::vector<int64_t> strides(spatial_dims, stride);
    const std::vector<int64_t> pads_begin(spatial_dims, pad);
    const std::vector<int64_t> pads_end(spatial_dims, pad);
    const std::vector<int64_t> dilations(spatial_dims, dilation);
    const std::string auto_pad {"None"};

    set_conv_common_attr(op_, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    logical_tensor_t diff_dst_lt
            = logical_tensor_init(0, diff_dst_shape, data_type::f32);
    logical_tensor_t wei_lt = logical_tensor_init(1, wei_shape, data_type::f32);
    logical_tensor_t diff_src_lt = logical_tensor_init(2, data_type::f32);

    std::vector<logical_tensor_t *> in_lts {&diff_dst_lt, &wei_lt};
    std::vector<logical_tensor_t *> out_lts {&diff_src_lt};

    op_schema_->shape_infer(&op_, in_lts, out_lts);
    const std::vector<int64_t> infered_diff_src_shape
            = logical_tensor_wrapper_t(diff_src_lt).vdims();
    EXPECT_EQ(infered_diff_src_shape, expected_diff_src_shape);
}

static inline void verify_shape_infer_for_convtranspose(
        const dnnl::graph::impl::op_kind_t op_kind_, std::string data_format,
        std::string filter_format, int64_t groups,
        const std::vector<int64_t> &in_data,
        const std::vector<int64_t> &in_weight,
        const std::vector<int64_t> &expected_out_shape) {
    using namespace dnnl::graph::impl;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    std::vector<int64_t> strides;
    std::vector<int64_t> pads_begin;
    std::vector<int64_t> pads_end;
    std::vector<int64_t> dilations;
    std::vector<int64_t> output_padding;
    strides.assign(in_data.size() - 2, 2);
    pads_begin.assign(in_data.size() - 2, 1);
    pads_end.assign(in_data.size() - 2, 2);
    dilations.assign(in_data.size() - 2, 1);
    output_padding.assign(in_data.size() - 2, 1);
    std::string auto_pad = "VALID";

    set_convtranspose_common_attr(op_, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups, output_padding);

    logical_tensor_t lt_data = logical_tensor_init(0, in_data, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, in_weight, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_data, &lt_weight};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    auto infered_pads_begin
            = op_.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
    auto infered_pads_end
            = op_.get_attr<std::vector<int64_t>>(op_attr::pads_end);
    std::vector<int64_t> expected_pads;
    expected_pads.assign(in_data.size() - 2, 0);
    EXPECT_EQ(infered_pads_begin, expected_pads);
    EXPECT_EQ(infered_pads_end, expected_pads);
}

static inline void verify_shape_infer_for_conv(
        const dnnl::graph::impl::op_kind_t op_kind_, std::string data_format,
        std::string filter_format, int64_t groups,
        const std::vector<int64_t> &in_data,
        const std::vector<int64_t> &in_weight,
        const std::vector<int64_t> &in_bias,
        const std::vector<int64_t> &expected_out_shape) {
    using namespace dnnl::graph::impl;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    std::vector<int64_t> strides;
    std::vector<int64_t> pads_begin;
    std::vector<int64_t> pads_end;
    std::vector<int64_t> dilations;
    strides.assign(in_data.size() - 2, 2);
    pads_begin.assign(in_data.size() - 2, 1);
    pads_end.assign(in_data.size() - 2, 2);
    dilations.assign(in_data.size() - 2, 1);
    std::string auto_pad = "VALID";

    set_conv_common_attr(op_, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);

    logical_tensor_t lt_data = logical_tensor_init(0, in_data, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, in_weight, data_type::f32);
    logical_tensor_t lt_bias = logical_tensor_init(0, in_bias, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_data, &lt_weight, &lt_bias};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    // shape inference without explicitly setting auto_broadcast
    // should be enabled by default
    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    auto infered_pads_begin
            = op_.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
    auto infered_pads_end
            = op_.get_attr<std::vector<int64_t>>(op_attr::pads_end);
    std::vector<int64_t> expected_pads;
    expected_pads.assign(in_data.size() - 2, 0);
    EXPECT_EQ(infered_pads_begin, expected_pads);
    EXPECT_EQ(infered_pads_end, expected_pads);
}

static inline void verify_shape_infer_for_conv_bprop_data(
        const dnnl::graph::impl::op_kind_t op_kind_, std::string data_format,
        std::string filter_format, int64_t groups,
        const std::vector<int64_t> &in_data,
        const std::vector<int64_t> &in_weight,
        const std::vector<int64_t> &in_output_shape,
        const std::vector<int64_t> &expected_out_shape) {
    using namespace dnnl::graph::impl;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads_begin = {1, 1};
    std::vector<int64_t> pads_end = {2, 2};
    std::vector<int64_t> output_padding = {3, 3};
    std::vector<int64_t> dilations = {1, 1};
    std::string auto_pad = "VALID";

    set_conv_common_attr(op_, strides, pads_begin, pads_end, dilations,
            auto_pad, data_format, filter_format, groups);
    op_.set_attr(op_attr::output_padding, output_padding);
    op_.set_attr(op_attr::output_shape, expected_out_shape);

    logical_tensor_t lt_data = logical_tensor_init(0, in_data, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, in_weight, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_data, &lt_weight};
    logical_tensor_t lt_out = logical_tensor_init(2, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    // shape inference without explicitly setting auto_broadcast
    // should be enabled by default
    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    auto infered_pads_begin
            = op_.get_attr<std::vector<int64_t>>(op_attr::pads_begin);
    auto infered_pads_end
            = op_.get_attr<std::vector<int64_t>>(op_attr::pads_end);
    const std::vector<int64_t> expected_pads = {0, 0};
    EXPECT_EQ(infered_pads_begin, expected_pads);
    EXPECT_EQ(infered_pads_end, expected_pads);
}

static inline void verify_identity_shape_infer_(
        const dnnl::graph::impl::op_kind_t op_kind_, const size_t out_lt_id,
        std::vector<dnnl::graph::impl::logical_tensor_t *> &in) {
    using namespace dnnl::graph::impl;
    const op_schema_t *op_schema_
            = op_schema_registry_t::get_op_schema(op_kind_);
    op_t op_ {op_kind_, op_t::kind2str(op_kind_)};

    logical_tensor_t lt_out = logical_tensor_init(
            out_lt_id, data_type::f32, layout_type::strided);
    std::vector<logical_tensor_t *> out {&lt_out};

    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    const std::vector<int64_t> expected_out_shape = {1, 3, 224, 224};
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    const std::vector<int64_t> infered_out_strides
            = logical_tensor_wrapper_t(lt_out).vstrides();
    const std::vector<int64_t> expected_out_strides
            = compute_dense_strides(expected_out_shape);
    EXPECT_EQ(infered_out_strides, expected_out_strides);
}

static inline void verify_single_in_identity_shape_infer(
        const dnnl::graph::impl::op_kind_t op_kind_) {
    using namespace dnnl::graph::impl;
    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    for (const auto &ltype : layout_types) {
        logical_tensor_t lt_in = logical_tensor_init(
                0, {1, 3, 224, 224}, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in {&lt_in};
        verify_identity_shape_infer_(op_kind_, 1, in);
    }
}

static inline void verify_two_ins_identity_shape_infer(
        const dnnl::graph::impl::op_kind_t op_kind_) {
    using namespace dnnl::graph::impl;
    const std::vector<layout_type_t> layout_types
            = {layout_type::strided, layout_type::opaque};

    for (const auto &ltype : layout_types) {
        logical_tensor_t lt_in1 = logical_tensor_init(
                0, {1, 3, 224, 224}, data_type::f32, ltype);
        logical_tensor_t lt_in2 = logical_tensor_init(
                1, {1, 3, 224, 224}, data_type::f32, ltype);
        std::vector<logical_tensor_t *> in {&lt_in1, &lt_in2};
        verify_identity_shape_infer_(op_kind_, 2, in);
    }
}

inline dnnl_graph_dim_t product(const std::vector<int64_t> &dims) {
    return dims.empty()
            ? 0
            : std::accumulate(dims.begin(), dims.end(), (dnnl_graph_dim_t)1,
                    std::multiplies<dnnl_graph_dim_t>());
}

inline void construct_f32_MHA(dnnl::graph::impl::graph_t *agraph,
        int batch_size = 1, int seq_len = 384, int num_head = 16,
        int head_dim = 1024) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests;

    int size_per_head = head_dim / num_head;
    dims MIXED_LAYER_INPUT_SHAPE = {batch_size, seq_len, head_dim};
    dims EXTENDED_ATTENTION_MASK_SHAPE = {batch_size, 1, 1, seq_len};
    dims QKV_RESHAPED_SHAPE = {batch_size, seq_len, num_head, size_per_head};
    dims QKV_TRANSPOSED_SHAPE = {batch_size, num_head, seq_len, size_per_head};
    dims KEY_TRANSPOSED_SHAPE = {batch_size, num_head, size_per_head, seq_len};
    dims MATMUL_QK_OUTPUT_SHAPE = {batch_size, num_head, seq_len, seq_len};
    dims MATMUL_V_OUTPUT_SHAPE = {batch_size, num_head, seq_len, size_per_head};

    dims CONST_SHAPE = {1};

    dims QKV_TRANSPOSED_ORDER = {0, 2, 1, 3};
    dims KEY_TRANSPOSED_ORDER = {0, 1, 3, 2};

    size_t lt_id = 0;

    auto query_gemm = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);
    auto qk_bmm = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);
    auto value_bmm = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);
    auto attention_mask_flt = unit::utils::logical_tensor_init(
            lt_id++, EXTENDED_ATTENTION_MASK_SHAPE, data_type::f32);

    auto query_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);
    auto query_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto key_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);
    auto key_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto key_transpose_out2 = unit::utils::logical_tensor_init(
            lt_id++, KEY_TRANSPOSED_SHAPE, data_type::f32);

    auto matmul_qk_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto fscore_scale = unit::utils::logical_tensor_init(
            lt_id++, CONST_SHAPE, data_type::f32);
    fscore_scale.property = property_type::constant;
    auto fscore_div_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto fscore_add_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);
    auto softmax_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto value_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);
    auto value_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto matmul_v_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_V_OUTPUT_SHAPE, data_type::f32);

    auto context_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);

    // reshape + transpose for query + key
    op_t query_reshape {0, op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(op_attr::special_zero, false);
    query_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    op_t query_transpose {1, op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    op_t key_reshape {2, op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(op_attr::special_zero, false);
    key_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    op_t key_transpose {3, op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    // alternative for transpose
    op_t key_transpose2 {4, op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr<std::vector<int64_t>>(
            op_attr::order, KEY_TRANSPOSED_ORDER);

    op_t matmul_qk {9, op_kind::MatMul, "matmul_qk"};

    op_t fscore_div {10, op_kind::Divide, "fscore_div"};
    fscore_div.set_attr(op_attr::auto_broadcast, std::string("numpy"));
    op_t fscore_add {11, op_kind::Add, "fscore_add"};
    fscore_add.set_attr(op_attr::auto_broadcast, std::string("numpy"));
    op_t softmax {12, op_kind::SoftMax, "softmax"};
    softmax.set_attr(op_attr::axis, (int64_t)3);

    // reshape + transpose for value
    op_t value_reshape {15, op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(op_attr::special_zero, false);
    value_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    op_t value_transpose {16, op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    op_t matmul_v {19, op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    op_t transpose_output {20, op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    op_t reshape_output {21, op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(op_attr::special_zero, false);
    reshape_output.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    query_reshape.add_input(query_gemm);
    query_reshape.add_output(query_reshape_out);
    query_transpose.add_input(query_reshape_out);
    query_transpose.add_output(query_transpose_out);
    key_reshape.add_input(qk_bmm);
    key_reshape.add_output(key_reshape_out);
    key_transpose.add_input(key_reshape_out);
    key_transpose.add_output(key_transpose_out);
    key_transpose2.add_input(key_transpose_out);
    key_transpose2.add_output(key_transpose_out2);

    matmul_qk.add_input(query_transpose_out);
    matmul_qk.add_input(key_transpose_out2);
    matmul_qk.add_output(matmul_qk_out);

    fscore_div.add_input(matmul_qk_out);
    fscore_div.add_input(fscore_scale);
    fscore_div.add_output(fscore_div_out);
    fscore_add.add_input(fscore_div_out);
    fscore_add.add_input(attention_mask_flt);
    fscore_add.add_output(fscore_add_out);
    softmax.add_input(fscore_add_out);
    softmax.add_output(softmax_out);

    value_reshape.add_input(value_bmm);
    value_reshape.add_output(value_reshape_out);
    value_transpose.add_input(value_reshape_out);
    value_transpose.add_output(value_transpose_out);
    matmul_v.add_input(softmax_out);
    matmul_v.add_input(value_transpose_out);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);

    agraph->add_op(&query_reshape);
    agraph->add_op(&query_transpose);
    agraph->add_op(&key_reshape);
    agraph->add_op(&key_transpose);
    agraph->add_op(&key_transpose2);
    agraph->add_op(&matmul_qk);

    agraph->add_op(&fscore_div);
    agraph->add_op(&fscore_add);
    agraph->add_op(&softmax);
    agraph->add_op(&value_reshape);
    agraph->add_op(&value_transpose);
    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
}

inline void construct_int8_MHA(dnnl::graph::impl::graph_t *agraph,
        int batch_size = 1, int seq_len = 384, int num_head = 16,
        int head_dim = 1024) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests;

    int size_per_head = head_dim / num_head;
    dims MIXED_LAYER_INPUT_SHAPE = {batch_size, seq_len, head_dim};
    dims EXTENDED_ATTENTION_MASK_SHAPE = {batch_size, 1, 1, seq_len};
    dims QKV_RESHAPED_SHAPE = {batch_size, seq_len, num_head, size_per_head};
    dims QKV_TRANSPOSED_SHAPE = {batch_size, num_head, seq_len, size_per_head};
    dims KEY_TRANSPOSED_SHAPE = {batch_size, num_head, size_per_head, seq_len};
    dims MATMUL_QK_OUTPUT_SHAPE = {batch_size, num_head, seq_len, seq_len};
    dims MATMUL_V_OUTPUT_SHAPE = {batch_size, num_head, seq_len, size_per_head};

    dims CONST_SHAPE = {1};

    dims QKV_TRANSPOSED_ORDER = {0, 2, 1, 3};
    dims KEY_TRANSPOSED_ORDER = {0, 1, 3, 2};

    size_t lt_id = 0;

    auto query_gemm = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);
    auto qk_bmm = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);
    auto value_bmm = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);
    auto attention_mask_flt = unit::utils::logical_tensor_init(
            lt_id++, EXTENDED_ATTENTION_MASK_SHAPE, data_type::f32);

    auto query_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);
    auto query_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto key_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);
    auto key_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto key_transpose_out2 = unit::utils::logical_tensor_init(
            lt_id++, KEY_TRANSPOSED_SHAPE, data_type::f32);

    auto query_quantize = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::u8);
    auto query_dequantize = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto key_quantize = unit::utils::logical_tensor_init(
            lt_id++, KEY_TRANSPOSED_SHAPE, data_type::u8);
    auto key_dequantize = unit::utils::logical_tensor_init(
            lt_id++, KEY_TRANSPOSED_SHAPE, data_type::f32);

    auto matmul_qk_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto fscore_scale = unit::utils::logical_tensor_init(
            lt_id++, CONST_SHAPE, data_type::f32);
    fscore_scale.property = property_type::constant;
    auto fscore_div_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto fscore_add_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);
    auto softmax_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto value_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);
    auto value_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto value_quantize = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::u8);
    auto value_dequantize = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto matmul_v_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_V_OUTPUT_SHAPE, data_type::f32);

    auto softmax_out_q = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::u8);
    auto softmax_out_deq = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto context_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);

    // reshape + transpose for query + key
    op_t query_reshape {0, op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(op_attr::special_zero, false);
    query_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    op_t query_transpose {1, op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    op_t key_reshape {2, op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(op_attr::special_zero, false);
    key_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    op_t key_transpose {3, op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    // alternative for transpose
    op_t key_transpose2 {4, op_kind::StaticTranspose, "key_transpose2"};
    key_transpose2.set_attr<std::vector<int64_t>>(
            op_attr::order, KEY_TRANSPOSED_ORDER);

    op_t quantize_query {5, op_kind::Quantize, "quantize_query"};
    op_t quantize_key {6, op_kind::Quantize, "quantize_key"};
    quantize_query.set_attr(op_attr::scales, std::vector<float>({0.12f}));
    quantize_query.set_attr(op_attr::zps, std::vector<int64_t>({2}));
    quantize_query.set_attr(op_attr::qtype, std::string("per_tensor"));
    quantize_query.set_attr(op_attr::axis, (int64_t)0);
    quantize_key.set_attr(op_attr::scales, std::vector<float>({0.12f}));
    quantize_key.set_attr(op_attr::zps, std::vector<int64_t>({2}));
    quantize_key.set_attr(op_attr::qtype, std::string("per_tensor"));
    quantize_key.set_attr(op_attr::axis, (int64_t)0);

    op_t dequantize_query {7, op_kind::Dequantize, "dequantize_query"};
    dequantize_query.set_attr(op_attr::scales, std::vector<float>({0.12f}));
    dequantize_query.set_attr(op_attr::zps, std::vector<int64_t>({2}));
    dequantize_query.set_attr(op_attr::qtype, std::string("per_tensor"));
    dequantize_query.set_attr(op_attr::axis, (int64_t)0);
    op_t dequantize_key {8, op_kind::Dequantize, "dequantize_key"};
    dequantize_key.set_attr(op_attr::scales, std::vector<float>({0.12f}));
    dequantize_key.set_attr(op_attr::zps, std::vector<int64_t>({2}));
    dequantize_key.set_attr(op_attr::qtype, std::string("per_tensor"));
    dequantize_key.set_attr(op_attr::axis, (int64_t)0);

    op_t matmul_qk {9, op_kind::MatMul, "matmul_qk"};

    op_t fscore_div {10, op_kind::Divide, "fscore_div"};
    fscore_div.set_attr(op_attr::auto_broadcast, std::string("numpy"));
    op_t fscore_add {11, op_kind::Add, "fscore_add"};
    fscore_add.set_attr(op_attr::auto_broadcast, std::string("numpy"));
    op_t softmax {12, op_kind::SoftMax, "softmax"};
    softmax.set_attr(op_attr::axis, (int64_t)3);
    // quantize-dequantize softmax's output
    op_t quantize_softmax {13, op_kind::Quantize, "quantize_softmax"};
    op_t dequantize_softmax {14, op_kind::Dequantize, "dequantize_softmax"};
    quantize_softmax.set_attr(op_attr::scales, std::vector<float>({0.12f}));
    quantize_softmax.set_attr(op_attr::zps, std::vector<int64_t>({2}));
    quantize_softmax.set_attr(op_attr::qtype, std::string("per_tensor"));
    quantize_softmax.set_attr(op_attr::axis, (int64_t)0);
    dequantize_softmax.set_attr(op_attr::scales, std::vector<float>({0.12f}));
    dequantize_softmax.set_attr(op_attr::zps, std::vector<int64_t>({2}));
    dequantize_softmax.set_attr(op_attr::qtype, std::string("per_tensor"));
    dequantize_softmax.set_attr(op_attr::axis, (int64_t)0);

    // reshape + transpose for value
    op_t value_reshape {15, op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(op_attr::special_zero, false);
    value_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    op_t value_transpose {16, op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    op_t quantize_value {17, op_kind::Quantize, "quantize_value"};
    op_t dequantize_value {18, op_kind::Dequantize, "dequantize_value"};
    quantize_value.set_attr(op_attr::scales, std::vector<float>({0.12f}));
    quantize_value.set_attr(op_attr::zps, std::vector<int64_t>({2}));
    quantize_value.set_attr(op_attr::qtype, std::string("per_tensor"));
    quantize_value.set_attr(op_attr::axis, (int64_t)0);
    dequantize_value.set_attr(op_attr::scales, std::vector<float>({0.12f}));
    dequantize_value.set_attr(op_attr::zps, std::vector<int64_t>({2}));
    dequantize_value.set_attr(op_attr::qtype, std::string("per_tensor"));
    dequantize_value.set_attr(op_attr::axis, (int64_t)0);

    op_t matmul_v {19, op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    op_t transpose_output {20, op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    op_t reshape_output {21, op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(op_attr::special_zero, false);
    reshape_output.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    query_reshape.add_input(query_gemm);
    query_reshape.add_output(query_reshape_out);
    query_transpose.add_input(query_reshape_out);
    query_transpose.add_output(query_transpose_out);
    key_reshape.add_input(qk_bmm);
    key_reshape.add_output(key_reshape_out);
    key_transpose.add_input(key_reshape_out);
    key_transpose.add_output(key_transpose_out);
    key_transpose2.add_input(key_transpose_out);
    key_transpose2.add_output(key_transpose_out2);

    quantize_query.add_input(query_transpose_out);
    quantize_query.add_output(query_quantize);
    dequantize_query.add_input(query_quantize);
    dequantize_query.add_output(query_dequantize);
    quantize_key.add_input(key_transpose_out2);
    quantize_key.add_output(key_quantize);
    dequantize_key.add_input(key_quantize);
    dequantize_key.add_output(key_dequantize);

    matmul_qk.add_input(query_dequantize);
    matmul_qk.add_input(key_dequantize);
    matmul_qk.add_output(matmul_qk_out);

    fscore_div.add_input(matmul_qk_out);
    fscore_div.add_input(fscore_scale);
    fscore_div.add_output(fscore_div_out);
    fscore_add.add_input(fscore_div_out);
    fscore_add.add_input(attention_mask_flt);
    fscore_add.add_output(fscore_add_out);
    softmax.add_input(fscore_add_out);
    softmax.add_output(softmax_out);
    quantize_softmax.add_input(softmax_out);
    quantize_softmax.add_output(softmax_out_q);
    dequantize_softmax.add_input(softmax_out_q);
    dequantize_softmax.add_output(softmax_out_deq);

    value_reshape.add_input(value_bmm);
    value_reshape.add_output(value_reshape_out);
    value_transpose.add_input(value_reshape_out);
    value_transpose.add_output(value_transpose_out);
    quantize_value.add_input(value_transpose_out);
    quantize_value.add_output(value_quantize);
    dequantize_value.add_input(value_quantize);
    dequantize_value.add_output(value_dequantize);
    matmul_v.add_input(softmax_out_deq);
    matmul_v.add_input(value_dequantize);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);

    agraph->add_op(&quantize_query);
    agraph->add_op(&dequantize_query);
    agraph->add_op(&quantize_key);
    agraph->add_op(&dequantize_key);
    agraph->add_op(&query_reshape);
    agraph->add_op(&query_transpose);
    agraph->add_op(&key_reshape);
    agraph->add_op(&key_transpose);
    agraph->add_op(&key_transpose2);
    agraph->add_op(&matmul_qk);

    agraph->add_op(&fscore_div);
    agraph->add_op(&fscore_add);
    agraph->add_op(&softmax);
    agraph->add_op(&quantize_softmax);
    agraph->add_op(&dequantize_softmax);
    agraph->add_op(&quantize_value);
    agraph->add_op(&dequantize_value);
    agraph->add_op(&value_reshape);
    agraph->add_op(&value_transpose);
    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
}

inline void construct_int8_bf16_MHA(dnnl::graph::impl::graph_t *agraph,
        int batch_size = 1, int seq_len = 384, int num_head = 16,
        int head_dim = 1024) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests;

    // construct a int8 MHA pattern first
    construct_int8_MHA(agraph, batch_size, seq_len, num_head, head_dim);
    agraph->build_graph();

    // change the f32 logical tensor to bf16
    for (auto &op : agraph->get_ops()) {
        for (auto &val : op->get_input_values()) {
            if (val->get_logical_tensor().data_type == impl::data_type::f32)
                val->set_data_type(impl::data_type::bf16);
        }

        for (auto &val : op->get_output_values()) {
            if (val->get_logical_tensor().data_type == impl::data_type::f32)
                val->set_data_type(impl::data_type::bf16);
        }
    }

    // insert bf16->f32 typecase op before quantize and f32->bf16 op after
    // dequantize
    std::vector<std::shared_ptr<impl::op_t>> target_ops;
    for (auto &op : agraph->get_ops()) {
        if (op->get_kind() == impl::op_kind::Quantize
                || op->get_kind() == impl::op_kind::Dequantize) {
            target_ops.emplace_back(op);
        }
    }

    std::vector<std::shared_ptr<impl::op_kind_t>> to_be_inserted;
    size_t new_lt_id_start = 1000;
    for (auto &op : target_ops) {
        // insert bf16->f32 typecase op before quantize
        if (op->get_kind() == impl::op_kind::Quantize) {
            auto bf16_to_f32
                    = agraph->create_op(op_kind::TypeCast, "bf16_to_f32");

            auto in_val = op->get_input_value(0);
            in_val->remove_consumer(*op, 0);
            in_val->add_consumer(*bf16_to_f32, bf16_to_f32->num_inputs());
            bf16_to_f32->add_input(in_val);

            auto new_lt = in_val->get_logical_tensor();
            new_lt.id = new_lt_id_start++;
            new_lt.data_type = impl::data_type::f32;
            auto new_val
                    = std::make_shared<value_t>(*bf16_to_f32, 0, new_lt, false);
            bf16_to_f32->add_output(new_val);

            new_val->add_consumer(*op, 0);
            op->connect_input(0, new_val);
        }

        // insert f32->bf16 op after dequantize
        if (op->get_kind() == impl::op_kind::Dequantize) {
            auto f32_to_bf16
                    = agraph->create_op(op_kind::TypeCast, "f32_to_bf16");

            auto out_val = op->get_output_value(0);
            f32_to_bf16->add_output(out_val);

            auto new_lt = out_val->get_logical_tensor();
            new_lt.id = new_lt_id_start++;
            new_lt.data_type = impl::data_type::f32;
            auto new_val = std::make_shared<value_t>(*op, 0, new_lt, false);
            op->connect_output(0, new_val);

            new_val->add_consumer(*f32_to_bf16, f32_to_bf16->num_inputs());
            f32_to_bf16->add_input(new_val);
        }
    }
}

inline void construct_reshaped_softmax_f32_mha(
        dnnl::graph::impl::graph_t *agraph, int batch_size = 1,
        int seq_len = 384, int num_head = 16, int head_dim = 1024) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests;

    int size_per_head = head_dim / num_head;
    dims MIXED_LAYER_INPUT_SHAPE = {batch_size, seq_len, head_dim};
    dims EXTENDED_ATTENTION_MASK_SHAPE = {batch_size, 1, 1, seq_len};
    dims QKV_RESHAPED_SHAPE = {batch_size, seq_len, num_head, size_per_head};
    dims QKV_TRANSPOSED_SHAPE = {batch_size, num_head, seq_len, size_per_head};
    dims KEY_TRANSPOSED_SHAPE = {batch_size, num_head, size_per_head, seq_len};
    dims MATMUL_QK_OUTPUT_SHAPE = {batch_size, num_head, seq_len, seq_len};
    dims PRE_SOFTMAX_SHAPE = {batch_size * num_head * seq_len, seq_len};
    dims MATMUL_V_OUTPUT_SHAPE = {batch_size, num_head, seq_len, size_per_head};

    dims QKV_TRANSPOSED_ORDER = {0, 2, 1, 3};
    dims KEY_TRANSPOSED_ORDER = {0, 2, 3, 1};

    dims PRE_SOFTMAX_RESHAPE = {batch_size * num_head * seq_len, -1};
    dims POST_SOFTMAX_RESHAPE = {batch_size, num_head, seq_len, seq_len};

    size_t lt_id = 0;

    auto query_gemm = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);
    auto qk_bmm = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);
    auto value_bmm = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);
    auto attention_mask_flt = unit::utils::logical_tensor_init(
            lt_id++, EXTENDED_ATTENTION_MASK_SHAPE, data_type::f32);

    auto query_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);
    auto query_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto key_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);
    auto key_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, KEY_TRANSPOSED_SHAPE, data_type::f32);

    auto matmul_qk_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto matmul_add_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto pre_softmax_out = unit::utils::logical_tensor_init(
            lt_id++, PRE_SOFTMAX_SHAPE, data_type::f32);

    auto softmax_out = unit::utils::logical_tensor_init(
            lt_id++, PRE_SOFTMAX_SHAPE, data_type::f32);

    auto post_softmax_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_QK_OUTPUT_SHAPE, data_type::f32);

    auto value_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);
    auto value_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_TRANSPOSED_SHAPE, data_type::f32);

    auto matmul_v_out = unit::utils::logical_tensor_init(
            lt_id++, MATMUL_V_OUTPUT_SHAPE, data_type::f32);

    auto context_transpose_out = unit::utils::logical_tensor_init(
            lt_id++, QKV_RESHAPED_SHAPE, data_type::f32);

    auto context_reshape_out = unit::utils::logical_tensor_init(
            lt_id++, MIXED_LAYER_INPUT_SHAPE, data_type::f32);

    // reshape + transpose for query + key
    op_t query_reshape {0, op_kind::StaticReshape, "query_reshape"};
    query_reshape.set_attr(op_attr::special_zero, false);
    query_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    op_t query_transpose {1, op_kind::StaticTranspose, "query_transpose"};
    query_transpose.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    op_t key_reshape {2, op_kind::StaticReshape, "key_reshape"};
    key_reshape.set_attr(op_attr::special_zero, false);
    key_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    op_t key_transpose {4, op_kind::StaticTranspose, "key_transpose"};
    key_transpose.set_attr<std::vector<int64_t>>(
            op_attr::order, KEY_TRANSPOSED_ORDER);

    op_t matmul_qk {9, op_kind::MatMul, "matmul_qk"};

    op_t matmul_add {10, op_kind::Add, "fscore_add"};
    matmul_add.set_attr(op_attr::auto_broadcast, std::string("numpy"));

    op_t pre_softmax_reshape {
            11, op_kind::StaticReshape, "pre_softmax_reshape"};
    pre_softmax_reshape.set_attr(op_attr::special_zero, false);
    pre_softmax_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, PRE_SOFTMAX_RESHAPE);
    op_t softmax {12, op_kind::SoftMax, "softmax"};
    softmax.set_attr(op_attr::axis, (int64_t)1);
    op_t post_softmax_reshape {
            13, op_kind::StaticReshape, "post_softmax_reshape"};
    post_softmax_reshape.set_attr(op_attr::special_zero, false);
    post_softmax_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, POST_SOFTMAX_RESHAPE);

    // reshape + transpose for value
    op_t value_reshape {15, op_kind::StaticReshape, "value_reshape"};
    value_reshape.set_attr(op_attr::special_zero, false);
    value_reshape.set_attr<std::vector<int64_t>>(
            op_attr::shape, QKV_RESHAPED_SHAPE);

    op_t value_transpose {16, op_kind::StaticTranspose, "value_transpose"};
    value_transpose.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    op_t matmul_v {19, op_kind::MatMul, "matmul_v"};

    // transpose + reshape before output
    op_t transpose_output {20, op_kind::StaticTranspose, "transpose_output"};
    transpose_output.set_attr<std::vector<int64_t>>(
            op_attr::order, QKV_TRANSPOSED_ORDER);

    op_t reshape_output {21, op_kind::StaticReshape, "reshape_output"};
    reshape_output.set_attr(op_attr::special_zero, false);
    reshape_output.set_attr<std::vector<int64_t>>(
            op_attr::shape, MIXED_LAYER_INPUT_SHAPE);

    query_reshape.add_input(query_gemm);
    query_reshape.add_output(query_reshape_out);
    query_transpose.add_input(query_reshape_out);
    query_transpose.add_output(query_transpose_out);
    key_reshape.add_input(qk_bmm);
    key_reshape.add_output(key_reshape_out);
    key_transpose.add_input(key_reshape_out);
    key_transpose.add_output(key_transpose_out);

    matmul_qk.add_input(query_transpose_out);
    matmul_qk.add_input(key_transpose_out);
    matmul_qk.add_output(matmul_qk_out);

    matmul_add.add_input(matmul_qk_out);
    matmul_add.add_input(attention_mask_flt);
    matmul_add.add_output(matmul_add_out);

    pre_softmax_reshape.add_input(matmul_add_out);
    pre_softmax_reshape.add_output(pre_softmax_out);
    softmax.add_input(pre_softmax_out);
    softmax.add_output(softmax_out);
    post_softmax_reshape.add_input(softmax_out);
    post_softmax_reshape.add_output(post_softmax_out);

    value_reshape.add_input(value_bmm);
    value_reshape.add_output(value_reshape_out);
    value_transpose.add_input(value_reshape_out);
    value_transpose.add_output(value_transpose_out);
    matmul_v.add_input(post_softmax_out);
    matmul_v.add_input(value_transpose_out);
    matmul_v.add_output(matmul_v_out);

    transpose_output.add_input(matmul_v_out);
    transpose_output.add_output(context_transpose_out);
    reshape_output.add_input(context_transpose_out);
    reshape_output.add_output(context_reshape_out);

    agraph->add_op(&query_reshape);
    agraph->add_op(&query_transpose);
    agraph->add_op(&key_reshape);
    agraph->add_op(&key_transpose);
    agraph->add_op(&matmul_qk);

    agraph->add_op(&matmul_add);
    agraph->add_op(&pre_softmax_reshape);
    agraph->add_op(&softmax);
    agraph->add_op(&post_softmax_reshape);
    agraph->add_op(&value_reshape);
    agraph->add_op(&value_transpose);
    agraph->add_op(&matmul_v);
    agraph->add_op(&transpose_output);
    agraph->add_op(&reshape_output);
}

inline void construct_chained_relu(dnnl::graph::impl::graph_t *agraph) {
    using namespace dnnl::graph::impl;
    using namespace dnnl::graph::tests;

    size_t lt_id = 0;

    dims shape = {1, 4, 16, 16};

    auto lt0 = unit::utils::logical_tensor_init(lt_id++, shape, data_type::f32);
    auto lt1 = unit::utils::logical_tensor_init(lt_id++, shape, data_type::f32);
    auto lt2 = unit::utils::logical_tensor_init(lt_id++, shape, data_type::f32);
    auto lt3 = unit::utils::logical_tensor_init(lt_id++, shape, data_type::f32);

    impl::op_t relu0(lt_id++, impl::op_kind::ReLU, "relu");
    impl::op_t relu1(lt_id++, impl::op_kind::ReLU, "relu");
    impl::op_t relu2(lt_id++, impl::op_kind::ReLU, "relu");

    relu0.add_input(lt0);
    relu0.add_output(lt1);
    relu1.add_input(lt1);
    relu1.add_output(lt2);
    relu2.add_input(lt2);
    relu2.add_output(lt3);

    agraph->add_op(&relu0);
    agraph->add_op(&relu1);
    agraph->add_op(&relu2);
}

class id_generator {
public:
    id_generator() : id_(0) {};
    size_t get_id() { return id_++; }

private:
    size_t id_;
};

inline impl::logical_tensor_t create_convolution(id_generator &id_gen,
        impl::graph_t &agraph, const impl::logical_tensor_t &src, int64_t ic,
        int64_t ks, int64_t oc, int64_t groups, const impl::dims &strides,
        const impl::dims &dilations, const impl::dims &pads_begin,
        const impl::dims &pads_end, const std::string &data_format,
        const std::string &filter_format, bool with_bias = false,
        bool with_bn = false, float epsilon = 1e-6f, bool with_relu = false,
        bool use_biasadd = false) {
    impl::op_t conv(id_gen.get_id(), impl::op_kind::Convolution, "conv");
    conv.set_attr<int64_t>(impl::op_attr::groups, groups);
    conv.set_attr<impl::dims>(impl::op_attr::strides, strides);
    conv.set_attr<impl::dims>(impl::op_attr::dilations, dilations);
    conv.set_attr<impl::dims>(impl::op_attr::pads_begin, pads_begin);
    conv.set_attr<impl::dims>(impl::op_attr::pads_end, pads_end);
    conv.set_attr<std::string>(impl::op_attr::data_format, data_format);
    conv.set_attr<std::string>(impl::op_attr::filter_format, filter_format);

    impl::dims wei_shape = (filter_format == "OIX")
            ? impl::dims {oc, ic / groups, ks, ks}
            : impl::dims {ks, ks, ic / groups, oc};

    auto wei = utils::logical_tensor_init(
            id_gen.get_id(), wei_shape, src.data_type);
    wei.property = impl::property_type::constant;
    auto dst = utils::logical_tensor_init(id_gen.get_id(), src.data_type);

    conv.add_input(src);
    conv.add_input(wei);
    if (with_bias && !use_biasadd) {
        auto bias = utils::logical_tensor_init(
                id_gen.get_id(), impl::dims {oc}, src.data_type);
        bias.property = impl::property_type::constant;
        conv.add_input(bias);
    }
    conv.add_output(dst);
    agraph.add_op(&conv);

    if (with_bias && use_biasadd) {
        impl::op_t biasadd_op(
                id_gen.get_id(), impl::op_kind::BiasAdd, "biasadd");
        biasadd_op.set_attr<std::string>(
                impl::op_attr::data_format, data_format);

        auto biasadd_src = dst;
        auto bias = utils::logical_tensor_init(
                id_gen.get_id(), impl::dims {oc}, biasadd_src.data_type);
        bias.property = impl::property_type::constant;

        dst = utils::logical_tensor_init(
                id_gen.get_id(), biasadd_src.data_type);

        biasadd_op.add_input(biasadd_src);
        biasadd_op.add_input(bias);
        biasadd_op.add_output(dst);

        agraph.add_op(&biasadd_op);
    }

    if (with_bn) {
        impl::op_t bn_op(
                id_gen.get_id(), impl::op_kind::BatchNormInference, "bn");
        bn_op.set_attr<std::string>(impl::op_attr::data_format, data_format);
        bn_op.set_attr<float>(impl::op_attr::epsilon, epsilon);

        int64_t bn_ic = oc;
        auto bn_src = dst;
        auto scale = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, src.data_type);
        auto shift = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, src.data_type);
        auto mean = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, src.data_type);
        auto var = utils::logical_tensor_init(
                id_gen.get_id(), {bn_ic}, src.data_type);

        scale.property = impl::property_type::constant;
        shift.property = impl::property_type::constant;
        mean.property = impl::property_type::constant;
        var.property = impl::property_type::constant;

        dst = utils::logical_tensor_init(id_gen.get_id(), src.data_type);

        bn_op.add_input(bn_src);
        bn_op.add_input(scale);
        bn_op.add_input(shift);
        bn_op.add_input(mean);
        bn_op.add_input(var);
        bn_op.add_output(dst);

        agraph.add_op(&bn_op);
    }

    if (with_relu) {
        impl::op_t relu_op(id_gen.get_id(), impl::op_kind::ReLU, "relu");
        auto relu_src = dst;
        dst = utils::logical_tensor_init(id_gen.get_id(), src.data_type);
        relu_op.add_input(relu_src);
        relu_op.add_output(dst);
        agraph.add_op(&relu_op);
    }

    return dst;
}

inline impl::logical_tensor_t create_add(id_generator &id_gen,
        impl::graph_t &agraph, const impl::logical_tensor_t &src0,
        const impl::logical_tensor_t &src1) {
    impl::op_t add(id_gen.get_id(), impl::op_kind::Add, "add");
    auto dst = utils::logical_tensor_init(id_gen.get_id(), src0.data_type);
    add.add_input(src0);
    add.add_input(src1);
    add.add_output(dst);
    agraph.add_op(&add);
    return dst;
}

inline impl::logical_tensor_t create_relu(id_generator &id_gen,
        impl::graph_t &agraph, const impl::logical_tensor_t &src) {
    impl::op_t relu_op(id_gen.get_id(), impl::op_kind::ReLU, "relu");
    auto dst = utils::logical_tensor_init(id_gen.get_id(), src.data_type);
    relu_op.add_input(src);
    relu_op.add_output(dst);
    agraph.add_op(&relu_op);
    return dst;
}

inline impl::logical_tensor_t create_dequantize(id_generator &id_gen,
        impl::graph_t &agraph, const impl::logical_tensor_t &src,
        const std::string &qtype, const std::vector<int64_t> &zps,
        const std::vector<float> &scales, int64_t axis) {
    impl::op_t dq_op(id_gen.get_id(), impl::op_kind::Dequantize, "dequantize");
    dq_op.set_attr<std::string>(impl::op_attr::qtype, qtype);
    dq_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, zps);
    dq_op.set_attr<std::vector<float>>(impl::op_attr::scales, scales);
    dq_op.set_attr<int64_t>(impl::op_attr::axis, axis);

    auto dst
            = utils::logical_tensor_init(id_gen.get_id(), impl::data_type::f32);
    dq_op.add_input(src);
    dq_op.add_output(dst);
    agraph.add_op(&dq_op);
    return dst;
}

inline impl::logical_tensor_t create_quantize(id_generator &id_gen,
        impl::graph_t &agraph, const impl::logical_tensor_t &src,
        impl::data_type_t dst_dtype, const std::string &qtype,
        const std::vector<int64_t> &zps, const std::vector<float> &scales,
        int64_t axis) {
    impl::op_t q_op(id_gen.get_id(), impl::op_kind::Quantize, "quantize");
    q_op.set_attr<std::string>(impl::op_attr::qtype, qtype);
    q_op.set_attr<std::vector<int64_t>>(impl::op_attr::zps, zps);
    q_op.set_attr<std::vector<float>>(impl::op_attr::scales, scales);
    q_op.set_attr<int64_t>(impl::op_attr::axis, axis);

    auto dst = utils::logical_tensor_init(id_gen.get_id(), dst_dtype);
    q_op.add_input(src);
    q_op.add_output(dst);
    agraph.add_op(&q_op);
    return dst;
}

inline impl::logical_tensor_t create_int8_convolution(id_generator &id_gen,
        impl::graph_t &agraph, const impl::logical_tensor_t &src, int64_t ic,
        int64_t ks, int64_t oc, int64_t groups, const impl::dims &strides,
        const impl::dims &dilations, const impl::dims &pads_begin,
        const impl::dims &pads_end, const std::string &data_format,
        const std::string &filter_format, bool with_bias, bool with_bn,
        float epsilon, bool with_relu,
        // args for int8 conv
        float src_scale, int64_t src_zp, float dst_scale, int64_t dst_zp,
        const std::vector<float> &wei_scales, impl::data_type_t dst_dtype,
        bool is_quantize_dst = true, bool use_biasadd = false,
        bool is_quantize_wei = false) {
    assertm(!with_bn, "int8 conv not support bn now");

    auto dq_src = create_dequantize(
            id_gen, agraph, src, "per_tensor", {src_zp}, {src_scale}, 0);

    impl::op_t conv(id_gen.get_id(), impl::op_kind::Convolution, "conv");
    conv.set_attr<int64_t>(impl::op_attr::groups, groups);
    conv.set_attr<impl::dims>(impl::op_attr::strides, strides);
    conv.set_attr<impl::dims>(impl::op_attr::dilations, dilations);
    conv.set_attr<impl::dims>(impl::op_attr::pads_begin, pads_begin);
    conv.set_attr<impl::dims>(impl::op_attr::pads_end, pads_end);
    conv.set_attr<std::string>(impl::op_attr::data_format, data_format);
    conv.set_attr<std::string>(impl::op_attr::filter_format, filter_format);

    impl::dims wei_shape = (filter_format == "OIX")
            ? impl::dims {oc, ic / groups, ks, ks}
            : impl::dims {ks, ks, ic / groups, oc};

    impl::logical_tensor_t int8_wei;
    if (is_quantize_wei) {
        auto f32_wei = utils::logical_tensor_init(
                id_gen.get_id(), wei_shape, impl::data_type::f32);
        f32_wei.property = impl::property_type::constant;
        int8_wei = create_quantize(id_gen, agraph, f32_wei, impl::data_type::s8,
                "per_channel", std::vector<int64_t>(oc, 0), wei_scales,
                (filter_format == "OIX") ? 0 : 3);
    } else {
        int8_wei = utils::logical_tensor_init(
                id_gen.get_id(), wei_shape, impl::data_type::s8);
        int8_wei.property = impl::property_type::constant;
    }

    auto dq_wei = create_dequantize(id_gen, agraph, int8_wei, "per_channel",
            std::vector<int64_t>(oc, 0), wei_scales,
            (filter_format == "OIX") ? 0 : 3);

    auto dst = utils::logical_tensor_init(id_gen.get_id(), dq_src.data_type);

    conv.add_input(dq_src);
    conv.add_input(dq_wei);
    if (with_bias && !use_biasadd) {
        auto bias = utils::logical_tensor_init(
                id_gen.get_id(), impl::dims {oc}, dq_src.data_type);
        bias.property = impl::property_type::constant;
        conv.add_input(bias);
    }
    conv.add_output(dst);
    agraph.add_op(&conv);

    if (with_bias && use_biasadd) {
        impl::op_t biasadd_op(
                id_gen.get_id(), impl::op_kind::BiasAdd, "biasadd");
        biasadd_op.set_attr<std::string>(
                impl::op_attr::data_format, data_format);

        auto biasadd_src = dst;
        auto bias = utils::logical_tensor_init(
                id_gen.get_id(), impl::dims {oc}, biasadd_src.data_type);
        bias.property = impl::property_type::constant;

        dst = utils::logical_tensor_init(
                id_gen.get_id(), biasadd_src.data_type);

        biasadd_op.add_input(biasadd_src);
        biasadd_op.add_input(bias);
        biasadd_op.add_output(dst);

        agraph.add_op(&biasadd_op);
    }

    if (with_relu) {
        impl::op_t relu_op(id_gen.get_id(), impl::op_kind::ReLU, "relu");
        auto relu_src = dst;
        dst = utils::logical_tensor_init(id_gen.get_id(), dq_src.data_type);
        relu_op.add_input(relu_src);
        relu_op.add_output(dst);
        agraph.add_op(&relu_op);
    }

    if (is_quantize_dst) {
        dst = create_quantize(id_gen, agraph, dst, dst_dtype, "per_tensor",
                {dst_zp}, {dst_scale}, 0);
    }

    return dst;
}

inline void construct_convolutional_bottleneck_resblock(
        dnnl::graph::impl::graph_t *agraph, id_generator &id_gen) {
    auto input = utils::logical_tensor_init(
            id_gen.get_id(), {8, 64, 56, 56}, impl::data_type::f32);
    auto conv0 = create_convolution(id_gen, *agraph, input, 64, 1, 64, 1,
            {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            true);
    auto conv1 = create_convolution(id_gen, *agraph, conv0, 64, 3, 64, 1,
            {1, 1}, {1, 1}, {1, 1}, {1, 1}, "NCX", "OIX", true, false, 1e-6f,
            true);
    auto conv2 = create_convolution(id_gen, *agraph, input, 64, 1, 256, 1,
            {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            false);
    auto conv3 = create_convolution(id_gen, *agraph, conv1, 64, 1, 256, 1,
            {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            false);
    auto add = create_add(id_gen, *agraph, conv3, conv2);
    auto relu3 = create_relu(id_gen, *agraph, add);
    (void)(relu3);
}

inline void construct_int8_conv_bias_relu_conv_bias_relu_block(
        dnnl::graph::impl::graph_t *agraph, id_generator &id_gen) {
    int64_t ic = 8, oc = 8, ks = 1;
    std::vector<int64_t> src_shape {1, ic, 12, 12};

    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0, zp_out = 78;
    std::vector<float> scale_wei(oc, 1 / 127.f);

    auto src = utils::logical_tensor_init(
            id_gen.get_id(), src_shape, impl::data_type::u8);

    auto int8_conv0 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);
    auto int8_conv1 = create_int8_convolution(id_gen, *agraph, int8_conv0, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);

    (void)(int8_conv1);
}

inline void construct_int8_identical_bottleneck_resblock(
        dnnl::graph::impl::graph_t *agraph, id_generator &id_gen) {
    int64_t ic = 8, oc = 8, ks = 1;
    std::vector<int64_t> src_shape {1, ic, 12, 12};

    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0, zp_out = 78;
    std::vector<float> scale_wei(oc, 1 / 127.f);

    auto src = utils::logical_tensor_init(
            id_gen.get_id(), src_shape, impl::data_type::u8);

    auto int8_conv0 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);
    auto int8_conv1 = create_int8_convolution(id_gen, *agraph, int8_conv0, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);

    auto int8_conv2 = create_int8_convolution(id_gen, *agraph, int8_conv1, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, /*no relu*/ false, scale_src, zp_src, scale_out,
            zp_out, scale_wei, impl::data_type::u8,
            /*not quantize dst*/ false);
    auto dq3 = create_dequantize(
            id_gen, *agraph, src, "per_tensor", {zp_src}, {scale_src}, 0);
    auto add0 = create_add(id_gen, *agraph, int8_conv2, dq3);
    auto relu0 = create_relu(id_gen, *agraph, add0);
    auto q2 = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
            "per_tensor", std::vector<int64_t> {zp_out},
            std::vector<float> {scale_out}, 0);
    (void)(q2);
}

inline void construct_int8_convolutional_bottleneck_resblock(
        dnnl::graph::impl::graph_t *agraph, id_generator &id_gen) {
    int64_t ic = 8, oc = 8, ks = 1;
    std::vector<int64_t> src_shape {1, ic, 12, 12};

    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0,
            zp_out
            = agraph->get_engine_kind() == impl::engine_kind::gpu ? 0 : 78;
    std::vector<float> scale_wei(oc, 1 / 127.f);

    auto src = utils::logical_tensor_init(
            id_gen.get_id(), src_shape, impl::data_type::u8);

    auto int8_conv0 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);
    auto int8_conv1 = create_int8_convolution(id_gen, *agraph, int8_conv0, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);

    auto int8_conv2 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            /*no relu*/ false, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);

    auto int8_conv3 = create_int8_convolution(id_gen, *agraph, int8_conv1, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, /*no relu*/ false, scale_src, zp_src, scale_out,
            zp_out, scale_wei, impl::data_type::u8,
            /*not quantize dst*/ false);
    auto dq3 = create_dequantize(id_gen, *agraph, int8_conv2, "per_tensor",
            {zp_src}, {scale_src}, 0);
    auto add0 = create_add(id_gen, *agraph, int8_conv3, dq3);
    auto relu0 = create_relu(id_gen, *agraph, add0);
    auto q2 = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
            "per_tensor", std::vector<int64_t> {zp_out},
            std::vector<float> {scale_out}, 0);
    (void)(q2);
}

inline void construct_int8_resnet50_stage2_block(
        dnnl::graph::impl::graph_t *agraph, id_generator &id_gen,
        size_t three_conv_block_num = 2) {
    int64_t ic = 8, oc = 8, ks = 1;
    std::vector<int64_t> src_shape {1, ic, 12, 12};

    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0, zp_out = 78;
    std::vector<float> scale_wei(oc, 1 / 127.f);

    auto src = utils::logical_tensor_init(
            id_gen.get_id(), src_shape, impl::data_type::u8);

    // 4-conv block
    auto int8_conv0 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);
    auto int8_conv1 = create_int8_convolution(id_gen, *agraph, int8_conv0, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);

    auto int8_conv2 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            /*no relu*/ false, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);

    auto int8_conv3 = create_int8_convolution(id_gen, *agraph, int8_conv1, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, /*no relu*/ false, scale_src, zp_src, scale_out,
            zp_out, scale_wei, impl::data_type::u8,
            /*not quantize dst*/ false);
    auto dq3 = create_dequantize(id_gen, *agraph, int8_conv2, "per_tensor",
            {zp_src}, {scale_src}, 0);
    auto add0 = create_add(id_gen, *agraph, int8_conv3, dq3);
    auto relu0 = create_relu(id_gen, *agraph, add0);
    auto q2 = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
            "per_tensor", std::vector<int64_t> {zp_out},
            std::vector<float> {scale_out}, 0);

    // Two 3-conv block
    impl::logical_tensor_t tmp = q2;
    for (size_t i = 0; i < three_conv_block_num; i++) {
        auto int8_conv0 = create_int8_convolution(id_gen, *agraph, tmp, ic, ks,
                oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
                false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out,
                scale_wei, impl::data_type::u8);
        auto int8_conv1 = create_int8_convolution(id_gen, *agraph, int8_conv0,
                ic, ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX",
                true, false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out,
                scale_wei, impl::data_type::u8);

        auto int8_conv2 = create_int8_convolution(id_gen, *agraph, int8_conv1,
                ic, ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX",
                true, false, 1e-6f, /*no relu*/ false, scale_src, zp_src,
                scale_out, zp_out, scale_wei, impl::data_type::u8,
                /*not quantize dst*/ false);
        auto dq3 = create_dequantize(
                id_gen, *agraph, tmp, "per_tensor", {zp_src}, {scale_src}, 0);
        auto add0 = create_add(id_gen, *agraph, int8_conv2, dq3);
        auto relu0 = create_relu(id_gen, *agraph, add0);
        tmp = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
                "per_tensor", std::vector<int64_t> {zp_out},
                std::vector<float> {scale_out}, 0);
    }
}

inline void construct_f32_resnet50_stage2_block(
        dnnl::graph::impl::graph_t *agraph, id_generator &id_gen,
        size_t three_conv_block_num = 2, bool use_biasadd = false) {
    int64_t ic = 8, oc = 8, ks = 1;
    std::vector<int64_t> src_shape {1, ic, 12, 12};

    auto src = utils::logical_tensor_init(
            id_gen.get_id(), src_shape, impl::data_type::f32);

    // 4-conv block
    auto conv0 = create_convolution(id_gen, *agraph, src, ic, ks, oc, 1, {1, 1},
            {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", use_biasadd, !use_biasadd,
            1e-6f, true, use_biasadd);
    auto conv1 = create_convolution(id_gen, *agraph, conv0, ic, ks, oc, 1,
            {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", use_biasadd,
            !use_biasadd, 1e-6f, true, use_biasadd);

    auto conv2 = create_convolution(id_gen, *agraph, src, ic, ks, oc, 1, {1, 1},
            {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", use_biasadd, !use_biasadd,
            1e-6f,
            /*no relu*/ false, use_biasadd);

    auto conv3 = create_convolution(id_gen, *agraph, conv1, ic, ks, oc, 1,
            {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", use_biasadd,
            !use_biasadd, 1e-6f,
            /*no relu*/ false, use_biasadd);
    auto add0 = create_add(id_gen, *agraph, conv3, conv2);
    auto relu0 = create_relu(id_gen, *agraph, add0);

    // Two 3-conv block
    impl::logical_tensor_t tmp = relu0;
    for (size_t i = 0; i < three_conv_block_num; i++) {
        auto conv0 = create_convolution(id_gen, *agraph, tmp, ic, ks, oc, 1,
                {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", use_biasadd,
                !use_biasadd, 1e-6f, true, use_biasadd);
        auto conv1 = create_convolution(id_gen, *agraph, conv0, ic, ks, oc, 1,
                {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", use_biasadd,
                !use_biasadd, 1e-6f, true, use_biasadd);
        auto conv2 = create_convolution(id_gen, *agraph, conv1, ic, ks, oc, 1,
                {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", use_biasadd,
                !use_biasadd, 1e-6f, /*no relu*/ false, use_biasadd);
        auto add0 = create_add(id_gen, *agraph, conv2, tmp);
        tmp = create_relu(id_gen, *agraph, add0);
    }
}

inline void construct_itex_int8_resnet50_stage2_block(
        dnnl::graph::impl::graph_t *agraph, id_generator &id_gen,
        size_t three_conv_block_num = 2) {
    int64_t ic = 8, oc = 8, ks = 1;
    std::vector<int64_t> src_shape {1, ic, 12, 12};

    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0, zp_out = 78;
    std::vector<float> scale_wei(oc, 1 / 127.f);

    auto src = utils::logical_tensor_init(
            id_gen.get_id(), src_shape, impl::data_type::u8);

    // 4-conv block
    auto int8_conv0 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8, true, true, true);
    auto int8_conv1 = create_int8_convolution(id_gen, *agraph, int8_conv0, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8, true, true, true);

    auto int8_conv2 = create_int8_convolution(id_gen, *agraph, int8_conv1, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f,
            /*no relu*/ false, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8, true, true, true);

    auto int8_conv3 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            /*no relu*/ false, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8,
            /*not quantize dst*/ false, true, true);
    auto dq3 = create_dequantize(id_gen, *agraph, int8_conv2, "per_tensor",
            {zp_src}, {scale_src}, 0);
    auto add0 = create_add(id_gen, *agraph, int8_conv3, dq3);
    auto relu0 = create_relu(id_gen, *agraph, add0);
    auto q2 = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
            "per_tensor", std::vector<int64_t> {zp_out},
            std::vector<float> {scale_out}, 0);

    // Two 3-conv block
    impl::logical_tensor_t tmp = q2;
    for (size_t i = 0; i < three_conv_block_num; i++) {
        auto int8_conv0 = create_int8_convolution(id_gen, *agraph, tmp, ic, ks,
                oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
                false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out,
                scale_wei, impl::data_type::u8, true, true, true);
        auto int8_conv1 = create_int8_convolution(id_gen, *agraph, int8_conv0,
                ic, ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX",
                true, false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out,
                scale_wei, impl::data_type::u8, true, true, true);

        auto int8_conv2 = create_int8_convolution(id_gen, *agraph, int8_conv1,
                ic, ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX",
                true, false, 1e-6f, /*no relu*/ false, scale_src, zp_src,
                scale_out, zp_out, scale_wei, impl::data_type::u8,
                /*not quantize dst*/ false, true, true);
        auto dq3 = create_dequantize(
                id_gen, *agraph, tmp, "per_tensor", {zp_src}, {scale_src}, 0);
        auto add0 = create_add(id_gen, *agraph, int8_conv2, dq3);
        auto relu0 = create_relu(id_gen, *agraph, add0);
        tmp = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
                "per_tensor", std::vector<int64_t> {zp_out},
                std::vector<float> {scale_out}, 0);
    }
}

inline void construct_int8_resnext101_stage3_block(
        dnnl::graph::impl::graph_t *agraph, id_generator &id_gen,
        size_t three_conv_block_num = 22) {
    int64_t ic = 8, oc = 8, ks = 1;
    std::vector<int64_t> src_shape {1, ic, 12, 12};

    float scale_src = 1 / 255.f, scale_out = 1;
    int64_t zp_src = 0, zp_out = 78;
    std::vector<float> scale_wei(oc, 1 / 127.f);

    auto src = utils::logical_tensor_init(
            id_gen.get_id(), src_shape, impl::data_type::u8);

    // 4-conv block
    auto int8_conv0 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);
    auto int8_conv1 = create_int8_convolution(id_gen, *agraph, int8_conv0, ic,
            ks, oc, 4, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);

    auto int8_conv2 = create_int8_convolution(id_gen, *agraph, src, ic, ks, oc,
            1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true, false, 1e-6f,
            /*no relu*/ false, scale_src, zp_src, scale_out, zp_out, scale_wei,
            impl::data_type::u8);

    auto int8_conv3 = create_int8_convolution(id_gen, *agraph, int8_conv1, ic,
            ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
            false, 1e-6f, /*no relu*/ false, scale_src, zp_src, scale_out,
            zp_out, scale_wei, impl::data_type::u8,
            /*not quantize dst*/ false);
    auto dq3 = create_dequantize(id_gen, *agraph, int8_conv2, "per_tensor",
            {zp_src}, {scale_src}, 0);
    auto add0 = create_add(id_gen, *agraph, int8_conv3, dq3);
    auto relu0 = create_relu(id_gen, *agraph, add0);
    auto q2 = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
            "per_tensor", std::vector<int64_t> {zp_out},
            std::vector<float> {scale_out}, 0);

    // 22 3-conv block
    impl::logical_tensor_t tmp = q2;
    for (size_t i = 0; i < three_conv_block_num; i++) {
        auto int8_conv0 = create_int8_convolution(id_gen, *agraph, tmp, ic, ks,
                oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX", true,
                false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out,
                scale_wei, impl::data_type::u8);
        auto int8_conv1 = create_int8_convolution(id_gen, *agraph, int8_conv0,
                ic, ks, oc, 4, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX",
                true, false, 1e-6f, true, scale_src, zp_src, scale_out, zp_out,
                scale_wei, impl::data_type::u8);

        auto int8_conv2 = create_int8_convolution(id_gen, *agraph, int8_conv1,
                ic, ks, oc, 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, "NCX", "OIX",
                true, false, 1e-6f, /*no relu*/ false, scale_src, zp_src,
                scale_out, zp_out, scale_wei, impl::data_type::u8,
                /*not quantize dst*/ false);
        auto dq3 = create_dequantize(
                id_gen, *agraph, tmp, "per_tensor", {zp_src}, {scale_src}, 0);
        auto add0 = create_add(id_gen, *agraph, int8_conv2, dq3);
        auto relu0 = create_relu(id_gen, *agraph, add0);
        tmp = create_quantize(id_gen, *agraph, relu0, impl::data_type::u8,
                "per_tensor", std::vector<int64_t> {zp_out},
                std::vector<float> {scale_out}, 0);
    }
}

} // namespace utils
} // namespace unit
} // namespace tests
} // namespace graph
} // namespace dnnl

#endif

/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/op_schema.hpp"

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
            return; \
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
        const std::map<std::string, bool> &attrs_data) {
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
    op_.set_attr("auto_broadcast", no_broadcast_attr_val);

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
        op_.set_attr("auto_broadcast", with_broadcast_attr_val);
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

static inline void set_conv_common_attr(dnnl::graph::impl::op_t &op_,
        std::vector<int64_t> &strides, std::vector<int64_t> &pads_begin,
        std::vector<int64_t> &pads_end, std::vector<int64_t> &dilations,
        std::string auto_pad, std::string data_format,
        std::string filter_format, int64_t groups) {
    op_.set_attr("strides", strides);
    op_.set_attr("pads_begin", pads_begin);
    op_.set_attr("pads_end", pads_end);
    op_.set_attr("dilations", dilations);
    op_.set_attr("auto_pad", auto_pad);
    op_.set_attr("data_format", data_format);
    op_.set_attr("filter_format", filter_format);
    op_.set_attr("groups", groups);
}

static inline void set_convtranspose_common_attr(dnnl::graph::impl::op_t &op_,
        std::vector<int64_t> &strides, std::vector<int64_t> &pads_begin,
        std::vector<int64_t> &pads_end, std::vector<int64_t> &dilations,
        std::string auto_pad, std::string data_format,
        std::string filter_format, int64_t groups,
        std::vector<int64_t> &output_padding) {
    op_.set_attr("strides", strides);
    op_.set_attr("pads_begin", pads_begin);
    op_.set_attr("pads_end", pads_end);
    op_.set_attr("dilations", dilations);
    op_.set_attr("auto_pad", auto_pad);
    op_.set_attr("data_format", data_format);
    op_.set_attr("filter_format", filter_format);
    op_.set_attr("groups", groups);
    op_.set_attr("output_padding", output_padding);
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
            = conv_op.get_attr<std::vector<int64_t>>("pads_begin");
    auto infered_pads_end = conv_op.get_attr<std::vector<int64_t>>("pads_end");
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

    auto infered_pads_begin = op_.get_attr<std::vector<int64_t>>("pads_begin");
    auto infered_pads_end = op_.get_attr<std::vector<int64_t>>("pads_end");
    std::vector<int64_t> expected_pads;
    expected_pads.assign(in_data.size() - 2, 0);
    EXPECT_EQ(infered_pads_begin, expected_pads);
    EXPECT_EQ(infered_pads_end, expected_pads);
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

    auto infered_pads_begin = op_.get_attr<std::vector<int64_t>>("pads_begin");
    auto infered_pads_end = op_.get_attr<std::vector<int64_t>>("pads_end");
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

    auto infered_pads_begin = op_.get_attr<std::vector<int64_t>>("pads_begin");
    auto infered_pads_end = op_.get_attr<std::vector<int64_t>>("pads_end");
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
    op_.set_attr("output_padding", output_padding);

    logical_tensor_t lt_data = logical_tensor_init(0, in_data, data_type::f32);
    logical_tensor_t lt_weight
            = logical_tensor_init(1, in_weight, data_type::f32);
    logical_tensor_t lt_output_shape
            = logical_tensor_init(2, in_output_shape, data_type::f32);
    std::vector<logical_tensor_t *> in {&lt_data, &lt_weight, &lt_output_shape};
    logical_tensor_t lt_out = logical_tensor_init(3, data_type::f32);
    std::vector<logical_tensor_t *> out {&lt_out};

    // shape inference without explicitly setting auto_broadcast
    // should be enabled by default
    op_schema_->shape_infer(&op_, in, out);
    const std::vector<int64_t> infered_out_shape
            = logical_tensor_wrapper_t(lt_out).vdims();
    EXPECT_EQ(infered_out_shape, expected_out_shape);

    auto infered_pads_begin = op_.get_attr<std::vector<int64_t>>("pads_begin");
    auto infered_pads_end = op_.get_attr<std::vector<int64_t>>("pads_end");
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

} // namespace utils
} // namespace unit
} // namespace tests
} // namespace graph
} // namespace dnnl

#endif

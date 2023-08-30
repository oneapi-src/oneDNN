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

#include <fstream>
#include <memory>
#include <numeric>
#include <vector>
#include "quantize_op.hpp"
#include <compiler/ir/graph/graph.hpp>
#include <util/math_utils.hpp>
#include <util/utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace quantize {

static void common_query_function(sc_op *node,
        const sc_data_format_t &out_format,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    out_formats.push_back({out_format});
    node->format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

quantize_infos_t get_quantize_info_from_attrs(const any_map_t &attrs) {
    quantize_infos_t infos;
    infos.dtype_
            = attrs.get_or_else(attr_keys::quan_dtype, sc_data_type_t::u8(1));
    infos.scales_
            = attrs.get_or_else(attr_keys::scales, std::vector<float> {1.f});
    infos.zero_points_
            = attrs.get_or_else(attr_keys::zero_points, std::vector<int> {0});
    infos.per_channel_ = attrs.get_or_else(attr_keys::per_channel, false)
            || infos.scales_.size() > 1;
    infos.channel_axis_ = attrs.get_or_else(attr_keys::channel_axis, 0);
    infos.asymmetric_ = attrs.get_or_else(attr_keys::asymmetric, true);
    assert(utils::is_one_of(infos.dtype_, datatypes::f32, datatypes::bf16,
                   datatypes::u8, datatypes::s8)
            && ((infos.per_channel_ && !infos.scales_.empty())
                    || (!infos.per_channel_ && infos.scales_.size() == 1
                            && infos.zero_points_.size() <= 1))
            && (infos.asymmetric_
                    || (!infos.asymmetric_
                            && (infos.zero_points_.empty()
                                    || (infos.zero_points_.size() == 1
                                            && infos.zero_points_[0] == 0)))));
    return infos;
}

quantize_op_t::quantize_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    assert(ins.size() == 1);
    assert(ins[0]->details_.dtype_.type_code_ == sc_data_etype::F32
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::S32);
    info_.inputs_ = ins;
    if (outs.empty()) {
        // fixme: correctly infer the shape for broadcast
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_ = ins[0]->details_;
        assert(attrs.has_key(attr_keys::quan_dtype));
        info_.outputs_[0]->details_.dtype_
                = attrs.get<sc_data_type_t>(attr_keys::quan_dtype);
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "quantize";
}

quantize_op_t::quantize_op_t(
        const std::vector<graph_tensor_ptr> &ins, const any_map_t &attrs)
    : quantize_op_t(ins, std::vector<graph_tensor_ptr>(), attrs) {}

void quantize_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    const auto qinfos = get_quantize_info_from_attrs(attrs_);
    assert(utils::is_one_of(qinfos.dtype_, datatypes::u8, datatypes::s8));
    auto scales = qinfos.scales_;
    scales = math_utils::vector_rcp(scales);
    std::shared_ptr<static_data_t> scales_ptr
            = std::make_shared<static_data_t>(scales);

    sc_dims plain_dims = {1};
    if (scales.size() > 1) {
        plain_dims.resize(inputs[0]->details_.get_plain_dims().size(), 1);
        plain_dims[qinfos.channel_axis_] = static_cast<int>(scales.size());
    }
    auto quantize_const_scales = graph->make("constant", {}, {},
            {{"values", scales_ptr}, {"dtype", datatypes::f32},
                    {"plain_dims", plain_dims}, {"format", sc_data_format_t()},
                    {"all_positive", true}});
    auto div_scale = graph->make("mul",
            {inputs[0], quantize_const_scales->get_outputs()[0]}, {}, {});

    auto zeropoints = qinfos.zero_points_;
    if (!zeropoints.empty()) {
        int zp_all_zero = std::all_of(zeropoints.begin(), zeropoints.end(),
                [](int i) { return i == 0; });
        if (!zp_all_zero) {
            std::vector<float> zeropoints_f32(
                    zeropoints.begin(), zeropoints.end());
            std::shared_ptr<static_data_t> zeropoints_ptr
                    = std::make_shared<static_data_t>(zeropoints_f32);
            auto quantize_const_zeropoints = graph->make("constant", {}, {},
                    {{"values", zeropoints_ptr}, {"dtype", datatypes::f32},
                            {"plain_dims",
                                    sc_dims {static_cast<sc_dim>(
                                            zeropoints_f32.size())}},
                            {"format", sc_data_format_t()}});

            div_scale = graph->make("add",
                    {div_scale->get_outputs()[0],
                            quantize_const_zeropoints->get_outputs()[0]},
                    {}, {});
        }
    }
    // maybe we need clip op in future
#if 0
        auto clip = graph->make("clip", sub_zp->get_outputs(), {},
                {{"clip_min",
                         qinfos.dtype_.is_etype(sc_data_etype::U8)
                                 ? 0.f
                                 : -128.f},
                        {"clip_max",
                                qinfos.dtype_.is_etype(
                                        sc_data_etype::U8)
                                        ? 255.f
                                        : 127.f}});
#endif
    auto int8_cast = graph->make("cast", div_scale->get_outputs(), {},
            {{"dtype", qinfos.dtype_}, {"saturated", true}});
    graph->make_output(int8_cast->get_outputs());
}

void quantize_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    common_query_function(this, info_.inputs_[0]->details_.get_format(),
            supported_ins, supported_outs);
}

dequantize_op_t::dequantize_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    assert(ins.size() == 1);
    assert(ins[0]->details_.dtype_.type_code_ == sc_data_etype::U8
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::S8
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::S32
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::F32);
    info_.inputs_ = ins;
    if (outs.empty()) {
        // fixme: correctly infer the shape for broadcast
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_ = ins[0]->details_;
        info_.outputs_[0]->details_.dtype_.type_code_ = sc_data_etype::F32;
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "dequantize";
}

dequantize_op_t::dequantize_op_t(
        const std::vector<graph_tensor_ptr> &ins, const any_map_t &attrs)
    : dequantize_op_t(ins, std::vector<graph_tensor_ptr>(), attrs) {}

void dequantize_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    auto qinfos = get_quantize_info_from_attrs(attrs_);
    qinfos.dtype_ = datatypes::f32;
    std::vector<float> scales = qinfos.scales_;
    std::shared_ptr<static_data_t> scales_ptr
            = std::make_shared<static_data_t>(scales);
    sc_dims scales_plain_dims = {1};
    if (scales.size() > 1) {
        scales_plain_dims.resize(
                inputs[0]->details_.get_plain_dims().size(), 1);
        scales_plain_dims[qinfos.channel_axis_]
                = static_cast<int>(scales.size());
    }
    auto ins = graph->make_input(inputs);
    auto const_scales = graph->make("constant", {}, {},
            {{"values", scales_ptr}, {"dtype", datatypes::f32},
                    {"plain_dims", scales_plain_dims},
                    {"format", sc_data_format_t()}, {"all_positive", true}});
    auto f32_cast = ins;
    if (inputs[0]->details_.dtype_.type_code_ != sc_data_etype::F32) {
        f32_cast = graph->make(
                "cast", ins->get_outputs(), {}, {{"dtype", qinfos.dtype_}});
    }

    bool all_zero = std::all_of(qinfos.zero_points_.begin(),
            qinfos.zero_points_.end(), [](int x) { return x == 0; });
    if (!all_zero) {
        std::vector<float> zero_points(
                qinfos.zero_points_.begin(), qinfos.zero_points_.end());
        auto const_zero_points = graph->make("constant", {}, {},
                {{"values", std::make_shared<static_data_t>(zero_points)},
                        {"dtype", datatypes::f32},
                        {"plain_dims",
                                sc_dims {static_cast<sc_dim>(
                                        zero_points.size())}},
                        {"format", sc_data_format_t()}});
        f32_cast = graph->make("sub",
                {f32_cast->get_outputs()[0],
                        const_zero_points->get_outputs()[0]},
                {}, {});
    }
    auto mul_scale = graph->make("mul",
            {f32_cast->get_outputs()[0], const_scales->get_outputs()[0]}, {},
            {});
    graph->make_output(mul_scale->get_outputs());
}

void dequantize_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    common_query_function(this, info_.inputs_[0]->details_.get_format(),
            supported_ins, supported_outs);
}

dynamic_quantize_op_t::dynamic_quantize_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    assert(ins.size() == 2 || ins.size() == 3);
    assert(ins[0]->details_.dtype_.type_code_ == sc_data_etype::F32
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::S32);
    assert(ins[1]->details_.dtype_.type_code_ == sc_data_etype::F32);
    if (ins.size() == 3) {
        assert(utils::is_one_of(ins[2]->details_.dtype_.type_code_,
                sc_data_etype::U8, sc_data_etype::S8, sc_data_etype::S32));
        assert(ins[2]->details_.get_plain_dims()
                == ins[1]->details_.get_plain_dims());
    }
    info_.inputs_ = ins;
    if (outs.empty()) {
        // fixme: correctly infer the shape for broadcast
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_ = ins[0]->details_;
        assert(attrs.has_key(attr_keys::quan_dtype));
        info_.outputs_[0]->details_.dtype_
                = attrs.get<sc_data_type_t>(attr_keys::quan_dtype);
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "dynamic_quantize";
}

void dynamic_quantize_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    const auto qinfos = get_quantize_info_from_attrs(attrs_);
    assert(utils::is_one_of(qinfos.dtype_, datatypes::u8, datatypes::s8));
    auto &inp = inputs[0];
    auto &scales = inputs[1];
    auto inp_op = graph->make_input(inputs);
    auto bc_axis = scales->details_.get_plain_dims() != sc_dims {1}
            ? std::vector<int> {qinfos.channel_axis_}
            : std::vector<int> {};
    auto div_scale
            = graph->make("div", {inp, scales}, {}, {{"bc_axis", bc_axis}});
    if (inputs.size() == 3) {
        auto zp_cast = graph->make(
                "cast", {inputs[2]}, {}, {{"dtype", datatypes::f32}});
        div_scale = graph->make("add",
                {div_scale->get_outputs()[0], zp_cast->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});
    }
    auto int8_cast = graph->make("cast", div_scale->get_outputs(), {},
            {{"dtype", qinfos.dtype_}, {"saturated", true}});
    graph->make_output(int8_cast->get_outputs());
}

void dynamic_quantize_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    common_query_function(this, info_.inputs_[0]->details_.get_format(),
            supported_ins, supported_outs);
}

dynamic_dequantize_op_t::dynamic_dequantize_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    assert(ins.size() == 2 || ins.size() == 3);
    assert(ins[0]->details_.dtype_.type_code_ == sc_data_etype::U8
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::S8
            || ins[0]->details_.dtype_.type_code_ == sc_data_etype::S32);
    assert(ins[1]->details_.dtype_.type_code_ == sc_data_etype::F32);
    if (ins.size() == 3) {
        assert(utils::is_one_of(ins[2]->details_.dtype_.type_code_,
                sc_data_etype::U8, sc_data_etype::S8, sc_data_etype::S32));
        assert(ins[2]->details_.get_plain_dims()
                == ins[1]->details_.get_plain_dims());
    }
    info_.inputs_ = ins;
    if (outs.empty()) {
        // fixme: correctly infer the shape for broadcast
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.outputs_[0]->details_ = ins[0]->details_;
        info_.outputs_[0]->details_.dtype_ = datatypes::f32;
    } else {
        info_.outputs_ = outs;
    }
    attrs_ = attrs;
    op_name_ = "dynamic_dequantize";
}

void dynamic_dequantize_op_t::get_graph_impl(
        std::shared_ptr<sc_graph_t> &graph) {
    // create new input logical tensors
    std::vector<graph_tensor_ptr> inputs, outputs;
    inputs = remake_logical_tensors(info_.inputs_);
    outputs = remake_logical_tensors(info_.outputs_);
    auto qinfos = get_quantize_info_from_attrs(attrs_);
    qinfos.dtype_ = datatypes::f32;
    auto &inp = inputs[0];
    auto &scales = inputs[1];
    auto inp_op = graph->make_input(inputs);
    auto f32_cast = graph->make("cast", {inp}, {}, {{"dtype", datatypes::f32}});
    auto bc_axis = scales->details_.get_plain_dims() != sc_dims {1}
            ? std::vector<int> {qinfos.channel_axis_}
            : std::vector<int> {};
    if (inputs.size() == 3) {
        auto zp_cast = graph->make(
                "cast", {inputs[2]}, {}, {{"dtype", datatypes::f32}});
        f32_cast = graph->make("sub",
                {f32_cast->get_outputs()[0], zp_cast->get_outputs()[0]}, {},
                {{"bc_axis", bc_axis}});
    }
    auto mul_scale = graph->make("mul", {f32_cast->get_outputs()[0], scales},
            {}, {{"bc_axis", bc_axis}});
    graph->make_output(mul_scale->get_outputs());
}

void dynamic_dequantize_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    common_query_function(this, info_.inputs_[0]->details_.get_format(),
            supported_ins, supported_outs);
}

} // namespace quantize

OP_REGISTER(quantize::quantize_op_t, quantize)
OP_REGISTER(quantize::dequantize_op_t, dequantize)
OP_REGISTER(quantize::dynamic_quantize_op_t, dynamic_quantize)
OP_REGISTER(quantize::dynamic_dequantize_op_t, dynamic_dequantize)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

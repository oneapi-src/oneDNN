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
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/op_schema.hpp"
#include "utils/utils.hpp"

#include "lower_down.hpp"
#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using value_ptr = std::shared_ptr<impl::value_t>;
using ltw = impl::logical_tensor_wrapper_t;

static bool has_optional_bias(op_kind_t kind) {
    std::set<op_kind_t> ops {impl::op_kind::Convolution, impl::op_kind::MatMul,
            impl::op_kind::ConvTranspose};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool has_int8_support(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::dnnl_convolution, impl::op_kind::MatMul,
            op_kind::dnnl_convtranspose, impl::op_kind::Reorder};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool has_post_ops(op_kind_t kind) {
    std::set<op_kind_t> ops {impl::op_kind::Convolution,
            op_kind::dnnl_convolution, impl::op_kind::ConvTranspose,
            op_kind::dnnl_convtranspose, impl::op_kind::MatMul,
            impl::op_kind::AvgPool, impl::op_kind::MaxPool, op_kind::dnnl_pool,
            impl::op_kind::ReLU, op_kind::dnnl_binary, op_kind::dnnl_batchnorm,
            impl::op_kind::BatchNormInference};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool is_output_scales_supported(op_kind_t kind) {
    // ops which don't support output scales
    std::set<op_kind_t> ops {
            impl::op_kind::AvgPool, impl::op_kind::MaxPool, op_kind::dnnl_pool};
    return ops.count(kind) == 0;
}

static std::pair<op_ptr, op_ptr> combine_scales(op_t *src_scales_op,
        op_t *src1_scales_op, op_t *dst_scales_op, op_kind_t binary_kind) {
    // combines scales for int8 patterns which contain binary post-op.
    // Should be used by OPs which don't support output scales attribute.
    // The way of combining scales depends on the binary algorithm.
    // Returned pair is in the following order:
    // new_dst_scales, new_src1_scales.
    // In cases where new_src1_scales can be removed (e.g. Multiply),
    // the 2nd op_ptr will be empty.
    const auto fuse_scales
            = [](const std::vector<float> &scales0,
                      const std::vector<float> &scales1,
                      const std::function<float(float, float)> &operation)
            -> std::vector<float> {
        std::vector<float> fused_scales(
                std::max(scales0.size(), scales1.size()), 1.f);
        if (scales0.size() >= scales1.size()) {
            for (size_t i = 0; i < scales1.size(); ++i) {
                fused_scales[i] = operation(scales0[i], scales1[0]);
            }
        } else {
            for (size_t i = 0; i < scales1.size(); ++i) {
                fused_scales[i] = operation(scales0[0], scales1[i]);
            }
        }
        return fused_scales;
    };
    const auto fuse_scales_attributes = [](const std::vector<op_t *> &scale_ops)
            -> std::pair<std::string, int64_t> {
        for (size_t i = 0; i < scale_ops.size(); ++i) {
            if (scale_ops[i]->get_attr<std::string>("qtype") == "per_channel") {
                // assumption: at least one scales per channel will make
                // combined scales per channel
                return std::make_pair(
                        "per_channel", scale_ops[i]->get_attr<int64_t>("axis"));
            }
        }
        // scales per tensor, defaulting axis
        return std::make_pair("per_tensor", static_cast<int64_t>(1));
    };

    const auto src_scales
            = src_scales_op->get_attr<std::vector<float>>("scales");
    assertm(std::all_of(src_scales.begin(), src_scales.end(),
                    [](float v) { return v != 0.f; }),
            "scales can't be zero");
    const auto src1_scales
            = src1_scales_op->get_attr<std::vector<float>>("scales");
    const auto inv_dst_scales
            = dst_scales_op->get_attr<std::vector<float>>("scales");

    std::vector<float> new_src1_scales;
    std::vector<float> new_dst_scales;
    std::string new_src1_qtype;
    std::string new_dst_qtype;
    int64_t new_src1_axis;
    int64_t new_dst_axis;
    const auto multiplier = std::multiplies<float>();
    const auto divider = std::divides<float>();
    switch (binary_kind) {
        case impl::op_kind::Add:
            new_src1_scales = fuse_scales(src1_scales, src_scales, divider);
            new_dst_scales
                    = fuse_scales(src_scales, inv_dst_scales, multiplier);
            std::tie(new_src1_qtype, new_src1_axis)
                    = fuse_scales_attributes({src_scales_op, src1_scales_op});
            std::tie(new_dst_qtype, new_dst_axis)
                    = fuse_scales_attributes({src_scales_op, dst_scales_op});
            break;
        case impl::op_kind::Multiply:
            new_dst_scales = fuse_scales(src_scales, src1_scales, multiplier);
            new_dst_scales
                    = fuse_scales(new_dst_scales, inv_dst_scales, multiplier);
            std::tie(new_dst_qtype, new_dst_axis) = fuse_scales_attributes(
                    {src_scales_op, src1_scales_op, dst_scales_op});
        default:
            assertm(false, "unsupported binary post-op was provided.");
            break;
    }

    op_ptr new_dst_scales_op = std::make_shared<op_t>(op_kind::mul_scales);
    op_ptr new_src1_scales_op;

    new_dst_scales_op->set_attr<std::vector<float>>("scales", new_dst_scales);
    new_dst_scales_op->set_attr<std::string>("qtype", new_dst_qtype);
    new_dst_scales_op->set_attr<int64_t>("axis", new_dst_axis);

    if (!new_src1_scales.empty()) {
        new_src1_scales_op = std::make_shared<op_t>(op_kind::mul_scales);
        new_src1_scales_op->set_attr<std::vector<float>>(
                "scales", new_src1_scales);
        new_src1_scales_op->set_attr<std::string>("qtype", new_src1_qtype);
        new_src1_scales_op->set_attr<int64_t>("axis", new_src1_axis);
    }

    return std::make_pair(new_dst_scales_op, new_src1_scales_op);
}

impl::status_t check_with_bias(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();

    for (auto &cur_op : subgraph) {
        if (!has_optional_bias(cur_op->get_kind())) continue;
        if (cur_op->num_inputs() == 3) {
            cur_op->set_attr<bool>("with_bias", true);
        } else {
            cur_op->set_attr<bool>("with_bias", false);
        }
    }
    return impl::status::success;
}

impl::status_t fuse_bias_add(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_t *> bias_add_ops;

    std::set<op_t *> visited;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::BiasAdd
                || visited.count(cur_op.get()) != 0)
            continue;

        bias_add_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    for (auto &bias_add : bias_add_ops) {
        auto in_val = bias_add->get_input_value(0);
        auto &prv_op = in_val->get_producer();
        if (!has_optional_bias(prv_op.get_kind())) continue;
        fuse_op_to_predecessor(bias_add, subgraph);
        prv_op.set_attr<bool>("with_bias", true);
    }

    return impl::status::success;
}

impl::status_t split_quant_dequant(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();

    std::vector<op_ptr> q_dq_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() == impl::op_kind::Quantize
                || cur_op->get_kind() == impl::op_kind::Dequantize) {
            q_dq_ops.emplace_back(cur_op);
        }
    }

    for (auto &cur_op : q_dq_ops) {
        const auto &zps = cur_op->get_attr<std::vector<int64_t>>("zps");
        const auto &scales = cur_op->get_attr<std::vector<float>>("scales");
        const auto &qtype = cur_op->get_attr<std::string>("qtype");
        const auto &axis = cur_op->get_attr<int64_t>("axis");

        auto in_vals = cur_op->get_input_values();
        auto out_vals = cur_op->get_output_values();
        assertm(in_vals.size() == 1 && out_vals.size() == 1,
                "static quantize/dequantize should only have one input and "
                "output");

        op_ptr op1, op2;
        if (cur_op->get_kind() == impl::op_kind::Dequantize) {
            // f32 = scales * (int8 - zps)
            op1 = std::make_shared<op_t>(op_kind::add_zps);
            op2 = std::make_shared<op_t>(op_kind::mul_scales);

            std::vector<int64_t> neg_zps = dnnl_impl::utils::fmap(
                    zps, [](int64_t zp) { return -zp; });
            op1->set_attr<std::vector<int64_t>>("zps", neg_zps);
            op2->set_attr<std::vector<float>>("scales", scales);
        } else {
            // int8 = f32 / scales + zps
            op1 = std::make_shared<op_t>(op_kind::mul_scales);
            op2 = std::make_shared<op_t>(op_kind::add_zps);

            assertm(std::all_of(scales.begin(), scales.end(),
                            [](float i) { return i != 0.f; }),
                    "scales can't be zero");

            std::vector<float> inv_scales = dnnl_impl::utils::fmap(
                    scales, [](float s) { return 1.f / s; });
            op1->set_attr<std::vector<float>>("scales", inv_scales);
            op2->set_attr<std::vector<int64_t>>("zps", zps);
        }
        op1->set_attr<int64_t>("axis", axis);
        op1->set_attr<std::string>("qtype", qtype);
        op2->set_attr<int64_t>("axis", axis);
        op2->set_attr<std::string>("qtype", qtype);

        // reconnect
        in_vals[0]->remove_consumer(*cur_op, 0);
        in_vals[0]->add_consumer(*op1, 0);
        op1->add_input(in_vals[0]);

        impl::logical_tensor_t new_lt
                = impl::empty_logical_tensor_with_default_id();
        auto new_val = std::make_shared<value_t>(*op1, 0, new_lt, true);
        op1->add_output(new_val);

        op2->add_input(new_val);
        new_val->add_consumer(*op2, 0);
        op2->add_output(out_vals[0]);

        // add new ops and delete quantize or dequantize op
        subgraph.emplace_back(op1);
        subgraph.emplace_back(op2);
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [cur_op](const op_ptr &op) { return *cur_op == *op; });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }
    return impl::status::success;
}

impl::status_t replace_quant_dequant_with_mul_scales(
        std::shared_ptr<subgraph_t> &sg) {
    // replaces Quant/Dequant OPs with mul_scales, zps attribute is ignored.
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> q_dq_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() == impl::op_kind::Quantize
                || cur_op->get_kind() == impl::op_kind::Dequantize) {
            q_dq_ops.emplace_back(cur_op);
        }
    }

    for (auto &cur_op : q_dq_ops) {
        // directly ignoring zps attribute
        const auto &scales = cur_op->get_attr<std::vector<float>>("scales");
        const auto &qtype = cur_op->get_attr<std::string>("qtype");
        const auto &axis = cur_op->get_attr<int64_t>("axis");

        auto in_vals = cur_op->get_input_values();
        auto out_vals = cur_op->get_output_values();
        assertm(in_vals.size() == 1 && out_vals.size() == 1,
                "static quantize/dequantize should only have one input and "
                "output");

        op_ptr mul_scales_op = std::make_shared<op_t>(op_kind::mul_scales);
        mul_scales_op->set_attr<std::string>("qtype", qtype);
        mul_scales_op->set_attr<int64_t>("axis", axis);
        if (impl::op_kind::Dequantize == cur_op->get_kind()) {
            mul_scales_op->set_attr<std::vector<float>>("scales", scales);
        } else { // Quantize
            assertm(std::all_of(scales.begin(), scales.end(),
                            [](float i) { return i != 0.f; }),
                    "scales can't be zero");

            std::vector<float> inv_scales = dnnl_impl::utils::fmap(
                    scales, [](float s) { return 1.f / s; });
            mul_scales_op->set_attr<std::vector<float>>("scales", inv_scales);
        }

        // reconnect
        in_vals[0]->remove_consumer(*cur_op, 0);
        in_vals[0]->add_consumer(*mul_scales_op, 0);
        mul_scales_op->add_input(in_vals[0]);
        mul_scales_op->add_output(out_vals[0]);

        // add new op and delete quantize or dequantize op
        subgraph.emplace_back(mul_scales_op);
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [cur_op](const op_ptr &op) { return *cur_op == *op; });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }
    return impl::status::success;
}

impl::status_t fuse_output_scales(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    auto &prm_attr_mgr = sg->prm_attr_mgr_;

    std::vector<std::pair<op_t *, op_t *>> fuse_groups;

    std::set<op_t *> visited;
    for (auto &cur_op : subgraph) {
        if (!has_int8_support(cur_op->get_kind())
                || visited.count(cur_op.get()) != 0)
            continue;

        assertm(cur_op->num_outputs() == 1,
                "cur_op should have only one output value.");
        auto out_val = cur_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;

        auto &next_op = consumers[0].get_op();
        if (next_op.get_kind() != op_kind::mul_scales) continue;

        fuse_groups.emplace_back(
                std::pair<op_t *, op_t *> {cur_op.get(), &next_op});
        visited.insert(cur_op.get());
        visited.insert(&next_op);
    }

    for (auto &fuse_group : fuse_groups) {
        auto base_op = fuse_group.first;
        auto scale_op = fuse_group.second;

        int64_t axis = scale_op->get_attr<int64_t>("axis");
        auto output_scales = scale_op->get_attr<std::vector<float>>("scales");

        int mask = output_scales.size() == 1 ? 0 : 1 << axis;

        int64_t key = -1;
        if (base_op->has_attr("primitive_attr_key")) {
            key = base_op->get_attr<int64_t>("primitive_attr_key");
        } else {
            key = prm_attr_mgr.init_attr();
            base_op->set_attr<int64_t>("primitive_attr_key", key);
        }

        dnnl::primitive_attr &prm_attr = prm_attr_mgr.get_attr(key);

        prm_attr.set_output_scales(mask, output_scales);

        // remove the fused scale op
        fuse_op_to_predecessor(scale_op, subgraph);
    }
    return impl::status::success;
}

impl::status_t replace_output_scales_with_binary(
        std::shared_ptr<subgraph_t> &sg) {
    // replace output scales with binary-multiply. It should be
    // used for OPs which don't support output scales attribute,
    // and do support only binary post-ops, e.g. Pooling and Eltwise.
    const auto get_next_op = [](const op_t *op) -> op_t * {
        const value_ptr out_val = op->get_output_value(0);
        if (!out_val->get_consumers().empty())
            return &out_val->get_consumers()[0].get_op();
        else
            return nullptr;
    };

    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<const op_t *> to_be_removed_ops;
    for (const auto &cur_op : subgraph) {
        if (is_output_scales_supported(cur_op->get_kind())) continue;

        // find output scales op
        op_t *next_op = get_next_op(cur_op.get());
        while (next_op && next_op->get_kind() == op_kind::dnnl_binary) {
            next_op = get_next_op(next_op);
        }
        // at this stage next_op is a first op after last binary,
        // we expect it to be a mul_scales
        if (!next_op || next_op->get_kind() != op_kind::mul_scales) continue;

        // replace scales op with binary-multiply op
        op_t *scales_op = next_op;
        op_ptr bin_mul_op = std::make_shared<op_t>(op_kind::dnnl_binary);
        bin_mul_op->set_attr<int64_t>(
                "alg_kind", static_cast<int64_t>(impl::op_kind::Multiply));
        auto in_val = scales_op->get_input_value(0);
        in_val->remove_consumer(*scales_op, 0);
        in_val->add_consumer(*bin_mul_op, 0);
        bin_mul_op->add_input(in_val);
        auto out_val = scales_op->get_output_value(0);
        bin_mul_op->add_output(out_val);

        // add constant scales op
        const auto scales = scales_op->get_attr<std::vector<float>>("scales");
        const auto qtype = scales_op->get_attr<std::string>("qtype");
        const auto axis = scales_op->get_attr<int64_t>("axis");
        const std::vector<int64_t> out_shape
                = ltw(out_val->get_logical_tensor()).vdims();
        std::vector<int64_t> new_shape(out_shape.size(), 1);
        if (qtype == "per_tensor") new_shape[axis] = out_shape[axis];
        op_ptr const_scales_op = std::make_shared<op_t>(op_kind::dnnl_constant);
        const_scales_op->set_attr("scales", scales);
        const_scales_op->set_attr("shape", new_shape);
        impl::logical_tensor_t const_scales_dst_lt
                = impl::empty_logical_tensor_with_default_id();
        auto const_scales_dst_value = std::make_shared<value_t>(
                *const_scales_op, 0, const_scales_dst_lt, true);
        const_scales_dst_value->set_layout_type(impl::layout_type::strided);
        const_scales_op->add_output(const_scales_dst_value);

        // connect binary-multiply with constant scales
        bin_mul_op->connect_input(1, const_scales_dst_value);

        to_be_inserted_ops.emplace_back(bin_mul_op);
        to_be_inserted_ops.emplace_back(const_scales_op);

        to_be_removed_ops.push_back(scales_op);
    }

    for (const auto &op : to_be_inserted_ops) {
        subgraph.emplace_back(op);
    }

    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &f_op) { return f_op.get() == op; });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }

    return impl::status::success;
}

impl::status_t folding_mul_scales(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    // lambda function to fold the consecutive mul_scales ops
    auto folding_mul_scales_func = [&]() {
        std::vector<std::pair<op_t *, op_t *>> folding_groups;
        std::set<op_t *> visited;
        for (const auto &cur_op : subgraph) {
            if (cur_op->get_kind() != op_kind::mul_scales
                    || visited.count(cur_op.get()) != 0)
                continue;

            assertm(cur_op->num_outputs() == 1,
                    "cur_op should have only one output value.");
            auto out_val = cur_op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            if (consumers.empty()) continue;

            auto &consumer_op = consumers[0].get_op();
            if (consumer_op.get_kind() != op_kind::mul_scales) continue;

            folding_groups.emplace_back(
                    std::pair<op_t *, op_t *> {cur_op.get(), &consumer_op});
            visited.insert(cur_op.get());
            visited.insert(&consumer_op);
        }

        if (folding_groups.empty()) return false;

        for (auto &folding_ops : folding_groups) {
            auto base_op = folding_ops.first;
            auto next_op = folding_ops.second;

            // update the scales
            const auto &scales_base
                    = base_op->get_attr<std::vector<float>>("scales");
            const auto &scales_next
                    = next_op->get_attr<std::vector<float>>("scales");
            std::vector<float> new_scales(
                    std::max(scales_base.size(), scales_next.size()), 1.f);
            // per-channel -> per-tensor
            if (scales_base.size() > scales_next.size()) {
                for (size_t i = 0; i < new_scales.size(); ++i)
                    new_scales[i] = scales_base[i] * scales_next[0];
            } else {
                for (size_t i = 0; i < new_scales.size(); ++i)
                    new_scales[i] = scales_base[0] * scales_next[i];
                // set attrs
                base_op->set_attr<int64_t>(
                        "axis", next_op->get_attr<int64_t>("axis"));
                base_op->set_attr<std::string>(
                        "qtype", next_op->get_attr<std::string>("qtype"));
            }
            base_op->set_attr<std::vector<float>>("scales", new_scales);

            fuse_op_to_predecessor(next_op, subgraph, 0);
        }
        return true;
    };

    bool changed = true;
    do {
        changed = folding_mul_scales_func();
    } while (changed);
    return impl::status::success;
}

impl::status_t fuse_to_int8_conv_or_deconv(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &op : subgraph) {
        if (op->get_kind() == impl::op_kind::Convolution
                || op->get_kind() == impl::op_kind::ConvTranspose) {
            auto &in0 = op->get_input_value(0)->get_producer();
            auto &in1 = op->get_input_value(1)->get_producer();
            if (in0.get_kind() != op_kind::mul_scales
                    || in1.get_kind() != op_kind::mul_scales)
                continue;

            fusion_groups.emplace_back(
                    std::vector<op_t *> {op.get(), &in0, &in1});
        }
    }

    for (auto &fusion_group : fusion_groups) {
        op_t *conv_op = fusion_group[0];
        op_t &in0 = *fusion_group[1];
        op_t &in1 = *fusion_group[2];

        op_kind_t tgt_op_kind;
        if (conv_op->get_kind() == impl::op_kind::Convolution)
            tgt_op_kind = op_kind::dnnl_convolution;
        else
            tgt_op_kind = op_kind::dnnl_convtranspose;
        op_ptr qconv_op = std::make_shared<op_t>(tgt_op_kind);
        op_ptr mul_op = std::make_shared<op_t>(op_kind::mul_scales);

        qconv_op->merge_attributes(conv_op->get_attributes());

        auto dq_src_scales = in0.get_attr<std::vector<float>>("scales");
        auto dq_wei_scales = in1.get_attr<std::vector<float>>("scales");
        std::vector<float> fused_scale(
                std::max(dq_src_scales.size(), dq_wei_scales.size()), 0);
        if (dq_src_scales.size() >= dq_wei_scales.size()) {
            for (int i = 0; i < dq_src_scales.size(); i++)
                fused_scale[i] = (dq_src_scales[i] * dq_wei_scales[0]);
            mul_op->set_attr<int64_t>("axis", in0.get_attr<int64_t>("axis"));
            mul_op->set_attr<std::string>(
                    "qtype", in0.get_attr<std::string>("qtype"));
        } else {
            // Currently for ConvTranspose, the output channel in weight tensor
            // (OC/g, IC, H, W) is not equal to the one in output tensor
            // (N, OC, H, W) if `groups` > 1, so the size of weight's
            // per-channel scale is not the same as the output channel in output
            // tensor, here we will broadcast scales from `OC/g` to `OC`.
            int64_t group = qconv_op->get_attr<int64_t>("groups");
            if (tgt_op_kind == op_kind::dnnl_convtranspose && group > 1) {
                fused_scale.resize(group * dq_wei_scales.size(), 0);
                for (int i = 0; i < fused_scale.size(); ++i)
                    fused_scale[i] = (dq_src_scales[0]
                            * dq_wei_scales[i % dq_wei_scales.size()]);
            } else {
                for (int i = 0; i < fused_scale.size(); ++i)
                    fused_scale[i] = (dq_src_scales[0] * dq_wei_scales[i]);
            }
            // FIXME(qun) the axis of output_scales should be related to data
            // format
            mul_op->set_attr<int64_t>("axis", 1); // hardcode to 1 for pytorch
            mul_op->set_attr<std::string>(
                    "qtype", in1.get_attr<std::string>("qtype"));
        }
        mul_op->set_attr<std::vector<float>>("scales", fused_scale);

        auto in0_ivalue = in0.get_input_value(0);
        auto in1_ivalue = in1.get_input_value(0);
        qconv_op->connect_input(0, in0_ivalue);
        qconv_op->connect_input(1, in1_ivalue);
        in0_ivalue->remove_consumer(in0, 0);
        in1_ivalue->remove_consumer(in1, 0);

        if (conv_op->num_inputs() == 3) { //with bias
            op_ptr mul_op1 = std::make_shared<op_t>(op_kind::mul_scales);

            assertm(std::all_of(fused_scale.begin(), fused_scale.end(),
                            [](float i) { return i != 0.f; }),
                    "scales can't be zero");

            std::vector<float> inv_scales(fused_scale.size());
            for (int i = 0; i < fused_scale.size(); i++)
                inv_scales[i] = 1.f / fused_scale[i];

            // FIXME(xxx) add other attrs
            mul_op1->set_attr<std::vector<float>>("scales", inv_scales);
            mul_op1->set_attr<int64_t>("axis", 0);

            auto bias_val = conv_op->get_input_value(2);
            bias_val->remove_consumer(*conv_op, 2);
            mul_op1->connect_input(0, bias_val);
            auto scaled_bias_lt = impl::empty_logical_tensor_with_default_id();
            auto scaled_bias_val = std::make_shared<value_t>(
                    *mul_op1, 0, scaled_bias_lt, true);
            mul_op1->add_output(scaled_bias_val);
            qconv_op->connect_input(2, scaled_bias_val);

            subgraph.emplace_back(mul_op1);
        }

        auto out_value = conv_op->get_output_value(0);

        auto new_lt1 = impl::empty_logical_tensor_with_default_id();
        auto new_value1
                = std::make_shared<value_t>(*qconv_op, 0, new_lt1, true);

        qconv_op->add_output(new_value1);
        out_value->set_producer(*mul_op);
        mul_op->connect_input(0, new_value1);
        mul_op->add_output(out_value);

        subgraph.emplace_back(qconv_op);
        subgraph.emplace_back(mul_op);

        for (auto &del_op : fusion_group) {
            auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [del_op](const op_ptr &f_op) {
                        return del_op == f_op.get();
                    });
            if (pos != subgraph.end()) subgraph.erase(pos);
        }
    }
    return impl::status::success;
}

impl::status_t fuse_to_int8_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::MatMul) continue;
        auto &in0 = cur_op->get_input_value(0)->get_producer();
        auto &in1 = cur_op->get_input_value(1)->get_producer();
        if (in0.get_kind() == op_kind::mul_scales
                && in1.get_kind() == op_kind::mul_scales)
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in0, &in1});
    }

    for (auto &fusion_group : fusion_groups) {
        op_t *matmul_op = fusion_group[0];
        op_t *in0 = fusion_group[1];
        op_t *in1 = fusion_group[2];

        op_ptr q_matmul_op = std::make_shared<op_t>(impl::op_kind::MatMul);
        q_matmul_op->merge_attributes(matmul_op->get_attributes());

        op_ptr mul_scales_op = std::make_shared<op_t>(op_kind::mul_scales);

        const auto &dq_src_scales = in0->get_attr<std::vector<float>>("scales");
        const auto &dq_wei_scales = in1->get_attr<std::vector<float>>("scales");
        std::vector<float> fused_scales(
                std::max(dq_src_scales.size(), dq_wei_scales.size()), 1.f);
        // src: per_tensor, weight: per_channel
        if (dq_src_scales.size() < dq_wei_scales.size()) {
            for (size_t i = 0; i < dq_wei_scales.size(); ++i)
                fused_scales[i] = dq_src_scales[0] * dq_wei_scales[i];
            // FIXME(wuxun): if quantization is per-channel, need to set axis
            // to the last dimension of dst
            // find the output edge op to get the ndims
            op_t begin_op = *matmul_op;
            while (!begin_op.get_output_value(0)->get_consumers().empty()) {
                begin_op = begin_op.get_output_value(0)
                                   ->get_consumers()[0]
                                   .get_op();
            }
            // if one of matmul's inputs is 1D tensor, the real ndims of
            // matmul's output should be (cur_ndims + 1), so we only need set
            // axis to cur_ndims
            int64_t new_axis
                    = begin_op.get_output_value(0)->get_logical_tensor().ndims
                    - 1;
            if (matmul_op->get_input_value(0)->get_logical_tensor().ndims == 1
                    || matmul_op->get_input_value(1)->get_logical_tensor().ndims
                            == 1)
                new_axis += 1;
            mul_scales_op->set_attr<int64_t>("axis", new_axis);
            mul_scales_op->set_attr<std::string>(
                    "qtype", in1->get_attr<std::string>("qtype"));
        } else {
            for (size_t i = 0; i < dq_src_scales.size(); ++i)
                fused_scales[i] = dq_src_scales[i] * dq_wei_scales[0];
            mul_scales_op->set_attr<int64_t>(
                    "axis", in0->get_attr<int64_t>("axis"));
            mul_scales_op->set_attr<std::string>(
                    "qtype", in0->get_attr<std::string>("qtype"));
        }
        mul_scales_op->set_attr<std::vector<float>>("scales", fused_scales);

        // update the connection relationship between matmul and mul_scales ops
        auto in0_value = in0->get_input_value(0);
        auto in1_value = in1->get_input_value(0);
        q_matmul_op->connect_input(0, in0_value);
        q_matmul_op->connect_input(1, in1_value);
        in0_value->remove_consumer(*in0, 0);
        in1_value->remove_consumer(*in1, 0);

        // with bias
        if (matmul_op->num_inputs() == 3) {
            op_ptr bias_mul_op = std::make_shared<op_t>(op_kind::mul_scales);

            assertm(std::all_of(fused_scales.begin(), fused_scales.end(),
                            [](float i) { return i != 0.f; }),
                    "scales can't be zero");

            std::vector<float> inv_scales(fused_scales.size(), 1.f);
            for (size_t i = 0; i < inv_scales.size(); ++i)
                inv_scales[i] = 1.f / fused_scales[i];
            bias_mul_op->set_attr<std::vector<float>>("scales", inv_scales);
            bias_mul_op->set_attr<int64_t>("axis", 0);

            auto bias_value = matmul_op->get_input_value(2);
            bias_value->remove_consumer(*matmul_op, 2);
            bias_mul_op->connect_input(0, bias_value);
            auto scaled_bias_lt = impl::empty_logical_tensor_with_default_id();
            auto scaled_bias_val = std::make_shared<value_t>(
                    *bias_mul_op, 0, scaled_bias_lt, true);
            bias_mul_op->add_output(scaled_bias_val);
            q_matmul_op->connect_input(2, scaled_bias_val);
            subgraph.emplace_back(bias_mul_op);
        }

        auto out_value = matmul_op->get_output_value(0);
        auto new_lt1 = impl::empty_logical_tensor_with_default_id();
        auto new_value1
                = std::make_shared<value_t>(*q_matmul_op, 0, new_lt1, true);

        q_matmul_op->add_output(new_value1);
        out_value->set_producer(*mul_scales_op);
        mul_scales_op->connect_input(0, new_value1);
        mul_scales_op->add_output(out_value);

        // delete original matmul and mul_scales ops
        for (auto &del_op : fusion_group) {
            auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [del_op](const op_ptr &f_op) {
                        return del_op == f_op.get();
                    });
            if (pos != subgraph.end()) subgraph.erase(pos);
        }

        subgraph.emplace_back(q_matmul_op);
        subgraph.emplace_back(mul_scales_op);
    }
    return impl::status::success;
}

impl::status_t fuse_to_int8_reorder(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::Reorder) continue;
        auto &in = cur_op->get_input_value(0)->get_producer();
        if (in.get_kind() == op_kind::mul_scales)
            fusion_groups.emplace_back(std::vector<op_t *> {cur_op.get(), &in});
    }

    for (auto &fusion_group : fusion_groups) {
        op_t *reorder_op = fusion_group[0];
        op_t *in_op = fusion_group[1];

        // update the connection relationship between reorder and mul_scales ops
        // basically switch the order of these two ops
        auto in_op_in_value = in_op->get_input_value(0);
        auto reorder_op_in_value = reorder_op->get_input_value(0);
        auto out_value = reorder_op->get_output_value(0);

        reorder_op->connect_input(0, in_op_in_value);
        in_op_in_value->remove_consumer(*in_op, 0);
        reorder_op->connect_output(0, reorder_op_in_value);
        reorder_op_in_value->set_producer(*reorder_op);
        reorder_op_in_value->remove_consumer(*reorder_op, 0);

        in_op->connect_input(0, reorder_op_in_value);
        in_op->connect_output(0, out_value);
        out_value->set_producer(*in_op);
    }

    return impl::status::success;
}

impl::status_t fuse_to_int8_pool(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_t *> fusion_ops;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::MaxPool
                && cur_op->get_kind() != impl::op_kind::AvgPool)
            continue;
        fusion_ops.emplace_back(cur_op.get());
    }

    if (fusion_ops.empty()) return impl::status::success;

    for (auto &pool_op : fusion_ops) {
        op_ptr q_pool_op = std::make_shared<op_t>(op_kind::dnnl_pool);
        q_pool_op->merge_attributes(pool_op->get_attributes());
        if (pool_op->get_kind() == impl::op_kind::MaxPool) {
            q_pool_op->set_attr<std::string>("kind", "maxpool");
        } else {
            q_pool_op->set_attr<std::string>("kind", "avgpool");
        }

        // mul_scales which feeds pooling will be always removed. If binary
        // post-op will occur, src scales data will be combined with dst
        // scales data.
        op_t &src_scales_op = pool_op->get_input_value(0)->get_producer();
        assertm(src_scales_op.get_kind() == op_kind::mul_scales,
                "the predecessor op of pool should be mul_scales.");
        value_ptr in_value = src_scales_op.get_input_value(0);
        in_value->remove_consumer(src_scales_op, 0);
        q_pool_op->connect_input(0, in_value);

        assertm(pool_op->get_output_value(0)->get_consumers().size() == 1,
                "pooling's successor op should only have one consumer.");
        op_t &pool_op_successor
                = pool_op->get_output_value(0)->get_consumers()[0].get_op();

        std::vector<const op_t *> deleted_ops {pool_op, &src_scales_op};

        if (pool_op_successor.get_kind() != op_kind::dnnl_binary) {
            // standalone pool case: detach pooling from the dst mul_scales
            op_t &dst_scales_op = pool_op_successor;
            assertm(dst_scales_op.get_kind() == op_kind::mul_scales,
                    "the successor op of pool should be mul_scales or binary.");
            value_ptr out_value = dst_scales_op.get_output_value(0);
            q_pool_op->add_output(out_value);

            deleted_ops.push_back(&dst_scales_op);
        } else {
            // pool + binary case
            op_t &binary_op = pool_op_successor;
            assertm(binary_op.num_inputs() == 2,
                    "binary op should have exactly two inputs");
            // binary could be commutative, depending on algorithms,
            // so we need to check here which input is which OP.
            const size_t src1_scales_offset
                    = (binary_op.get_input_value(0)->get_producer().get_kind()
                              == op_kind::mul_scales)
                    ? 0
                    : 1;
            op_t &src1_scales_op
                    = binary_op.get_input_value(src1_scales_offset)
                              ->get_producer();
            assertm(src1_scales_op.get_kind() == op_kind::mul_scales,
                    "the 2nd binary input producer should be mul_scales.");

            assertm(binary_op.get_output_value(0)->get_consumers().size() == 1,
                    "binary's successor op should only have one consumer.");
            op_t &dst_scales_op = binary_op.get_output_value(0)
                                          ->get_consumers()[0]
                                          .get_op();
            assertm(dst_scales_op.get_kind() == op_kind::mul_scales,
                    "the successor op of binary should be mul_scales.");

            // <new_dst_scales_op, new_src1_scales_op (could be nullptr)>
            std::pair<op_ptr, op_ptr> combined_scales = combine_scales(
                    &src_scales_op, &src1_scales_op, &dst_scales_op,
                    static_cast<op_kind_t>(
                            binary_op.get_attr<int64_t>("alg_kind")));

            // make new connections
            value_ptr src_bin_value
                    = binary_op.get_input_value(1 - src1_scales_offset);
            q_pool_op->add_output(src_bin_value);

            impl::logical_tensor_t new_dst_bin_lt
                    = impl::empty_logical_tensor_with_default_id();
            auto new_dst_bin_value = std::make_shared<value_t>(
                    binary_op, 0, new_dst_bin_lt, true);
            binary_op.connect_output(0, new_dst_bin_value);

            op_ptr new_dst_scales_op = combined_scales.first;
            value_ptr dst_scales_value = dst_scales_op.get_output_value(0);
            new_dst_scales_op->connect_input(0, new_dst_bin_value);
            new_dst_scales_op->add_output(dst_scales_value);

            subgraph.emplace_back(new_dst_scales_op);

            value_ptr src1_scales_value = src1_scales_op.get_input_value(0);
            src1_scales_value->remove_consumer(src1_scales_op, 0);
            if (combined_scales.second != nullptr) {
                op_ptr new_src1_scales_op = combined_scales.second;
                new_src1_scales_op->connect_input(0, src1_scales_value);

                impl::logical_tensor_t new_src1_scales_lt
                        = impl::empty_logical_tensor_with_default_id();
                auto new_src1_scales_value = std::make_shared<value_t>(
                        *new_src1_scales_op, 0, new_src1_scales_lt, true);
                new_src1_scales_op->add_output(new_src1_scales_value);
                binary_op.connect_input(
                        src1_scales_offset, new_src1_scales_value);

                subgraph.emplace_back(new_src1_scales_op);
            } else {
                // src1 mul_scales input will be connected directly to the
                // binary op
                binary_op.connect_input(src1_scales_offset, src1_scales_value);
            }

            deleted_ops.push_back(&src1_scales_op);
            deleted_ops.push_back(&dst_scales_op);
        }

        for (const auto &del_op : deleted_ops) {
            auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [del_op](const op_ptr &f_op) {
                        return f_op.get() == del_op;
                    });
            if (pos != subgraph.end()) subgraph.erase(pos);
        }

        subgraph.emplace_back(q_pool_op);
    }
    return impl::status::success;
}

impl::status_t fuse_to_shuffle(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::StaticReshape) continue;

        if (cur_op->get_output_value(0)->get_consumers().size() != 1) continue;
        auto &next0 = cur_op->get_output_value(0)->get_consumers()[0].get_op();
        if (next0.get_kind() != impl::op_kind::StaticTranspose) continue;

        if (next0.get_output_value(0)->get_consumers().size() != 1) continue;
        auto &next1 = next0.get_output_value(0)->get_consumers()[0].get_op();
        if (next1.get_kind() != impl::op_kind::StaticReshape) continue;

        fusion_groups.emplace_back(
                std::vector<op_t *> {cur_op.get(), &next0, &next1});
    }

    for (auto &fusion_group : fusion_groups) {
        op_t *reshape0 = fusion_group[0];
        op_t *transpose = fusion_group[1];
        op_t *reshape1 = fusion_group[2];

        op_ptr shuffle = std::make_shared<op_t>(op_kind::dnnl_shuffle);

        value_ptr in_value = reshape0->get_input_value(0);
        value_ptr out_value = reshape1->get_output_value(0);

        const auto src_shape = ltw(in_value->get_logical_tensor()).vdims();
        const auto attr_shape
                = reshape0->get_attr<std::vector<int64_t>>("shape");
        const auto res = std::mismatch(
                src_shape.cbegin(), src_shape.cend(), attr_shape.cbegin());
        const size_t axis = std::distance(src_shape.cbegin(), res.first);
        const int64_t group = attr_shape[axis];
        shuffle->set_attr<int64_t>("axis", static_cast<int64_t>(axis));
        shuffle->set_attr<int64_t>("group", group);

        shuffle->connect_input(0, in_value);
        in_value->remove_consumer(*reshape0, 0);

        shuffle->add_output(out_value);

        for (auto &del_op : fusion_group) {
            auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [del_op](const op_ptr &f_op) {
                        return del_op == f_op.get();
                    });
            if (pos != subgraph.end()) subgraph.erase(pos);
        }

        subgraph.emplace_back(shuffle);
    }

    return impl::status::success;
}

status_t fuse_post_ops(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    auto &prm_attr_mgr = sg->prm_attr_mgr_;

    // lambda function to fuse one post op into base primitive
    auto fuse_post_ops_func = [&](std::vector<op_ptr> &subgraph,
                                      primitive_attr_mgr_t &prm_attr_mgr,
                                      bool &changed) -> impl::status_t {
        std::vector<std::pair<op_t *, op_t *>> fuse_groups;

        std::set<op_t *> visited;
        impl::topo_order_visit(
                impl::graph_t(subgraph).get_output_ops(), [&](impl::op_t *op) {
                    const auto &pops_fusible_map = get_post_ops_fusible_map();

                    auto base_op_kind = op->get_kind();
                    if (!pops_fusible_map.count(base_op_kind)
                            || visited.count(op) != 0)
                        return impl::status::success;

                    auto out_val = op->get_output_values()[0];
                    auto consumers = out_val->get_consumers();

                    // The base op should have and only have one consumer, it's
                    // the post op to be fused
                    if (consumers.size() != 1) return impl::status::success;
                    auto &post_op = consumers[0].get_op();

                    // check if fusible
                    // TODO(qun) make sure bn only fuse relu
                    auto post_op_kind = post_op.get_kind();
                    bool not_fusible = (!pops_fusible_map.at(base_op_kind)
                                                       .count(post_op_kind))
                            || (post_op_kind == op_kind::dnnl_binary
                                    && !post_binary_fusible(op, &post_op))
                            || (post_op_kind == impl::op_kind::Convolution
                                    && !post_depthwise_conv_fusible(&post_op));
                    if (not_fusible) { return impl::status::success; }

                    // push fusible pair to fuse group for later fusion
                    fuse_groups.emplace_back(
                            std::pair<op_t *, op_t *> {op, &post_op});
                    visited.insert(op);
                    visited.insert(&post_op);
                    return impl::status::success;
                });

        if (fuse_groups.empty()) {
            changed = false;
            return impl::status::success;
        }

        for (auto &fuse_group : fuse_groups) {
            auto base_op = fuse_group.first;
            auto post_op = fuse_group.second;
            // post op fuse to which predecessor
            size_t fuse_op_predecessor_offset = base_op->get_output_value(0)
                                                        ->get_consumers()[0]
                                                        .get_offset();

            int64_t key = -1;
            if (base_op->has_attr("primitive_attr_key")) {
                key = base_op->get_attr<int64_t>("primitive_attr_key");
            } else {
                key = prm_attr_mgr.init_attr();
                base_op->set_attr<int64_t>("primitive_attr_key", key);
            }

            dnnl::primitive_attr &prm_attr = prm_attr_mgr.get_attr(key);

            dnnl::post_ops pops = prm_attr.get_post_ops();

            bool with_op_replacement = false;

            if (post_op->get_kind() == op_kind::dnnl_eltwise) {
                float scale = 1.f;
                float alpha = 0;
                float beta = 0;

                const auto post_op_kind = static_cast<impl::op_kind_t>(
                        post_op->get_attr<int64_t>("alg_kind"));
                assertm(is_eltwise_kind(post_op_kind),
                        "alg_kind of dnnl_eltwise should be able to mapped to "
                        "eltwise op kind");
                const auto alg = get_eltwise_alg_map().at(post_op_kind);

                // for BatchNormForwardTraining, set dnnl_fuse_norm_relu flag
                // instead of post op
                if ((base_op->get_kind() == op_kind::dnnl_batchnorm
                            && base_op->get_attr<bool>("is_training"))
                        && post_op_kind == impl::op_kind::ReLU) {
                    base_op->set_attr<bool>("fuse_relu", true);
                    // remove the fused post_ops op
                    fuse_op_to_predecessor(
                            post_op, subgraph, fuse_op_predecessor_offset);
                    continue;
                }

                alpha = post_op->get_attr<float>("alpha");
                beta = post_op->get_attr<float>("beta");

                auto out_val = post_op->get_output_values()[0];
                auto consumers = out_val->get_consumers();
                if (!consumers.empty()) {
                    auto &next_op = consumers[0].get_op();
                    // set eltwise post-ops scale
                    if (next_op.get_kind() == op_kind::mul_scales) {
                        scale = next_op.get_attr<std::vector<float>>(
                                "scales")[0];
                        fuse_op_to_predecessor(&next_op, subgraph);
                    }
                }
                pops.append_eltwise(scale, alg, alpha, beta);
            } else if (post_op->get_kind() == op_kind::dnnl_binary
                    && static_cast<impl::op_kind_t>(
                               post_op->get_attr<int64_t>("alg_kind"))
                            == impl::op_kind::Add) {
                // If the other in value of Add has mul_scales producer,
                // then this pattern is a int8 pattern
                size_t mul_scale_op_offset = 2;
                auto other_in_val = post_op->get_input_value(
                        1 - fuse_op_predecessor_offset);
                if (other_in_val->has_producer()
                        && other_in_val->get_producer().get_kind()
                                == op_kind::mul_scales) {
                    mul_scale_op_offset = 1 - fuse_op_predecessor_offset;
                }
                if (mul_scale_op_offset != 2
                        && is_output_scales_supported(base_op->get_kind())) {
                    // for int8 cases (excluding OPs which don't support
                    // output scales attribute)
                    auto in_val = post_op->get_input_value(mul_scale_op_offset);
                    auto &mul_scale_op = in_val->get_producer();
                    auto scales = mul_scale_op.get_attr<std::vector<float>>(
                            "scales");
                    assert(scales.size() == 1); // per tensor

                    auto tmp = mul_scale_op.get_input_value(0);
                    auto &add_zps_op = tmp->get_producer();
                    auto zps = add_zps_op.get_attr<std::vector<int64_t>>("zps");
                    assert(scales.size() == zps.size());

                    fuse_op_to_successor(&add_zps_op, subgraph);
                    fuse_op_to_successor(&mul_scale_op, subgraph);

                    auto out_val = post_op->get_output_values()[0];
                    auto consumers = out_val->get_consumers();
                    if (!consumers.empty()) {
                        auto &next_op = consumers[0].get_op();
                        // set sum post-ops' second input scale
                        if (next_op.get_kind() == op_kind::mul_scales) {
                            float tmp_scale
                                    = next_op.get_attr<std::vector<float>>(
                                            "scales")[0];
                            scales[0] *= tmp_scale;
                            fuse_op_to_predecessor(&next_op, subgraph);

                            scale_t ori_scales;
                            int ori_mask;
                            prm_attr.get_output_scales(ori_mask, ori_scales);
                            for (auto &v : ori_scales)
                                v *= tmp_scale;
                            prm_attr.set_output_scales(ori_mask, ori_scales);
                        }
                    }
                    pops.append_sum(scales[0], static_cast<int32_t>(-zps[0]));
                    base_op->set_attr<bool>("with_sum", true);
                } else {
                    // - the add operation may need broadcast
                    auto fused_in = post_op->get_input_value(
                            fuse_op_predecessor_offset);
                    auto other_in = post_op->get_input_value(
                            1 - fuse_op_predecessor_offset);
                    auto dst = post_op->get_output_value(0);

                    if (ltw(fused_in->get_logical_tensor()).vdims()
                            == ltw(other_in->get_logical_tensor()).vdims()) {
                        if (base_op->get_kind() == op_kind::dnnl_eltwise) {
                            memory::desc post_src = make_dnnl_memory_desc(
                                    other_in->get_logical_tensor());
                            pops.append_binary(algorithm::binary_add, post_src);
                            base_op->set_attr<bool>("with_binary", true);
                        } else {
                            // use sum post-ops for no-broadcast add
                            pops.append_sum(1.f);
                            base_op->set_attr<bool>("with_sum", true);
                        }
                    } else {
                        // use binary post-ops for broadcast add
                        memory::desc post_src = make_dnnl_memory_desc(
                                other_in->get_logical_tensor());

                        pops.append_binary(algorithm::binary_add, post_src);
                        base_op->set_attr<bool>("with_binary", true);
                    }
                }
            } else if (post_op->get_kind() == op_kind::dnnl_binary
                    && static_cast<impl::op_kind_t>(
                               post_op->get_attr<int64_t>("alg_kind"))
                            != impl::op_kind::Add) {
                memory::desc post_src = make_dnnl_memory_desc(
                        post_op->get_input_value(1 - fuse_op_predecessor_offset)
                                ->get_logical_tensor());
                const auto algo
                        = get_binary_alg_map().at(static_cast<impl::op_kind_t>(
                                post_op->get_attr<int64_t>("alg_kind")));
                pops.append_binary(algo, post_src);
                base_op->set_attr<bool>("with_binary", true);
            } else if (post_op->get_kind() == impl::op_kind::Convolution) {
                const auto get_dnn_dt = [](const value_ptr &val) {
                    const auto graph_dt
                            = ltw(val->get_logical_tensor()).data_type();
                    return static_cast<dnnl::memory::data_type>(graph_dt);
                };

                const size_t wei_offset = 1;
                const size_t dst_offset = 0;
                const auto wei_dt
                        = get_dnn_dt(post_op->get_input_value(wei_offset));
                const auto dst_dt
                        = get_dnn_dt(post_op->get_output_value(dst_offset));
                const auto bia_dt = dnnl::memory::data_type::undef;
                const int mask = 0;
                const std::string dw_type
                        = (post_op->get_attr<std::vector<int64_t>>("strides")[0]
                                  == 1)
                        ? "k3s1p1"
                        : "k3s2p1";
                if (dw_type == "k3s1p1")
                    pops.append_dw_k3s1p1(wei_dt, bia_dt, dst_dt, mask,
                            std::vector<float> {});
                else
                    pops.append_dw_k3s2p1(wei_dt, bia_dt, dst_dt, mask,
                            std::vector<float> {});

                op_ptr conv_dw
                        = std::make_shared<op_t>(op_kind::conv_depthwise);
                const auto dw_groups = post_op->get_attr<int64_t>("groups");
                const auto dw_filter_format
                        = post_op->get_attr<std::string>("filter_format");
                conv_dw->set_attr("dw_groups", dw_groups);
                conv_dw->set_attr("dw_filter_format", dw_filter_format);
                conv_dw->set_attr("dw_type", dw_type);

                auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                        [base_op](const op_ptr &f_op) {
                            return base_op == f_op.get();
                        });
                assert(pos != subgraph.end());
                replace_op(*pos, conv_dw);

                fuse_op_to_predecessor(
                        post_op, subgraph, fuse_op_predecessor_offset);
                // conv_depthwise op may have additional output, which is
                // intermediate base conv output. It is needed in layout
                // propagation step.
                auto base_out_lt
                        = base_op->get_output_value(0)->get_logical_tensor();
                op_t &conv_dw_ref = *conv_dw.get();
                auto intermediate_out = std::make_shared<value_t>(
                        conv_dw_ref, conv_dw->num_outputs(), base_out_lt, true);
                conv_dw->connect_output(
                        conv_dw->num_outputs(), intermediate_out);

                subgraph.erase(pos);
                subgraph.emplace_back(conv_dw);

                with_op_replacement = true;
            } else {
                // unsupported post ops
                continue;
            }

            prm_attr.set_post_ops(pops);

            if (!with_op_replacement) {
                // remove the fused post_ops op
                fuse_op_to_predecessor(
                        post_op, subgraph, fuse_op_predecessor_offset);
            }
        }

        changed = true;
        return impl::status::success;
    };

    int cnt = 0;
    const int max_num_limit = static_cast<int>(subgraph.size());

    bool changed = true;
    do {
        auto ret = fuse_post_ops_func(subgraph, prm_attr_mgr, changed);
        if (ret != impl::status::success) return ret;
        cnt++;
    } while (changed && cnt <= max_num_limit);

    assertm(cnt <= max_num_limit + 1,
            "Failed to fuse all post ops since there has unsupported ones.");
    if (cnt > max_num_limit + 1) return impl::status::unsupported;
    return status::success;
}

impl::status_t fuse_zero_points(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    auto &prm_attr_mgr = sg->prm_attr_mgr_;

    std::vector<op_t *> zp_ops;

    std::set<op_t *> visited;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::add_zps
                || visited.count(cur_op.get()) != 0)
            continue;

        zp_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    for (auto &zp_op : zp_ops) {
        assertm(zp_op->num_outputs() == 1,
                "zp_op should have only one output value.");
        auto in_val = zp_op->get_input_values()[0];
        auto out_val = zp_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        bool is_input_zps = consumers.size() == 1
                && has_int8_support(consumers[0].get_op().get_kind());
        bool is_output_zps = in_val->has_producer()
                && has_int8_support(in_val->get_producer().get_kind());

        if (is_input_zps) {
            auto &next_op = consumers[0].get_op();
            auto offset = consumers[0].get_offset();
            if (offset == 0 || offset == 1) {
                int64_t key = -1;
                if (next_op.has_attr("primitive_attr_key")) {
                    key = next_op.get_attr<int64_t>("primitive_attr_key");
                } else {
                    key = prm_attr_mgr.init_attr();
                    next_op.set_attr<int64_t>("primitive_attr_key", key);
                }

                dnnl::primitive_attr &prm_attr = prm_attr_mgr.get_attr(key);

                auto zps = zp_op->get_attr<std::vector<int64_t>>("zps");
                bool not_all_zero
                        = std::find_if(zps.begin(), zps.end(),
                                  [](const int64_t &zp) { return zp != 0; })
                        != zps.end();
                if (not_all_zero) {
                    assertm(zps.size() == 1,
                            "zp attr only support scalar zp, need to use "
                            "runtime arg to support vector zp");

                    int64_t axis = zp_op->get_attr<int64_t>("axis");
                    int mask = zps.size() == 1 ? 0 : 1 << axis;

                    std::vector<int32_t> neg_int32_zps
                            = dnnl_impl::utils::fmap(zps, [](int64_t zp) {
                                  return static_cast<int32_t>(-zp);
                              });

                    prm_attr.set_zero_points(
                            offset == 0 ? DNNL_ARG_SRC : DNNL_ARG_WEIGHTS, mask,
                            neg_int32_zps);
                }
            }

            fuse_op_to_successor(zp_op, subgraph);
        } else if (is_output_zps) {
            auto in_val = zp_op->get_input_values()[0];
            auto &prv_op = in_val->get_producer();

            int64_t key = -1;
            if (prv_op.has_attr("primitive_attr_key")) {
                key = prv_op.get_attr<int64_t>("primitive_attr_key");
            } else {
                key = prm_attr_mgr.init_attr();
                prv_op.set_attr<int64_t>("primitive_attr_key", key);
            }

            dnnl::primitive_attr &prm_attr = prm_attr_mgr.get_attr(key);

            int64_t axis = zp_op->get_attr<int64_t>("axis");
            auto zps = zp_op->get_attr<std::vector<int64_t>>("zps");

            const size_t num_zps = zps.size();
            int mask = num_zps == 1 ? 0 : 1 << axis;
            std::vector<int32_t> int32_zps(num_zps, 0);
            for (size_t i = 0; i < num_zps; i++) {
                int32_zps[i] = static_cast<int32_t>(zps[i]);
            }
            prm_attr.set_zero_points(DNNL_ARG_DST, mask, int32_zps);

            fuse_op_to_predecessor(zp_op, subgraph);
        } else {
            // Nothing to do
        }
    }
    return impl::status::success;
}

impl::status_t fuse_mul_scales_add_zps(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<std::pair<op_t *, op_t *>> fuse_groups;
    std::set<op_t *> visited;
    for (const auto &cur_op : subgraph) {
        if ((cur_op->get_kind() != op_kind::mul_scales
                    && cur_op->get_kind() != op_kind::add_zps)
                || visited.count(cur_op.get()) != 0)
            continue;

        auto out_val = cur_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;

        auto &consumer_op = consumers[0].get_op();
        if ((consumer_op.get_kind() != op_kind::mul_scales
                    && consumer_op.get_kind() != op_kind::add_zps))
            continue;

        fuse_groups.emplace_back(
                std::pair<op_t *, op_t *> {cur_op.get(), &consumer_op});
        visited.insert(cur_op.get());
        visited.insert(&consumer_op);
    }

    if (fuse_groups.empty()) return impl::status::success;

    for (auto &fuse_ops : fuse_groups) {
        op_t *op1 = fuse_ops.first;
        op_t *op2 = fuse_ops.second;

        bool add_zps_first = op1->get_kind() == op_kind::add_zps
                && op2->get_kind() == op_kind::mul_scales;

        const int64_t axis = op1->get_attr<int64_t>("axis");
        const std::string &qtype = op1->get_attr<std::string>("qtype");
        const std::vector<float> &scales = add_zps_first
                ? op2->get_attr<std::vector<float>>("scales")
                : op1->get_attr<std::vector<float>>("scales");
        const std::vector<int64_t> &zps = add_zps_first
                ? op1->get_attr<std::vector<int64_t>>("zps")
                : op2->get_attr<std::vector<int64_t>>("zps");

        op_ptr fused_op = std::make_shared<op_t>(impl::op_kind::Reorder);
        fused_op->set_attr<bool>("change_layout", false);
        fused_op->set_attr<int64_t>("axis", axis);
        fused_op->set_attr<std::string>("qtype", qtype);
        fused_op->set_attr<std::vector<float>>("scales", scales);
        if (std::find_if(zps.begin(), zps.end(), [](const int64_t &zp) -> bool {
                return zp != 0;
            }) != zps.end()) { // not all zero
            std::string attr_name = add_zps_first ? "src_zps" : "dst_zps";
            fused_op->set_attr<std::vector<int64_t>>(attr_name, zps);
        }

        auto in_val = op1->get_input_value(0);
        in_val->remove_consumer(*op1, 0);
        fused_op->connect_input(0, in_val);

        auto out_val = op2->get_output_value(0);
        fused_op->add_output(out_val);
        out_val->set_producer(*fused_op);

        subgraph.emplace_back(fused_op);

        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op1](const op_ptr &f_op) { return op1 == f_op.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);

        pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op2](const op_ptr &f_op) { return op2 == f_op.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }
    return impl::status::success;
}

impl::status_t insert_bn_folding(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_t *> bn_ops;

    std::set<op_t *> visited;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::BatchNormInference
                || visited.count(cur_op.get()) != 0)
            continue;

        bn_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    for (auto &bn_op : bn_ops) {
        auto &prv_op = bn_op->get_input_value(0)->get_producer();
        if (prv_op.get_kind() != impl::op_kind::Convolution) continue;

        op_ptr bn_folding_op = std::make_shared<op_t>(op_kind::dnnl_bn_folding);
        bn_folding_op->merge_attributes(bn_op->get_attributes());
        bn_folding_op->set_attr<std::string>(
                "filter_format", prv_op.get_attr<std::string>("filter_format"));
        bn_folding_op->set_attr<bool>("with_bias", prv_op.num_inputs() == 3);

        // add conv's weight and bias (if exist) to bn folder op
        for (size_t i = 1; i < prv_op.num_inputs(); i++) {
            auto tmp = prv_op.get_input_value(i);
            tmp->remove_consumer(prv_op, i);
            tmp->add_consumer(*bn_folding_op, bn_folding_op->num_inputs());
            bn_folding_op->add_input(tmp);
        }

        // add bn's scale, shift, mean and variance to bn folder op
        for (size_t i = 1; i < bn_op->num_inputs(); i++) {
            auto tmp = bn_op->get_input_value(i);
            tmp->remove_consumer(*bn_op, i);
            tmp->add_consumer(*bn_folding_op, bn_folding_op->num_inputs());
            bn_folding_op->add_input(tmp);
        }

        auto updated_conv_wei = std::make_shared<value_t>(*bn_folding_op, 0,
                impl::empty_logical_tensor_with_default_id(), true);
        bn_folding_op->add_output(updated_conv_wei);
        updated_conv_wei->add_consumer(prv_op, 1);
        prv_op.connect_input(1, updated_conv_wei);

        auto updated_conv_bias = std::make_shared<value_t>(*bn_folding_op, 1,
                impl::empty_logical_tensor_with_default_id(), true);
        bn_folding_op->add_output(updated_conv_bias);
        updated_conv_bias->add_consumer(prv_op, 2);
        prv_op.connect_input(2, updated_conv_bias);

        auto bn_out_val = bn_op->get_output_value(0);
        prv_op.connect_output(0, bn_out_val);

        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [bn_op](const op_ptr &op) { return *bn_op == *op; });
        if (pos != subgraph.end()) subgraph.erase(pos);

        subgraph.emplace_back(bn_folding_op);
    }
    return impl::status::success;
}

impl::status_t conv_bwd_data_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<op_ptr> to_be_removed_ops;

    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::ConvolutionBackpropData)
            continue;

        // insert permute
        bool need_permute_0 = cur_op->has_attr("data_format")
                ? (cur_op->get_attr<std::string>("data_format") == "NXC")
                : false;
        bool need_permute_1 = cur_op->has_attr("filter_format")
                ? (cur_op->get_attr<std::string>("filter_format") == "XIO")
                : false;

        if (need_permute_0) {
            // input permute
            op_ptr in_perm_op = std::make_shared<impl::op_t>(op_kind::permute);
            in_perm_op->set_attr<std::string>("permute_kind", "permute");
            in_perm_op->set_attr<std::string>("from_format", "NXC");
            in_perm_op->set_attr<std::string>("to_format", "NCX");
            insert_op_before(in_perm_op, cur_op, 0);
            to_be_inserted_ops.emplace_back(in_perm_op);

            // output permute
            op_ptr out_perm_op = std::make_shared<impl::op_t>(op_kind::permute);
            out_perm_op->set_attr<std::string>("permute_kind", "permute");
            out_perm_op->set_attr<std::string>("from_format", "NCX");
            out_perm_op->set_attr<std::string>("to_format", "NXC");
            insert_op_after(out_perm_op, cur_op, 0);
            to_be_inserted_ops.emplace_back(out_perm_op);

            cur_op->set_attr<std::string>("data_format", "NCX");
        }

        if (need_permute_1) {
            op_ptr perm_op = std::make_shared<impl::op_t>(op_kind::permute);
            perm_op->set_attr<std::string>("permute_kind", "permute");
            perm_op->set_attr<std::string>("from_format", "XIO");
            perm_op->set_attr<std::string>("to_format", "OIX");
            insert_op_before(perm_op, cur_op, 1);
            to_be_inserted_ops.emplace_back(perm_op);
            cur_op->set_attr<std::string>("filter_format", "OIX");
        }

        // insert to_group
        auto groups = cur_op->get_attr<int64_t>("groups");
        if (groups > 1) {
            op_ptr to_group_op
                    = std::make_shared<impl::op_t>(op_kind::to_group);
            to_group_op->set_attr<int64_t>("groups", groups);
            insert_op_before(to_group_op, cur_op, 1);
            to_be_inserted_ops.emplace_back(to_group_op);
            cur_op->set_attr<int64_t>("groups", 1);
        }

        // replace original op to dnnl specific op
        op_ptr new_op = std::make_shared<op_t>(op_kind::dnnl_conv_bwd_data);
        replace_op(cur_op, new_op);
        to_be_inserted_ops.emplace_back(new_op);
        to_be_removed_ops.emplace_back(cur_op);
    }

    for (const auto &op : to_be_inserted_ops) {
        subgraph.emplace_back(op);
    }

    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }
    return impl::status::success;
}

impl::status_t fuse_mul_sigmoid_to_swish(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<std::vector<op_t *>> swish_patterns;
    std::vector<size_t> mul_other_offsets;

    // find all swish pattern in subgraph
    std::set<op_t *> visited;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::Sigmoid
                || visited.count(cur_op.get()) != 0)
            continue;
        visited.insert(cur_op.get());

        // check if the sigmoid op belongs to a swish pattern.
        // A swish pattern is composed by a sigmoid op and a multiply op:
        //        any
        //       /   \
        //  sigmoid   |
        //       \   /
        //      multiply
        //         |
        //        any
        auto sigmoid_out = cur_op->get_output_value(0);
        auto sigmoid_csm = sigmoid_out->get_consumers();
        if (sigmoid_csm.size() != 1) continue;

        auto &csm_op = sigmoid_csm[0].get_op();
        if (csm_op.get_kind() != impl::op_kind::Multiply) continue;

        size_t offset = sigmoid_csm[0].get_offset(); // offset should be 0 or 1
        size_t mul_other_offset = 1 - offset;
        auto mul_other_in = csm_op.get_input_value(mul_other_offset);
        auto sigmoid_in = cur_op->get_input_value(0);
        if (mul_other_in.get() != sigmoid_in.get()) continue;

        // all checks passed, found a swish pattern
        std::vector<op_t *> pattern {cur_op.get(), &csm_op};
        swish_patterns.emplace_back(pattern);
        mul_other_offsets.emplace_back(mul_other_offset);
    }

    if (swish_patterns.empty()) return impl::status::success;

    // fuse swish pattern to a swish op
    for (size_t i = 0; i < swish_patterns.size(); i++) {
        op_t *sigmoid_op = swish_patterns[i][0];
        op_t *mul_op = swish_patterns[i][1];
        size_t mul_other_offset = mul_other_offsets[i];

        op_ptr swish_op = std::make_shared<op_t>(op_kind::dnnl_swish);
        swish_op->set_attr<float>("alpha", (float)1.0);

        auto in_val = sigmoid_op->get_input_value(0);
        in_val->remove_consumer(*sigmoid_op, 0);
        in_val->remove_consumer(*mul_op, mul_other_offset);
        swish_op->connect_input(0, in_val);

        auto out_val = mul_op->get_output_value(0);
        swish_op->add_output(out_val);
        out_val->set_producer(*swish_op);

        subgraph.emplace_back(swish_op);

        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [sigmoid_op](const op_ptr &f_op) {
                    return sigmoid_op == f_op.get();
                });
        if (pos != subgraph.end()) subgraph.erase(pos);

        pos = std::find_if(subgraph.begin(), subgraph.end(),
                [mul_op](const op_ptr &f_op) { return mul_op == f_op.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }
    return impl::status::success;
}

impl::status_t fuse_typecast_to_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::MatMul) continue;
        auto &in0 = cur_op->get_input_value(0)->get_producer();
        auto &in1 = cur_op->get_input_value(1)->get_producer();
        if (in0.get_kind() == impl::op_kind::TypeCast
                && in1.get_kind() == impl::op_kind::TypeCast
                && in0.get_input_value(0)->has_producer()
                && in1.get_input_value(0)->has_producer()
                && in0.get_input_value(0)->get_producer().get_kind()
                        == impl::op_kind::Dequantize
                && in1.get_input_value(0)->get_producer().get_kind()
                        == impl::op_kind::Dequantize)
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in0, &in1});
    }

    for (auto &fusion_group : fusion_groups) {
        op_t *matmul_op = fusion_group[0];
        op_t *in0 = fusion_group[1];
        op_t *in1 = fusion_group[2];

        op_ptr q_matmul_op = std::make_shared<op_t>(impl::op_kind::MatMul);
        q_matmul_op->merge_attributes(matmul_op->get_attributes());

        // update the connection relationship between matmul and typecast ops
        auto in0_value = in0->get_input_value(0);
        auto in1_value = in1->get_input_value(0);
        q_matmul_op->connect_input(0, in0_value);
        q_matmul_op->connect_input(1, in1_value);
        in0_value->remove_consumer(*in0, 0);
        in1_value->remove_consumer(*in1, 0);

        // handle bias
        if (matmul_op->num_inputs() == 3) {
            auto bias_value = matmul_op->get_input_value(2);
            bias_value->remove_consumer(*matmul_op, 2);
            q_matmul_op->connect_input(2, bias_value);
        }

        auto out_val = matmul_op->get_output_value(0);
        q_matmul_op->add_output(out_val);
        out_val->set_producer(*q_matmul_op);
        out_val->set_data_type(
                matmul_op->get_input_value(0)->get_logical_tensor().data_type);

        // delete original matmul and typecast ops
        for (auto &del_op : fusion_group) {
            auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [del_op](const op_ptr &f_op) {
                        return del_op == f_op.get();
                    });
            if (pos != subgraph.end()) subgraph.erase(pos);
        }

        subgraph.emplace_back(q_matmul_op);
    }
    return impl::status::success;
}

impl::status_t fuse_typecast_to_add(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::Add) continue;
        if (!(cur_op->get_input_value(0)->has_producer()
                    && cur_op->get_input_value(1)->has_producer()))
            continue;
        auto &in0 = cur_op->get_input_value(0)->get_producer();
        auto &in1 = cur_op->get_input_value(1)->get_producer();
        if (in0.get_kind() == impl::op_kind::TypeCast
                && in1.get_kind() == impl::op_kind::MatMul
                && in0.get_input_value(0)->get_producer().get_kind()
                        == impl::op_kind::Dequantize) {
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in0});
        } else if (in1.get_kind() == impl::op_kind::TypeCast
                && in0.get_kind() == impl::op_kind::MatMul
                && in1.get_input_value(0)->get_producer().get_kind()
                        == impl::op_kind::Dequantize) {
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in1});
        } else {
        }
    }

    for (auto &fusion_group : fusion_groups) {
        op_t *add_op = fusion_group[0];
        op_t *typecast_op = fusion_group[1];

        op_ptr new_add_op = std::make_shared<op_t>(impl::op_kind::Add);
        new_add_op->merge_attributes(add_op->get_attributes());

        // update the connection relationship between add and typecast ops
        auto tc_in = typecast_op->get_input_value(0);
        auto in0 = add_op->get_input_value(0);
        auto in1 = add_op->get_input_value(1);
        in0->remove_consumer(*add_op, 0);
        in1->remove_consumer(*add_op, 1);
        if (in0->get_producer().get_kind() == impl::op_kind::TypeCast) {
            new_add_op->connect_input(0, tc_in);
            new_add_op->connect_input(1, in1);
            tc_in->remove_consumer(*typecast_op, 0);
            tc_in->set_data_type(in0->get_logical_tensor().data_type);
        } else {
            new_add_op->connect_input(1, tc_in);
            new_add_op->connect_input(0, in0);
            tc_in->remove_consumer(*typecast_op, 0);
            tc_in->set_data_type(in1->get_logical_tensor().data_type);
        }

        auto out_val = add_op->get_output_value(0);
        new_add_op->add_output(out_val);
        out_val->set_producer(*new_add_op);

        // delete original matmul and typecast ops
        for (auto &del_op : fusion_group) {
            auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [del_op](const op_ptr &f_op) {
                        return del_op == f_op.get();
                    });
            if (pos != subgraph.end()) subgraph.erase(pos);
        }

        subgraph.emplace_back(new_add_op);
    }
    return impl::status::success;
}

impl::status_t fuse_post_typecast_to_matmul(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    const std::set<op_kind_t> post_ops_kinds {impl::op_kind::GELU};
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::MatMul) continue;
        auto out = cur_op->get_output_value(0);
        if (out->get_consumers().size() != 1) continue;
        auto &next_op = out->get_consumers()[0].get_op();
        if (post_ops_kinds.count(next_op.get_kind())) {
            auto post_out = next_op.get_output_value(0);
            if (post_out->get_consumers().size() != 1) continue;
            auto &tc_op = post_out->get_consumers()[0].get_op();
            if (tc_op.get_kind() != impl::op_kind::TypeCast) continue;
            auto tc_out = tc_op.get_output_value(0);
            if (tc_out->get_consumers().size() != 1) continue;
            auto &q_op = tc_out->get_consumers()[0].get_op();
            if (q_op.get_kind() != impl::op_kind::Quantize) continue;
            post_out->remove_consumer(tc_op, 0);
            tc_out->remove_consumer(q_op, 0);
            q_op.connect_input(0, post_out);
            out->set_data_type(impl::data_type::f32);
            post_out->set_data_type(impl::data_type::f32);
            fusion_groups.emplace_back(std::vector<op_t *> {&tc_op});
        } else {
            if (next_op.get_kind() != impl::op_kind::TypeCast) continue;
            auto tc_out = next_op.get_output_value(0);
            if (tc_out->get_consumers().size() != 1) continue;
            auto &q_op = tc_out->get_consumers()[0].get_op();
            if (q_op.get_kind() != impl::op_kind::Quantize) continue;
            out->remove_consumer(next_op, 0);
            tc_out->remove_consumer(q_op, 0);
            q_op.connect_input(0, out);
            out->set_data_type(impl::data_type::f32);
            fusion_groups.emplace_back(std::vector<op_t *> {&next_op});
        }
    }

    for (auto &fusion_group : fusion_groups)
        // delete original typecast ops
        for (auto &del_op : fusion_group) {
            auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [del_op](const op_ptr &f_op) {
                        return del_op == f_op.get();
                    });
            if (pos != subgraph.end()) subgraph.erase(pos);
        }
    return impl::status::success;
}

impl::status_t lower_down_to_dnnl_batchnorm(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<op_ptr> to_be_removed_ops, to_be_inserted_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::BatchNormInference
                && cur_op->get_kind()
                        != impl::op_kind::BatchNormForwardTraining)
            continue;

        // create new dnnl_batchnorm
        op_ptr new_op = std::make_shared<op_t>(op_kind::dnnl_batchnorm);

        // decide if this is for training or inference
        if (cur_op->get_kind() == impl::op_kind::BatchNormInference)
            new_op->set_attr<bool>("is_training", false);
        else
            new_op->set_attr<bool>("is_training", true);

        // replace original oneDNN Graph ops with dnnl_batchnorm
        replace_op(cur_op, new_op);
        to_be_inserted_ops.emplace_back(new_op);
        to_be_removed_ops.emplace_back(cur_op);
    }

    for (const auto &op : to_be_inserted_ops) {
        subgraph.emplace_back(op);
    }

    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }

    return impl::status::success;
}

impl::status_t fuse_to_dnnl_sum(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    op_ptr sum_op = std::make_shared<op_t>(op_kind::dnnl_sum);

    auto graph_in_vals = impl::graph_t(subgraph).get_input_values();
    auto graph_out_vals = impl::graph_t(subgraph).get_output_values();

    int input_idx = 0;
    for (auto &cur_op : subgraph) {
        // reconnect graph's input val to dnnl_sum
        auto input_values = cur_op->get_input_values();
        for (auto &in_val : input_values) {
            if (std::find(graph_in_vals.begin(), graph_in_vals.end(),
                        in_val.get())
                    == graph_in_vals.end())
                continue;

            in_val->remove_consumer(*cur_op, 0);
            sum_op->connect_input(input_idx++, in_val);
        }

        // reconnect graph's output val to dnnl_sum
        auto output_values = cur_op->get_output_values();
        for (auto &out_val : output_values) {
            if (std::find(graph_out_vals.begin(), graph_out_vals.end(),
                        out_val.get())
                    == graph_out_vals.end())
                continue;

            sum_op->add_output(out_val);
            out_val->set_producer(*sum_op);
        }
    }
    // remove original add/mul ops
    subgraph.clear();

    // add dnnl_sum to subgraph
    subgraph.emplace_back(sum_op);

    return impl::status::success;
}

impl::status_t binary_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<op_ptr> to_be_removed_ops;

    const static std::set<impl::op_kind_t> binary_op_set = {impl::op_kind::Add,
            impl::op_kind::Multiply, impl::op_kind::Divide,
            impl::op_kind::Minimum, impl::op_kind::Maximum};

    auto &subgraph = sg->get_mutable_ops();

    for (auto &cur_op : subgraph) {
        if (!binary_op_set.count(cur_op->get_kind())) continue;

        // check doable
        auto src0_lt = cur_op->get_input_value(0)->get_logical_tensor();
        auto src1_lt = cur_op->get_input_value(1)->get_logical_tensor();
        if (!binary_doable(ltw(src0_lt).vdims(), ltw(src1_lt).vdims())) {
            return status::invalid_shape;
        }

        // insert expand op
        int32_t src0_ndims = src0_lt.ndims;
        int32_t src1_ndims = src1_lt.ndims;
        int32_t target_ndims = std::max(src0_ndims, src1_ndims);
        std::vector<int32_t> in_ndims {src0_ndims, src1_ndims};
        for (size_t i = 0; i < cur_op->num_inputs(); ++i) {
            if (in_ndims[i] == target_ndims) { continue; }

            auto expand_op = std::make_shared<op_t>(op_kind::expand);
            expand_op->set_attr<int64_t>("expand_to", target_ndims);
            insert_op_before(expand_op, cur_op, i);
            to_be_inserted_ops.emplace_back(expand_op);
        }

        // replace original op to dnnl specific op
        op_ptr new_op = std::make_shared<op_t>(op_kind::dnnl_binary);
        new_op->set_attr<int64_t>(
                "alg_kind", static_cast<int64_t>(cur_op->get_kind()));
        replace_op(cur_op, new_op);
        to_be_inserted_ops.emplace_back(new_op);
        to_be_removed_ops.emplace_back(cur_op);
    }

    for (const auto &op : to_be_inserted_ops) {
        subgraph.emplace_back(op);
    }

    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }

    return impl::status::success;
}

impl::status_t fuse_adjacent_reorders(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<op_ptr> to_be_removed_ops;

    const static std::set<impl::op_kind_t> reorder_op_set
            = {impl::op_kind::Reorder, op_kind::dnnl_u8_to_s8};

    auto &subgraph = sg->get_mutable_ops();

    auto fuse_two_adjacent_reorders = [](std::vector<op_ptr> &subgraph,
                                              bool &changed) -> impl::status_t {
        std::vector<std::pair<op_t *, op_t *>> fuse_groups;

        std::set<const op_t *> visited;
        impl::topo_order_visit(
                impl::graph_t(subgraph).get_output_ops(), [&](impl::op_t *op) {
                    if (!reorder_op_set.count(op->get_kind())
                            || visited.count(op) != 0)
                        return impl::status::success;

                    assertm(op->num_outputs() == 1,
                            "cur_op should have only one output value.");
                    auto out_val = op->get_output_values()[0];
                    auto consumers = out_val->get_consumers();
                    if (consumers.size() != 1) return impl::status::success;
                    auto &next_op = consumers[0].get_op();

                    // check if fusible
                    if (reorder_op_set.count(next_op.get_kind()) == 0) {
                        return impl::status::success;
                    }

                    // if reorder do per channel scaling, their axis should be
                    // same
                    int64_t cur_axis = op->has_attr("axis")
                            ? op->get_attr<int64_t>("axis")
                            : -1;
                    int64_t next_axis = next_op.has_attr("axis")
                            ? next_op.get_attr<int64_t>("axis")
                            : -1;
                    if (cur_axis != -1 && next_axis != -1
                            && cur_axis != next_axis) {
                        return impl::status::success;
                    }

                    // Skip fusion if reorder's output has extra info, because
                    // oneDNN doesn't have good supports for such fused cases.
                    // TODO(qun) improve the skip rule
                    auto fused_out_lt
                            = next_op.get_output_value(0)->get_logical_tensor();
                    auto fused_out_md = make_dnnl_memory_desc(fused_out_lt);
                    if (fused_out_md.data.extra.flags
                            != dnnl_memory_extra_flag_none) {
                        return impl::status::success;
                    }

                    // push fusible pair to fuse group for later fusion
                    fuse_groups.emplace_back(
                            std::pair<op_t *, op_t *> {op, &next_op});
                    visited.insert(op);
                    visited.insert(&next_op);
                    return impl::status::success;
                });

        if (fuse_groups.empty()) {
            changed = false;
            return impl::status::success;
        }

        for (auto &fuse_group : fuse_groups) {
            auto op1 = fuse_group.first;
            auto op2 = fuse_group.second;

            // get the src_zps, dst_zps and scales.
            auto get_scales_zps = [](const op_t *op, std::vector<float> &scales,
                                          std::vector<int64_t> &src_zps,
                                          std::vector<int64_t> &dst_zps,
                                          size_t &num) {
                if (op->get_kind() == op_kind::dnnl_u8_to_s8) {
                    scales = std::vector<float> {1.0};
                    src_zps = std::vector<int64_t> {0};
                    dst_zps = std::vector<int64_t> {-128};
                } else {
                    scales = op->has_attr("scales")
                            ? op->get_attr<std::vector<float>>("scales")
                            : std::vector<float> {1.0};
                    src_zps = op->has_attr("src_zps")
                            ? op->get_attr<std::vector<int64_t>>("src_zps")
                            : std::vector<int64_t> {0};
                    dst_zps = op->has_attr("dst_zps")
                            ? op->get_attr<std::vector<int64_t>>("dst_zps")
                            : std::vector<int64_t> {0};
                }
                num = std::max(std::max(scales.size(), src_zps.size()),
                        dst_zps.size());
            };

            size_t num1, num2;
            std::vector<float> scales1, scales2;
            std::vector<int64_t> src_zps1, dst_zps1, src_zps2, dst_zps2;
            get_scales_zps(op1, scales1, src_zps1, dst_zps1, num1);
            get_scales_zps(op2, scales2, src_zps2, dst_zps2, num2);

            // broadcast to same dimension
            size_t max_num = std::max(num1, num2);
            if (scales1.size() < max_num) {
                scales1.resize(max_num, scales1[0]);
            }
            if (src_zps1.size() < max_num) {
                src_zps1.resize(max_num, src_zps1[0]);
            }
            if (dst_zps1.size() < max_num) {
                dst_zps1.resize(max_num, dst_zps1[0]);
            }

            if (scales2.size() < max_num) {
                scales2.resize(max_num, scales2[0]);
            }
            if (src_zps2.size() < max_num) {
                src_zps2.resize(max_num, src_zps2[0]);
            }
            if (dst_zps2.size() < max_num) {
                dst_zps2.resize(max_num, dst_zps2[0]);
            }

            // fuse the scales and zps according to the the formula of reorder
            // op: dst = scales*(src+src_zps)+dst_zps;
            std::vector<float> fused_scales;
            std::vector<int64_t> fused_src_zps, fused_dst_zps;
            fused_src_zps = src_zps1;
            fused_scales.reserve(max_num);
            fused_dst_zps.reserve(max_num);
            for (size_t i = 0; i < max_num; i++) {
                fused_scales.emplace_back(scales1[i] * scales2[i]);
                fused_dst_zps.emplace_back(
                        scales2[i] * (dst_zps1[i] + src_zps2[i]) + dst_zps2[i]);
            }

            int64_t axis = -1;
            if (op1->has_attr("axis")) {
                axis = op1->get_attr<int64_t>("axis");
            }
            if (op2->has_attr("axis")) {
                axis = op2->get_attr<int64_t>("axis");
            }

            bool change_layout
                    = (op1->has_attr("change_layout")
                              && op1->get_attr<bool>("change_layout"))
                    || (op2->has_attr("change_layout")
                            && op2->get_attr<bool>("change_layout"));

            // create fused op
            op_ptr fused_op = std::make_shared<op_t>(impl::op_kind::Reorder);
            fused_op->set_attr<bool>("change_layout", change_layout);
            if (axis != -1) fused_op->set_attr<int64_t>("axis", axis);

            if (std::find_if(fused_scales.begin(), fused_scales.end(),
                        [](const float &s) { return std::abs(s - 1.f) > 1e-6; })
                    != fused_scales.end()) {
                fused_op->set_attr<std::vector<float>>("scales", fused_scales);
            }

            if (std::find_if(fused_src_zps.begin(), fused_src_zps.end(),
                        [](const int64_t &zp) { return zp != 0; })
                    != fused_src_zps.end()) {
                fused_op->set_attr<std::vector<int64_t>>(
                        "src_zps", fused_src_zps);
            }

            if (std::find_if(fused_dst_zps.begin(), fused_dst_zps.end(),
                        [](const int64_t &zp) { return zp != 0; })
                    != fused_dst_zps.end()) {
                fused_op->set_attr<std::vector<int64_t>>(
                        "dst_zps", fused_dst_zps);
            }

            // connect fused op to subgraph and remove original ones
            auto in_val = op1->get_input_value(0);
            in_val->remove_consumer(*op1, 0);
            fused_op->connect_input(0, in_val);

            auto out_val = op2->get_output_value(0);
            fused_op->add_output(out_val);
            out_val->set_producer(*fused_op);

            subgraph.emplace_back(fused_op);

            auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [op1](const op_ptr &f_op) { return op1 == f_op.get(); });
            if (pos != subgraph.end()) subgraph.erase(pos);

            pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [op2](const op_ptr &f_op) { return op2 == f_op.get(); });
            if (pos != subgraph.end()) subgraph.erase(pos);
        }
        return impl::status::success;
    };

    int cnt = 0;
    const int max_num_limit = static_cast<int>(subgraph.size());

    bool changed = true;
    do {
        auto ret = fuse_two_adjacent_reorders(subgraph, changed);
        if (ret != impl::status::success) return ret;
        cnt++;
    } while (changed && cnt <= max_num_limit);

    assertm(cnt <= max_num_limit + 1, "reorder fusion failed.");
    if (cnt > max_num_limit + 1) return impl::status::unsupported;

    return impl::status::success;
}

impl::status_t fuse_typecast_to_quantize(std::shared_ptr<subgraph_t> &sg) {
    auto &subgraph = sg->get_mutable_ops();
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::Quantize
                || !cur_op->get_input_value(0)->has_producer())
            continue;

        auto &in = cur_op->get_input_value(0)->get_producer();
        if (in.get_kind() == impl::op_kind::TypeCast)
            fusion_groups.emplace_back(std::vector<op_t *> {cur_op.get(), &in});
    }

    for (auto &fusion_group : fusion_groups) {
        op_t *in0 = fusion_group[1];
        fuse_op_to_successor(in0, subgraph);
    }
    return impl::status::success;
}

impl::status_t eltwise_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> to_be_inserted_ops;
    std::vector<op_ptr> to_be_removed_ops;

    auto &subgraph = sg->get_mutable_ops();

    for (auto &cur_op : subgraph) {
        if (!is_eltwise_kind(cur_op->get_kind())) continue;
        // replace original op to dnnl specific op
        op_ptr new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);

        // record the original op kind, which will mapped to dnnl alg kind
        // during compilation
        new_op->set_attr<int64_t>(
                "alg_kind", static_cast<int64_t>(cur_op->get_kind()));

        // convert the frontend op attr to backend op attr
        if (cur_op->has_attr("alpha")) {
            new_op->set_attr<float>("alpha", cur_op->get_attr<float>("alpha"));
        } else if (cur_op->has_attr("min")) {
            new_op->set_attr<float>("alpha", cur_op->get_attr<float>("min"));
        } else {
            new_op->set_attr<float>("alpha", 0);
        }

        if (cur_op->has_attr("beta")) {
            new_op->set_attr<float>("beta", cur_op->get_attr<float>("beta"));
        } else if (cur_op->has_attr("max")) {
            new_op->set_attr<float>("beta", cur_op->get_attr<float>("max"));
        } else {
            new_op->set_attr<float>("beta", 0);
        }

        replace_op(cur_op, new_op);
        to_be_inserted_ops.emplace_back(new_op);
        to_be_removed_ops.emplace_back(cur_op);
    }

    for (const auto &op : to_be_inserted_ops) {
        subgraph.emplace_back(op);
    }

    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }

    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

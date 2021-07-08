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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "interface/c_types_map.hpp"
#include "interface/op_schema.hpp"

#include "lower_down.hpp"
#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_t = impl::op_t;
using op_ptr = std::shared_ptr<impl::op_t>;
using value_ptr = std::shared_ptr<impl::value_t>;
using ltw = impl::logical_tensor_wrapper;

// define epsilon to avoid divide-by-zero when computing the inverse of scales
// note that it may affect the model's accuracy
const float scale_eps = 1e-9f;

static bool has_optional_bias(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::Convolution, op_kind::MatMul};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool has_int8_support(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::dnnl_convolution, op_kind::MatMul};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool has_post_ops(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::dnnl_convolution, op_kind::MatMul};
    return ops.count(kind) != 0;
}

void check_with_bias(std::vector<op_ptr> &subgraph) {
    for (auto &cur_op : subgraph) {
        if (!has_optional_bias(cur_op->get_kind())) continue;
        if (cur_op->num_inputs() == 3) {
            cur_op->set_attr<bool>("with_bias", true);
        } else {
            cur_op->set_attr<bool>("with_bias", false);
        }
    }
}

void fuse_bias_add(std::vector<op_ptr> &subgraph) {
    std::vector<op_t *> bias_add_ops;

    std::set<op_t *> visited;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::BiasAdd
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
}

void split_quant_dequant(std::vector<op_ptr> &subgraph) {
    std::vector<op_ptr> q_dq_ops;
    for (auto &cur_op : subgraph) {
        if (cur_op->get_kind() == op_kind::Quantize
                || cur_op->get_kind() == op_kind::Dequantize) {
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
        if (cur_op->get_kind() == op_kind::Dequantize) {
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

            std::vector<float> inv_scales = dnnl_impl::utils::fmap(
                    scales, [](float s) { return 1.f / (s + scale_eps); });
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
    return;
}

void fuse_output_scales(
        std::vector<op_ptr> &subgraph, primitive_attr_mgr &prm_attr_mgr) {
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
}

void folding_mul_scales(std::vector<op_ptr> &subgraph) {
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
}

void fuse_to_int8_conv(std::vector<op_ptr> &subgraph) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &op : subgraph) {
        if (op->get_kind() == op_kind::Convolution) {
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

        op_ptr qconv_op = std::make_shared<op_t>(op_kind::dnnl_convolution);
        op_ptr mul_op = std::make_shared<op_t>(op_kind::mul_scales);

        qconv_op->merge_attributes(conv_op->get_attributes());

        auto dq_src_scales = in0.get_attr<std::vector<float>>("scales");
        auto dq_wei_scales = in1.get_attr<std::vector<float>>("scales");
        std::vector<float> fused_scale(
                std::max(dq_src_scales.size(), dq_wei_scales.size()), 0);
        if (dq_src_scales.size() > dq_wei_scales.size()) {
            for (int i = 0; i < dq_src_scales.size(); i++)
                fused_scale[i] = (dq_src_scales[i] * dq_wei_scales[0]);
            mul_op->set_attr<int64_t>("axis", in0.get_attr<int64_t>("axis"));
            mul_op->set_attr<std::string>(
                    "qtype", in0.get_attr<std::string>("qtype"));
        } else {
            for (int i = 0; i < dq_wei_scales.size(); i++)
                fused_scale[i] = (dq_src_scales[0] * dq_wei_scales[i]);
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

            std::vector<float> inv_scales(fused_scale.size());
            for (int i = 0; i < fused_scale.size(); i++)
                // add epsilon to avoid divide zero
                inv_scales[i] = 1.f / (fused_scale[i] + scale_eps);

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
}

void fuse_to_int8_matmul(std::vector<op_ptr> &subgraph) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != op_kind::MatMul) continue;
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

        op_ptr q_matmul_op = std::make_shared<op_t>(op_kind::MatMul);
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
            mul_scales_op->set_attr<int64_t>("axis",
                    begin_op.get_output_value(0)->get_logical_tensor().ndims
                            - 1);
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
            std::vector<float> inv_scales(fused_scales.size(), 1.f);
            for (size_t i = 0; i < inv_scales.size(); ++i)
                inv_scales[i] = 1.f / (fused_scales[i] + scale_eps);
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
}

void fuse_to_int8_pool(std::vector<op_ptr> &subgraph) {
    std::vector<op_t *> fusion_ops;
    for (const auto &cur_op : subgraph) {
        // currently only support int8 MaxPooling
        if (cur_op->get_kind() != op_kind::MaxPool) continue;
        fusion_ops.emplace_back(cur_op.get());
    }

    if (fusion_ops.empty()) return;
    for (auto &pool_op : fusion_ops) {
        op_ptr q_pool_op = std::make_shared<op_t>(op_kind::dnnl_maxpool);
        q_pool_op->merge_attributes(pool_op->get_attributes());

        // oneDNN int8 pooling primitive doesn't require scales and zps, so we
        // just fuse the Quantize and Dequantize op into this newly created pool
        // op
        op_t &dequant_op = pool_op->get_input_value(0)->get_producer();
        assertm(dequant_op.get_kind() == op_kind::Dequantize,
                "the predecessor op of pool should be a Dequant op.");
        value_ptr in_value = dequant_op.get_input_value(0);
        q_pool_op->connect_input(0, in_value);
        in_value->remove_consumer(dequant_op, 0);

        assertm(pool_op->get_output_value(0)->get_consumers().size() == 1,
                "pooling's successor Quant op should only have one consumer.");
        const op_t &quant_op
                = pool_op->get_output_value(0)->get_consumers()[0].get_op();
        assertm(quant_op.get_kind() == op_kind::Quantize,
                "the successor op of pool should be a Quantize op.");
        value_ptr out_value = quant_op.get_output_value(0);
        q_pool_op->add_output(out_value);
        out_value->set_producer(*q_pool_op);

        // delete original pool and Quant/Dequant ops
        std::vector<const op_t *> deleted_ops {pool_op, &dequant_op, &quant_op};
        for (const auto &del_op : deleted_ops) {
            auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                    [del_op](const op_ptr &f_op) {
                        return del_op == f_op.get();
                    });
            if (pos != subgraph.end()) subgraph.erase(pos);
        }

        subgraph.emplace_back(q_pool_op);
    }
}

status_t fuse_post_ops(
        std::vector<op_ptr> &subgraph, primitive_attr_mgr &prm_attr_mgr) {
    const std::set<op_kind_t> post_ops_kinds {op_kind::ReLU, op_kind::GELU,
            op_kind::Sigmoid, op_kind::Elu, op_kind::HardTanh, op_kind::Abs,
            op_kind::Sqrt, op_kind::Square, op_kind::Tanh, op_kind::Add};

    const std::set<op_kind_t> eltwise_kinds {op_kind::ReLU, op_kind::GELU,
            op_kind::Sigmoid, op_kind::Elu, op_kind::HardTanh, op_kind::Abs,
            op_kind::Sqrt, op_kind::Square, op_kind::Tanh};

    const std::map<op_kind_t, dnnl::algorithm> eltwise_alg_map {
            {op_kind::ReLU, dnnl::algorithm::eltwise_relu},
            {op_kind::GELU, dnnl::algorithm::eltwise_gelu_erf},
            {op_kind::Sigmoid, dnnl::algorithm::eltwise_logistic},
            {op_kind::Elu, dnnl::algorithm::eltwise_elu},
            {op_kind::HardTanh, dnnl::algorithm::eltwise_clip},
            {op_kind::Abs, dnnl::algorithm::eltwise_abs},
            {op_kind::Sqrt, dnnl::algorithm::eltwise_sqrt},
            {op_kind::Square, dnnl::algorithm::eltwise_square},
            {op_kind::Tanh, dnnl::algorithm::eltwise_tanh}};

    // lambda function to fuse one post op into base primitive
    auto fuse_post_ops_func = [&](std::vector<op_ptr> &subgraph,
                                      primitive_attr_mgr &prm_attr_mgr) {
        std::vector<std::pair<op_t *, op_t *>> fuse_groups;

        std::set<op_t *> visited;
        for (auto &cur_op : subgraph) {
            if (!has_post_ops(cur_op->get_kind())
                    || visited.count(cur_op.get()) != 0)
                continue;

            assertm(cur_op->num_outputs() == 1,
                    "cur_op should have only one output value.");
            auto out_val = cur_op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            if (consumers.empty()) continue;

            auto &next_op = consumers[0].get_op();
            if (post_ops_kinds.count(next_op.get_kind()) == 0) continue;

            fuse_groups.emplace_back(
                    std::pair<op_t *, op_t *> {cur_op.get(), &next_op});
            visited.insert(cur_op.get());
            visited.insert(&next_op);
        }

        if (fuse_groups.empty()) return false;

        for (auto &fuse_group : fuse_groups) {
            auto base_op = fuse_group.first;
            auto post_op = fuse_group.second;

            int64_t key = -1;
            if (base_op->has_attr("primitive_attr_key")) {
                key = base_op->get_attr<int64_t>("primitive_attr_key");
            } else {
                key = prm_attr_mgr.init_attr();
                base_op->set_attr<int64_t>("primitive_attr_key", key);
            }

            dnnl::primitive_attr &prm_attr = prm_attr_mgr.get_attr(key);

            dnnl::post_ops pops = prm_attr.get_post_ops();

            if (eltwise_kinds.count(post_op->get_kind()) != 0) {
                float scale = 1.f;
                auto alg = eltwise_alg_map.at(post_op->get_kind());
                float alpha = 0;
                float beta = 0;

                if (post_op->has_attr("alpha")) {
                    alpha = post_op->get_attr<float>("alpha");
                } else if (post_op->has_attr("min")) {
                    alpha = post_op->get_attr<float>("min");
                }

                if (post_op->has_attr("beta")) {
                    beta = post_op->get_attr<float>("beta");
                } else if (post_op->has_attr("max")) {
                    beta = post_op->get_attr<float>("max");
                }

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
            } else if (post_op->get_kind() == op_kind::Add) {
                auto in_val1 = post_op->get_input_value(1);
                if (in_val1->has_producer()) {
                    // for int8 cases
                    auto &mul_scale_op = in_val1->get_producer();
                    auto scales = mul_scale_op.get_attr<std::vector<float>>(
                            "scales");
                    assert(scales.size() == 1); // per tensor

                    auto tmp = mul_scale_op.get_input_value(0);
                    auto &add_zps_op = tmp->get_producer();
                    auto zps = add_zps_op.get_attr<std::vector<int64_t>>("zps");
                    assert(scales.size() == zps.size()
                            && zps[0] == 0); // symmetric

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
                    pops.append_sum(scales[0]);
                } else {
                    // for fp32 cases, the another input of sum op have no
                    // producer
                    pops.append_sum(1.f);
                }

                base_op->set_attr<bool>("with_sum", true);
            } else {
                // unsupported post ops
                continue;
            }

            prm_attr.set_post_ops(pops);

            // remove the fused post_ops op
            fuse_op_to_predecessor(post_op, subgraph);
        }

        return true;
    };

    int cnt = 0;
    const int max_num_limit = static_cast<int>(subgraph.size());

    bool changed = true;
    do {
        changed = fuse_post_ops_func(subgraph, prm_attr_mgr);
        cnt++;
    } while (changed && cnt <= max_num_limit);

    assertm(cnt <= max_num_limit,
            "Failed to fuse all post ops since there has unsupported ones.");
    if (cnt > max_num_limit) return impl::status::unsupported;
    return status::success;
}

void fuse_zero_points(
        std::vector<op_ptr> &subgraph, primitive_attr_mgr &prm_attr_mgr) {
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
        auto out_val = zp_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (!consumers.empty()) {
            auto &next_op = consumers[0].get_op();
            auto offset = consumers[0].get_offset();
            // only fuse conv/matmul's src and weight zps
            if (!has_int8_support(next_op.get_kind())) continue;

            if (offset == 0) {
                int64_t key = -1;
                if (next_op.has_attr("primitive_attr_key")) {
                    key = next_op.get_attr<int64_t>("primitive_attr_key");
                } else {
                    key = prm_attr_mgr.init_attr();
                    next_op.set_attr<int64_t>("primitive_attr_key", key);
                }

                dnnl::primitive_attr &prm_attr = prm_attr_mgr.get_attr(key);

                int64_t axis = zp_op->get_attr<int64_t>("axis");
                auto zps = zp_op->get_attr<std::vector<int64_t>>("zps");

                int mask = zps.size() == 1 ? 0 : 1 << axis;

                std::vector<int32_t> neg_int32_zps = dnnl_impl::utils::fmap(zps,
                        [](int64_t zp) { return static_cast<int32_t>(-zp); });

                prm_attr.set_zero_points(DNNL_ARG_SRC, mask, neg_int32_zps);
            }

            fuse_op_to_successor(zp_op, subgraph);
        } else {
            auto in_val = zp_op->get_input_values()[0];
            auto &prv_op = in_val->get_producer();
            if (!has_int8_support(prv_op.get_kind())) continue;

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

            int mask = zps.size() == 1 ? 0 : 1 << axis;
            std::vector<int32_t> int32_zps;
            for (auto &zp : zps) {
                int32_zps.emplace_back(static_cast<int32_t>(zp));
            }
            prm_attr.set_zero_points(DNNL_ARG_DST, mask, int32_zps);

            fuse_op_to_predecessor(zp_op, subgraph);
        }
    }
}

void fuse_mul_scales_add_zps(std::vector<op_ptr> &subgraph) {
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

    if (fuse_groups.empty()) return;

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

        op_ptr fused_op = std::make_shared<op_t>(op_kind::Reorder);
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
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

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
using ltw = impl::logical_tensor_wrapper;

static bool has_optional_bias(op_kind_t kind) {
    std::set<op_kind_t> ops {impl::op_kind::Convolution, impl::op_kind::MatMul};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool has_int8_support(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::dnnl_convolution, impl::op_kind::MatMul};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool has_post_ops(op_kind_t kind) {
    std::set<op_kind_t> ops {impl::op_kind::Convolution,
            op_kind::dnnl_convolution, impl::op_kind::MatMul};
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
}

void split_quant_dequant(std::vector<op_ptr> &subgraph) {
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
        if (op->get_kind() == impl::op_kind::Convolution) {
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
}

void fuse_to_int8_matmul(std::vector<op_ptr> &subgraph) {
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
}

void fuse_to_int8_pool(std::vector<op_ptr> &subgraph) {
    std::vector<op_t *> fusion_ops;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::MaxPool
                && cur_op->get_kind() != impl::op_kind::AvgPool)
            continue;
        fusion_ops.emplace_back(cur_op.get());
    }

    if (fusion_ops.empty()) return;
    for (auto &pool_op : fusion_ops) {
        op_ptr q_pool_op = std::make_shared<op_t>(op_kind::dnnl_pool);
        q_pool_op->merge_attributes(pool_op->get_attributes());
        if (pool_op->get_kind() == impl::op_kind::MaxPool) {
            q_pool_op->set_attr<std::string>("kind", "maxpool");
        } else {
            q_pool_op->set_attr<std::string>("kind", "avgpool");
        }

        // oneDNN int8 pooling primitive doesn't require scales and zps, so we
        // just fuse the Quantize and Dequantize op into this newly created pool
        // op
        op_t &dequant_op = pool_op->get_input_value(0)->get_producer();
        assertm(dequant_op.get_kind() == impl::op_kind::Dequantize,
                "the predecessor op of pool should be a Dequant op.");
        value_ptr in_value = dequant_op.get_input_value(0);
        q_pool_op->connect_input(0, in_value);
        in_value->remove_consumer(dequant_op, 0);

        assertm(pool_op->get_output_value(0)->get_consumers().size() == 1,
                "pooling's successor Quant op should only have one consumer.");
        const op_t &quant_op
                = pool_op->get_output_value(0)->get_consumers()[0].get_op();
        assertm(quant_op.get_kind() == impl::op_kind::Quantize,
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
    const std::set<op_kind_t> post_ops_kinds {impl::op_kind::ReLU,
            impl::op_kind::GELU, impl::op_kind::Sigmoid, impl::op_kind::Elu,
            impl::op_kind::HardTanh, impl::op_kind::Abs, impl::op_kind::Sqrt,
            impl::op_kind::Square, impl::op_kind::Tanh, impl::op_kind::Add,
            impl::op_kind::Divide, op_kind::dnnl_swish};

    const std::set<op_kind_t> eltwise_kinds {impl::op_kind::ReLU,
            impl::op_kind::GELU, impl::op_kind::Sigmoid, impl::op_kind::Elu,
            impl::op_kind::HardTanh, impl::op_kind::Abs, impl::op_kind::Sqrt,
            impl::op_kind::Square, impl::op_kind::Tanh, op_kind::dnnl_swish};

    const std::map<op_kind_t, dnnl::algorithm> eltwise_alg_map {
            {impl::op_kind::ReLU, dnnl::algorithm::eltwise_relu},
            {impl::op_kind::GELU, dnnl::algorithm::eltwise_gelu_erf},
            {impl::op_kind::Sigmoid, dnnl::algorithm::eltwise_logistic},
            {impl::op_kind::Elu, dnnl::algorithm::eltwise_elu},
            {impl::op_kind::HardTanh, dnnl::algorithm::eltwise_clip},
            {impl::op_kind::Abs, dnnl::algorithm::eltwise_abs},
            {impl::op_kind::Sqrt, dnnl::algorithm::eltwise_sqrt},
            {impl::op_kind::Square, dnnl::algorithm::eltwise_square},
            {impl::op_kind::Tanh, dnnl::algorithm::eltwise_tanh},
            {op_kind::dnnl_swish, dnnl::algorithm::eltwise_swish}};

    // lambda function to fuse one post op into base primitive
    auto fuse_post_ops_func = [&](std::vector<op_ptr> &subgraph,
                                      primitive_attr_mgr &prm_attr_mgr,
                                      bool &changed) -> impl::status_t {
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

        if (fuse_groups.empty()) {
            changed = false;
            return impl::status::success;
        }

        for (auto &fuse_group : fuse_groups) {
            auto base_op = fuse_group.first;
            auto post_op = fuse_group.second;
            // post op fuse to which predecessor
            size_t fuse_op_predecessor_offset = 0;

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
            } else if (post_op->get_kind() == impl::op_kind::Add) {
                // get mul_scale op from Add's input
                // as Add is commutative, mul_scale op can be
                // 0 / 1 input of Add
                size_t mul_scale_op_offset = 2;
                for (size_t i = 0; i < 2; ++i) {
                    auto in_val = post_op->get_input_value(i);
                    if (in_val->has_producer()
                            && in_val->get_producer().get_kind()
                                    == op_kind::mul_scales) {
                        mul_scale_op_offset = i;
                        fuse_op_predecessor_offset = 1 - i;
                        break;
                    }
                }
                if (mul_scale_op_offset != 2) {
                    // for int8 cases
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
                    // for fp32 cases:
                    // - the add op's src1 have no producer
                    // - the add operation may need broadcast
                    auto fused_in = post_op->get_input_value(
                            fuse_op_predecessor_offset);
                    auto other_in = post_op->get_input_value(
                            1 - fuse_op_predecessor_offset);
                    auto dst = post_op->get_output_value(0);

                    if (ltw(fused_in->get_logical_tensor()).vdims()
                            == ltw(other_in->get_logical_tensor()).vdims()) {
                        // use sum post-ops for no-broadcast add
                        pops.append_sum(1.f);
                        base_op->set_attr<bool>("with_sum", true);
                    } else {
                        // use binary post-ops for broadcast add
                        auto dst_lt = dst->get_logical_tensor();
                        const logical_tensor_wrapper dst_ltw(dst_lt);
                        int dst_ndims = dst_ltw.ndims();
                        memory::desc post_src = make_dnnl_memory_desc(
                                other_in->get_logical_tensor());
                        post_src = expand(post_src, dst_ndims);

                        // insert a expand op to preprocess the post src
                        op_ptr expand_op
                                = std::make_shared<op_t>(op_kind::expand);
                        insert_op_before(expand_op.get(), post_op,
                                1 - fuse_op_predecessor_offset);
                        subgraph.emplace_back(expand_op);

                        // post binary only supports per tensor and per channel
                        // broadcast, which means the expand shape of post src
                        // should be all one or the
                        // post_src_dim[c_axis]==dst_dim[c_axis]
                        std::string data_fmt = base_op->has_attr("data_format")
                                ? base_op->get_attr<std::string>("data_format")
                                : "NCX";
                        int c_axis = (data_fmt == "NXC") ? (dst_ndims - 1) : 1;
                        for (int i = dst_ndims - 1; i >= 0; i--) {
                            if (post_src.dims()[i] == 1) continue;

                            if (i != c_axis
                                    || dst_ltw.dims()[i]
                                            != post_src.dims()[i]) {
                                return impl::status::compile_fail;
                            }
                        }
                        pops.append_binary(algorithm::binary_add, post_src);
                        base_op->set_attr<bool>("with_binary", true);
                    }
                }
            } else if (post_op->get_kind() == impl::op_kind::Divide) {
                const auto &post_src = make_dnnl_memory_desc(
                        post_op->get_input_value(1)->get_logical_tensor());
                pops.append_binary(dnnl::algorithm::binary_div, post_src);
            } else {
                // unsupported post ops
                continue;
            }

            prm_attr.set_post_ops(pops);

            // remove the fused post_ops op
            fuse_op_to_predecessor(
                    post_op, subgraph, fuse_op_predecessor_offset);
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
}

void insert_bn_folding(std::vector<op_ptr> &subgraph) {
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
}

void conv_bwd_data_canonicalization(std::vector<op_ptr> &subgraph) {
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
        subgraph.emplace_back(std::move(op));
    }
    for (const auto &op : to_be_removed_ops) {
        auto pos = std::find_if(subgraph.begin(), subgraph.end(),
                [op](const op_ptr &tmp) { return op.get() == tmp.get(); });
        if (pos != subgraph.end()) subgraph.erase(pos);
    }
}

void fuse_mul_sigmoid_to_swish(std::vector<op_ptr> &subgraph) {
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

    if (swish_patterns.empty()) return;

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
}

void fuse_typecast_to_matmul(std::vector<op_ptr> &subgraph) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : subgraph) {
        if (cur_op->get_kind() != impl::op_kind::MatMul) continue;
        auto &in0 = cur_op->get_input_value(0)->get_producer();
        auto &in1 = cur_op->get_input_value(1)->get_producer();
        if (in0.get_kind() == impl::op_kind::TypeCast
                && in1.get_kind() == impl::op_kind::TypeCast
                && cur_op->get_input_value(0)->get_logical_tensor().data_type
                        == impl::data_type::bf16
                && cur_op->get_input_value(1)->get_logical_tensor().data_type
                        == impl::data_type::bf16
                && in0.get_input_value(0)->has_producer()
                && in1.get_input_value(0)->has_producer()
                && in0.get_input_value(0)->get_producer().get_kind()
                        == impl::op_kind::Dequantize
                && in1.get_input_value(0)->get_producer().get_kind()
                        == impl::op_kind::Dequantize)
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in0, &in1});
    }

    if (fusion_groups.empty()) return;

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
        out_val->set_data_type(impl::data_type::bf16);

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
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/graph.hpp"
#include "graph/interface/op_schema.hpp"
#include "graph/interface/shape_infer.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/fusion_info.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/op_executable.hpp"

#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_t = op_t;
using op_ptr = std::shared_ptr<op_t>;
using value_ptr = std::shared_ptr<value_t>;
using ltw = logical_tensor_wrapper_t;

static status_t pool_fwd_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_pool);
    if (op->get_kind() == graph::op_kind::MaxPool) {
        new_op->set_attr<std::string>(op_attr::kind, "maxpool");
    } else {
        new_op->set_attr<std::string>(op_attr::kind, "avgpool");
    }
    new_op->merge_attributes(op->get_attributes());
    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t avgpool_bwd_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_pool_bwd);
    new_op->set_attr<std::string>(op_attr::kind, "avgpool");
    new_op->merge_attributes(op->get_attributes());
    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t binary_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_binary);
    new_op->set_attr<int64_t>(op_attr::alg_kind,
            static_cast<int64_t>(get_binary_alg_map().at(op->get_kind())));
    new_op->merge_attributes(op->get_attributes());
    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t bias_add_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_binary);
    new_op->set_attr<int64_t>(op_attr::alg_kind,
            static_cast<int64_t>(dnnl::algorithm::binary_add));
    new_op->set_attr<bool>(op_attr::is_bias_add, true);
    new_op->merge_attributes(op->get_attributes());
    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t eltwise_fwd_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
    new_op->set_attr<int64_t>(op_attr::alg_kind,
            static_cast<int64_t>(get_eltwise_alg_map().at(op->get_kind())));
    merge_common_eltwise_attrs(op, new_op);
    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t eltwise_bwd_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise_bwd);
    merge_common_eltwise_attrs(op, new_op);
    const bool use_dst = op->has_attr(op_attr::use_dst)
            ? op->get_attr<bool>(op_attr::use_dst)
            : false;
    new_op->set_attr(op_attr::use_dst, use_dst);

    auto kind = op->get_kind();
    auto bwd_algo = get_eltwise_bwd_alg(kind, use_dst);
    auto fwd_algo = get_eltwise_bwd_alg(kind, false);
    if (bwd_algo == algorithm::undef) {
        DEBUG_PRINT_ERROR("Unsupported eltwise bwd op.");
        return status::unimplemented;
    }
    new_op->set_attr<int64_t>(
            op_attr::alg_kind, static_cast<int64_t>(bwd_algo));
    new_op->set_attr<int64_t>(
            op_attr::fwd_alg_kind, static_cast<int64_t>(fwd_algo));

    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t softplus_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    op_ptr new_op;
    const auto beta = op->get_attr<float>(op_attr::beta);
    const auto algo = dnnl::algorithm::eltwise_soft_relu;
    if (op->get_kind() == graph::op_kind::SoftPlus) {
        new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
    } else { // SoftPlusBackward
        new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise_bwd);
        new_op->set_attr(op_attr::fwd_alg_kind, static_cast<int64_t>(algo));
        new_op->set_attr(op_attr::use_dst, false);
    }
    new_op->set_attr<int64_t>(op_attr::alg_kind, static_cast<int64_t>(algo));
    new_op->set_attr<float>(op_attr::alpha, beta);

    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t batchnorm_fwd_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_batchnorm);

    // decide if this is for training or inference
    if (op->get_kind() == graph::op_kind::BatchNormInference)
        new_op->set_attr<bool>(op_attr::is_training, false);
    else
        new_op->set_attr<bool>(op_attr::is_training, true);
    new_op->merge_attributes(op->get_attributes());

    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t reduction_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_reduction);
    new_op->set_attr<int64_t>(
            op_attr::alg_kind, static_cast<int64_t>(op->get_kind()));
    new_op->set_attr<int64_t>(op_attr::alg_kind,
            static_cast<int64_t>(get_reduction_alg_map().at(op->get_kind())));
    if (op->get_kind() == graph::op_kind::ReduceL1)
        new_op->set_attr<float>(op_attr::p, 1.0f);
    else if (op->get_kind() == graph::op_kind::ReduceL2)
        new_op->set_attr<float>(op_attr::p, 2.0f);
    new_op->merge_attributes(op->get_attributes());

    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t reorder_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
    new_op->set_attr<bool>(op_attr::change_layout, true);
    new_op->merge_attributes(op->get_attributes());

    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t typecast_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
    new_op->set_attr<bool>(op_attr::change_layout, false);
    new_op->merge_attributes(op->get_attributes());

    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t reciprocal_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
    new_op->set_attr<int64_t>(op_attr::alg_kind,
            static_cast<int64_t>(dnnl::algorithm::eltwise_pow));
    new_op->set_attr<float>(op_attr::alpha, 1.f);
    new_op->set_attr<float>(op_attr::beta, -1.f);

    rewriter.replace_op(op, new_op);
    insert_empty_scratchpad(new_op);
    return status::success;
}

static status_t static_reshape_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_reshape);
    new_op->merge_attributes(op->get_attributes());
    rewriter.replace_op(op, new_op);
    return status::success;
}

static status_t static_transpose_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    auto new_op = std::make_shared<op_t>(op_kind::dnnl_transpose);
    new_op->merge_attributes(op->get_attributes());
    rewriter.replace_op(op, new_op);
    return status::success;
}

static status_t dummy_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    UNUSED(op);
    UNUSED(rewriter);
    return status::success;
}

static status_t maxpool_bwd_handler(
        const std::shared_ptr<op_t> &cur_op, subgraph_rewriter_t &rewriter) {
    // For MaxPoolBackward op, we get diff_src (the output) shape from it's
    // src input. While dnnl_pool_bwd op didn't define src input (because
    // it's not used in primitive computation, and AvgPoolBackward also
    // didn't have src input), we must transfer the shape info from the src
    // input to dnnl_pool_bwd op's op_attr::input_shape attribute. So, we
    // need to check that the src input must have shape info.
    auto src_lt = cur_op->get_input_value(0)->get_logical_tensor();
    logical_tensor_wrapper_t src_ltw(src_lt);
    if (src_ltw.is_shape_unknown()) {
        DEBUG_PRINT_ERROR(
                "MaxPoolBackward op's src input must have valid shape");
        return status::invalid_shape;
    }

    op_ptr maxpool_bwd = std::make_shared<op_t>(op_kind::dnnl_pool_bwd);
    maxpool_bwd->merge_attributes(cur_op->get_attributes());
    maxpool_bwd->set_attr<std::string>(op_attr::kind, "maxpool");
    maxpool_bwd->set_attr<std::vector<int64_t>>(
            op_attr::src_shape, src_ltw.vdims());

    // connect diff_dst
    auto diff_dst_value = cur_op->get_input_value(1);
    diff_dst_value->remove_consumer(*cur_op, 1);
    diff_dst_value->add_consumer(*maxpool_bwd, 0);
    maxpool_bwd->add_input(diff_dst_value);

    // no indices. we need to insert a maxpool fwd to re-compute the
    // indices from src
    op_ptr maxpool_fwd = std::make_shared<op_t>(op_kind::dnnl_pool);
    maxpool_fwd->merge_attributes(cur_op->get_attributes());
    maxpool_fwd->set_attr<std::string>(op_attr::kind, "maxpool");

    // connect src value to fwd op
    auto src_value = cur_op->get_input_value(0);
    src_value->remove_consumer(*cur_op, 0);
    src_value->add_consumer(*maxpool_fwd, 0);
    maxpool_fwd->add_input(src_value);

    // create dst value for fwd op
    // this might be an extra end edge since no consumers
    logical_tensor_t maxpool_fwd_dst = empty_logical_tensor_with_default_id();
    maxpool_fwd_dst.data_type = src_value->get_logical_tensor().data_type;
    value_ptr maxpool_fwd_dst_value
            = std::make_shared<value_t>(*maxpool_fwd, 0, maxpool_fwd_dst);
    maxpool_fwd->add_output(maxpool_fwd_dst_value);

    // create scratchpad value for fwd op
    insert_empty_scratchpad(maxpool_fwd);

    // create ws value for fwd op
    logical_tensor_t maxpool_fwd_ws = empty_logical_tensor_with_default_id();
    value_ptr maxpool_fwd_ws_value
            = std::make_shared<value_t>(*maxpool_fwd, 2, maxpool_fwd_ws);
    maxpool_fwd->add_output(maxpool_fwd_ws_value);

    // connect forward op's ws value to bwd op
    maxpool_fwd_ws_value->add_consumer(*maxpool_bwd, 1);
    maxpool_bwd->add_input(maxpool_fwd_ws_value);

    rewriter.to_insert(maxpool_fwd);

    // connect the forward src as the dnnl_pool_bwd op's 3rd input (used
    // to store the logical tensor which will be converted to a md to
    // create the forward hint)
    src_value->add_consumer(*maxpool_bwd, 2);
    maxpool_bwd->add_input(src_value);

    // connect diff_src
    auto diff_src_value = cur_op->get_output_value(0);
    maxpool_bwd->add_output(diff_src_value);

    // connect scratchpad
    insert_empty_scratchpad(maxpool_bwd);

    rewriter.to_insert(maxpool_bwd);
    rewriter.to_remove(cur_op);

    return status::success;
}

static status_t squared_difference_handler(
        const std::shared_ptr<op_t> &cur_op, subgraph_rewriter_t &rewriter) {
    if (cur_op->get_kind() == graph::op_kind::SquaredDifference) {
        op_ptr subtract = std::make_shared<op_t>(op_kind::dnnl_binary);
        subtract->set_attr<int64_t>(op_attr::alg_kind,
                static_cast<int64_t>(dnnl::algorithm::binary_sub));

        rewriter.replace_op(cur_op, subtract);

        op_ptr square = std::make_shared<op_t>(op_kind::dnnl_eltwise);
        square->set_attr<int64_t>(op_attr::alg_kind,
                static_cast<int64_t>(dnnl::algorithm::eltwise_square));

        const float default_attr_value = 0.0f;
        square->set_attr<float>(op_attr::alpha, default_attr_value);
        square->set_attr<float>(op_attr::beta, default_attr_value);

        rewriter.insert_op_after(square, subtract, 0);
        insert_empty_scratchpad(subtract);
        insert_empty_scratchpad(square);
    }

    return status::success;
}

static status_t static_quant_handler(
        const std::shared_ptr<op_t> &op, subgraph_rewriter_t &rewriter) {
    const auto &zps = op->get_attr<std::vector<int64_t>>(op_attr::zps);
    const auto &scales = op->get_attr<std::vector<float>>(op_attr::scales);
    const auto &qtype = op->get_attr<std::string>(op_attr::qtype);
    const auto &axis = op->get_attr<int64_t>(op_attr::axis);

    auto in_vals = op->get_input_values();
    auto out_vals = op->get_output_values();
    assertm(in_vals.size() == 1 && out_vals.size() == 1,
            "static quantize/dequantize should only have one input and "
            "output");

    // int8 = f32 / scales + zps
    op_ptr mul_scales_op = std::make_shared<op_t>(op_kind::dnnl_mul_scales);
    op_ptr add_zps_op = std::make_shared<op_t>(op_kind::dnnl_add_zps);

    assertm(std::all_of(scales.begin(), scales.end(),
                    [](float i) { return i != 0.f; }),
            "scales can't be zero");

    std::vector<float> inv_scales
            = dnnl_impl::utils::fmap(scales, [](float s) { return 1.f / s; });
    mul_scales_op->set_attr<std::vector<float>>(op_attr::scales, inv_scales);
    add_zps_op->set_attr<std::vector<int64_t>>(op_attr::zps, zps);

    mul_scales_op->set_attr<int64_t>(op_attr::axis, axis);
    mul_scales_op->set_attr<std::string>(op_attr::qtype, qtype);
    add_zps_op->set_attr<int64_t>(op_attr::axis, axis);
    add_zps_op->set_attr<std::string>(op_attr::qtype, qtype);

    // reconnect
    in_vals[0]->remove_consumer(*op, 0);
    in_vals[0]->add_consumer(*mul_scales_op, 0);
    mul_scales_op->add_input(in_vals[0]);

    logical_tensor_t new_lt = empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*mul_scales_op, 0, new_lt, true);
    new_val->set_data_type(in_vals[0]->get_logical_tensor().data_type);

    mul_scales_op->add_output(new_val);

    add_zps_op->add_input(new_val);
    new_val->add_consumer(*add_zps_op, 0);
    add_zps_op->add_output(out_vals[0]);

    // add new ops and delete quantize op
    rewriter.to_insert(mul_scales_op);
    rewriter.to_insert(add_zps_op);
    rewriter.to_remove(op);

    return status::success;
}

static status_t static_dequant_handler(
        const std::shared_ptr<op_t> &cur_op, subgraph_rewriter_t &rewriter) {
    const auto &zps = cur_op->get_attr<std::vector<int64_t>>(op_attr::zps);
    const auto &scales = cur_op->get_attr<std::vector<float>>(op_attr::scales);
    const auto &qtype = cur_op->get_attr<std::string>(op_attr::qtype);
    const auto &axis = cur_op->get_attr<int64_t>(op_attr::axis);

    auto in_vals = cur_op->get_input_values();
    auto out_vals = cur_op->get_output_values();
    assertm(in_vals.size() == 1 && out_vals.size() == 1,
            "static dequantize should only have one input and output");

    // f32 = scales * (int8 - zps)
    op_ptr sub_zps_op = std::make_shared<op_t>(op_kind::dnnl_sub_zps);
    op_ptr mul_scales_op = std::make_shared<op_t>(op_kind::dnnl_mul_scales);

    sub_zps_op->set_attr<std::vector<int64_t>>(op_attr::zps, zps);
    mul_scales_op->set_attr<std::vector<float>>(op_attr::scales, scales);

    sub_zps_op->set_attr<int64_t>(op_attr::axis, axis);
    sub_zps_op->set_attr<std::string>(op_attr::qtype, qtype);
    mul_scales_op->set_attr<int64_t>(op_attr::axis, axis);
    mul_scales_op->set_attr<std::string>(op_attr::qtype, qtype);

    // reconnect
    in_vals[0]->remove_consumer(*cur_op, 0);
    in_vals[0]->add_consumer(*sub_zps_op, 0);
    sub_zps_op->add_input(in_vals[0]);

    logical_tensor_t new_lt = empty_logical_tensor_with_default_id();
    auto new_val = std::make_shared<value_t>(*sub_zps_op, 0, new_lt, true);
    new_val->set_data_type(in_vals[0]->get_logical_tensor().data_type);

    sub_zps_op->add_output(new_val);

    mul_scales_op->add_input(new_val);
    new_val->add_consumer(*mul_scales_op, 0);
    mul_scales_op->add_output(out_vals[0]);

    // add new ops and delete dequantize op
    rewriter.to_insert(sub_zps_op);
    rewriter.to_insert(mul_scales_op);
    rewriter.to_remove(cur_op);

    return status::success;
}

static status_t dynamic_quant_handler(
        const std::shared_ptr<op_t> &cur_op, subgraph_rewriter_t &rewriter) {
    const auto &qtype = cur_op->get_attr<std::string>(op_attr::qtype);
    const auto &axis = cur_op->get_attr<int64_t>(op_attr::axis);

    auto &in_vals = cur_op->get_input_values();
    auto &out_vals = cur_op->get_output_values();
    assertm((in_vals.size() == 3 || in_vals.size() == 2)
                    && out_vals.size() == 1,
            "dynamic quantize must have 2 or 3 inputs and 1 output");

    // DynamicQuantize has optional zps
    bool has_zps = in_vals.size() == 3;

    value_ptr src = in_vals[0], scales = in_vals[1], dst = out_vals[0], zps;
    if (has_zps) zps = in_vals[2];

    // int8 = f32 / scales + zps
    op_ptr mul_scales = std::make_shared<op_t>(op_kind::dnnl_mul_scales);

    mul_scales->connect_input(1, scales);
    scales->remove_consumer(*cur_op, 1);
    mul_scales->set_attr<int64_t>(op_attr::axis, axis);
    mul_scales->set_attr<std::string>(op_attr::qtype, qtype);
    mul_scales->set_attr<bool>(op_attr::with_runtime_scales, true);

    // connect mul_scales to subgraph
    mul_scales->connect_input(0, src);
    src->remove_consumer(*cur_op, 0);
    mul_scales->add_output(dst);
    rewriter.to_insert(mul_scales);

    // op used to inverse the scales
    auto inv_scales_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
    // y = alpha*x^beta
    inv_scales_op->set_attr<int64_t>(op_attr::alg_kind,
            static_cast<int64_t>(dnnl::algorithm::eltwise_pow));
    inv_scales_op->set_attr<float>(op_attr::alpha, 1.0f);
    inv_scales_op->set_attr<float>(op_attr::beta, -1.0f);
    rewriter.insert_op_before(inv_scales_op, mul_scales, 1);
    insert_empty_scratchpad(inv_scales_op);

    if (has_zps) {
        op_ptr add_zps = std::make_shared<op_t>(op_kind::dnnl_add_zps);
        add_zps->connect_input(1, zps);
        zps->remove_consumer(*cur_op, 2);
        add_zps->set_attr<int64_t>(op_attr::axis, axis);
        add_zps->set_attr<std::string>(op_attr::qtype, qtype);
        add_zps->set_attr<bool>(op_attr::with_runtime_zps, true);

        // connect add_zps to subgraph
        rewriter.insert_op_after(add_zps, mul_scales, 0, 0);
    }

    rewriter.to_remove(cur_op);

    return status::success;
}

static status_t dynamic_dequant_handler(
        const std::shared_ptr<op_t> &cur_op, subgraph_rewriter_t &rewriter) {
    const auto &qtype = cur_op->get_attr<std::string>(op_attr::qtype);
    const auto &axis = cur_op->get_attr<int64_t>(op_attr::axis);

    auto &in_vals = cur_op->get_input_values();
    auto &out_vals = cur_op->get_output_values();
    assertm((in_vals.size() == 3 || in_vals.size() == 2)
                    && out_vals.size() == 1,
            "dynamic dequantize must have 2 or 3 inputs and 1 output");

    // DynamicDequantize has optional zps
    bool has_zps = in_vals.size() == 3;

    value_ptr src = in_vals[0], scales = in_vals[1], dst = out_vals[0], zps;
    if (has_zps) zps = in_vals[2];

    // f32 = scales * (int8 - zps)
    // connect scales to mul_scales op
    op_ptr mul_scales = std::make_shared<op_t>(op_kind::dnnl_mul_scales);
    mul_scales->connect_input(1, scales);
    scales->remove_consumer(*cur_op, 1);
    mul_scales->set_attr<int64_t>(op_attr::axis, axis);
    mul_scales->set_attr<std::string>(op_attr::qtype, qtype);
    mul_scales->set_attr<bool>(op_attr::with_runtime_scales, true);

    // connect mul_scales op to subgraph
    mul_scales->connect_input(0, src);
    src->remove_consumer(*cur_op, 0);
    mul_scales->add_output(dst);
    rewriter.to_insert(mul_scales);

    if (has_zps) {
        op_ptr sub_zps = std::make_shared<op_t>(op_kind::dnnl_sub_zps);
        sub_zps->connect_input(1, zps);
        zps->remove_consumer(*cur_op, 2);
        sub_zps->set_attr<int64_t>(op_attr::axis, axis);
        sub_zps->set_attr<std::string>(op_attr::qtype, qtype);
        sub_zps->set_attr<bool>(op_attr::with_runtime_zps, true);

        // connect sub_zps op to subgraph
        rewriter.insert_op_before(sub_zps, mul_scales, 0, 0);
    }

    rewriter.to_remove(cur_op);

    return status::success;
}

#define ITEM(kind, func) \
    { \
        graph::op_kind::kind, handler_func { (func) } \
    }

static const std::unordered_map<graph::op_kind_t, handler_func> handler_table {
        // matmul
        ITEM(MatMul, common_handler<op_kind::kDnnl_matmul>),
        // conv
        ITEM(Convolution, common_handler<op_kind::kDnnl_convolution>),
        ITEM(ConvolutionBackwardData,
                common_handler<op_kind::kDnnl_conv_bwd_data>),
        ITEM(ConvolutionBackwardWeights,
                common_handler<op_kind::kDnnl_conv_bwd_weights>),
        // convtranspose
        ITEM(ConvTranspose, common_handler<op_kind::kDnnl_convtranspose>),
        ITEM(ConvTransposeBackwardData,
                common_handler<op_kind::kDnnl_convtranspose_bwd_data>),
        ITEM(ConvTransposeBackwardWeights,
                common_handler<op_kind::kDnnl_convtranspose_bwd_weights>),
        // pooling
        ITEM(MaxPool, pool_fwd_handler),
        ITEM(AvgPool, pool_fwd_handler),
        ITEM(AvgPoolBackward, avgpool_bwd_handler),
        ITEM(MaxPoolBackward, maxpool_bwd_handler),
        // softmax
        ITEM(SoftMax, common_handler<op_kind::kDnnl_softmax>),
        ITEM(LogSoftmax, common_handler<op_kind::kDnnl_logsoftmax>),
        ITEM(SoftMaxBackward, common_handler<op_kind::kDnnl_softmax_bwd>),
        ITEM(LogSoftmaxBackward, common_handler<op_kind::kDnnl_logsoftmax_bwd>),
        // binary
        ITEM(Add, binary_handler),
        ITEM(Subtract, binary_handler),
        ITEM(Multiply, binary_handler),
        ITEM(Divide, binary_handler),
        ITEM(Minimum, binary_handler),
        ITEM(Maximum, binary_handler),
        // eltwise fwd
        ITEM(Abs, eltwise_fwd_handler),
        ITEM(Clamp, eltwise_fwd_handler),
        ITEM(Elu, eltwise_fwd_handler),
        ITEM(Exp, eltwise_fwd_handler),
        ITEM(GELU, eltwise_fwd_handler),
        ITEM(HardSigmoid, eltwise_fwd_handler),
        ITEM(HardSwish, eltwise_fwd_handler),
        ITEM(LeakyReLU, eltwise_fwd_handler),
        ITEM(Log, eltwise_fwd_handler),
        ITEM(Mish, eltwise_fwd_handler),
        ITEM(ReLU, eltwise_fwd_handler),
        ITEM(Round, eltwise_fwd_handler),
        ITEM(Sigmoid, eltwise_fwd_handler),
        ITEM(Sqrt, eltwise_fwd_handler),
        ITEM(Square, eltwise_fwd_handler),
        ITEM(Tanh, eltwise_fwd_handler),
        // eltwise bwd
        ITEM(AbsBackward, eltwise_bwd_handler),
        ITEM(ClampBackward, eltwise_bwd_handler),
        ITEM(EluBackward, eltwise_bwd_handler),
        ITEM(GELUBackward, eltwise_bwd_handler),
        ITEM(HardSigmoidBackward, eltwise_bwd_handler),
        ITEM(HardSwishBackward, eltwise_bwd_handler),
        ITEM(MishBackward, eltwise_bwd_handler),
        ITEM(ReLUBackward, eltwise_bwd_handler),
        ITEM(SigmoidBackward, eltwise_bwd_handler),
        ITEM(SqrtBackward, eltwise_bwd_handler),
        ITEM(TanhBackward, eltwise_bwd_handler),
        // batchnorm
        ITEM(BatchNormInference, batchnorm_fwd_handler),
        ITEM(BatchNormForwardTraining, batchnorm_fwd_handler),
        ITEM(BatchNormTrainingBackward,
                common_handler<op_kind::kDnnl_batchnorm_bwd>),
        // prelu
        ITEM(PReLU, common_handler<op_kind::kDnnl_prelu>),
        ITEM(PReLUBackward, common_handler<op_kind::kDnnl_prelu_bwd>),
        // reduction
        ITEM(ReduceL1, reduction_handler),
        ITEM(ReduceL2, reduction_handler),
        ITEM(ReduceMax, reduction_handler),
        ITEM(ReduceMean, reduction_handler),
        ITEM(ReduceMin, reduction_handler),
        ITEM(ReduceProd, reduction_handler),
        ITEM(ReduceSum, reduction_handler),
        // softplus
        ITEM(SoftPlus, softplus_handler),
        ITEM(SoftPlusBackward, softplus_handler),
        // interpolate
        ITEM(Interpolate, common_handler<op_kind::kDnnl_resampling>),
        ITEM(InterpolateBackward,
                common_handler<op_kind::kDnnl_resampling_bwd>),
        // layernorm
        ITEM(LayerNorm, common_handler<op_kind::kDnnl_layernorm>),
        ITEM(LayerNormBackward, common_handler<op_kind::kDnnl_layernorm_bwd>),
        // quantization
        ITEM(Quantize, static_quant_handler),
        ITEM(Dequantize, static_dequant_handler),
        ITEM(DynamicQuantize, dynamic_quant_handler),
        ITEM(DynamicDequantize, dynamic_dequant_handler),
        // data formatting
        ITEM(StaticReshape, static_reshape_handler),
        ITEM(StaticTranspose, static_transpose_handler),
        // misc
        ITEM(BiasAdd, bias_add_handler),
        ITEM(Reorder, reorder_handler),
        ITEM(TypeCast, typecast_handler),
        ITEM(Reciprocal, reciprocal_handler),
        ITEM(Concat, common_handler<op_kind::kDnnl_concat>),
        ITEM(SquaredDifference, squared_difference_handler),
        // utility
        ITEM(Wildcard, dummy_handler),
        ITEM(End, dummy_handler),
};

#undef ITEM

status_t lower_down(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        auto kind = cur_op->get_kind();
        if (!handler_table.count(kind)) {
            assertm(false,
                    "All spec ops should be lowered to internal ops, except "
                    "for some utility ops like End, Wildcard");
            return status::invalid_graph_op;
        }

        // lower this spec op to dnnl backend internal op
        const auto &handler = handler_table.at(kind);
        auto status = handler(cur_op, rewriter);
        if (status != status::success) return status;
    }

    rewriter.run();
    return infer_shape(sg);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

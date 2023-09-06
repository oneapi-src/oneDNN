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
#include <tuple>
#include <utility>
#include <vector>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/graph.hpp"
#include "graph/interface/op_schema.hpp"
#include "graph/interface/shape_infer.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/fusion_info.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/op_executable.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
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

static bool has_optional_bias(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::dnnl_convolution, op_kind::dnnl_matmul,
            op_kind::dnnl_convtranspose};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool has_int8_support(op_kind_t kind) {
    std::set<op_kind_t> ops {op_kind::dnnl_convolution, op_kind::dnnl_matmul,
            op_kind::dnnl_convtranspose, op_kind::dnnl_reorder};
    return ops.count(kind) != 0;
}

// TODO(xxx): extend to support other ops
static bool is_output_scales_supported(op_kind_t kind) {
    // ops which don't support output scales
    std::set<op_kind_t> ops {op_kind::dnnl_pool, op_kind::dnnl_eltwise};
    return ops.count(kind) == 0;
}

status_t check_with_bias(std::shared_ptr<subgraph_t> &sg) {
    for (auto &cur_op : sg->get_ops()) {
        if (!has_optional_bias(cur_op->get_kind())) continue;
        if (cur_op->num_inputs() == 3) {
            cur_op->set_attr<bool>(op_attr::with_bias, true);
        } else {
            cur_op->set_attr<bool>(op_attr::with_bias, false);
        }
    }
    return status::success;
}

status_t fuse_bias_add(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> bias_add_ops;
    subgraph_rewriter_t rewriter(sg);

    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_binary
                || visited.count(cur_op.get()) != 0)
            continue;

        if (!cur_op->has_attr(op_attr::is_bias_add)
                || !cur_op->get_attr<bool>(op_attr::is_bias_add))
            continue;

        // bias add may be standalone
        if (!cur_op->get_input_value(0)->has_producer()) continue;

        bias_add_ops.emplace_back(cur_op);
        visited.insert(cur_op.get());
    }

    for (auto &bias_add : bias_add_ops) {
        auto in_val = bias_add->get_input_value(0);
        auto &prv_op = in_val->get_producer();
        if (!has_optional_bias(prv_op.get_kind())) continue;
        rewriter.fuse_op_to_predecessor(bias_add);
        prv_op.set_attr<bool>(op_attr::with_bias, true);
    }

    rewriter.run();
    return status::success;
}

// replace mul_scales and add_zps with binary_mul and binary_add respectively
status_t replace_quant_data_with_binary_post_op(
        std::shared_ptr<subgraph_t> &sg) {
    const auto get_next_op = [](const op_t *op) -> op_t * {
        const value_ptr out_val = op->get_output_value(0);
        if (!out_val->get_consumers().empty()) {
            size_t offset = out_val->get_consumers()[0].get_offset();
            auto &next_op = out_val->get_consumers()[0].get_op();
            if (next_op.get_kind() == op_kind::dnnl_add_zps
                    || next_op.get_kind() == op_kind::dnnl_sub_zps
                    || next_op.get_kind() == op_kind::dnnl_mul_scales)
                // cur_op's output must be mul_scales and add_zps ops' first
                // input
                return offset == 0 ? &next_op : nullptr;
            else
                return &out_val->get_consumers()[0].get_op();
        } else
            return nullptr;
    };

    const std::set<op_kind_t> accepted_kinds_in_chain
            = {op_kind::dnnl_binary, op_kind::dnnl_mul_scales,
                    op_kind::dnnl_add_zps, op_kind::dnnl_sub_zps};

    subgraph_rewriter_t rewriter(sg);
    std::set<op_t *> visited;
    for (const auto &cur_op : sg->get_ops()) {
        if ((is_output_scales_supported(cur_op->get_kind())
                    && cur_op->get_kind() != op_kind::dnnl_softmax
                    && cur_op->get_kind() != op_kind::dnnl_layernorm)
                || visited.count(cur_op.get()))
            continue;

        visited.insert(cur_op.get());
        op_t *next_op = get_next_op(cur_op.get());
        while (next_op && accepted_kinds_in_chain.count(next_op->get_kind())) {
            // TODO(xxx): handle the case where other binary input is a
            // 'add_zps'. For patterns like 'int8 pool + binary-mul',
            // 'mul_scales' which feeds 'binary-mul' will get removed.
            // Standalone 'add_zps' is not supported right now, and here we can
            // cover that by treating it as other 'add_zps' OPs in a chain.
            if (next_op->get_kind() == op_kind::dnnl_binary
                    || visited.count(next_op)) {
                next_op = get_next_op(next_op);
                continue;
            }

            // replace quant related op with binary
            op_t *quant_data_op = next_op;
            auto algo = (quant_data_op->get_kind() == op_kind::dnnl_mul_scales)
                    ? dnnl::algorithm::binary_mul
                    : quant_data_op->get_kind() == op_kind::dnnl_add_zps
                    ? dnnl::algorithm::binary_add
                    : dnnl::algorithm::binary_sub;
            op_ptr bin_op = std::make_shared<op_t>(op_kind::dnnl_binary);
            bin_op->set_attr<int64_t>(
                    op_attr::alg_kind, static_cast<int64_t>(algo));
            auto in_val = quant_data_op->get_input_value(0);
            in_val->remove_consumer(*quant_data_op, 0);
            in_val->add_consumer(*bin_op, 0);
            bin_op->add_input(in_val);
            auto out_val = quant_data_op->get_output_value(0);
            bin_op->add_output(out_val);
            in_val->set_data_type(out_val->get_logical_tensor().data_type);
            insert_empty_scratchpad(bin_op);

            // add quant data as a constant input
            const auto qtype
                    = quant_data_op->get_attr<std::string>(op_attr::qtype);
            const std::vector<int64_t> out_shape
                    = ltw(out_val->get_logical_tensor()).vdims();
            std::vector<int64_t> new_shape(out_shape.size(), 1);

            // axis in  [-r, r-1], it may less 0
            const auto axis = (quant_data_op->get_attr<int64_t>(op_attr::axis)
                                      + out_shape.size())
                    % out_shape.size();

            if (qtype != "per_tensor") new_shape[axis] = out_shape[axis];
            op_ptr const_data_op;
            if (quant_data_op->get_kind() == op_kind::dnnl_mul_scales) {
                const auto scales = quant_data_op->get_attr<std::vector<float>>(
                        op_attr::scales);
                const_data_op
                        = std::make_shared<op_t>(op_kind::dnnl_constant_scales);
                const_data_op->set_attr(op_attr::scales, scales);
            } else { // add_zps
                const auto zps = quant_data_op->get_attr<std::vector<int64_t>>(
                        op_attr::zps);
                const_data_op
                        = std::make_shared<op_t>(op_kind::dnnl_constant_zps);
                const_data_op->set_attr(op_attr::zps, zps);
            }
            const_data_op->set_attr(op_attr::shape, new_shape);
            logical_tensor_t const_data_dst_lt
                    = empty_logical_tensor_with_default_id();
            auto const_data_dst_value = std::make_shared<value_t>(
                    *const_data_op, 0, const_data_dst_lt, true);
            auto out_dtype = const_data_op->has_attr(op_attr::zps)
                    ? graph::data_type::s32
                    : graph::data_type::f32;
            const_data_dst_value->set_data_type(out_dtype);
            const_data_dst_value->set_layout_type(layout_type::strided);
            const_data_op->add_output(const_data_dst_value);

            // connect binary and constant data
            bin_op->connect_input(1, const_data_dst_value);

            rewriter.to_insert(bin_op);
            rewriter.to_insert(const_data_op);
            rewriter.to_remove(quant_data_op->shared_from_this());

            visited.insert(next_op);
            next_op = get_next_op(next_op);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t convert_to_runtime_src_scales(std::shared_ptr<subgraph_t> &sg) {
    std::set<op_t *> visited;
    std::vector<op_t *> scales_ops;

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_mul_scales
                || visited.count(cur_op.get()) != 0)
            continue;

        scales_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : scales_ops) {
        assertm(cur_op->num_outputs() == 1,
                "scale_op should have only one output value.");
        auto out_val = cur_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;
        if (!impl::utils::one_of(consumers[0].get_op().get_kind(),
                    op_kind::dnnl_matmul, op_kind::dnnl_convolution,
                    op_kind::dnnl_convtranspose, op_kind::dnnl_reorder))
            continue;

        // make scales as a constant input
        op_ptr const_data_op;
        const auto scales
                = cur_op->get_attr<std::vector<float>>(op_attr::scales);
        const_data_op = std::make_shared<op_t>(op_kind::dnnl_constant_scales);
        const_data_op->set_attr(op_attr::scales, scales);
        std::vector<int64_t> dst_shape(1, scales.size());
        const_data_op->set_attr(op_attr::shape, dst_shape);
        logical_tensor_t const_data_dst_lt
                = empty_logical_tensor_with_default_id();
        auto const_data_dst_value = std::make_shared<value_t>(
                *const_data_op, 0, const_data_dst_lt, true);
        const_data_dst_value->set_data_type(graph::data_type::f32);
        const_data_dst_value->set_layout_type(layout_type::strided);
        const_data_dst_value->set_strides({1});
        const_data_op->add_output(const_data_dst_value);
        cur_op->set_attr(op_attr::with_runtime_scales, true);
        cur_op->remove_attr(op_attr::scales);

        // connect mul_scale and constant data
        cur_op->connect_input(1, const_data_dst_value);
        rewriter.to_insert(const_data_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t convert_to_runtime_src_zero_points(std::shared_ptr<subgraph_t> &sg) {
    std::set<op_t *> visited;
    std::vector<op_t *> zp_ops;

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_sub_zps
                || visited.count(cur_op.get()) != 0)
            continue;

        zp_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &zp_op : zp_ops) {
        assertm(zp_op->num_outputs() == 1,
                "zp_op should have only one output value.");
        auto out_val = zp_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();

        if (!impl::utils::one_of(consumers[0].get_op().get_kind(),
                    op_kind::dnnl_matmul, op_kind::dnnl_convolution,
                    op_kind::dnnl_convtranspose, op_kind::dnnl_reorder))
            continue;

        // make zps as a constant input
        op_ptr const_data_op;
        auto zps = zp_op->get_attr<std::vector<int64_t>>(op_attr::zps);
        // adjusted zp
        std::vector<int64_t> adj_zps = {zps[0]};
        const_data_op = std::make_shared<op_t>(op_kind::dnnl_constant_zps);
        const_data_op->set_attr(op_attr::zps, adj_zps);
        std::vector<int64_t> dst_shape(1, adj_zps.size());
        const_data_op->set_attr(op_attr::shape, dst_shape);
        logical_tensor_t const_data_dst_lt
                = empty_logical_tensor_with_default_id();
        auto const_data_dst_value = std::make_shared<value_t>(
                *const_data_op, 0, const_data_dst_lt, true);
        const_data_dst_value->set_data_type(graph::data_type::s32);
        const_data_dst_value->set_layout_type(layout_type::strided);
        const_data_dst_value->set_strides({1});
        const_data_op->add_output(const_data_dst_value);
        zp_op->set_attr(op_attr::with_runtime_zps, true);
        zp_op->remove_attr(op_attr::zps);

        // connect add_zp and constant data
        zp_op->connect_input(1, const_data_dst_value);
        rewriter.to_insert(const_data_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t convert_to_runtime_dst_zero_points(std::shared_ptr<subgraph_t> &sg) {
    std::set<op_t *> visited;
    std::vector<op_t *> zp_ops;

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_add_zps
                || visited.count(cur_op.get()) != 0)
            continue;

        zp_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &zp_op : zp_ops) {
        assertm(zp_op->num_outputs() == 1,
                "zp_op should have only one output value.");
        auto in_val = zp_op->get_input_values()[0];
        bool is_output_zps = in_val->has_producer()
                && impl::utils::one_of(in_val->get_producer().get_kind(),
                        op_kind::dnnl_matmul, op_kind::dnnl_convolution,
                        op_kind::dnnl_convtranspose, op_kind::dnnl_reorder);

        if (!is_output_zps) continue;

        // make scales as a constant input
        op_ptr const_data_op;
        auto zps = zp_op->get_attr<std::vector<int64_t>>(op_attr::zps);
        // adjusted zp
        std::vector<int64_t> adj_zps = {zps[0]};
        const_data_op = std::make_shared<op_t>(op_kind::dnnl_constant_zps);
        const_data_op->set_attr(op_attr::zps, adj_zps);
        std::vector<int64_t> dst_shape(1, adj_zps.size());
        const_data_op->set_attr(op_attr::shape, dst_shape);
        logical_tensor_t const_data_dst_lt
                = empty_logical_tensor_with_default_id();
        auto const_data_dst_value = std::make_shared<value_t>(
                *const_data_op, 0, const_data_dst_lt, true);
        const_data_dst_value->set_data_type(graph::data_type::s32);
        const_data_dst_value->set_layout_type(layout_type::strided);
        const_data_dst_value->set_strides({1});
        const_data_op->add_output(const_data_dst_value);
        zp_op->set_attr(op_attr::with_runtime_zps, true);
        zp_op->remove_attr(op_attr::zps);

        // connect add_zp and constant data
        zp_op->connect_input(1, const_data_dst_value);
        rewriter.to_insert(const_data_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t fold_mul_scales(std::shared_ptr<subgraph_t> &sg) {
    // lambda function to fold the consecutive mul_scales ops
    auto fold_mul_scales_func = [&]() {
        std::vector<std::pair<op_t *, op_t *>> folding_groups;
        std::set<op_t *> visited;
        for (const auto &cur_op : sg->get_ops()) {
            if (cur_op->get_kind() != op_kind::dnnl_mul_scales
                    || visited.count(cur_op.get()) != 0)
                continue;

            assertm(cur_op->num_outputs() == 1,
                    "cur_op should have only one output value.");
            auto out_val = cur_op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            if (consumers.empty()) continue;

            auto &consumer_op = consumers[0].get_op();
            if (consumer_op.get_kind() != op_kind::dnnl_mul_scales) continue;

            folding_groups.emplace_back(
                    std::pair<op_t *, op_t *> {cur_op.get(), &consumer_op});
            visited.insert(cur_op.get());
            visited.insert(&consumer_op);
        }

        if (folding_groups.empty()) return false;

        subgraph_rewriter_t rewriter(sg);
        for (auto &folding_ops : folding_groups) {
            auto base_op = folding_ops.first;
            auto next_op = folding_ops.second;

            // update the scales
            const auto &scales_base
                    = base_op->get_attr<std::vector<float>>(op_attr::scales);
            const auto &scales_next
                    = next_op->get_attr<std::vector<float>>(op_attr::scales);
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
                base_op->set_attr<int64_t>(op_attr::axis,
                        next_op->get_attr<int64_t>(op_attr::axis));
                base_op->set_attr<std::string>(op_attr::qtype,
                        next_op->get_attr<std::string>(op_attr::qtype));
            }
            base_op->set_attr<std::vector<float>>(op_attr::scales, new_scales);

            rewriter.fuse_op_to_predecessor(next_op->shared_from_this(), 0);
        }
        rewriter.run();
        return true;
    };

    bool changed = true;
    do {
        changed = fold_mul_scales_func();
    } while (changed);
    return status::success;
}

impl::status_t fold_sub_zps_add_zps(std::shared_ptr<subgraph_t> &sg) {
    // lambda function to fold the consecutive zps ops
    auto fold_zps_func = [&]() {
        std::vector<std::pair<op_t *, op_t *>> folding_groups;
        std::set<op_t *> visited;
        for (const auto &cur_op : sg->get_ops()) {
            if (cur_op->get_kind() != op_kind::dnnl_sub_zps
                    || visited.count(cur_op.get()) != 0)
                continue;

            assertm(cur_op->num_outputs() == 1,
                    "cur_op should have only one output value.");
            auto out_val = cur_op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            if (consumers.empty()) continue;

            auto &consumer_op = consumers[0].get_op();
            if (consumer_op.get_kind() != op_kind::dnnl_add_zps) continue;

            folding_groups.emplace_back(
                    std::pair<op_t *, op_t *> {cur_op.get(), &consumer_op});
            visited.insert(cur_op.get());
            visited.insert(&consumer_op);
        }

        if (folding_groups.empty()) return false;

        subgraph_rewriter_t rewriter(sg);
        for (auto &folding_ops : folding_groups) {
            auto previous_op = folding_ops.first;
            auto base_op = folding_ops.second;

            // update the scales
            const auto &zps_previous
                    = previous_op->get_attr<std::vector<int64_t>>(op_attr::zps);
            const auto &zps_base
                    = base_op->get_attr<std::vector<int64_t>>(op_attr::zps);
            std::vector<int64_t> new_zps(
                    std::max(zps_previous.size(), zps_base.size()), 0);
            // per-channel -> per-tensor
            if (zps_base.size() > zps_previous.size()) {
                for (size_t i = 0; i < new_zps.size(); ++i)
                    new_zps[i] = zps_base[i] - zps_previous[0];
            } else {
                for (size_t i = 0; i < new_zps.size(); ++i)
                    new_zps[i] = zps_base[0] - zps_previous[i];
                // set attrs
                base_op->set_attr<int64_t>(op_attr::axis,
                        previous_op->get_attr<int64_t>(op_attr::axis));
                base_op->set_attr<std::string>(op_attr::qtype,
                        previous_op->get_attr<std::string>(op_attr::qtype));
            }
            base_op->set_attr<std::vector<int64_t>>(op_attr::zps, new_zps);

            rewriter.fuse_op_to_predecessor(previous_op->shared_from_this(), 0);
        }
        rewriter.run();
        return true;
    };

    bool changed = true;
    do {
        changed = fold_zps_func();
    } while (changed);
    return impl::status::success;
}

// FIXME(xx) This pass works correctly only when all inputs/outputs scales/zps
// are same, since we are simply ignoring the scales and zps. We can improve
// this pass to support different per-tensor scale since oneDNN concat primitive
// support inputs scaling
status_t fuse_to_int8_concat(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_t *> fusion_ops;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_concat) continue;

        bool matched = true;
        for (auto &in : cur_op->get_input_values()) {
            if (!in->has_producer()
                    || in->get_producer().get_kind()
                            != op_kind::dnnl_mul_scales) {
                matched = false;
                break;
            }

            auto producer_in = in->get_producer().get_input_value(0);
            if (!producer_in->has_producer()
                    || producer_in->get_producer().get_kind()
                            != op_kind::dnnl_sub_zps) {
                matched = false;
                break;
            }
        }

        if (!matched) continue;

        fusion_ops.emplace_back(cur_op.get());
    }

    if (fusion_ops.empty()) return status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &concat_op : fusion_ops) {
        for (size_t i = 0; i < concat_op->num_inputs(); ++i) {
            op_t &scale_op = concat_op->get_input_value(i)->get_producer();
            op_t &zp_op = scale_op.get_input_value(0)->get_producer();
            rewriter.fuse_op_to_successor(zp_op.shared_from_this());
            rewriter.fuse_op_to_successor(scale_op.shared_from_this());
        }

        assertm(concat_op->get_output_value(0)->get_consumers().size() == 1,
                "concat's successor op should only have one consumer.");
        op_t &scale_op
                = concat_op->get_output_value(0)->get_consumers()[0].get_op();
        op_t &zp_op = scale_op.get_output_value(0)->get_consumers()[0].get_op();
        rewriter.fuse_op_to_predecessor(zp_op.shared_from_this());
        rewriter.fuse_op_to_predecessor(scale_op.shared_from_this());
    }

    rewriter.run();
    return status::success;
}

// Moves quant related OPs after pool op.
// This function has effect only if post-ops are present.
//
//       |                   |
//      zps0               zps0
//       |       |           |         |
//    scales0   zps1       pool       zps1
//       |       |           |         |
//     pool   scales1     scales0   scales1
//       \      /              \      /
//        binary      =>        binary
//           |                     |
//        scales2               scales2
//           |                     |
//          zps2                  zps2
//           |                     |
//
status_t fuse_to_int8_pool(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> pool_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_pool
                && cur_op->get_input_value(0)->has_producer()
                && !cur_op->get_output_value(0)->get_consumers().empty()
                && cur_op->get_input_value(0)->get_producer().get_kind()
                        == op_kind::dnnl_mul_scales) {
            pool_ops.emplace_back(cur_op);
        }
    }

    if (pool_ops.empty()) return status::success;

    for (auto &pool_op : pool_ops) {
        value_ptr pool_in_val = pool_op->get_input_value(0);
        value_ptr pool_out_val = pool_op->get_output_value(0);
        op_t &scales_op = pool_in_val->get_producer();

        auto csm = pool_out_val->get_consumers()[0];
        op_t &csm_op = csm.get_op();
        const size_t csm_offset = csm.get_offset();

        value_ptr scales_in_val = scales_op.get_input_value(0);
        if (!scales_in_val->has_producer()) continue;

        // connect pooling with a scales input
        scales_in_val->remove_consumer(scales_op, 0);
        pool_op->connect_input(0, scales_in_val);

        // connect zps with a pooling using a fresh value
        logical_tensor_t pool_to_scales_lt
                = empty_logical_tensor_with_default_id();
        auto pool_to_scales_val = std::make_shared<value_t>(
                *pool_op, 0, pool_to_scales_lt, true);
        pool_to_scales_val->set_data_type(
                scales_in_val->get_logical_tensor().data_type);
        pool_op->connect_output(0, pool_to_scales_val);
        scales_op.connect_input(0, pool_to_scales_val);

        // connect scales with a binary using a fresh value
        logical_tensor_t scales_to_bin_lt
                = empty_logical_tensor_with_default_id();
        auto scales_to_bin_val = std::make_shared<value_t>(
                scales_op, 0, scales_to_bin_lt, true);
        scales_to_bin_val->set_data_type(
                scales_in_val->get_logical_tensor().data_type);
        scales_op.connect_output(0, scales_to_bin_val);
        csm_op.connect_input(csm_offset, scales_to_bin_val);
    }

    return infer_shape(sg);
}

// Moves unfused src zps after pool op, to implement it as a post-binary
//
//       |       |           |         |
//     zps0   zps1         pool       zps1
//       |       |           |         |
//     pool   scales1      zps0     scales1
//       \      /              \      /
//        binary      =>        binary
//           |                     |
//        scales2               scales2
//           |                     |
//          zps2                  zps2
//           |                     |
//
status_t defer_src_zps_for_pool(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> pool_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_pool
                && cur_op->get_input_value(0)->has_producer()
                && !cur_op->get_output_value(0)->get_consumers().empty()
                && cur_op->get_input_value(0)->get_producer().get_kind()
                        == op_kind::dnnl_sub_zps) {
            pool_ops.emplace_back(cur_op);
        }
    }

    if (pool_ops.empty()) return status::success;

    for (auto &pool_op : pool_ops) {
        value_ptr pool_in_val = pool_op->get_input_value(0);
        value_ptr pool_out_val = pool_op->get_output_value(0);

        op_t &sub_zps_op = pool_in_val->get_producer();

        auto csm = pool_out_val->get_consumers()[0];
        op_t &csm_op = csm.get_op();
        const size_t csm_offset = csm.get_offset();

        value_ptr zps_in_val = sub_zps_op.get_input_value(0);

        // connect pooling with a zps input
        zps_in_val->remove_consumer(*pool_op, 0);
        pool_op->connect_input(0, zps_in_val);

        // connect zps with a pooling using a fresh value
        logical_tensor_t pool_to_zps_lt
                = empty_logical_tensor_with_default_id();
        auto pool_to_zps_val
                = std::make_shared<value_t>(*pool_op, 0, pool_to_zps_lt, true);
        pool_to_zps_val->set_data_type(
                zps_in_val->get_logical_tensor().data_type);
        pool_op->connect_output(0, pool_to_zps_val);
        sub_zps_op.connect_input(0, pool_to_zps_val);

        // connect zps with a binary using a fresh value
        logical_tensor_t zps_to_bin_lt = empty_logical_tensor_with_default_id();
        auto zps_to_bin_val
                = std::make_shared<value_t>(sub_zps_op, 0, zps_to_bin_lt, true);
        zps_to_bin_val->set_data_type(
                zps_in_val->get_logical_tensor().data_type);
        sub_zps_op.connect_output(0, zps_to_bin_val);
        csm_op.connect_input(csm_offset, zps_to_bin_val);
    }

    return infer_shape(sg);
}

status_t fuse_to_shuffle(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_reshape) continue;

        if (cur_op->get_output_value(0)->get_consumers().size() != 1) continue;
        auto &next0 = cur_op->get_output_value(0)->get_consumers()[0].get_op();
        if (next0.get_kind() != op_kind::dnnl_transpose) continue;

        if (next0.get_output_value(0)->get_consumers().size() != 1) continue;
        auto &next1 = next0.get_output_value(0)->get_consumers()[0].get_op();
        if (next1.get_kind() != op_kind::dnnl_reshape) continue;

        fusion_groups.emplace_back(
                std::vector<op_t *> {cur_op.get(), &next0, &next1});
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *reshape0 = fusion_group[0];
        op_t *reshape1 = fusion_group[2];
        op_t *transpose = fusion_group[1];

        const auto res = shuffle_fusible(reshape0, reshape1, transpose);
        const bool fusible = res.first;
        if (!fusible) continue;

        op_ptr shuffle = std::make_shared<op_t>(op_kind::dnnl_shuffle);

        value_ptr in_value = reshape0->get_input_value(0);
        value_ptr out_value = reshape1->get_output_value(0);

        const auto src_shape = ltw(in_value->get_logical_tensor()).vdims();
        const auto attr_shape
                = reshape0->get_attr<std::vector<int64_t>>(op_attr::shape);
        const size_t axis = res.second.first;
        const int64_t group = res.second.second;
        shuffle->set_attr<int64_t>(op_attr::axis, static_cast<int64_t>(axis));
        shuffle->set_attr<int64_t>(op_attr::groups, group);

        shuffle->connect_input(0, in_value);
        in_value->remove_consumer(*reshape0, 0);

        shuffle->add_output(out_value);

        insert_empty_scratchpad(shuffle);

        for (auto &del_op : fusion_group) {
            rewriter.to_remove(del_op->shared_from_this());
        }

        rewriter.to_insert(shuffle);
    }

    rewriter.run();
    return status::success;
}

status_t fuse_post_ops(std::shared_ptr<subgraph_t> &sg) {
    // lambda function to fuse one post op into base primitive
    auto fuse_post_ops_func = [&](bool &changed) -> status_t {
        auto &mgr = sg->fusion_info_mgr_;
        subgraph_rewriter_t rewriter(sg);

        std::vector<std::pair<op_t *, op_t *>> fuse_groups;

        std::set<op_t *> visited;
        status_t ret = topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
            const auto &pops_fusible_map = get_post_ops_fusible_map();

            auto base_op_kind = op->get_kind();
            // only fuse two ops each time
            if (!pops_fusible_map.count(base_op_kind) || visited.count(op) != 0
                    || !fuse_groups.empty())
                return status::success;

            auto out_val = op->get_output_values()[0];
            auto consumers = out_val->get_consumers();

            // The base op should have and only have one consumer, it's
            // the post op to be fused
            if (consumers.size() != 1) return status::success;
            auto &post_op = consumers[0].get_op();

            // check if fusible
            // TODO(qun) make sure bn only fuse relu
            auto post_op_kind = post_op.get_kind();
            bool not_fusible
                    = (!pops_fusible_map.at(base_op_kind).count(post_op_kind))
                    || (post_op_kind == op_kind::dnnl_binary
                            && !post_binary_fusible(op, &post_op))
                    || (post_op_kind == op_kind::dnnl_convolution
                            && !post_depthwise_conv_fusible(op, &post_op));
            if (not_fusible) { return status::success; }

            // push fusible pair to fuse group for later fusion
            fuse_groups.emplace_back(std::pair<op_t *, op_t *> {op, &post_op});
            visited.insert(op);
            visited.insert(&post_op);
            return status::success;
        });

        if (ret != status::success) return ret;

        if (fuse_groups.empty()) {
            changed = false;
            return status::success;
        }

        for (auto &fuse_group : fuse_groups) {
            auto base_op = fuse_group.first;
            auto post_op = fuse_group.second;
            // post op fuse to which predecessor
            size_t fuse_op_predecessor_offset = base_op->get_output_value(0)
                                                        ->get_consumers()[0]
                                                        .get_offset();

            int64_t key = -1;
            if (base_op->has_attr(op_attr::fusion_info_key)
                    && base_op->get_attr<int64_t>(op_attr::fusion_info_key)
                            != -1) {
                key = base_op->get_attr<int64_t>(op_attr::fusion_info_key);
            } else {
                key = mgr.init_info();
                base_op->set_attr<int64_t>(op_attr::fusion_info_key, key);
            }

            fusion_info_t &fusion_info = mgr.get_mutable_info(key);

            if (post_op->get_kind() == op_kind::dnnl_eltwise) {
                float scale = 1.f;

                const auto alg = static_cast<dnnl::algorithm>(
                        post_op->get_attr<int64_t>(op_attr::alg_kind));

                // for BatchNormForwardTraining, set dnnl_fuse_norm_relu flag
                // instead of post op
                if ((base_op->get_kind() == op_kind::dnnl_batchnorm
                            && base_op->get_attr<bool>(op_attr::is_training))
                        && alg == dnnl::algorithm::eltwise_relu) {
                    base_op->set_attr<bool>(op_attr::fuse_relu, true);
                    // remove the fused post_ops op
                    rewriter.fuse_op_to_predecessor(post_op->shared_from_this(),
                            fuse_op_predecessor_offset);
                    auto tmp_ptr = base_op->shared_from_this();
                    insert_empty_workspace(tmp_ptr);
                    continue;
                }
                fusion_info.append_post_eltwise(
                        post_op->shared_from_this(), scale);
            } else if (post_op->get_kind() == op_kind::dnnl_binary
                    && static_cast<dnnl::algorithm>(
                               post_op->get_attr<int64_t>(op_attr::alg_kind))
                            == dnnl::algorithm::binary_add) {
                // If the other in value of Add has mul_scales producer,
                // then this pattern is a int8 pattern
                size_t mul_scale_op_offset = 2;
                auto other_in_val0 = post_op->get_input_value(
                        1 - fuse_op_predecessor_offset);

                if (other_in_val0->has_producer()
                        && (other_in_val0->get_producer().get_kind()
                                        == op_kind::dnnl_mul_scales
                                || other_in_val0->get_producer().get_kind()
                                        == op_kind::dnnl_sub_zps)) {
                    mul_scale_op_offset = 1 - fuse_op_predecessor_offset;
                }

                auto other_in_val1
                        = post_op->get_input_value(fuse_op_predecessor_offset);

                if (mul_scale_op_offset != 2
                        && is_output_scales_supported(base_op->get_kind())
                        && ltw(other_in_val0->get_logical_tensor()).vdims()
                                == ltw(other_in_val1->get_logical_tensor())
                                           .vdims()) {
                    // for int8 cases (excluding OPs which don't support
                    // output scales attribute and its inputs don't have
                    // same dims)
                    auto in_val = post_op->get_input_value(mul_scale_op_offset);
                    auto &pre_op = in_val->get_producer();
                    std::vector<float> scales {1.f};
                    int32_t zp = 0;
                    if (pre_op.get_kind() == op_kind::dnnl_mul_scales) {
                        scales = pre_op.get_attr<std::vector<float>>(
                                op_attr::scales);
                        assert(scales.size() == 1); // per tensor
                        auto tmp = pre_op.get_input_value(0);

                        if (tmp->has_producer()
                                && tmp->get_producer().get_kind()
                                        == op_kind::dnnl_sub_zps) {
                            auto &sub_op = tmp->get_producer();
                            auto zps = sub_op.get_attr<std::vector<int64_t>>(
                                    op_attr::zps);
                            zp = static_cast<int32_t>(zps[0]);
                            assert(scales.size() == zps.size());
                            rewriter.fuse_op_to_successor(
                                    sub_op.shared_from_this());
                        }
                    } else {
                        auto zps = pre_op.get_attr<std::vector<int64_t>>(
                                op_attr::zps);
                        zp = static_cast<int32_t>(zps[0]);
                        assert(scales.size() == zps.size());
                    }
                    rewriter.fuse_op_to_successor(pre_op.shared_from_this());
                    fusion_info.append_post_binary(post_op->shared_from_this(),
                            std::vector<size_t> {base_op->num_inputs()},
                            scales[0], zp);
                } else {
                    fusion_info.append_post_binary(post_op->shared_from_this(),
                            std::vector<size_t> {base_op->num_inputs()});
                }
            } else if (post_op->get_kind() == op_kind::dnnl_binary
                    && static_cast<dnnl::algorithm>(
                               post_op->get_attr<int64_t>(op_attr::alg_kind))
                            != dnnl::algorithm::binary_add) {
                fusion_info.append_post_binary(post_op->shared_from_this(),
                        std::vector<size_t> {base_op->num_inputs()});
            } else if (post_op->get_kind() == op_kind::dnnl_convolution) {
                // TODO(xx) if dw_conv has bias, we also need to put it into the
                // unfused input indices
                fusion_info.append_post_dw_conv(post_op->shared_from_this(),
                        std::vector<size_t> {base_op->num_inputs()});
            } else {
                // unsupported post ops
                continue;
            }

            // remove the fused post_ops op
            rewriter.fuse_op_to_predecessor(
                    post_op->shared_from_this(), fuse_op_predecessor_offset);
        }

        rewriter.run();
        changed = true;
        return status::success;
    };

    int cnt = 0;
    const int max_num_limit = static_cast<int>(sg->num_ops());

    bool changed = true;
    do {
        auto ret = fuse_post_ops_func(changed);
        if (ret != status::success) return ret;
        cnt++;
    } while (changed && cnt <= max_num_limit);

    assertm(cnt <= max_num_limit + 1,
            "Failed to fuse all post ops since there has unsupported ones.");
    if (cnt > max_num_limit + 1) return status::unimplemented;
    return status::success;
}

status_t fuse_src_zero_points(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;

    std::vector<op_t *> zp_ops;

    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_sub_zps
                || visited.count(cur_op.get()) != 0)
            continue;
        zp_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &zp_op : zp_ops) {
        assertm(zp_op->num_outputs() == 1,
                "zp_op should have only one output value.");
        auto out_val = zp_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();

        if (!has_int8_support(consumers[0].get_op().get_kind())) continue;

        auto &next_op = consumers[0].get_op();
        auto offset = consumers[0].get_offset();
        if (offset == 0 || offset == 1) {
            int64_t key = -1;
            if (next_op.has_attr(op_attr::fusion_info_key)) {
                key = next_op.get_attr<int64_t>(op_attr::fusion_info_key);
            } else {
                key = mgr.init_info();
                next_op.set_attr<int64_t>(op_attr::fusion_info_key, key);
            }
            fusion_info_t &fusion_info = mgr.get_mutable_info(key);
            bool not_all_zero = true;
            if (zp_op->has_attr(op_attr::with_runtime_zps)
                    && zp_op->get_attr<bool>(op_attr::with_runtime_zps)) {
                if (zp_op->num_inputs() > 1
                        && zp_op->get_input_value(1)->has_producer()
                        && zp_op->get_input_op(1)->get_kind()
                                == op_kind::dnnl_constant_zps) {
                    auto &const_op = zp_op->get_input_value(1)->get_producer();
                    auto zps = const_op.get_attr<std::vector<int64_t>>(
                            op_attr::zps);
                    not_all_zero = !utils::all_zero(zps);
                    if (!not_all_zero) {
                        rewriter.to_remove((&const_op)->shared_from_this());
                    }
                }
                value_ptr in0_val = zp_op->get_input_value(0);
                in0_val->remove_consumer(*zp_op, 0);
                value_ptr in1_val = zp_op->get_input_value(1);
                in1_val->remove_consumer(*zp_op, 1);
                value_ptr out_val = zp_op->get_output_value(0);
                auto consumers = out_val->get_consumers();
                in0_val->add_consumer(next_op, offset);
                next_op.connect_input(offset, in0_val);
                if (not_all_zero) {
                    next_op.add_input(in1_val);
                    in1_val->add_consumer(next_op, next_op.num_inputs() - 1);
                    fusion_info.set_zero_points(
                            zp_op->shared_from_this(), true, offset);
                }
                rewriter.to_remove(zp_op->shared_from_this());
            } else {
                auto zps = zp_op->get_attr<std::vector<int64_t>>(op_attr::zps);
                not_all_zero = !utils::all_zero(zps);
                if (not_all_zero) {
                    assertm(zps.size() == 1,
                            "zp attr only support scalar zp, need to use "
                            "runtime arg to support vector zp");
                    fusion_info.set_zero_points(
                            zp_op->shared_from_this(), true, offset);
                }
                rewriter.fuse_op_to_successor(zp_op->shared_from_this());
            }
            if (not_all_zero) {
                fusion_info.set_zero_points(
                        zp_op->shared_from_this(), true, offset);
            }
        }
    }
    rewriter.run();
    return infer_shape(sg);
}

status_t fuse_src_scales(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;

    std::vector<op_t *> scales_ops;

    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_mul_scales
                || visited.count(cur_op.get()) != 0)
            continue;
        scales_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &scale_op : scales_ops) {
        assertm(scale_op->num_outputs() == 1,
                "scale_op should have only one output value.");
        auto out_val = scale_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;
        if (!impl::utils::one_of(consumers[0].get_op().get_kind(),
                    op_kind::dnnl_matmul, op_kind::dnnl_convolution,
                    op_kind::dnnl_convtranspose, op_kind::dnnl_reorder))
            continue;

        auto &next_op = consumers[0].get_op();
        auto offset = consumers[0].get_offset();
        if (offset == 0 || offset == 1) {
            int64_t key = -1;
            if (next_op.has_attr(op_attr::fusion_info_key)) {
                key = next_op.get_attr<int64_t>(op_attr::fusion_info_key);
            } else {
                key = mgr.init_info();
                next_op.set_attr<int64_t>(op_attr::fusion_info_key, key);
            }
            fusion_info_t &fusion_info = mgr.get_mutable_info(key);
            if (scale_op->has_attr(op_attr::with_runtime_scales)
                    && scale_op->get_attr<bool>(op_attr::with_runtime_scales)) {
                value_ptr in0_val = scale_op->get_input_value(0);
                in0_val->remove_consumer(*scale_op, 0);
                value_ptr in1_val = scale_op->get_input_value(1);
                in1_val->remove_consumer(*scale_op, 1);
                value_ptr out_val = scale_op->get_output_value(0);
                auto consumers = out_val->get_consumers();
                in0_val->add_consumer(next_op, offset);
                next_op.connect_input(offset, in0_val);
                next_op.add_input(in1_val);
                in1_val->add_consumer(next_op, next_op.num_inputs() - 1);
                fusion_info.set_runtime_scales(
                        scale_op->shared_from_this(), true, offset);
                rewriter.to_remove(scale_op->shared_from_this());
            } else {
                assertm(false, "src scales must be runtime scales.");
            }
        }
    }
    rewriter.run();
    return infer_shape(sg);
}

status_t fuse_dst_scales(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;
    subgraph_rewriter_t rewriter(sg);

    std::vector<std::pair<op_t *, op_t *>> fuse_groups;

    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_convolution
                    && cur_op->get_kind() != op_kind::dnnl_matmul
                    && cur_op->get_kind() != op_kind::dnnl_convtranspose
                    && cur_op->get_kind() != op_kind::dnnl_softmax
                    && cur_op->get_kind() != op_kind::dnnl_layernorm
                    && cur_op->get_kind() != op_kind::dnnl_reorder)
                || visited.count(cur_op.get()) != 0)
            continue;
        auto out_val = cur_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.size() != 1) continue;
        auto &next_op = consumers[0].get_op();
        if (next_op.get_kind() != op_kind::dnnl_mul_scales) continue;
        fuse_groups.emplace_back(
                std::pair<op_t *, op_t *> {cur_op.get(), &next_op});
        visited.insert(cur_op.get());
        visited.insert(&next_op);
    }

    for (auto &fuse_group : fuse_groups) {
        auto base_op = fuse_group.first;
        auto scale_op = fuse_group.second;

        int64_t key = -1;
        if (base_op->has_attr(op_attr::fusion_info_key)
                && base_op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
            key = base_op->get_attr<int64_t>(op_attr::fusion_info_key);
        } else {
            key = mgr.init_info();
            base_op->set_attr<int64_t>(op_attr::fusion_info_key, key);
        }

        fusion_info_t &fusion_info = mgr.get_mutable_info(key);
        fusion_info.set_runtime_scales(scale_op->shared_from_this(), false, 0);
        rewriter.fuse_op_to_predecessor(scale_op->shared_from_this());
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t convert_to_runtime_dst_scales(std::shared_ptr<subgraph_t> &sg) {
    std::set<op_t *> visited;
    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_mul_scales
                || cur_op->num_inputs() != 1
                || !cur_op->get_input_value(0)->has_producer()
                || !impl::utils::one_of(cur_op->get_input_op(0)->get_kind(),
                        op_kind::dnnl_softmax, op_kind::dnnl_layernorm,
                        op_kind::dnnl_convolution, op_kind::dnnl_matmul,
                        op_kind::dnnl_convtranspose, op_kind::dnnl_reorder)
                || visited.count(cur_op.get()))
            continue;

        visited.insert(cur_op.get());
        // make scales as a constant input
        op_ptr const_data_op;
        auto scales = cur_op->get_attr<std::vector<float>>(op_attr::scales);
        // TODO(Xinyu): do not inv scales in qdata once oscales removed.
        scales = dnnl_impl::utils::fmap(
                scales, [](float s) { return 1.f / s; });
        const_data_op = std::make_shared<op_t>(op_kind::dnnl_constant_scales);
        const_data_op->set_attr(op_attr::scales, scales);
        std::vector<int64_t> dst_shape(1, scales.size());
        const_data_op->set_attr(op_attr::shape, dst_shape);
        logical_tensor_t const_data_dst_lt
                = empty_logical_tensor_with_default_id();
        auto const_data_dst_value = std::make_shared<value_t>(
                *const_data_op, 0, const_data_dst_lt, true);
        const_data_dst_value->set_data_type(graph::data_type::f32);
        const_data_dst_value->set_layout_type(layout_type::strided);
        const_data_dst_value->set_strides({1});
        const_data_op->add_output(const_data_dst_value);
        cur_op->set_attr(op_attr::with_runtime_scales, true);
        cur_op->remove_attr(op_attr::scales);

        // connect mul_scale and constant data
        cur_op->connect_input(1, const_data_dst_value);
        rewriter.to_insert(const_data_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t convert_bias_to_f32(std::shared_ptr<subgraph_t> &sg) {
    std::set<op_t *> visited;
    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : sg->get_ops()) {
        if (!impl::utils::one_of(cur_op->get_kind(), op_kind::dnnl_convolution,
                    op_kind::dnnl_matmul)
                || cur_op->num_inputs() < 3
                || !cur_op->get_input_value(0)->has_producer()
                || !cur_op->get_input_value(1)->has_producer()
                || cur_op->get_input_op(0)->get_kind()
                        != op_kind::dnnl_mul_scales
                || cur_op->get_input_op(1)->get_kind()
                        != op_kind::dnnl_mul_scales
                || ltw(cur_op->get_input_value(2)->get_logical_tensor())
                                .data_type()
                        != impl::data_type::bf16
                || visited.count(cur_op.get()))
            continue;

        visited.insert(cur_op.get());
        op_ptr tc_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        rewriter.insert_op_before(tc_op, cur_op->shared_from_this(), 2);
        // Some of oneDNN's conv primitive implementation can't support bf16
        // bias
        tc_op->get_output_value(0)->set_data_type(graph::data_type::f32);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t fuse_dst_zero_points(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;
    std::vector<op_t *> zp_ops;
    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_add_zps
                || visited.count(cur_op.get()) != 0)
            continue;
        zp_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &zp_op : zp_ops) {
        assertm(zp_op->num_outputs() == 1,
                "zp_op should have only one output value.");
        auto out_val = zp_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();

        auto in_val = zp_op->get_input_values()[0];
        if (!in_val->has_producer()) continue;
        auto &prv_op = in_val->get_producer();

        if (!has_int8_support(prv_op.get_kind())) continue;

        int64_t key = -1;
        if (prv_op.has_attr(op_attr::fusion_info_key)) {
            key = prv_op.get_attr<int64_t>(op_attr::fusion_info_key);
        } else {
            key = mgr.init_info();
            prv_op.set_attr<int64_t>(op_attr::fusion_info_key, key);
        }

        fusion_info_t &fusion_info = mgr.get_mutable_info(key);
        fusion_info.set_zero_points(zp_op->shared_from_this(), false, 0);
        rewriter.fuse_op_to_predecessor(zp_op->shared_from_this());
    }
    rewriter.run();
    return infer_shape(sg);
}

status_t insert_bn_folding(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_t *> bn_ops;

    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_batchnorm
                || visited.count(cur_op.get()) != 0)
            continue;

        // only fold bn inference
        if (cur_op->get_attr<bool>(op_attr::is_training)) continue;

        // only folding conv_bn case
        auto in = cur_op->get_input_value(0);
        if (!in->has_producer()
                || in->get_producer().get_kind() != op_kind::dnnl_convolution)
            continue;

        // (TODO) skip on gpu when inputs dtype are mixtured because dnnl binary
        // primitive requires src0 and src1 has the same dtype. need support
        // dtype promotion when using binary primitive
        if (sg->get_engine_kind() == graph::engine_kind::gpu
                && cur_op->get_input_value(0)->get_logical_tensor().data_type
                        != cur_op->get_input_value(1)
                                   ->get_logical_tensor()
                                   .data_type) {
            continue;
        }

        bn_ops.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &bn_op : bn_ops) {
        auto &prv_op = bn_op->get_input_value(0)->get_producer();
        if (prv_op.get_kind() != op_kind::dnnl_convolution) continue;

        op_ptr bn_folding_op = std::make_shared<op_t>(op_kind::dnnl_bn_folding);

        bn_folding_op->set_attr<float>(
                op_attr::epsilon, bn_op->get_attr<float>(op_attr::epsilon));
        bn_folding_op->set_attr<std::string>(op_attr::data_format,
                bn_op->get_attr<std::string>(op_attr::data_format));

        bn_folding_op->set_attr<std::string>(op_attr::weights_format,
                prv_op.get_attr<std::string>(op_attr::weights_format));
        bn_folding_op->set_attr<bool>(
                op_attr::with_bias, prv_op.num_inputs() == 3);

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
                empty_logical_tensor_with_default_id(), true);
        updated_conv_wei->set_data_type(
                prv_op.get_input_value(1)->get_logical_tensor().data_type);
        bn_folding_op->add_output(updated_conv_wei);
        updated_conv_wei->add_consumer(prv_op, 1);
        prv_op.connect_input(1, updated_conv_wei);

        auto updated_conv_bias = std::make_shared<value_t>(*bn_folding_op, 1,
                empty_logical_tensor_with_default_id(), true);
        // when bias is none, f32 zero bias will be allocated
        const auto bias_dtype = prv_op.num_inputs() == 3
                ? prv_op.get_input_value(2)->get_logical_tensor().data_type
                : graph::data_type::f32;
        updated_conv_bias->set_data_type(bias_dtype);
        bn_folding_op->add_output(updated_conv_bias);
        updated_conv_bias->add_consumer(prv_op, 2);
        prv_op.connect_input(2, updated_conv_bias);

        // add scratchpad output for bn_folding
        insert_empty_scratchpad(bn_folding_op);

        auto bn_out_val = bn_op->get_output_value(0);
        prv_op.connect_output(0, bn_out_val);

        rewriter.to_remove(bn_op->shared_from_this());
        rewriter.to_insert(bn_folding_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t expand_convtranspose_scales(std::shared_ptr<subgraph_t> &sg) {
    for (const auto &op : sg->get_ops()) {
        if (op->get_kind() == op_kind::dnnl_convtranspose
                && op->get_input_value(0)->has_producer()
                && op->get_input_value(1)->has_producer()) {
            auto &in0 = op->get_input_value(0)->get_producer();
            auto &in1 = op->get_input_value(1)->get_producer();
            if (in0.get_kind() != op_kind::dnnl_mul_scales
                    || in1.get_kind() != op_kind::dnnl_mul_scales)
                continue;

            if (in1.has_attr(op_attr::qtype)
                    && in1.get_attr<std::string>(op_attr::qtype)
                            == "per_tensor")
                continue;
            auto dq_wei_scales
                    = in1.get_attr<std::vector<float>>(op_attr::scales);
            int64_t group = op->get_attr<int64_t>(op_attr::groups);
            if (group > 1) {
                // Currently for ConvTranspose, the output channel in weight tensor
                // (IC, OC/g, H, W) is not equal to the one in output tensor
                // (N, OC, H, W) if `groups` > 1, so the size of weight's
                // per-channel scale is not the same as the output channel in output
                // tensor, here we will broadcast scales from `OC/g` to `OC`.
                std::vector<float> expand_scales(
                        group * dq_wei_scales.size(), 0);
                for (size_t i = 0; i < expand_scales.size(); ++i)
                    expand_scales[i] = dq_wei_scales[i % dq_wei_scales.size()];
                in1.set_attr(op_attr::scales, expand_scales);
            }
        }
    }
    return status::success;
}

status_t conv_bwd_data_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_conv_bwd_data) continue;

        // insert permute
        bool need_permute_0 = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;
        bool need_permute_1 = cur_op->has_attr(op_attr::weights_format)
                ? (cur_op->get_attr<std::string>(op_attr::weights_format)
                        == "XIO")
                : false;

        if (need_permute_0) {
            // input permute
            auto in_ndims
                    = cur_op->get_input_value(0)->get_logical_tensor().ndims;
            auto in_perm = get_permutation(in_ndims, "NXC", "NCX");

            op_ptr in_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            in_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, in_perm);
            rewriter.insert_op_before(in_perm_op, cur_op, 0);

            // output permute
            auto out_ndims
                    = cur_op->get_output_value(0)->get_logical_tensor().ndims;
            auto out_perm = get_permutation(out_ndims, "NCX", "NXC");

            op_ptr out_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            out_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, out_perm);
            rewriter.insert_op_after(out_perm_op, cur_op, 0);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");
            if (cur_op->has_attr(op_attr::dst_shape)) {
                auto nxc_dst_shape = cur_op->get_attr<dims>(op_attr::dst_shape);
                auto ncx_dst_shape = canonicalize(nxc_dst_shape, "NXC");
                cur_op->set_attr<dims>(op_attr::dst_shape, ncx_dst_shape);
            }
        }

        if (need_permute_1) {
            auto wei_ndims
                    = cur_op->get_input_value(1)->get_logical_tensor().ndims;
            auto wei_perm = get_permutation(wei_ndims, "XIO", "OIX");

            op_ptr perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, wei_perm);
            rewriter.insert_op_before(perm_op, cur_op, 1);
            cur_op->set_attr<std::string>(op_attr::weights_format, "OIX");
        }

        // insert to_group
        auto groups = cur_op->get_attr<int64_t>(op_attr::groups);
        if (groups > 1) {
            op_ptr to_group_op = std::make_shared<op_t>(op_kind::dnnl_to_group);
            to_group_op->set_attr<int64_t>(op_attr::groups, groups);
            rewriter.insert_op_before(to_group_op, cur_op, 1);
            cur_op->set_attr<int64_t>(op_attr::groups, 1);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t conv_bwd_weights_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_conv_bwd_weights
                && cur_op->get_kind()
                        != op_kind::dnnl_convtranspose_bwd_weights)
            continue;

        const auto filter_shape_attr = cur_op->get_attr<std::vector<int64_t>>(
                op_attr::weights_shape);
        const bool is_filter_shape_default = std::all_of(
                filter_shape_attr.begin(), filter_shape_attr.end(),
                [](int64_t d) { return d == 0; });
        if (is_filter_shape_default) {
            const std::vector<int64_t> filter_shape
                    = ltw(cur_op->get_output_value(0)->get_logical_tensor())
                              .vdims();
            cur_op->set_attr(op_attr::weights_shape, filter_shape);
        }

        // insert permute
        bool need_permute_0 = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;
        bool need_permute_1 = cur_op->has_attr(op_attr::weights_format)
                ? (cur_op->get_attr<std::string>(op_attr::weights_format)
                        != "OIX")
                : false;

        if (need_permute_0) {
            // input permute
            auto in0_ndims
                    = cur_op->get_input_value(0)->get_logical_tensor().ndims;
            auto in0_perm = get_permutation(in0_ndims, "NXC", "NCX");

            op_ptr in0_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            in0_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, in0_perm);
            rewriter.insert_op_before(in0_perm_op, cur_op, 0);

            auto in1_ndims
                    = cur_op->get_input_value(1)->get_logical_tensor().ndims;
            auto in1_perm = get_permutation(in1_ndims, "NXC", "NCX");

            op_ptr in1_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            in1_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, in1_perm);
            rewriter.insert_op_before(in1_perm_op, cur_op, 1);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");
        }
        // output permute
        if (need_permute_1) {
            auto out_ndims
                    = cur_op->get_output_value(0)->get_logical_tensor().ndims;
            std::string filter_format
                    = cur_op->get_attr<std::string>(op_attr::weights_format);
            std::vector<int64_t> out_perm
                    = get_permutation(out_ndims, "OIX", filter_format);

            op_ptr out_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            out_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, out_perm);
            rewriter.insert_op_after(out_perm_op, cur_op, 0);

            const auto filter_shape_attr
                    = cur_op->get_attr<std::vector<int64_t>>(
                            op_attr::weights_shape);
            const auto filter_shape_as_oix
                    = canonicalize(filter_shape_attr, filter_format);
            cur_op->set_attr<dims>(op_attr::weights_shape, filter_shape_as_oix);
            cur_op->set_attr<std::string>(op_attr::weights_format, "OIX");
        }

        // insert from_group
        auto groups = cur_op->get_attr<int64_t>(op_attr::groups);
        if (groups > 1) {
            op_ptr from_group_op
                    = std::make_shared<op_t>(op_kind::dnnl_from_group);
            from_group_op->set_attr<int64_t>(op_attr::groups, groups);
            rewriter.insert_op_after(from_group_op, cur_op, 0);

            if (cur_op->get_kind() == op_kind::dnnl_convtranspose_bwd_weights)
                from_group_op->set_attr<bool>(op_attr::is_convtranspose, true);
        }

        cur_op->set_attr<bool>(op_attr::canonicalized, true);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t pool_fwd_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_pool) continue;

        // insert permute
        bool need_permute = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;

        if (need_permute) {
            // src permute
            auto in0_ndims
                    = cur_op->get_input_value(0)->get_logical_tensor().ndims;
            auto in0_perm = get_permutation(in0_ndims, "NXC", "NCX");

            op_ptr in0_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            in0_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, in0_perm);
            rewriter.insert_op_before(in0_perm_op, cur_op, 0);

            // dst permute
            auto out0_ndims
                    = cur_op->get_output_value(0)->get_logical_tensor().ndims;
            auto out0_perm = get_permutation(out0_ndims, "NCX", "NXC");

            op_ptr out0_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            out0_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, out0_perm);
            rewriter.insert_op_after(out0_perm_op, cur_op, 0);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t pool_bwd_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_pool_bwd) continue;

        // insert permute
        bool need_permute = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;

        if (need_permute) {
            // diff_dst permute
            auto in0_ndims
                    = cur_op->get_input_value(0)->get_logical_tensor().ndims;
            auto in0_perm = get_permutation(in0_ndims, "NXC", "NCX");

            op_ptr in0_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            in0_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, in0_perm);
            rewriter.insert_op_before(in0_perm_op, cur_op, 0);

            // src permute
            if (cur_op->get_attr<std::string>(op_attr::kind) == "maxpool") {
                auto src_ndims = cur_op->get_input_value(2)
                                         ->get_logical_tensor()
                                         .ndims;
                auto src_perm = get_permutation(src_ndims, "NXC", "NCX");

                op_ptr src_perm_op
                        = std::make_shared<op_t>(op_kind::dnnl_permute);
                src_perm_op->set_attr<std::vector<int64_t>>(
                        op_attr::permutation, src_perm);
                rewriter.insert_op_before(src_perm_op, cur_op, 2);
            }

            // diff_src permute
            auto out0_ndims
                    = cur_op->get_output_value(0)->get_logical_tensor().ndims;
            auto out0_perm = get_permutation(out0_ndims, "NCX", "NXC");

            op_ptr out_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            out_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, out0_perm);
            rewriter.insert_op_after(out_perm_op, cur_op, 0);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");

            if (cur_op->has_attr(op_attr::src_shape)) {
                auto nxc_dst_shape = cur_op->get_attr<dims>(op_attr::src_shape);
                auto ncx_dst_shape = canonicalize(nxc_dst_shape, "NXC");
                cur_op->set_attr<dims>(op_attr::src_shape, ncx_dst_shape);
            }
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t fuse_mul_sigmoid_to_swish(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> swish_patterns;
    std::vector<size_t> mul_other_offsets;

    // find all swish pattern in subgraph
    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_eltwise
                || visited.count(cur_op.get()) != 0)
            continue;

        if (static_cast<dnnl::algorithm>(
                    cur_op->get_attr<int64_t>(op_attr::alg_kind))
                != dnnl::algorithm::eltwise_logistic)
            continue;

        visited.insert(cur_op.get());

        /* check if the sigmoid op belongs to a swish pattern.
        // A swish pattern is composed by a sigmoid op and a multiply op:
        //        any
        //       /   \
        //  sigmoid   |
        //       \   /
        //      multiply
        //         |
        //        any
        */
        auto sigmoid_out = cur_op->get_output_value(0);
        auto sigmoid_csm = sigmoid_out->get_consumers();
        if (sigmoid_csm.size() != 1) continue;

        auto &csm_op = sigmoid_csm[0].get_op();
        if (csm_op.get_kind() != op_kind::dnnl_binary) continue;

        if (static_cast<dnnl::algorithm>(
                    csm_op.get_attr<int64_t>(op_attr::alg_kind))
                != dnnl::algorithm::binary_mul)
            continue;

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

    if (swish_patterns.empty()) return status::success;

    // fuse swish pattern to a swish op
    subgraph_rewriter_t rewriter(sg);
    for (size_t i = 0; i < swish_patterns.size(); i++) {
        op_t *sigmoid_op = swish_patterns[i][0];
        op_t *mul_op = swish_patterns[i][1];
        size_t mul_other_offset = mul_other_offsets[i];

        op_ptr swish_op = std::make_shared<op_t>(op_kind::dnnl_eltwise);
        swish_op->set_attr<int64_t>(op_attr::alg_kind,
                static_cast<int64_t>(dnnl::algorithm::eltwise_swish));
        swish_op->set_attr<float>(op_attr::alpha, (float)1.0);

        auto in_val = sigmoid_op->get_input_value(0);
        in_val->remove_consumer(*sigmoid_op, 0);
        in_val->remove_consumer(*mul_op, mul_other_offset);
        swish_op->connect_input(0, in_val);

        auto out_val = mul_op->get_output_value(0);
        swish_op->add_output(out_val);
        out_val->set_producer(*swish_op);

        insert_empty_scratchpad(swish_op);

        rewriter.to_insert(swish_op);
        rewriter.to_remove(sigmoid_op->shared_from_this());
        rewriter.to_remove(mul_op->shared_from_this());
    }

    rewriter.run();
    return status::success;
}

status_t fuse_typecast_to_matmul_or_conv(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_matmul
                    && cur_op->get_kind() != op_kind::dnnl_convolution)
                || !cur_op->get_input_value(0)->has_producer()
                || !cur_op->get_input_value(1)->has_producer())
            continue;
        auto &in0 = cur_op->get_input_value(0)->get_producer();
        auto &in1 = cur_op->get_input_value(1)->get_producer();
        if (is_typecast(&in0) && is_typecast(&in1)
                && in0.get_input_value(0)->has_producer()
                && in1.get_input_value(0)->has_producer()
                && in0.get_input_value(0)->get_producer().get_kind()
                        == op_kind::dnnl_mul_scales
                && in1.get_input_value(0)->get_producer().get_kind()
                        == op_kind::dnnl_mul_scales)
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in0, &in1});
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *base_op = fusion_group[0];
        op_t *in0 = fusion_group[1];
        op_t *in1 = fusion_group[2];

        // update the connection relationship between matmul/convolution and
        // typecast ops
        auto in0_value = in0->get_input_value(0);
        auto in1_value = in1->get_input_value(0);
        base_op->connect_input(0, in0_value);
        base_op->connect_input(1, in1_value);
        in0_value->remove_consumer(*in0, 0);
        in1_value->remove_consumer(*in1, 0);

        rewriter.to_remove(in0->shared_from_this());
        rewriter.to_remove(in1->shared_from_this());
    }

    rewriter.run();
    return status::success;
}

status_t fuse_typecast_to_add(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_binary
                || static_cast<dnnl::algorithm>(
                           cur_op->get_attr<int64_t>(op_attr::alg_kind))
                        != dnnl::algorithm::binary_add)
            continue;
        if (!(cur_op->get_input_value(0)->has_producer()
                    && cur_op->get_input_value(1)->has_producer()))
            continue;

        auto &in0 = cur_op->get_input_value(0)->get_producer();
        auto &in1 = cur_op->get_input_value(1)->get_producer();
        if (is_typecast(&in0)
                && (in1.get_kind() == op_kind::dnnl_matmul
                        || in1.get_kind() == op_kind::dnnl_convolution)) {
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in0});
        } else if (is_typecast(&in1)
                && (in0.get_kind() == op_kind::dnnl_matmul
                        || in0.get_kind() == op_kind::dnnl_convolution)) {
            fusion_groups.emplace_back(
                    std::vector<op_t *> {cur_op.get(), &in1});
        } else {
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *add_op = fusion_group[0];
        op_t *typecast_op = fusion_group[1];

        op_ptr new_add_op = std::make_shared<op_t>(op_kind::dnnl_binary);
        new_add_op->merge_attributes(add_op->get_attributes());

        // update the connection relationship between add and typecast ops
        auto tc_in = typecast_op->get_input_value(0);
        auto in0 = add_op->get_input_value(0);
        auto in1 = add_op->get_input_value(1);
        in0->remove_consumer(*add_op, 0);
        in1->remove_consumer(*add_op, 1);
        if (is_typecast(&in0->get_producer())) {
            new_add_op->connect_input(0, tc_in);
            new_add_op->connect_input(1, in1);
            tc_in->remove_consumer(*typecast_op, 0);
        } else {
            new_add_op->connect_input(1, tc_in);
            new_add_op->connect_input(0, in0);
            tc_in->remove_consumer(*typecast_op, 0);
        }

        auto out_val = add_op->get_output_value(0);
        new_add_op->add_output(out_val);
        out_val->set_producer(*new_add_op);

        auto scratchpad_val = add_op->get_output_value(1);
        new_add_op->connect_output(1, scratchpad_val);

        // delete original matmul/convolution and typecast ops
        for (auto &del_op : fusion_group) {
            rewriter.to_remove(del_op->shared_from_this());
        }

        rewriter.to_insert(new_add_op);
    }

    rewriter.run();
    return status::success;
}

status_t fuse_post_typecast_to_predecessor(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (!impl::utils::one_of(cur_op->get_kind(), op_kind::dnnl_matmul,
                    op_kind::dnnl_convolution, op_kind::dnnl_eltwise,
                    op_kind::dnnl_binary, op_kind::dnnl_softmax,
                    op_kind::dnnl_layernorm))
            continue;
        auto out = cur_op->get_output_value(0);
        if (out->get_consumers().size() != 1) continue;
        auto &next_op = out->get_consumers()[0].get_op();

        if (!is_typecast(&next_op)) continue;
        auto tc_out = next_op.get_output_value(0);
        if (tc_out->get_consumers().size() > 1) continue;
        if (tc_out->get_consumers().size() == 1) {
            // bf16-int8 mix precision case
            auto &next_next_op = tc_out->get_consumers()[0].get_op();
            out->remove_consumer(next_op, 0);
            tc_out->remove_consumer(next_next_op, 0);
            next_next_op.connect_input(0, out);
            out->set_data_type(tc_out->get_logical_tensor().data_type);
            fusion_groups.emplace_back(std::vector<op_t *> {&next_op});
        } else {
            // tc has no consumer in the subgraph
            // which means the fp32-in-bf16 out case
            cur_op->connect_output(0, tc_out);
            fusion_groups.emplace_back(std::vector<op_t *> {&next_op});
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups)
        // delete original typecast ops
        for (auto &del_op : fusion_group) {
            rewriter.to_remove(del_op->shared_from_this());
        }
    rewriter.run();
    return status::success;
}

status_t fuse_reciprocal_mul_to_div(std::shared_ptr<subgraph_t> &sg) {
    /* transformation below graphs
    Case 1:
        in0       in1
         \         |               in0    in1
          \     reciprocal          \      /
           \       /        --->      div
              mul                      |
               |                     out0
              out0

    Case 2:
        in0       in1
         \         |               in1    in0
      reciprocal   |                \      /
           \       /        --->      div
              mul                      |
               |                      out0
              out0
    */

    std::vector<std::pair<op_t *, op_t *>> div_patterns;
    std::vector<size_t> mul_other_offsets;
    std::set<op_t *> visited;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_eltwise
                || visited.count(cur_op.get()) != 0)
            continue;

        auto is_reciprocal = [&cur_op]() -> bool {
            bool ok = static_cast<dnnl::algorithm>(
                              cur_op->get_attr<int64_t>(op_attr::alg_kind))
                    == dnnl::algorithm::eltwise_pow;
            if (!ok) return false;

            // check attribute alpha
            float alpha = 0.f;
            if (cur_op->has_attr(op_attr::alpha))
                alpha = cur_op->get_attr<float>(op_attr::alpha);
            if (!utils::compare_float(alpha, 1.f)) return false;

            // check attribute beta
            float beta = 0.f;
            if (cur_op->has_attr(op_attr::beta))
                beta = cur_op->get_attr<float>(op_attr::beta);
            if (!utils::compare_float(beta, -1.f)) return false;
            return true;
        };

        if (!is_reciprocal()) continue;

        visited.insert(cur_op.get());

        auto reciprocal_out = cur_op->get_output_value(0);
        auto reciprocal_csm = reciprocal_out->get_consumers();
        if (reciprocal_csm.size() != 1) continue;

        auto &csm_op = reciprocal_csm[0].get_op();
        if (csm_op.get_kind() != op_kind::dnnl_binary
                || static_cast<dnnl::algorithm>(
                           csm_op.get_attr<int64_t>(op_attr::alg_kind))
                        != dnnl::algorithm::binary_mul)
            continue;

        // offset should be 0 or 1
        size_t offset = reciprocal_csm[0].get_offset();
        size_t mul_other_offset = 1 - offset;
        mul_other_offsets.emplace_back(mul_other_offset);

        div_patterns.emplace_back(
                std::pair<op_t *, op_t *> {cur_op.get(), &csm_op});
    }

    if (div_patterns.empty()) return status::success;

    subgraph_rewriter_t rewriter(sg);
    for (size_t i = 0; i < div_patterns.size(); ++i) {
        auto reciprocal_op = div_patterns[i].first;
        auto mul_op = div_patterns[i].second;
        auto mul_other_offset = mul_other_offsets[i];

        op_ptr div_op = std::make_shared<op_t>(op_kind::dnnl_binary);
        div_op->set_attr<int64_t>(op_attr::alg_kind,
                static_cast<int64_t>(dnnl::algorithm::binary_div));

        auto mul_other_in_val = mul_op->get_input_value(mul_other_offset);
        mul_other_in_val->remove_consumer(*mul_op, mul_other_offset);
        div_op->connect_input(0, mul_other_in_val);

        auto reciprocal_in_val = reciprocal_op->get_input_value(0);
        reciprocal_in_val->remove_consumer(*reciprocal_op, 0);
        div_op->connect_input(1, reciprocal_in_val);

        auto mul_out_val = mul_op->get_output_value(0);
        div_op->add_output(mul_out_val);
        mul_out_val->set_producer(*div_op);

        insert_empty_scratchpad(div_op);

        rewriter.to_insert(div_op);
        rewriter.to_remove(reciprocal_op->shared_from_this());
        rewriter.to_remove(mul_op->shared_from_this());
    }
    rewriter.run();
    return status::success;
}

status_t batchnorm_bwd_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_batchnorm_bwd) continue;

        // insert permute
        bool need_permute = cur_op->has_attr(op_attr::data_format)
                ? (cur_op->get_attr<std::string>(op_attr::data_format) == "NXC")
                : false;

        if (need_permute) {
            // input0 permute
            auto in0_ndims
                    = cur_op->get_input_value(0)->get_logical_tensor().ndims;
            auto in0_perm = get_permutation(in0_ndims, "NXC", "NCX");

            op_ptr in_perm_op_0 = std::make_shared<op_t>(op_kind::dnnl_permute);
            in_perm_op_0->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, in0_perm);
            rewriter.insert_op_before(in_perm_op_0, cur_op, 0);

            // input1 permute
            auto in1_ndims
                    = cur_op->get_input_value(1)->get_logical_tensor().ndims;
            auto in1_perm = get_permutation(in1_ndims, "NXC", "NCX");

            op_ptr in_perm_op_1 = std::make_shared<op_t>(op_kind::dnnl_permute);
            in_perm_op_1->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, in1_perm);
            rewriter.insert_op_before(in_perm_op_1, cur_op, 1);

            // output permute
            auto out_ndims
                    = cur_op->get_output_value(0)->get_logical_tensor().ndims;
            auto out_perm = get_permutation(out_ndims, "NCX", "NXC");

            op_ptr out_perm_op = std::make_shared<op_t>(op_kind::dnnl_permute);
            out_perm_op->set_attr<std::vector<int64_t>>(
                    op_attr::permutation, out_perm);
            rewriter.insert_op_after(out_perm_op, cur_op, 0);

            cur_op->set_attr<std::string>(op_attr::data_format, "NCX");
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t fuse_to_dnnl_sum(std::shared_ptr<subgraph_t> &sg) {
    auto is_non_broadcast_add = [](const op_t *op) {
        return op->get_kind() == op_kind::dnnl_binary
                && static_cast<dnnl::algorithm>(
                           op->get_attr<int64_t>(op_attr::alg_kind))
                == dnnl::algorithm::binary_add
                && op->has_attr(op_attr::auto_broadcast)
                && op->get_attr<std::string>(op_attr::auto_broadcast) == "none";
    };

    std::vector<std::vector<op_ptr>> op_lists;
    std::set<op_t *> visited;
    for (auto &op : sg->get_ops()) {
        if (!is_non_broadcast_add(op.get()) || visited.count(op.get()))
            continue;
        std::vector<op_ptr> list;
        list.emplace_back(op);
        visited.insert(op.get());

        op_t *cur = op.get();
        while (true) {
            auto csms = cur->get_output_value(0)->get_consumers();
            bool match = csms.size() == 1
                    && is_non_broadcast_add(&csms[0].get_op());
            if (match) {
                cur = &csms[0].get_op();
                list.emplace_back(cur->shared_from_this());
                visited.insert(cur);
            } else {
                break;
            }
        }

        // no need to fuse single binary add
        if (list.size() > 1) op_lists.emplace_back(list);
    }

    if (op_lists.empty()) return status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &list : op_lists) {
        op_ptr sum_op = std::make_shared<op_t>(op_kind::dnnl_sum);

        auto graph_in_vals = graph_t(list).get_input_values();
        auto graph_out_vals = graph_t(list).get_output_values();

        int input_idx = 0;
        for (auto &cur_op : list) {
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

                // scratchpad output is binary op's 2nd output, we skip it
                if (out_val->get_offset() != 0) continue;

                sum_op->add_output(out_val);
                out_val->set_producer(*sum_op);
            }
        }
        // insert the scratchpad output to match the schema definition
        insert_empty_scratchpad(sum_op);

        // remove original add/mul ops
        for (const auto &op : list) {
            rewriter.to_remove(op);
        }

        // add dnnl_sum to subgraph
        rewriter.to_insert(sum_op);
    }

    rewriter.run();
    return status::success;
}

status_t binary_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_binary) continue;

        bool is_bias_add = cur_op->has_attr(op_attr::is_bias_add)
                ? cur_op->get_attr<bool>(op_attr::is_bias_add)
                : false;

        // check doable
        auto src0_lt = cur_op->get_input_value(0)->get_logical_tensor();
        auto src1_lt = cur_op->get_input_value(1)->get_logical_tensor();

        bool shape_check_ok = true;
        if (is_bias_add) {
            // special check for BiasAdd
            const auto &data_format = cur_op->has_attr(op_attr::data_format)
                    ? cur_op->get_attr<std::string>(op_attr::data_format)
                    : "NCX";
            const auto channel_num
                    = logical_tensor_wrapper_t(src0_lt).get_src_c(data_format);
            shape_check_ok = channel_num == src1_lt.dims[0];
        } else {
            shape_check_ok
                    = binary_doable(ltw(src0_lt).vdims(), ltw(src1_lt).vdims());
        }

        if (!shape_check_ok) return status::invalid_shape;

        // insert unsqueeze op
        int32_t src0_ndims = src0_lt.ndims;
        int32_t src1_ndims = src1_lt.ndims;
        int32_t target_ndims = std::max(src0_ndims, src1_ndims);
        std::vector<int32_t> in_ndims {src0_ndims, src1_ndims};
        for (size_t i = 0; i < cur_op->num_inputs(); ++i) {
            if (in_ndims[i] == target_ndims) { continue; }

            std::vector<int64_t> axes(target_ndims - in_ndims[i]);
            std::iota(axes.begin(), axes.end(), 0);

            // Only for NCX format BiasAdd, we need to unsqueeze the 1D bias to
            // [1, C, 1, 1]
            const bool channel_first = is_bias_add
                    && (!cur_op->has_attr(op_attr::data_format)
                            || cur_op->get_attr<std::string>(
                                       op_attr::data_format)
                                    == "NCX");
            if (channel_first && axes.size() >= 2) {
                axes.erase(axes.begin() + 1);
                axes.emplace_back(-1);
            }

            auto unsqueeze_op = std::make_shared<op_t>(op_kind::dnnl_unsqueeze);
            unsqueeze_op->set_attr<std::vector<int64_t>>(op_attr::axes, axes);
            rewriter.insert_op_before(unsqueeze_op, cur_op, i);
        }

        // set attr
        cur_op->set_attr<bool>(op_attr::canonicalized, true);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t binary_broadcast_swap(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_binary) continue;
        const auto alg_kind = static_cast<dnnl::algorithm>(
                cur_op->get_attr<int64_t>(op_attr::alg_kind));
        if (alg_kind != dnnl::algorithm::binary_add
                && alg_kind != dnnl::algorithm::binary_mul)
            continue;

        // check doable
        auto src0_lt = cur_op->get_input_value(0)->get_logical_tensor();
        auto src1_lt = cur_op->get_input_value(1)->get_logical_tensor();

        if (logical_tensor_wrapper_t(src0_lt).nelems()
                >= logical_tensor_wrapper_t(src1_lt).nelems())
            continue;

        op_ptr binary_op = std::make_shared<op_t>(op_kind::dnnl_binary);
        binary_op->merge_attributes(cur_op->get_attributes());

        // swap src0 and src1 value
        auto src0_value = cur_op->get_input_value(0);
        auto src1_value = cur_op->get_input_value(1);
        src1_value->remove_consumer(*cur_op, 1);
        src1_value->add_consumer(*binary_op, 0);
        binary_op->add_input(src1_value);
        src0_value->remove_consumer(*cur_op, 0);
        src0_value->add_consumer(*binary_op, 1);
        binary_op->add_input(src0_value);
        // connect out0 and out1 value
        auto out0_value = cur_op->get_output_value(0);
        binary_op->add_output(out0_value);
        auto out1_value = cur_op->get_output_value(1);
        binary_op->add_output(out1_value);

        rewriter.to_insert(binary_op);
        rewriter.to_remove(cur_op);
    }

    rewriter.run();
    return status::success;
}

extern "C" dnnl_status_t dnnl_memory_desc_create_with_string_tag(
        dnnl_memory_desc_t *, int, const dnnl_dims_t, dnnl_data_type_t,
        const char *);

status_t fuse_adjacent_reorders(std::shared_ptr<subgraph_t> &sg) {
    const static std::set<op_kind_t> reorder_op_set = {op_kind::dnnl_reorder};

    auto fuse_two_adjacent_reorders = [&](bool &changed) -> status_t {
        auto &mgr = sg->fusion_info_mgr_;
        auto &p_engine = sg->p_engine_;
        auto &pd_cache = sg->pd_cache_;

        std::vector<std::pair<op_t *, op_t *>> fuse_groups;

        std::set<const op_t *> visited;
        status_t ret;
        ret = topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
            if (!reorder_op_set.count(op->get_kind()) || visited.count(op) != 0)
                return status::success;

            auto out_val = op->get_output_values()[0];
            auto consumers = out_val->get_consumers();
            if (consumers.size() != 1) return status::success;
            auto &next_op = consumers[0].get_op();

            // check if fusible
            if (reorder_op_set.count(next_op.get_kind()) == 0) {
                return status::success;
            }

            // todo: fuse reorder with runtime args
            if (op->num_inputs() > 1 || next_op.num_inputs() > 1)
                return status::success;

            // two reorders should have same shape
            auto next_op_out = next_op.get_output_value(0);
            auto lhs = out_val->get_logical_tensor();
            auto rhs = next_op_out->get_logical_tensor();
            if (ltw(lhs).vdims() != ltw(rhs).vdims()) {
                return status::success;
            }

            // if reorder do per channel scaling, their axis should be
            // same
            int64_t cur_axis = op->has_attr(op_attr::axis)
                    ? op->get_attr<int64_t>(op_attr::axis)
                    : -1;
            int64_t next_axis = next_op.has_attr(op_attr::axis)
                    ? next_op.get_attr<int64_t>(op_attr::axis)
                    : -1;
            if (cur_axis != -1 && next_axis != -1 && cur_axis != next_axis) {
                return status::success;
            }

            // Skip fusion if reorder's output has extra info, because oneDNN
            // doesn't have good supports for such fused cases. Note that since
            // onednn didn't provide api to check extra flags, here we construct
            // a temp md without extra flag, and then compare it with the origin
            // md. If they are not equal, the origin md may has extra flags.
            auto fused_out_lt
                    = next_op.get_output_value(0)->get_logical_tensor();
            auto fused_out_md = make_dnnl_memory_desc(fused_out_lt);
            auto format_tag = get_format_tag_str(fused_out_md);
            const auto &dims = fused_out_md.get_dims();
            const auto &dtype = fused_out_md.get_data_type();
            dnnl_memory_desc_t temp_md;
            dnnl_memory_desc_create_with_string_tag(&temp_md,
                    static_cast<int>(dims.size()), dims.data(),
                    static_cast<dnnl_data_type_t>(dtype), format_tag.data());
            if (!dnnl_memory_desc_equal(fused_out_md.get(), temp_md)) {
                // destroy md before return
                dnnl_memory_desc_destroy(temp_md);
                return status::success;
            }

            // push fusible pair to fuse group for later fusion
            fuse_groups.emplace_back(std::pair<op_t *, op_t *> {op, &next_op});
            visited.insert(op);
            visited.insert(&next_op);
            // destroy md before return
            dnnl_memory_desc_destroy(temp_md);
            return status::success;
        });

        if (ret != status::success) return ret;

        if (fuse_groups.empty()) {
            changed = false;
            return status::success;
        }

        subgraph_rewriter_t rewriter(sg);
        for (auto &fuse_group : fuse_groups) {
            auto op1 = fuse_group.first;
            auto op2 = fuse_group.second;

            // get the src_zps, dst_zps and scales.
            auto get_scales_zps = [](const op_t *op, std::vector<float> &scales,
                                          std::vector<int64_t> &src_zps,
                                          std::vector<int64_t> &dst_zps,
                                          size_t &num) {
                scales = op->has_attr(op_attr::scales)
                        ? op->get_attr<std::vector<float>>(op_attr::scales)
                        : std::vector<float> {1.0};
                src_zps = op->has_attr(op_attr::src_zps)
                        ? op->get_attr<std::vector<int64_t>>(op_attr::src_zps)
                        : std::vector<int64_t> {0};
                dst_zps = op->has_attr(op_attr::dst_zps)
                        ? op->get_attr<std::vector<int64_t>>(op_attr::dst_zps)
                        : std::vector<int64_t> {0};

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
            // op: dst = scales*(src-src_zps)+dst_zps;
            std::vector<float> fused_scales;
            std::vector<int64_t> fused_src_zps, fused_dst_zps;
            fused_src_zps = src_zps1;
            fused_scales.reserve(max_num);
            fused_dst_zps.reserve(max_num);
            for (size_t i = 0; i < max_num; i++) {
                fused_scales.emplace_back(scales1[i] * scales2[i]);
                fused_dst_zps.emplace_back(
                        scales2[i] * (dst_zps1[i] - src_zps2[i]) + dst_zps2[i]);
            }

            int64_t axis = -1;
            if (op1->has_attr(op_attr::axis)) {
                axis = op1->get_attr<int64_t>(op_attr::axis);
            }
            if (op2->has_attr(op_attr::axis)) {
                axis = op2->get_attr<int64_t>(op_attr::axis);
            }

            bool change_layout
                    = (op1->has_attr(op_attr::change_layout)
                              && op1->get_attr<bool>(op_attr::change_layout))
                    || (op2->has_attr(op_attr::change_layout)
                            && op2->get_attr<bool>(op_attr::change_layout));

            // create fused op
            op_ptr fused_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
            fused_op->set_attr<bool>(op_attr::change_layout, change_layout);
            if (axis != -1) fused_op->set_attr<int64_t>(op_attr::axis, axis);

            if (!std::all_of(fused_scales.begin(), fused_scales.end(),
                        [](const float &s) { return s == 1.f; })) {
                fused_op->set_attr<std::vector<float>>(
                        op_attr::scales, fused_scales);
            }

            if (std::find_if(fused_src_zps.begin(), fused_src_zps.end(),
                        [](const int64_t &zp) { return zp != 0; })
                    != fused_src_zps.end()) {
                fused_op->set_attr<std::vector<int64_t>>(
                        op_attr::src_zps, fused_src_zps);
            }

            if (std::find_if(fused_dst_zps.begin(), fused_dst_zps.end(),
                        [](const int64_t &zp) { return zp != 0; })
                    != fused_dst_zps.end()) {
                fused_op->set_attr<std::vector<int64_t>>(
                        op_attr::dst_zps, fused_dst_zps);
            }

            // connect fused op to subgraph and remove original ones
            auto in_val = op1->get_input_value(0);
            in_val->remove_consumer(*op1, 0);
            fused_op->connect_input(0, in_val);

            auto out_val = op2->get_output_value(0);
            fused_op->add_output(out_val);
            out_val->set_producer(*fused_op);

            auto scratchpad_val = insert_empty_scratchpad(fused_op);
            // remove pd in pd_cache since fused_op share the same id
            if (pd_cache.find(fused_op.get()) != pd_cache.end()) {
                pd_cache.erase(fused_op.get());
            }
            const auto &pd = reorder_executable_t::create_desc(
                    fused_op, *p_engine, mgr, pd_cache);
            const memory::desc scratchpad_desc = pd.scratchpad_desc();
            auto status = fill_layout_info(scratchpad_val, scratchpad_desc);
            if (status != status::success) return status;

            rewriter.to_insert(fused_op);
            rewriter.to_remove(op1->shared_from_this());
            rewriter.to_remove(op2->shared_from_this());
        }
        rewriter.run();
        return status::success;
    };

    int cnt = 0;
    const int max_num_limit = static_cast<int>(sg->num_ops());

    bool changed = true;
    do {
        auto ret = fuse_two_adjacent_reorders(changed);
        if (ret != status::success) return ret;
        cnt++;
    } while (changed && cnt <= max_num_limit);

    assertm(cnt <= max_num_limit + 1, "reorder fusion failed.");
    if (cnt > max_num_limit + 1) return status::unimplemented;

    return status::success;
}

status_t fuse_typecast_to_mul_scales(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::vector<op_t *>> fusion_groups;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_mul_scales
                || !cur_op->get_input_value(0)->has_producer())
            continue;

        auto &in = cur_op->get_input_value(0)->get_producer();
        if (is_typecast(&in))
            fusion_groups.emplace_back(std::vector<op_t *> {cur_op.get(), &in});
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &fusion_group : fusion_groups) {
        op_t *in0 = fusion_group[1];
        rewriter.fuse_op_to_successor(in0->shared_from_this());
    }
    rewriter.run();
    return status::success;
}

status_t convert_runtime_mul_scales(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_t *> mul_scales;
    std::set<op_t *> visited;
    for (const auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_mul_scales)
                || visited.count(cur_op.get()) != 0)
            continue;

        // This pass only handle static quantization
        bool dync_quantization = cur_op->has_attr(op_attr::with_runtime_scales)
                && cur_op->get_attr<bool>(op_attr::with_runtime_scales);
        if (dync_quantization) continue;

        mul_scales.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &mul_scale : mul_scales) {
        // make scales as a constant input
        op_ptr const_data_op;
        const auto scales
                = mul_scale->get_attr<std::vector<float>>(op_attr::scales);
        const_data_op = std::make_shared<op_t>(op_kind::dnnl_constant_scales);
        const_data_op->set_attr(op_attr::scales, scales);
        std::vector<int64_t> dst_shape(1, scales.size());
        const_data_op->set_attr(op_attr::shape, dst_shape);
        logical_tensor_t const_data_dst_lt
                = empty_logical_tensor_with_default_id();
        auto const_data_dst_value = std::make_shared<value_t>(
                *const_data_op, 0, const_data_dst_lt, true);
        const_data_dst_value->set_data_type(graph::data_type::f32);
        const_data_dst_value->set_layout_type(layout_type::strided);
        const_data_dst_value->set_strides({1});
        const_data_op->add_output(const_data_dst_value);
        mul_scale->set_attr(op_attr::with_runtime_scales, true);
        mul_scale->remove_attr(op_attr::scales);

        // connect mul_scale and constant data
        mul_scale->connect_input(1, const_data_dst_value);
        rewriter.to_insert(const_data_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t convert_runtime_zero_points(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_t *> zps;
    std::set<op_t *> visited;
    for (const auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_sub_zps
                    && cur_op->get_kind() != op_kind::dnnl_add_zps)
                || visited.count(cur_op.get()) != 0)
            continue;

        // This pass only handle static quantization
        bool dync_quantization = cur_op->has_attr(op_attr::with_runtime_zps)
                && cur_op->get_attr<bool>(op_attr::with_runtime_zps);
        if (dync_quantization) continue;

        zps.emplace_back(cur_op.get());
        visited.insert(cur_op.get());
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &zp_op : zps) {
        // make zps as a constant input
        op_ptr const_data_op;
        auto zps = zp_op->get_attr<std::vector<int64_t>>(op_attr::zps);
        // adjusted zp
        std::vector<int64_t> adj_zps = {zps[0]};
        const_data_op = std::make_shared<op_t>(op_kind::dnnl_constant_zps);
        const_data_op->set_attr(op_attr::zps, adj_zps);
        std::vector<int64_t> dst_shape(1, adj_zps.size());
        const_data_op->set_attr(op_attr::shape, dst_shape);
        logical_tensor_t const_data_dst_lt
                = empty_logical_tensor_with_default_id();
        auto const_data_dst_value = std::make_shared<value_t>(
                *const_data_op, 0, const_data_dst_lt, true);
        const_data_dst_value->set_data_type(graph::data_type::s32);
        const_data_dst_value->set_layout_type(layout_type::strided);
        const_data_dst_value->set_strides({1});
        const_data_op->add_output(const_data_dst_value);
        zp_op->set_attr(op_attr::with_runtime_zps, true);
        zp_op->remove_attr(op_attr::zps);

        // connect zp and constant data
        zp_op->connect_input(1, const_data_dst_value);
        rewriter.to_insert(const_data_op);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t fuse_dynamic_mul_scales_add_zps(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::pair<op_ptr, op_ptr>> fuse_groups;
    std::set<op_t *> visited;
    for (const auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_mul_scales)
                || visited.count(cur_op.get()) != 0)
            continue;

        // This pass only handle dynamic quantization
        if (!cur_op->has_attr(op_attr::with_runtime_scales)
                || !cur_op->get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        auto out_val = cur_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;

        auto &consumer_op = consumers[0].get_op();
        if (consumer_op.get_kind() != op_kind::dnnl_add_zps) continue;

        if (!consumer_op.has_attr(op_attr::with_runtime_zps)
                || !consumer_op.get_attr<bool>(op_attr::with_runtime_zps))
            continue;

        fuse_groups.emplace_back(std::pair<op_ptr, op_ptr> {
                cur_op, (&consumer_op)->shared_from_this()});
        visited.insert(cur_op.get());
        visited.insert(&consumer_op);
    }

    if (fuse_groups.empty()) return status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &fuse_ops : fuse_groups) {
        op_ptr &mul_scales = fuse_ops.first; // mul_scales
        op_ptr &add_zps = fuse_ops.second; // add_zps

        const int64_t axis = mul_scales->get_attr<int64_t>(op_attr::axis);
        const std::string &qtype
                = mul_scales->get_attr<std::string>(op_attr::qtype);

        op_ptr fused_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        fused_op->set_attr<bool>(op_attr::change_layout, false);
        fused_op->set_attr<int64_t>(op_attr::axis, axis);
        fused_op->set_attr<std::string>(op_attr::qtype, qtype);

        // src must be the 0-th input
        auto src = mul_scales->get_input_value(0);
        src->remove_consumer(*mul_scales, 0);
        fused_op->connect_input(0, src);

        // fuse scales as arg src scales
        auto scales = mul_scales->get_input_value(1);
        scales->remove_consumer(*mul_scales, 1);
        fused_op->connect_input(1, scales);
        fused_op->set_attr<bool>(op_attr::with_runtime_scales, true);

        // fuse dst zps
        auto zps = add_zps->get_input_value(1);
        zps->remove_consumer(*add_zps, 1);
        fused_op->connect_input(2, zps);
        fused_op->set_attr<bool>(op_attr::with_runtime_dst_zps, true);

        auto dst = add_zps->get_output_value(0);
        fused_op->add_output(dst);
        dst->set_producer(*fused_op);

        insert_empty_scratchpad(fused_op);

        rewriter.to_insert(fused_op);
        rewriter.to_remove(mul_scales);
        rewriter.to_remove(add_zps);
    }

    rewriter.run();
    return status::success;
}

status_t fuse_dynamic_sub_zps_mul_scales(std::shared_ptr<subgraph_t> &sg) {
    std::vector<std::pair<op_ptr, op_ptr>> fuse_groups;
    std::set<op_t *> visited;
    for (const auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_sub_zps)
                || visited.count(cur_op.get()) != 0)
            continue;

        // This pass only handle dynamic dequantization
        if (!cur_op->has_attr(op_attr::with_runtime_zps)
                || !cur_op->get_attr<bool>(op_attr::with_runtime_zps))
            continue;

        auto out_val = cur_op->get_output_values()[0];
        auto consumers = out_val->get_consumers();
        if (consumers.empty()) continue;

        auto &consumer_op = consumers[0].get_op();
        if (consumer_op.get_kind() != op_kind::dnnl_mul_scales) continue;
        if (!consumer_op.has_attr(op_attr::with_runtime_scales)
                || !consumer_op.get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        fuse_groups.emplace_back(std::pair<op_ptr, op_ptr> {
                cur_op, (&consumer_op)->shared_from_this()});
        visited.insert(cur_op.get());
        visited.insert(&consumer_op);
    }

    if (fuse_groups.empty()) return status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &fuse_ops : fuse_groups) {
        op_ptr &op1 = fuse_ops.first; // sub_zps
        op_ptr &op2 = fuse_ops.second; // mul_scales

        const int64_t axis = op1->get_attr<int64_t>(op_attr::axis);
        const std::string &qtype = op1->get_attr<std::string>(op_attr::qtype);

        op_ptr fused_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        fused_op->set_attr<bool>(op_attr::change_layout, false);
        fused_op->set_attr<int64_t>(op_attr::axis, axis);
        fused_op->set_attr<std::string>(op_attr::qtype, qtype);

        // src must be the 0-th input
        auto src = op1->get_input_value(0);
        src->remove_consumer(*op1, 0);
        fused_op->connect_input(0, src);

        // fuse scales as output scales
        auto scales = op2->get_input_value(1);
        scales->remove_consumer(*op2, 1);
        fused_op->connect_input(1, scales);
        fused_op->set_attr<bool>(op_attr::with_runtime_scales, true);

        // fuse src zps
        auto zps = op1->get_input_value(1);
        zps->remove_consumer(*op1, 1);
        fused_op->connect_input(2, zps);
        fused_op->set_attr<bool>(op_attr::with_runtime_src_zps, true);

        auto dst = op2->get_output_value(0);
        fused_op->add_output(dst);
        dst->set_producer(*fused_op);

        insert_empty_scratchpad(fused_op);

        rewriter.to_insert(fused_op);
        rewriter.to_remove(op1);
        rewriter.to_remove(op2);
    }

    rewriter.run();
    return status::success;
}

impl::status_t convert_dynamic_quantize_ops(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> convert_ops;
    std::set<op_t *> visited;
    for (const auto &cur_op : sg->get_ops()) {
        if ((cur_op->get_kind() != op_kind::dnnl_mul_scales
                    && cur_op->get_kind() != op_kind::dnnl_add_zps
                    && cur_op->get_kind() != op_kind::dnnl_sub_zps)
                || visited.count(cur_op.get()) != 0)
            continue;

        // This pass only handle single dynamic sub_scale,add_zp,sub_zp
        if (cur_op->get_kind() == op_kind::dnnl_mul_scales) {
            if (!cur_op->has_attr(op_attr::with_runtime_scales)
                    || !cur_op->get_attr<bool>(op_attr::with_runtime_scales))
                continue;
        } else {
            if (!cur_op->has_attr(op_attr::with_runtime_zps)
                    || !cur_op->get_attr<bool>(op_attr::with_runtime_zps))
                continue;
        }

        convert_ops.emplace_back(cur_op);
        visited.insert(cur_op.get());
    }

    if (convert_ops.empty()) return impl::status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &cur_op : convert_ops) {
        const int64_t axis = cur_op->get_attr<int64_t>(op_attr::axis);
        const std::string &qtype
                = cur_op->get_attr<std::string>(op_attr::qtype);

        op_ptr fused_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
        fused_op->set_attr<bool>(op_attr::change_layout, false);
        fused_op->set_attr<int64_t>(op_attr::axis, axis);
        fused_op->set_attr<std::string>(op_attr::qtype, qtype);

        // src must be the 0-th input
        auto src = cur_op->get_input_value(0);
        src->remove_consumer(*cur_op, 0);
        fused_op->connect_input(0, src);

        auto another_src = cur_op->get_input_value(1);
        another_src->remove_consumer(*cur_op, 1);
        fused_op->connect_input(1, another_src);
        if (cur_op->get_kind() == op_kind::dnnl_mul_scales) {
            // fuse scales as arg src scales
            fused_op->set_attr<bool>(op_attr::with_runtime_scales, true);
        } else if (cur_op->get_kind() == op_kind::dnnl_add_zps) {
            // fuse dst zps
            fused_op->set_attr<bool>(op_attr::with_runtime_dst_zps, true);
        } else {
            // fuse src zps
            fused_op->set_attr<bool>(op_attr::with_runtime_src_zps, true);
        }

        auto dst = cur_op->get_output_value(0);
        fused_op->add_output(dst);
        dst->set_producer(*fused_op);

        insert_empty_scratchpad(fused_op);

        rewriter.to_insert(fused_op);
        rewriter.to_remove(cur_op);
    }

    rewriter.run();
    return impl::status::success;
}

status_t reorder_canonicalization(std::shared_ptr<subgraph_t> &sg) {
    subgraph_rewriter_t rewriter(sg);

    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_reorder) continue;

        size_t index = 1; // the start index of optional runtime scales and zps

        // optionally skip the runtime scales
        if (cur_op->has_attr(op_attr::with_runtime_scales)
                && cur_op->get_attr<bool>(op_attr::with_runtime_scales)) {
            index++;
        }

        // check runtime src_zps and add typecast if necessary
        if (cur_op->has_attr(op_attr::with_runtime_src_zps)
                && cur_op->get_attr<bool>(op_attr::with_runtime_src_zps)) {
            auto src_zps = cur_op->get_input_value(index);
            if (src_zps->get_logical_tensor().data_type
                    != graph::data_type::s32) {
                auto tc_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
                tc_op->set_attr<bool>(op_attr::change_layout, false);
                rewriter.insert_op_before(tc_op, cur_op, index);
                insert_empty_scratchpad(tc_op);
                tc_op->get_output_value(0)->set_data_type(
                        graph::data_type::s32);
                index++;
            }
        }

        // check runtime dst_zps and add typecast if necessary
        if (cur_op->has_attr(op_attr::with_runtime_dst_zps)
                && cur_op->get_attr<bool>(op_attr::with_runtime_dst_zps)) {
            auto dst_zps = cur_op->get_input_value(index);
            if (dst_zps->get_logical_tensor().data_type
                    != graph::data_type::s32) {
                auto tc_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
                tc_op->set_attr<bool>(op_attr::change_layout, false);
                rewriter.insert_op_before(tc_op, cur_op, index);
                tc_op->get_output_value(0)->set_data_type(
                        graph::data_type::s32);
                index++;
            }
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t common_reorder_elimination(std::shared_ptr<subgraph_t> &sg) {
    // Eliminate two equal reorder with same input
    auto cse_func = [&](bool &changed) {
        std::vector<op_t *> fusion;

        // find two fusible ops
        for (auto &op : sg->get_ops()) {
            if (op->get_kind() != graph::op_kind::Reorder) continue;

            auto ins = op->get_input_values();
            // equal op must share same inputs, so it's enough to only look up
            // ins[0]'s consumers.
            auto csms = ins[0]->get_consumers();
            bool found_equal_op = false;
            for (auto csm : csms) {
                auto &csm_op = csm.get_op();
                // the same op, skip
                if (&csm_op == op.get()) continue;

                // not equal op, skip
                bool equal_op = op->get_kind() == csm_op.get_kind()
                        && op->has_same_attr_values(csm_op);
                if (!equal_op) continue;

                // not same inputs, skip
                auto csm_ins = csm_op.get_input_values();
                if (csm_ins.size() != ins.size()) continue;
                size_t i;
                for (i = 0; i < csm_ins.size(); i++) {
                    if (csm_ins[i].get() != ins[i].get()) break;
                }
                if (i < csm_ins.size()) continue;

                // not equal outputs, skip
                auto &outs = op->get_output_values();
                auto &csm_outs = csm_op.get_output_values();
                if (csm_outs.size() != outs.size()) continue;
                for (i = 0; i < csm_outs.size(); i++) {
                    auto lt1 = csm_outs[i]->get_logical_tensor();
                    auto lt2 = outs[i]->get_logical_tensor();
                    if (make_dnnl_memory_desc(lt1)
                            != make_dnnl_memory_desc(lt2))
                        break;
                }
                if (i < csm_outs.size()) continue;

                // all condition matched
                fusion.emplace_back(op.get());
                fusion.emplace_back(&csm_op);
                found_equal_op = true;
                break;
            }
            if (found_equal_op) break;
        }

        if (fusion.empty()) {
            changed = false;
            return status::success;
        }

        // remove op2 and add it's consumers to op1
        auto op1 = fusion[0], op2 = fusion[1];
        auto op2_ins = op2->get_input_values();
        for (size_t i = 0; i < op2_ins.size(); i++) {
            op2_ins[i]->remove_consumer(*op2, i);
        }

        auto op1_outs = op1->get_output_values();
        auto op2_outs = op2->get_output_values();
        for (size_t i = 0; i < op2_outs.size(); i++) {
            auto &csms = op2_outs[i]->get_consumers();
            for (auto &csm : csms) {
                op1_outs[i]->add_consumer(csm.get_op(), csm.get_offset());
                csm.get_op().connect_input(csm.get_offset(), op1_outs[i]);
            }
        }

        subgraph_rewriter_t rewriter(sg);
        rewriter.to_remove(op2->shared_from_this());
        rewriter.run();

        changed = true;
        return status::success;
    };

    int cnt = 0;
    const int max_iter_num = static_cast<int>(sg->num_ops());

    bool changed = true;
    do {
        auto ret = cse_func(changed);
        if (ret != status::success) return ret;
        cnt++;
    } while (changed && cnt <= max_iter_num);

    assertm(cnt <= max_iter_num + 1,
            "Failed to eliminate common reorders since the pass can't "
            "converge.");
    if (cnt > max_iter_num + 1) return status::unimplemented;

    return status::success;
}

// combine scales around binary post op
//
//         |                     |        |
//      base_op               base_op    zps1
//         |         |           |        |
//       zps0       zps1        zps0    new_scales1 (optional)
//         |         |            \      /
//      scales0   scales1          binary
//           \      /                 |
//            binary      =>     new_scales2
//               |                    |
//            scales2                zps2
//               |                    |
//              zps2
//               |
//
//    where base_op is one of [pool]
//
//    binary-add case:
//    - new_scales1 = scales1 / scales0
//    - new_scales2 = scales0 * scales2
//    binary-mul case:
//    - new_scales1 will be removed
//    - new_scales2 = scales0 * scales1 * scales2
status_t combine_binary_post_op_scales(std::shared_ptr<subgraph_t> &sg) {
    const auto fuse_scales
            = [](const std::vector<float> &scales0,
                      const std::vector<float> &scales1,
                      const std::function<float(float, float)> &operation)
            -> std::vector<float> {
        std::vector<float> fused_scales(
                std::max(scales0.size(), scales1.size()), 1.f);
        if (scales0.size() >= scales1.size()) {
            for (size_t i = 0; i < scales0.size(); ++i) {
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
            if (scale_ops[i]->get_attr<std::string>(op_attr::qtype)
                    == "per_channel") {
                // assumption: at least one scales per channel will make
                // combined scales per channel
                return std::make_pair("per_channel",
                        scale_ops[i]->get_attr<int64_t>(op_attr::axis));
            }
        }
        // scales per tensor, defaulting axis
        return std::make_pair("per_tensor", static_cast<int64_t>(1));
    };

    std::vector<op_ptr> bin_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_binary) {
            value_ptr bin_in0_val = cur_op->get_input_value(0);
            value_ptr bin_in1_val = cur_op->get_input_value(1);
            value_ptr bin_out_val = cur_op->get_output_value(0);
            if (!bin_in0_val->has_producer() || !bin_in1_val->has_producer()
                    || bin_out_val->get_consumers().empty())
                continue;

            if (bin_in0_val->get_producer().get_kind()
                            != op_kind::dnnl_mul_scales
                    || bin_in1_val->get_producer().get_kind()
                            != op_kind::dnnl_mul_scales
                    || bin_out_val->get_consumers()[0].get_op().get_kind()
                            != op_kind::dnnl_mul_scales)
                continue;

            bin_ops.emplace_back(cur_op);
        }
    }

    if (bin_ops.empty()) return status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &bin_op : bin_ops) {
        value_ptr bin_in0_val = bin_op->get_input_value(0);
        value_ptr bin_in1_val = bin_op->get_input_value(1);
        value_ptr bin_out_val = bin_op->get_output_value(0);
        if (!bin_in0_val->has_producer() || !bin_in1_val->has_producer()
                || bin_out_val->get_consumers().empty())
            continue;

        op_t &scales_in0_op = bin_in0_val->get_producer();
        assertm(scales_in0_op.get_kind() == op_kind::dnnl_mul_scales,
                "the first predecessor of a binary op should be mul_scales.");
        if (scales_in0_op.has_attr(op_attr::with_runtime_scales)
                && scales_in0_op.get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        op_t &scales_in1_op = bin_in1_val->get_producer();
        assertm(scales_in1_op.get_kind() == op_kind::dnnl_mul_scales,
                "the second predecessor of a binary op should be mul_scales.");
        if (scales_in1_op.has_attr(op_attr::with_runtime_scales)
                && scales_in1_op.get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        op_t &scales_out_op = bin_out_val->get_consumers()[0].get_op();
        assertm(scales_out_op.get_kind() == op_kind::dnnl_mul_scales,
                "the successor of a binary op should be mul_scales.");
        if (scales_out_op.has_attr(op_attr::with_runtime_scales)
                && scales_out_op.get_attr<bool>(op_attr::with_runtime_scales))
            continue;

        const size_t base_op_branch_idx = [&scales_in0_op]() {
            op_t &zps_op = scales_in0_op.get_input_value(0)->get_producer();
            if (zps_op.get_input_value(0)->has_producer()) {
                const auto zps_predecessor_kind
                        = zps_op.get_input_value(0)->get_producer().get_kind();
                if (zps_predecessor_kind == op_kind::dnnl_eltwise
                        || zps_predecessor_kind == op_kind::dnnl_pool) {
                    return 0;
                }
            }
            return 1;
        }();
        op_t &base_scales_op
                = (base_op_branch_idx) ? scales_in1_op : scales_in0_op;
        op_t &other_scales_op
                = (base_op_branch_idx) ? scales_in0_op : scales_in1_op;

        const auto in0_scales
                = base_scales_op.get_attr<std::vector<float>>(op_attr::scales);
        const auto in1_scales
                = other_scales_op.get_attr<std::vector<float>>(op_attr::scales);
        const auto inv_out_scales
                = scales_out_op.get_attr<std::vector<float>>(op_attr::scales);
        const auto bin_kind = static_cast<dnnl::algorithm>(
                bin_op->get_attr<int64_t>(op_attr::alg_kind));

        std::vector<float> new_scales_in1, new_scales_in0;
        std::string new_qtype_in1;
        std::string new_qtype_in0;
        int64_t new_axis_in1 = 0;
        int64_t new_axis_in0 = 0;
        bool drop_other_scales = false;
        const auto multiplier = std::multiplies<float>();
        switch (bin_kind) {
            case dnnl::algorithm::binary_add:
                assertm(std::all_of(in0_scales.begin(), in0_scales.end(),
                                [](float v) { return v != 0.f; }),
                        "scales can't be zero");
                new_scales_in0
                        = fuse_scales(in0_scales, inv_out_scales, multiplier);
                new_scales_in1
                        = fuse_scales(in1_scales, inv_out_scales, multiplier);
                std::tie(new_qtype_in1, new_axis_in1) = fuse_scales_attributes(
                        {&scales_in1_op, &scales_out_op});
                std::tie(new_qtype_in0, new_axis_in0) = fuse_scales_attributes(
                        {&scales_in0_op, &scales_out_op});
                break;
            case dnnl::algorithm::binary_mul:
                drop_other_scales = true;
                new_scales_in0
                        = fuse_scales(in0_scales, in1_scales, multiplier);
                new_scales_in0 = fuse_scales(
                        new_scales_in0, inv_out_scales, multiplier);
                std::tie(new_qtype_in0, new_axis_in0) = fuse_scales_attributes(
                        {&scales_in0_op, &scales_in1_op, &scales_out_op});
                break;
            default:
                assertm(false, "unsupported binary post-op was provided.");
                break;
        }

        // drop out scales op
        rewriter.fuse_op_to_predecessor(scales_out_op.shared_from_this());

        // if possible, drop other scales, and connect zps with binary
        // otherwise, update other scales data
        if (drop_other_scales) {
            rewriter.fuse_op_to_successor(other_scales_op.shared_from_this());
        } else {
            other_scales_op.set_attr(op_attr::scales, new_scales_in1)
                    .set_attr(op_attr::qtype, new_qtype_in1)
                    .set_attr(op_attr::axis, new_axis_in1);
        }

        // update output scales data
        base_scales_op.set_attr(op_attr::scales, new_scales_in0)
                .set_attr(op_attr::qtype, new_qtype_in0)
                .set_attr(op_attr::axis, new_axis_in0);
    }

    rewriter.run();
    return infer_shape(sg);
}

status_t remove_quant_data_with_no_effect(std::shared_ptr<subgraph_t> &sg) {
    auto is_dequantize = [](const op_ptr &op) {
        value_ptr quant_data_out_val = op->get_output_value(0);
        value_ptr quant_data_in_val = op->get_input_value(0);
        return op->get_kind() == op_kind::dnnl_sub_zps
                || (op->get_kind() == op_kind::dnnl_mul_scales
                        && quant_data_in_val->get_logical_tensor().data_type
                                != quant_data_out_val->get_logical_tensor()
                                           .data_type);
    };
    std::vector<op_ptr> quant_data_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_mul_scales
                || cur_op->get_kind() == op_kind::dnnl_add_zps
                || cur_op->get_kind() == op_kind::dnnl_sub_zps) {
            bool dync_quantization
                    = cur_op->has_attr(op_attr::with_runtime_scales)
                    && cur_op->get_attr<bool>(op_attr::with_runtime_scales);
            if (dync_quantization) continue;
            dync_quantization = cur_op->has_attr(op_attr::with_runtime_zps)
                    && cur_op->get_attr<bool>(op_attr::with_runtime_zps);
            if (dync_quantization) continue;
            quant_data_ops.emplace_back(cur_op);
        }
    }

    if (quant_data_ops.empty()) return status::success;

    subgraph_rewriter_t rewriter(sg);
    for (auto &quant_data_op : quant_data_ops) {
        bool to_remove = false;
        if (quant_data_op->get_kind() == op_kind::dnnl_mul_scales) {
            const auto scales = quant_data_op->get_attr<std::vector<float>>(
                    op_attr::scales);
            to_remove = std::all_of(scales.begin(), scales.end(), [](float s) {
                float expected = 1.0f;
                float eps = 0.000001f;
                return std::abs(s - expected) <= eps;
            });
        } else {
            const auto zps = quant_data_op->get_attr<std::vector<int64_t>>(
                    op_attr::zps);
            to_remove = std::all_of(
                    zps.begin(), zps.end(), [](int64_t z) { return z == 0; });
        }

        if (to_remove) {
            // for dnnl_mul_scales out_value dataType is f32
            // for dnnl_add_zps out_value dataType is s8/u8
            // post logical tensor should be retained
            value_ptr quant_data_out_val = quant_data_op->get_output_value(0);
            value_ptr quant_data_in_val = quant_data_op->get_input_value(0);
            if (is_dequantize(quant_data_op)) {
                if (!quant_data_out_val->get_consumers().empty())
                    rewriter.fuse_op_to_successor(quant_data_op);
                else if (quant_data_in_val->has_producer()) {
                    quant_data_in_val->get_producer().connect_output(
                            quant_data_in_val->get_offset(),
                            quant_data_out_val);
                    rewriter.to_remove(quant_data_op);
                } else {
                    op_ptr tc_op
                            = std::make_shared<op_t>(op_kind::dnnl_reorder);
                    rewriter.replace_op(quant_data_op, tc_op);
                }
            } else {
                if (quant_data_in_val->has_producer()) {
                    quant_data_in_val->get_producer().connect_output(
                            quant_data_in_val->get_offset(),
                            quant_data_out_val);
                    rewriter.to_remove(quant_data_op);
                } else {
                    if (quant_data_op->get_kind() == op_kind::dnnl_mul_scales) {
                        rewriter.fuse_op_to_successor(quant_data_op);
                    } else {
                        op_ptr tc_op
                                = std::make_shared<op_t>(op_kind::dnnl_reorder);
                        rewriter.replace_op(quant_data_op, tc_op);
                    }
                }
            }
        }
    }

    rewriter.run();
    return status::success;
}

impl::status_t lift_up_typecast(std::shared_ptr<subgraph_t> &sg) {
    while (true) {
        std::vector<std::pair<op_t *, op_t *>> to_be_swapped;
        for (auto &op : sg->get_ops()) {
            bool ok = is_typecast(op.get())
                    && op->get_input_value(0)->has_producer();
            if (!ok) continue;

            op_t *producer = op->get_input_op(0);
            ok = producer->get_kind() == op_kind::dnnl_reshape
                    || producer->get_kind() == op_kind::dnnl_transpose
                    || is_layout_reorder(producer);
            if (!ok) continue;

            to_be_swapped.emplace_back(
                    std::pair<op_t *, op_t *> {producer, op.get()});
        }

        if (to_be_swapped.empty()) break;
        subgraph_rewriter_t rewriter(sg);
        for (auto &pair : to_be_swapped) {
            op_t *producer = pair.first;
            op_t *tc = pair.second;

            rewriter.swap_neighboring_si_ops(
                    producer->shared_from_this(), tc->shared_from_this());
        }
    }
    return infer_shape(sg);
}

impl::status_t lift_up_quantize(std::shared_ptr<subgraph_t> &sg) {
    while (true) {
        std::vector<std::pair<op_t *, op_t *>> to_be_swapped;
        for (auto &op : sg->get_ops()) {
            bool ok = impl::utils::one_of(op->get_kind(),
                              op_kind::dnnl_mul_scales, op_kind::dnnl_add_zps)
                    && op->get_input_value(0)->has_producer();
            if (!ok) continue;

            ok = op->has_attr(op_attr::qtype)
                    && op->get_attr<std::string>(op_attr::qtype)
                            == "per_tensor";
            if (!ok) continue;

            op_t *producer = op->get_input_op(0);
            ok = producer->get_kind() == op_kind::dnnl_reshape
                    || producer->get_kind() == op_kind::dnnl_transpose
                    || is_layout_reorder(producer);
            if (!ok) continue;

            to_be_swapped.emplace_back(
                    std::pair<op_t *, op_t *> {producer, op.get()});
        }

        if (to_be_swapped.empty()) break;
        subgraph_rewriter_t rewriter(sg);
        for (auto &pair : to_be_swapped) {
            op_t *producer = pair.first;
            op_t *quant = pair.second;

            rewriter.swap_neighboring_si_ops(
                    producer->shared_from_this(), quant->shared_from_this());
        }
    }
    return infer_shape(sg);
}

impl::status_t lift_up_weight_reshape_for_depthwiseconv(
        std::shared_ptr<subgraph_t> &sg) {
    std::unordered_map<op_t *, std::vector<op_t *>> to_be_swapped;
    for (auto &op : sg->get_ops()) {
        if (op->get_kind() != op_kind::dnnl_convolution) continue;

        // check the current op is depthwiseconv
        const auto groups = op->get_attr<int64_t>(op_attr::groups);
        const size_t wei_offset = 1;
        const auto wei_dims
                = ltw(op->get_input_value(wei_offset)->get_logical_tensor())
                          .vdims();
        const auto wei_format = (op->has_attr(op_attr::weights_format))
                ? op->get_attr<std::string>(op_attr::weights_format)
                : "XIO";
        const int64_t ndims = wei_dims.size();
        const int64_t outchannel
                = (wei_format == "OIX") ? wei_dims[0] : wei_dims[ndims - 1];
        const int64_t inputchannel
                = (wei_format == "OIX") ? wei_dims[1] : wei_dims[ndims - 2];

        if (groups == 0 || outchannel % groups != 0 || inputchannel != 1)
            continue;

        if (!op->get_input_value(1)->has_producer()) break;
        op_t *reshape_op = op->get_input_op(1);
        if (reshape_op->get_kind() != op_kind::dnnl_reshape) continue;
        op_t *producer = reshape_op;
        while (true) {
            if (!producer->get_input_value(0)->has_producer()) break;
            producer = producer->get_input_op(0);
            if (!impl::utils::one_of(producer->get_kind(),
                        op_kind::dnnl_add_zps, op_kind::dnnl_sub_zps,
                        op_kind::dnnl_mul_scales))
                break;
            if (wei_format == "XIO") {
                producer->set_attr<int64_t>(op_attr::axis, ndims - 1);
            }
            if (to_be_swapped.count(reshape_op))
                to_be_swapped[reshape_op].emplace_back(producer);
            else
                to_be_swapped[reshape_op] = {producer};
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &pair : to_be_swapped) {
        op_t *baseop = pair.first;
        for (auto swaped : pair.second)
            rewriter.swap_neighboring_reshape_ops(
                    swaped->shared_from_this(), baseop->shared_from_this());
    }

    return infer_shape(sg);
}

impl::status_t fuse_dst_transpose_to_matmul(std::shared_ptr<subgraph_t> &sg) {
    std::vector<op_ptr> transpose_ops;
    for (auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() == op_kind::dnnl_transpose
                && cur_op->get_input_value(0)->has_producer()
                && cur_op->get_input_value(0)->get_producer().get_kind()
                        == op_kind::dnnl_matmul
                && !cur_op->get_output_value(0)->get_consumers().empty()
                && (cur_op->get_output_value(0)
                                        ->get_consumers()[0]
                                        .get_op()
                                        .get_kind()
                                == op_kind::dnnl_reshape
                        || is_layout_reorder(&cur_op->get_output_value(0)
                                                      ->get_consumers()[0]
                                                      .get_op()))) {
            transpose_ops.emplace_back(cur_op);
        }
    }

    subgraph_rewriter_t rewriter(sg);
    for (auto &transpose_op : transpose_ops) {
        value_ptr in_val = transpose_op->get_input_value(0);
        auto in_lt = in_val->get_logical_tensor();
        value_ptr out_val = transpose_op->get_output_value(0);
        auto out_lt = out_val->get_logical_tensor();
        std::vector<int64_t> order
                = transpose_op->get_attr<std::vector<int64_t>>(op_attr::order);
        // if order < 0, convert it to positive order
        if (!order.empty()) {
            for (int64_t &axis : order) {
                if (axis < 0) axis += ltw(in_lt).ndims();
            }
        } else {
            return impl::status::success;
        }

        std::vector<int> axes = dnnl_impl::utils::fmap(order,
                [](int64_t index) { return static_cast<int32_t>(index); });
        // calculate the expected transposed layout by permuting the md
        auto expected_stride = get_dense_strides(ltw(out_lt).vdims());
        auto &consumer = transpose_op->get_output_value(0)
                                 ->get_consumers()[0]
                                 .get_op();
        if (is_layout_reorder(&consumer)) {
            value_ptr reorder_out_val = consumer.get_output_value(0);
            if (ltw(reorder_out_val->get_logical_tensor()).layout_type()
                    == layout_type::strided) {
                expected_stride
                        = ltw(reorder_out_val->get_logical_tensor()).vstrides();
                rewriter.fuse_op_to_predecessor(consumer.shared_from_this());
            }
        }
        dnnl::memory::desc out_md {ltw(out_lt).vdims(),
                static_cast<dnnl::memory::data_type>(ltw(out_lt).data_type()),
                expected_stride};
        dnnl::memory::desc expected_out_md = out_md.permute_axes(axes);
        const auto &strides = expected_out_md.get_strides();
        in_val->set_strides(strides);
        auto &matmul = transpose_op->get_input_value(0)->get_producer();
        matmul.set_attr(op_attr::keep_dst_layout, true);
    }

    return impl::status::success;
}

impl::status_t swap_relu_mul_scales(std::shared_ptr<subgraph_t> &sg) {
    while (true) {
        std::vector<std::pair<graph::op_t *, graph::op_t *>> to_be_swapped;
        for (auto &op : sg->get_ops()) {
            bool ok = op->get_kind() == op_kind::dnnl_mul_scales
                    && op->get_input_value(0)->has_producer();
            if (!ok) continue;
            graph::op_t *producer = op->get_input_op(0);
            ok = producer->get_kind() == op_kind::dnnl_eltwise;
            if (!ok) continue;
            const auto alg = static_cast<dnnl::algorithm>(
                    producer->get_attr<int64_t>(op_attr::alg_kind));
            ok = alg == dnnl::algorithm::eltwise_relu;
            if (!ok) continue;

            // only support batchnorminference+relu+mul_scale
            ok = producer->get_input_value(0)->has_producer();
            if (!ok) continue;
            const graph::op_t &prv_op
                    = producer->get_input_value(0)->get_producer();
            if (prv_op.get_kind() == op_kind::dnnl_batchnorm
                    && !prv_op.get_attr<bool>(op_attr::is_training)) {
                to_be_swapped.emplace_back(
                        std::pair<graph::op_t *, graph::op_t *> {
                                producer, op.get()});
            } else {
                continue;
            }
        }
        if (to_be_swapped.empty()) break;
        subgraph_rewriter_t rewriter(sg);
        for (auto &pair : to_be_swapped) {
            graph::op_t *relu = pair.first;
            graph::op_t *mul_scales = pair.second;
            rewriter.swap_neighboring_si_ops(
                    relu->shared_from_this(), mul_scales->shared_from_this());
        }
    }
    return infer_shape(sg);
}

impl::status_t fold_pre_mul_scale_into_bn(std::shared_ptr<subgraph_t> &sg) {
    const auto get_next_op = [](const op_ptr &op) -> op_ptr {
        const value_ptr out_val = op->get_output_value(0);
        if (!out_val->get_consumers().empty()) {
            size_t offset = out_val->get_consumers()[0].get_offset();
            auto &next_op = out_val->get_consumers()[0].get_op();
            return offset == 0 && next_op.get_kind() == op_kind::dnnl_batchnorm
                    ? next_op.shared_from_this()
                    : nullptr;
        }
        return nullptr;
    };

    subgraph_rewriter_t rewriter(sg);
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_mul_scales) continue;
        auto next_op = get_next_op(cur_op);

        if (next_op && !next_op->get_attr<bool>(op_attr::is_training)) {
            auto gamma_quant_op = dnnl_impl::clone_mul_scales(cur_op);
            auto mean_quant_op = dnnl_impl::clone_mul_scales(cur_op);
            dnnl_impl::inverse_mul_scales(mean_quant_op);

            rewriter.insert_op_before(gamma_quant_op, next_op, 1, 0, 0);
            rewriter.insert_op_before(mean_quant_op, next_op, 3, 0, 0);

            auto quant_data_out_val = cur_op->get_output_value(0);
            auto quant_data_in_val = cur_op->get_input_value(0);
            next_op->connect_input(0, quant_data_in_val);
            quant_data_out_val->remove_consumer(*next_op, 0);
            if (quant_data_out_val->get_consumers().empty()) {
                rewriter.to_remove(cur_op);
            }
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

impl::status_t fold_post_mul_scale_into_bn(std::shared_ptr<subgraph_t> &sg) {
    const auto get_prev_op = [](const op_ptr &op) -> op_ptr {
        const auto in_val = op->get_input_value(0);
        if (in_val->has_producer()) {
            auto &bn_op = in_val->get_producer();
            return bn_op.get_kind() == op_kind::dnnl_batchnorm
                    ? bn_op.shared_from_this()
                    : nullptr;
        }
        return nullptr;
    };

    subgraph_rewriter_t rewriter(sg);
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != op_kind::dnnl_mul_scales) continue;
        auto bn_op = get_prev_op(cur_op);
        if (bn_op && !bn_op->get_attr<bool>(op_attr::is_training)) {
            auto gamma_quant_op = dnnl_impl::clone_mul_scales(cur_op);
            auto beta_quant_op = dnnl_impl::clone_mul_scales(cur_op);
            rewriter.insert_op_before(gamma_quant_op, bn_op, 1, 0, 0);
            rewriter.insert_op_before(beta_quant_op, bn_op, 2, 0, 0);
            value_ptr quant_data_out_val = cur_op->get_output_value(0);
            bn_op->connect_output(0, quant_data_out_val);
            rewriter.to_remove(cur_op);
        }
    }

    rewriter.run();
    return infer_shape(sg);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

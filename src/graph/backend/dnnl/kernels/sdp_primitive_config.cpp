/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "graph/backend/dnnl/kernels/sdp_primitive_config.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"

#include "common/compiler_workarounds.hpp"

#define VCHECK_SDP_PRIMITIVE(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_primitive_kernel_t, (cond), status, \
            msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

op_ptr sdp_primitive_config_t::get_post_op(const op_ptr &op) const {
    const auto out_val = op->get_output_value(0);
    const auto &consumers = out_val->get_consumers();
    if (consumers.size() != 1) return nullptr;
    return consumers[0].get_op().shared_from_this();
}

status_t sdp_primitive_config_t::locate_io(std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {

    using dnnl::impl::utils::one_of;

    auto follow_back = [](std::shared_ptr<value_t> val) {
        while (val->has_producer() && val->get_producer().num_inputs() == 1)
            val = val->get_producer().get_input_value(0);
        return val;
    };

    auto in_tensor_list = [](const value_t *val,
                                  const std::vector<logical_tensor_t> &list) {
        for (auto &t : list)
            if (val->get_logical_tensor().id == t.id) return true;
        return false;
    };

    // Locate ops of interest: matmuls, scale, mask
    op_ptr mm1 = nullptr, mm2 = nullptr, scale = nullptr, add = nullptr,
           final_op = nullptr;
    const std::unordered_set<op_kind_t> mm1_post_op_kind
            = {op_kind::dnnl_binary, op_kind::dnnl_softmax, op_kind::dnnl_mask};
    for (const auto &cur_op : sg->get_ops()) {
        if (in_tensor_list(cur_op->get_output_value(0).get(), outputs))
            final_op = cur_op;
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
        auto post_op = get_post_op(cur_op);
        if (post_op && mm1_post_op_kind.count(post_op->get_kind())) {
            // Locate mm1 and all post ops(scale and mask) here.
            // 1. locate mm1
            VCHECK_SDP_PRIMITIVE(mm1 == nullptr, status::unimplemented,
                    "Multiple mm1 found");
            mm1 = cur_op;
            // At least one of scale and mask exists
            if (post_op->get_kind() == op_kind::dnnl_binary) {
                auto binary_alg = static_cast<alg_kind_t>(
                        post_op->get_attr<int64_t>(op_attr::alg_kind));
                // 2. locate scale if have
                if (one_of(binary_alg, alg_kind::binary_mul,
                            alg_kind::binary_div)) {
                    scale = post_op;
                    invert_scale_ = (binary_alg == alg_kind::binary_div);
                    // Update `post_op` to the next op of scale
                    post_op = get_post_op(post_op);
                }

                // 3. locate mask if have
                if (post_op->get_kind() == op_kind::dnnl_binary) {
                    add = post_op;
                } else if (post_op->get_kind() == op_kind::dnnl_mask) {
                    // implicit causal mask
                    causal_mask_ = true;
                }
            } else if (post_op->get_kind() == op_kind::dnnl_mask) {
                causal_mask_ = true;
            }
        } else {
            VCHECK_SDP_PRIMITIVE(mm2 == nullptr, status::unimplemented,
                    "Multiple mm2 found");
            mm2 = cur_op;
        }
    }

    // Locate input/outputs: Q, K, V, dst, scale, mask
    mm1_ = mm1;
    mm2_ = mm2;
    VCHECK_SDP_PRIMITIVE((mm1 && mm2 && final_op), status::unimplemented,
            "Not all ops are found");

    q_ = mm1->get_input_value(0);
    k_ = mm1->get_input_value(1);
    v_ = mm2->get_input_value(1);

    if (quantized_) {
        // The input order of fused matmul is: src_0, src_1, scale, zero points
        if (mm1->num_inputs() > 2) k_scale_ = mm1->get_input_value(2);
        if (mm2->num_inputs() > 2) v_scale_ = mm2->get_input_value(2);

        // asymmetric quantization for key.
        if (4 == mm1->num_inputs()) k_zero_points_ = mm1->get_input_value(3);
        // asymmetric quantization for value.
        if (4 == mm2->num_inputs()) v_zero_points_ = mm2->get_input_value(3);
    }

    auto k_follow = follow_back(k_);
    for (auto &t : inputs)
        if (k_follow->get_logical_tensor().id == t.id) {
            kv_head_number_ = t.dims[1];
        }
    dst_ = (final_op->get_kind() == op_kind::dnnl_transpose)
            ? final_op->get_input_value(0)
            : final_op->get_output_value(
                    0); /* for some reason final transpose is not fused into mm2 */

    if (scale) {
        auto s0 = follow_back(scale->get_input_value(0));
        auto s1 = follow_back(scale->get_input_value(1));
        scale_ = in_tensor_list(s1.get(), inputs) ? s1 : s0;
    }

    if (add) {
        auto m0 = add->get_input_value(0), m1 = add->get_input_value(1);
        if (in_tensor_list(m1.get(), inputs)) {
            attn_mask_ = m1;
        } else if (in_tensor_list(m0.get(), inputs)) {
            attn_mask_ = m0;
        } else if (m1->has_producer()
                && m1->get_producer().get_kind() == op_kind::dnnl_unsqueeze
                && in_tensor_list(
                        m1->get_producer().get_input_value(0).get(), inputs)) {
            // consider the case when mask is not 4D,
            // unsqueeze op is inserted to broadcast the mask
            attn_mask_ = m1;
        } else if (m0->has_producer()
                && m0->get_producer().get_kind() == op_kind::dnnl_unsqueeze
                && in_tensor_list(
                        m0->get_producer().get_input_value(0).get(), inputs)) {
            attn_mask_ = m0;
        } else {
            VCHECK_SDP_PRIMITIVE(
                    false, status::unimplemented, "explicit mask is not found");
        }
    }

    return status::success;
}

status_t sdp_primitive_config_t::initial_check(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    // At least 3 inputs: Q, K, V
    VCHECK_SDP_PRIMITIVE(inputs.size() >= 3, status::invalid_arguments,
            "At least 3 inputs are required");

    // step1(pattern check): Not support sdpa variants with select as mask
    // We already have a pattern matcher to ensure that the sdpa patterns
    // dispatch to here are knows ones, and we have quant check in sdpa base
    // kernel, so here we only check specific variants based on support matrix.
    const std::unordered_set<graph::op_kind_t> mm1_post_op_kind
            = {graph::op_kind::Divide, graph::op_kind::Multiply,
                    graph::op_kind::Add, graph::op_kind::Select,
                    graph::op_kind::SoftMax};
    op_ptr mm1 = nullptr, mm2 = nullptr, scale = nullptr;
    for (const auto &cur_op : sg->get_ops()) {
        const auto &op_kind = cur_op->get_kind();
        if (op_kind == graph::op_kind::DynamicDequantize
                && cur_op->get_attr<std::string>(op_attr::qtype)
                        == "per_group") {
            if (!cur_op->has_attr(op_attr::group_shape))
                return status::invalid_arguments;
            const auto &group_shape = cur_op->get_attr<std::vector<int64_t>>(
                    op_attr::group_shape);
            const auto &input_lt
                    = cur_op->get_input_value(0)->get_logical_tensor();
            const auto &input_dims = ltw(input_lt).dims();
            if (static_cast<int>(group_shape.size()) != ltw(input_lt).ndims())
                return status::invalid_arguments;
            // Due to the precision issue of ukernel implementation, we only
            // support group_num=1 case for now.
            for (size_t idx = 0; idx < group_shape.size(); ++idx) {
                if (group_shape[idx] != 1
                        && group_shape[idx] != input_dims[idx])
                    return status::unimplemented;
            }
            // TODO(zhitao): execute the reorder for scale and zps mannually if the
            // transpose attribute is specified as true.
            auto post_op = get_post_op(cur_op);
            if (post_op && post_op->get_kind() == graph::op_kind::MatMul
                    && post_op->has_attr(op_attr::transpose_b)
                    && post_op->get_attr<bool>(op_attr::transpose_b))
                return status::unimplemented;
        }
        if (op_kind != graph::op_kind::MatMul) continue;
        auto post_op = get_post_op(cur_op);
        if (post_op && mm1_post_op_kind.count(post_op->get_kind())) {
            mm1 = cur_op;
            // Not support select between mm1 and scale(optional)
            // GPT-J:[mm1] --> [select] --> [scale]* --> [mask]* --> ...
            VCHECK_SDP_PRIMITIVE(post_op->get_kind() != graph::op_kind::Select,
                    status::unimplemented,
                    "Not support select between mm1 and scale(optional)");
            // scale
            if (post_op->get_kind() == graph::op_kind::Divide
                    || post_op->get_kind() == graph::op_kind::Multiply) {
                // Scale exists, update post_op and traverse to next op
                scale = post_op;
                post_op = get_post_op(post_op);
            }
            // mask
            if (post_op) {
                if (post_op->get_kind() == graph::op_kind::Add) {
                    // Mask exists, update post_op and traverse to next op
                    post_op = get_post_op(post_op);
                }
                // Not support select after scale(optional) and mask(optional)
                // Distill-Bert:[mm1] --> [scale]* --> [mask]* --> [select] --> ...
                VCHECK_SDP_PRIMITIVE(post_op
                                && post_op->get_kind()
                                        != graph::op_kind::Select,
                        status::unimplemented,
                        "Not support select after scale(optional) and "
                        "mask(optional)");
            }
        } else {
            mm2 = cur_op;
        }
    }

    auto find_graph_inport = [&inputs](const std::shared_ptr<value_t> &val) {
        auto tmp_val = val;
        while (tmp_val->has_producer()) {
            const op_t &prod_op = tmp_val->get_producer();
            tmp_val = prod_op.get_input_value(0);
        }
        for (int i = 0; i < (int)inputs.size(); i++) {
            if (tmp_val->get_logical_tensor().id == inputs[i].id) { return i; }
        }
        // If the corresponding input is not found, return an invalid value
        return -1;
    };

    VCHECK_SDP_PRIMITIVE(
            mm1 && mm2, status::invalid_graph, "mm1 or mm2 is not found");

    // step3(dims check): only support 4-dims now.
    int q_id = find_graph_inport(mm1->get_input_value(0));
    int k_id = find_graph_inport(mm1->get_input_value(1));
    int v_id = find_graph_inport(mm2->get_input_value(1));

    VCHECK_SDP_PRIMITIVE(q_id != -1 && k_id != -1 && v_id != -1,
            status::unimplemented, "Q, K, V are not found");
    VCHECK_SDP_PRIMITIVE(ltw(inputs[q_id]).vdims().size() == 4
                    && ltw(inputs[k_id]).vdims().size() == 4
                    && ltw(inputs[v_id]).vdims().size() == 4,
            status::unimplemented, "Q, K, V should be 4-dims");

    // sdp_primitive only supports single scale value.
    if (scale) {
        const auto &s = scale->get_input_value(1)->get_logical_tensor();
        VCHECK_SDP_PRIMITIVE(ltw(s).nelems() == 1, status::unimplemented,
                "Scale should be single value");
    }

    return status::success;
}

status_t sdp_primitive_config_t::init(std::shared_ptr<subgraph_t> &sg,
        const dnnl::engine &p_engine,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {

    CHECK(locate_io(sg, inputs, outputs));

    // Retrieve mds and create pd, primitive
    auto md_q = make_dnnl_memory_desc(q_->get_logical_tensor());
    auto md_k = make_dnnl_memory_desc(k_->get_logical_tensor());
    auto md_v = make_dnnl_memory_desc(v_->get_logical_tensor());
    auto md_dst = make_dnnl_memory_desc(dst_->get_logical_tensor());

    dnnl::memory::desc md_mask;
    if (attn_mask_)
        md_mask = make_dnnl_memory_desc(attn_mask_->get_logical_tensor());

    auto scale_dt = impl::data_type::undef;
    if (scale_) scale_dt = scale_->get_logical_tensor().data_type;

    dnnl::primitive_attr attr, qk_attr, vs_attr;

    auto &mgr = sg->fusion_info_mgr_;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_fpmath_mode(
            static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode().mode_));

    if (mm1_->has_attr(op_attr::fusion_info_key)
            && mm1_->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = mm1_->get_attr<int64_t>(op_attr::fusion_info_key);
        qk_attr = make_dnnl_primitive_attr(mm1_, mgr.get_info(key));
    }
    if (mm2_->has_attr(op_attr::fusion_info_key)
            && mm2_->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = mm2_->get_attr<int64_t>(op_attr::fusion_info_key);
        vs_attr = make_dnnl_primitive_attr(mm2_, mgr.get_info(key));
    }

    CHECK(create_sdpa_pd(sdpa_pd_, p_engine.get(), md_q.get(), md_k.get(),
            md_v.get(), md_dst.get(), md_mask.get(), scale_dt, invert_scale_,
            kv_head_number_, causal_mask_, attr.get(), qk_attr.get(),
            vs_attr.get()));

    auto status = sdpa_pd_->create_primitive(sdpa_prim_, p_engine.get());

    VCONDCHECK(graph, create, dispatch, sdp, status == status::success, status,
            "could not create sdp primitive, falling back\n");
    return status;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

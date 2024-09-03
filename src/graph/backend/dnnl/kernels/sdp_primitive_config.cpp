/*******************************************************************************
* Copyright 2024 Intel Corporation
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
            = {op_kind::dnnl_binary, op_kind::dnnl_softmax};
    for (const auto &cur_op : sg->get_ops()) {
        if (in_tensor_list(cur_op->get_output_value(0).get(), outputs))
            final_op = cur_op;
        if (cur_op->get_kind() != op_kind::dnnl_matmul) continue;
        auto post_op = get_post_op(cur_op);
        if (post_op && mm1_post_op_kind.count(post_op->get_kind())) {
            // Locate mm1 and all post ops(scale and mask) here.
            // 1. locate mm1
            if (mm1) return status::unimplemented;
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
                }
            }
        } else {
            if (mm2) return status::unimplemented;
            mm2 = cur_op;
        }
    }

    // Locate input/outputs: Q, K, V, dst, scale, mask
    if (!mm1 || !mm2 || !final_op) return status::unimplemented;
    q_ = mm1->get_input_value(0);
    k_ = mm1->get_input_value(1);
    v_ = mm2->get_input_value(1);
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
        attn_mask_ = in_tensor_list(m1.get(), inputs) ? m1 : m0;
    }

    return status::success;
}

status_t sdp_primitive_config_t::initial_check(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs) {
    // At least 3 inputs: Q, K, V
    if (inputs.size() < 3) return status::invalid_arguments;

    // step1(pattern check): Not support sdpa variants with select as mask
    // We already have a pattern matcher to ensure that the sdpa patterns
    // dispatch to here are knows ones, and we have quant check in sdpa base
    // kernel, so here we only check specific variants based on support matrix.
    const std::unordered_set<graph::op_kind_t> mm1_post_op_kind
            = {graph::op_kind::Divide, graph::op_kind::Multiply,
                    graph::op_kind::Add, graph::op_kind::Select,
                    graph::op_kind::SoftMax};
    op_ptr mm1 = nullptr, mm2 = nullptr;
    for (const auto &cur_op : sg->get_ops()) {
        if (cur_op->get_kind() != graph::op_kind::MatMul) continue;
        auto post_op = get_post_op(cur_op);
        if (post_op && mm1_post_op_kind.count(post_op->get_kind())) {
            mm1 = cur_op;
            // Not support select between mm1 and scale(optional)
            // GPT-J:[mm1] --> [select] --> [scale]* --> [mask]* --> ...
            if (post_op->get_kind() == graph::op_kind::Select) {
                return status::unimplemented;
            }
            // scale
            if (post_op->get_kind() == graph::op_kind::Divide
                    || post_op->get_kind() == graph::op_kind::Multiply) {
                // Scale exists, update post_op and traverse to next op
                post_op = get_post_op(post_op);
            }
            // mask
            if (post_op->get_kind() == graph::op_kind::Add) {
                // Mask exists, update post_op and traverse to next op
                post_op = get_post_op(post_op);
            }

            // Not support select after scale(optional) and mask(optional)
            // Distill-Bert:[mm1] --> [scale]* --> [mask]* --> [select] --> ...
            if (post_op->get_kind() == graph::op_kind::Select) {
                return status::unimplemented;
            }
        } else {
            mm2 = cur_op;
        }
    }

    // step2(data type check): only support fp16 now.
    auto in_lt = inputs[0];
    if (in_lt.data_type != dnnl_data_type_t::dnnl_f16)
        return status::unimplemented;

    auto find_graph_inport = [&inputs](const std::shared_ptr<value_t> &val) {
        for (int i = 0; i < (int)inputs.size(); i++) {
            if (val->get_logical_tensor().id == inputs[i].id) { return i; }
        }
        // If the corresponding input is not found, return an invalid value
        return -1;
    };

    // step3(dims check): only support 4-dims now.
    int q_id = find_graph_inport(mm1->get_input_value(0));
    int k_id = find_graph_inport(mm1->get_input_value(1));
    int v_id = find_graph_inport(mm2->get_input_value(1));

    bool ok = true;
    ok = ok && (q_id != -1) && (k_id != -1) && (v_id != -1);
    if (!ok) return status::unimplemented;
    ok = ok && ltw(inputs[q_id]).vdims().size() == 4
            && ltw(inputs[k_id]).vdims().size() == 4
            && ltw(inputs[v_id]).vdims().size() == 4;

    return ok ? status::success : status::unimplemented;
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

    dnnl::primitive_attr attr;

    auto &mgr = sg->fusion_info_mgr_;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_fpmath_mode(static_cast<dnnl::fpmath_mode>(mgr.get_fpmath_mode()));

    CHECK(create_sdpa_pd(sdpa_pd_, p_engine.get(), md_q.get(), md_k.get(),
            md_v.get(), md_dst.get(), md_mask.get(), scale_dt, invert_scale_,
            attr.get()));

    auto status = sdpa_pd_->create_primitive(sdpa_prim_, p_engine.get());

    if (status != status::success) {
        if (get_verbose(verbose_t::create_dispatch, component_t::graph)) {
            verbose_printf(
                    "graph,create:dispatch,sdpa,could not create primitive, "
                    "falling back\n");
        }
    }

    return status;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

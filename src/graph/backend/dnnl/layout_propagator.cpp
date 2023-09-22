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

#include <memory>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/value.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/layout_propagator.hpp"
#include "graph/backend/dnnl/op_executable.hpp"

#include "graph/interface/shape_infer.hpp"

#include "graph/backend/dnnl/dnnl_shape_infer.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_t = op_t;
using op_ptr = std::shared_ptr<op_t>;
using value_ptr = std::shared_ptr<value_t>;
using ltw = logical_tensor_wrapper_t;

status_t insert_reorder_before(op_ptr &op, size_t offset,
        const dnnl::memory::desc &opt_mdesc, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    value_ptr in_val = op->get_input_value(offset);
    const logical_tensor_t &in_lt = in_val->get_logical_tensor();
    // just return if real input layout is the same as optimal layout or
    // input layout type is ANY
    if (make_dnnl_memory_desc(in_lt) == opt_mdesc || ltw(in_lt).is_any())
        return status;

    // create reorder op, connect it to graph and add it's scratchpad output
    auto reorder_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
    rewriter.insert_op_before(reorder_op, op, offset);
    auto scratchpad_val = insert_empty_scratchpad(reorder_op);
    // set optimal layout to reorder's output
    auto reorder_out_val = reorder_op->get_output_value(0);
    status = fill_layout_info(reorder_out_val, opt_mdesc);
    if (status != status::success) return status;
    // fill shape info
    reorder_out_val->set_data_type(ltw(in_lt).data_type());
    reorder_out_val->set_dims(ltw(in_lt).vdims());

    // set layout info for scratchpad output
    const auto &pd = reorder_executable_t::create_desc(
            reorder_op, p_engine, mgr, pd_cache);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t insert_reorder_after(op_ptr &op, size_t offset,
        const dnnl::memory::desc &opt_mdesc, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    value_ptr out_val = op->get_output_value(offset);
    const logical_tensor_t &out_lt = out_val->get_logical_tensor();
    // just return if real output layout is the same as optimal layout or
    // output layout type is ANY
    if (make_dnnl_memory_desc(out_lt) == opt_mdesc || ltw(out_lt).is_any())
        return status;

    // create reorder op, connect it to graph and add it's scratchpad output
    auto reorder_op = std::make_shared<op_t>(op_kind::dnnl_reorder);
    rewriter.insert_op_after(reorder_op, op, offset);
    auto scratchpad_val = insert_empty_scratchpad(reorder_op);
    // set optimal layout to reorder's input
    auto reorder_in_val = reorder_op->get_input_value(0);
    status = fill_layout_info(reorder_in_val, opt_mdesc);
    if (status != status::success) return status;
    // fill shape info
    reorder_in_val->set_data_type(ltw(out_lt).data_type());
    reorder_in_val->set_dims(ltw(out_lt).vdims());

    // set layout info for scratchpad output
    const auto &pd = reorder_executable_t::create_desc(
            reorder_op, p_engine, mgr, pd_cache);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_conv(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    // always create pd using any format
    const auto &pd
            = conv_fwd_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    // insert reorders for conv's inputs
    insert_reorder_before(
            op, 0, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    if (status != status::success) return status;

    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        insert_reorder_before(
                op, 2, pd.bias_desc(), p_engine, mgr, pd_cache, rewriter);
        value_ptr bias = op->get_input_value(2);
        status = fill_layout_info(bias, pd.bias_desc());
        if (status != status::success) return status;
    }

    fusion_info_t fusion_info;
    if (op->has_attr(op_attr::fusion_info_key)
            && op->get_attr<int64_t>(op_attr::fusion_info_key) != -1) {
        int64_t key = op->get_attr<int64_t>(op_attr::fusion_info_key);
        fusion_info = mgr.get_info(key);
    }
    if (fusion_info.has_post_dw_conv()) {
        const auto &dw_conv = fusion_info.get_post_dw_conv();
        auto dw_wei_idx = dw_conv->get_unfused_input_indices();
        // TODO(xx) support bias
        assertm(dw_wei_idx.size() == 1,
                "not support bias for fused dw_conv now");

        const auto &dw_wei_opt_mdesc = pd.query_md(query::exec_arg_md,
                DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
        insert_reorder_before(op, dw_wei_idx[0], dw_wei_opt_mdesc, p_engine,
                mgr, pd_cache, rewriter);
        value_ptr dw_wei = op->get_input_value(2);
        status = fill_layout_info(dw_wei, dw_wei_opt_mdesc);
        if (status != status::success) return status;
    }
    // insert a reorder if output layout is different from output optimal layout
    // 1) output layout is opaque
    // 2) output is any, directly set optimal layout
    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    // fill scratchpads dimensions and data type to scratchpad value_t
    // according to op schema, scratchpad must be be second output
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_deconv(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd
            = deconv_fwd_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    // insert reorders for deconv's inputs
    insert_reorder_before(
            op, 0, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    if (status != status::success) return status;

    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        insert_reorder_before(
                op, 2, pd.bias_desc(), p_engine, mgr, pd_cache, rewriter);
        value_ptr bias = op->get_input_value(2);
        status = fill_layout_info(bias, pd.bias_desc());
        if (status != status::success) return status;
    }
    // insert a reorder if output layout is different from output optimal layout
    // 1) output layout is opaque
    // 2) output is any, directly set optimal layout
    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_deconv_bwd_data(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    // always create pd using any format
    const auto &pd = deconv_bwd_data_executable_t::create_desc(
            op, p_engine, mgr, pd_cache);

    // insert reorders for inputs
    insert_reorder_before(
            op, 0, pd.diff_dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_dst = op->get_input_value(0);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    if (status != status::success) return status;

    // insert a reorder if output layout is different from output optimal layout
    // 1) output layout is opaque
    // 2) output is any, directly set optimal layout
    insert_reorder_after(
            op, 0, pd.diff_src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    if (status != status::success) return status;

    // fill scratchpads dimensions and data type to scratchpad value_t
    // according to op schema, scratchpad must be be second output
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_deconv_bwd_weights(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd = deconv_bwd_weights_executable_t::create_desc(
            op, p_engine, mgr, pd_cache);

    insert_reorder_before(
            op, 0, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.diff_dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_dst = op->get_input_value(1);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 0, pd.diff_weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_weights = op->get_output_value(0);
    status = fill_layout_info(diff_weights, pd.diff_weights_desc());
    if (status != status::success) return status;

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_eltwise(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    // When input's layout is specified (opaque or strided),
    // we can propagate it to output.
    const auto &pd
            = eltwise_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_eltwise_bwd(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd = eltwise_bwd_executable_t::create_desc(
            op, p_engine, mgr, pd_cache);

    // to hit an optimized kernel, input/output of forward and both diff_dst and
    // diff_src should use the same memory format. Primitive is created based on
    // a backward data and here we are conditionally aligning forward data
    // format.
    auto opt_desc = (op->has_attr(op_attr::use_dst)
                            && op->get_attr<bool>(op_attr::use_dst))
            ? pd.dst_desc()
            : pd.src_desc();
    insert_reorder_before(op, 0, opt_desc, p_engine, mgr, pd_cache, rewriter);
    value_ptr data = op->get_input_value(0);
    status = fill_layout_info(data, opt_desc);
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.diff_dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_dst = op->get_input_value(1);
    status = fill_layout_info(diff_dst, opt_desc);
    if (status != status::success) return status;
    insert_reorder_after(
            op, 0, pd.diff_src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    if (status != status::success) return status;

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_binary(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    using ltw = logical_tensor_wrapper_t;
    status_t status = status::success;

    // if with zero dimension, the binary op will take no effect, we just
    // complete the layout propagation process by using dummy dst md.
    if (ltw(op->get_input_value(0)->get_logical_tensor()).has_zero_dim()
            || ltw(op->get_input_value(1)->get_logical_tensor())
                       .has_zero_dim()) {
        value_ptr dst = op->get_output_value(0);
        status = fill_layout_info(dst,
                to_ncx_format(
                        make_dnnl_memory_desc(dst->get_logical_tensor())));
        if (status != status::success) return status;

        return fill_layout_info(op->get_output_value(1), dnnl::memory::desc {});
    }

    const auto &pd
            = binary_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_concat(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd
            = concat_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    for (size_t i = 0; i < op->num_inputs(); ++i) {
        insert_reorder_before(op, i, pd.src_desc(static_cast<int>(i)), p_engine,
                mgr, pd_cache, rewriter);
        status = fill_layout_info(
                op->get_input_value(i), pd.src_desc(static_cast<int>(i)));
        if (status != status::success) return status;
    }

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    status = fill_layout_info(op->get_output_value(0), pd.dst_desc());
    if (status != status::success) return status;

    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_shuffle(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd
            = shuffle_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    value_ptr src = op->get_input_value(0);
    value_ptr dst = op->get_output_value(0);

    assertm(!ltw(src->get_logical_tensor()).is_any(),
            "shuffle's src can't be any layout");

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_matmul(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    using ltw = logical_tensor_wrapper_t;
    status_t status = status::success;

    // if with zero dimension, the matmul op will take no effect, we just
    // complete the layout propagation process by using dummy dst md.
    if (ltw(op->get_input_value(0)->get_logical_tensor()).has_zero_dim()
            || ltw(op->get_input_value(1)->get_logical_tensor())
                       .has_zero_dim()) {
        value_ptr dst = op->get_output_value(0);
        status = fill_layout_info(dst,
                to_ncx_format(
                        make_dnnl_memory_desc(dst->get_logical_tensor())));
        if (status != status::success) return status;

        return fill_layout_info(op->get_output_value(1), dnnl::memory::desc {});
    }

    const auto &pd
            = matmul_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    // insert reorders for matmul's inputs
    insert_reorder_before(
            op, 0, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    if (status != status::success) return status;

    if (op->has_attr(op_attr::with_bias)
            && op->get_attr<bool>(op_attr::with_bias)) {
        insert_reorder_before(
                op, 2, pd.bias_desc(), p_engine, mgr, pd_cache, rewriter);
        value_ptr bias = op->get_input_value(2);
        status = fill_layout_info(bias, pd.bias_desc());
        if (status != status::success) return status;
    }
    // insert a reorder if output layout is different from output optimal layout
    // 1) output layout is opaque
    // 2) output is any, directly set optimal layout
    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_pool(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd
            = pool_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    // make scratchpad as pool's last output
    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    if (status != status::success) return status;

    if (op->has_attr(op_attr::is_training)
            && op->get_attr<bool>(op_attr::is_training)) {
        value_ptr workspace_val = op->get_output_value(2);
        const memory::desc &ws_md = pd.workspace_desc();
        workspace_val->set_dims(ws_md.get_dims());
        workspace_val->set_data_type(
                static_cast<data_type_t>(ws_md.get_data_type()));
        status = fill_layout_info(workspace_val, ws_md);
    }
    return status;
}

status_t layout_propagator_for_pool_bwd(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd
            = pool_bwd_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_before(
            op, 0, pd.diff_dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_dst = op->get_input_value(0);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 0, pd.diff_src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    if (status != status::success) return status;

    // make scratchpad as pool's last output
    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_batchnorm(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;

    const auto &pd
            = batchnorm_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_before(
            op, 0, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    if (op->get_attr<bool>(op_attr::is_training)) {
        value_ptr running_mean = op->get_output_value(1);
        value_ptr running_variance = op->get_output_value(2);
        value_ptr batch_mean = op->get_output_value(3);
        value_ptr batch_variance = op->get_output_value(4);

        status = fill_layout_info(running_mean, pd.mean_desc());
        if (status != status::success) return status;
        status = fill_layout_info(running_variance, pd.variance_desc());
        if (status != status::success) return status;
        status = fill_layout_info(batch_mean, pd.mean_desc());
        if (status != status::success) return status;
        status = fill_layout_info(batch_variance, pd.variance_desc());
        if (status != status::success) return status;
    }

    size_t scratchpad_index = op->num_outputs() - 1;

    // if batchnorm's prop_kind is forward_training and fused with ReLU, it will
    // have a workspace output
    if (op->has_attr(op_attr::fuse_relu)
            && op->get_attr<bool>(op_attr::fuse_relu)) {
        scratchpad_index = op->num_outputs() - 2;
        value_ptr workspace_val = op->get_output_value(op->num_outputs() - 1);
        status = fill_layout_info(workspace_val, pd.workspace_desc());
        if (status != status::success) return status;
    }

    value_ptr scratchpad_val = op->get_output_value(scratchpad_index);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());

    return status;
}

status_t layout_propagator_for_batchnorm_bwd(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd = batchnorm_bwd_executable_t::create_desc(
            op, p_engine, mgr, pd_cache);

    insert_reorder_before(
            op, 0, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.diff_dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_dst = op->get_input_value(1);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 2, pd.mean_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr mean = op->get_input_value(2);
    status = fill_layout_info(mean, pd.mean_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 3, pd.variance_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr var = op->get_input_value(3);
    status = fill_layout_info(var, pd.variance_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 0, pd.diff_src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.diff_src_desc());
    if (status != status::success) return status;

    if (op->num_outputs() > 2) {
        value_ptr diff_gamma = op->get_output_value(1);
        value_ptr diff_beta = op->get_output_value(2);

        status = fill_layout_info(diff_gamma, pd.diff_weights_desc());
        if (status != status::success) return status;
        status = fill_layout_info(diff_beta, pd.diff_weights_desc());
        if (status != status::success) return status;
    }

    value_ptr scratchpad_val = op->get_output_values().back();
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_prelu(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd
            = prelu_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_before(
            op, 0, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    value_ptr scratchpad_val = op->get_output_value(1);
    // make scratchpad as prelu's last output
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_prelu_bwd(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd
            = prelu_bwd_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_before(
            op, 0, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    if (status != status::success) return status;

    value_ptr diff_dst = op->get_input_value(2);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 0, pd.diff_src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 1, pd.diff_weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_wei = op->get_output_value(1);
    status = fill_layout_info(diff_wei, pd.diff_weights_desc());
    if (status != status::success) return status;

    value_ptr scratchpad_val = op->get_output_value(2);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_layernorm(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd
            = layernorm_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    if (op->num_outputs() > 2) {
        // keep_stats is true
        value_ptr mean = op->get_output_value(1);
        value_ptr variance = op->get_output_value(2);
        status = fill_layout_info(mean, pd.mean_desc());
        if (status != status::success) return status;
        status = fill_layout_info(variance, pd.variance_desc());
        if (status != status::success) return status;
    }

    // scratchpad is layernorm's last output
    value_ptr scratchpad_val = op->get_output_values().back();
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_layernorm_bwd(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd = layernorm_bwd_executable_t::create_desc(
            op, p_engine, mgr, pd_cache);

    size_t in_index {0};
    insert_reorder_before(
            op, in_index, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(in_index++);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_before(op, in_index, pd.diff_dst_desc(), p_engine, mgr,
            pd_cache, rewriter);
    value_ptr diff_dst = op->get_input_value(in_index++);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, in_index, pd.mean_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr mean = op->get_input_value(in_index++);
    status = fill_layout_info(mean, pd.mean_desc());
    if (status != status::success) return status;

    insert_reorder_before(op, in_index, pd.variance_desc(), p_engine, mgr,
            pd_cache, rewriter);
    value_ptr var = op->get_input_value(in_index++);
    status = fill_layout_info(var, pd.variance_desc());
    if (status != status::success) return status;

    size_t out_index {0};
    insert_reorder_after(op, out_index, pd.diff_src_desc(), p_engine, mgr,
            pd_cache, rewriter);
    value_ptr diff_src = op->get_output_value(out_index++);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    if (status != status::success) return status;

    const bool use_affine = op->get_attr<bool>(op_attr::use_affine);
    if (use_affine) {
        const auto &diff_scale_opt_mdesc
                = pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SCALE);
        insert_reorder_after(op, out_index, diff_scale_opt_mdesc, p_engine, mgr,
                pd_cache, rewriter);
        value_ptr diff_scale = op->get_output_value(out_index++);
        status = fill_layout_info(diff_scale, diff_scale_opt_mdesc);
        if (status != status::success) return status;

        const auto &diff_shift_opt_mdesc
                = pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_SHIFT);
        insert_reorder_after(op, out_index, diff_shift_opt_mdesc, p_engine, mgr,
                pd_cache, rewriter);
        value_ptr diff_shift = op->get_output_value(out_index++);
        status = fill_layout_info(diff_shift, diff_shift_opt_mdesc);
        if (status != status::success) return status;
    }
    auto scratchpad_val = op->get_output_value(op->num_outputs() - 1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_permute(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);

    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    auto perm = op->get_attr<std::vector<int64_t>>(op_attr::permutation);
    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);

        auto int32_perm = dnnl_impl::utils::cast_to_int32(perm);
        dnnl::memory::desc out_md = in_md.permute_axes(int32_perm);
        status = fill_layout_info(dst, out_md);
    } else if (!ltw(out_lt).is_any() && ltw(in_lt).is_any()) {
        dnnl::memory::desc out_md = make_dnnl_memory_desc(out_lt);
        std::vector<int32_t> inverse_perm(perm.size(), -1);
        for (size_t i = 0; i < perm.size(); i++) {
            inverse_perm[static_cast<size_t>(perm[i])]
                    = static_cast<int32_t>(i);
        }
        dnnl::memory::desc in_md = out_md.permute_axes(inverse_perm);
        status = fill_layout_info(src, in_md);
    } else if (!ltw(in_lt).is_any() && !ltw(out_lt).is_any()) {
        // case `conv (opaque) -> permute -> output (strided)` or
        // case `input (strided) -> permute -> conv (opaque)`
        dnnl::memory::desc out_md = make_dnnl_memory_desc(out_lt);
        std::vector<int32_t> inverse_perm(perm.size(), -1);
        for (size_t i = 0; i < perm.size(); i++) {
            inverse_perm[static_cast<size_t>(perm[i])]
                    = static_cast<int32_t>(i);
        }
        dnnl::memory::desc tmp_in_md = out_md.permute_axes(inverse_perm);

        // if the input md derived from output md is different from the real
        // input mem desc, just insert a reorder before the op
        if (make_dnnl_memory_desc(in_lt) != tmp_in_md)
            insert_reorder_before(
                    op, 0, tmp_in_md, p_engine, mgr, pd_cache, rewriter);
    }
    return status;
}

status_t layout_propagator_for_to_group(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    UNUSED(p_engine);
    UNUSED(mgr);
    UNUSED(pd_cache);
    UNUSED(rewriter);

    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md;
        auto groups = op->get_attr<int64_t>(op_attr::groups);
        // avoid dividing by zero at below.
        if (groups == 0) return status::invalid_shape;
        if (op->has_attr(op_attr::is_convtranspose)
                && op->get_attr<bool>(op_attr::is_convtranspose)) {
            auto permuted_weight = transpose(in_md, 0, 1);
            auto permuted_group_weight = to_grouped(permuted_weight, groups);
            out_md = transpose(permuted_group_weight, 1, 2);
        } else {
            out_md = to_grouped(in_md, groups);
        }
        status = fill_layout_info(dst, out_md);
    }
    return status;
}

status_t layout_propagator_for_from_group(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto get_dst_md
            = [](const dnnl::memory::desc &src_md,
                      bool is_convtranspose) -> dnnl::memory::desc {
        if (is_convtranspose) {
            auto permuted_dst = transpose(src_md, 1, 2);
            auto permuted_dst_no_groups = from_grouped(permuted_dst);
            return !permuted_dst_no_groups
                    ? permuted_dst_no_groups
                    : transpose(permuted_dst_no_groups, 0, 1);
        } else {
            return from_grouped(src_md);
        }
    };
    const auto get_strides = [](const dnnl::memory::desc &src_md,
                                     bool is_convtranspose) -> dims {
        if (is_convtranspose) {
            // chain of (transpose -> from_grouped -> transpose) requires
            // such permuted strides, otherwise reshape will fail
            auto strides
                    = get_dense_strides(transpose(src_md, 0, 1).get_dims());
            std::swap(strides[0], strides[1]);
            return strides;
        } else {
            return get_dense_strides(src_md.get_dims());
        }
    };

    value_ptr src = op->get_input_value(0);
    value_ptr dst = op->get_output_value(0);
    auto src_lt = src->get_logical_tensor();
    auto dst_lt = dst->get_logical_tensor();

    if (ltw(src_lt).is_any()) return status;

    const bool is_convtranspose = op->has_attr(op_attr::is_convtranspose)
            ? op->get_attr<bool>(op_attr::is_convtranspose)
            : false;
    const auto src_md = make_dnnl_memory_desc(src_lt);
    dnnl::memory::desc inferred_dst_md = get_dst_md(src_md, is_convtranspose);
    // from_grouped uses the 'allow_empty' option when reshaping, so if reshape
    // will not succeed (e.g. padding exists inside dims we want to join),
    // inferred_dst_md will be an empty memory descriptor.
    if (!inferred_dst_md) {
        dnnl::memory::desc strided_dst_md(src_md.get_dims(),
                src_md.get_data_type(), get_strides(src_md, is_convtranspose));
        insert_reorder_before(
                op, 0, strided_dst_md, p_engine, mgr, pd_cache, rewriter);
        inferred_dst_md = get_dst_md(strided_dst_md, is_convtranspose);
    }

    if (ltw(dst_lt).is_any()) {
        status = fill_layout_info(dst, inferred_dst_md);
    } else {
        insert_reorder_after(
                op, 0, inferred_dst_md, p_engine, mgr, pd_cache, rewriter);
    }
    return status;
}

static status_t layout_propagator_for_reshape_like_ops(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter,
        const dnnl::memory::dims &target_dims) {
    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    assertm(!ltw(in_lt).is_any(), "input layout must be specified");

    if (target_dims.empty()) {
        // scalar, just set empty strides to make the dst strided
        dst->set_strides({});
        return status::success;
    }

    if (ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md
                = in_md.reshape(target_dims, /* allow empty */ true);

        // out_md will be empty if the in_md is not reshape-able. We need to
        // reorder the in_md first and then reshape the reordered reshape-able
        // md.
        if (!out_md) {
            dnnl::memory::desc reshapable_md(in_md.get_dims(),
                    in_md.get_data_type(), get_ncx_format(in_md.get_ndims()));
            insert_reorder_before(
                    op, 0, reshapable_md, p_engine, mgr, pd_cache, rewriter);
            out_md = reshapable_md.reshape(target_dims);
        }

        status = fill_layout_info(dst, out_md);
    } else if (ltw(out_lt).is_strided()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        dnnl::memory::desc out_md = make_dnnl_memory_desc(out_lt);
        // check if the out_md is reshape-able
        dnnl::memory::desc expected_in_md
                = out_md.reshape(in_md.get_dims(), /* allow empty */ true);
        if (expected_in_md) {
            // If the out_md is reshape-able, the expected_in_md must be
            // reshape-able too. Then we just need to check if the real in_md
            // has same layout as the expected_in_md, and insert only one
            // possible reorder if needed.
            if (expected_in_md != in_md) {
                insert_reorder_before(op, 0, expected_in_md, p_engine, mgr,
                        pd_cache, rewriter);
            }
            // finally, we have a chain of: in_md -> (optional reorder) ->
            // expected_in_md -> reshape -> out_md
        } else {
            // Check if the in_md is reshape-able.
            dnnl::memory::desc reshaped_in_md
                    = in_md.reshape(target_dims, /* allow empty */ true);
            if (!reshaped_in_md) {
                dnnl::memory::desc reshapable_md(in_md.get_dims(),
                        in_md.get_data_type(),
                        get_ncx_format(in_md.get_ndims()));
                insert_reorder_before(op, 0, reshapable_md, p_engine, mgr,
                        pd_cache, rewriter);
                reshaped_in_md = reshapable_md.reshape(target_dims);
            }
            // If the reshaped_in_md is not same as the specified out_md, we
            // insert reorder
            if (reshaped_in_md != out_md) {
                insert_reorder_after(op, 0, reshaped_in_md, p_engine, mgr,
                        pd_cache, rewriter);
            }
            // finally, we have a chain of: in_md -> (optional reorder) ->
            // reshapable_md -> reshape -> reshaped_in_md -> (optional reorder)
            // -> out_md. The optional reorder will only occurs when both in_md
            // and out_md are not reshapable.
        }
    }
    return status;
}

status_t layout_propagator_for_reshape(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    auto out_lt = op->get_output_value(0)->get_logical_tensor();
    auto target_dims = ltw(out_lt).vdims();
    status_t status = layout_propagator_for_reshape_like_ops(
            op, p_engine, mgr, pd_cache, rewriter, target_dims);
    return status;
}

status_t layout_propagator_for_transpose(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    assertm(!ltw(in_lt).is_any(), "transpose's src can't be any layout now");

    std::vector<int64_t> order
            = op->get_attr<std::vector<int64_t>>(op_attr::order);
    // if order < 0, convert it to positive order
    if (!order.empty()) {
        for (int64_t &axis : order) {
            if (axis < 0) axis += ltw(in_lt).ndims();
        }
    } else {
        // FIXME(xx) handle this case
        assertm(false, "not handled yet");
    }

    /// The order in spec op is used as:
    /// for (i = 0; i < ndims(); i++)
    ///     new_shape[i] = org_shape[order[i]];
    ///
    /// The axes for permute_axes function is used as:
    /// for (i = 0; i < ndims(); i++)
    ///     new_shape[axes[i]] = org_shape[i];
    ///
    /// So, we need to convert the order to axes
    std::vector<int> axes(order.size(), -1);
    for (size_t i = 0; i < order.size(); i++) {
        size_t new_shape_idx = i;
        size_t org_shape_idx = order[i];
        axes[org_shape_idx] = static_cast<int>(new_shape_idx);
    }

    // calculate the expected transposed layout by permuting the md
    dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
    dnnl::memory::desc expected_out_md = in_md.permute_axes(axes);
    if (ltw(out_lt).is_any()) {
        status = fill_layout_info(dst, expected_out_md);
    } else {
        // if the output layout is specified, we need to check if it's matched
        // with the expected out layout. If not, we should insert a reorder op
        // to convert the transposed layout to the specified one.
        dnnl::memory::desc out_md = make_dnnl_memory_desc(out_lt);
        if (expected_out_md != out_md) {
            insert_reorder_after(
                    op, 0, expected_out_md, p_engine, mgr, pd_cache, rewriter);
        }
    }
    return status;
}

status_t layout_propagator_for_unsqueeze(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    UNUSED(rewriter);

    status_t status = status::success;
    value_ptr src = op->get_input_value(0);
    value_ptr dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
        // 'out_lt' shape should be known at this stage
        status = fill_layout_info(dst, in_md.reshape(ltw(out_lt).vdims()));
    }
    return status;
}

status_t layout_propagator_for_squeeze(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    auto out_lt = op->get_output_value(0)->get_logical_tensor();
    auto target_dims = ltw(out_lt).vdims();
    status_t status = layout_propagator_for_reshape_like_ops(
            op, p_engine, mgr, pd_cache, rewriter, target_dims);
    return status;
}

status_t layout_propagator_for_reorder(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    UNUSED(rewriter);

    status_t status = status::success;
    std::shared_ptr<value_t> src, dst;
    src = op->get_input_value(0);
    dst = op->get_output_value(0);
    auto in_lt = src->get_logical_tensor();
    auto out_lt = dst->get_logical_tensor();

    if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
        assertm(!op->has_attr(op_attr::change_layout)
                        || !op->get_attr<bool>(op_attr::change_layout),
                "the dnnl_reorder op's input and output layout must be known "
                "if it changes layout");

        // for logical tensor with opaque layout, make_dnnl_memory_desc
        // cannot help manually modify the logical tensor. The previously
        // created memory will be quired according to logic tensor id.
        // So data_type of quired memory may be not same as the logical tensor.
        // We used to be able to change the data type manually in oneDNN
        // v2.7, but we can't modify the data type in this way anymore
        // since v3.0.
        auto out_md = make_dnnl_memory_desc(in_lt);
        if (in_lt.data_type != out_lt.data_type) {
            auto format_tag = get_format_tag_str(out_md);
            const auto &dims = out_md.get_dims();
            dnnl_memory_desc_t tmp_md;
            dnnl_memory_desc_create_with_string_tag(&tmp_md,
                    static_cast<int>(dims.size()), dims.data(),
                    static_cast<dnnl_data_type_t>(out_lt.data_type),
                    format_tag.data());
            status = fill_layout_info(dst, tmp_md);
        } else {
            status = fill_layout_info(dst, out_md);
        }
    } else if (!ltw(out_lt).is_any() && ltw(in_lt).is_any()) {
        assertm(!op->has_attr(op_attr::change_layout)
                        || !op->get_attr<bool>(op_attr::change_layout),
                "the dnnl_reorder op's input and output layout must be known "
                "if it changes layout");

        out_lt.data_type = ltw(in_lt).data_type();
        auto in_md = make_dnnl_memory_desc(out_lt);
        status = fill_layout_info(src, in_md);
    }
    if (status != status::success) return status;

    // set layout info for scratchpad output
    if (op->num_outputs() == 1) { insert_empty_scratchpad(op); }

    const auto &pd
            = reorder_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_mul_scales(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    return layout_propagator_for_reorder(op, p_engine, mgr, pd_cache, rewriter);
}

status_t layout_propagator_for_bn_folding(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    UNUSED(rewriter);

    status_t status = status::success;
    // skip the scratchpad
    for (size_t i = 0; i < op->num_outputs() - 1; i++) {
        auto in_lt = op->get_input_value(i)->get_logical_tensor();
        auto out_lt = op->get_output_value(i)->get_logical_tensor();
        if (!ltw(in_lt).is_any() && ltw(out_lt).is_any()) {
            dnnl::memory::desc in_md = make_dnnl_memory_desc(in_lt);
            auto dst = op->get_output_value(i);
            status = fill_layout_info(dst, in_md);
            if (status != status::success) return status;
        }
    }

    auto pd = bn_folding_t::create_desc(op, p_engine, mgr, pd_cache);
    // scratchpad is bn_folding's last inputs
    auto val = op->get_output_value(2);
    status = fill_layout_info(val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_conv_bwd_data(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd = conv_bwd_data_executable_t::create_desc(
            op, p_engine, mgr, pd_cache);

    insert_reorder_before(
            op, 0, pd.diff_dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_dst = op->get_input_value(0);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr wei = op->get_input_value(1);
    status = fill_layout_info(wei, pd.weights_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 0, pd.diff_src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    if (status != status::success) return status;

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_conv_bwd_weights(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd = conv_bwd_weights_executable_t::create_desc(
            op, p_engine, mgr, pd_cache);

    insert_reorder_before(
            op, 0, pd.src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr src = op->get_input_value(0);
    status = fill_layout_info(src, pd.src_desc());
    if (status != status::success) return status;

    insert_reorder_before(
            op, 1, pd.diff_dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_dst = op->get_input_value(1);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 0, pd.diff_weights_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_weights = op->get_output_value(0);
    status = fill_layout_info(diff_weights, pd.diff_weights_desc());
    if (status != status::success) return status;

    // fill scratchpads dimensions and data type to scratchpad value_t
    auto scratchpad_val = op->get_output_value(1);
    const memory::desc scratchpad_desc = pd.scratchpad_desc();
    status = fill_layout_info(scratchpad_val, scratchpad_desc);
    return status;
}

status_t layout_propagator_for_resampling(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd
            = resampling_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    // make scratchpad as resampling's last output
    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_resampling_bwd(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    const auto &pd = resampling_bwd_executable_t::create_desc(
            op, p_engine, mgr, pd_cache);

    insert_reorder_after(
            op, 0, pd.diff_src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    if (status != status::success) return status;

    auto scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_sum(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    value_ptr dst = op->get_output_value(0);
    bool input_has_any_format = false;
    for (const auto &in_val : op->get_input_values()) {
        if (ltw(in_val->get_logical_tensor()).is_any()) {
            input_has_any_format = true;
            break;
        }
    }

    MAYBE_UNUSED(input_has_any_format);
    assertm(!input_has_any_format,
            "input format of sum primitive cannot be any.");

    const auto &pd = sum_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    if (ltw(dst->get_logical_tensor()).is_any()) {
        insert_reorder_after(
                op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
        dst = op->get_output_value(0);
        status = fill_layout_info(dst, pd.dst_desc());
        if (status != status::success) return status;
    }

    // scratchpad is dnnl_sum's last output
    value_ptr scratchpad_val = op->get_output_values().back();
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_softmax(op_ptr &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache,
        subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    value_ptr src = op->get_input_value(0);
    assertm(!ltw(src->get_logical_tensor()).is_any(),
            "softmax/logsoftmax's src can't be any layout now");

    const auto &pd
            = softmax_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_softmax_bwd(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    value_ptr dst = op->get_input_value(1);
    assertm(!ltw(dst->get_logical_tensor()).is_any(),
            "softmax/logsoftmax bwd's dst can't be any layout now");

    const auto &pd = softmax_bwd_executable_t::create_desc(
            op, p_engine, mgr, pd_cache);

    insert_reorder_before(
            op, 0, pd.diff_dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_dst = op->get_input_value(0);
    status = fill_layout_info(diff_dst, pd.diff_dst_desc());
    if (status != status::success) return status;

    insert_reorder_after(
            op, 0, pd.diff_src_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr diff_src = op->get_output_value(0);
    status = fill_layout_info(diff_src, pd.diff_src_desc());
    if (status != status::success) return status;

    // according to op schema, scratchpad must be be second output
    auto scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_reduction(op_ptr &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    status_t status = status::success;
    value_ptr src = op->get_input_value(0);
    assertm(!ltw(src->get_logical_tensor()).is_any(),
            "reduction's src can't be any layout now");

    const auto &pd
            = reduction_executable_t::create_desc(op, p_engine, mgr, pd_cache);

    insert_reorder_after(
            op, 0, pd.dst_desc(), p_engine, mgr, pd_cache, rewriter);
    value_ptr dst = op->get_output_value(0);
    status = fill_layout_info(dst, pd.dst_desc());
    if (status != status::success) return status;

    value_ptr scratchpad_val = op->get_output_value(1);
    status = fill_layout_info(scratchpad_val, pd.scratchpad_desc());
    return status;
}

status_t layout_propagator_for_constant_filler(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    UNUSED(op);
    UNUSED(p_engine);
    UNUSED(mgr);
    UNUSED(pd_cache);
    UNUSED(rewriter);
    return status::success;
}

status_t layout_propagator_for_sub_zps(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    UNUSED(op);
    UNUSED(p_engine);
    UNUSED(mgr);
    UNUSED(pd_cache);
    UNUSED(rewriter);
    assertm(false,
            "dnnl_sub_zps op is only for fusion purpose, we shouldn't do "
            "layout propagation for it");
    return status::invalid_graph_op;
}

status_t layout_propagator_for_add_zps(std::shared_ptr<op_t> &op,
        const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
        pd_cache_t &pd_cache, subgraph_rewriter_t &rewriter) {
    UNUSED(op);
    UNUSED(p_engine);
    UNUSED(mgr);
    UNUSED(pd_cache);
    UNUSED(rewriter);
    assertm(false,
            "dnnl_add_zps op is only for fusion purpose, we shouldn't do "
            "layout propagation for it");
    return status::invalid_graph_op;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

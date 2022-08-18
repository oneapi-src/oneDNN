/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/value.hpp"

#include "graph/utils/compatible.hpp"

#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/op_executable.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
using op_ptr = std::shared_ptr<op_t>;

/// After the lower down, infer shape, infer type and layout propagation passes,
/// each op in the subgraph will has complete attributes and each edge will have
/// complete shape/dtype/layout information. We can create executable for these
/// ops.
status_t compile_ops(std::shared_ptr<subgraph_t> &sg) {
    auto &mgr = sg->fusion_info_mgr_;
    const auto &p_engine = *(sg->p_engine_);
    auto &pd_cache = sg->pd_cache_;

    return topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        auto cur_op = op->shared_from_this();
        std::shared_ptr<op_executable_t> exec;

        if (cur_op->get_kind() == op_kind::dnnl_convolution
                || cur_op->get_kind() == op_kind::dnnl_conv_depthwise) {
            exec = std::make_shared<conv_fwd_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_convtranspose) {
            exec = std::make_shared<deconv_fwd_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_convtranspose_bwd_data) {
            exec = std::make_shared<deconv_bwd_data_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind()
                == op_kind::dnnl_convtranspose_bwd_weights) {
            exec = std::make_shared<deconv_bwd_weights_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_matmul) {
            exec = std::make_shared<matmul_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_eltwise) {
            exec = std::make_shared<eltwise_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_eltwise_bwd) {
            exec = std::make_shared<eltwise_bwd_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_shuffle) {
            exec = std::make_shared<shuffle_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_prelu) {
            exec = std::make_shared<prelu_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_prelu_bwd) {
            exec = std::make_shared<prelu_bwd_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_pool) {
            exec = std::make_shared<pool_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_pool_bwd) {
            exec = std::make_shared<pool_bwd_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_concat) {
            exec = std::make_shared<concat_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_mul_scales
                || cur_op->get_kind() == op_kind::dnnl_reorder) {
            exec = std::make_shared<reorder_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_constant_scales) {
            exec = std::make_shared<fvec_to_fvec_filler>(
                    cur_op, op_attr::scales);
        } else if (cur_op->get_kind() == op_kind::dnnl_constant_zps) {
            exec = std::make_shared<i64vec_to_i32vec_filler>(
                    cur_op, op_attr::zps);
        } else if (cur_op->get_kind() == op_kind::dnnl_permute
                || cur_op->get_kind() == op_kind::dnnl_to_group
                || cur_op->get_kind() == op_kind::dnnl_from_group
                || cur_op->get_kind() == op_kind::dnnl_expand
                || cur_op->get_kind() == op_kind::dnnl_squeeze
                || cur_op->get_kind() == op_kind::dnnl_reshape
                || cur_op->get_kind() == op_kind::dnnl_transpose) {
            // For preprocess ops. The memory_reparser will not do
            // computation, it only re-parses the existing buffer.
            exec = std::make_shared<memory_reparser_t>();
        } else if (cur_op->get_kind() == op_kind::dnnl_bn_folding) {
            exec = std::make_shared<bn_folding_t>(cur_op, p_engine);
        } else if (cur_op->get_kind() == op_kind::dnnl_conv_bwd_data) {
            exec = std::make_shared<conv_bwd_data_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_conv_bwd_weights) {
            exec = std::make_shared<conv_bwd_weights_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_batchnorm) {
            exec = std::make_shared<batchnorm_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_batchnorm_bwd) {
            exec = std::make_shared<batchnorm_bwd_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_layernorm) {
            exec = std::make_shared<layernorm_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_layernorm_bwd) {
            exec = std::make_shared<layernorm_bwd_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_resampling) {
            exec = std::make_shared<resampling_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_resampling_bwd) {
            exec = std::make_shared<resampling_bwd_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_sum) {
            exec = std::make_shared<sum_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_binary) {
            exec = std::make_shared<binary_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_softmax
                || cur_op->get_kind() == op_kind::dnnl_logsoftmax) {
            exec = std::make_shared<softmax_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_softmax_bwd
                || cur_op->get_kind() == op_kind::dnnl_logsoftmax_bwd) {
            exec = std::make_shared<softmax_bwd_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_reduction) {
            exec = std::make_shared<reduction_executable_t>(
                    cur_op, p_engine, mgr, pd_cache);
        } else {
            assertm(false, "unimplemented op, can't compile it");
            return status::unimplemented;
        }

        sg->execs_.emplace_back(exec);
        sg->is_constant_.push_back(op->has_attr(op_attr::is_constant)
                && op->get_attr<bool>(op_attr::is_constant));
        return status::success;
    });
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

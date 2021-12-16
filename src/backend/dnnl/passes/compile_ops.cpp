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

#include <memory>
#include <vector>
#include <unordered_map>

#include "interface/c_types_map.hpp"
#include "interface/value.hpp"
#include "utils/compatible.hpp"

#include "backend/dnnl/passes/compile_ops.hpp"
#include "backend/dnnl/passes/op_executable.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
using op_ptr = std::shared_ptr<impl::op_t>;

/// After the lower down, infer shape, infer type and layout propagation passes,
/// each op in the subgraph will has complete attributes and each edge will have
/// complete shape/dtype/layout information. We can create executable for these
/// ops.
impl::status_t compile_ops(std::shared_ptr<subgraph_t> &sg) {
    auto &prm_attr_mgr = sg->prm_attr_mgr_;
    const auto &p_engine = *(sg->p_engine_);
    auto &pd_cache = sg->pd_cache_;

    return impl::topo_order_visit(sg->get_output_ops(), [&](impl::op_t *op) {
        auto cur_op = op->shared_from_this();
        std::shared_ptr<op_executable_t> exec;

        if (cur_op->get_kind() == impl::op_kind::Convolution
                || cur_op->get_kind() == op_kind::dnnl_convolution
                || cur_op->get_kind() == op_kind::conv_depthwise) {
            exec = std::make_shared<conv_fwd_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::ConvTranspose
                || cur_op->get_kind() == op_kind::dnnl_convtranspose) {
            exec = std::make_shared<deconv_fwd_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::MatMul) {
            exec = std::make_shared<matmul_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_eltwise) {
            exec = std::make_shared<eltwise_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_shuffle) {
            exec = std::make_shared<shuffle_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::PReLU
                || cur_op->get_kind() == op_kind::dnnl_prelu) {
            exec = std::make_shared<prelu_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::MaxPool
                || cur_op->get_kind() == impl::op_kind::AvgPool
                || cur_op->get_kind() == op_kind::dnnl_pool) {
            exec = std::make_shared<pool_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::Concat) {
            exec = std::make_shared<concat_executable_t>(
                    cur_op, p_engine, prm_attr_mgr);
        } else if (cur_op->get_kind() == op_kind::mul_scales
                || cur_op->get_kind() == impl::op_kind::Reorder
                || cur_op->get_kind() == impl::op_kind::TypeCast
                || cur_op->get_kind() == op_kind::dnnl_u8_to_s8) {
            exec = std::make_shared<reorder_executable_t>(
                    cur_op, p_engine, prm_attr_mgr);
        } else if (cur_op->get_kind() == op_kind::dnnl_constant) {
            exec = std::make_shared<const_memory_filler_t>(cur_op);
        } else if (cur_op->get_kind() == op_kind::permute
                || cur_op->get_kind() == op_kind::to_group
                || cur_op->get_kind() == op_kind::expand
                || cur_op->get_kind() == op_kind::squeeze
                || cur_op->get_kind() == op_kind::unsqueeze
                || cur_op->get_kind() == impl::op_kind::StaticReshape
                || cur_op->get_kind() == impl::op_kind::StaticTranspose) {
            // For preprocess ops. The memory_reparser will not do
            // computation, it only re-parses the existing buffer.
            exec = std::make_shared<memory_reparser_t>();
        } else if (cur_op->get_kind() == op_kind::dnnl_bn_folding) {
            exec = std::make_shared<bn_folding_t>(cur_op, p_engine);
        } else if (cur_op->get_kind() == op_kind::dnnl_conv_bwd_data) {
            exec = std::make_shared<conv_bwd_data_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_batchnorm) {
            exec = std::make_shared<batchnorm_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::LayerNorm) {
            exec = std::make_shared<layernorm_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::Interpolate) {
            exec = std::make_shared<resampling_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_sum) {
            exec = std::make_shared<sum_executable_t>(
                    cur_op, p_engine, prm_attr_mgr);
        } else if (cur_op->get_kind() == op_kind::dnnl_binary) {
            exec = std::make_shared<binary_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::SoftMax) {
            exec = std::make_shared<softmax_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == impl::op_kind::LogSoftmax) {
            exec = std::make_shared<logsoftmax_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else if (cur_op->get_kind() == op_kind::dnnl_reduction) {
            exec = std::make_shared<reduction_executable_t>(
                    cur_op, p_engine, prm_attr_mgr, pd_cache);
        } else {
            assertm(false, "unimplemented op, can't compile it");
            return impl::status::compile_fail;
        }

        sg->execs_.emplace_back(exec);
        sg->is_constant_.push_back(op->has_attr("is_constant")
                && op->get_attr<bool>("is_constant"));
        return impl::status::success;
    });
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

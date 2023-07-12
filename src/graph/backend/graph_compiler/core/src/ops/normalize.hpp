/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_NORMALIZE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_NORMALIZE_HPP

#include <memory>
#include <vector>
#include "compiler/ir/graph/graph_op.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

enum class normalize_kind {
    layernorm = 0,
    instancenorm,
};

class normalize_common_t : public graph_op_t,
                           public op_traits::auto_copyable_t {
public:
    normalize_common_t() = default;
    normalize_common_t(const normalize_kind &kind,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void get_graph_impl(std::shared_ptr<sc_graph_t> &graph) override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
};

/**
 * The layer norm op
 * Inputs:
 *  - in[0] - the data input
 *  - in[1] - (Only avaliable when use_affine = True) gamma. @see use_affine
 *  - in[2] - (Only avaliable when use_affine = True) beta. @see use_affine
 * Outputs:
 *  - The result tensor
 *  - Mean(Optional)
 *  - Variance(Optional)
 * Attrs:
 *  - keep_stats: bool - Default = true. Whether to output mean&&var.
 *  - begin_norm_axis: int - Default = -1. which axis to start layer
 * normalization. This will convert to rd_axis in this op.
 *  - use_affine: bool - Default = true. If true, output = output * gamma + beta
 *  - epsilon: float - Default = 1e-5
 *  - rd_axis: vector<int> Internal use. Set reduced axis directly. Do not set
 * begin_norm_axis if you intend to use this parameter.
 * */
class layernorm_op_t : public normalize_common_t {
public:
    layernorm_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : normalize_common_t(normalize_kind::layernorm, ins, outs,
                layernorm_op_attrs(ins, attrs)) {}

private:
    any_map_t layernorm_op_attrs(
            const std::vector<graph_tensor_ptr> &ins, const any_map_t &attrs) {
        any_map_t new_attrs = attrs;
        new_attrs.set("keep_stats", attrs.get_or_else("keep_stats", true));
        new_attrs.set("use_affine", attrs.get_or_else("use_affine", true));
        new_attrs.set("epsilon", attrs.get_or_else("epsilon", 1e-5f));
        if (attrs.has_key("begin_norm_axis")) { // oneDNN specification
            int begin_norm_axis = attrs.get<int>("begin_norm_axis");
            if (begin_norm_axis < 0) {
                begin_norm_axis += ins[0]->details_.get_plain_dims().size();
            }
            if (begin_norm_axis > int(ins[0]->details_.get_plain_dims().size())
                    || begin_norm_axis < 0) {
                throw std::runtime_error(
                        "layernorm_op_t::begin_norm_axis boundary exceed.");
            }
            std::vector<int> rd_axis;
            for (size_t i = begin_norm_axis;
                    i < ins[0]->details_.get_plain_dims().size(); i++)
                rd_axis.emplace_back(i);
            new_attrs.set("rd_axis", rd_axis);
        } else if (attrs.has_key("rd_axis")) { // internal use
            new_attrs.set("rd_axis", attrs.get<std::vector<int>>("rd_axis"));
        } else { // nothing specific
            std::vector<int> rd_axis
                    = {int(ins[0]->details_.get_plain_dims().size() - 1)};
            new_attrs.set("rd_axis", rd_axis);
        }
        return new_attrs;
    }
};

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/kernels/batchnorm.hpp"
#include "graph/backend/dnnl/patterns/fusions.hpp"
#include "graph/backend/dnnl/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace pattern {

namespace pm = graph::utils::pm;
using in_edges_t = pm::in_edges_t;
using pb_graph_t = pm::pb_graph_t;
using FCreateV2Pattern = graph::pass::FCreateV2Pattern;

DNNL_BACKEND_REGISTER_PATTERN_DEF_BEGIN(bn_fusion)

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, bn_relu_fusion)
        .set_priority(8.8f)
        .set_kind(partition_kind::batch_norm_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto bn = pgraph->append_op(
                            graph::op_kind::BatchNormInference);
                    pgraph->append_op(
                            graph::op_kind::ReLU, {in_edge(0, bn, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<batchnorm_fwd_t>();
        });

DNNL_BACKEND_REGISTER_TRANSFORMATION_PATTERN(dnnl, bn_bwd_relu_bwd_fusion)
        .set_priority(8.8f)
        .set_kind(partition_kind::misc_post_ops)
        .set_attr<FCreateV2Pattern>("FCreateV2Pattern",
                [](const std::shared_ptr<pb_graph_t> &pgraph) -> void {
                    auto relu_bwd
                            = pgraph->append_op(graph::op_kind::ReLUBackprop);
                    pgraph->append_op(graph::op_kind::BatchNormTrainingBackprop,
                            {in_edge(0, relu_bwd, 0)});
                })
        .set_attr<FCreateKernel>("FCreateKernel", []() -> kernel_ptr {
            return std::make_shared<batchnorm_bwd_t>();
        });

DNNL_BACKEND_REGISTER_PATTERN_DEF_END

} // namespace pattern
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

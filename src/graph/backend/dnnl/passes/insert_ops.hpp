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
#ifndef GRAPH_BACKEND_DNNL_PASSES_INSERT_OPS_HPP
#define GRAPH_BACKEND_DNNL_PASSES_INSERT_OPS_HPP

#include <memory>
#include <vector>

#include "graph/interface/c_types_map.hpp"

#include "graph/backend/dnnl/passes/lower_down.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t insert_permute(std::shared_ptr<subgraph_t> &sg);

status_t insert_permute_for_shuffle(std::shared_ptr<subgraph_t> &sg);

status_t insert_to_group_for_conv_or_deconv(std::shared_ptr<subgraph_t> &sg);

status_t insert_to_group_for_reorder(std::shared_ptr<subgraph_t> &sg);

/// Insert a transpose op for matmul's input tensors
///
/// Only valid for below scenarios:
/// (1) src or weight's ndims is greater than 1
/// (2) either `transpose_a` or `transpose_b` is true
status_t insert_transpose_for_matmul(std::shared_ptr<subgraph_t> &sg);

/// Insert reshape pair for ndx2d matmul for better performance
///
/// For ndx2d matmul:
/// 1) reshape src0 to 2d(keep last dimension and flatten others)
/// 2) reshape dst back to nd after compilation
status_t insert_reshape_for_ndx2d_matmul(std::shared_ptr<subgraph_t> &sg);

/// Insert an expand-squeeze pair for matmul
///
/// The usage of expand op:
///     There maybe three scenarios as below:
///     (1) one of inputs (src or weight) has only one dimension, DNNL require
//      at two dimensions, so need to insert dim 1 before/after the current dim
///     (2) The batch dimensions of src and weight are not matched, need to
///     expand
///     (3) bias dimensions are not matched with dst, need to expand
///
/// The usage of squeeze op:
///     Only will be inserted when previously expand op(s) inserted
///     For example, considering two inputs [3,4]x[4], the second input will be
///     expanded into [4,1], hence the output shape should be [3,1]. However,
///     this is inconsistent with the results derived from the shape inference.
///     So we use squeeze here to remove the extra 1 dimension to produce output
///     with [3].
status_t insert_expand_and_squeeze_for_matmul(std::shared_ptr<subgraph_t> &sg);

/// Insert an dnnl_reorder op for matmul's weight tensor to shift the weight
/// from u8 to s8
///
/// Only valid for below scenarios: src and weight's dtype are both uint8
status_t insert_u8_to_s8_for_matmul(std::shared_ptr<subgraph_t> &sg);

/// Insert expand_to op to make PReLU input shapes meet DNNL requirements
///
/// expand_to inserts 1 at the beginning of the weight dims as many
/// as needed, so that the weights have the same number of ndims as src.
/// When performing channel to broadcast and data format is NCX,
/// after expanding weights, we additionally have to permute them.
status_t insert_expand_for_prelu(std::shared_ptr<subgraph_t> &sg);

/// Insert expand_to and squeeze op for PReLUBackprop:
///     The weights dims we receive from the frameworks do not meet DNNL
///     requirements. Its number of dimensions are less than the number of src
///     dimensions. Therefore, we need to use expand and squeeze to satisfy
///     certain conditions.
///
/// The usage of expand op:
///     expand_to inserts 1 at the beginning of the weight dims and diff weights
///     dims as many as needed, so that the weights have the same number of
///     ndims as src.
///     When performing broadcast to channel and data format is NCX,
///     after expanding weights, we additionally have to permute them, so that
///     the weights and src have the same channel dim.
///
/// The usage of squeeze op:
///     squeeze op is the inverse of expand and is needed to restore the
///     original dimensions of the diff weights at the output.
status_t insert_expand_and_squeeze_for_prelu_bwd(
        std::shared_ptr<subgraph_t> &sg);

/// Insert expand and squeeze op for reduction:
///     Both OPs will only be inserted when reduction 'keep_dims' attribute is
///     equal to false. Their goal is to make subgraph compatible with oneDNN
///     requirements (no support for dropping axes on which reduction was
///     performed).
///
/// The usage of expand op:
///     It will be placed before each post-op src1 input (if exists).
///
/// The usage of squeeze op:
///     It will be placed after reduction op or (if present) after its last
///     supported post-op. Squeeze will take responsibility for removing
///     reduced dimensions, meaning that reductions 'keep_dims' attribute will
///     be changed to true.
status_t insert_expand_and_squeeze_for_reduction(
        std::shared_ptr<subgraph_t> &sg);

status_t insert_maxpool_forward(std::shared_ptr<subgraph_t> &sg);

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

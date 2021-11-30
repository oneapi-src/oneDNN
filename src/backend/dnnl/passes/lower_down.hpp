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
#ifndef BACKEND_DNNL_PASSES_LOWER_DOWN_HPP
#define BACKEND_DNNL_PASSES_LOWER_DOWN_HPP

#include <memory>
#include <vector>
#include <unordered_map>

#include "interface/c_types_map.hpp"

#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

impl::status_t check_with_bias(std::shared_ptr<subgraph_t> &sg);

impl::status_t fuse_bias_add(std::shared_ptr<subgraph_t> &sg);

impl::status_t split_quant_dequant(std::shared_ptr<subgraph_t> &sg);

impl::status_t folding_mul_scales(std::shared_ptr<subgraph_t> &sg);

impl::status_t fuse_to_int8_conv_or_deconv(std::shared_ptr<subgraph_t> &sg);

impl::status_t fuse_to_int8_matmul(std::shared_ptr<subgraph_t> &sg);

impl::status_t fuse_to_int8_pool(std::shared_ptr<subgraph_t> &sg);

impl::status_t fuse_output_scales(std::shared_ptr<subgraph_t> &sg);

impl::status_t fuse_post_ops(std::shared_ptr<subgraph_t> &sg);

impl::status_t fuse_zero_points(std::shared_ptr<subgraph_t> &sg);

impl::status_t fuse_mul_scales_add_zps(std::shared_ptr<subgraph_t> &sg);

impl::status_t insert_bn_folding(std::shared_ptr<subgraph_t> &sg);

impl::status_t conv_bwd_data_canonicalization(std::shared_ptr<subgraph_t> &sg);

impl::status_t fuse_mul_sigmoid_to_swish(std::shared_ptr<subgraph_t> &sg);

/// translate mixed int8/bf16 matmul subgraph to x8x8bf16 subgraph
///
///     | (u8/s8)  | (u8/s8)               | (u8/s8)  | (u8/s8)
///  dequant    dequant                 dequant    dequant
///     | (f32)    | (f32)                 | (f32)    | (f32)
///  typecast  typecast         -->         \        /
/// (bf16) \     / (bf16)                     matmul
///        matmul                               | (bf16)
///          | (bf16)
///
impl::status_t fuse_typecast_to_matmul(std::shared_ptr<subgraph_t> &sg);

/// translate mixed int8/bf16 matmul+add subgraph to x8x8bf16 subgraph
///
///     | (u8/s8)  | (u8/s8)               | (u8/s8)          | (u8/s8)
///  dequant    dequant    | (u8/s8)            dequant    dequant    | (u8/s8)
/// (f32) \     / (f32) dequant                (f32) \     / (f32) dequant
///        matmul      / (fp32)                       matmul      / (fp32)
///           \     typecast                            \ (fp32) /
///     (bf16) \   / (bf16)                                 add
///             add                                          | (bf16)
///              | (bf16)
impl::status_t fuse_typecast_to_add(std::shared_ptr<subgraph_t> &sg);

/// translate mixed int8/bf16 matmul(+post_ops) subgraph to int8 subgraph
///
///     | (u8/s8)  | (u8/s8)               | (u8/s8)  | (u8/s8)
///  dequant    dequant                 dequant    dequant
///     \ (fp32)   / (fp32)     -->         \ (fp32)  / (fp32)
///        matmul                             matmul
///          | (bf16)                           | (f32)
///      (post_ops)                         (post_ops)
///          | (bf16)                           | (f32)
///       typecast                            quant
///          | (fp32)                           | (u8/s8)
///        quant
///          | (u8/s8)
impl::status_t fuse_post_typecast_to_matmul(std::shared_ptr<subgraph_t> &sg);

/// replace all oneDNN Graph BatchNormalizationInference/
/// BatchNormalizationTraining with dnnl_batchnorm and set corresponding
/// attributes
impl::status_t lower_down_to_dnnl_batchnorm(std::shared_ptr<subgraph_t> &sg);

/// translate the subgraph containing chain of Adds into dnnl_sum
///   in0   in1
///     \    /
///      Add   in2         in0  in1  in2
///        \   /             \   |   / ...
///         Add  in3    -->     sum
///           \   /
///            Add
///            ...
impl::status_t fuse_to_dnnl_sum(std::shared_ptr<subgraph_t> &sg);

// This pass is used to lower the oneDNN Graph binary op (like Add, Multiply,
// ...) to DNNL backend internl dnnl_binary op and insert expand op before
// inputs to make the input shape meet the requirement of oneDNN binary
// primitive
impl::status_t binary_canonicalization(std::shared_ptr<subgraph_t> &sg);

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

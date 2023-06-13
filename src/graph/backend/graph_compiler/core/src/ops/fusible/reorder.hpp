/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_REORDER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_REORDER_HPP

#include <memory>
#include <vector>
#include <compiler/ir/graph/graph_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
void find_vectorized_axis(std::vector<expr> const &blocking_dims_expr,
        sc_data_format_t const &format, int &last_origin_axis,
        int &origin_axis_vectorized);

size_t throw_if_negative(int dim);
static const int TARGET_AXIS_NOT_DEFINE = -1;
std::vector<expr> get_reorder_block2plain_indexes(sc_graph_t &graph,
        const std::vector<expr> &in_indexes, const sc_data_format_t &format,
        const sc_dims &plain_dims, expr &condition, expr &last_axis_offset,
        expr &other_axis_condition,
        const int target_axis = TARGET_AXIS_NOT_DEFINE);
std::vector<expr> get_reorder_plain2block_indexes(
        const std::vector<expr> &in_indexes, const sc_data_format_t &format);
bool can_be_fast_transpose(const context_ptr &ctx, std::vector<int> &inp_a_axis,
        std::vector<int> &inp_b_axis, std::vector<int> &out_a_axis,
        std::vector<int> &out_b_axis, const sc_dims &plain_dims,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const tensor_slice &src,
        const tensor_slice &dst, const sc_data_type_t &dtype, bool is_dynamic,
        bool dynamic_no_padding, bool &use_lanesx16);
void compute_fast_transpose(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        const std::vector<int> &inp_a_axis, const std::vector<int> &inp_b_axis,
        const std::vector<int> &out_a_axis, const std::vector<int> &out_b_axis,
        size_t wkld = 0UL, bool is_dynamic = false,
        bool dynamic_no_padding = false, bool use_lanesx16 = false);
bool can_be_vnni_reorder(const context_ptr &ctx, std::vector<int> &inp_n_axis,
        std::vector<int> &inp_k_axis, std::vector<int> &out_n_axis,
        std::vector<int> &out_k_axis, const sc_dims &plain_dims,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const tensor_slice &src,
        const tensor_slice &dst, const sc_data_type_t &dtype,
        bool &is_vnni_reorder, bool is_dynamic, bool dynamic_no_padding,
        bool &use_x16step);
void do_vnni_reorder(std::vector<stmt_c> &cur_list, std::vector<expr> &rows,
        sc_data_type_t &rows_dtype, const bool is_vnni_reorder,
        const int bf16_step);
void vnni_reorder_insert_kernel(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        std::vector<int> &inp_n_axis, std::vector<int> &inp_k_axis,
        std::vector<int> &out_n_axis, std::vector<int> &out_k_axis,
        size_t wkld = 0UL, bool is_dynamic = false,
        bool dynamic_no_padding = false);
void compute_vnni_reorder(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        std::vector<int> &inp_n_axis, std::vector<int> &inp_k_axis,
        std::vector<int> &out_n_axis, std::vector<int> &out_k_axis,
        size_t wkld = 0UL, const bool &is_vnni_reorder = false,
        bool is_dynamic = false, bool dynamic_no_padding = false,
        bool use_x16step = false);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

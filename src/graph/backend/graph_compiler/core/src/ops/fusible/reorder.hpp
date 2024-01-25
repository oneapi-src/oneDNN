/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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

enum class sc_trans_kernel {
    NO_TRANS, // don't use transpose
    F32_8X8_TRANS, // use f32 8x8 transpose (f32 default)
    F32_16X16_TRANS, // use f32 16x16 transpose (currently this is not enabled)
    U8S8_16X16_TRANS, // use u8s8 16x16 transpose (u8s8 default)
    BIT16_16X16_TRANS, // use bf16x16 transpose (Enabled when the cpu supports
    // the highest instruction set is avx2)
    BIT16_32X8_TRANS // use bf16 32x8 transpose ( bf16 default)
};

enum class sc_vnni_kernel {
    NO_VNNI, // don't use vnni reorder/transpose
    INSERT_REORDER_VNNI, // use inset kernel (step = 16)
    X8_REORDER_VNNI, // use 8 step unpack kernel
    X16_REORDER_VNNI, // use 16 step unpack kernel (default)
    BF16_TRANSPOSE_VNNI, // bf16 transpose (default bf16 vnni transpose)
    U8S8_TRANSPOSE_VNNI, // u8s8 transpose (default u8s8 vnni transpose)
};

void find_vectorized_axis(std::vector<expr> const &blocking_dims_expr,
        sc_data_format_t const &format, int &last_origin_axis,
        int &origin_axis_vectorized);

void find_vectorized_axis(const tensor_slice &tsl,
        sc_data_format_t const &format, int &last_origin_axis,
        int &origin_axis_vectorized);

int collect_axis_shape_size(
        sc_dims &blocking_dims, const std::vector<int> &axis);
size_t throw_if_negative(int dim);
static const int TARGET_AXIS_NOT_DEFINE = -1;
std::vector<expr> get_reorder_block2plain_indexes(sc_graph_t &graph,
        const std::vector<expr> &in_indexes, const sc_data_format_t &format,
        const sc_dims &plain_dims, expr &condition, expr &last_axis_offset,
        expr &other_axis_condition,
        const int target_axis = TARGET_AXIS_NOT_DEFINE);
std::vector<expr> get_reorder_plain2block_indexes(
        const std::vector<expr> &in_indexes, const sc_data_format_t &format);
bool can_be_fast_transpose(const sc_graph_t &graph, const context_ptr &ctx,
        std::vector<int> &inp_a_axis, std::vector<int> &inp_b_axis,
        std::vector<int> &out_a_axis, std::vector<int> &out_b_axis,
        const sc_dims &plain_dims, const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const tensor_slice &src,
        const tensor_slice &dst, const sc_data_type_t &dtype, bool is_dynamic,
        bool dynamic_no_padding, sc_trans_kernel &trans_kernel_used);
void compute_fast_transpose(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        const std::vector<int> &inp_a_axis, const std::vector<int> &inp_b_axis,
        const std::vector<int> &out_a_axis, const std::vector<int> &out_b_axis,
        const graph_tensor_ptr &expand_gt, size_t wkld, bool is_dynamic,
        bool dynamic_no_padding, const sc_trans_kernel trans_kernel_used);
bool can_be_vnni_reorder(const context_ptr &ctx, std::vector<int> &inp_n_axis,
        std::vector<int> &inp_k_axis, std::vector<int> &out_n_axis,
        std::vector<int> &out_k_axis, const sc_dims &plain_dims,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, const tensor_slice &src,
        const tensor_slice &dst, const sc_data_type_t &dtype,
        bool &is_vnni_reorder, bool is_dynamic, bool dynamic_no_padding,
        sc_vnni_kernel &vnni_kernel_used);
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
        const graph_tensor_ptr &expand_gt, size_t wkld = 0UL,
        bool is_dynamic = false, bool dynamic_no_padding = false);
void compute_vnni_reorder(sc_graph_t &graph, const context_ptr &ctx,
        const tensor_slice &src, tensor_slice &dst,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format, sc_data_type_t dtype,
        const sc_dims &plain_dims, bool output_loop, any_map_t &attrs,
        std::vector<int> &inp_n_axis, std::vector<int> &inp_k_axis,
        std::vector<int> &out_n_axis, std::vector<int> &out_k_axis,
        const graph_tensor_ptr &expand_gt, size_t wkld = 0UL,
        const bool &is_vnni_reorder = false, bool is_dynamic = false,
        bool dynamic_no_padding = false,
        const sc_vnni_kernel vnni_kernel_used = sc_vnni_kernel::NO_VNNI);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_UTILS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_UTILS_HPP

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "fusible_op.hpp"
#include "fusion_data.hpp"
#include <unordered_map>

namespace sc {

enum class cmp_res : int {
    unknown = -1,
    equal = 0,
    l_less_r = 1,
    l_larger_r = 2,
};

using slice_range_map = std::unordered_map<int, slice_range_list>;
slice_range_map search_known_slice_ranges(
        sc_op *cur, fslice_map &fsmap, infer_status_map_t &stat_map);
void set_unknown_slice_ranges(fusible_op_t *cur,
        const slice_range_map &known_ranges_map, fslice_map &fsmap,
        infer_status_map_t &stat_map);
void infer_binary_slice_ranges(
        fusible_op_t *cur, fslice_map &fsmap, infer_status_map_t &stat_map);

std::unordered_map<int, bound_axis> search_known_bound_axis(
        sc_op *cur, bound_axis_map &bdax_map);
void set_unknown_axis_binding(sc_op *cur,
        const std::unordered_map<int, bound_axis> &known_axis_map,
        bound_axis_map &bdax_map);

void identical_infer_binding_axis(fusible_op_t *cur, bound_axis_map &bdax_map);
void identical_pre_binding_axis(fusible_op_t *cur, bound_axis_map &bdax_map);

sc_dims get_expr_to_dims(const std::vector<expr> &dims);
size_t get_dims_product(const sc_dims &dims);
// the dim can be squeezed is 1
int get_number_of_squeeze_dims(const sc_dims &dims);

bool slice_full_on_axis(
        const sc_dims &dim, slice_range ranges, const std::vector<int> &axis);

bool slice_divisible_on_axis(
        const sc_dims &dim, slice_range ranges, const std::vector<int> &axis);

inline uint16_t vectorize_step(const context_ptr &ctx, sc_data_etype detype) {
    return std::min(uint16_t(16), ctx->get_max_vector_lanes(detype));
}
bool loop_can_be_fused(const for_loop &loop);

class outer_loop_generator_t;
ir_module_ptr fusible_op_get_func(fusible_op_t *op, outer_loop_generator_t &gen,
        const context_ptr &ctx, bool check_parallel);

struct mask_compute_func_t {
    mask_compute_func_t(const std::function<stmt(const std::vector<expr> &,
                    std::vector<expr::lvalue_proxy_t> &)> &func)
        : impl_(func) {}
    stmt operator()(const std::vector<expr> &in,
            std::vector<expr::lvalue_proxy_t> &out,
            const expr &cur_idx = expr(), const expr &upper_bound = expr(),
            uint32_t lanes = 16) const;
    std::function<stmt(
            const std::vector<expr> &, std::vector<expr::lvalue_proxy_t> &)>
            impl_;
};

using fusion_compute_func_t = std::function<stmt(
        const std::vector<expr> &, std::vector<expr::lvalue_proxy_t> &)>;

void compute_vectorized_op(sc_graph_t &graph,
        const std::vector<const tensor_slice *> &src, const tensor_slice &dst,
        sc_op_info_t &info, const vectorized_info_t &vx_info,
        const mask_compute_func_t &compute_lanes,
        const mask_compute_func_t &compute_scalar, any_map_t &attrs,
        size_t wkld = 0UL, bool use_mask = false,
        const tensor_slice *expand_loop_by
        = nullptr /*by default expand loop by dst*/,
        bool unroll_inner_loop = false);
expr make_select_by_mask(const expr &, const expr &, const expr &, uint32_t);
expr generate_mask_var_by_step(stmt &mask_def, const expr &cur_step,
        int32_t step, const expr &sup_condition = expr());
void compute_mask_and_generate_condition(sc_graph_t &graph,
        const std::vector<const tensor_slice *> &src, const sc_dims &plain_dims,
        sc_data_format_t format, const std::vector<expr> &iter_vars, int lanes,
        std::unordered_map<expr, std::pair<expr, expr>> &conditions,
        int &last_axis_mask);
void compute_block_elemwise(const std::vector<const tensor_slice *> &src,
        const tensor_slice &dst, sc_op_info_t &info,
        fusion_compute_func_t compute);

std::vector<int> transform_axis_plain2blocking(
        const logical_tensor_t &lt, const std::vector<int> &plain_axis);

std::vector<int> transform_axis_plain2blocking(
        const graph_tensor_ptr &gt, const std::vector<int> &plain_axis);

std::vector<int> transform_axis_blocking2plain(
        const logical_tensor_t &lt, const std::vector<int> &blocking_axis);

std::string fusion_create_var_idx();
std::string fusion_create_idx();

void create_fusible_output_anchor(stmt &parent, const tensor_slice &dst,
        const std::vector<expr> &loop_vars,
        const std::vector<int> &anchor_pos_in_loop,
        const vectorized_info_t &vx_info, any_map_t &attrs);

cmp_res cmp_slice_range(const slice_range_list &left_slice_range_list,
        const slice_range_list &right_slice_range_list);

// workload penalty coefficient for transpose/reorder measured by
// for(i, 0, 128){
//     for(j, 0, 256){
//         B[j, i] = A[i, j];
//     }
// }
// TODO(xxx): currently we mark this penalty on op, we will add loop analysis
// pass for tensor sequential access analysis in future
static constexpr size_t workload_penalty_coefficient = 16UL;

float evaluate_loop_parallel_balance(const std::vector<for_loop> &loops);
expr cast_to_s32(const expr &in);
// compare expr in slice equal or not, constant slice may have different
// datatypes but same value as we use `int` for static.
bool slice_expr_equals(const expr &in1, const expr &in2);
} // namespace sc

#endif

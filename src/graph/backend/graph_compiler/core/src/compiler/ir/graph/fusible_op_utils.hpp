/*******************************************************************************
 * Copyright 2020-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_UTILS_HPP

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "fusible_op.hpp"
#include "fusion_data.hpp"
#include "util/variant.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

enum class cmp_res : int {
    unknown = -1,
    equal = 0,
    l_less_r = 1,
    l_larger_r = 2,
};

using slice_range_map = std::unordered_map<int, slice_range_list>;
slice_range_map search_known_input_slice(sc_op *cur, fslice_map &fsmap);
void set_unknown_input_slice(fusible_op_t *cur,
        const slice_range_map &known_ranges_map, fslice_map &fsmap);
infer_status_code infer_binary_slice_ranges(
        fusible_op_t *cur, fslice_map &fsmap);

std::unordered_map<int, binding_axis> search_known_input_axis(
        sc_op *cur, binding_axis_map &bdax_map);
void set_unknown_binding_axis(sc_op *cur,
        const std::unordered_map<int, binding_axis> &known_axis_map,
        binding_axis_map &bdax_map);
void call_output_user_axis_binding(sc_op *cur, binding_axis_map &bdax_map);
void infer_identical_binding_axis(
        fusible_op_t *cur, binding_axis_map &bdax_map);
void pre_infer_identical_binding_axis(
        fusible_op_t *cur, binding_axis_map &bdax_map);

sc_dims get_expr_to_dims(const std::vector<expr> &dims);
size_t get_dims_product(const sc_dims &dims);
// the dim can be squeezed is 1
int get_number_of_squeeze_dims(const sc_dims &dims);

bool range_from_outer_loop(const std::pair<expr, expr> &range);

bool slice_full_on_axis(const sc_dims &dim, const slice_range &ranges,
        const std::vector<int> &axis);

bool slice_divisible_on_axis(const sc_dims &dim, const slice_range &ranges,
        const std::vector<int> &axis);

bool slice_divisible_by_factor(const slice_range &ranges,
        const std::vector<int> &axis, const int factor);

bool slice_larger_than_bound_on_axis(const slice_range &ranges,
        const std::vector<int> &axis, const int factor, const int lower_bound);

int get_slice_size(const slice_range &ranges, const int dtype_size = 1);

inline uint16_t vectorize_step(const context_ptr &ctx, sc_data_etype detype) {
    // eg: bf16 or s8u8 always promote to f32 or s32 to do calculation, we need
    // to limited bf16 max lanes is 8 under avx2 environment.
    auto avx2_lanes_require_dtype = [](const sc_data_etype detype) {
        return detype == sc_data_etype::BF16 || detype == sc_data_etype::S8
                || detype == sc_data_etype::U8;
    };
    if (!ctx->machine_.cpu_flags_.fAVX512F
            && avx2_lanes_require_dtype(detype)) {
        assert(ctx->machine_.cpu_flags_.fAVX2);
        return std::min(uint16_t(8), ctx->get_max_vector_lanes(detype));
    }
    return std::min(uint16_t(16), ctx->get_max_vector_lanes(detype));
}
bool loop_can_be_fused(const for_loop &loop);

// use new fusion mgr to lowering single fusible op
ir_module_ptr fusible_op_get_func(fusible_op_t *op, const context_ptr &ctx);

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
bool is_op_input_blocking_shape(const sc_op_info_t &info);

void compute_vectorized_op(const context_ptr &ctx, sc_graph_t &graph,
        const std::vector<const tensor_slice *> &src, const tensor_slice &dst,
        sc_op_info_t &info, const vectorized_info_t &vx_info,
        const mask_compute_func_t &compute_lanes,
        const mask_compute_func_t &compute_scalar, any_map_t &attrs,
        const graph_tensor_ptr &expand_gt, size_t wkld = 0UL,
        bool use_mask = false,
        const tensor_slice *expand_loop_by
        = nullptr /*by default expand loop by dst*/,
        bool unroll_inner_loop = false);
expr make_select_by_mask(const expr &, const expr &, const expr &, uint32_t);
expr generate_mask_var_by_step(stmt &mask_def, const expr &cur_step,
        int32_t step, const expr &sup_condition = expr(),
        bool direct_sup_cond = false);
expr generate_mask_by_step_directly(const expr &cur_step, int32_t step,
        const expr &sup_condition = expr(), bool direct_sup_cond = false);
expr calculate_mask_cur_step(
        const expr &len, const expr &iter_var, const int32_t lanes);
expr indexing_from_diff_cond(const bool is_last_dim_1, const bool has_tail,
        const tensor_slice &input, std::vector<expr> &input_idx,
        const int32_t lanes, expr &res_idx, const expr &axis_len,
        const expr &iter_var, const expr &floor, bool just_tail_part = false);
expr last_dim_generate_mask(const expr &iter_var, const expr &floor,
        expr const &last_dim_len, int const &lanes,
        bool just_tail_part = false);
void vec_backend_require(const context_ptr &ctx, bool &use_vectorized);
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

cmp_res cmp_slice_range(const slice_range_list &left_slice_range_list,
        const slice_range_list &right_slice_range_list);

bool is_dynamic_slice_range_list(const slice_range_list &in_slice_range_list);

// workload penalty coefficient for transpose/reorder measured by
// for(i, 0, 128){
//     for(j, 0, 256){
//         B[j, i] = A[i, j];
//     }
// }
// TODO(xxx): currently we mark this penalty on op, we will add loop
// analysis pass for tensor sequential access analysis in future
static constexpr size_t workload_penalty_coefficient = 16UL;

float evaluate_loop_parallel_balance(const std::vector<for_loop> &loops,
        bool check_use_full_threads = false);
// return static loop parallelism coefficient to satisfy the parallelism and
// the related condition expr.
float evaluate_loop_parallel_balance(const std::vector<for_loop> &loops,
        expr &cond, bool check_use_full_threads = false);
expr cast_to_s32(const expr &in);
// compare expr in slice equal or not, constant slice may have different
// datatypes but same value as we use `int` for static.
bool slice_expr_equals(const expr &in1, const expr &in2);
// sort op inputs by layout input index, the order is [layout input index,
// others...]
std::vector<graph_tensor_ptr> get_sorted_inputs_by_layout_input(
        const sc_op_ptr &op);

/**
 * @return Bool: return if given slice comes from the inner most anchor with non
 * dividable lanes
 * @param ctx: context, used to query max lanes
 * @param slice: given slice range
 * @param dtype: data type kind, used to query max lanes
 * @param floor: if return is True, recording `floor` value for last dims
 * divided by max lanes
 * @param tail: if return is True, recording `tail` value for last dims divided
 * by max lanes
 * */
bool innermost_slice_with_non_dividable_lanes(const context_ptr &ctx,
        const slice_range &slice, const sc_data_type_t &dtype, sc_dim &floor,
        sc_dim &tail);

variant<float, int64_t> numeric_limits_minimum(sc_data_etype type_code);
variant<float, int64_t> numeric_limits_maximum(sc_data_etype type_code);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

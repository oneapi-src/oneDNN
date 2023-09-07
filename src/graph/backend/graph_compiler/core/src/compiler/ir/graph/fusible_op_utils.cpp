/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include <assert.h>
#include <atomic>
#include <unordered_map>

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include "fusible_op.hpp"
#include "fusion_mgr.hpp"
#include "outer_loop_generator.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/cpu/local_tensor_lower.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <runtime/config.hpp>
#include <util/optional_find.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// incrementor for loop
static std::atomic<int> idx = {0};
// incrementor for temp var
static std::atomic<int> var_idx = {0};
// helper function to get current var_idx
std::string fusion_create_var_idx() {
    return std::string("_") + std::to_string(var_idx++);
}

std::string fusion_create_idx() {
    return std::string("_") + std::to_string(idx++);
}

static std::vector<tensor_slice *> convert_t(std::vector<tensor_slice> &src) {
    std::vector<tensor_slice *> dst(src.size());
    std::transform(src.begin(), src.end(), dst.begin(),
            [](tensor_slice &t) { return &t; });
    return dst;
}

static std::vector<const tensor_slice *> convert_const_t(
        std::vector<tensor_slice> &src) {
    std::vector<const tensor_slice *> dst(src.size());
    std::transform(src.begin(), src.end(), dst.begin(),
            [](tensor_slice &t) { return &t; });
    return dst;
}

static std::vector<tensor_slice> make_tensor_slice(sc_graph_t &graph,
        const std::vector<graph_tensor_ptr> &data,
        const std::string &tensor_name, std::vector<expr> &flattened) {
    std::vector<tensor_slice> expected;
    for (size_t i = 0; i < data.size(); ++i) {
        std::vector<expr> dims
                = data[i]->details_.get_blocking_dims_expr(graph);
        std::vector<expr> strides = dims_to_dense_stride(dims);
        expr aexpr = builder::make_stensor(tensor_name + std::to_string(i),
                dims, strides, data[i]->details_.dtype_);
        flattened.emplace_back(aexpr);
        expected.emplace_back(tensor_slice(aexpr));
    }
    return expected;
}

ir_module_ptr fusible_op_get_func(fusible_op_t *op, outer_loop_generator_t &gen,
        const context_ptr &ctx, bool check_parallel) {
    fusion_manager fmgr;
    fmgr.get_graph().sync_dynamic_info_with_graph(op->get_owner_graph());
    std::vector<graph_tensor_ptr> ins;
    std::vector<graph_tensor_ptr> outs;
    for (auto &in : op->get_inputs()) {
        ins.emplace_back(fmgr.make<input_op>(in->details_)->get_outputs()[0]);
    }
    for (auto &out : op->get_outputs()) {
        outs.emplace_back(
                std::make_shared<graph_tensor>(nullptr, out->details_));
    }
    auto copyable = op->dyn_cast<op_traits::copyable_t>();
    COMPILE_ASSERT(
            copyable, "The fusible op should be copyable: " << op->op_name_);
    auto copied = copyable->copy(ins, outs, fmgr.get_graph());
    copied->info_.cur_impl_ = op->info_.cur_impl_;
    COMPILE_ASSERT(copied->get_outputs().size() == 1,
            "Currently only support 1 output only");
    fmgr.make<output_op>(copied->get_outputs()[0]);
    auto base_idx = gen.get_base_tsr_idx();
    fmgr.put_input_first(
            fmgr.get_graph().get_input_ops()[base_idx]->dyn_cast<input_op>());
    return lower_fusion_manager(ctx, &gen, op, &fmgr, check_parallel);
}

ir_module_ptr fusible_op_get_func(fusible_op_t *op, const context_ptr &ctx) {
    sc_graph_t g;
    g.sync_dynamic_info_with_graph(op->get_owner_graph());
    if (op->get_owner_graph().attrs_.get_or_else("temp.force_static", false)) {
        g.attrs_.set("temp.force_static", true);
    }
    if (op->get_owner_graph().is_dynamic()) {
        g.attrs_.set("temp.parent_graph_dynamic", true);
    }
    std::vector<graph_tensor_ptr> ins;
    std::vector<graph_tensor_ptr> outs;
    for (auto &in : op->get_inputs()) {
        ins.emplace_back(std::make_shared<graph_tensor>(nullptr, in->details_));
    }
    for (auto &out : op->get_outputs()) {
        outs.emplace_back(
                std::make_shared<graph_tensor>(nullptr, out->details_));
    }
    g.make_input(ins);
    auto copyable = op->dyn_cast<op_traits::copyable_t>();
    COMPILE_ASSERT(
            copyable, "The fusible op should be copyable: " << op->op_name_);
    auto copied = copyable->copy(ins, outs, g);
    copied->info_.cur_impl_ = op->info_.cur_impl_;
    COMPILE_ASSERT(copied->get_outputs().size() == 1,
            "Currently only support 1 output only");
    g.make_output(outs);
    g.attrs_.set(mixed_partition_hint::single_op_graph, true);
    // create dummy parti
    auto parti = std::make_shared<mixed_parti_t>(ctx,
            std::const_pointer_cast<sc_op>(op->shared_from_this()), nullptr);
    // create graph-to-original ops maping
    std::unordered_map<sc_op_ptr, sc_op_ptr> graph2orig_ops
            = {{copied, op->shared_from_this()}};
    // try optimize partition
    if (!op->attrs_.get_or_else("temp.no_optimize_op", false)
            && try_optimize_parti(parti.get(), g, graph2orig_ops)) {
        // redo partition
        std::vector<mixed_parti_t::ptr> op2parti(g.ops_.size());
        do_partition(ctx, g, op2parti);
        // collect legal partition
        auto res = collect_parti_set(op2parti, false);
        // Expect only one partition found
        COMPILE_ASSERT(res.size() == 1,
                "Only sinlge partition is expected, but got " << res.size());
        // reset new partition
        parti = res[0];
        // validate optimization
        if (!parti->validate_optimization()) {
            // redo without optimization
            op->attrs_.set("temp.no_optimize_op", true);
            auto ret = fusible_op_get_func(op, ctx);
            // remove temp attr
            op->attrs_.remove("temp.no_optimize_op");
            return ret;
        }
    } else {
        parti = std::make_shared<mixed_parti_t>(ctx,
                std::const_pointer_cast<sc_op>(copied->shared_from_this()),
                std::make_shared<op_dep_matrix_t>(g));
    }
    auto mx_op = parti->transform_to_mixed_op();
    mx_op->set_owner_graph(&g);
    // copy logigcal id
    mx_op->logical_op_id_ = op->logical_op_id_;
    return mx_op->get_func(ctx);
}

sc_dims get_expr_to_dims(const std::vector<expr> &dim) {
    sc_dims dim_int;
    dim_int.reserve(dim.size());
    for (const expr &d : dim) {
        auto cd = do_cast_and_fold(d);
        COMPILE_ASSERT(cd.isa<constant_c>(), "non-constant value found.");
        dim_int.emplace_back(get_const_as_int(cd.static_as<constant_c>()));
    }
    return dim_int;
}

stmt mask_compute_func_t::operator()(const std::vector<expr> &in,
        std::vector<expr::lvalue_proxy_t> &out, const expr &cur_idx,
        const expr &upper_bound, uint32_t lanes) const {
    auto ret = impl_(in, out);
    if (cur_idx.defined() && upper_bound.defined()) {
        auto bld = builder::get_current_builder();
        bld->emit(ret);
        return builder::make_assign_unattached(out[0],
                make_select_by_mask(out[0], cur_idx, upper_bound, lanes));
    }
    return ret;
}

expr make_select_by_mask(const expr &lhs_vec, const expr &cur_index,
        const expr &upper_bound, uint32_t lanes) {
    sc_data_type_t var_dtype;
    auto bld = builder::get_current_builder();
    auto upper_bound_int = builder::make_cast(datatypes::s32, upper_bound);
    auto cur_index_int = builder::make_cast(datatypes::s32, cur_index);
    int step = static_cast<int>(lanes);
    // upper_bound - cur_index
    auto cur_step = builder::make_min(
            builder::make_max(
                    builder::make_sub(upper_bound_int, cur_index_int), 0),
            step);
    stmt mask_def;
    auto mask = generate_mask_var_by_step(mask_def, cur_step, step);
    bld->emit(mask_def);
    expr rhs_vec = make_expr<constant_node>(
            std::vector<union_val>(lanes, UINT64_C(0)),
            sc_data_type_t(lhs_vec->dtype_.type_code_, lanes));
    return builder::make_select(mask, lhs_vec, rhs_vec);
}

void choose_mask_vartype_init_value(
        sc_data_type_t &var_dtype, uint64_t &init_value, int32_t step) {
    switch (step) {
        case 4: {
            var_dtype = datatypes::u8;
            init_value = 15;
            break;
        }
        case 8: {
            var_dtype = datatypes::u8;
            init_value = std::numeric_limits<uint8_t>::max();
            break;
        }
        case 16: {
            var_dtype = datatypes::u16;
            init_value = std::numeric_limits<uint16_t>::max();
            break;
        }
        case 32: {
            var_dtype = datatypes::u32;
            init_value = std::numeric_limits<uint32_t>::max();
            break;
        }
        case 64: {
            var_dtype = datatypes::index;
            init_value = std::numeric_limits<uint64_t>::max();
            break;
        }
        default: COMPILE_ASSERT(false, "invalid lanes: " << step);
    }
}

expr calculate_mask_cur_step(
        const expr &len, const expr &iter_var, const int32_t lanes) {
    auto last_axis_offset = cast_to_s32(len) - cast_to_s32(iter_var);
    // mask = min(max(0, last_dim_len -
    // last_dim_idx),real_step) To choose [0 ~
    // step] mask
    return builder::make_min(
            builder::make_max(builder::make_constant(0), last_axis_offset),
            lanes);
}

// generate mask = var < floor ? 0b1111 : 0b00..111;
expr last_dim_generate_mask(const expr &iter_var, const expr &floor,
        expr const &last_dim_len, int const &lanes, bool just_tail_part) {
    // just_tail_part means that the floor and tail parts of the for loop are
    // calculated separately. Only the tail part needs to calculate the mask.
    auto s32_var = cast_to_s32(iter_var);
    auto s32_floor = cast_to_s32(floor);
    auto s32_dim_len = cast_to_s32(last_dim_len);
    expr condition = s32_var < s32_floor;
    expr tail_len = lanes + s32_var - s32_dim_len;
    sc_data_type_t var_dtype;
    uint64_t init_value;
    choose_mask_vartype_init_value(var_dtype, init_value, lanes);
    auto full_mask = builder::make_constant({init_value}, var_dtype);
    if (floor.isa<constant>()) {
        int floor_int = get_expr_as_int(floor);
        int dim_len = get_expr_as_int(last_dim_len);
        int res_mask = init_value >> (lanes + floor_int - dim_len);

        return just_tail_part ? builder::make_cast(var_dtype, res_mask)
                              : builder::make_select(condition, full_mask,
                                      builder::make_cast(var_dtype, res_mask));
    }
    return just_tail_part
            ? (full_mask >> builder::make_cast(var_dtype, tail_len))
            : builder::make_select(condition, full_mask,
                    (full_mask >> builder::make_cast(var_dtype, tail_len)));
}

expr generate_mask_var_by_step(stmt &mask_def, const expr &cur_step,
        int32_t step, const expr &sup_condition) {
    // notice: cur_step must be s32
    sc_data_type_t var_dtype;
    uint64_t init_value;
    choose_mask_vartype_init_value(var_dtype, init_value, step);
    auto mask_select
            = generate_mask_by_step_directly(cur_step, step, sup_condition);
    auto mask = builder::make_var(
            var_dtype, "__mask_" + std::to_string(var_idx++));
    mask_def = builder::make_var_tensor_def_unattached(
            mask, linkage::local, mask_select);
    return mask;
}

expr generate_mask_by_step_directly(
        const expr &cur_step, int32_t step, const expr &sup_condition) {
    // notice: cur_step must be s32
    sc_data_type_t var_dtype;
    uint64_t init_value;
    choose_mask_vartype_init_value(var_dtype, init_value, step);
    auto full_mask = builder::make_constant({init_value}, var_dtype);
    auto empty_mask = builder::make_constant({UINT64_C(0)}, var_dtype);
    auto empty_mask_condition = (sup_condition.defined())
            ? (cur_step == 0 || !sup_condition)
            : (cur_step == 0);
    return builder::make_select(empty_mask_condition, empty_mask,
            builder::make_select(cur_step == step, full_mask,
                    full_mask
                            >> builder::make_cast(var_dtype, step - cur_step)));
}

/** Determine whether masks are needed during elementwise computation and
 * generate conditional expressions for the mask
 * @param graph the graph
 * @param src input slice
 * @param plain_dims plain shapes
 * @param format input format
 * @param iter_vars input loop vars
 * @param lanes simd lanes
 * @param condition key is related iter var, value is two exprs: first is
 * current accumulated index, second is its plain shape upperbound.
 * @param last_axis_mask mask count, how many elements should be computed in
 * this time. -1 means all.
 * */
void compute_mask_and_generate_condition(sc_graph_t &graph,
        const std::vector<const tensor_slice *> &src, const sc_dims &plain_dims,
        sc_data_format_t format, const std::vector<expr> &iter_vars, int lanes,
        std::unordered_map<expr, std::pair<expr, expr>> &conditions,
        int &last_axis_mask) {
    auto blocking_dims
            = sc_data_format_t::get_blocking_shapes(plain_dims, format);
    auto padded_dims
            = sc_data_format_t::get_padded_plain_shapes(blocking_dims, format);
    auto &format_code = format.format_code_;
    if (plain_dims == padded_dims) { return; }
    auto offset = src[0]->get_offset();
    auto shapes = src[0]->get_shape();
    size_t ndims = format_code.ndims();
    assert(offset.size() == ndims && shapes.size() == ndims
            && iter_vars.size() == ndims);
    auto plain2block = format_code.collect_p2b_mapping();
    for (size_t i = 0; i < plain2block.size(); i++) {
        auto &orig_dim = i;
        if (plain_dims[orig_dim] == padded_dims[orig_dim]
                || plain2block[i].size() == 1) {
            continue;
        }
        auto &block_dim = plain2block[i][plain2block[i].size() - 1];
        auto blocks = format_code.collect_blocking_index(orig_dim);
        int padding_count = 0;
        conditions[iter_vars[block_dim]].first
                = iter_vars[block_dim] + offset[block_dim];
        for (int b = static_cast<int>(blocks.size()) - 1; b >= 0; b--) {
            if (b > 0 && blocks[b - 1] % blocks[b] != 0) { padding_count++; }
            conditions[iter_vars[block_dim]].first
                    = conditions[iter_vars[block_dim]].first
                    + (iter_vars[plain2block[i][b]] + offset[plain2block[i][b]])
                            * format.blocks_[blocks[b]];
        }
        conditions[iter_vars[block_dim]].second
                = graph.dim_to_expr(plain_dims[orig_dim]);
        COMPILE_ASSERT(padding_count < 2,
                "Currently we don't support multi-level padding mask.");
        if (block_dim == format_code.ndims() - 1) {
            assert(lanes > 1);
            last_axis_mask = plain_dims[orig_dim] % lanes;
        }
    }
}

void create_fusible_output_anchor(std::vector<stmt> &parent,
        const tensor_slice &dst, const std::vector<expr> &loop_vars,
        const std::vector<int> &anchor_pos_in_loop,
        const vectorized_info_t &vx_info, any_map_t &attrs) {
    if (attrs.has_key(op_attr_key::inner_anchor)) {
        // insert inner anchor (cache-level)
        auto tsr = dst.get_real_tensor();
        auto range = dst.get_ranges();
        if (range.size() != loop_vars.size()) return;
        COMPILE_ASSERT(std::all_of(anchor_pos_in_loop.begin(),
                               anchor_pos_in_loop.end(),
                               [&loop_vars](int pos) {
                                   return pos >= 0
                                           && pos <= static_cast<int>(
                                                      loop_vars.size());
                               }),
                "Could not create fusible output anchor at loop position: "
                        << utils::print_vector(anchor_pos_in_loop)
                        << ", due to only " << loop_vars.size()
                        << " loops found")
        // reset offset
        for (size_t j = 0; j < loop_vars.size(); j++) {
            if (anchor_pos_in_loop.end()
                    != std::find(anchor_pos_in_loop.begin(),
                            anchor_pos_in_loop.end(), static_cast<int>(j)))
                continue;
            if (!range[j].second.isa<constant>()) return;
            if (get_expr_as_int(range[j].second) == 1) continue;
            if (!range[j].first.isa<constant>()) return;
            range[j].first = loop_vars[j];
            range[j].second = ((static_cast<int>(j) == vx_info.axis)
                            ? expr(int(vx_info.lanes))
                            : expr(1));
        }
        auto s = make_stmt<stmts_node_t>(std::vector<stmt> {});
        auto fanchor = fuse_anchor_t(s,
                std::make_pair(std::vector<tensor_slice> {tensor_slice(
                                       tsr, std::move(range))},
                        std::vector<tensor_slice> {}));
        // redirect gen_fanchor
        attrs[op_attr_key::inner_anchor] = fanchor;
        parent.emplace_back(s);
    }
}

void create_fusible_output_anchor(stmt &parent, const tensor_slice &dst,
        const std::vector<expr> &loop_vars,
        const std::vector<int> &anchor_pos_in_loop,
        const vectorized_info_t &vx_info, any_map_t &attrs) {
    std::vector<stmt> ss = parent.isa<stmts>() ? parent.static_as<stmts>()->seq_
                                               : std::vector<stmt> {parent};
    create_fusible_output_anchor(
            ss, dst, loop_vars, anchor_pos_in_loop, vx_info, attrs);
    parent = make_stmt<stmts_node_t>(std::move(ss));
}

/** Get indexing based on different conditions
 * @param is_lastdim_meet_require whether the shape len >= threshold
 * @param has_tail whether has tail
 * @param input input slice
 * @param input_idx input index
 * @param lanes simd lanes
 * @param res_idx the final indexing we want to get
 * @param axis_len length of indexing axis
 * @param iter_var var of indexing axis
 * @param just_tail_part floor part and tail part is calculated separately
 * */
expr indexing_from_diff_cond(const bool is_lastdim_meet_require,
        const bool has_tail, const tensor_slice &input,
        std::vector<expr> &input_idx, const int32_t lanes, expr &res_idx,
        const expr &axis_len, const expr &iter_var, const expr &floor,
        bool just_tail_part) {
    if (is_lastdim_meet_require) {
        res_idx = builder::make_indexing(input.tptr_, input_idx);
    } else if (has_tail && utils::is_one_of(lanes, 4, 8, 16, 32, 64)) {
        auto mask = last_dim_generate_mask(
                iter_var, floor, axis_len, lanes, just_tail_part);
        res_idx = builder::make_indexing(input.tptr_, input_idx, lanes, mask);
    } else {
        res_idx = builder::make_indexing(input.tptr_, input_idx, lanes);
    }
    return res_idx;
}

// Whether all the input shapes are blocking formats.
bool is_op_input_blocking_shape(const sc_op_info_t &info) {
    return std::all_of(info.inputs_.begin(), info.inputs_.end(),
            [](const graph_tensor_ptr &in) {
                return in->details_.get_format().is_blocking();
            });
}

void vec_backend_require(const context_ptr &ctx, bool &use_vectorized) {
// llvm and g++ will perform special optimization on the scalar version,
// resulting in the performance of our vectorized version not being as good
// as the scalar version. Currently, these two backends still maintain the
// scalar method of tail processing. The builtin will use our vectorized
// version.
#if SC_BUILTIN_JIT_ENABLED
    if (ctx->flags_.jit_kind_ == jit_kind::xbyak) {
        use_vectorized = true;
    } else {
        use_vectorized = false;
    }
#else
    use_vectorized = false;
#endif
}

void compute_vectorized_op(const context_ptr &ctx, sc_graph_t &graph,
        const std::vector<const tensor_slice *> &src, const tensor_slice &dst,
        sc_op_info_t &info, const vectorized_info_t &vx_info,
        const mask_compute_func_t &compute_lanes,
        const mask_compute_func_t &compute_scalar, any_map_t &attrs,
        size_t wkld, bool use_mask, const tensor_slice *expand_loop_by,
        bool unroll_inner_loop) {
    if (!expand_loop_by) { expand_loop_by = &dst; }
    bool use_vectorized = false;
    vec_backend_require(ctx, use_vectorized);
    // In order to support non-stride test, we add dense_stride flag.
    // If it is non-stride shape, we just use step = 1 to do
    // this.
    int graph_input_size = info.inputs_.size();
    bool dense_stride = std::all_of(info.inputs_.begin(), info.inputs_.end(),
            [](const graph_tensor_ptr &in) { return in->details_.is_dense(); });
    bool is_blocking_shape = is_op_input_blocking_shape(info);
    // nested loop vars
    std::vector<expr> iter_vars;
    // the indices for multiple inputs. First dim: the input, Second dim:
    // the dimemsions in the tensor
    std::vector<std::vector<expr>> src_indices_floor(src.size());
    std::vector<std::vector<expr>> src_indices_tail(src.size());
    // the indices for the output tensor
    std::vector<expr> dst_idx_floor;
    std::vector<expr> dst_idx_tail;
    for (unsigned i = 0; i < expand_loop_by->nslice_dims(); i++) {
        // make the loop var for the for-loop
        iter_vars.emplace_back(
                range_from_outer_loop(expand_loop_by->get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_fuseiter")
                                        + std::to_string(idx++)));
        // for each input tensor
        for (size_t j = 0; j < src.size(); j++) {
            auto &src_idx_floor = src_indices_floor.at(j);
            auto &src_idx_tail = src_indices_tail.at(j);
            // push an index
            src_idx_floor.emplace_back(iter_vars.back());
            src_idx_tail.emplace_back(iter_vars.back());
        }
        // push an index for output tensor
        dst_idx_floor.emplace_back(iter_vars.back());
        dst_idx_tail.emplace_back(iter_vars.back());
    }
    auto tail_var = builder::make_var(
            datatypes::index, std::string("_fuseiter") + std::to_string(idx++));
    for (size_t j = 0; j < src.size(); j++) {
        auto &src_idx_tail = src_indices_tail.at(j);
        src_idx_tail[vx_info.axis] = tail_var;
    }
    dst_idx_tail[vx_info.axis] = tail_var;
    expr indexed_target_floor;

    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    int lanes = static_cast<int>(vx_info.lanes);
    auto slice_len = expand_loop_by->get_shape().at(vx_info.axis);
    auto floor = do_cast_and_fold(slice_len / lanes * lanes);
    auto tail = do_cast_and_fold(slice_len % lanes);
    int floor_int = 0;
    int tail_int = 0;
    if (floor.isa<constant>()) {
        floor_int = get_expr_as_int(floor);
        tail_int = get_expr_as_int(tail);
        COMPILE_ASSERT((floor_int + tail_int), "Don't support shape len = 0.");
    }
    const int INVALID_AXIS_MASK = -64;
    int last_axis_mask = INVALID_AXIS_MASK;
    std::unordered_map<expr, std::pair<expr, expr>> conditions;
    if (use_mask) {
        compute_mask_and_generate_condition(graph, src,
                info.inputs_[0]->details_.get_plain_dims(),
                info.inputs_[0]->details_.get_format(), iter_vars,
                vx_info.lanes, conditions, last_axis_mask);
    }
    if (last_axis_mask != INVALID_AXIS_MASK && floor_int > 0) {
        COMPILE_ASSERT(tail_int == 0,
                "Currently we only support mask in vectorize compute not "
                "tail.");
    }
    std::vector<stmt> tcur;
    stmt cur;
    int loop_size = static_cast<int>(expand_loop_by->get_shape().size());
    bool tail_threshold = tail.isa<constant>() && tail_int <= 1;
    bool use_scalar
            = !use_vectorized || tail_threshold || lanes == 1 || !dense_stride;
    // recover schedule loop
    for (int i = loop_size - 1; i >= 0; i--) {
        stmt body;
        // currently vx_axis should be last axis
        if (loop_size == vx_info.axis + 1 && i == vx_info.axis) {
            if (dense_stride && (!floor.isa<constant>() || floor_int)) {
                bld->push_scope();
                // if the shape is less than lanes, we don't use mask to
                // process.

                indexing_from_diff_cond(false, false, dst, dst_idx_floor, lanes,
                        indexed_target_floor, slice_len, iter_vars.at(i),
                        floor);
                std::vector<expr> indexed_input_floor;
                expr input_floor_idx;
                for (unsigned j = 0; j < src.size(); j++) {
                    indexed_input_floor.emplace_back(indexing_from_diff_cond(
                            false, false, *src.at(j), src_indices_floor.at(j),
                            lanes, input_floor_idx, slice_len, iter_vars.at(i),
                            floor));
                }
                std::vector<expr::lvalue_proxy_t> target_floor
                        = {expr::lvalue_proxy_t(indexed_target_floor, false)};
                auto cond_it = conditions.find(iter_vars[i]);
                if (cond_it != conditions.end()) {
                    assert(last_axis_mask != INVALID_AXIS_MASK);
                    cur = compute_lanes(indexed_input_floor, target_floor,
                            cond_it->second.first, cond_it->second.second,
                            vx_info.lanes);
                } else {
                    cur = compute_lanes(indexed_input_floor, target_floor);
                }
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                stmt s = bld->pop_scope();
                auto ss = std::vector<stmt> {s};
                if (!tail_int)
                    create_fusible_output_anchor(
                            ss, dst, iter_vars, {i + 1}, vx_info, attrs);
                cur = ss.size() > 1 ? make_stmt<stmts_node_t>(std::move(ss))
                                    : s;
                if (iter_vars.at(i).isa<var>()) {
                    cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                            floor, expr(lanes), cur, true, for_type::NORMAL);
                    if (unroll_inner_loop) {
                        cur->attr()[stmt_attr_key::unroll_loop] = 0;
                    }
                }
                tcur.emplace_back(cur);
            }
            if (((!tail.isa<constant>() && !is_blocking_shape) || tail_int)
                    || !dense_stride) {
                bld->push_scope();

                std::vector<expr> indexed_input_tail;
                expr mask;
                if (!use_scalar) {
                    mask = last_dim_generate_mask(
                            tail_var, floor, slice_len, lanes, true);
                }
                expr indexed_target_tail = builder::make_indexing(
                        dst.tptr_, dst_idx_tail, use_scalar ? 1 : lanes, mask);
                for (unsigned j = 0; j < src.size(); j++) {
                    indexed_input_tail.emplace_back(builder::make_indexing(
                            src.at(j)->tptr_, src_indices_tail.at(j),
                            use_scalar ? 1 : lanes, mask));
                }
                std::vector<expr::lvalue_proxy_t> target_tail
                        = {expr::lvalue_proxy_t(indexed_target_tail, false)};
                if (use_scalar) {
                    cur = compute_scalar(indexed_input_tail, target_tail);
                } else {
                    cur = compute_lanes(indexed_input_tail, target_tail);
                }
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(tail_var,
                        !dense_stride ? expr(0) : floor, slice_len,
                        use_scalar ? expr(1) : expr(lanes), bld->pop_scope(),
                        true, for_type::NORMAL);
                if (unroll_inner_loop) {
                    cur->attr()[stmt_attr_key::unroll_loop] = 0;
                }
                tcur.emplace_back(cur);
                // create fusible output anchor as demand
                std::vector<int> anchor_pos_in_loop(1);
                anchor_pos_in_loop.emplace_back(i);
                create_fusible_output_anchor(
                        tcur, dst, iter_vars, {i}, vx_info, attrs);
            }
        } else if (iter_vars.at(i).isa<var>()) {
            // Do not generate those dummy loop
            if (!tcur.empty() && tcur[0].defined()) {
                body = make_stmt<stmts_node_t>(std::move(tcur));
                tcur.clear();
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), expand_loop_by->get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else if (cur.defined()) {
                body = make_stmt<stmts_node_t>(
                        std::vector<stmt> {std::move(cur)});
                // address special condition, like temp_buffer is used
                cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)),
                        expr(0), expand_loop_by->get_shape().at(i), expr(1),
                        std::move(body), true, for_type::NORMAL);
            } else {
                // if cur not defined, means last axis of tensor slice has
                // range 1, e.g. tensor_slice{{i, 100},{0, 1}}
                indexed_target_floor
                        = builder::make_indexing(dst.tptr_, dst_idx_floor);
                std::vector<expr> indexed_input_floor;
                for (unsigned j = 0; j < src.size(); j++) {
                    indexed_input_floor.emplace_back(builder::make_indexing(
                            src.at(j)->tptr_, src_indices_floor.at(j)));
                }

                std::vector<expr::lvalue_proxy_t> target_floor
                        = {expr::lvalue_proxy_t(indexed_target_floor, false)};
                bld->push_scope();
                cur = compute_scalar(indexed_input_floor, target_floor);
                cur->attr()[op_traits::workload_computable_t::workload_number]
                        = wkld;
                bld->emit(cur);
                cur = make_stmt<for_loop_node_t>(iter_vars.at(i), expr(0),
                        expand_loop_by->get_shape().at(i), expr(1),
                        bld->pop_scope(), true, for_type::NORMAL);
                if (unroll_inner_loop) {
                    cur->attr()[stmt_attr_key::unroll_loop] = 0;
                }
            }
        }
    }
    if (!tcur.empty() && tcur[0].defined()) {
        // TODO(xxx): currenly we don't add merge_loop attribute for this
        // special case, need stronger loop analysis.
        for (auto &it : tcur) {
            bld->emit(it);
        }
        // TODO(yifei): analyze whether this is safe enough
        cur->attr()[stmt_attr_key::merge_loop] = true;
    } else {
        cur->attr()[stmt_attr_key::merge_loop] = true;
        bld->emit(cur);
    }
}

size_t get_dims_product(const sc_dims &dims) {
    sc_dim ret = 1;
    // todo: find out how to use this function in dynamic cases.
    for (unsigned i = 0; i < dims.size(); ++i) {
        if (!is_dynamic_dim(dims[i]) && dims[i]) { ret *= dims[i]; }
    }
    assert(ret > 0 && "Overflow or non-constant shape detected");
    return ret;
}

int get_number_of_squeeze_dims(const sc_dims &dims) {
    int ret = 0;
    for (auto &it : dims) {
        if (it == 1) { ret++; }
    }
    return ret;
}

bool loop_can_be_fused(const for_loop &loop) {
    return get_expr_as_int(loop->step_) == INT64_C(1);
}

slice_range_map search_known_slice_ranges(
        sc_op *cur, fslice_map &fsmap, infer_status_map_t &stat_map) {
    slice_range_map known_ranges_map;
    auto input_size = cur->get_inputs().size();
    for (size_t i = 0; i < input_size; i++) {
        auto &input = cur->get_inputs()[i];
        if (!fsmap.get(input).empty()) {
            known_ranges_map[i] = fsmap.get(input);
        }
    }
    if (known_ranges_map.empty()) {
        stat_map.append_ops_by_status(cur, infer_status_code::UNKNOWN);
    }
    return known_ranges_map;
}

void set_unknown_slice_ranges(fusible_op_t *cur,
        const slice_range_map &known_ranges_map, fslice_map &fsmap,
        infer_status_map_t &stat_map) {
    // set other unknown ranges.
    auto input_size = cur->get_inputs().size();
    for (size_t i = 0; i < input_size; i++) {
        auto input = cur->get_inputs()[i];
        auto &inp_slice = fsmap.get(input);
        if (input->producer_owner_->isa<input_op>()
                && input->producer_owner_->dyn_cast<input_op>()
                           ->is_arg_input()) {
            inp_slice = *utils::find_map_value(known_ranges_map, i).get();
        } else {
            if (inp_slice.empty()) {
                inp_slice = *utils::find_map_value(known_ranges_map, i).get();
                if (!stat_map.is_recursive_mode()) continue;
                if (auto inp_op
                        = input->producer_owner_->dyn_cast<fusible_op_t>()) {
                    inp_op->pre_slice_ranges(fsmap, stat_map);
                }
            }
        }
    }
}

std::unordered_map<int, bound_axis> search_known_bound_axis(
        sc_op *cur, bound_axis_map &bdax_map) {
    std::unordered_map<int, bound_axis> known_axis_map;
    auto input_size = cur->get_inputs().size();
    for (size_t i = 0; i < input_size; i++) {
        auto &input = cur->get_inputs()[i];
        if (!bdax_map.get(input).empty()) {
            known_axis_map[i] = bdax_map.get(input);
        }
    }
    COMPILE_ASSERT(!known_axis_map.empty(),
            "No binded input axis found for " << cur->op_name_
                                              << cur->logical_op_id_)
    return known_axis_map;
}

void call_output_user_axis_binding(sc_op *cur, bound_axis_map &bdax_map) {
    for (auto &out : cur->get_outputs()) {
        for (auto &user : out->uses_) {
            if (auto bd_op = user.second->dyn_cast<
                             op_traits::mixed_partition_acceptable>()) {
                bd_op->infer_binding_axis(bdax_map);
            }
        }
    }
}

void set_unknown_axis_binding(sc_op *cur,
        const std::unordered_map<int, bound_axis> &known_axis_map,
        bound_axis_map &bdax_map) {
    // set other unknown axis.
    auto input_size = cur->get_inputs().size();
    for (size_t i = 0; i < input_size; i++) {
        auto input = cur->get_inputs()[i];
        auto &inp_axis = bdax_map.get(input);
        if (inp_axis.empty()) {
            inp_axis = *utils::find_map_value(known_axis_map, i).get();
            auto producer = input->producer_owner_;
            if (producer->isa<input_op>()) continue;
            if (auto inp_op = producer->dyn_cast<
                              op_traits::mixed_partition_acceptable>()) {
                inp_op->pre_binding_axis(bdax_map);
                // in avoid of more than one users cases
                call_output_user_axis_binding(producer, bdax_map);
            }
        }
    }
    // call output
    call_output_user_axis_binding(cur, bdax_map);
}

std::vector<int> transform_axis_plain2blocking(
        const logical_tensor_t &lt, const std::vector<int> &plain_axis) {
    auto fmt = lt.get_format();
    // If format is any, just return.
    if (fmt.is_any()) { return plain_axis; }
    std::vector<int> real_axis;
    auto p2bmp = fmt.format_code_.collect_p2b_mapping();
    for (auto &i : plain_axis) {
        std::vector<int> res;
        res.resize(p2bmp[i].size());
        std::transform(p2bmp[i].begin(), p2bmp[i].end(), res.begin(),
                [](const int &v) { return v; });
        real_axis.insert(real_axis.end(), res.begin(), res.end());
    }
    std::sort(real_axis.begin(), real_axis.end());
    return real_axis;
}

std::vector<int> transform_axis_plain2blocking(
        const graph_tensor_ptr &gt, const std::vector<int> &plain_axis) {
    return transform_axis_plain2blocking(gt->details_, plain_axis);
}

std::vector<int> transform_axis_blocking2plain(
        const logical_tensor_t &lt, const std::vector<int> &blocking_axis) {
    auto fmt = lt.get_format();
    // If format is any, just return.
    if (fmt.is_any()) { return blocking_axis; }
    std::vector<int> plain_axis;
    auto p2bmp = fmt.format_code_.collect_p2b_mapping();
    for (auto &ax : blocking_axis) {
        for (size_t i = 0; i < p2bmp.size(); i++) {
            auto blk_axis_i = p2bmp[i];
            if (blk_axis_i.end()
                    != std::find(blk_axis_i.begin(), blk_axis_i.end(), ax)) {
                plain_axis.emplace_back(i);
                break;
            }
        }
    }
    // check if empty to make g++12 happy
    if (!plain_axis.empty()) {
        // remove duplicated axis.
        std::sort(plain_axis.begin(), plain_axis.end());
        plain_axis.erase(std::unique(plain_axis.begin(), plain_axis.end()),
                plain_axis.end());
    }
    return plain_axis;
}

bool is_dynamic_slice_range_list(const slice_range_list &in_slice_range_list) {
    for (auto &range : in_slice_range_list) {
        auto shapes = get_slice_shape(range);
        for (auto &shape : shapes) {
            auto folded_shape = do_cast_and_fold(shape);
            if (!folded_shape.isa<constant>()) { return true; }
        }
    }
    return false;
}

/**
 * Compare left and right fsmap
 * */
cmp_res cmp_slice_range(const slice_range_list &left_slice_range_list,
        const slice_range_list &right_slice_range_list) {
    size_t left_slice_size = 0, right_slice_size = 0;
    COMPILE_ASSERT(
            !left_slice_range_list.empty() && !right_slice_range_list.empty(),
            "slice range should be set");
    if (is_dynamic_slice_range_list(left_slice_range_list)
            || is_dynamic_slice_range_list(right_slice_range_list)) {
        auto &left_slice_range = left_slice_range_list[0];
        auto &right_slice_range = right_slice_range_list[0];
        assert(left_slice_range.size() == right_slice_range.size());
        for (size_t i = left_slice_range.size(); i > 0; i--) {
            auto left_shape = do_cast_and_fold(left_slice_range[i - 1].second);
            auto right_shape
                    = do_cast_and_fold(right_slice_range[i - 1].second);
            if (!left_shape->equals(right_shape)
                    && !(left_shape.isa<constant>()
                            && right_shape.isa<constant>()
                            && get_expr_as_int(left_shape)
                                    == get_expr_as_int(right_shape))) {
                if (left_shape.isa<constant>() && right_shape.isa<constant>()) {
                    if (get_expr_as_int(left_shape)
                            > get_expr_as_int(right_shape)) {
                        left_slice_size++;
                    } else {
                        right_slice_size++;
                    }
                } else if (left_shape.isa<constant>()) {
                    right_slice_size++;
                } else if (right_shape.isa<constant>()) {
                    left_slice_size++;
                }
            }
        }
    } else {
        for (auto &left_slice_range : left_slice_range_list) {
            auto left_slice_shape
                    = get_expr_to_dims(get_slice_shape(left_slice_range));
            if (get_dims_product(left_slice_shape) > left_slice_size) {
                left_slice_size = get_dims_product(left_slice_shape);
            }
        }
        for (auto &right_slice_range : right_slice_range_list) {
            auto right_slice_shape
                    = get_expr_to_dims(get_slice_shape(right_slice_range));
            if (get_dims_product(right_slice_shape) > right_slice_size) {
                right_slice_size = get_dims_product(right_slice_shape);
            }
        }
    }
    // if right anchor is more smaller than the leftrent one
    if (left_slice_size == right_slice_size) {
        return cmp_res::equal;
    } else if (left_slice_size < right_slice_size) {
        return cmp_res::l_less_r;
    } else {
        return cmp_res::l_larger_r;
    }
}

bool is_reshaped_tensor(const expr &tsr) {
    COMPILE_ASSERT(tsr.isa<tensorptr>(),
            "except for tensor node, only tensorptr node is expected, but "
            "got " << tsr);
    if (tsr.static_as<tensorptr>()->is_slice_) return false;
    auto base = tsr.static_as<tensorptr>()->base_;
    COMPILE_ASSERT(base.isa<indexing>(),
            "tensor_ptr base should be indexing, but got: " << base);
    for (auto &idx : base.static_as<indexing>()->idx_) {
        if (!idx.isa<constant>() || get_expr_as_int(idx) != 0) return false;
    }
    auto base_tensor = base.static_as<indexing>()->ptr_;
    COMPILE_ASSERT(base_tensor.isa<tensor>(), "Tensor type is expected")
    auto base_dims = base_tensor.static_as<tensor>()->dims_;
    auto new_dims = tsr.static_as<tensorptr>()->shape_;
    return get_dims_product(get_expr_to_dims(base_dims))
            == get_dims_product(get_expr_to_dims(new_dims));
}

static std::vector<expr> get_dense_stride(const std::vector<expr> &shape) {
    std::vector<expr> result(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
        result[i] = result[i + 1] * shape[i + 1];
    }
    return result;
}

expr transform_tsr2stsr_with_range(const expr &tsr, const slice_range &range) {
    auto new_dims = get_slice_shape(range);
    std::vector<expr> new_strides;
    tensor t;
    if (tsr.isa<tensor>()) {
        t = tsr.static_as<tensor>();
        new_strides = t->strides_;
    } else {
        COMPILE_ASSERT(is_reshaped_tensor(tsr),
                "reshaped tensor is expected, but got " << tsr);
        t = tsr.static_as<tensorptr>()
                    ->base_.static_as<indexing>()
                    ->ptr_.static_as<tensor>();
        new_strides = get_dense_stride(tsr.static_as<tensorptr>()->shape_);
    }
    return builder::make_stensor(
            t->name_ + "_strd", new_dims, new_strides, t->elem_dtype_);
}

expr transform_tsl2stsr(const tensor_slice &tsl) {
    return transform_tsr2stsr_with_range(
            tsl.get_real_tensor(), tsl.get_ranges());
}

expr transform_tsr2tptr_with_range(const expr &tsr, const slice_range &range) {
    auto new_dims = get_slice_shape(range);
    return builder::tensor_ptr(
            tsr, get_slice_idx(range), get_slice_shape(range), true);
}

expr transform_tptr2stsr(const expr &tptr) {
    COMPILE_ASSERT(tptr.isa<tensorptr>(),
            "tensort pointer node is expected, but got " << tptr);
    auto tp = tptr.static_as<tensorptr>();
    COMPILE_ASSERT(tp->base_.isa<indexing>(), "indexing node is expected");
    auto tsr = tp->base_->ptr_;
    COMPILE_ASSERT(
            tsr.isa<tensor>(), "tensor node is expected, but got " << tsr);
    auto t = tsr.static_as<tensor>();
    return builder::make_stensor(
            t->name_ + "_strd", tp->shape_, t->strides_, t->elem_dtype_);
}

float evaluate_loop_parallel_balance(
        const std::vector<for_loop> &loops, bool check_use_full_threads) {
    expr dummy;
    return evaluate_loop_parallel_balance(loops, dummy, check_use_full_threads);
}

float evaluate_loop_parallel_balance(const std::vector<for_loop> &loops,
        expr &cond, bool check_use_full_threads) {
    expr dyn_loops = 1;
    sc_dim stc_loops = 1;
    const int run_threads = runtime_config_t::get().get_num_threads();
    cond = false;
    // the minor dyn cost is used for case `dyn_var1 + dyn_var2` and
    // `dyn_var1` only comparison.
    float minor_dyn_parallelism = 0.f;
    const float minor_dyn_parallelism_step = 1e-4f;
    for (size_t i = 0; i < loops.size(); i++) {
        auto &loop = loops[i];
        if (!(loop->iter_begin_.isa<constant_c>()
                    && loop->iter_end_.isa<constant_c>())) {
            dyn_loops = dyn_loops * (loop->iter_end_ - loop->iter_begin_);
            minor_dyn_parallelism += minor_dyn_parallelism_step;
        } else {
            auto begin = get_expr_as_int(loop->iter_begin_),
                 end = get_expr_as_int(loop->iter_end_);
            COMPILE_ASSERT(
                    end > begin, "loop end is expected to larger than begin")
            stc_loops = stc_loops * (end - begin);
            dyn_loops = dyn_loops * static_cast<uint64_t>(stc_loops);
        }
    }
    if (check_use_full_threads) {
        cond = dyn_loops < run_threads;
        return stc_loops >= run_threads;
    }
    cond = !((dyn_loops % run_threads == 0) || (dyn_loops > run_threads * 8));
    if (stc_loops == 1) { return 0.0f + minor_dyn_parallelism; }
    bool parallelism = (stc_loops / run_threads > 8)
            || (stc_loops % run_threads == 0 && stc_loops >= run_threads);
    float cal_parallelism = ((stc_loops % run_threads) / float(run_threads));
    if (stc_loops < run_threads) { cal_parallelism += minor_dyn_parallelism; }
    return parallelism ? 1.0f : cal_parallelism;
}

bool range_from_outer_loop(const std::pair<expr, expr> &range) {
    return !range.first.isa<constant>() && range.second.isa<constant>()
            && get_expr_as_int(range.second) == 1;
}

bool slice_full_on_axis(const sc_dims &dim, const slice_range &ranges,
        const std::vector<int> &axis) {
    for (auto &ax : axis) {
        auto first = do_cast_and_fold(ranges[ax].first);
        auto second = do_cast_and_fold(ranges[ax].second);
        // slice range length should equal to dims
        if (second.isa<constant>()) {
            if (get_const_as_int(second.checked_as<constant>()) != dim[ax]) {
                return false;
            } else if (dim[ax] == 1) {
                continue;
            }
        }
        if (!first.isa<constant>()) {
            if (first->node_type_ == sc_expr_type::mul) {
                auto rv = constant_folding::get_operand_from_binary(first)
                                  .second;
                // {i * block, block} case where `block_size==dims[i]`
                if (rv.isa<constant>()
                        && get_const_as_int(rv.static_as<constant>())
                                == dim[ax])
                    continue;
            }
            return false;
        } else if (get_const_as_int(first.static_as<constant>()) != 0) {
            return false;
        }
    }
    return true;
}

bool slice_divisible_on_axis(const sc_dims &dim, const slice_range &ranges,
        const std::vector<int> &axis) {
    for (auto &ax : axis) {
        auto second = do_cast_and_fold(ranges[ax].second);
        // slice range length should be divisible by dims
        if (second.isa<constant>()
                && dim[ax] % get_const_as_int(second.checked_as<constant>())
                        != 0) {
            return false;
        }
    }
    return true;
}

bool slice_divisible_by_factor(const slice_range &ranges,
        const std::vector<int> &axis, const int factor) {
    for (auto &ax : axis) {
        auto second = do_cast_and_fold(ranges[ax].second);
        // slice range length should be divisible by dims
        if (second.isa<constant>()) {
            if (get_const_as_int(second.checked_as<constant>()) % factor != 0) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

bool slice_larger_than_bound_on_axis(const slice_range &ranges,
        const std::vector<int> &axis, const int factor, const int lower_bound) {
    auto total_len = 1;
    for (auto &ax : axis) {
        auto second = do_cast_and_fold(ranges[ax].second);
        if (second.isa<constant>()) {
            total_len *= get_const_as_int(second.checked_as<constant>());
        } else {
            return false;
        }
    }
    if (total_len / factor < lower_bound) { return false; }
    return true;
}

bool innermost_slice_with_non_dividable_lanes(const context_ptr &ctx,
        const slice_range &slice, const sc_data_type_t &dtype, sc_dim &floor,
        sc_dim &tail) {
    if (is_dynamic_slice_range_list({slice})) return false;
    auto shape = get_slice_shape(slice);
    auto dims = get_expr_to_dims(shape);
    auto last_dim = dims.back();
    // check dims except last are all one
    if (get_dims_product(dims) != (size_t)last_dim) return false;
    // get max lanes
    auto lanes = vectorize_step(ctx, dtype.type_code_);
    if ((last_dim > lanes) && (last_dim % lanes != 0)) {
        floor = last_dim / lanes * lanes;
        tail = last_dim % lanes;
        return true;
    }
    return false;
}

int get_slice_size(const slice_range &ranges, const int dtype_size) {
    auto total_size = dtype_size;
    for (auto &range : ranges) {
        auto second = do_cast_and_fold(range.second);
        if (second.isa<constant>()) {
            total_size *= get_const_as_int(second.checked_as<constant>());
        } else {
            return -1;
        }
    }
    return total_size;
}

bool slice_expr_equals(const expr &in1, const expr &in2) {
    auto fin1 = do_cast_and_fold(in1);
    auto fin2 = do_cast_and_fold(in2);
    if (fin1->equals(fin2)) { return true; }
    // datatype may not equal during slice compare.
    if (fin1.isa<constant>() && fin2.isa<constant>()) {
        return get_expr_as_int(fin1) == get_expr_as_int(fin2);
    }
    return false;
}

expr cast_to_s32(const expr &in) {
    return builder::make_cast(datatypes::s32, in);
}

std::vector<graph_tensor_ptr> get_sorted_inputs_by_layout_input(
        const sc_op_ptr &op) {
    int layout_input_index
            = op->attrs_.get_or_else(op_attr_key::layout_input_index, 0);
    if (layout_input_index == 0) { return op->get_inputs(); }
    std::vector<graph_tensor_ptr> ret;
    auto inps = op->get_inputs();
    ret.reserve(inps.size());
    ret.push_back(inps[layout_input_index]);
    for (int i = 0; i < static_cast<int>(inps.size()); i++) {
        if (i == layout_input_index) { continue; }
        ret.push_back(inps[i]);
    }
    return ret;
}

variant<float, int64_t> numeric_limits_minimum(sc_data_etype type_code) {
    if (type_code == sc_data_etype::F32 || type_code == sc_data_etype::BF16) {
        return -std::numeric_limits<float>::infinity();
    } else if (type_code == sc_data_etype::F16) {
        return (float)-65504;
    } else if (type_code == sc_data_etype::U8
            || type_code == sc_data_etype::U32) {
        return int64_t(0);
    } else if (type_code == sc_data_etype::S8) {
        return int64_t(-128);
    } else if (type_code == sc_data_etype::S32) {
        return int64_t(std::numeric_limits<int32_t>::min());
    } else {
        COMPILE_ASSERT(0, "unsupported data_etype");
    }
}

variant<float, int64_t> numeric_limits_maximum(sc_data_etype type_code) {
    if (type_code == sc_data_etype::F32 || type_code == sc_data_etype::BF16) {
        return std::numeric_limits<float>::infinity();
    } else if (type_code == sc_data_etype::F16) {
        return (float)65504;
    } else if (type_code == sc_data_etype::S8) {
        return int64_t(127);
    } else if (type_code == sc_data_etype::S32) {
        return int64_t(std::numeric_limits<int32_t>::max());
    } else if (type_code == sc_data_etype::U8) {
        return int64_t(255);
    } else if (type_code == sc_data_etype::U32) {
        return int64_t(std::numeric_limits<uint32_t>::max());
    } else {
        COMPILE_ASSERT(0, "unsupported data_etype");
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

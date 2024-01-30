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

#include "fused_op.hpp"
#include <algorithm>
#include <atomic>
#include <utility>
#include "anchor_loop_generator.hpp"
#include "fusible_op_utils.hpp"
#include "lowering.hpp"
#include "pass/pass.hpp"
#include "tunable_op.hpp"
#include "visitor.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/dynamic_internal_info.hpp>
#include <compiler/ir/graph/dynamic_utils.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/dead_write_eliminate.hpp>
#include <compiler/ir/transform/dyn_tsr_transform.hpp>
#include <compiler/ir/transform/func_inline.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <ops/convolution.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/padding.hpp>
#include <ops/fusible/pooling.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/shape_of_tensor.hpp>
#include <ops/fusible/ternary_elemwise.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <ops/managed_matmul_core.hpp>
#include <ops/matmul_core.hpp>
#include <runtime/config.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <unordered_map>
#include <unordered_set>

SC_MODULE(graph.fused_op)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

fusion_partition_t *fusion_partition_t::get_root() const {
    if (merged_to) {
        assert(ops.empty());
        return merged_to->get_root();
    }
    return const_cast<fusion_partition_t *>(this);
}

bool fusion_partition_t::is_ok_to_add(
        sc_op *op, const op_dep_matrix_t &g) const {
    if (merged_to) { return get_root()->is_ok_to_add(op, g); }
    for (auto &in : op->get_inputs()) {
        // if the input is in the partition, don't need to check
        if (contains(in->producer_owner_)) { continue; }
        // if "in" depends on an op in the partition
        if (main_tunable_op
                && g.lookup(main_tunable_op->logical_op_id_,
                           in->producer_owner_->logical_op_id_)
                        == 1) {
            return false;
        }
        for (auto &op_in_set : ops) {
            auto result = g.lookup(op_in_set->logical_op_id_,
                    in->producer_owner_->logical_op_id_);
            // if "in" depends on an op in the partition
            if (result == 1) { return false; }
        }
    }
    return true;
}

bool fusion_partition_t::contains(sc_op *op) const {
    if (merged_to) { return get_root()->contains(op); }
    return op == main_tunable_op.get()
            || ops.find(op->shared_from_this()) != ops.end();
}

// merge the ops in "other" to "this"
void fusion_partition_t::merge(const ptr &other) const {
    auto ths_root = get_root();
    auto other_root = other->get_root();
    if (ths_root == other_root) { return; }
    ths_root->ops.insert(other_root->ops.begin(), other_root->ops.end());
    assert(other_root->main_tunable_op == nullptr);
    other_root->ops.clear();
    other_root->merged_to = ths_root->shared_from_this();
}

namespace graph {
void mark_read_or_write_buffers(std::vector<expr> &args, bool is_read) {
    const char *name = is_read ? "read_buffer" : "write_buffer";
    for (auto &tsr : args) {
        tsr->attr()[name] = true;
    }
}

func_t create_func_decl_for_op(
        sc_op *op, std::vector<expr> &ins, std::vector<expr> &outs) {
    auto &graph = op->get_owner_graph();
    ins = ins.empty() ? graph::tensor_detail_to_ir_tensor(
                  graph, "__ins_", op->get_inputs())
                      : ins;
    outs = outs.empty() ? graph::tensor_detail_to_ir_tensor(
                   graph, "__outs_", op->get_outputs())
                        : outs;
    graph::mark_read_or_write_buffers(ins, true);
    graph::mark_read_or_write_buffers(outs, false);
    std::vector<expr> args = outs;
    args.insert(args.end(), ins.begin(), ins.end());
    std::string func_name;
    if (auto layer_name
            = op->attrs_.get_or_null<std::string>(op_attr_key::layer_name)) {
        COMPILE_ASSERT(!layer_name->empty() && isalpha(layer_name->front()),
                "Bad layername: " << *layer_name);
        func_name = *layer_name;
    } else {
        func_name = op->op_name_;
    }
    func_name += "__";
    func_name += std::to_string(op->logical_op_id_);
    auto func = builder::make_func(func_name, args,
            make_stmt<stmts_node_t>(std::vector<stmt>()), datatypes::boolean);
    // func->attr()["Gflop"] = gen_ptr->get_gflop();
    return func;
}

func_t create_query_func_decl_for_op(sc_op *op, std::vector<expr> &ins,
        std::vector<expr> &ori_ins, std::vector<expr> &outs,
        std::vector<expr> &in_fmts, std::vector<expr> &ori_in_fmts,
        std::vector<expr> &out_fmts, std::vector<expr> &out_sizes,
        expr &kernel) {
    func_t func = create_func_decl_for_op(op, ins, outs);
    in_fmts.resize(ins.size());
    out_fmts.resize(outs.size());
    out_sizes.resize(outs.size());
    assert(op->isa<mixed_fuse_op_t>());
    if ((op->isa<mixed_fuse_op_t>()
                && !op->stc_cast<mixed_fuse_op_t>()
                            ->get_internal_tunable_input_indices()
                            .empty())) {
        size_t ori_sz = op->stc_cast<mixed_fuse_op_t>()
                                ->get_internal_tunable_input_indices()
                                .size();
        auto &graph = op->get_owner_graph();
        for (size_t i = 0; i < ori_sz; i++) {
            auto ori_in = graph::tensor_detail_to_ir_tensor(graph,
                    std::string("__ori_ins_") + std::to_string(i),
                    op->get_inputs()[i]->details_);
            ori_in->attr().set(attr_keys::always_trans, true);
            ori_ins.emplace_back(ori_in);
            ori_in_fmts.emplace_back(builder::make_tensor(
                    ori_in.static_as<tensor>()->name_ + "_format",
                    {UINT64_C(1)}, datatypes::index));
        }
    }
    std::transform(ins.begin(), ins.end(), in_fmts.begin(), [](const expr &in) {
        in->attr().set(attr_keys::always_trans, true);
        return builder::make_tensor(in.static_as<tensor>()->name_ + "_format",
                {UINT64_C(1)}, datatypes::index);
    });
    std::transform(
            outs.begin(), outs.end(), out_fmts.begin(), [](const expr &in) {
                in->attr().set(attr_keys::always_trans, true);
                return builder::make_tensor(
                        in.static_as<tensor>()->name_ + "_format",
                        {UINT64_C(1)}, datatypes::index);
            });
    std::transform(
            outs.begin(), outs.end(), out_sizes.begin(), [](const expr &in) {
                return builder::make_tensor(
                        in.static_as<tensor>()->name_ + "_size", {UINT64_C(1)},
                        datatypes::index);
            });
    // table should be get from module global data, here is only for the
    // alignment to other functions.
    auto table = builder::make_var(datatypes::pointer, "dummy_table");
    std::vector<expr> args = func->params_;
    args.insert(args.begin(), table);
    args.insert(args.end(), ori_ins.begin(), ori_ins.end());
    args.insert(args.end(), out_fmts.begin(), out_fmts.end());
    args.insert(args.end(), in_fmts.begin(), in_fmts.end());
    args.insert(args.end(), ori_in_fmts.begin(), ori_in_fmts.end());
    args.insert(args.end(), out_sizes.begin(), out_sizes.end());
    kernel = builder::make_tensor("func_kernel",
            {1 + get_num_of_internal_funcs(op->shared_from_this())},
            datatypes::index);
    args.push_back(kernel);
    func->params_ = args;
    func->name_ = std::string("query_format_") + func->name_;
    return func;
}
} // namespace graph
static std::atomic<int> out_idx(0);

struct fused_exprs_t {
    expr dummy_size;
    expr dummy_kernel;
    expr combined_keys;
    expr combined_algs;
};

struct general_fused_params_t {
    builder::ir_builder_t &bld;
    ir_module_ptr modu;
    sc_graph_t &graph;
    sc_op_ptr node;
    std::unordered_map<graph_tensor_ptr, tsr_info_t> &ltsr_rtsr;
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> &fmgr_2_orig;
    std::unordered_map<graph_tensor_ptr, bool> &visited;
    int &inner_tsr_count;
    int &cur_combined_op_idx;
    int &cur_combined_key_idx;
    int &cur_ori_inp_idx;
    int &cur_internal_idx;
    fused_exprs_t exprs;
};

tsr_info_t get_or_create_tsr_and_fmt(
        general_fused_params_t &gp, const graph_tensor_ptr &in) {
    auto it = gp.ltsr_rtsr.find(in);
    if (it != gp.ltsr_rtsr.end()) { return it->second; }
    auto &bld = gp.bld;
    auto rtsr
            = builder::make_tensor("tsr_" + std::to_string(gp.inner_tsr_count),
                    {sizeof(runtime::dynamic_tensor_t)}, datatypes::u8);
    auto shape_tsr = builder::make_tensor(
            "dyn_shape_tsr_" + std::to_string(gp.inner_tsr_count),
            {in->details_.get_plain_dims().size()}, datatypes::index);
    shape_tsr->attr().set(attr_keys::no_dead_write, true);
    shape_tsr->attr().set(attr_keys::no_tensor2var, true);
    bool fmt_init = in->details_.get_format_candidates().size() <= 1;
    auto out_fmt = builder::make_tensor(
            "format_" + std::to_string(gp.inner_tsr_count), {1UL},
            datatypes::index);
    bld.push_var_tensor_def(rtsr);
    bld.push_var_tensor_def(shape_tsr);
    bld.push_evaluate(builder::make_write_struct(rtsr, shape_tsr,
            dyn_tsr_struct_t::name, dyn_tsr_struct_t::fields::dim_ptr));
    bld.push_evaluate(builder::make_write_struct(rtsr,
            builder::make_constant(
                    {in->details_.get_plain_dims().size()}, datatypes::s32),
            dyn_tsr_struct_t::name, dyn_tsr_struct_t::fields::ndims));
    uint64_t etype = in->details_.dtype_.is_etype_pointer()
            ? in->details_.dtype_.get_pointer_element().as_etype_int()
            : in->details_.dtype_.as_etype_int();
    bld.push_evaluate(builder::make_write_struct(rtsr,
            builder::make_constant({etype}, datatypes::u32),
            dyn_tsr_struct_t::name, dyn_tsr_struct_t::fields::dtype));
    auto plain_shapes = gp.node->get_owner_graph().dims_to_expr(
            in->details_.get_plain_dims());
    uint64_t dyn_mask_int = 0;
    for (size_t i = 0; i < plain_shapes.size(); i++) {
        bld.push_assign(
                builder::make_indexing(shape_tsr, {i}), plain_shapes[i]);
        dyn_mask_int |= (uint64_t(!plain_shapes[i].isa<constant>()) << i);
    }
    bld.push_evaluate(builder::make_write_struct(rtsr,
            builder::make_constant({dyn_mask_int}, datatypes::u8),
            dyn_tsr_struct_t::name, dyn_tsr_struct_t::fields::dyn_mask));
    uint64_t init_format = 0;
    if (fmt_init) {
        init_format = uint64_t(in->details_.get_format().to_runtime());
    }
    bld.push_var_tensor_def(out_fmt);
    bld.push_assign(builder::make_indexing(out_fmt, {0}), init_format);

    gp.inner_tsr_count++;
    auto ret = tsr_info_t(rtsr, expr(), out_fmt, gp.exprs.dummy_size);
    gp.ltsr_rtsr[in] = ret;
    return ret;
}

static bool need_inner_query(
        general_fused_params_t &gp, const sc_op_ptr &node, int &main_idx) {
    auto &inputs = node->get_inputs();
    auto &outputs = node->get_outputs();
    // check if the op is associated with const.
    for (size_t i = 0; i < inputs.size(); i++) {
        auto &in = inputs[i];
        // original ltensor is legal and is constant
        if (!gp.visited[in]) { return true; }
        auto it = gp.fmgr_2_orig.find(in);
        if (it != gp.fmgr_2_orig.end() && !it->second->uses_.empty()
                && it->second->producer_owner_
                && it->second->producer_owner_->attrs_.get_or_else(
                           "constant", const_kind::not_const)
                        != const_kind::not_const) {
            return true;
        }
        if (!in->uses_.empty() && in->producer_owner_
                && in->producer_owner_->isa<reorder_op_t>()) {
            return true;
        }
    }
    if (node->isa<binary_elementwise_op_t>()) {
        int bc_idx = node->stc_cast<binary_elementwise_op_t>()
                             ->get_broadcast_input();
        main_idx = bc_idx == -1 ? 0 : 1 - bc_idx;
    }
    // check the op is linked to output, need to query output size.
    for (size_t i = 0; i < outputs.size(); i++) {
        auto &out = outputs[i];
        for (size_t j = 0; j < out->uses_.size(); j++) {
            if (out->uses_[j].second->isa<output_op>()) { return true; }
        }
    }
    return false;
}

void update_op_visited(general_fused_params_t &gp, const sc_op_ptr &node) {
    for (auto &in : node->get_inputs()) {
        gp.visited[in] = true;
    }
    for (auto &out : node->get_outputs()) {
        gp.visited[out] = true;
    }
}

void add_global_table_var(general_fused_params_t &gp,
        const std::string &table_name, const op_dispatch_tables_ptr &table_ptr,
        const expr &table_var) {
    gp.modu->add_op_table(std::make_pair(table_name, table_ptr));
    auto table_def = builder::make_var_tensor_def_unattached(
            table_var, linkage::private_global);
    gp.modu->add_global_var(table_def.checked_as<define>());
}

void declare_dummy_and_combined_tsrs(
        general_fused_params_t &gp, int total_key_num, int dispatch_op_num) {
    auto &bld = gp.bld;
    bld.push_scope();
    // create dummy tensor/var for inner query.
    gp.exprs.dummy_kernel
            = builder::make_var(datatypes::pointer, "dummy_kernel");
    bld.push_var_tensor_def(gp.exprs.dummy_kernel, linkage::local);
    gp.exprs.dummy_size
            = builder::make_tensor("dummy_size", {1}, datatypes::index);
    bld.push_var_tensor_def(gp.exprs.dummy_size, linkage::local);
    if (total_key_num) {
        expr combined_keys = builder::make_tensor(
                "combined_keys", {total_key_num}, datatypes::pointer);
        gp.exprs.combined_keys = combined_keys;
        bld.push_var_tensor_def(combined_keys);
    }
    if (dispatch_op_num) {
        expr combined_algs = builder::make_tensor(
                "combined_algs", {dispatch_op_num}, datatypes::s32);
        gp.exprs.combined_algs = combined_algs;
        bld.push_var_tensor_def(combined_algs);
    }
}

void set_original_tensor_and_format_for_tunables(general_fused_params_t &gp,
        sc_op *node_before, const std::vector<expr> &ori_ins,
        const std::vector<expr> &ori_in_fmts, expr &ori_tsr, expr &ori_fmt) {
    tsr_info_t tsr_info;
    if (node_before->isa<reorder_op_t>()) {
        tsr_info = get_or_create_tsr_and_fmt(gp, node_before->get_inputs()[0]);
    } else {
        auto ltsr = node_before->get_outputs()[0];
        auto it = gp.fmgr_2_orig.find(ltsr);
        // if find in fmgr_2_orig, it is a input op, else is a internal ltsr.
        if (it != gp.fmgr_2_orig.end()) {
            assert(node_before->isa<input_op>());
            ltsr = it->second;
        }
        tsr_info = get_or_create_tsr_and_fmt(gp, ltsr);
    }
    ori_tsr = tsr_info.tensor_;
    ori_fmt = tsr_info.format_;
}

void create_query_function_by_graph(general_fused_params_t &gp,
        const expr &kernel, const std::vector<expr> &ori_ins,
        const std::vector<expr> &ori_in_fmts,
        std::vector<int> &each_op_num_keys, int total_key_num,
        int dispatch_op_num,
        const std::vector<size_t> &tunable_inp_indices
        = std::vector<size_t>()) {
    auto &bld = gp.bld;
    auto combined_keys = gp.exprs.combined_keys;
    auto combined_algs = gp.exprs.combined_algs;
    auto &cur_combined_key_idx = gp.cur_combined_key_idx;
    auto &cur_combined_op_idx = gp.cur_combined_op_idx;
    auto &cur_internal_idx = gp.cur_internal_idx;
    std::vector<std::string> table_names(gp.graph.ops_.size());
    auto create_internal_query_func = [&](const sc_op_ptr &op) {
        // Can not use can_op_be_dispatched as tsr and format need
        // pass through each op.
        if (op->isa<input_op>() || op->isa<output_op>()
                || op->isa<constant_op_t>()) {
            return;
        }
        expr dummy_kernel = gp.exprs.dummy_kernel;
        expr dummy_size = gp.exprs.dummy_size;
        auto ctx = gp.modu->ctx_;
        int main_idx = 0;
        size_t in_size = op->get_inputs().size();
        size_t out_size = op->get_outputs().size();
        auto table_name = gp.node->op_name_ + "__"
                + std::to_string(gp.node->logical_op_id_) + "_inner__"
                + std::to_string(op->logical_op_id_) + "_table";
        if (op->info_.internal_info_) {
            op->info_.internal_info_->dispatch_table_name_ = table_name;
        }
        auto table_var = builder::make_var(datatypes::pointer, table_name);
        auto table_ptr = std::make_shared<op_dispatch_tables_t>();
        std::vector<tsr_info_t> op_outs(out_size), op_ins(in_size);
        for (size_t i = 0; i < out_size; i++) {
            op_outs[i] = get_or_create_tsr_and_fmt(gp, op->get_outputs()[i]);
        }
        for (size_t i = 0; i < in_size; i++) {
            op_ins[i] = get_or_create_tsr_and_fmt(gp, op->get_inputs()[i]);
        }
        if (op->isa<ops::matmul_core_op_t>()
                || op->isa<ops::managed_matmul_core_op_t>()
                || op->isa<ops::conv_fwd_core_op_t>()) {
            auto table_ptr = std::make_shared<op_dispatch_tables_t>();
            // create origin tsr and dispatch key for tunable ops
            expr ori_in0, ori_in1, ori_in_fmt0, ori_in_fmt1;
            auto node_before_in0 = op->get_inputs()[0]->producer_owner_;
            auto node_before_in1 = op->get_inputs()[1]->producer_owner_;
            set_original_tensor_and_format_for_tunables(gp, node_before_in0,
                    ori_ins, ori_in_fmts, ori_in0, ori_in_fmt0);
            set_original_tensor_and_format_for_tunables(gp, node_before_in1,
                    ori_ins, ori_in_fmts, ori_in1, ori_in_fmt1);
            add_global_table_var(gp, table_name, table_ptr, table_var);
            auto internal_kernel = op->need_dynamic_internal_query()
                    ? builder::tensor_ptr(kernel, {cur_internal_idx})
                    : make_expr<constant_node>(UINT64_C(0), datatypes::pointer);
            std::vector<expr> args = {table_var, op_outs[0].tensor_,
                    op_ins[0].tensor_, op_ins[1].tensor_, ori_in0, ori_in1,
                    op_outs[0].format_, op_ins[0].format_, op_ins[1].format_,
                    ori_in_fmt0, ori_in_fmt1, op_outs[0].size_, internal_kernel,
                    builder::tensor_ptr(combined_algs, {cur_combined_op_idx})};
            bld.push_evaluate(call_op_dynamic_query_function(op, args));
            initialize_dispatch_table_with_op(ctx, op, table_ptr);
            // set combined tensor
            bld.push_assign(builder::make_indexing(
                                    combined_keys, {cur_combined_key_idx++}),
                    op_ins[0].format_);
            bld.push_assign(builder::make_indexing(
                                    combined_keys, {cur_combined_key_idx++}),
                    op_ins[1].format_);
            bld.push_assign(builder::make_indexing(
                                    combined_keys, {cur_combined_key_idx++}),
                    op_outs[0].format_);
            each_op_num_keys[cur_combined_op_idx] = 3;
            cur_combined_op_idx++;
            if (op->isa<ops::managed_matmul_core_op_t>()) {
                cur_internal_idx++;
            }
        } else if (op->isa<unary_elementwise_op_impl_t>()) {
            if (need_inner_query(gp, op, main_idx)) {
                add_global_table_var(gp, table_name, table_ptr, table_var);
                initialize_dispatch_table_with_op(ctx, op, table_ptr);
                std::vector<expr> args = {table_var, op_outs[0].tensor_,
                        op_ins[0].tensor_, op_outs[0].format_,
                        op_ins[0].format_, op_outs[0].size_, dummy_kernel};
                bld.push_evaluate(call_op_dynamic_query_function(op, args));
            } else {
                auto &out = op->get_outputs()[0];
                gp.ltsr_rtsr[out] = op_ins[main_idx];
            }
        } else if (op->isa<padding_op_t>()) {
            add_global_table_var(gp, table_name, table_ptr, table_var);
            initialize_dispatch_table_with_op(ctx, op, table_ptr);
            std::vector<expr> args = {table_var, op_outs[0].tensor_,
                    op_ins[0].tensor_, op_outs[0].format_, op_ins[0].format_,
                    op_outs[0].size_, dummy_kernel};
            bld.push_evaluate(call_op_dynamic_query_function(op, args));
        } else if (op->isa<pooling_op_t>()) {
            add_global_table_var(gp, table_name, table_ptr, table_var);
            initialize_dispatch_table_with_op(ctx, op, table_ptr);
            std::vector<expr> args = {table_var, op_outs[0].tensor_,
                    op_ins[0].tensor_, op_outs[0].format_, op_ins[0].format_,
                    op_outs[0].size_, dummy_kernel};
            bld.push_evaluate(call_op_dynamic_query_function(op, args));
        } else if (op->isa<binary_elementwise_op_impl_t>()) {
            if (need_inner_query(gp, op, main_idx)) {
                add_global_table_var(gp, table_name, table_ptr, table_var);
                initialize_dispatch_table_with_op(ctx, op, table_ptr);
                std::vector<expr> args = {table_var, op_outs[0].tensor_,
                        op_ins[0].tensor_, op_ins[1].tensor_,
                        op_outs[0].format_, op_ins[0].format_,
                        op_ins[1].format_, op_outs[0].size_, dummy_kernel};
                bld.push_evaluate(call_op_dynamic_query_function(op, args));
            } else {
                auto &out = op->get_outputs()[0];
                gp.ltsr_rtsr[out] = op_ins[main_idx];
            }
        } else if (op->isa<reorder_op_t>()) {
            // Currently reorder is the last op of fusion pattern, so always
            // query.
            add_global_table_var(gp, table_name, table_ptr, table_var);
            std::vector<expr> args = {table_var, op_outs[0].tensor_,
                    op_ins[0].tensor_, op_outs[0].format_, op_ins[0].format_,
                    op_outs[0].size_, dummy_kernel,
                    builder::tensor_ptr(combined_algs, {cur_combined_op_idx})};
            bld.push_evaluate(call_op_dynamic_query_function(op, args));
            // set combined key tensor
            bld.push_assign(builder::make_indexing(
                                    combined_keys, {cur_combined_key_idx++}),
                    op_ins[0].format_);
            bld.push_assign(builder::make_indexing(
                                    combined_keys, {cur_combined_key_idx++}),
                    op_outs[0].format_);
            each_op_num_keys[cur_combined_op_idx] = 2;
            cur_combined_op_idx++;
        } else if (op->isa<reduce_op_t>()) {
            // always query
            add_global_table_var(gp, table_name, table_ptr, table_var);
            initialize_dispatch_table_with_op(ctx, op, table_ptr);
            std::vector<expr> args = {table_var, op_outs[0].tensor_,
                    op_ins[0].tensor_, op_outs[0].format_, op_ins[0].format_,
                    op_outs[0].size_, dummy_kernel};
            bld.push_evaluate(call_op_dynamic_query_function(op, args));
        } else if (op->isa<tensor_view_op_t>()) {
            // always query
            add_global_table_var(gp, table_name, table_ptr, table_var);
            initialize_dispatch_table_with_op(ctx, op, table_ptr);
            std::vector<expr> args = {table_var, op_outs[0].tensor_,
                    op_ins[0].tensor_, op_outs[0].format_, op_ins[0].format_,
                    op_outs[0].size_, dummy_kernel};
            bld.push_evaluate(call_op_dynamic_query_function(op, args));
        } else if (op->isa<select_op_t>()) {
            add_global_table_var(gp, table_name, table_ptr, table_var);
            initialize_dispatch_table_with_op(ctx, op, table_ptr);
            std::vector<expr> args = {table_var, op_outs[0].tensor_,
                    op_ins[0].tensor_, op_ins[1].tensor_, op_ins[2].tensor_,
                    op_outs[0].format_, op_ins[0].format_, op_ins[1].format_,
                    op_ins[2].format_, op_outs[0].size_, dummy_kernel};
            bld.push_evaluate(call_op_dynamic_query_function(op, args));
        } else if (op->isa<shape_of_tensor_op_t>()) {
            // do nothing
        } else {
            COMPILE_ASSERT(false,
                    "Currently dynamic fusbile op only support "
                    "unary/binary.");
        }
        update_op_visited(gp, op);
    };
    visit_fused_graph_by_query_order(gp.graph, create_internal_query_func);

    // final query the fused op kernel.
    assert(gp.cur_combined_key_idx == total_key_num
            && gp.cur_combined_op_idx == dispatch_op_num);
    assert(gp.cur_internal_idx - 1 == get_num_of_internal_funcs(gp.node));
    auto main_table_name = gp.node->op_name_ + "__"
            + std::to_string(gp.node->logical_op_id_) + "_ptr_table";
    auto main_table_var
            = builder::make_var(datatypes::pointer, main_table_name);
    auto main_table_ptr = std::make_shared<op_dispatch_tables_t>();
    add_global_table_var(gp, main_table_name, main_table_ptr, main_table_var);
    expr each_op_num_keys_tsr = builder::make_tensor("each_op_num_keys",
            {dispatch_op_num}, datatypes::s32, address_space::automatic,
            std::make_shared<static_data_t>(each_op_num_keys));
    gp.modu->add_global_var(builder::make_var_tensor_def_unattached(
            each_op_num_keys_tsr, linkage::private_global)
                                    .static_as<define>());
    bld.push_evaluate(builtin::call_fused_op_query_combined(main_table_var,
            combined_keys.defined() ? combined_keys : get_ir_null(),
            combined_algs.defined() ? combined_algs : get_ir_null(),
            each_op_num_keys_tsr, dispatch_op_num, kernel));
}

mixed_fuse_op_t::mixed_fuse_op_t(const std::string &name,
        const std::vector<mixed_parti_t::ptr> &parti_list,
        const ir_module_ptr &mod, const sc_graph_t &graph,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    parti_list_ = parti_list;
    mod_ = mod;
    sub_graph_ = copy_graph(graph);
    op_name_ = name;
    attrs_ = attrs;
}

bool mixed_fuse_op_t::need_dynamic_internal_query_impl() const {
    return !std::all_of(sub_graph_.ops_.begin(), sub_graph_.ops_.end(),
            [](const sc_op_ptr &op) {
                return !op->need_dynamic_internal_query();
            });
}

ir_module_ptr mixed_fuse_op_t::get_func(context_ptr ctx) {
    func_t func;
    bool use_cache
            = get_owner_graph().attrs_.get_or_else("temp.force_static", false);
    ir_module_ptr modu;
    if (!use_cache && can_op_be_dispatched(shared_from_this())) {
        modu = std::make_shared<ir_module_t>(ctx);
        // max fusion policy don't need any conditions.
        expr max_loop_parallelism_cond;
        func_t max_fusion_func, max_loop_parallel_func;
        ir_module_ptr max_fusion_modu, max_loop_parallel_modu;
        std::vector<expr> ins, outs;
        func = graph::create_func_decl_for_op(this, ins, outs);
        outs.insert(outs.end(), ins.begin(), ins.end());
        std::for_each(outs.begin(), outs.end(), [](const expr &arg) {
            arg->attr().set(attr_keys::always_trans, true);
        });
        if (need_dynamic_internal_query()) {
            auto internal_func_arg = builder::make_tensor(
                    "extra_internal_funcs",
                    {get_num_of_internal_funcs(sub_graph_)}, datatypes::index);
            func->params_.emplace_back(internal_func_arg);
            func->decl_->params_.emplace_back(internal_func_arg);
            outs.emplace_back(internal_func_arg);
        }
        func->name_ = op_name_;
        func->decl_->name_ = op_name_;
        func->name_ += "_" + std::to_string(logical_op_id_);
        func->decl_->name_ += "_" + std::to_string(logical_op_id_);
        auto return_stmt = builder::make_returns_unattached(true);
        stmt policy_dispatch;
        {
            // max_loop_parallelism policy
            auto cpy_graph = copy_graph(sub_graph_);
            cpy_graph.attrs_.set("temp.dynamic_fusion_policy",
                    dynamic_fusion_policy_t::max_loop_parallelism);
            mixed_partition(cpy_graph, ctx);
            std::vector<sc_op_ptr> lower_args(cpy_graph.get_output_ops());
            auto input_ops = cpy_graph.get_input_ops();
            lower_args.insert(
                    lower_args.end(), input_ops.begin(), input_ops.end());
            cpy_graph.attrs_.set("temp.force_static", true);
            max_loop_parallel_modu
                    = lower_graph(ctx, cpy_graph, lower_args, false);
            max_loop_parallel_func = max_loop_parallel_modu->get_entry_func();
            max_loop_parallel_func->name_ = op_name_;
            max_loop_parallel_func->decl_->name_ = op_name_;
            max_loop_parallel_func->name_
                    += "_max_loop_parallism_" + std::to_string(logical_op_id_);
            max_loop_parallel_func->decl_->name_
                    += "_max_loop_parallism_" + std::to_string(logical_op_id_);
            max_loop_parallelism_cond = cpy_graph.attrs_.get<expr>(
                    "temp.fusion_policy_condition");
            schedule_loops(max_loop_parallel_func->body_);
        }
        // if condition is true or false after simplify, keep only one module
        // for less functions.
        max_loop_parallelism_cond = do_cast_and_fold(max_loop_parallelism_cond);
        if (max_loop_parallelism_cond->equals(expr(true))) {
            modu->merge(*max_loop_parallel_modu);
            policy_dispatch = builder::make_evaluate_unattached(
                    builder::make_call(max_loop_parallel_func->decl_, outs));
        } else {
            // max_fusion policy
            auto cpy_graph = copy_graph(sub_graph_);
            cpy_graph.attrs_.set("temp.dynamic_fusion_policy",
                    dynamic_fusion_policy_t::max_fusion);
            mixed_partition(cpy_graph, ctx);
            std::vector<sc_op_ptr> lower_args(cpy_graph.get_output_ops());
            auto input_ops = cpy_graph.get_input_ops();
            lower_args.insert(
                    lower_args.end(), input_ops.begin(), input_ops.end());
            cpy_graph.attrs_.set("temp.force_static", true);
            max_fusion_modu = lower_graph(ctx, cpy_graph, lower_args, false);
            max_fusion_func = max_fusion_modu->get_entry_func();
            max_fusion_func->name_ = op_name_;
            max_fusion_func->name_
                    += "_max_fusion_" + std::to_string(logical_op_id_);
            max_fusion_func->decl_->name_
                    += "_max_fusion_" + std::to_string(logical_op_id_);
            schedule_loops(max_fusion_func->body_);
            modu->merge(*max_fusion_modu);
            if (max_loop_parallelism_cond->equals(expr(false))) {
                policy_dispatch = builder::make_evaluate_unattached(
                        builder::make_call(max_fusion_func->decl_, outs));
            } else {
                modu->merge(*max_loop_parallel_modu);
                policy_dispatch = builder::make_if_else_unattached(
                        max_loop_parallelism_cond,
                        builder::make_evaluate_unattached(builder::make_call(
                                max_loop_parallel_func->decl_, outs)),
                        builder::make_evaluate_unattached(builder::make_call(
                                max_fusion_func->decl_, outs)));
            }
        }
        func->body_ = builder::make_stmts_unattached(
                {policy_dispatch, return_stmt});
        modu->add_func({func});
        modu->set_entry_func_idx(modu->get_contents().size() - 1);
    } else {
        // if mod_ is not empty, usually when redo occurs in partition stage.
        if (mod_) return mod_;
        COMPILE_ASSERT(parti_list_.size() == 1,
                "partition size is expected for 1, but got "
                        << parti_list_.size())
        func = parti_list_[0]->func_;
        func->name_ = op_name_;
        func->decl_->name_ = op_name_;
        func->name_ += "_" + std::to_string(logical_op_id_);
        func->decl_->name_ += "_" + std::to_string(logical_op_id_);
        schedule_loops(func->body_);
        modu = std::make_shared<ir_module_t>(ctx);
        modu->add_func({func});
        modu->set_entry_func_idx(0);
        if (need_dynamic_internal_query()) {
            for (auto &parti : parti_list_) {
                assert(parti->dyn_inter_);
                modu->merge(*parti->dyn_inter_->mod_);
            }
        }
    }

    return modu;
}

void schedule_loop_body(const stmt &body, node_ptr_map *node_remap) {
    stmt target_loop = body;
    if (target_loop.isa<stmts>()) {
        auto ss = target_loop.static_as<stmts>();
        COMPILE_ASSERT(ss->seq_.size() == 1 && ss->seq_[0].isa<for_loop>(),
                "for loop node is expected");
        target_loop = ss->seq_[0];
    }
    COMPILE_ASSERT(target_loop.isa<for_loop>(), "for loop node is expected");
    for_loop outer_most_loop = target_loop.checked_as<for_loop>();
    outer_most_loop->kind_ = for_type::PARALLEL;
    assert(outer_most_loop.defined());
    const int run_threads = runtime_config_t::get().get_num_threads();
    for_loop cur_loop = outer_most_loop;
    std::vector<for_loop> loops;
    auto fused_number = 1;
    while (true) {
        if (cur_loop->iter_end_.isa<constant>()
                && cur_loop->iter_begin_.isa<constant>()) {
            fused_number *= (get_expr_as_int(cur_loop->iter_end_)
                    - get_expr_as_int(cur_loop->iter_begin_));
        }
        if (fused_number / run_threads > 12
                || (fused_number >= run_threads
                        && (fused_number % run_threads) == 0))
            break;
        auto inner_loop = get_inner_for_loop(cur_loop.get());
        if (inner_loop.defined() && !inner_loop->num_threads_
                && (inner_loop->step_.isa<constant>()
                        && get_expr_as_int(inner_loop->step_) == 1)
                && !inner_loop->attr().get_or_else(
                        stmt_attr_key::no_loop_fuse, false)) {
            node_ptr_map cur_remap;
            outer_most_loop->fuse(
                    inner_loop, node_remap ? &cur_remap : nullptr);
            cur_loop = inner_loop;
            if (node_remap) {
                for (auto &exp_m : (*node_remap)) {
                    auto iter = cur_remap.find(exp_m.second);
                    if (iter != cur_remap.end()) {
                        exp_m.second = iter->second;
                        cur_remap.erase(iter);
                    }
                }
                node_remap->insert(cur_remap.begin(), cur_remap.end());
            }
        } else {
            break;
        }
    }
}

static op_traits::may_prefetch_t *find_prefetch_op(const sc_graph_t &graph) {
    // fix-me (yijie): should provide a way to find the main op of the partition
    // find the first tunable & prefetchable op
    op_traits::may_prefetch_t *found_op = nullptr;
    op_visitor_t::dfs_topology_sort(graph.ops_.size())
            .visit_graph(
                    graph, [&found_op](op_visitor_t *vis, const sc_op_ptr &op) {
                        if (found_op) { return; }
                        if (op->isa<tunable_op_t>()
                                && op->isa<op_traits::may_prefetch_t>()
                                && op->attrs_.get_or_else(
                                        mixed_partition_hint::first_prefetch_op,
                                        false)) {
                            for (auto &ins : op->get_inputs()) {
                                if (ins->producer_owner_->isa<input_op>()) {
                                    found_op = op->dyn_cast<
                                            op_traits::may_prefetch_t>();
                                    return;
                                }
                            }
                        }
                    });
    return found_op;
}

std::vector<int> mixed_fuse_op_t::query_prefetch(const context_ptr &ctx,
        bool is_global, const std::vector<tensor_slice> &ins) {
    if (auto found_op = find_prefetch_op(sub_graph_)) {
        auto ret = found_op->query_prefetch(ctx, is_global, ins);
        auto &ins = dynamic_cast<sc_op *>(found_op)->get_inputs();
        // check that the inputs to prefetch are from the sub-graph inputs
        std::vector<int> indices;
        auto graph_inputs = sub_graph_.get_input_ops();
        for (auto in_of_op : ret) {
            if (auto the_in_op
                    = ins.at(in_of_op)->producer_owner_->dyn_cast<input_op>()) {
                // find the index of the input op in graph inputs
                int idx = std::find(graph_inputs.begin(), graph_inputs.end(),
                                  the_in_op->shared_from_this())
                        - graph_inputs.begin();
                // if idx is not in indices, push
                if (std::find(indices.begin(), indices.end(), idx)
                        == indices.end()) {
                    indices.emplace_back(idx);
                }
            }
        }
        return indices;
    }
    return {};
}

void mixed_fuse_op_t::generate_prefetcher_body_for_tensor(
        const context_ptr &ctx, const std::vector<expr> &func_args,
        const std::vector<expr> &ins, const std::vector<int> &indices) {
    if (auto found_op = find_prefetch_op(sub_graph_)) {
        auto graph_inputs = sub_graph_.get_input_ops();
        auto &op_ins = dynamic_cast<sc_op *>(found_op)->get_inputs();
        std::vector<int> indices_in_op;
        for (auto idx : indices) {
            auto input_op_ptr = graph_inputs.at(idx).get();
            for (size_t i = 0; i < op_ins.size(); i++) {
                if (op_ins[i]->producer_owner_ == input_op_ptr) {
                    indices_in_op.emplace_back(i);
                }
            }
        }
        found_op->generate_prefetcher_body_for_tensor(
                ctx, func_args, ins, indices_in_op);
    }
}

void mixed_fuse_op_t::schedule_loops(const stmt &body) {
    if (body.isa<for_loop>())
        schedule_loop_body(body);
    else if (body.isa<stmts>()) {
        for (auto &st : body.checked_as<stmts>()->seq_) {
            if (st.isa<for_loop>()) {
                schedule_loop_body(st);
            } else {
                // recursively call
                schedule_loops(st);
            }
        }
    }
}

std::vector<size_t> mixed_fuse_op_t::get_internal_tunable_input_indices() {
    auto graph_input_ops = sub_graph_.get_input_ops();
    std::vector<size_t> ret;
    for (size_t i = 0; i < graph_input_ops.size(); i++) {
        auto &uses = graph_input_ops[i]->get_outputs()[0]->uses_;
        for (auto &use : uses) {
            if (use.second.lock() && use.second->isa<tunable_op_t>()) {
                ret.emplace_back(i);
                break;
            }
        }
    }
    return ret;
}

dispatch_set_ptr &mixed_fuse_op_t::get_dispatch_key_set() {
    if (!info_.dispatch_key_set_) {
        int dummy_num;
        info_.dispatch_key_set_ = std::make_shared<combined_dispatch_key_set_t>(
                get_inner_dispatch_ops(&dummy_num));
    }
    return info_.dispatch_key_set_;
}

std::vector<sc_op_ptr> mixed_fuse_op_t::get_inner_dispatch_ops(
        int *total_key_num) {
    std::vector<sc_op_ptr> ret;
    if (total_key_num) { *total_key_num = 0; }
    ret = get_graph_inner_dispatch_ops(sub_graph_, total_key_num);
    return ret;
}

void mixed_fuse_op_t::update_internal_graph_format(
        const combined_op_dispatch_key_t &key, const context_ptr &ctx) {
    int key_idx = 0;
    update_graph_format_by_key(
            ctx, shared_from_this(), sub_graph_, key, key_idx, 0, 0);
    assert(key_idx == static_cast<int>(key.size()));
}

ir_module_ptr mixed_fuse_op_t::get_dynamic_query_func(const context_ptr &ctx) {
    auto modu = std::make_shared<ir_module_t>(ctx);
    sub_graph_.sync_dynamic_info_with_graph(get_owner_graph());
    std::unordered_map<graph_tensor_ptr, graph_tensor_ptr> fmgr_2_orig;
    auto &node_inputs = get_inputs();
    const auto &graph_input_ops = sub_graph_.get_input_ops();
    const auto &graph_output_ops = sub_graph_.get_output_ops();
    assert(node_inputs.size() == graph_input_ops.size());
    for (size_t i = 0; i < node_inputs.size(); i++) {
        fmgr_2_orig[graph_input_ops[i]->get_outputs()[0]] = node_inputs[i];
    }
    // inner graph logical tensor visit states, for query pruning.
    std::unordered_map<graph_tensor_ptr, bool> visited;
    std::vector<expr> ins, outs, in_fmts, out_fmts, ori_ins, ori_in_fmts,
            out_sizes;
    expr main_table_var, kernel;
    size_t inp_idx = 0, out_idx = 0;
    auto func = graph::create_query_func_decl_for_op(this, ins, ori_ins, outs,
            in_fmts, ori_in_fmts, out_fmts, out_sizes, kernel);
    // inner logical tensor => real tensor, out_size and format.
    std::unordered_map<graph_tensor_ptr, tsr_info_t> ltsr_rtsr;
    builder::ir_builder_t bld;
    int total_key_num = 0;
    int inner_tsr_count = 0;
    int dispatch_op_num
            = static_cast<int>(get_inner_dispatch_ops(&total_key_num).size());
    int cur_combined_op_idx = 0, cur_combined_key_idx = 0, cur_ori_inp_idx = 0,
        cur_internal_idx = 1;
    // create general params.
    general_fused_params_t gp {bld, modu, sub_graph_, shared_from_this(),
            ltsr_rtsr, fmgr_2_orig, visited, inner_tsr_count,
            cur_combined_op_idx, cur_combined_key_idx, cur_ori_inp_idx,
            cur_internal_idx, fused_exprs_t()};
    // construct combined tensors for final query.
    std::vector<int> each_op_num_keys(dispatch_op_num, 0);
    // build query function body
    // declare dummy kernel and dummy size, combined tsrs
    declare_dummy_and_combined_tsrs(gp, total_key_num, dispatch_op_num);
    for (auto &inp : graph_input_ops) {
        auto &ltsr = inp->get_outputs()[0];
        ltsr_rtsr[ltsr]
                = tsr_info_t(ins[inp_idx], expr(), in_fmts[inp_idx], expr());
        inp_idx++;
    }
    for (auto &out : graph_output_ops) {
        auto &ltsr = out->get_inputs()[0];
        ltsr_rtsr[ltsr] = tsr_info_t(
                outs[out_idx], expr(), out_fmts[out_idx], out_sizes[out_idx]);
        out_idx++;
    }
    auto query_idx = get_internal_tunable_input_indices();
    for (size_t i = 0; i < query_idx.size(); i++) {
        auto &ori_inp_idx = query_idx[i];
        auto &ltsr = node_inputs[ori_inp_idx];
        ltsr_rtsr[ltsr]
                = tsr_info_t(ori_ins[i], expr(), ori_in_fmts[i], expr());
    }
    // create query functions of valid ops inside graph and final query
    // function.
    create_query_function_by_graph(gp, kernel, ori_ins, ori_in_fmts,
            each_op_num_keys, total_key_num, dispatch_op_num);
    bld.push_returns(true);
    auto body = bld.pop_scope();
    func->body_ = std::move(body);
    modu->add_func({func});
    modu->set_entry_func_idx(0);
    return modu;
}

void mixed_fuse_op_t::create_internal_dispatch_funcs(const context_ptr &ctx,
        ir_module_ptr &mod,
        const std::shared_ptr<const thread_pool_mode_t> &use_mtp) {
    // todo: currently we only support one op with internal func query.
    for (auto &op : sub_graph_.ops_) {
        if (op->need_dynamic_internal_query()) {
            COMPILE_ASSERT(op->info_.internal_info_
                            && !op->info_.internal_info_->dispatch_table_name_
                                        .empty(),
                    "Not set the dispatch table in mixed op.");
            auto &table_name = op->info_.internal_info_->dispatch_table_name_;
            int dyn_idx = 0;
            op->info_.internal_info_->parti_in_ltsrs_
                    = info_.internal_info_->parti_in_ltsrs_;
            op->info_.internal_info_->parti_out_ltsrs_
                    = info_.internal_info_->parti_out_ltsrs_;
            op->get_internal_dispatch_key_set(ctx)->for_each_key_process(
                    std::bind(create_dispatch_funcs_by_keys, ctx, std::ref(mod),
                            table_name, op, std::placeholders::_1, expr(),
                            std::ref(dyn_idx), use_mtp,
                            /*internal*/ true));
        }
    }
}

void mixed_fuse_op_t::get_graph_impl(std::shared_ptr<sc_graph_t> &graph) {
    throw std::runtime_error("mixed_fuse_op_t::get_graph Not implemented");
}

struct inplace_recursion_context_t {
    int depth_ = 0;
    // UNDEF means a tensor is not directly or indirectly connected to an input
    enum kind_t { UNDEF = 0, NO_INPLACE, ZERO_OFFSET_INPLACE, FREE_INPLACE };

    // the graph input tensor -> its index in std::vector<kind_t>
    const std::unordered_map<graph_tensor_ptr, int> &tsr_2_in_index_;
    // the map of all graph tensors -> a vector of graph inputs. Each element of
    // the vector represents the in-place status of a graph input
    std::unordered_map<graph_tensor_ptr, std::vector<kind_t>> result_;

    inplace_recursion_context_t(
            const std::unordered_map<graph_tensor_ptr, int> &tsr_2_in_index)
        : tsr_2_in_index_(tsr_2_in_index) {}

    // merges the in-place results. Used when an Op depends on multiple graph
    // inputs and we need to merge the status of the same input
    static kind_t merge_result(kind_t a, kind_t b, bool good_op) {
        if (good_op) {
            if (a == NO_INPLACE || b == NO_INPLACE) { return NO_INPLACE; }
            if (a == UNDEF) { return b; }
            if (b == UNDEF) { return a; }
            if (a == ZERO_OFFSET_INPLACE || b == ZERO_OFFSET_INPLACE) {
                return ZERO_OFFSET_INPLACE;
            }
            return FREE_INPLACE;
        } else {
            // if the current op is not a "good" op, we need to mark all inputs
            // it depends on with NO_INPLACE
            if (a != UNDEF || b != UNDEF) { return NO_INPLACE; }
            return UNDEF;
        }
    }

    // the main recursion function to recursively find the in-place status
    const std::vector<kind_t> *call(const graph_tensor_ptr &tsr) {
        depth_++;
        if (depth_ > 500) { return nullptr; }
        auto itr = result_.find(tsr);
        if (itr != result_.end()) { return &itr->second; }
        auto &ret
                = (result_[tsr] = std::vector<kind_t>(tsr_2_in_index_.size()));
        // if the tensor is used more than once, we simply skip it for the sake
        // of correctness. We can obviously do better than this.
        auto producer = tsr->producer_owner_;
        bool good_op = tsr->uses_.size() <= 1UL;
        if (producer->isa<input_op>()) {
            // we define that input tensor can in-place reuse itself.
            ret.at(tsr_2_in_index_.find(tsr)->second)
                    = good_op ? FREE_INPLACE : NO_INPLACE;
            depth_--;
            return &ret;
        }
        bool is_binary = producer->isa<binary_elementwise_op_t>();
        // if it is an broadcast op, we cannot in-place reuse the broadcast
        // input
        int bcast_idx = -1;
        if (is_binary) {
            bcast_idx = producer->stc_cast<binary_elementwise_op_t>()
                                ->get_broadcast_input();
        }
        bool must_zero_offset = producer->isa<cast_op_t>() || is_binary
                || producer->isa<unary_elementwise_op_t>();
        bool can_be_free = producer->isa<tensor_view_op_t>();
        good_op = good_op && (must_zero_offset || can_be_free);
        // we allow cast ops in the dependency chain, even if the size of dtype
        // is not the same. Here we only check the logical dependency instead of
        // memory position
        auto &inputs = producer->get_inputs();
        for (size_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
            auto &intsr = inputs[input_idx];
            auto *sub_result = call(intsr);
            if (!sub_result) { return nullptr; }
            for (size_t i = 0; i < ret.size(); i++) {
                auto result = (*sub_result)[i];
                // if the op's input is broadcast, all the graph input tensors
                // it depends on should be NO_INPLACE
                if ((int64_t)bcast_idx == (int64_t)input_idx
                        && result != UNDEF) {
                    result = NO_INPLACE;
                }
                ret[i] = merge_result(ret[i], result, good_op);
                if (must_zero_offset && ret[i] == FREE_INPLACE) {
                    ret[i] = ZERO_OFFSET_INPLACE;
                }
            }
        }
        depth_--;
        return &ret;
    }
};

float mixed_fuse_op_t::get_gflop() {
    return sub_graph_.get_gflop();
}

sc_op_ptr mixed_fuse_op_t::copy( // NOLINT
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &graph) {
    auto ret = graph.make<mixed_fuse_op_t>(op_name_, parti_list_, nullptr,
            copy_graph(sub_graph_), ins, outs, attrs_);
    ret->sub_graph_.sync_dynamic_info_with_graph(graph);
    return ret;
}

std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>
mixed_fuse_op_t::get_inplace_map() {
    std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>> ret;
    auto in_ops = sub_graph_.get_input_ops();
    // create a map from input tensors to its index
    std::unordered_map<graph_tensor_ptr, int> tsr_2_index;
    std::vector<graph_tensor_ptr> index_2_tsr;
    for (auto &in : in_ops) {
        for (auto &tsr : in->get_outputs()) {
            auto idx = tsr_2_index.size();
            tsr_2_index[tsr] = idx;
            index_2_tsr.emplace_back(tsr);
        }
    }
    inplace_recursion_context_t ctx {tsr_2_index};

    // for each output tensors...
    auto out_ops = sub_graph_.get_output_ops();
    size_t out_idx = 0;
    for (auto &out : out_ops) {
        for (auto &outtsr : out->get_inputs()) {
            std::vector<tensor_inplace_info_t> can_inplace;
            auto *rec_ret = ctx.call(outtsr);
            if (!rec_ret) {
                SC_MODULE_WARN << "Max recursion count reached for tensor "
                                  "inplace optimization "
                                  "for fused op";
                return {};
            }
            for (size_t i = 0; i < rec_ret->size(); i++) {
                if ((*rec_ret)[i]
                        == inplace_recursion_context_t::ZERO_OFFSET_INPLACE) {
                    // zero offset means that the output->input dependency
                    // chain contains elementwise ops. We need to ensure
                    // that each memory position of the output strictly
                    // depend on the same memory position of the input
                    if (utils::get_sizeof_type(
                                index_2_tsr.at(i)->details_.dtype_)
                            == utils::get_sizeof_type(
                                    outtsr->details_.dtype_)) {
                        can_inplace.emplace_back(
                                tensor_inplace_info_t {static_cast<int>(i),
                                        inplace_kind::ZERO_OFFSET});
                    }
                } else if ((*rec_ret)[i]
                        == inplace_recursion_context_t::FREE_INPLACE) {
                    can_inplace.emplace_back(tensor_inplace_info_t {
                            static_cast<int>(i), inplace_kind::FREE});
                }
            }
            if (!can_inplace.empty()) {
                ret.emplace_back(out_idx, std::move(can_inplace));
            }
        }
        out_idx++;
    }
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

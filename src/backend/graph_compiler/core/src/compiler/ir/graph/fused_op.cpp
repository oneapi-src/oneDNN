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

#include "fused_op.hpp"
#include <atomic>
#include <utility>
#include "fusion_mgr.hpp"
#include "outer_loop_generator.hpp"
#include "pass/pass.hpp"
#include "tunable_op.hpp"
#include "visitor.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/transform/func_inline.hpp>
#include <compiler/ir/transform/scope_flatten.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <unordered_map>
namespace sc {
namespace graph {
void mark_read_or_write_buffers(std::vector<expr> &args, bool is_read) {
    const char *name = is_read ? "read_buffer" : "write_buffer";
    for (auto &tsr : args) {
        tsr->attr()[name] = true;
    }
}

func_t create_func_decl_for_op(
        sc_op *op, std::vector<expr> &ins, std::vector<expr> &outs) {
    ins = graph::tensor_detail_to_ir_tensor("__ins_", op->get_inputs());
    outs = graph::tensor_detail_to_ir_tensor("__outs_", op->get_outputs());
    graph::mark_read_or_write_buffers(ins, true);
    graph::mark_read_or_write_buffers(outs, false);
    std::vector<expr> args = outs;
    args.insert(args.end(), ins.begin(), ins.end());
    auto func_name = op->op_name_;
    func_name += "__";
    func_name += std::to_string(op->logical_op_id_);
    auto func = builder::make_func(func_name, args,
            make_stmt<stmts_node_t>(std::vector<stmt>()), datatypes::boolean);
    // func->attr()["Gflop"] = gen_ptr->get_gflop();
    return func;
}
} // namespace graph
static std::atomic<int> out_idx(0);

op_traits::post_fusion_acceptable_t *fused_op_t::get_main_op() const {
    COMPILE_ASSERT(main_op_.ops_.size() == 2, "Bad internal graph");
    auto op = main_op_.ops_[1]->dyn_cast<op_traits::post_fusion_acceptable_t>();
    COMPILE_ASSERT(op, "The main op is not post_fusion_acceptable_t");
    return op;
}

bool fused_op_t::is_valid(const context_ptr &ctx) {
    if (main_op_.empty()) { return true; }
    return main_op_.ops_[1]->is_valid(ctx);
}

std::shared_ptr<sc_graph_t> fused_op_t::get_graph() {
    throw std::runtime_error("fused_op_t::get_graph Not implemented");
    return nullptr;
}

fused_op_t::fused_op_t(const std::string &name, sc_graph_t &&main_op,
        std::shared_ptr<fusion_manager> fuse_mgr,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : mgr_(std::move(fuse_mgr)), main_op_(std::move(main_op)) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    attrs_ = attrs;
    if (!main_op_.ops_.empty()) {
        attrs_.set("horizontal_merge",
                main_op_.ops_[1]->attrs_.get_or_else(
                        "horizontal_merge", horizontal_merge_type::no_merge));
    }
    op_name_ = name;
}

bool fused_op_t::compare_contents(const sc_op *other) const {
    if (!sc_op::compare_contents(other)) { return false; }
    if (auto other_fused = other->dyn_cast<const fused_op_t>()) {
        if (main_op_.empty() != other_fused->main_op_.empty()) { return false; }
        if (!main_op_.empty()) {
            auto mainop = dynamic_cast<sc_op *>(get_main_op());
            auto other_mainop
                    = dynamic_cast<sc_op *>(other_fused->get_main_op());
            if (!mainop->compare_contents(other_mainop)) { return false; }
        }
        return compare_graph(mgr_->get_graph(), other_fused->mgr_->get_graph());
    }
    return false;
}

// may need refactor when enable graph hash
size_t fused_op_t::hash_contents() const {
    size_t seed = 0;
    hash_combine(seed, sc_op::hash_contents());
    if (!main_op_.empty()) {
        auto mainop = dynamic_cast<sc_op *>(get_main_op());
        hash_combine(seed, mainop->hash_contents());
    }
    return seed;
}

ir_module_ptr fused_op_t::get_func(context_ptr ctx) {
    std::vector<sc_op_ptr> out_failed;
    auto ret = try_get_func(ctx, false, out_failed);
    COMPILE_ASSERT(ret && out_failed.empty(),
            "Fusion failed. Fallback not implemented");
    return ret;
}

ir_module_ptr fused_op_t::try_get_func(const context_ptr &ctx, bool just_check,
        std::vector<sc_op_ptr> &out_failed) {
    auto modu = std::make_shared<ir_module_t>(ctx);
    // if no base Op, do fusion on fusible ops
    if (main_op_.empty()) {
        auto &graph = mgr_->get_graph();
        op_dep_matrix_t dep(graph);
        // find correct base input idx for generator
        size_t inp_idx = 0;
        // collect ops which results to broadcast op
        std::vector<sc_op *> bc_dep_ops;
        for (auto &op : graph.ops_) {
            if (auto be_op = op->dyn_cast<binary_elementwise_op_t>()) {
                int bc_idx = be_op->get_broadcast_input();
                if (bc_idx >= 0) { bc_dep_ops.emplace_back(op.get()); }
            }
        }

        for (size_t i = 0; i < graph.get_input_ops().size(); i++) {
            auto ins = graph.get_input_ops()[i];
            // sucessfully found flag
            bool found = true;
            for (auto &op : bc_dep_ops) {
                if (dep.lookup(ins->logical_op_id_, op->logical_op_id_) == 1) {
                    if (auto be_op = op->dyn_cast<binary_elementwise_op_t>()) {
                        int bc_idx = be_op->get_broadcast_input();
                        // ensued by above check
                        COMPILE_ASSERT(bc_idx >= 0,
                                "Implicit broadcast op is expected.");
                        auto bc_arg = op->get_inputs()[bc_idx]->producer_owner_;
                        auto non_bc_arg
                                = op->get_inputs()[1 - bc_idx]->producer_owner_;
                        if (dep.lookup(ins->logical_op_id_,
                                    non_bc_arg->logical_op_id_)
                                        == 1
                                || ins.get() == non_bc_arg) {
                            continue;
                        }
                        if (dep.lookup(
                                    ins->logical_op_id_, bc_arg->logical_op_id_)
                                        == 1
                                || ins.get() == bc_arg) {
                            found = false;
                            break;
                        }
                    } else {
                        COMPILE_ASSERT(
                                0, "Unexpected Op named: " << op->op_name_);
                    }
                }
            }
            if (found) {
                inp_idx = i;
                break;
            }
        }
        // reset input_idx
        mgr_->put_input_first(
                graph.get_input_ops()[inp_idx]->dyn_cast<input_op>());
        outer_loop_generator_t gen(inp_idx);
        return try_lower_fusion_manager(
                ctx, &gen, this, mgr_.get(), true, just_check, out_failed);
    }
    auto mainop = dynamic_cast<tunable_op_t *>(get_main_op());
    COMPILE_ASSERT(mainop, "Expecting tunable_op");

    COMPILE_ASSERT(mainop->get_outputs().size() == 1,
            "Expecting single output tunable op");
    auto gen_ptr = mainop->create_generator();
    mainop->set_config_if_empty(ctx, gen_ptr.get());

    std::vector<expr> ins;
    // real_outs are the output tensors in the function arguments
    std::vector<expr> real_outs;
    auto func = graph::create_func_decl_for_op(this, ins, real_outs);
    assert(mainop->get_outputs().size() <= real_outs.size());
    assert(mainop->get_outputs().size() == 1);
    assert(keep_outputs_.size() == 1);
    // finds if an output can be computed in-place on an "input" of the fusion
    // graph
    auto inplacemap = mgr_->query_inplace();
    // outs are the output tensors for the original outputs of the main op
    std::vector<expr> outs;
    if (keep_outputs_[0]) {
        outs = {real_outs[0]};
    } else if (!inplacemap.at(0).empty() && inplacemap[0][0] == 0) {
        // a really naive inplace optimization. We currently only allow
        // output of fusion replace output of the original output of tunable
        // Op
        outs = {real_outs[0]};
    } else {
        // no in-place available, define the output args independently
        outs = graph::tensor_detail_to_ir_tensor(
                "__origouts_" + std::to_string(out_idx++),
                mainop->get_outputs());
        assert(outs.size() == 1);
    }
    auto main_op_input_size = mainop->get_inputs().size();
    COMPILE_ASSERT(get_inputs().size() >= main_op_input_size,
            "Input tsr number of Fused op should >= Input tsr number of "
            "original op");
    // additional inputs arg tensors for fusion
    std::vector<expr> additional_ins;
    std::vector<expr> origin_ins(ins.begin(), ins.begin() + main_op_input_size);
    for (size_t i = 0; i < get_inputs().size() - main_op_input_size; i++) {
        additional_ins.emplace_back(ins[i + main_op_input_size]);
    }

    // =======================
    // Start of building function body
    // =======================
    builder::ir_builder_t bld;
    bld.push_scope();
    // for each original outputs
    for (auto &local_out : outs) {
        // if the output is in final output args, do not need to define the
        // output buffer as local tensor
        bool need_define_local = true;
        for (auto &v : real_outs) {
            if (v.ptr_same(local_out)) {
                need_define_local = false;
                break;
            }
        }
        if (need_define_local) {
            local_out->attr()[tensor_shrinker_attrs::may_shrink] = true;
            bld.push_var_tensor_def(local_out);
        }
    }
    std::vector<for_loop> loops;
    bool status = gen_ptr->generate(ctx, mainop->get_config().get(), mgr_.get(),
            origin_ins, outs, loops);
    assert(status);
    bld.push_returns(true);
    auto body = bld.pop_scope();

    // =======================
    // End of building function body
    // =======================
    std::vector<fusion_anchor_data> fuse_state;
    std::vector<expr> fuse_outs;
    if (keep_outputs_[0]) {
        assert(real_outs.size() > 1);
        fuse_outs = std::vector<expr>(real_outs.begin() + 1, real_outs.end());
    } else {
        fuse_outs = real_outs;
    }
    out_failed = mgr_->prepare_and_check(
            ctx, fuse_state, fuse_outs, additional_ins);
    if (!out_failed.empty()) {
        mgr_->clear_anchor();
        return nullptr;
    }
    if (just_check) {
        mgr_->clear_anchor();
        return nullptr;
    }
    bool can_in_brg = mgr_->can_register_brgemm_fusion(body);
    if (!can_in_brg) { mgr_->break_brgemm_fusion(); }
    mgr_->commit(modu, fuse_state);
    // register fusion in brgemm.
    if (can_in_brg) {
        body = mgr_->get_brgemm_fusion_register()
                       .remake_brgemm_intrinsic_by_fusion(body);
    }
    func->body_ = std::move(body);
    gen_ptr->schedule_loops(
            ctx, mainop->get_config().get(), func->body_, loops);
    modu->add_func({func});
    modu->set_entry_func_idx(0);
    return modu;
}

std::shared_ptr<sc_graph_t> horizontal_fused_op_t::get_graph() {
    throw std::runtime_error("horiaontal_fused_op::get_graph Not implemented");
    return nullptr;
}

horizontal_fused_op_t::horizontal_fused_op_t(const std::string &name,
        const std::vector<sc_op_ptr> &ops_to_merge,
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : ops_to_merge_(ops_to_merge) {
    info_.inputs_ = ins;
    info_.outputs_ = outs;
    op_name_ = name;
    attrs_ = attrs;
}

static std::unordered_map<graph_tensor_ptr, expr> make_ltrs2rtrs_map(
        const std::vector<graph_tensor_ptr> &ltrs,
        const std::vector<expr> &rtrs) {
    assert(ltrs.size() == rtrs.size());
    std::unordered_map<graph_tensor_ptr, expr> m;
    for (size_t i = 0; i < ltrs.size(); i++) {
        m[ltrs[i]] = rtrs[i];
    }
    return m;
}

static std::vector<expr> get_op_rtrs(
        std::unordered_map<graph_tensor_ptr, expr> &m,
        const std::vector<graph_tensor_ptr> &ltrs) {
    std::vector<expr> rtrs;
    rtrs.reserve(ltrs.size());
    for (const graph_tensor_ptr &ltr : ltrs) {
        assert(m.find(ltr) != m.end());
        rtrs.emplace_back(m[ltr]);
    }
    return rtrs;
}

void horizontal_fused_op_t::schedule_loops(const stmt &body) {
    COMPILE_ASSERT(body.isa<stmts>(), "body has only one stmt.");
    scope_flatten(body.checked_as<stmts>(), -1);
    std::vector<stmt> &body_seq = body.checked_as<stmts>()->seq_;
    std::vector<for_loop> loops;
    std::vector<stmt> not_loops;
    stmt return_stmt;
    for (auto &st : body_seq) {
        if (st.isa<for_loop>()) {
            loops.push_back(st.checked_as<for_loop>());
        } else if (!st.isa<returns>()) {
            not_loops.push_back(st);
        } else {
            return_stmt = st;
        }
    }
    std::vector<stmt> new_seq(not_loops.begin(), not_loops.end());
    new_seq.insert(new_seq.end(), loops.begin(), loops.end());
    new_seq.push_back(return_stmt);
    body_seq = std::move(new_seq);
    COMPILE_ASSERT(loops.size() > 1,
            "No need to horizontal fuse as parallel loop number is less than "
            "2.");
    for (size_t i = 1; i < loops.size(); i++) {
        loops[0]->parallel_merge(body, loops[i]);
    }
}

ir_module_ptr horizontal_fused_op_t::get_func(context_ptr ctx) {
    auto modu = std::make_shared<ir_module_t>(ctx);
    std::vector<expr> ins, outs;
    auto func = graph::create_func_decl_for_op(this, ins, outs);
    std::unordered_map<graph_tensor_ptr, expr> ins_ltrs2rtrs
            = make_ltrs2rtrs_map(info_.inputs_, ins);
    std::unordered_map<graph_tensor_ptr, expr> outs_ltrs2rtrs
            = make_ltrs2rtrs_map(info_.outputs_, outs);
    func_inliner_t inliner;
    builder::ir_builder_t bld;
    bld.push_scope();
    for (auto &op : ops_to_merge_) {
        op->info_.inputs_ = remake_logical_tensors(
                op->attrs_.get<std::vector<graph_tensor_ptr>>("op_ins"));
        op->info_.outputs_ = remake_logical_tensors(
                op->attrs_.get<std::vector<graph_tensor_ptr>>("op_outs"));
        auto mod_to_merge = op->get_func(ctx);
        auto &global_vars = mod_to_merge->get_module_vars();
        for (auto &def_v : global_vars) {
            modu->add_global_var(def_v);
        }
        auto f = mod_to_merge->get_entry_func();
        tensor_shrinker_t pass;
        f = std::const_pointer_cast<func_base>(pass(f));
        std::vector<expr> op_in_args = get_op_rtrs(ins_ltrs2rtrs,
                op->attrs_.get<std::vector<graph_tensor_ptr>>("op_ins"));
        std::vector<expr> op_out_args = get_op_rtrs(outs_ltrs2rtrs,
                op->attrs_.get<std::vector<graph_tensor_ptr>>("op_outs"));
        op_out_args.insert(
                op_out_args.end(), op_in_args.begin(), op_in_args.end());
        auto callf = make_expr<call_node>(f, op_out_args);
        inliner.inline_at(callf, bld.get_current_scope().body, 0, global_vars);
    }
    bld.push_returns(true);
    func->body_ = bld.pop_scope();
    schedule_loops(func->body_);
    modu->add_func({func});
    modu->set_entry_func_idx(0);
    return modu;
}

} // namespace sc

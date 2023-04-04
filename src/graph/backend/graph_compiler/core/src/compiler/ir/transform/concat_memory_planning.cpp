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

#include "concat_memory_planning.hpp"

#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/util_module_passes.hpp>
#include <compiler/ir/viewer.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/config.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(pass.concat_memory_planning);

SC_DECL_PASS_INFO(concat_memory_planning, SC_PASS_DEPENDS_ON(),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

static bool is_standalone_concat_call(call_c &v) {
    return v->attr_
            && v->attr_->get_or_else(
                    concat_optim_attr_keys::is_standalone_concat, false);
}

// Collect the {input buffer, (output buffer, offset)} info of all concat ops.
// Collcet the concat calls that should be deleted.
class concat_memory_planning_preprocess_t : public ir_viewer_t {
public:
    concat_memory_planning_preprocess_t(
            std::unordered_map<expr, std::pair<expr, std::vector<expr>>>
                    &concat_in_out,
            std::unordered_set<expr_c> &to_be_deleted)
        : concat_in_out(concat_in_out), to_be_deleted(to_be_deleted) {}
    // map the input buffer to the output buffer and offset
    std::unordered_map<expr, std::pair<expr, std::vector<expr>>> &concat_in_out;
    // concat evaluations to be deleted
    std::unordered_set<expr_c> &to_be_deleted;

    void view(call_c v) override {
        std::string func_name = v->get_prototype()->name_;
        // if this is a standalone concat op, we can directly check the args
        if (is_standalone_concat_call(v)) {
            SC_MODULE_INFO << "Meet a standalone concat call node: " << v;
            std::vector<std::vector<expr>> inputs_offsets;
            for (size_t i = 1; i < v->args_.size(); ++i) {
                if (v->args_[i]->attr_
                        && v->args_[i]->attr_->has_key(
                                concat_optim_attr_keys::pass_memory_offset)) {
                    auto &offset = v->args_[i]->attr().get<std::vector<expr>>(
                            concat_optim_attr_keys::pass_memory_offset);
                    if (offset.empty()) {
                        SC_MODULE_WARN
                                << "Input #" << i << ": " << v->args_[i]
                                << " has empty offset set, skip this concat";
                        return;
                    }
                    inputs_offsets.push_back(offset);
                } else {
                    SC_MODULE_INFO << "Input #" << i << ": " << v->args_[i]
                                   << " has no offset set, can not do memory "
                                      "planning, skip this concat";
                    return;
                }
            }
            COMPILE_ASSERT(inputs_offsets.size() == v->args_.size() - 1,
                    "Get wrong number of inputs offsets");
            for (size_t i = 1; i < v->args_.size(); ++i) {
                concat_in_out[v->args_[i]]
                        = std::make_pair(v->args_[0], inputs_offsets[i - 1]);
            }
            to_be_deleted.insert(v);
        }
    }
};

static void find_final_tensor_and_offset(
        std::unordered_map<expr, std::pair<expr, std::vector<expr>>>
                &concat_in_out) {
    for (auto &pair : concat_in_out) {
        expr curr = pair.first;
        size_t n_dims = curr.static_as<tensor>()->dims_.size();
        std::vector<expr> final_offset(n_dims, 0);
        while (concat_in_out.find(curr) != concat_in_out.end()) {
            auto parent = concat_in_out[curr].first;
            auto offset = concat_in_out[curr].second;
            curr = parent;
            for (size_t i = 0; i < n_dims; ++i) {
                final_offset[i] = final_offset[i] + offset[i];
            }
        }
        for (size_t i = 0; i < n_dims; ++i) {
            final_offset[i] = do_cast_and_fold(final_offset[i]);
        }
        pair.second = {curr, final_offset};
    }
}

// We have to visit the parent nodes of concat, so we can not merge this into
// preprocess.
class concat_memory_planning_process_t : public ir_visitor_t {
public:
    concat_memory_planning_process_t(
            std::unordered_map<expr, std::pair<expr, std::vector<expr>>>
                    &concat_in_out,
            std::unordered_set<expr_c> &to_be_deleted)
        : concat_in_out(concat_in_out), to_be_deleted(to_be_deleted) {}
    std::unordered_map<expr, std::pair<expr, std::vector<expr>>> &concat_in_out;
    // concat evaluations to be deleted, collected from preprocess
    std::unordered_set<expr_c> &to_be_deleted;
    // the concat ops' output buffers which are defined within the function
    std::unordered_set<expr> output_of_concats;

    using ir_visitor_t::dispatch;
    func_c dispatch(func_c v) override {
        // the args of function
        std::unordered_set<expr> args;
        for (auto &arg : v->params_) {
            args.insert(arg);
        }
        for (auto &in_out_pair : concat_in_out) {
            auto &out_buf = in_out_pair.second.first;
            // if the concat output buffer is defined within the function, not
            // an input/output param of the function
            if (args.count(out_buf) == 0) { output_of_concats.insert(out_buf); }
        }
        return ir_visitor_t::dispatch(v);
    }

    // Move forward the define of output buffer of concat.
    // Delete the defines of input buffers and concat evaluates.
    stmt_c visit(stmts_c v) override {
        const std::vector<stmt> &stmts_vec = v->seq_;
        std::unordered_map<expr, std::pair<size_t, size_t>>
                output_define_ori_pos_new_pos;
        // input buffer defines or concat evaluations
        std::unordered_set<size_t> to_delete_stmt_pos;
        for (size_t i = 0; i < stmts_vec.size(); ++i) {
            auto s = stmts_vec[i];
            if (s.isa<define>()) {
                auto var = s.static_as<define>()->var_;
                // if this stmt defines an input to a concat,
                if (concat_in_out.find(var) != concat_in_out.end()) {
                    to_delete_stmt_pos.insert(i);
                    auto out_var = concat_in_out[var].first;
                    // then we will move the define of output to this position
                    if (output_of_concats.count(out_var)
                            && output_define_ori_pos_new_pos.find(out_var)
                                    == output_define_ori_pos_new_pos.end()) {
                        output_define_ori_pos_new_pos[out_var].second = i;
                    }
                }
                // if this defines the output to a concat
                if (output_of_concats.find(var) != output_of_concats.end()) {
                    to_delete_stmt_pos.insert(i);
                    output_define_ori_pos_new_pos[var].first = i;
                }
            }
            if (s.isa<evaluate>()) {
                if (to_be_deleted.count(s.static_as<evaluate>()->value_)) {
                    to_delete_stmt_pos.insert(i);
                }
            }
        }

        std::unordered_map<size_t, size_t> new_ori;
        for (const auto &pos : output_define_ori_pos_new_pos) {
            size_t ori_pos = pos.second.first;
            size_t new_pos = pos.second.second;
            new_ori[new_pos] = ori_pos;
        }
        std::vector<stmt> new_seq;
        for (size_t i = 0; i < stmts_vec.size(); ++i) {
            if (new_ori.find(i) != new_ori.end()) {
                new_seq.push_back(stmts_vec[new_ori[i]]);
                // TODO(niuxiaoguang): the inputs and output of concats
                // should not occupy same address, but there seems no direct
                // way to add this constraint, so we simply do not do buffer
                // rescheduling on the output of concat. This may affects
                // performance in some cases. Improve this in the future.
                new_seq.back()->attr()["pass.tsr_dont_buf_sched"] = true;
            } else if (to_delete_stmt_pos.count(i)) {
                continue;
            } else {
                new_seq.push_back(dispatch(stmts_vec[i]).remove_const());
            }
        }
        return make_stmt<stmts_node_t>(std::move(new_seq));
    }

    stmt_c visit(define_c v) override { return v; }

    // Replace the inputs of concat with tensorptrs on the output of concat.
    expr_c visit(tensor_c v) override {
        expr e = v.static_as<expr>();
        if (concat_in_out.find(e) != concat_in_out.end()) {
            auto tsrptr = builder::tensor_ptr(concat_in_out[e].first,
                    concat_in_out[e].second, v->dims_, true);
            SC_MODULE_INFO << "Meet tensor_c: " << v
                           << " of shape: " << utils::print_vector(v->dims_)
                           << ", replace it with tensorptr: " << tsrptr;
            return tsrptr;
        } else {
            return v;
        }
    }
};

static const_ir_module_ptr optimize_standalone_concat_in_main_entry(
        const_ir_module_ptr in_mod) {
    SC_MODULE_INFO << "Start run concat_memory_planning pass on main entry";
    if (in_mod->get_entry_func() == nullptr) { return in_mod; }
    std::unordered_map<expr, std::pair<expr, std::vector<expr>>> concat_in_out;
    std::unordered_set<expr_c> to_be_deleted;
    concat_memory_planning_preprocess_t pre(concat_in_out, to_be_deleted);
    pre.dispatch(in_mod->get_entry_func());

    if (concat_in_out.empty()) {
        SC_MODULE_INFO
                << "Finish run concat_memory_planning pass on main entry";
        return in_mod;
    } else {
        find_final_tensor_and_offset(concat_in_out);

        concat_memory_planning_process_t pro(concat_in_out, to_be_deleted);
        auto ret_mod = std::make_shared<ir_module_t>(*in_mod);
        for (auto &funct : ret_mod->get_contents()) {
            if (funct == ret_mod->get_entry_func()) {
                funct = std::const_pointer_cast<func_base>(pro.dispatch(funct));
            }
        }
        SC_MODULE_INFO
                << "Finish run concat_memory_planning pass on main entry";
        return ret_mod;
    }
}

static expr find_final_tsr(expr v) {
    if (v.isa<tensor>()) {
        return v;
    } else if (v.isa<indexing>()) {
        return find_final_tsr(v.checked_as<indexing>()->ptr_);
    } else if (v.isa<tensorptr>()) {
        return find_final_tsr(v.checked_as<tensorptr>()->base_);
    } else {
        COMPILE_ASSERT(false, "Cannot find final tensor for: " << v);
        return v;
    }
}

// Collect the {input buffer, (output buffer, offset)} info of all concat ops.
// Some assignment operations of concat will be useless after optimization, so
// collect them to a set.
class concat_in_mxp_memory_planning_preprocess_t : public ir_viewer_t {
public:
    concat_in_mxp_memory_planning_preprocess_t(
            std::unordered_map<expr, std::pair<expr, std::vector<expr>>>
                    &concat_in_out,
            std::unordered_set<stmt_c> &to_be_deleted)
        : concat_in_out(concat_in_out), to_be_deleted(to_be_deleted) {}
    // the args of function
    std::unordered_set<expr> args;
    // map the input buffer to the output buffer and offset
    std::unordered_map<expr, std::pair<expr, std::vector<expr>>> &concat_in_out;
    // assignment statements that will be deleted later
    std::unordered_set<stmt_c> &to_be_deleted;

    func_c dispatch(func_c v) override {
        for (auto &arg : v->params_) {
            args.insert(arg);
        }
        return ir_viewer_t::dispatch(v);
    }

    void view(tensor_c v) override {
        if (args.count(v.static_as<expr>()) == 0 && v->attr_
                && v->attr_->has_key(concat_optim_attr_keys::pass_memory_offset)
                && v->attr_->has_key(
                        concat_optim_attr_keys::pass_memory_offset_to)) {
            // this buffer will be replaced by tensorptr
            auto &offset = v->attr_->get<std::vector<expr>>(
                    concat_optim_attr_keys::pass_memory_offset);
            COMPILE_ASSERT(!offset.empty(), "Input has empty offset set");
            auto &offset_to = v->attr_->get<expr>(
                    concat_optim_attr_keys::pass_memory_offset_to);
            auto input_expr = v.static_as<expr>();
            concat_in_out[input_expr] = std::make_pair(offset_to, offset);
        }
    }

    // collcet the redundant assignment stmts of concat operations
    void view(assign_c v) override {
        if (v->var_.isa<indexing>() && v->value_.isa<indexing>()) {
            // left[...] = right[...] ==> left = &left[...][...]
            expr left = find_final_tsr(v->var_);
            expr right = find_final_tsr(v->value_);
            if (args.count(right) == 0 // this buffer is not a function param
                    && concat_in_out.find(right) != concat_in_out.end()
                    && concat_in_out.at(right).first.ptr_same(left)) {
                // right is concat input and left is concat output
                to_be_deleted.insert(v);
            }
        }
    }
};

// Replace the inputs of concat with tensorptrs on the output of concat.
// We have to visit the parent nodes of concat, so we can not merge this into
// preprocess.
class concat_in_mxp_memory_planning_process_t : public ir_visitor_t {
public:
    concat_in_mxp_memory_planning_process_t(
            std::unordered_map<expr, std::pair<expr, std::vector<expr>>>
                    &concat_in_out,
            std::unordered_set<stmt_c> &to_be_deleted)
        : concat_in_out(concat_in_out), to_be_deleted(to_be_deleted) {}
    std::unordered_map<expr, std::pair<expr, std::vector<expr>>> &concat_in_out;
    std::unordered_set<stmt_c> &to_be_deleted;

    using ir_visitor_t::dispatch;

    stmt_c visit(stmts_c v) override {
        const std::vector<stmt> &stmts_vec = v->seq_;
        std::vector<stmt> new_seq;
        for (size_t i = 0; i < stmts_vec.size(); ++i) {
            auto s = stmts_vec[i];
            if (s.isa<assign>() && to_be_deleted.count(s)) {
                continue;
            } else {
                new_seq.push_back(dispatch(s).remove_const());
            }
        }
        return make_stmt<stmts_node_t>(std::move(new_seq));
    }

    stmt_c visit(define_c v) override { return v; }

    expr_c visit(tensor_c v) override {
        expr var = v.static_as<expr>();
        if (concat_in_out.find(var) != concat_in_out.end()) {
            auto tsrptr = builder::tensor_ptr(concat_in_out[var].first,
                    concat_in_out[var].second, v->dims_, true);
            SC_MODULE_INFO << "Meet tensor_c: " << v
                           << ", replace it with tensorptr: " << tsrptr;
            return tsrptr;
        }
        return v;
    }
};

static const_ir_module_ptr optimize_concat_in_mxp(
        const const_ir_module_ptr &in_mod) {
    SC_MODULE_INFO
            << "Start run concat_memory_planning pass on mixed partitions";
    auto out_mod = std::make_shared<ir_module_t>(*in_mod);
    for (auto &funct : out_mod->get_contents()) {
        if (funct == out_mod->get_entry_func()) { continue; }
        std::unordered_map<expr, std::pair<expr, std::vector<expr>>>
                concat_in_out;
        std::unordered_set<stmt_c> to_be_deleted;
        concat_in_mxp_memory_planning_preprocess_t pre(
                concat_in_out, to_be_deleted);
        pre.dispatch(funct);
        if (concat_in_out.empty()) {
            continue;
        } else {
            concat_in_mxp_memory_planning_process_t pro(
                    concat_in_out, to_be_deleted);
            funct->body_ = pro.dispatch(funct->body_).remove_const();
        }
    }
    SC_MODULE_INFO
            << "Finish run concat_memory_planning pass on mixed partitions";
    return out_mod;
}

const_ir_module_ptr concat_memory_planning_t::operator()(
        const_ir_module_ptr in_mod) {
    SC_MODULE_INFO << "Start run concat_memory_planning pass";
    auto out_mod = optimize_standalone_concat_in_main_entry(
            optimize_concat_in_mxp(in_mod));
    SC_MODULE_INFO << "Finish run concat_memory_planning pass";
    return out_mod;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

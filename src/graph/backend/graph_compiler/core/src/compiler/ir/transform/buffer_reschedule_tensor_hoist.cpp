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

#include "buffer_reschedule_tensor_hoist.hpp"
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/config/context.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/barrier.hpp>
#include <runtime/config.hpp>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(pass.buffer_reschedule_tensor_hoist);

SC_DECL_PASS_INFO(buffer_rescheduling_tensor_hoisting,
        SC_PASS_DEPENDS_ON(index2var, tensor_init, index_flattener,
                dead_write_eliminator, ir_simplifier),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

static const char *processed_by_brth = "processed_by_brth";
static const char *attr_hoisted_tensor = "hoisted";

static bool is_par_for(stmt &curr) {
    return curr.isa<for_loop_c>()
            && curr.static_as<for_loop_c>()->kind_ == for_type::PARALLEL;
}

static bool is_par_for_or_define(stmt &curr) {
    return (is_par_for(curr)
            || (curr.isa<define_c>()
                    && !(curr.checked_as<define_c>()->init_.defined())));
}

static bool not_innermost_par_for(std::vector<stmt> &stmts) {
    return std::all_of(stmts.begin(), stmts.end(), is_par_for_or_define);
}

static bool contains_par_for(std::vector<stmt> &stmts) {
    return std::any_of(stmts.begin(), stmts.end(), is_par_for);
}

class buffer_rescheduling_tensor_hoisting_impl_t : public ir_visitor_t {
private:
    context_ptr ctx_;
    int buffer_schedule_type_;
    bool eliminate_dead_writes_;
    bool do_inplace_opt_;

    buffer_scheduler_t buffer_sche_ {
            ctx_, eliminate_dead_writes_, do_inplace_opt_};

    uint64_t par_for_level_id_ = 0;
    uint64_t define_id_ = 0;

    stmt copy_define(
            define &ori_define, int num_copies, uint64_t par_for_level_id) {
        ++define_id_;
        auto ori_tsr = ori_define->var_.static_as<tensor>();
        std::string name = ori_tsr->name_;
        expr_c shape = num_copies;
        shape = do_cast_and_fold(
                builder::make_mul(num_copies, ori_tsr->dims_[0]));
        std::shared_ptr<static_data_t> new_data_init(nullptr);
        if (ori_tsr->init_value_) {
            auto size = ori_tsr->init_value_->size_;
            if (size == 0) {
                new_data_init = tensor_node::get_zero_tensor_initializer();
            } else {
                std::unique_ptr<char[]> ddata(new char[size * num_copies]);
                for (int i = 0; i < num_copies; i++) {
                    memcpy(ddata.get() + i * size, ori_tsr->init_value_->data_,
                            size);
                }
                new_data_init = std::make_shared<static_data_t>(
                        ddata.get(), size * num_copies);
            }
        }
        auto hoisted_tsr = builder::make_tensor(std::string("hoisted_") + name
                        + "_id" + std::to_string(define_id_),
                {shape}, ori_tsr->elem_dtype_, ori_tsr->address_space_,
                new_data_init);
        hoisted_tsr->attr()[attr_keys::can_be_scheduled] = true;
        hoisted_tsr->attr()[attr_hoisted_tensor] = true;
        auto hoisted_def = builder::make_var_tensor_def_unattached(hoisted_tsr);

        expr thread_id = make_expr<intrin_call_node>(intrin_type::get_group_id,
                std::vector<expr> {par_for_level_id}, any_map_t());

        ori_define->init_ = builder::tensor_ptr(
                hoisted_tsr, {thread_id * ori_tsr->dims_[0]});

        return hoisted_def;
    }

public:
    buffer_rescheduling_tensor_hoisting_impl_t(context_ptr ctx,
            int buffer_schedule_type, bool eliminate_dead_writes,
            bool do_inplace_opt = false)
        : ctx_(std::move(ctx))
        , buffer_schedule_type_(buffer_schedule_type)
        , eliminate_dead_writes_(eliminate_dead_writes)
        , do_inplace_opt_(do_inplace_opt) {}

    using ir_visitor_t::dispatch;
    func_c dispatch(func_c v) override {
        buffer_sche_.top_level_ = v;
        return ir_visitor_t::dispatch(std::move(v));
    }

    stmt_c visit(for_loop_c v) override {
        if (v->attr_ && v->attr_->get_or_else(processed_by_brth, false)) {
            return ir_visitor_t::visit(v);
        }
        if (v->kind_ == for_type::PARALLEL) {
            SC_MODULE_INFO << "Input parallel for:\n" << v;
            std::vector<stmt> &old_body = v->body_.checked_as<stmts>()->seq_;
            bool not_innermost = contains_par_for(old_body);
            if (not_innermost) {
                // there is a nested parallel_for_loop
                // within v, process it first (recursively)
                auto new_body = make_stmt<stmts_node_t>(std::vector<stmt> {});
                for (size_t i = 0; i < old_body.size(); ++i) {
                    stmt &curr = old_body[i];
                    if (is_par_for(curr)) {
                        ++par_for_level_id_;
                        // dispatch: transform the original nested_par_for to
                        // hoisted_defines + new nested_par_for
                        auto new_for = dispatch(curr).remove_const();
                        if (new_for.isa<stmts>()) {
                            auto new_for_stmts
                                    = new_for.checked_as<stmts>()->seq_;
                            new_body->seq_.insert(new_body->seq_.end(),
                                    new_for_stmts.begin(), new_for_stmts.end());
                        } else {
                            new_body->seq_.emplace_back(new_for);
                        }
                        --par_for_level_id_;
                    } else { // other stmts remain unchanged
                        new_body->seq_.emplace_back(old_body[i]);
                    }
                }
                old_body = new_body->seq_;
                SC_MODULE_INFO << "After recursive process:\n" << v;
            }

            // first, do buffer rescheduling on the body
            auto body_stmts = v->body_.checked_as<stmts>();
            body_stmts->attr()[attr_keys::buf_sched_type]
                    = buffer_schedule_type_;
            auto sched_body = buffer_sche_(body_stmts);
            old_body = sched_body.checked_as<stmts>()->seq_;
            SC_MODULE_INFO << "After buffer reschedule:\n" << v;

            if (!not_innermost) {
                SC_MODULE_INFO << "The innermost level only buffer reschedule, "
                                  "do not hoist";
                // using visit of parent class
                auto ret = ir_visitor_t::visit(v).remove_const();
                ret->attr()[processed_by_brth] = true;
                return ret;
            }
            // then, extract defines from par_for body
            std::vector<stmt> defines;
            auto others = make_stmt<stmts_node_t>(std::vector<stmt> {});
            for (size_t i = 0; i < old_body.size(); ++i) {
                stmt &curr = old_body[i];
                if (curr.isa<define_c>()
                        && curr.checked_as<define_c>()->var_.isa<tensor>()
                        && !(curr.checked_as<define_c>()->init_.defined())) {
                    define ori_def = curr.checked_as<define_c>().remove_const();
                    // create a copy of tensor define for each thread, but
                    // not for each iteration
                    int num_copies = v->num_threads_ != 0
                            ? v->num_threads_
                            : runtime_config_t::get().get_num_threads();
                    auto hoisted_def = copy_define(
                            ori_def, num_copies, par_for_level_id_);
                    defines.push_back(hoisted_def);
                    // indexing of ori_def is modified
                    others->seq_.emplace_back(ori_def);
                } else {
                    // will revisit inner for_loops to replace the outdated
                    // indexings
                    others->seq_.push_back(dispatch(curr).remove_const());
                }
            }
            auto for_without_defines = builder::make_for_loop_unattached(
                    v->var_, v->iter_begin_, v->iter_end_, v->step_, others,
                    v->incremental_, v->kind_, v->num_threads_);
            for_without_defines->attr()[processed_by_brth] = true;
            // par_for with defines in body -->
            // hoisted defines + par_for without defines in body
            if (defines.empty()) {
                SC_MODULE_INFO << "Output parallel for:\n"
                               << for_without_defines;
                return for_without_defines;
            } else {
                defines.push_back(for_without_defines);
                stmts new_for = make_stmt<stmts_node_t>(std::move(defines));
                SC_MODULE_INFO << "Output parallel for:\n" << new_for;
                return new_for;
            }
        } else { // not parallel
            return ir_visitor_t::visit(v); // using visit of parent class
        }
    }
};

func_c buffer_rescheduling_tensor_hoisting_t::operator()(func_c f) {
    SC_MODULE_INFO << "Start run buffer_rescheduling_tensor_hoisting pass";
    int buffer_sche_type = ctx_->flags_.buffer_schedule_;
    if (f->attr_) {
        if (f->attr_->has_key(attr_keys::buf_sched_type)) {
            int type = f->attr_->template get<int>(attr_keys::buf_sched_type);
            if (type < 0 || type > 3) {
                SC_MODULE_WARN
                        << "The attr pass.buf_sched_type should be >0 and <3";
            } else {
                buffer_sche_type = type;
            }
        }
    }

    buffer_rescheduling_tensor_hoisting_impl_t impl(
            ctx_, buffer_sche_type, eliminate_dead_writes_, do_inplace_opt_);
    f = impl.dispatch(std::move(f));
    SC_MODULE_INFO << "Finish run buffer_rescheduling_tensor_hoisting pass";
    return f;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

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
#include "nested_parallel_flatten.hpp"
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/barrier.hpp>
#include <runtime/config.hpp>
#include <unordered_map>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(nested_parallel_flattener,
        SC_PASS_DEPENDS_ON(validator, buffer_rescheduling_tensor_hoisting),
        SC_PASS_REQUIRE_STATE(CONST_FOLDED, IR_SIMPLIFIED),
        SC_PASS_REQUIRE_NOT_STATE(), SC_PASS_SET_STATE(),
        SC_PASS_UNSET_STATE(CONST_FOLDED, IR_SIMPLIFIED));

// In this pass, we do and only do nested parallel flattening.
// Tensor hoisting is already done in buffer_rescheduling_tensor_hoisting pass.
class nested_parallel_flatten_impl_t : public ir_visitor_t {
    struct parallel_info_t {
        int num_groups_;
        int threads_per_group_;
        expr thread_id_;
        expr group_id_;
        expr barriers_;
        parallel_info_t(int num_groups, int threads_per_group)
            : num_groups_(num_groups), threads_per_group_(threads_per_group) {}
    };

    std::vector<parallel_info_t> info_;
    std::vector<stmt> *top_level_parallel_seq_ = nullptr;
    expr global_tid_;
    int runtime_threads_ = runtime_config_t::get().get_num_threads();
    int count_ = 0;
    int var_count_ = 0;
    int for_count_ = 0;
    bool cannot_parallel_ = false;
    bool need_pre_barrier_ = false;
    bool need_post_barrier_ = false;

public:
    using ir_visitor_t::dispatch;

    std::string make_name(const char *n) {
        std::string name = n;
        name += std::to_string(count_);
        name += '_';
        name += std::to_string(info_.size());
        name += '_';
        name += std::to_string(++var_count_);
        return name;
    }

    expr get_barrier_for_current_for() {
        COMPILE_ASSERT(top_level_parallel_seq_ && info_.size() > 1UL,
                "Invalid for-loop");
        int num_barrier = 1;
        constexpr uint64_t barrier_size = sizeof(runtime::barrier_t);
        expr idx = info_[info_.size() - 2].group_id_ * barrier_size;
        for (int64_t i = info_.size() - 2; i >= 0; i--) {
            num_barrier *= info_[i].num_groups_;
            if (i != 0) {
                idx = info_[i - 1].group_id_ * (num_barrier * barrier_size)
                        + idx;
            }
        }
        if (info_.back().barriers_.defined()) {
            return builder::tensor_ptr(info_.back().barriers_, {idx});
        }

        info_.back().barriers_ = builder::make_tensor(make_name("_barrier"),
                {num_barrier * barrier_size}, datatypes::u8);
        top_level_parallel_seq_->emplace_back(
                builder::make_var_tensor_def_unattached(
                        info_.back().barriers_));
        top_level_parallel_seq_->emplace_back(builder::make_evaluate_unattached(
                builtin::get_init_barrier_func()(info_.back().barriers_,
                        num_barrier,
                        uint64_t(info_[info_.size() - 2].threads_per_group_))));
        return builder::tensor_ptr(info_.back().barriers_, {idx});
    }

    void gen_call_to_barrier(
            std::vector<stmt> *cur_insert_point, int post_barrier_id) {
        auto b = get_barrier_for_current_for();
        auto the_call = builtin::get_barrier_arrive_func()(
                b, get_ir_null(), get_ir_null());
        cur_insert_point->emplace_back(
                builder::make_evaluate_unattached(the_call));
        if (post_barrier_id >= 0) {
            the_call->attr()["post_barrier_id"] = post_barrier_id;
        }
    }

    bool is_trace_call(const stmt &v) {
        return v.cast<evaluate>()
                .map([](const evaluate &v) { return v->value_.as<call>(); })
                .map([](const call &v) {
                    return dynamic_cast<func_base *>(v->func_.get());
                })
                .filter([](func_base *f) {
                    return f->attr_
                            && f->attr_->get_or_else(
                                    function_attrs::is_trace_func, false);
                })
                .has_value();
    }

    /*
transforming:
void work() {
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < 1000; i++) {
        aaa(i);
        bbb(i);
#pragma omp parallel for num_threads(2)
        for (int j = 0; j < 123; j++) {
            ccc(j);
            ddd(j);
#pragma omp parallel for num_threads(8)
          for (int k = 0; k < 56; k++) {
              eee(k);
              fff(k);
          }
        }
    }
}
=====================================
to:
void work() {
    barrier bar1[4]; // 1 is group level. 4 is the thread number of first group,
                     // each bar sync 16 threads
    barrier bar2[4][2]; // 2 is group level. 2 is the thread number of first
                        // group, each bar sync 8 threads
#pragma omp parallel for num_threads(64)
    for (int tid0 = 0; tid0 < 64; tid0++) {
        int gid1 = tid0 / 16; // group id level 1, 0-3
        int local_tid1 = tid0 % 16; // thread id in group level 1, 0-15
        int start1, end1;
        balance211(gid1, 0, 1000, &start1, &end1);
        // simulate the line : for (int i = 0; i < 1000; i++) {
        // 64-thread group will execute following
        for (int i = start1; i < end1; i++) {
            if (local_tid1 == 0) {
                // only 1 threads in 16-thread group will execute this
                aaa(i);
                bbb(i);
            }
            // start of the code: for (int j = 0; j < 123; j++) {
            // sync 16 threads in the group
            int gid2 = local_tid1 / 8; // group id level 2, 0-1
            int local_tid2 = local_tid1 % 8; // thread id in group level 2
            int start2, end2;
            balance211(gid2, 0, 123, &start2, &end2);
            bar1[gid1].enter_and_wait();
            for (int j = start2; j < end2; j++) {
                if (local_tid2 == 0) {
                    ccc(j);
                    ddd(j);
                }
                // last level has 8 threads in each group
                int gid3 = local_tid1 / 1;
                int local_tid3 = local_tid1 % 1; // thread id in group level 3
                int start3, end3;
                balance211(gid3, 0, 56, &start3, &end3);
                bar2[gid1][gid2].enter_and_wait();
                for (int k = start3; k < end3; k++) {
                    eee(k);
                    fff(k);
                }
                bar2[gid1][gid2].enter_and_wait();
            }
            bar1[gid1].enter_and_wait();
        }
    }
}
    */
    void transform_loop(const for_loop_c &v, int num_threads_parent_group,
            std::vector<stmt> &seq, expr tid0, bool need_pre_barrier, // NOLINT
            bool need_post_barrier) {
        int cur_post_barrier_id = for_count_;
        for_count_++;
        COMPILE_ASSERT(info_.empty() || info_.front().threads_per_group_ != 0,
                "Cannot handle nested parallel-for without num threads in most "
                "outer parallel-for");
        int num_threads = v->num_threads_;
        if (num_threads == 0) { num_threads = num_threads_parent_group; }
        bool divisible = num_threads_parent_group % num_threads == 0;
        uint64_t threads_per_group = num_threads_parent_group / num_threads;
        COMPILE_ASSERT(threads_per_group > 0,
                "Too many threads in this parallel: " << v);

        info_.emplace_back(num_threads, threads_per_group);
        auto gid1 = builder::make_var(datatypes::index, make_name("_gid"));
        info_.back().group_id_ = gid1;
        auto tid1 = builder::make_var(datatypes::index, make_name("_tid"));
        info_.back().thread_id_ = tid1;
        seq.emplace_back(builder::make_var_tensor_def_unattached(
                gid1, linkage::local, tid0 / threads_per_group));
        seq.emplace_back(builder::make_var_tensor_def_unattached(
                tid1, linkage::local, tid0 % threads_per_group));
        std::vector<stmt> *cur_insert_point = &seq;
        // if parent threads per group is not divisible by the current num of
        // threads, gid may excess num_threads, need to skip it
        if (!divisible) {
            auto gid_ok_body = make_stmt<stmts_node_t>(std::vector<stmt> {});
            stmts gid_skip_body;
            if (need_pre_barrier) {
                gid_skip_body = make_stmt<stmts_node_t>(std::vector<stmt> {});
                gen_call_to_barrier(&gid_skip_body->seq_, -1);
            }
            seq.emplace_back(builder::make_if_else_unattached(
                    gid1 < make_expr<constant_node>(
                            uint64_t(num_threads), datatypes::index),
                    gid_ok_body, gid_skip_body));
            cur_insert_point = &gid_ok_body->seq_;
            // the balance211 & for body code will be omited in the if body
        }
        expr begin, end;
        builtin::generate_balance211(
                num_threads, v->iter_begin_, v->iter_end_, v->step_, gid1,
                [&](const char *v) { return make_name(v); }, &begin, nullptr,
                &end, cur_insert_point);
        if (need_pre_barrier) { gen_call_to_barrier(cur_insert_point, -1); }

        auto new_body = make_stmt<stmts_node_t>(std::vector<stmt> {});
        auto step_expr = v->step_->dtype_ == datatypes::index
                ? v->step_
                : constant_folder_t()(
                        builder::make_cast(datatypes::index, v->step_));
        cur_insert_point->emplace_back(builder::make_for_loop_unattached(
                v->var_, begin, end, step_expr, new_body, v->incremental_,
                for_type::NORMAL));

        auto &old_body = v->body_.checked_as<stmts>()->seq_;
        stmts single_thread_body;
        bool local_need_pre_barrier = false;
        // convert old body of v to new_body
        for (size_t i = 0; i < old_body.size(); i++) {
            if (old_body[i].isa<for_loop>()
                    && old_body[i].static_as<for_loop>()->kind_
                            == for_type::PARALLEL) {
                cannot_parallel_ = false;
                // if there are single-threaded sections in the for-body, we
                // need sync the threads
                need_pre_barrier_ = local_need_pre_barrier;
                need_post_barrier_ = false;
                // check if we need to insert post barrier. If the current
                // parallel-for is the last statement in parent parallel-for, we
                // don't need the barrier
                for (size_t n = i + 1; n < old_body.size(); n++) {
                    // if the next stmt is a pure definition, we can ignore it
                    if (old_body[n].isa<define_c>()) {
                        auto &initv = old_body[n].static_as<define_c>()->init_;
                        if (!initv.defined() || initv.isa<constant>()) {
                            continue;
                        }
                    } else if (is_trace_call(old_body[n])) {
                        continue;
                    }

                    // otherwise, we cannot remove the barrier because the stmt
                    // may depend on the current loop
                    need_post_barrier_ = true;
                    break;
                }
                auto body = dispatch(old_body[i])
                                    .remove_const()
                                    .checked_as<stmts>();
                new_body->seq_.insert(new_body->seq_.end(), body->seq_.begin(),
                        body->seq_.end());
                single_thread_body = stmts();
                local_need_pre_barrier = false;
            } else if (old_body[i]
                               .cast<define>()
                               .filter([](const define &v) {
                                   return !v->init_.defined()
                                           || !v->init_.isa<indexing>();
                               })
                               .has_value()) {
                // if the statement is a define node and the init value is not
                // indexing (bypass a LLVM bug)
                if (old_body[i].static_as<define>()->init_.defined()) {
                    single_thread_body = stmts();
                    new_body->seq_.emplace_back(
                            dispatch(old_body[i]).remove_const());
                } else {
                    // if the def node is a pure definition, lift it to the
                    // begining of the block and don't break the current
                    // for-body
                    new_body->seq_.insert(new_body->seq_.begin(),
                            dispatch(old_body[i]).remove_const());
                }
            } else if (is_trace_call(old_body[i])) {
                new_body->seq_.emplace_back(
                        dispatch(old_body[i]).remove_const());
            } else {
                cannot_parallel_ = true;
                auto dispatched = dispatch(old_body[i]).remove_const();
                bool is_set_idle_func_call = dispatched.isa<evaluate>()
                        && dispatched.static_as<evaluate>()
                                   ->value_.isa<intrin_call>()
                        && dispatched.static_as<evaluate>()
                                        ->value_.static_as<intrin_call>()
                                        ->type_
                                == intrin_type::set_thread_idle_func;
                if (is_set_idle_func_call) {
                    dispatched.static_as<evaluate_c>()
                            ->value_.static_as<intrin_call>()
                            ->attr()["post_barrier_id"]
                            = for_count_;
                }
                if (threads_per_group > 1 && !is_set_idle_func_call) {
                    if (dispatched.isa<stmts>()
                            && dispatched.static_as<stmts>()->seq_.empty()) {
                        // if it is a hoisted tensor..., don't need to add to
                        // the body
                    } else {
                        local_need_pre_barrier = true;
                        if (!single_thread_body.defined()) {
                            single_thread_body = make_stmt<stmts_node_t>(
                                    std::vector<stmt> {});
                            new_body->seq_.emplace_back(
                                    builder::make_if_else_unattached(
                                            tid1 == UINT64_C(0),
                                            single_thread_body, stmt()));
                        }
                        single_thread_body->seq_.emplace_back(dispatched);
                    }

                } else {
                    new_body->seq_.emplace_back(dispatched);
                }
            }
        }
        if (need_post_barrier) {
            gen_call_to_barrier(&seq, cur_post_barrier_id);
        }

        info_.pop_back();
    }

    stmt_c visit(for_loop_c v) override {
        COMPILE_ASSERT(
                v->num_threads_ >= 0 && v->num_threads_ <= runtime_threads_,
                "Bad thread count: " << v);
        if (v->kind_ == for_type::PARALLEL) {
            COMPILE_ASSERT(!cannot_parallel_,
                    "Cannot parallel here. The inner parallel for must be "
                    "directly nested in parent parallel-for. "
                            << v);
            if (info_.empty()) {
                if (v->num_threads_ == 0
                        || v->num_threads_ == runtime_threads_) {
                    // first level loop is using all threads, do not transform
                    info_.emplace_back(0, 0);
                    top_level_parallel_seq_ = nullptr;
                    auto ret = ir_visitor_t::visit(v);
                    info_.pop_back();
                    return ret;
                }

                auto body_lv0 = make_stmt<stmts_node_t>(std::vector<stmt> {});
                top_level_parallel_seq_ = &body_lv0->seq_;
                auto body_lv1 = make_stmt<stmts_node_t>(std::vector<stmt> {});
                auto tid0
                        = builder::make_var(datatypes::index, make_name("tid"));
                count_++;
                // now refine the top-level number of threads
                int num_threads = v->num_threads_;
                COMPILE_ASSERT(runtime_threads_ >= num_threads,
                        "num_threads of the loop excesses the total number of "
                        "threads: "
                                << v);
                // use the greatest number of total threads divisible by the
                // current num_threads
                num_threads = runtime_threads_ / num_threads * num_threads;
                global_tid_ = tid0;

                auto for_lv1 = builder::make_for_loop_unattached(tid0,
                        UINT64_C(0), uint64_t(num_threads), UINT64_C(1),
                        body_lv1, true, for_type::PARALLEL);
                transform_loop(
                        v, num_threads, body_lv1->seq_, tid0, false, false);
                body_lv0->seq_.emplace_back(for_lv1);
                if (v->attr_
                        && v->attr_->get_or_else(
                                stmt_attr_key::no_post_barrier, false)) {
                    for_lv1->attr()[stmt_attr_key::no_post_barrier] = true;
                }
                global_tid_ = expr();
                top_level_parallel_seq_ = nullptr;
                cannot_parallel_ = false;
                return body_lv0;
            } else {
                // not first level parallel
                assert(!info_.empty());
                auto &parent_info = info_.back();
                auto body_lv1 = make_stmt<stmts_node_t>(std::vector<stmt> {});
                transform_loop(v, parent_info.threads_per_group_,
                        body_lv1->seq_, parent_info.thread_id_,
                        need_pre_barrier_,
                        need_post_barrier_
                                && !(v->attr_
                                        && v->attr_->get_or_else(
                                                stmt_attr_key::no_post_barrier,
                                                false)));
                return body_lv1;
            }
        } else {
            // not parallel
            return ir_visitor_t::visit(v);
        }
    }

    stmt_c visit(stmts_c v) override {
        std::vector<stmt_c> newseq;
        newseq.reserve(v->seq_.size());
        bool changed = false;
        for (auto &s : v->seq_) {
            auto news = dispatch(s);
            if (!news.ptr_same(s)) { changed = true; }
            if (s.isa<for_loop>() && news.isa<stmts>()) {
                auto &inner = news.static_as<stmts>()->seq_;
                newseq.insert(newseq.end(), inner.begin(), inner.end());
            } else {
                newseq.emplace_back(news);
            }
        }
        if (!changed) {
            return v;
        } else {
            return copy_attr(*v, builder::make_stmts_unattached(newseq));
        }
    }

    expr_c visit(intrin_call_c v) override {
        if (v->type_ == intrin_type::get_group_id) {
            uint64_t level_id
                    = get_const_as_int(v->args_[0].checked_as<constant_c>());
            COMPILE_ASSERT(
                    level_id < info_.size(), "Level of group out of range");
            return info_[level_id].group_id_;
        } else if (v->type_ == intrin_type::get_group_thread_id) {
            int64_t level_id
                    = get_const_as_int(v->args_[0].checked_as<constant_c>());
            COMPILE_ASSERT(level_id < (int64_t)info_.size(),
                    "Level of group out of range");
            if (level_id < 0) {
                if (global_tid_.defined()) {
                    return builder::make_cast(datatypes::s32, global_tid_);
                } else {
                    return v;
                }
            } else {
                return info_[level_id].thread_id_;
            }
        } else {
            return ir_visitor_t::visit(v);
        }
    }
};

func_c nested_parallel_flattener_t::operator()(func_c f) {
    nested_parallel_flatten_impl_t impl;
    f = impl.dispatch(std::move(f));
    return f;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

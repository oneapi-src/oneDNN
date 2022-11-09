/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

namespace sc {

SC_DECL_PASS_INFO(nested_parallel_flattener, SC_PASS_DEPENDS_ON(validator),
        SC_PASS_REQUIRE_STATE(CONST_FOLDED, IR_SIMPLIFIED),
        SC_PASS_REQUIRE_NOT_STATE(), SC_PASS_SET_STATE(),
        SC_PASS_UNSET_STATE(CONST_FOLDED, IR_SIMPLIFIED));

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
    std::vector<stmt> outof_parallel_defs_;
    std::unordered_map<expr, std::pair<expr, expr>> outof_parallel_map_;
    std::vector<stmt> *top_level_parallel_seq_ = nullptr;
    int runtime_threads_ = runtime_config_t::get().get_num_threads();
    int count_ = 0;
    int var_count_ = 0;
    bool cannot_parallel_ = false;
    bool need_pre_barrier_ = false;
    bool need_post_barrier_ = false;

public:
    using ir_visitor_t::dispatch;

    // replace with new indexing.
    expr_c visit(indexing_c v) override {
        auto it = outof_parallel_map_.find(v->ptr_);
        if (it != outof_parallel_map_.end()) {
            assert(v->idx_.size() == 1);
            return builder::make_indexing(it->second.first,
                    {it->second.second + v->idx_[0]}, v->dtype_.lanes_,
                    v->mask_);
        }
        return v;
    }

    std::string make_name(const char *n) {
        std::string name = n;
        name += std::to_string(count_);
        name += '_';
        name += std::to_string(info_.size());
        name += '_';
        name += std::to_string(++var_count_);
        return name;
    }

    void do_generate_balance211(int num_threads, const for_loop_c &v,
            const expr &gid, std::vector<stmt> &out_seq, expr &out_start,
            expr &out_end) {
        auto_caster_t caster;
        constant_folder_t folder;
        auto def_var = [&](const char *name, const expr &init_v) {
            auto ret = builder::make_var(datatypes::index, make_name(name));
            out_seq.emplace_back(builder::make_var_tensor_def_unattached(
                    ret, linkage::local, do_cast_and_fold(init_v)));
            return ret;
        };
        expr the_tid, my_jobs_e, my_jobs_2_e;
        if (v->iter_begin_.isa<constant>() && v->iter_end_.isa<constant>()
                && v->step_.isa<constant>()) {
            // if is constant-for (in most cases)
            uint64_t end = get_const_as_int(v->iter_end_.static_as<constant>());
            uint64_t begin
                    = get_const_as_int(v->iter_begin_.static_as<constant>());
            uint64_t step = get_const_as_int(v->step_.static_as<constant>());
            auto len = end - begin;
            auto num_jobs = utils::divide_and_ceil(len, step);
            uint64_t my_jobs = utils::divide_and_ceil(num_jobs, num_threads);
            COMPILE_ASSERT(my_jobs > 0, "Bad number of jobs");
            if (num_jobs % num_threads == 0) {
                // fast path. the jobs is divisible by the thread number
                out_start = def_var("_start", gid * (my_jobs * step) + begin);
                out_end = def_var("_end", out_start + (my_jobs * step));
                return;
            }
            uint64_t my_jobs_2 = my_jobs - 1;
            the_tid = num_jobs - my_jobs_2 * num_threads;
            my_jobs_e = my_jobs;
            my_jobs_2_e = my_jobs_2;

        } else {
            expr end = v->iter_end_;
            expr begin = v->iter_begin_;
            expr step = v->step_;
            expr len = end - begin;
            auto num_jobs = def_var("num_jobs", (len + step - 1) / step);
            my_jobs_e = def_var(
                    "my_jobs", (num_jobs + (num_threads - 1)) / num_threads);
            // assert(my_jobs > 0);
            my_jobs_2_e = def_var("my_jobs2", my_jobs_e - 1);
            the_tid = def_var("the_tid", num_jobs - my_jobs_2_e * num_threads);
        }
        expr cur_jobs
                = builder::make_select(gid < the_tid, my_jobs_e, my_jobs_2_e);
        expr my_begin = v->iter_begin_
                + builder::make_select(gid <= the_tid, gid * my_jobs_e,
                          the_tid * my_jobs_e + (gid - the_tid) * my_jobs_2_e)
                        * v->step_;
        my_begin = folder(caster(my_begin)).remove_const();
        out_start = def_var("_start", my_begin);

        expr my_end = out_start + cur_jobs * v->step_;
        my_end = folder(caster(my_end)).remove_const();
        out_end = def_var("_end", my_end);
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

    void gen_call_to_barrier(std::vector<stmt> *cur_insert_point) {
        auto b = get_barrier_for_current_for();
        cur_insert_point->emplace_back(builder::make_evaluate_unattached(
                builtin::get_barrier_arrive_func()(b)));
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
                gen_call_to_barrier(&gid_skip_body->seq_);
            }
            seq.emplace_back(builder::make_if_else_unattached(
                    gid1 < make_expr<constant_node>(
                            uint64_t(num_threads), datatypes::index),
                    gid_ok_body, gid_skip_body));
            cur_insert_point = &gid_ok_body->seq_;
            // the balance211 & for body code will be omited in the if body
        }
        expr begin, end;
        do_generate_balance211(
                num_threads, v, gid1, *cur_insert_point, begin, end);
        if (need_pre_barrier) { gen_call_to_barrier(cur_insert_point); }

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
        // convert old body of v to new_body
        for (size_t i = 0; i < old_body.size(); i++) {
            if (old_body[i].isa<for_loop>()
                    && old_body[i].static_as<for_loop>()->kind_
                            == for_type::PARALLEL) {
                cannot_parallel_ = false;
                // if there are single-threaded sections in the for-body, we
                // need sync the threads
                need_pre_barrier_ = single_thread_body.defined();
                need_post_barrier_ = false;
                // check if we need to insert post barrier. If the current
                // parallel-for is the last statement in parent parallel-for, we
                // don't need the barrier
                for (size_t n = i + 1; n < old_body.size(); n++) {
                    // if the next stmt is a pure definition, we can ignore it
                    if (old_body[i].isa<define_c>()) {
                        auto &initv = old_body[i].static_as<define_c>()->init_;
                        if (!initv.defined() || initv.isa<constant>()) {
                            continue;
                        }
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
            } else {
                cannot_parallel_ = true;
                if (threads_per_group > 1) {
                    auto dispatched = dispatch(old_body[i]).remove_const();
                    if (dispatched.isa<stmts>()
                            && dispatched.static_as<stmts>()->seq_.empty()) {
                        // if it is a hoisted tensor..., don't need to add to
                        // the body
                    } else {
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
                    new_body->seq_.emplace_back(
                            dispatch(old_body[i]).remove_const());
                }
            }
        }
        if (need_post_barrier) { gen_call_to_barrier(&seq); }

        info_.pop_back();
    }

    /* Currently only for tensor defined between nested parallel. The tensor out
     of parallel has number of outer parallel loop copies of origin. */
    stmt_c visit(define_c v) override {
        // not to transform
        if (info_.empty() || info_.back().threads_per_group_ <= 1
                || v->var_.isa<var>() || v->init_.defined()) {
            return ir_visitor_t::visit(v);
        }
        // this pass occures after index flatten.
        // number of copies
        auto num_of_copies = runtime_threads_ / info_.back().threads_per_group_;
        expr_c shape = num_of_copies;
        std::string *name;
        sc_data_type_t elem_dtype;
        address_space addspace = address_space::automatic;
        expr offset_idx;

        auto tsr = v->var_.static_as<tensor>();
        assert(tsr->dims_.size() == 1 && tsr->strides_.size() == 1);
        shape = do_cast_and_fold(
                builder::make_mul(num_of_copies, tsr->dims_[0]));
        name = &tsr->name_;
        elem_dtype = tsr->elem_dtype_;
        if (elem_dtype.lanes_ > 1) { return ir_visitor_t::visit(v); }

        addspace = tsr->address_space_;
        auto accu_idx = info_.back().num_groups_;
        offset_idx = info_.back().group_id_;
        for (auto it = info_.rbegin() + 1; it != info_.rend(); it++) {
            offset_idx = offset_idx + it->group_id_ * accu_idx;
            accu_idx = accu_idx * it->num_groups_;
        }
        offset_idx = do_cast_and_fold(
                builder::make_mul(offset_idx, tsr->dims_[0]));
        std::shared_ptr<static_data_t> new_data_init(nullptr);
        if (tsr->init_value_) {
            auto size = tsr->init_value_->size_;
            if (size == 0) {
                new_data_init = tensor_node::get_zero_tensor_initializer();
            } else {
                std::unique_ptr<char[]> ddata(new char[size * num_of_copies]);
                for (int i = 0; i < num_of_copies; i++) {
                    memcpy(ddata.get() + i * size, tsr->init_value_->data_,
                            size);
                }
                new_data_init = std::make_shared<static_data_t>(
                        ddata.get(), size * num_of_copies);
            }
        }
        auto new_tsr
                = builder::make_tensor(std::string("outof_parallel_") + *name,
                        {shape}, elem_dtype, addspace, new_data_init);
        auto new_def = builder::make_var_tensor_def_unattached(new_tsr);

        outof_parallel_map_.insert(
                std::make_pair(v->var_, std::make_pair(new_tsr, offset_idx)));
        outof_parallel_defs_.emplace_back(new_def);
        return builder::make_stmts_unattached({});
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

                auto for_lv1 = builder::make_for_loop_unattached(tid0,
                        UINT64_C(0), uint64_t(num_threads), UINT64_C(1),
                        body_lv1, true, for_type::PARALLEL);
                transform_loop(
                        v, num_threads, body_lv1->seq_, tid0, false, false);
                body_lv0->seq_.insert(body_lv0->seq_.end(),
                        outof_parallel_defs_.begin(),
                        outof_parallel_defs_.end());
                body_lv0->seq_.emplace_back(for_lv1);
                top_level_parallel_seq_ = nullptr;
                cannot_parallel_ = false;
                outof_parallel_defs_.clear();
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
};

func_c nested_parallel_flattener_t::operator()(func_c f) {
    nested_parallel_flatten_impl_t impl;
    f = impl.dispatch(std::move(f));
    return f;
}

} // namespace sc

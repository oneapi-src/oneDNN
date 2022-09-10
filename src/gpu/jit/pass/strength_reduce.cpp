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

#include "gpu/jit/pass/strength_reduce.hpp"

#include "gpu/jit/pass/simplify.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class loop_strength_reducer_t : public ir_mutator_t {
public:
    loop_strength_reducer_t() {
        // Create top-level dummy loop.
        loops_.emplace_back();
    }

    ~loop_strength_reducer_t() override {
        // Sanity check, all stores must be applied.
        ir_assert(post_inc_stores.empty());
    }

    object_t _mutate(const for_t &obj) override {
        loops_.emplace_back(obj);
        auto new_obj = ir_mutator_t::_mutate(obj);
        return inject_stores_and_pop_loop(new_obj);
    }

    object_t _mutate(const let_t &obj) override {
        int loop_level = int(loops_.size()) - 1;
        auto ret = lets_.insert(
                {obj.var, let_info_t(obj.var, obj.value, loop_level)});
        ir_assert(ret.second);
        MAYBE_UNUSED(ret);
        auto new_obj = ir_mutator_t::_mutate(obj);
        lets_.erase(obj.var);
        return new_obj;
    }

    object_t _mutate(const stmt_group_t &obj) override {
        if (obj.body.is<for_t>()) {
            loops_.emplace_back(obj.body);
            const for_t *for_obj = obj.body.as_ptr<for_t>();
            auto body = for_obj ? ir_mutator_t::_mutate(*for_obj) : for_obj;
            if (body.is_same(obj.body)) return obj;
            auto new_obj = stmt_group_t::make(obj.label, body);
            return inject_stores_and_pop_loop(new_obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

    // Pattern to handle:
    //     for (...) {
    //         store(buf_ptr, ...) <- Write (producer).
    //         // ...
    //         stmt_t(..., buf_ptr, ...) <- Read (consumer).
    //     }
    object_t _mutate(const store_t &obj) override {
        if (loops_.size() == 1) return ir_mutator_t::_mutate(obj);

        // Try to reduce strength, moving the store up.
        int init_store_level = -1;
        stmt_t init_store_stmt = obj;
        post_inc_store_info_t post_inc_store(obj);
        for (int level = int(loops_.size()) - 1; level >= 1; level--) {
            auto &loop_info = loops_[level];
            int refs = count_object(loop_info.loop, obj.buf);
            // Producer and consumer - must be 2 references.
            if (refs != 2) break;

            // Try to insert the store before level-th loop.
            auto &store = init_store_stmt.as<store_t>();
            auto &store_value = store.value;
            auto &loop_var = loop_info.loop_var();

            auto cur_value = substitute_let(store_value, level);
            auto next_value = substitute(cur_value, loop_var, loop_var + 1);
            auto inc = simplify(next_value - cur_value);

            // Cannot eliminate loop variable, break.
            if (contains_object(inc, loop_var)) break;

            // Not scalar, break.
            if (!store_value.type().is_scalar()) break;

            // Success, replace store by post-increment store.
            init_store_level = level;

            auto new_store_value
                    = substitute(cur_value, loop_var, loop_info.loop_init());
            init_store_stmt = store_t::make(store.buf, store.off,
                    simplify(new_store_value), store.stride);

            post_inc_store.update(loop_info, inc);
        }

        // Can't do anything, return as is.
        if (init_store_level == -1) return ir_mutator_t::_mutate(obj);

        // Move this store up, remove from here.
        loops_[init_store_level].init_stores.push_back(init_store_stmt);
        if (!post_inc_store.is_empty()) {
            auto ret = post_inc_stores.insert({obj.buf, post_inc_store});
            ir_assert(ret.second);
            MAYBE_UNUSED(ret);
        }
        return stmt_t();
    }

    object_t _mutate(const func_call_t &obj) override {
        for (auto &kv : post_inc_stores) {
            int refs = count_object(obj, kv.first);
            if (refs == 1) {
                auto ret = stmt_seq_t::make(obj, kv.second.stmt());
                post_inc_stores.erase(kv.first);
                return std::move(ret);
            }
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    struct loop_info_t {
        loop_info_t(const stmt_t &loop = {}) : loop(loop) {}

        const expr_t &loop_var() const { return loop.as<for_t>().var; }

        const expr_t &loop_init() const { return loop.as<for_t>().init; }

        const expr_t &loop_bound() const { return loop.as<for_t>().bound; }

        expr_t loop_extent() const { return loop_bound() - loop_init(); }

        // Loop being analyzed.
        stmt_t loop;
        // Stores to insert before the loop.
        std::vector<stmt_t> init_stores;

        std::vector<stmt_t> lets;
    };

    struct let_info_t {
        let_info_t(const expr_t &var, const expr_t &value, int loop_level)
            : var(var), value(value), loop_level(loop_level) {}

        expr_t var;
        expr_t value;
        int loop_level;
    };

    struct post_inc_store_info_t {
        post_inc_store_info_t(const store_t &obj)
            : store(&obj), inc(0), last_iter_cond(true), compensation(0) {}

        stmt_t stmt() const {
            auto load
                    = load_t::make(store->value.type(), store->buf, store->off);
            return store_t::make(store->buf, store->off, load + inc);
        }

        bool is_empty() const { return is_zero(inc); }

        void update(const loop_info_t &loop, const expr_t &loop_inc) {
            inc = simplify(iif_t::make(
                    last_iter_cond, inc - compensation + loop_inc, inc));
            if (last_iter_cond.is_equal(expr_t(true))) {
                last_iter_cond = (loop.loop_var() == loop.loop_bound() - 1);
            } else {
                last_iter_cond = last_iter_cond
                        & (loop.loop_var() == loop.loop_bound() - 1);
            }
            compensation = simplify(loop.loop_extent() * loop_inc);
        }

        const store_t *store;
        expr_t inc;

        expr_t last_iter_cond;
        expr_t compensation;
    };

    // Recursively substitutes all variable from let statements located under
    // the given loop level.
    expr_t substitute_let(const expr_t &_e, int loop_level) const {
        auto e = _e;
        for (;;) {
            bool found = false;
            auto vars = find_unique_objects<var_t>(e);
            for (auto &v : vars) {
                auto it = lets_.find(v);
                if (it == lets_.end()) continue;
                auto &let_info = it->second;
                // Do not substitute top-level let variables.
                if (let_info.loop_level < loop_level) continue;
                found = true;
                e = substitute(e, v, let_info.value);
            }
            if (!found) break;
        }
        return e;
    }

    // Injects initial store statements if any.
    object_t inject_stores_and_pop_loop(const stmt_t &_s) {
        stmt_t s = _s;
        auto &stores = loops_.back().init_stores;
        for (auto it = stores.rbegin(); it != stores.rend(); ++it) {
            s = stmt_seq_t::make(*it, s);
        }
        loops_.pop_back();
        // The top-level dummy loop shouldn't be removed.
        ir_assert(loops_.size() >= 1);
        return std::move(s);
    }

    // Loops, ordered from outermost to innermost. The first loop is dummy, to
    // represent let statements in the top-level scope.
    std::vector<loop_info_t> loops_;

    // Buffers whose references are to be updated.
    object_map_t<expr_t, post_inc_store_info_t> post_inc_stores;

    // Let statements available at the current IR node.
    object_map_t<expr_t, let_info_t> lets_;
};

stmt_t loop_strength_reduce(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = loop_strength_reducer_t().mutate(s);
    trace_pass("loop_strength_reduce", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

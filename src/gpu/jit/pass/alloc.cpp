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

#include "gpu/jit/pass/alloc.hpp"

#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class alloc_lifter_t : public ir_mutator_t {
public:
    alloc_lifter_t(const stmt_t &root, bool reuse_headers)
        : reuse_headers_(reuse_headers) {
        if (!reuse_headers_) return;
        auto calls = find_objects<func_call_t>(root);
        for (auto &c : calls) {
            if (!is_func_call<send_t>(c)) continue;
            auto header_buf = send_t::arg_mem_off(c);
            ir_assert(is_var(header_buf)) << header_buf;
            header_bufs_.insert(header_buf);
        }
    }

    object_t _mutate(const alloc_t &obj) override {
        if (!do_lift(obj)) return ir_mutator_t::_mutate(obj);
        // Remove alloc and insert it before the compute loop.
        allocs_.push_back(&obj);
        return obj.body;
    }

    object_t _mutate(const stmt_group_t &obj) override {
        bool is_compute_loop = (obj.label == stmt_label_t::compute_loop());
        if (is_compute_loop) in_compute_loop_ = true;
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (is_compute_loop) {
            in_compute_loop_ = false;
            // Outermost loop.
            for (auto it = allocs_.rbegin(); it != allocs_.rend(); ++it) {
                auto &a = it->as<alloc_t>();
                new_obj = alloc_t::make(
                        a.buf, a.size, a.kind, a.attrs, new_obj);
            }
            allocs_.resize(0);
        }
        return new_obj;
    }

private:
    bool do_lift(const alloc_t &obj) const {
        if (!in_compute_loop_) return false;
        if (reuse_headers_) {
            bool is_header_alloc = (header_bufs_.count(obj.buf) != 0);
            return !is_header_alloc;
        }
        return true;
    }

    bool reuse_headers_;
    object_set_t<expr_t> header_bufs_;

    bool in_compute_loop_ = false;
    std::vector<stmt_t> allocs_;
};

stmt_t lift_alloc(const stmt_t &s, ir_context_t &ir_ctx, bool reuse_headers) {
    trace_start();
    auto ret = alloc_lifter_t(s, reuse_headers).mutate(s);
    trace_pass("lift_alloc", ret, ir_ctx);
    return ret;
}

class alloc_let_optimizer_t : public ir_mutator_t {
public:
    // Also track alloc_t and for_t to validate all variable usages.
    object_t _mutate(const alloc_t &obj) override {
        return mutate_scope(obj, obj.buf);
    }

    object_t _mutate(const for_t &obj) override {
        level_++;
        auto new_obj = mutate_scope(obj, obj.var);
        level_--;
        return new_obj;
    }

    object_t _mutate(const let_t &obj) override {
        return mutate_scope(obj, obj.var);
    }

    object_t _mutate(const store_t &obj) override {
        auto &base = (obj.buf.is<var_t>() ? obj.buf : obj.buf.as<ptr_t>().base);
        // Do not count store references. If there are only stores to a buffer
        // and no other usages, the buffer can be safely removed.
        skip_var_ = base;
        auto new_obj = ir_mutator_t::_mutate(obj);
        skip_var_ = expr_t();
        return new_obj;
    }

    object_t _mutate(const var_t &obj) override {
        ir_assert(refs_.count(obj) == 1)
                << "Variable is not defined: " << expr_t(&obj);
        if (!skip_var_.is_same(obj)) refs_[&obj].update(increment_, level_);
        return ir_mutator_t::_mutate(obj);
    }

private:
    struct ref_info_t {
        ref_info_t(int level = 0)
            : refs(0), min_level(level), max_level(level) {}

        void update(int increment, int level) {
            refs += increment;
            max_level = std::max(max_level, level);
        }

        bool is_same_level() const { return min_level == max_level; }

        int refs;
        int min_level;
        int max_level;
    };

    template <typename T>
    object_t mutate_scope(const T &obj, const expr_t &var) {
        auto ret = refs_.insert({var, ref_info_t(level_)});
        ir_assert(ret.second) << stmt_t(obj);
        MAYBE_UNUSED(ret);

        auto new_obj = ir_mutator_t::_mutate(obj);
        auto &ref_info = refs_[var];

        if (std::is_same<T, let_t>()) {
            new_obj = mutate_let(new_obj.template as<let_t>(), ref_info);
        } else if (std::is_same<T, alloc_t>()) {
            new_obj = mutate_alloc(new_obj.template as<alloc_t>(), ref_info);
        }

        refs_.erase(var);
        return new_obj;
    }

    object_t mutate_let(const let_t &obj, const ref_info_t &ref_info) {
        ir_assert(ref_info.refs >= 1);
        if (ref_info.refs == 1) {
            // Variable is not used.
            remove_refs(obj);
            return obj.body;
        }
        // Check following conditions to substitute let value:
        // - 2 references: one from producer, one from consumer - means single usage
        // - Consumer and producer are on the same level (same loop)
        // - Variable is not external
        if (ref_info.refs == 2 && ref_info.is_same_level()
                && !obj.value.is_empty()) {
            return substitute(obj.body, obj.var, obj.value);
        }
        return obj;
    }

    object_t mutate_alloc(const alloc_t &obj, const ref_info_t &ref_info) {
        ir_assert(ref_info.refs >= 1);
        // Buffer is not used, single reference from alloc_t itself. Remove
        // stores to the buffer if any.
        if (ref_info.refs == 1) return remove_stores(obj.body, obj.buf);
        return obj;
    }

    void remove_refs(const let_t &obj) {
        increment_ = -1;
        mutate(obj.value);
        increment_ = 1;
    }

    // Removes all nested stores to the buffer.
    stmt_t remove_stores(const stmt_t &stmt, const expr_t &buf) {
        auto ret = stmt;
        auto stores = find_objects<store_t>(stmt);
        for (auto &_s : stores) {
            auto &s = _s.as<store_t>();
            auto &base = (s.buf.is<var_t>() ? s.buf : s.buf.as<ptr_t>().base);
            if (base.is_same(buf)) ret = substitute(ret, _s, stmt_t());
        }
        return ret;
    }

    int increment_ = 1;
    int level_ = 0;

    expr_t skip_var_;
    object_map_t<expr_t, ref_info_t> refs_;
};

stmt_t optimize_alloc_let(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = alloc_let_optimizer_t().mutate(s);
    trace_pass("optimize_alloc_let", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

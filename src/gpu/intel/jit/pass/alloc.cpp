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

#include "gpu/intel/jit/pass/alloc.hpp"

#include "gemmstone/../../dsl/ir/pass/trace.hpp"
#include "gpu/intel/jit/ir/eltwise.hpp"
#include "gpu/intel/jit/ir/legacy.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
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
            gpu_assert(is_var(header_buf)) << header_buf;
            header_bufs_.insert(std::move(header_buf));
        }
    }

    object_t _mutate(const alloc_t &obj) override {
        if (!do_lift(obj)) return ir_mutator_t::_mutate(obj);
        // Remove alloc and insert it before the compute loop.
        allocs_.emplace_back(&obj);
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
    ir::trace_start();
    auto ret = alloc_lifter_t(s, reuse_headers).mutate(s);
    ir::trace_pass("lift_alloc", ret, ir_ctx);
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
        gpu_assert(refs_.count(obj) == 1)
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
        gpu_assert(ret.second) << var;
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
        gpu_assert(ref_info.refs >= 1);
        if (ref_info.refs == 1) {
            // Variable is not used.
            remove_refs(obj);
            return obj.body;
        }
        // Check following conditions to substitute let value:
        // - 2 references: one from producer, one from consumer - means single usage
        // - Consumer and producer are on the same level (same loop)
        // - Variable is not external
        if (ref_info.refs == 2 && ref_info.is_same_level() && obj.value) {
            return substitute(obj.body, obj.var, obj.value);
        }
        return obj;
    }

    object_t mutate_alloc(const alloc_t &obj, const ref_info_t &ref_info) {
        gpu_assert(ref_info.refs >= 1);
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
    ir::trace_start();
    auto ret = alloc_let_optimizer_t().mutate(s);
    ir::trace_pass("optimize_alloc_let", ret, ir_ctx);
    return ret;
}

class alloc_injector_t : public ir_mutator_t {
public:
    alloc_injector_t(const stmt_t &root, const std::vector<stmt_t> &allocs)
        : allocs_(allocs) {
        for (auto &_a : allocs) {
            auto &a = _a.as<alloc_t>();
            if (a.kind != alloc_kind_t::global) gpu_assert(a.size > 0) << _a;
            alloc_map_.insert({a.buf, _a});
            buf_cur_refs_[a.buf] = 0;
        }
        mutate(root);
        buf_total_refs_ = buf_cur_refs_;
        for (auto &kv : buf_cur_refs_)
            kv.second = 0;
        in_ctor_ = false; // NOLINT(cppcoreguidelines-prefer-member-initializer)
    }

#define HANDLE_IR_OBJECT(type) \
    object_t _mutate(const type &obj) override { return mutate_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT
    object_t _mutate(const var_t &obj) override {
        if (alloc_map_.find(obj) != alloc_map_.end()) buf_cur_refs_[obj]++;
        return obj;
    }

private:
    template <typename T>
    object_t mutate_stmt(const T &obj) {
        if (in_ctor_) return ir_mutator_t::_mutate(obj);
        if (obj.template is<stmt_seq_t>()) { return mutate_stmt_seq(obj); }
        auto undef_bufs = get_undef_bufs();
        auto new_obj = ir_mutator_t::_mutate(obj);
        new_obj = maybe_inject(new_obj, undef_bufs);
        return new_obj;
    }

    // Handle stmt_seq_t in a special way:
    // 1. Walk through the sequence and record the first and the last statement
    //    where a buffer is referenced
    // 2. Inject alloc statements according to the usage
    object_t mutate_stmt_seq(const object_t &obj) {
        auto stmt_vec = obj.as<stmt_seq_t>().vec;
        gpu_assert(!stmt_vec.empty());
        int nstmts = (int)stmt_vec.size();
        // Mutate statments and record buffer usage in the form: buf: [first, last].
        object_map_t<expr_t, int> last_undef;
        object_map_t<expr_t, std::pair<int, int>> entries;
        for (int i = 0; i < nstmts; i++) {
            auto &s = stmt_vec[i];
            for (auto &b : get_undef_bufs()) {
                auto it = alloc_map_.find(b);
                if (it == alloc_map_.end() || it->second.is_empty()) continue;
                last_undef[b] = i;
            }
            s = mutate(s);
            for (auto &kv : last_undef) {
                auto &buf = kv.first;
                if (entries.count(buf) != 0) continue;
                if (buf_cur_refs_[buf] == buf_total_refs_[buf]) {
                    entries[buf] = std::make_pair(kv.second, i);
                }
            }
        }
        // Sort buffers based on the number of statements they span. This is to
        // inject more local allocations first.
        std::vector<expr_t> bufs;
        for (auto &kv : entries) {
            if (alloc_map_.at(kv.first).is_empty()) continue;
            bufs.push_back(kv.first);
        }
        std::sort(bufs.begin(), bufs.end(),
                [&](const expr_t &a, const expr_t &b) {
            auto &ea = entries.at(a);
            auto &eb = entries.at(b);
            int a_span = (ea.second - ea.first);
            int b_span = (eb.second - eb.first);
            if (a_span == b_span)
                return a.as<var_t>().name < b.as<var_t>().name;
            return a_span < b_span;
        });
        // Use union-find to incrementally merge statements based on the common
        // buffers.
        std::vector<int> parent(nstmts);
        std::iota(parent.begin(), parent.end(), 0);
        std::function<int(int)> _find;
        std::function<void(int, int)> _union;
        _find = [&](int i) {
            if (parent[i] == i) return i;
            return parent[i] = _find(parent[i]);
        };
        _union = [&](int i, int j) {
            i = _find(i);
            j = _find(j);
            parent[j] = i;
        };
        std::vector<stmt_t> new_stmt_seq = std::move(stmt_vec);
        for (auto &buf : bufs) {
            auto &e = entries.at(buf);
            stmt_t stmt;
            for (int i = e.first; i <= e.second; i++) {
                int idx = _find(i);
                stmt = stmt.append(new_stmt_seq[idx]);
                new_stmt_seq[idx] = stmt_t();
                _union(e.first, i);
            }
            auto it = alloc_map_.find(buf);
            auto &a = it->second.as<alloc_t>();
            stmt = alloc_t::make(a.buf, a.size, a.kind, a.attrs, stmt);
            new_stmt_seq[_find(e.first)] = stmt;
            it->second = stmt_t();
        }
        stmt_t new_obj;
        for (auto &s : new_stmt_seq) {
            if (s.is_empty()) continue;
            new_obj = new_obj.append(s);
        }
        return std::move(new_obj);
    }

    object_set_t<expr_t> get_undef_bufs() const {
        object_set_t<expr_t> ret;
        for (auto &kv : buf_cur_refs_)
            if (kv.second == 0) ret.insert(kv.first);
        return ret;
    }

    object_t maybe_inject(
            const object_t &obj, const object_set_t<expr_t> &undef_bufs) {
        auto new_obj = obj;
        for (auto &kv : alloc_map_) {
            if (kv.second.is_empty()) continue;
            auto &buf = kv.first;
            auto &a = kv.second.as<alloc_t>();
            if (do_inject(buf, undef_bufs)) {
                new_obj = alloc_t::make(
                        a.buf, a.size, a.kind, a.attrs, new_obj);
                kv.second = stmt_t();
            }
        }
        return new_obj;
    }

    bool do_inject(
            const expr_t &buf, const object_set_t<expr_t> &undef_bufs) const {
        if (buf.is_empty()) return false; // Already injected.
        int cur_refs = buf_cur_refs_.at(buf);
        int total_refs = buf_total_refs_.at(buf);
        bool was_undef = (undef_bufs.count(buf) != 0);
        return was_undef && (cur_refs == total_refs);
    }

    bool in_ctor_ = true;
    std::vector<stmt_t> allocs_;
    object_map_t<expr_t, stmt_t> alloc_map_;
    object_map_t<expr_t, int> buf_total_refs_;
    object_map_t<expr_t, int> buf_cur_refs_;
};

class alloc_remover_t : public ir_mutator_t {
public:
    alloc_remover_t(std::vector<stmt_t> &allocs) : allocs_(allocs) {}

    object_t _mutate(const alloc_t &obj) override {
        allocs_.push_back(
                alloc_t::make(obj.buf, obj.size, obj.kind, obj.attrs));
        return mutate(obj.body);
    }

private:
    std::vector<stmt_t> &allocs_;
};

stmt_t inject_alloc_stmts(const stmt_t &stmt, const std::vector<stmt_t> &allocs,
        bool put_innermost, bool update_existing) {
    if (update_existing)
        gpu_assert(put_innermost)
                << "update_existing can be used only with put_innermost.";
    if (!put_innermost) {
        auto ret = stmt;
        for (auto &_a : allocs) {
            auto &a = _a.as<alloc_t>();
            ret = alloc_t::make(a.buf, a.size, a.kind, a.attrs, ret);
        }
        return ret;
    }
    if (update_existing) {
        std::vector<stmt_t> _allocs;
        alloc_remover_t remover(_allocs);
        auto _stmt = remover.mutate(stmt);
        _allocs.insert(_allocs.end(), allocs.begin(), allocs.end());
        return inject_alloc_stmts(_stmt, _allocs, put_innermost);
    }
    alloc_injector_t injector(stmt, allocs);
    return injector.mutate(stmt);
}

stmt_t inject_alloc_stmts(const stmt_t &stmt, const buffer_manager_t &buf_mgr) {
    std::vector<stmt_t> allocs;
    for (auto &e : buf_mgr.entries()) {
        allocs.push_back(e.second.create_alloc_stmt());
    }
    return inject_alloc_stmts(stmt, allocs, /*put_innermost=*/true);
}

stmt_t inject_let_stmts(const stmt_t &stmt, const std::vector<stmt_t> &lets) {
    stmt_t ret = stmt;
    for (auto it = lets.rbegin(); it != lets.rend(); ++it) {
        auto &let = it->as<let_t>();
        ret = let_t::make(let.var, let.value, ret);
    }
    return ret;
}

class access_map_attribute_injector_t : public ir_mutator_t {
public:
    access_map_attribute_injector_t(ir_context_t &ir_ctx) : ir_ctx_(ir_ctx) {}

    object_t _mutate(const alloc_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        if (bufs_.count(obj.buf) == 0) return new_obj;
        auto new_attrs = obj.attrs;
        new_attrs.emplace_back(bufs_.at(obj.buf));
        return alloc_t::make(obj.buf, obj.size, obj.kind, new_attrs,
                new_obj.as<alloc_t>().body);
    }

    object_t _mutate(const load_t &obj) override {
        auto stride = (obj.has_default_stride()) ? obj.type.base().size()
                                                 : obj.stride;
        auto off = (obj.off.is<int_imm_t>()) ? to_int(obj.off) : 0;
        append_access(obj.buf, off, obj.type.elems() * stride);
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const store_t &obj) override {
        auto stride = (obj.has_default_stride())
                ? obj.value.type().base().size()
                : obj.stride;
        auto off = (obj.off.is<int_imm_t>()) ? to_int(obj.off) : 0;
        append_access(obj.buf, off, obj.value.type().elems() * stride);
        return ir_mutator_t::_mutate(obj);
    }

    object_t _mutate(const func_call_t &obj) override {
        auto layout_size = [](int64_t e, const dsl::layout_t &l) {
            return utils::div_up(int(e) * l.type().base().bitsize(), 8);
        };
        auto append_block = [](int64_t s, const dsl::layout_t::block_t &b) {
            return ((b.stride.is_fixed()) ? int64_t(b.stride) : s) * b.size;
        };
        auto min_grf = [&]() { return ir_ctx_.grf_size() * 2; };
        auto append_access_ = [&](const expr_t &e, const dsl::layout_t &l) {
            int total_blk = 1, inner_blk = 1;
            for (auto &b : l.blocks()) {
                total_blk = int(append_block(total_blk, b));
                if (layout_size(inner_blk, l) < min_grf())
                    inner_blk = int(append_block(inner_blk, b));
            }
            if (total_blk % inner_blk == 0) {
                for (int i = 0; i < total_blk; i += inner_blk)
                    append_access(
                            e, layout_size(i, l), layout_size(inner_blk, l));
            } else {
                append_access(e, 0, layout_size(l.elems(), l));
            }
        };
        // DPAS/MAD SRC0-SRC2 are covered by bank conflict alloc but DST isn't
        if (auto *send = obj.func.as_ptr<send_t>()) {
            append_access(send_t::arg_reg_buf(obj), 0, send->payload_size());
        } else if (auto *mad = obj.func.as_ptr<mad_t>()) {
            append_access(mad_t::arg_dst(obj), 0, mad->dst_size());
        } else if (auto *dpas = obj.func.as_ptr<dpas_t>()) {
            append_access(dpas_t::arg_dst(obj), 0, dpas->dst_size());
        } else if (auto *reorder = obj.func.as_ptr<reorder_t>()) {
            append_access_(reorder_t::arg_src_buf(obj), reorder->src_layout);
            append_access_(reorder_t::arg_dst_buf(obj), reorder->dst_layout);
        } else if (auto *reduce = obj.func.as_ptr<reduce_t>()) {
            append_access_(reduce_t::arg_src_buf(obj), reduce->src_layout);
            append_access_(reduce_t::arg_dst_buf(obj), reduce->dst_layout);
        } else if (auto *eltwise = obj.func.as_ptr<eltwise_t>()) {
            const auto dst_dt = (eltwise->dst_dt != ngen::DataType::invalid)
                    ? dsl::type_t(eltwise->dst_dt)
                    : dsl::type_t::f32();
            auto size = to_int(eltwise_t::arg_elems(obj)) * dst_dt.size();
            auto buf = eltwise_t::arg_data(obj);
            for (int i = 0; i < size; i += min_grf())
                append_access(buf, i, std::min(size - i, min_grf()));
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    void append_access(const expr_t &buf, int off, int size) {
        ir::access_map_alloc_attr_t::accesses_t accs;
        auto it = bufs_.find(get_base(buf));
        if (it != bufs_.end())
            accs = it->second.as<ir::access_map_alloc_attr_t>().accs;
        else
            it = bufs_.insert({get_base(buf), {}}).first;
        if (buf.is<ptr_t>() && is_const(buf.as<ptr_t>().off))
            off += to_int(buf.as<ptr_t>().off);
        accs.emplace_back(off, size);
        it->second = ir::access_map_alloc_attr_t::make(accs);
    }

    object_eq_map_t<expr_t, alloc_attr_t> bufs_;
    ir_context_t &ir_ctx_;
};

class var_counter_t : public ir_visitor_t {
public:
    var_counter_t(const object_set_t<expr_t> &vars) {
        for (auto &v : vars) {
            counts[v] = 0;
        }
    }

    void _visit(const var_t &obj) override {
        auto it = counts.find(obj);
        if (it == counts.end()) return;
        it->second++;
    }

    object_map_t<expr_t, int> counts;
};

object_map_t<expr_t, int> count_vars(
        const stmt_t &stmt, const object_set_t<expr_t> &vars) {
    var_counter_t counter(vars);
    counter.visit(stmt);
    return counter.counts;
}

class let_injector_t : public ir_mutator_t {
public:
    object_t _mutate(const stmt_seq_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto &stmt_vec = new_obj.as<stmt_seq_t>().vec;
        int nstmts = (int)stmt_vec.size();
        // 1. Collect total var references for dangling lets.
        object_set_t<expr_t> let_vars;
        for (auto &s : stmt_vec) {
            if (is_dangling_let(s)) {
                auto &var = s.as<let_t>().var;
                let_vars.insert(var);
            }
        }
        if (let_vars.empty()) return new_obj;
        auto total_refs = count_vars(new_obj, let_vars);

        // 2. Find scopes for dangling lets.
        object_map_t<expr_t, stmt_t> var2let;
        object_map_t<stmt_t, int> let_scope_ends;
        object_map_t<expr_t, int> cur_refs;
        for (auto &v : let_vars)
            cur_refs[v] = 0;
        for (int i = 0; i < nstmts; i++) {
            auto &s = stmt_vec[i];
            if (is_dangling_let(s)) {
                var2let[s.as<let_t>().var] = s;
                let_scope_ends[s] = i;
            }
            for (auto &kv : count_vars(s, let_vars)) {
                auto &var = kv.first;
                cur_refs[var] += kv.second;
                if (cur_refs[var] == total_refs[var]) {
                    let_vars.erase(var);
                    let_scope_ends[var2let.at(var)] = i;
                }
            }
        }

        // 3. Nest let statements according to the scopes.
        std::vector<entry_t> entries;
        entries.emplace_back();
        for (int i = 0; i < nstmts; i++) {
            auto &s = stmt_vec[i];
            if (is_dangling_let(s)) {
                entry_t e;
                e.let_stmt = s;
                entries.push_back(e);
            } else {
                entries.back().append(s);
            }
            while (!entries.empty()) {
                auto &last = entries.back();
                if (last.let_stmt.is_empty()) break;
                int end = let_scope_ends.at(last.let_stmt);
                if (end > i) break;
                auto new_stmt = last.make_let();
                entries.pop_back();
                entries.back().append(new_stmt);
            }
        }

        stmt_t ret;
        for (auto &e : entries) {
            gpu_assert(e.let_stmt.is_empty()) << e.let_stmt;
            ret = ret.append(e.body);
        }

        return std::move(ret);
    }

private:
    static bool is_dangling_let(const stmt_t &s) {
        auto *let = s.as_ptr<let_t>();
        return let && let->body.is_empty();
    }

    struct entry_t {
        stmt_t body;
        stmt_t let_stmt;

        stmt_t make_let() const { return replace_stmt_body(let_stmt, body); }

        void append(const stmt_t &s) { body = body.append(s); }
    };
};

stmt_t inject_dangling_let_stmts(const stmt_t &stmt) {
    return let_injector_t().mutate(stmt);
}

stmt_t inject_access_map_attribute(const stmt_t &stmt, ir_context_t &ir_ctx) {
    ir::trace_start();
    auto retn = access_map_attribute_injector_t(ir_ctx).mutate(stmt);
    ir::trace_pass("inject_access_map_attribute", retn, ir_ctx);
    return retn;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

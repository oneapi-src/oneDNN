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

#include "gpu/jit/conv/pipeline.hpp"

#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reorder.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper structure for for_t.
struct loop_info_t {
    loop_info_t() = default;

    loop_info_t(const stmt_t &s) {
        ir_assert(s.is<for_t>()) << s;
        auto &loop = s.as<for_t>();
        stmt = s;
        var = loop.var;
        init_ = loop.init;
        bound_ = loop.bound;
        expr_t e_size = simplify(bound_ - init_);
        ir_assert(is_const(e_size));
        size_ = to_cpp<int>(e_size);
    }

    int init() const {
        ir_assert(is_const(init_));
        return to_cpp<int>(init_);
    }

    int bound() const {
        ir_assert(is_const(bound_));
        return to_cpp<int>(bound_);
    }

    int size() const { return size_; }
    const stmt_t &body() const { return stmt.as<for_t>().body; }
    int unroll() const { return stmt.as<for_t>().unroll; }

    stmt_t stmt;
    expr_t var;

private:
    expr_t init_;
    expr_t bound_;
    int size_;
};

// Iterates through multiple nested loops with fixed bounds. Used to unroll
// such nested loops.
class multi_loop_iterator_t {
public:
    // Ordered from innermost to outermost.
    multi_loop_iterator_t(const std::vector<loop_info_t> &loops)
        : loops_(loops) {
        for (auto &l : loops)
            var_values_.push_back(l.init());
    }

    int var_value(const expr_t &var) const {
        for (size_t i = 0; i < loops_.size(); i++) {
            if (loops_[i].var.is_same(var)) return var_values_[i];
        }
        ir_error_not_expected();
        return 0;
    }

    void advance(int n = 1) {
        if (loops_.empty()) return;
        for (int i_n = 0; i_n < n; i_n++) {
            for (size_t i = 0; i < loops_.size(); i++) {
                auto &l = loops_[i];
                if (++var_values_[i] < l.bound()) break;
                var_values_[i] = l.init();
            }
            ir_assert(var_values_.back() < loops_.back().bound());
        }
    }

    bool is_outer_loop_end() const {
        if (loops_.empty()) return true;
        for (size_t i = 0; i < loops_.size() - 1; i++) {
            auto &l = loops_[i];
            if (var_values_[i] != l.bound() - 1) return false;
        }
        return true;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "multi_loop_iterator_t(";
        for (size_t i = 0; i < loops_.size(); i++) {
            oss << (i != 0 ? ", " : "");
            oss << loops_[i].var << " = " << var_values_[i];
        }
        oss << ")";
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    std::vector<loop_info_t> loops_;
    std::vector<int> var_values_;
};

// Extracts different parts of the compute iteration and verifies the loop nest
// is properly formed and can be further injected with SLM buffering.
class compute_step_visitor_t : public ir_visitor_t {
public:
    stmt_t find_stmt_group(const stmt_label_t &label) const {
        auto groups = find_stmt_groups(label);
        if (groups.empty()) return stmt_t();
        ir_assert(groups.size() == 1);
        return groups[0];
    }

    std::vector<stmt_t> find_stmt_groups(const stmt_label_t &label) const {
        std::vector<stmt_t> ret;
        for (auto &_g : stmt_groups_) {
            auto &g = _g.as<stmt_group_t>();
            if (g.label == label) ret.push_back(_g);
        }
        return ret;
    }

    const std::vector<stmt_t> &inner_let_stmts() const {
        return inner_let_stmts_;
    }

#define HANDLE_IR_OBJECT(type) \
    void _visit(const type &obj) override { visit_stmt(obj); }

    HANDLE_STMT_IR_OBJECTS()

#undef HANDLE_IR_OBJECT

    template <typename T>
    void visit_stmt(const T &obj) {
        bool is_for = obj.template is<for_t>();
        bool is_stmt_group = obj.template is<stmt_group_t>();
        bool is_let = obj.template is<let_t>();
        bool is_stmt_seq = obj.template is<stmt_seq_t>();

        // Loop may contain:
        // - Another loop
        // - Container statement (stmt_seq_t or stmt_group_t)
        // - Let statement (in the innermost loop only)
        // - Barrier
        if (loop_level_ > 0) {
            bool ok = false;
            if (is_for || is_let || is_stmt_group || is_stmt_seq) {
                ok = true;
            } else if (obj.template is<func_call_t>()) {
                auto &call = obj.template as<func_call_t>();
                ok = call.func.is_equal(funcs::barrier_func());
            }

            if (!ok) {
                ir_error_not_expected()
                        << "Found unexpected statement inside loop.\n"
                        << stmt_t(obj);
            }
        }

        bool is_compute_loop = false;
        if (is_stmt_group) {
            auto label = obj.template as<stmt_group_t>().label;
            stmt_groups_.push_back(obj);
            if (utils::one_of(label, stmt_label_t::g2s_load(),
                        stmt_label_t::g2s_store(), stmt_label_t::g2r_load(),
                        stmt_label_t::s2r_load(), stmt_label_t::prefetch(),
                        stmt_label_t::mul())) {
                // Leaf labels, do not visit them.
                return;
            }
            if (label == stmt_label_t::compute_loop()) {
                is_compute_loop = true;
                in_compute_loop_ = true;
            }
        }

        if (is_for && in_compute_loop_) loop_level_++;
        found_loop_ = false;
        ir_visitor_t::_visit(obj);
        if (in_compute_loop_ && is_let) {
            if (found_loop_)
                ir_error_not_expected()
                        << "Let is allowed in the innermost loop only.";

            inner_let_stmts_.push_back(replace_stmt_body(obj, stmt_t()));
        }
        if (is_for && in_compute_loop_) {
            loop_level_--;
            found_loop_ = true;
        }

        if (is_compute_loop) in_compute_loop_ = false;
    }

private:
    bool found_loop_ = false;
    bool in_compute_loop_ = false;
    int loop_level_ = 0;

    std::vector<stmt_t> stmt_groups_;
    std::vector<stmt_t> inner_let_stmts_;
};

// Provides access to different parts of the inner compute iteration.
class compute_step_t {
public:
    compute_step_t(const stmt_t &parent) {
        compute_step_visitor_t v;
        v.visit(parent);

        compute_loop_ = v.find_stmt_group(stmt_label_t::compute_loop());
        g2s_load_ = v.find_stmt_group(stmt_label_t::g2s_load());
        g2s_store_ = v.find_stmt_group(stmt_label_t::g2s_store());
        prefetch_ = v.find_stmt_group(stmt_label_t::prefetch());
        g2r_load_ = v.find_stmt_groups(stmt_label_t::g2r_load());
        s2r_load_ = v.find_stmt_groups(stmt_label_t::s2r_load());
        mul_ = v.find_stmt_groups(stmt_label_t::mul());
        c_zero_out_ = v.find_stmt_group(stmt_label_t::c_zero_out());
        inner_let_stmts_ = v.inner_let_stmts();

        ir_assert(g2r_load_.size() == mul_.size());
        ir_assert(s2r_load_.size() == mul_.size());

        // Assign preload/mul tags to let statements.
        for (auto &_let : inner_let_stmts_) {
            auto &var = _let.as<let_t>().var;
            bool is_preload = (count_object(g2s_load_, var) > 0)
                    || (count_object(prefetch_, var) > 0);
            bool is_mul = count_object(g2r_load_, var) > 0
                    || count_object(mul_, var) > 0;
            if (is_preload) preload_lets_.insert(_let);
            if (is_mul) mul_lets_.insert(_let);
        }

        // Propagate preload/mul tags up based on dependencies between let
        // statements.
        std::vector<let_info_t> let_infos;
        object_set_t<stmt_t> seen;
        std::function<void(const stmt_t &)> propagate;
        propagate = [&](const stmt_t &_let) {
            if (seen.count(_let) > 0) return;
            auto &let = _let.as<let_t>();
            for (auto &_child : inner_let_stmts_) {
                auto &child = _child.as<let_t>();
                if (_child.is_same(_let)) continue;
                if (contains_object(child.value, let.var)) {
                    // Visit child let statements first.
                    propagate(_child);
                    // Propagate child preload/mul values to this let statement.
                    if (is_preload_let(_child)) preload_lets_.insert(_let);
                    if (is_mul_let(_child)) mul_lets_.insert(_let);
                }
            }
            auto let_info = create_let_info(
                    let, is_preload_let(_let), is_mul_let(_let));
            let_infos.push_back(let_info);
            seen.insert(_let);
        };
        for (auto &_let : inner_let_stmts_)
            propagate(_let);

        // Duplicate lets that are used in both preload and mul contexts.
        duplicate_lets(let_infos);
    }

    // See ir_core.hpp for the description.
    const stmt_t &compute_loop() const { return compute_loop_; }
    const stmt_t &g2s_load() const { return g2s_load_; }
    const stmt_t &g2s_store() const { return g2s_store_; }
    const stmt_t &prefetch() const { return prefetch_; }
    const std::vector<stmt_t> &g2r_load() const { return g2r_load_; }
    const std::vector<stmt_t> &s2r_load() const { return s2r_load_; }
    const std::vector<stmt_t> &mul() const { return mul_; }
    const stmt_t &c_zero_out() const { return c_zero_out_; }
    const std::vector<stmt_t> &inner_let_stmts() const {
        return inner_let_stmts_;
    }

    bool is_preload_let(const stmt_t &s) const {
        return preload_lets_.count(s) > 0;
    }
    bool is_mul_let(const stmt_t &s) const { return mul_lets_.count(s) > 0; }

private:
    struct let_info_t {
        let_info_t(const expr_t &var) : var(var) {}

        expr_t var;
        expr_t preload_var;
        expr_t mul_var;

        bool is_preload() const { return !preload_var.is_empty(); }
        bool is_mul() const { return !mul_var.is_empty(); }

        bool needs_update() const { return is_preload() && is_mul(); }
    };

    let_info_t create_let_info(const let_t &let, bool is_preload, bool is_mul) {
        let_info_t info(let.var);
        if (is_preload && !is_mul) {
            info.preload_var = let.var;
        } else if (!is_preload && is_mul) {
            info.mul_var = let.var;
        } else if (is_preload && is_mul) {
            info.preload_var = create_var_with_suffix(let.var, "p");
            info.mul_var = create_var_with_suffix(let.var, "m");
        }
        return info;
    }

    void duplicate_lets(const std::vector<let_info_t> &let_infos) {
        int nlets = int(inner_let_stmts_.size());
        ir_assert(int(let_infos.size()) == nlets);

        std::vector<stmt_t> new_lets;
        for (int i = nlets - 1; i >= 0; i--) {
            auto &info = let_infos[i];
            auto &old_let = inner_let_stmts_[i].as<let_t>();
            if (!info.needs_update()) {
                auto new_value = update_var(old_let.value, let_infos,
                        info.is_preload(), info.is_mul());
                auto new_let = inner_let_stmts_[i];
                if (!new_value.is_same(old_let.value)) {
                    new_let = let_t::make(old_let.var, new_value, old_let.body);
                    if (info.is_preload()) {
                        preload_lets_.erase(&old_let);
                        preload_lets_.insert(new_let);
                    }
                    if (info.is_mul()) {
                        mul_lets_.erase(&old_let);
                        mul_lets_.insert(new_let);
                    }
                }
                new_lets.push_back(new_let);
                continue;
            }

            preload_lets_.erase(&old_let);
            mul_lets_.erase(&old_let);

            auto preload_value
                    = update_var(old_let.value, let_infos, true, false);
            auto preload_let = let_t::make(
                    info.preload_var, preload_value, old_let.body);

            auto mul_value = update_var(old_let.value, let_infos, false, true);
            auto mul_let = let_t::make(info.mul_var, mul_value, old_let.body);

            preload_lets_.insert(preload_let);
            new_lets.push_back(preload_let);

            mul_lets_.insert(mul_let);
            new_lets.push_back(mul_let);

            // Update statements.
            g2s_load_ = update_var(g2s_load_, let_infos, true, false);
            g2s_store_ = update_var(g2s_store_, let_infos, true, false);
            prefetch_ = update_var(prefetch_, let_infos, true, false);
            g2r_load_ = update_var(g2r_load_, let_infos, false, true);
            s2r_load_ = update_var(s2r_load_, let_infos, false, true);
            mul_ = update_var(mul_, let_infos, false, true);
        }

        std::reverse(new_lets.begin(), new_lets.end());
        inner_let_stmts_ = new_lets;
    }

    template <typename T>
    static std::vector<T> update_var(const std::vector<T> &vec,
            const std::vector<let_info_t> &let_infos, bool is_preload,
            bool is_mul) {
        std::vector<T> ret;
        for (auto &v : vec)
            ret.push_back(update_var(v, let_infos, is_preload, is_mul));
        return ret;
    }

    static object_t update_var(const object_t &obj,
            const std::vector<let_info_t> &let_infos, bool is_preload,
            bool is_mul) {
        auto ret = obj;
        for (auto &info : let_infos) {
            if (!info.needs_update()) continue;
            if (!contains_object(ret, info.var)) continue;
            if (is_preload) {
                ir_assert(info.is_preload());
                ret = substitute(ret, info.var, info.preload_var);
            } else if (is_mul) {
                ir_assert(info.is_mul());
                ret = substitute(ret, info.var, info.mul_var);
            }
        }
        return ret;
    }

    static expr_t create_var_with_suffix(
            const expr_t &_var, const std::string &suffix) {
        auto &var = _var.as<var_t>();
        auto new_name = var.name + "_" + suffix;
        return var_t::make(var.type, new_name);
    }

    stmt_t compute_loop_;
    stmt_t g2s_load_;
    stmt_t g2s_store_;
    stmt_t prefetch_;
    std::vector<stmt_t> g2r_load_;
    std::vector<stmt_t> s2r_load_;
    std::vector<stmt_t> mul_;
    stmt_t c_zero_out_;

    std::vector<stmt_t> inner_let_stmts_;

    // Due to loop unrolling the inner let statements may depend on different
    // indices of the outer loops. There are two contexts:
    // - "preload" loop iteration, e.g. index I
    // - "multiplication" loop iteration, e.g. index (I + nbuf)
    // Preloads (either via SLM or via prefetches) for the corresponding
    // multiplication are executed several iterations before the real
    // multiplication. That's why we need to know exactly in which context the
    // given let statement is used. It might be that the same variable is used
    // from two different contexts. In this case it is duplicated and
    // initialized with different values for each case.
    object_set_t<stmt_t> preload_lets_;
    object_set_t<stmt_t> mul_lets_;
};

// Helper class to access the outer loop index after pipelining. Pipelining
// in general requires tracking two versions of a loop index:
// - Multiplication version - corresponding to the iteration that is currently
//   used for multiplication
// - Preload version - corresponding to the iteration that is currently used
//   for preload for one of the next multiplications
// The multiplication version is a few steps behind the preload version.
class outer_loop_info_t : public loop_info_t {
public:
    outer_loop_info_t() = default;

    outer_loop_info_t(const stmt_t &s, ir_context_t &ir_ctx) : loop_info_t(s) {
        // Outer loop may not be used for unrolling hence loop iterations must
        // not use its index. If this doesn't hold, introduce a GRF buffer to
        // represent that variable and apply post-increment updates after each
        // outer loop iteration.
        if (count_object(s.as<for_t>().body, var) != 0) {
            has_var_refs_ = true;
            mul_var_buf_ = ir_ctx.create_tmp_var(
                    type_t::byte_ptr(), var.as<var_t>().name + "_mul_buf");
            preload_var_buf_ = ir_ctx.create_tmp_var(
                    type_t::byte_ptr(), var.as<var_t>().name + "_preload_buf");

            auto mul_alloc = alloc_t::make(
                    mul_var_buf_, var.type().size(), alloc_kind_t::grf);
            auto preload_alloc = alloc_t::make(
                    preload_var_buf_, var.type().size(), alloc_kind_t::grf);
            allocs_.push_back(mul_alloc);
            allocs_.push_back(preload_alloc);

            auto mul_init = store_t::make(mul_var_buf_, 0, init());
            auto preload_init = store_t::make(preload_var_buf_, 0, init());
            init_stmt_ = mul_init.append(preload_init);

            mul_post_inc_stmt_
                    = store_t::make(mul_var_buf_, 0, mul_var_load() + 1);
            preload_post_inc_stmt_ = store_t::make(
                    preload_var_buf_, 0, preload_var_load() + 1);
        }
    }

    bool has_var_refs() const { return has_var_refs_; }

    expr_t mul_var_load() const {
        return load_t::make(var.type(), mul_var_buf_, 0);
    }
    expr_t preload_var_load() const {
        return load_t::make(var.type(), preload_var_buf_, 0);
    }

    stmt_t inject_alloc_stmts(const stmt_t &stmt) const {
        return jit::inject_alloc_stmts(stmt, allocs_);
    }

    const stmt_t &init_stmt() const { return init_stmt_; }

    const stmt_t &mul_post_inc_stmt() const { return mul_post_inc_stmt_; }
    const stmt_t &preload_post_inc_stmt() const {
        return preload_post_inc_stmt_;
    }

private:
    bool has_var_refs_ = false;

    // Helper expressions/statements to partially unroll the loop.
    expr_t mul_var_buf_;
    expr_t preload_var_buf_;
    std::vector<stmt_t> allocs_;
    stmt_t init_stmt_;
    stmt_t mul_post_inc_stmt_;
    stmt_t preload_post_inc_stmt_;
};

class compute_loop_nest_visitor_t : public ir_visitor_t {
public:
    int compute_loop_level() const { return compute_loop_level_; }

    const std::vector<loop_info_t> &loops() const { return loops_; }

    void _visit(const stmt_group_t &obj) override {
        bool is_compute_loop = (obj.label == stmt_label_t::compute_loop());
        if (is_compute_loop) {
            in_compute_loop_ = true;
            compute_loop_level_ = level_;
        }
        ir_visitor_t::_visit(obj);
        if (is_compute_loop) in_compute_loop_ = false;
    }

    void _visit(const for_t &obj) override {
        level_++;
        ir_visitor_t::_visit(obj);
        if (in_compute_loop_) loops_.emplace_back(obj);
        level_--;
    }

private:
    bool in_compute_loop_ = false;
    int compute_loop_level_ = -1;
    std::vector<loop_info_t> loops_;
    int level_ = 0;
};

// Helper class to work with loop nest of the compute loop.
class compute_loop_nest_t {
public:
    compute_loop_nest_t() = default;

    compute_loop_nest_t(const stmt_t &root, ir_context_t &ir_ctx)
        : root_(root) {
        compute_loop_nest_visitor_t visitor;
        visitor.visit(root);

        compute_loop_level_ = visitor.compute_loop_level();
        loops_ = visitor.loops();

        if (loops_.empty()) {
            outer_loop_size_ = 1;
            return;
        }

        outer_loop_ = outer_loop_info_t(loops_.back().stmt, ir_ctx);
        outer_loop_size_ = outer_loop_.size();
    }

    // Returns the loop level of the compute_loop statement group corresponding
    // to the number of outer loops.
    int compute_loop_level() const { return compute_loop_level_; }

    // Returns loops inside compute_loop statement group.
    const std::vector<loop_info_t> &loops() const { return loops_; }

    // Number of iterations of all loops.
    int size() const {
        int ret = 1;
        for (auto &l : loops_)
            ret *= l.size();
        return ret;
    }

    // Number of iterations in the outermost loop (see comments in ctor).
    int outer_loop_size() const { return outer_loop_size_; }

    const outer_loop_info_t &outer_loop_info() const { return outer_loop_; }

    template <typename F>
    void for_each_loop_var(const F &f) const {
        for (auto &l : loops_)
            f(l.var);
    }

    // Number of iterations of all loops except the outermost.
    int inner_loops_size() const { return size() / outer_loop_size(); }

private:
    stmt_t root_;
    int compute_loop_level_ = -1;
    std::vector<loop_info_t> loops_;

    int outer_loop_size_;
    outer_loop_info_t outer_loop_;
};

struct compute_params_t {
    compute_params_t() = default;

    compute_params_t(int slm_bufs, int gmem_bufs, int slm_buf_size,
            int prefetch_bufs, int inner_loops_iters)
        : slm_bufs(slm_bufs)
        , gmem_bufs(gmem_bufs)
        , slm_buf_size(slm_buf_size)
        , prefetch_bufs(prefetch_bufs) {
        use_slm = (slm_buf_size > 0);
        use_prefetch = (prefetch_bufs > 0);
        ir_assert(!use_slm || !use_prefetch)
                << "Can't have both SLM buffering and prefetch enabled.";
        if (use_slm) {
            ir_assert(utils::one_of(slm_bufs, 1, 2, 3));
            ir_assert(utils::one_of(gmem_bufs, 1, 2));
            preload_bufs = slm_bufs;
            unroll = math::lcm(slm_bufs * gmem_bufs, inner_loops_iters);
        } else if (use_prefetch) {
            preload_bufs = prefetch_bufs;
            ir_assert(slm_bufs == 0);
            ir_assert(gmem_bufs == 0);
            unroll = math::lcm(prefetch_bufs, inner_loops_iters);
        } else {
            preload_bufs = 0;
            ir_assert(slm_bufs == 0);
            ir_assert(gmem_bufs == 0);
            unroll = inner_loops_iters;
        }
    }

    int slm_bufs;
    int gmem_bufs;
    int slm_buf_size;
    int prefetch_bufs;
    int preload_bufs;
    int unroll;

    bool use_slm;
    bool use_prefetch;
};

// Helper class to implement SLM buffering.
class compute_iterator_t {
public:
    compute_iterator_t(const compute_params_t &params,
            const compute_loop_nest_t &loop_nest)
        : params(params)
        , preload_loop_it(loop_nest.loops())
        , mul_loop_it(loop_nest.loops()) {

        int compute_iters = loop_nest.size();
        iters = compute_iters;
        ir_assert(iters >= 1) << "Empty loop is not expected.";

        iters += std::max(0, preload_bufs() - 1) + std::max(0, gmem_bufs() - 1);
        ramp_up_iters
                = std::max(1, preload_bufs() + std::max(0, gmem_bufs() - 1));
        ramp_down_iters = std::min(
                std::max(0, preload_bufs() - 1) + std::max(0, gmem_bufs() - 1),
                iters - ramp_up_iters);
        body_iters = iters - ramp_up_iters - ramp_down_iters;
        body_iters = utils::rnd_dn(body_iters, params.unroll);
        ramp_down_iters = iters - ramp_up_iters - body_iters;

        ir_assert(ramp_up_iters + body_iters + ramp_down_iters == iters);

        iter = 0;
        linear_id = 0;
        riter = iters - 1;
    }

    int unroll() const { return params.unroll; }

    int preload_bufs() const { return params.preload_bufs; }

    int slm_bufs() const { return params.slm_bufs; }

    int gmem_bufs() const { return params.gmem_bufs; }

    compute_iterator_t &operator++() {
        if (do_preload()) preload_loop_it.advance();
        if (do_mul()) mul_loop_it.advance();
        ++iter;
        ++linear_id;
        --riter;
        return *this;
    }

    void advance(int n) {
        if (n == 0) return;

        ir_assert(n % params.unroll == 0);
        ir_assert(iter + n <= iters);

        if (preload_bufs() > 0) ir_assert(do_preload());
        ir_assert(do_mul());

        iter += n;
        riter -= n;

        if (preload_bufs() > 0) preload_loop_it.advance(n);
        mul_loop_it.advance(n);
    }

    bool do_mul() const {
        return iter >= std::max(0, preload_bufs() - 1)
                + std::max(0, gmem_bufs() - 1);
    }

    bool is_first_mul() const {
        return iter
                == std::max(0, preload_bufs() - 1)
                + std::max(0, gmem_bufs() - 1);
    }
    bool is_last_mul() const { return riter == 0; }

    bool is_last_g2s_store() const {
        if (!do_g2s_store()) return false;
        return riter == slm_bufs() - 1;
    }

    bool is_last_preload() const {
        if (!do_preload()) return false;
        return riter == (preload_bufs() - 1) + std::max(0, gmem_bufs() - 1);
    }

    bool is_last_g2s_load() const {
        if (!do_g2s_load()) return false;
        return is_last_preload();
    }

    bool is_last_prefetch() const {
        if (!do_prefetch()) return false;
        return is_last_preload();
    }

    bool do_preload() const {
        if (preload_bufs() == 0) return false;
        return riter >= (preload_bufs() - 1) + std::max(0, gmem_bufs() - 1);
    }

    bool do_prefetch() const {
        if (!params.use_prefetch) return false;
        return do_preload();
    }

    bool do_g2s_load() const {
        if (!params.use_slm) return false;
        return do_preload();
    }

    bool do_g2s_store() const {
        if (!params.use_slm) return false;
        ir_assert(gmem_bufs() >= 1);
        return iter >= (gmem_bufs() - 1) && riter >= (slm_bufs() - 1);
    }

    int gmem_write_buf_index() const {
        ir_assert(do_g2s_load());
        return iter % gmem_bufs();
    }

    int gmem_read_buf_index() const {
        ir_assert(do_g2s_store());
        return (iter - (gmem_bufs() - 1)) % gmem_bufs();
    }

    int slm_read_offset_update() const {
        ir_assert(params.use_slm);
        ir_assert(do_mul());

        int slm_iter = iter - (gmem_bufs() - 1) - (slm_bufs() - 1);
        int cur_slm_idx = slm_iter % slm_bufs();
        int next_slm_idx = (slm_iter + 1) % slm_bufs();
        int ret = next_slm_idx * params.slm_buf_size
                - cur_slm_idx * params.slm_buf_size;
        return ret;
    }

    int slm_write_offset_update() const {
        ir_assert(params.use_slm);
        ir_assert(do_g2s_store());

        int slm_iter = iter - (gmem_bufs() - 1);
        int cur_slm_idx = slm_iter % slm_bufs();
        int next_slm_idx = (slm_iter + 1) % slm_bufs();
        int ret = next_slm_idx * params.slm_buf_size
                - cur_slm_idx * params.slm_buf_size;
        return ret;
    }

    compute_params_t params;
    multi_loop_iterator_t preload_loop_it;
    multi_loop_iterator_t mul_loop_it;

    // ramp_up_iters + body_iters + ramp_down_iters == iters
    int iters;
    int ramp_up_iters;
    int body_iters;
    int ramp_down_iters;

    // Invariant: iter + riter = iters - 1
    int iter;
    int riter;

    int linear_id;
};

// Basic LRU SBID allocator, tries to use the same SBIDs for the same GRF
// buffers.
class sbid_manager_t {
public:
    sbid_manager_t(ngen::HW hw = ngen::HW::Unknown)
        : sbid_count_(hw >= ngen::HW::XeHPC ? 32 : 16)
        , tuple_func_(builtin_t::make("tuple")) {}

    ngen_proxy::SBID get_sbid(const expr_t &buf, int index = 0) {
        auto key = tuple_func_.call({buf, expr_t(index)});

        int free_idx = -1;
        for (int i = 0; i < sbid_count_; i++) {
            auto &e = entries_[i];
            if (key.is_equal(e.key)) {
                e.time = cur_time_++;
                return ngen_proxy::SBID(i);
            }
            if (free_idx == -1 && e.key.is_empty()) free_idx = i;
        }

        // Not found but there is a free SBID.
        if (free_idx != -1) {
            entries_[free_idx] = {key, cur_time_++};
            return ngen_proxy::SBID(free_idx);
        }

        // Find the oldest SBID and use it.
        int old_idx = 0;
        int old_time = entries_[0].time;
        for (int i = 1; i < sbid_count_; i++) {
            if (entries_[i].time < old_time) {
                old_idx = i;
                old_time = entries_[i].time;
            }
        }

        entries_[old_idx] = entry_t({key, cur_time_++});
        return ngen_proxy::SBID(old_idx);
    }

private:
    struct entry_t {
        stmt_t key;
        int time;
    };

    static const int max_sbid_count = 32;
    std::array<entry_t, max_sbid_count> entries_;

    int sbid_count_ = 0;
    func_t tuple_func_;
    int cur_time_ = 0;
};

// Helper to assign SBIDs to IR function calls.
class sbid_assigner_t {
public:
    sbid_assigner_t(ngen::HW hw) : local_sbid_mgr_(hw) {}

    sbid_assigner_t(sbid_manager_t &external_sbid_mgr)
        : external_sbid_mgr_(&external_sbid_mgr) {}

    stmt_t assign(const stmt_t &stmt) {
        auto stmt_vec = flatten_statements(stmt);
        stmt_t ret = stmt;
        int prefetch_idx = 0;
        for (auto &_s : stmt_vec) {
            if (!_s.is<func_call_t>()) continue;
            auto s = _s;
            if (is_func_call<send_t>(s)) {
                auto &send = s.as<func_call_t>().func.as<send_t>();
                int idx = (send.is_prefetch() || send.is_prefetch_2d()
                                ? prefetch_idx++
                                : 0);
                auto sbid = get_sbid(send_t::arg_reg_buf(s), idx);
                s = update_call_with_sbid(s, sbid);
            } else if (is_func_call<dpas_t>(s)) {
                auto &c = s.as<func_call_t>();
                auto *mod_attr = c.attr.as_ptr<instruction_modifier_attr_t>();
                if (!c.func.as<dpas_t>().is_dp4a() && // dp4a-s do not need SBID
                        (!mod_attr || !mod_attr->mod.is_atomic)) {
                    // Last dpas in Atomic chain.
                    auto sbid = get_sbid(dpas_t::arg_src1(s));
                    s = update_call_with_sbid(s, sbid);
                }
            } else if (s.is<func_call_t>()) {
                auto &c = s.as<func_call_t>();
                if (c.func.is_equal(funcs::signal_func())
                        || c.func.is_equal(funcs::slm_fence_func())
                        || c.func.is_equal(funcs::barrier_func())) {
                    // Use 0 as the key for signals and SLM fences.
                    auto sbid = get_sbid(expr_t(0));
                    s = update_call_with_sbid(s, sbid);
                }
            } else {
                ir_error_not_expected() << s;
            }
            ret = substitute(ret, _s, s);
        }
        return ret;
    }

private:
    ngen_proxy::SBID get_sbid(const expr_t &ptr, int index = 0) {
        auto &sbid_mgr
                = (external_sbid_mgr_ ? *external_sbid_mgr_ : local_sbid_mgr_);
        return sbid_mgr.get_sbid(ptr, index);
    }

    static stmt_t update_call_with_sbid(
            const stmt_t &s, const ngen_proxy::SBID &sbid) {
        return instruction_modifier_attr_t::make(
                ngen_proxy::InstructionModifier().with_sbid(sbid))
                .apply_to(s);
    }

    sbid_manager_t local_sbid_mgr_;
    sbid_manager_t *external_sbid_mgr_ = nullptr;
};

// Work around due to limited scoping functionality in current generator
// Prepends all newly created var_t names with given prefix.
class var_prepender_t : public ir_mutator_t {
public:
    var_prepender_t(const std::string &prefix) : prefix_(prefix) {}
    object_t _mutate(const for_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto new_var = var_t::make(
                obj.var.type(), prefix_ + obj.var.as<var_t>().name);
        new_obj = substitute(new_obj, obj.var, new_var);
        return new_obj;
    }
    object_t _mutate(const let_t &obj) override {
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto new_var = var_t::make(
                obj.var.type(), prefix_ + obj.var.as<var_t>().name);
        new_obj = substitute(new_obj, obj.var, new_var);
        return new_obj;
    }

private:
    std::string prefix_;
};

object_t prepend_new_vars(const object_t &root, const std::string &prefix) {
    var_prepender_t mutator(prefix);
    return mutator.mutate(root);
}

// Perform pipelining operation. The goal is to transform
// the loop structure from:
//
// for i in range(init, bound):
//     A_block(i);
//     B_block(i);
//
// to the following
//
// for i in range(init, init + length):
//     A_block(i);
// for i in range(init, bound):
//     if (i < bound - length):
//         A_block(i + length);
//      B_block(i);
//
// Since A_block and B_block have to be independent to maintain correctness,
// this transform ignores the operations within the for_loop and relies on a
// correct substitution for A_block and B_block.

struct pipeline_ctx_t {
    pipeline_ctx_t(const stmt_t &prologue, const stmt_t &body)
        : prologue_(prologue), body_(body) {}
    stmt_t stmt() const { return prologue_.append(body_); }
    stmt_t prologue() { return prologue_; }
    stmt_t body() { return body_; }

private:
    stmt_t prologue_;
    stmt_t body_;
};

pipeline_ctx_t pipeline(
        int length, const loop_info_t &loop, stmt_t A_block, stmt_t B_block) {

    expr_t idx = loop.var;
    int bound = loop.bound();
    int init = loop.init();

    int pipe_len = std::min(init + length, bound);

    stmt_t prologue = prepend_new_vars(
            for_t::make(idx, init, pipe_len, A_block,
                    pipe_len <= loop.unroll() ? pipe_len : 1),
            "prefetch_");

    expr_t A_idx = idx + pipe_len;
    stmt_t body;
    if (init < (bound - pipe_len))
        body = if_t::make(
                idx < (bound - pipe_len), substitute(A_block, idx, A_idx));
    body = body.append(B_block);
    body = for_t::make(idx, init, bound, body, loop.unroll());

    return pipeline_ctx_t(prologue, body);
}

class prefetch_pipeliner_t {
public:
    prefetch_pipeliner_t(
            const stmt_t &root, const conv_config_t &cfg, ir_context_t &ir_ctx)
        : root_(root), cfg_(cfg), ir_ctx_(ir_ctx) {}
    stmt_t inject() {
        auto compute_loop_stmt
                = find_stmt_group(root_, stmt_label_t::compute_loop());
        if (!compute_loop_stmt.has_value()) return root_;
        auto compute_loop = compute_loop_stmt.value();
        auto loop_nest = compute_loop_nest_t(compute_loop, ir_ctx_);
        auto &loops = loop_nest.loops();

        // No loops to pipeline
        if (loops.size() == 0) return root_;
        auto &loop_body = loops[0].body();

        auto A_block_stmt
                = find_stmt_group(loop_body, stmt_label_t::prefetch());
        if (!A_block_stmt.has_value()) return root_;
        auto A_block = A_block_stmt.value();
        auto B_block = remove_stmt_group(loop_body, stmt_label_t::prefetch());
        size_t prefetch_count = 0;
        size_t max_nested_prefetch = 2;
        for (size_t i = 0; i < loops.size(); i++) {
            if (prefetch_count < max_nested_prefetch) {
                if (!contains_object(A_block, loops[i].var)) {
                    // No point in prefetching a constant in a loop
                    B_block = for_t::make(loops[i].var, loops[i].init(),
                            loops[i].bound(), B_block, loops[i].unroll());
                    continue;
                }

                auto next = pipeline(
                        cfg_.prefetch().bufs(), loops[i], A_block, B_block);
                A_block = next.prologue();
                B_block = next.body();
                prefetch_count++;

            } else {
                B_block = for_t::make(loops[i].var, loops[i].init(),
                        loops[i].bound(), A_block.append(B_block),
                        loops[i].unroll());
                A_block = stmt_t();
            }
        }
        return substitute(root_, compute_loop, A_block.append(B_block));
    }

private:
    stmt_t root_;
    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
};

stmt_t inject_prefetch_pipeline(
        const stmt_t &s, ir_context_t &ir_ctx, const conv_config_t &cfg) {
    trace_start();
    auto ret = prefetch_pipeliner_t(s, cfg, ir_ctx).inject();
    trace_pass("inject_prefetch_pipeline", ret, ir_ctx);
    return ret;
}

// Helper class to handle synchronization between threads for cooperative SLM
// load and stores for double and triple buffering. Name conventions:
// - Lx step - load from global memory to GRF (to be stored in SLM buffer x)
// - Mx step - load from SLM buffer x to GRF and multiplication
// - Sx step - store from GRF to SLM buffer x
// - Rx event - SLM buffer x is available for reading
// - Wx event - SLM buffer x is available for writing
// Scheme for single buffering:
//     L0
//     barrier
//     S0
//     barrier
//     M0
// Schemes for double and triple buffering are below.
class slm_sync_manager_t {
public:
    slm_sync_manager_t(const conv_config_t &cfg, bool with_unroll)
        : slm_bufs_(cfg.slm().bufs())
        , gmem_bufs_(cfg.slm().gmem_bufs())
        , with_unroll_(with_unroll) {
        switch (slm_bufs_) {
            case 2: ver_ = version_t::x2; break;
            case 3: ver_ = version_t::x3_v3; break;
            default: ver_ = version_t::undef;
        }
        if (cfg.slm().sync_version() != -1) {
            ver_ = (version_t)cfg.slm().sync_version();
        }
        switch (slm_bufs_) {
            case 2: ir_assert(ver_ == version_t::x2); break;
            case 3:
                ir_assert(utils::one_of(ver_, version_t::x3_v1,
                        version_t::x3_v2, version_t::x3_v3));
                break;
            default: ir_assert(ver_ == version_t::undef);
        }
    }

    stmt_t before_loop_prepend(const stmt_t &_s) const {
        if (with_unroll_) return _s;
        auto s = _s;
        if (is_x3_v1() || is_x3_v2() || is_x3_v3()) {
            // Emit initial signal, to match wait-signal pairs in the loop.
            s = funcs::signal().append(s);
        }
        return s;
    }

    stmt_t after_loop(const stmt_t &_s) const {
        auto s = _s;
        if (slm_bufs_ == 3) {
            s = s.append(funcs::barrier_wait());
            // Wait with V3 guarantees that all SLM writes are synced, other
            // versions need additional synchronization.
            if (!is_x3_v3()) s = s.append(funcs::barrier());
        }
        return s;
    }

    stmt_t before_L(const stmt_t &_s, bool do_mul) const {
        bool emit = false;
        if (!with_unroll_) emit = true;
        if (with_unroll_ && do_mul) emit = true;

        auto s = _s;
        if (is_x3_v2() && emit) { s = s.append(funcs::barrier_wait()); }

        return s;
    }

    stmt_t before_L_prepend(const stmt_t &_s, bool do_mul) const {
        return before_L(stmt_t(), do_mul).append(_s);
    }

    stmt_t after_L(const stmt_t &_s, bool do_mul) const {
        bool emit = false;
        if (!with_unroll_) emit = true;
        if (with_unroll_ && do_mul) emit = true;

        auto s = _s;
        if (is_x3_v1() && emit) s = s.append(funcs::barrier_wait());
        return s;
    }

    stmt_t after_L_prepend(const stmt_t &_s, bool do_mul) const {
        return after_L(stmt_t(), do_mul).append(_s);
    }

    stmt_t before_S(const stmt_t &_s, bool do_mul, bool is_last_mul = false,
            int iter = -1) const {
        bool emit = false;
        if (!with_unroll_) emit = true;
        if (with_unroll_ && iter != -1
                && iter >= (slm_bufs_ - 1) + (gmem_bufs_ - 1) - 1)
            emit = true;

        auto s = _s;
        if (is_x3_v3() && emit) {
            s = s.append(funcs::barrier_wait());
        } else if ((is_x3_v1() || is_x3_v2()) && emit) {
            // In general we have to use SLM fence before signal to flush all
            // previous SLM stores. However any SLM load behaves as implicit
            // SLM fence for all previous SLM stores. This means we don't need
            // explicit SLM fence when we perform SLM load/multiplication
            // before signal.
            if (!do_mul) s = s.append(funcs::slm_fence());
            if (!is_last_mul) s = s.append(funcs::signal());
        }
        return s;
    }

    stmt_t after_S(
            const stmt_t &_s, bool is_last_mul = false, int iter = -1) const {
        auto s = _s;
        if (is_x2()) {
            s = s.append(funcs::barrier());
        } else if (is_x3_v3()) {
            bool emit = false;
            if (!with_unroll_) emit = true;
            if (with_unroll_ && !is_last_mul && iter != -1
                    && iter >= (slm_bufs_ - 1) + (gmem_bufs_ - 1) - 2)
                emit = true;
            if (emit) {
                s = s.append(funcs::slm_fence());
                s = s.append(funcs::signal());
            }
        }
        return s;
    }

    bool is_x2() const { return ver_ == version_t::x2; }
    bool is_x3_v1() const { return ver_ == version_t::x3_v1; }
    bool is_x3_v2() const { return ver_ == version_t::x3_v2; }
    bool is_x3_v3() const { return ver_ == version_t::x3_v3; }

private:
    enum class version_t {
        undef,
        // Double buffering scheme:
        //     L0
        //     M1
        //     S0
        //     barrier
        //     L1
        //     M0
        //     S1
        //     barrier
        x2,
        // Triple buffering scheme V1 (wait before M)
        //     L0
        //     wait R1/W0
        //     M1
        //     signal R2/W1
        //     S0
        //     L1
        //     wait R2/W1
        //     M2
        //     signal R0/W2
        //     S1
        //     L2
        //     wait R0/W2
        //     M0
        //     signal R1/W0
        //     S2
        x3_v1,
        // Triple buffering scheme V2 (wait before L)
        //     wait R1/W0
        //     L0
        //     M1
        //     signal R2/W1
        //     S0
        //     wait R2/W1
        //     L1
        //     M2
        //     signal R0/W2
        //     S1
        //     wait R0/W2
        //     L2
        //     M0
        //     signal R1/W0
        //     S2
        x3_v2,
        // Triple buffering scheme V3 (signal after store)
        // There are no SLM loads between S and signal so explicit fence is
        // required.
        //     L0
        //     M1
        //     wait R2/W0
        //     S0
        //     fence and signal R0/W1
        //     L1
        //     M2
        //     wait R0/W1
        //     S1
        //     fence and signal R1/W2
        //     L2
        //     M0
        //     wait R1/W2
        //     S2
        //     fence and signal R2/W0
        x3_v3
    };

    int slm_bufs_;
    int gmem_bufs_;
    bool with_unroll_;
    version_t ver_;
};

static bool assign_sbids(const conv_config_t &cfg) {
    return cfg.is_dpas_or_dpasw_fma();
}

class simple_slm_buffering_injector_t {
public:
    simple_slm_buffering_injector_t(const stmt_t &root, ir_context_t &ir_ctx,
            const conv_config_t &cfg, int ab_slm_size)
        : ir_ctx_(ir_ctx)
        , cfg_(cfg)
        , ab_slm_size_(ab_slm_size)
        , root_(root)
        , alloc_mgr_(root_)
        , step_(root)
        , loop_nest_(root, ir_ctx)
        , slm_sync_mgr_(cfg, /*with_unroll=*/false) {}

    stmt_t inject() {
        ir_assert(cfg_.slm().gmem_bufs() == 1)
                << "GRF buffering is not supported.";
        if (utils::one_of(cfg_.slm().bufs(), 0, 1)) return root_;

        ir_assert(cfg_.slm().a() == cfg_.slm().b())
                << "Mixed SLM/GMEM loads are not supported.";

        auto loop = step_.compute_loop();

        // SLM indices are allocated as follows:
        // slm_idx[0] -> slm_buf_store
        // slm_idx[1] -> slm_buf_compute
        // slm_idx[2] -> slm_counter
        auto slm_idx_buf
                = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "slm_idx");
        int slm_idx_size = type_t::s32().size();

        auto slm_idx_load = [&](int off, int elems) {
            return load_t::make(
                    type_t::s32(elems), slm_idx_buf, slm_idx_size * off);
        };

        // Initialize slm_idx.
        int off = 0;
        auto store0 = store_t::make(slm_idx_buf, off, 0);
        off += slm_idx_size;

        auto store1 = store_t::make(slm_idx_buf, off, 1);
        off += slm_idx_size;

        auto store2 = store_t::make(
                slm_idx_buf, off, int_imm_t::make(0, type_t::s32()));

        auto slm_idx_init = store0.append(store1).append(store2);

        auto slm_idx_load2 = slm_idx_load(0, 2);
        auto slm_idx_load4 = slm_idx_load(0, 4);
        auto slm_idx_store = store_t::make(slm_idx_buf, 0,
                slm_idx_load4 + shuffle_t::make_broadcast(1, 4));

        // Update slm_idx.
        auto mask = (slm_idx_load2
                == shuffle_t::make_broadcast(cfg_.slm().bufs(), 2));
        auto slm_idx_store_fix = store_t::make(slm_idx_buf, 0,
                shuffle_t::make_broadcast(int_imm_t::make(0, type_t::s32()), 2),
                store_t::default_stride, mask);

        auto slm_idx_update = slm_idx_store.append(slm_idx_store_fix);

        loop = slm_idx_init.append(loop);

        auto &g2s_load_orig = step_.g2s_load();
        auto &g2s_store_orig = step_.g2s_store();
        auto &s2r_load = step_.s2r_load();
        auto &mul = step_.mul();

        auto g2s_load = g2s_load_orig;
        auto g2s_store = g2s_store_orig;

        ir_assert(s2r_load.size() == mul.size());

        stmt_t s2r_mul;
        for (int i = 0; i < int(mul.size()); i++) {
            s2r_mul = s2r_mul.append(s2r_load[i]);
            loop = substitute(loop, s2r_load[i], stmt_t(), 1);
            s2r_mul = s2r_mul.append(mul[i]);
            loop = substitute(loop, mul[i], stmt_t(), 1);
        }

        loop = remove_synchronization(loop);

        s2r_mul = sub_slm_bufs(s2r_mul, slm_idx_load(1, 1));
        g2s_store = sub_slm_bufs(g2s_store, slm_idx_load(0, 1));
        g2s_store = g2s_store.append(slm_idx_update);

        auto s2r_mul_body = s2r_mul;
        auto s2r_mul_tail = s2r_mul;
        auto slm_counter = slm_idx_load(2, 1);
        auto cond = (slm_counter >= cfg_.slm().bufs() - 1);

        if (cfg_.slm().bufs() == 2) {
            s2r_mul_body = if_t::make(cond, s2r_mul_body);
        } else {
            // In general we have to use SLM fence before signal to flush all
            // previous SLM stores. However any SLM load behaves as implicit
            // SLM fence for all previous SLM stores. This means we don't need
            // explicit SLM fence when we perform SLM load/multiplication
            // before signal.
            auto with_mul = slm_sync_mgr_.before_S(s2r_mul_body, true);
            auto without_mul = slm_sync_mgr_.before_S(stmt_t(), false);
            s2r_mul_body = if_t::make(cond, with_mul, without_mul);
        }

        g2s_store = slm_sync_mgr_.after_S(g2s_store);
        g2s_load = slm_sync_mgr_.before_L_prepend(g2s_load, true);
        g2s_load = slm_sync_mgr_.after_L(g2s_load, true);

        if (!g2s_load.is_same(g2s_load_orig)) {
            loop = substitute(loop, g2s_load_orig, g2s_load, 1);
        }

        loop = substitute(
                loop, g2s_store_orig, s2r_mul_body.append(g2s_store), 1);

        loop = slm_sync_mgr_.before_loop_prepend(loop);

        // Complete the remaining iterations.
        int slm_bufs = cfg_.slm().bufs();
        int rem_iters = slm_bufs - 1;
        int mul_start = std::max(0, rem_iters - loop_nest_.size());
        multi_loop_iterator_t multi(loop_nest_.loops());
        multi.advance(loop_nest_.size() - rem_iters + mul_start);

        loop = slm_sync_mgr_.after_loop(loop);
        for (int i = 0; i < rem_iters; i++) {
            if (i >= mul_start) {
                auto tmp_mul_tail = s2r_mul_tail;
                loop_nest_.for_each_loop_var([&](const expr_t &v) {
                    expr_t iter(multi.var_value(v));
                    tmp_mul_tail = substitute(tmp_mul_tail, v, iter);
                });
                // SLM load/multiplication works as implicit SLM fence.
                loop = loop.append(tmp_mul_tail);
                multi.advance();
            }
            loop = loop.append(slm_idx_update);
        }

        if (assign_sbids(cfg_))
            loop = sbid_assigner_t(ir_ctx_.hw_cfg().hw()).assign(loop);

        const auto grf_size = ir_ctx_.hw_cfg().grf_size();
        loop = alloc_t::make(slm_idx_buf, grf_size, alloc_kind_t::grf, loop);

        alloc_updater_t alloc_updater;
        auto slm_buffers = alloc_mgr_.find_buffers(alloc_kind_t::slm);
        ir_assert(slm_buffers.size() == 1);
        auto &slm_buf = slm_buffers[0];
        int non_ab_slm_size = alloc_mgr_.alloc_size(slm_buf) - ab_slm_size_;
        alloc_updater.resize(
                slm_buf, non_ab_slm_size + ab_slm_size_ * slm_bufs);

        auto ret = substitute(root_, step_.compute_loop(), loop, 1);
        ret = alloc_updater.update(ret);
        return ret;
    }

    static stmt_t remove_synchronization(const stmt_t &s) {
        auto ret = s;
        for (auto &_c : find_objects<func_call_t>(s)) {
            auto &c = _c.as<func_call_t>();
            if (c.func.is_equal(funcs::signal_func())
                    || c.func.is_equal(funcs::slm_fence_func())
                    || c.func.is_equal(funcs::barrier_func())) {
                ret = substitute(ret, _c, stmt_t(), 1);
            }
        }
        return ret;
    }

    stmt_t sub_slm_bufs(const stmt_t &stmt, const expr_t &slm_idx) const {
        auto stmt_vec = flatten_statements(stmt);

        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            if (!is_func_call<send_t>(s)) continue;

            auto &send = s.as<func_call_t>().func.as<send_t>();

            // This is not send to SLM, skip.
            if (!send.is_slm()) continue;

            auto new_args = s.as<func_call_t>().args;
            send_t::arg_mem_off(new_args) += ab_slm_size_ * slm_idx;
            auto new_send = send.call(new_args);
            ret = substitute(ret, s, new_send, 1);
        }

        return ret;
    }

    ir_context_t &ir_ctx_;
    const conv_config_t &cfg_;
    int ab_slm_size_;

    stmt_t root_;
    alloc_manager_t alloc_mgr_;
    compute_step_t step_;
    compute_loop_nest_t loop_nest_;
    slm_sync_manager_t slm_sync_mgr_;
};

stmt_t inject_simple_slm_buffering(const stmt_t &s, ir_context_t &ir_ctx,
        const conv_config_t &cfg, int ab_slm_size) {
    trace_start();
    auto ret = simple_slm_buffering_injector_t(s, ir_ctx, cfg, ab_slm_size)
                       .inject();
    trace_pass("inject_simple_slm_buffering", ret, ir_ctx);
    return ret;
}

class unrolling_injector_t {
public:
    unrolling_injector_t(const stmt_t &root, const conv_config_t &cfg,
            ir_context_t &ir_ctx, int ab_slm_size)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , ab_slm_size_(ab_slm_size)
        , root_(root)
        , alloc_mgr_(root_)
        , step_(root)
        , loop_nest_(root, ir_ctx)
        , slm_sync_mgr_(cfg, /*with_unroll=*/true) {
        int inner_iters = loop_nest_.inner_loops_size();
        params_ = compute_params_t(cfg_.slm().bufs(), cfg_.slm().gmem_bufs(),
                ab_slm_size, cfg_.prefetch().bufs(), inner_iters);
        if (params_.use_slm) {
            for (auto &b :
                    find_send_buffers(step_.g2s_load(), /*is_mem=*/false)) {
                g2s_reg_bufs_.emplace_back(b, alloc_mgr_.alloc_size(b));
            }
        }

        // Can't fuse top-level zero-out statement unless the compute loop is
        // top-level as well.
        fuse_zero_out_with_fma_ = (loop_nest_.compute_loop_level() == 0);
    }

    stmt_t inject() {
        compute_iterator_t it(params_, loop_nest_);
        stmt_t body;

        sbid_manager_t sbid_mgr(cfg_.hw());

        auto &outer_loop_info = loop_nest_.outer_loop_info();

        auto append_outer_post_inc = [&](const stmt_t &_s) {
            auto &mul = outer_loop_info.mul_post_inc_stmt();
            auto &preload = outer_loop_info.preload_post_inc_stmt();
            auto s = _s;
            if (it.mul_loop_it.is_outer_loop_end() && it.do_mul()) {
                s = s.append(mul);
            }
            if (it.preload_loop_it.is_outer_loop_end() && it.do_preload()) {
                s = s.append(preload);
            }
            return s;
        };

        bmnk_dim_helper_t h(cfg_);
        int k_iter_blk = h.iter_dim(gemm_dims::k);
        int reduce_iter_bytes = k_iter_blk * cfg_.prb().a_data_type_size;
        // Add periodic signal-wait thread group synchronization in some cases.
        // This is to ensure threads access close reduction blocks and able to
        // reuse their common data from L1.
        bool do_sync
                = (cfg_.hw() >= ngen::HW::XeHPC) && (reduce_iter_bytes > 32);
        if (cfg_.slm()) do_sync = false;
        // Distance in iterations between signal and wait.
        int sync_dist = 3;

        // Ramp-up.
        for (int i = 0; i < it.ramp_up_iters; i++) {
            body = stmt_seq_t::make(body, create_iteration(it, sbid_mgr));
            body = append_outer_post_inc(body);
            ++it;
        }

        // Body.
        if (it.body_iters > 0) {
            int extent = it.body_iters / it.unroll();
            bool has_loop = (extent > 1);

            stmt_t loop_body;
            bool do_sync_wait = false;
            for (int i = 0; i < it.unroll(); i++) {
                if (do_sync && i % sync_dist == 0) {
                    loop_body = loop_body.append(do_sync_wait
                                    ? funcs::barrier_wait()
                                    : funcs::signal());
                    do_sync_wait = !do_sync_wait;
                }
                loop_body = loop_body.append(create_iteration(
                        it, sbid_mgr, /*in_loop_body=*/has_loop));
                ir_assert(it.do_mul());
                loop_body = append_outer_post_inc(loop_body);
                ++it;
            }
            if (do_sync && do_sync_wait)
                loop_body = loop_body.append(funcs::barrier_wait());
            if (!has_loop) {
                body = body.append(loop_body);
            } else {
                ir_assert(extent > 0);
                auto for_var = ir_ctx_.create_tmp_var(type_t::s32(), "i");
                body = body.append(for_t::make(for_var, 0, extent, loop_body));
            }
            it.advance(it.body_iters - it.unroll());
        }

        // Ramp-down.
        for (int i = 0; i < it.ramp_down_iters; i++) {
            ir_assert(it.do_mul());
            body = body.append(create_iteration(it, sbid_mgr));
            body = append_outer_post_inc(body);
            ++it;
        }

        if (outer_loop_info.has_var_refs()) {
            body = outer_loop_info.init_stmt().append(body);
            body = outer_loop_info.inject_alloc_stmts(body);
        }

        // When compute loop is part of outer loop and SLM buffering is used
        // then synchronization is required between outer iterations.
        if (loop_nest_.compute_loop_level() != 0 && params_.use_slm) {
            body = funcs::barrier().append(body);
        }

        body = stmt_group_t::make(stmt_label_t::compute_loop(), body);
        auto ret = substitute(root_, step_.compute_loop(), body, 1);

        if (params_.use_slm) {
            alloc_updater_t alloc_updater;

            // Update buffer sizes.
            for (auto &b : g2s_reg_bufs_) {
                alloc_updater.resize(b.buf,
                        alloc_mgr_.alloc_size(b.buf) * cfg_.slm().gmem_bufs());
            }

            auto slm_buffers = alloc_mgr_.find_buffers(alloc_kind_t::slm);
            if (!slm_buffers.empty()) {
                ir_assert(slm_buffers.size() == 1);

                auto &slm_buf = slm_buffers[0];
                int non_ab_slm_size
                        = alloc_mgr_.alloc_size(slm_buf) - ab_slm_size_;
                alloc_updater.resize(slm_buf,
                        non_ab_slm_size + ab_slm_size_ * cfg_.slm().bufs());
            }

            ret = alloc_updater.update(ret);
        }

        // Remove zero-out statement for C (handled by sub_fma_acc_with_zero).
        if (fuse_zero_out_with_fma_)
            ret = substitute(ret, step_.c_zero_out(), stmt_t(), 1);

        return ret;
    }

private:
    struct buffer_info_t {
        buffer_info_t(const expr_t &buf, int size) : buf(buf), size(size) {}

        expr_t buf;
        int size;
    };

    stmt_t create_iteration(const compute_iterator_t &it,
            sbid_manager_t &sbid_mgr, bool in_loop_body = false) const {
        auto g2s_load = step_.g2s_load();
        auto g2s_store = step_.g2s_store();
        auto prefetch = step_.prefetch();
        auto g2r_load = step_.g2r_load();
        auto s2r_load = step_.s2r_load();
        auto mul = step_.mul();
        auto lets = step_.inner_let_stmts();
        auto &outer_loop_info = loop_nest_.outer_loop_info();

        loop_nest_.for_each_loop_var([&](const expr_t &v) {
            expr_t mul_var_value;
            expr_t preload_var_value;
            if (v.is_same(outer_loop_info.var) && in_loop_body
                    && outer_loop_info.has_var_refs()) {
                mul_var_value = outer_loop_info.mul_var_load();
                preload_var_value = outer_loop_info.preload_var_load();
            } else {
                mul_var_value = it.mul_loop_it.var_value(v);
                preload_var_value = it.preload_loop_it.var_value(v);
            }
            g2s_load = const_fold(substitute(g2s_load, v, preload_var_value));
            g2s_store = const_fold(substitute(g2s_store, v, preload_var_value));
            prefetch = const_fold(substitute(prefetch, v, preload_var_value));
            for (auto &m : mul) {
                m = const_fold(substitute(m, v, mul_var_value));
            }
            for (auto &s : g2r_load) {
                s = const_fold(substitute(s, v, mul_var_value));
            }
            for (auto &s : s2r_load) {
                if (count_object(s, v) > 0) ir_error_not_expected();
                s = const_fold(substitute(s, v, preload_var_value));
            }
            for (int i = 0; i < int(lets.size()); i++) {
                auto &let = lets[i];
                auto &orig_let = step_.inner_let_stmts()[i];
                expr_t var_value;
                bool is_preload_let = step_.is_preload_let(orig_let);
                bool is_mul_let = step_.is_mul_let(orig_let);
                if (is_preload_let && !is_mul_let) {
                    var_value = preload_var_value;
                } else if (is_mul_let && !is_preload_let) {
                    var_value = mul_var_value;
                } else {
                    ir_assert(count_object(let.as<let_t>().value, v) == 0)
                            << "Unexpected reference to variable " << v
                            << " from " << let;
                    continue;
                }
                let = const_fold(substitute(let, v, var_value));
            }
        });

        if (params_.use_slm) {
            g2s_load = sub_gmem_bufs(g2s_load, it, /*is_read=*/false);
            g2s_store = sub_gmem_bufs(g2s_store, it, /*is_read=*/true);

            g2s_store = sub_slm_bufs(g2s_store, it, /*is_read=*/false);
            for (auto &s : s2r_load) {
                s = sub_slm_bufs(s, it, /*is_read=*/true);
            }
        }

        if (it.is_first_mul() && fuse_zero_out_with_fma_) {
            for (auto &m : mul) {
                m = sub_fma_acc_with_zero(m);
            }
        }

        if (it.is_last_g2s_store())
            g2s_store = remove_post_inc_stores(g2s_store);
        if (it.is_last_g2s_load()) g2s_load = remove_post_inc_stores(g2s_load);
        if (it.is_last_prefetch()) prefetch = remove_post_inc_stores(prefetch);
        if (it.is_last_mul()) {
            for (auto &s : s2r_load)
                s = remove_post_inc_stores(s);
            for (auto &s : g2r_load)
                s = remove_post_inc_stores(s);
        }

        stmt_t iter_stmt;

        iter_stmt = slm_sync_mgr_.before_L(iter_stmt, it.do_mul());
        if (it.do_g2s_load()) iter_stmt = iter_stmt.append(g2s_load);
        iter_stmt = slm_sync_mgr_.after_L(iter_stmt, it.do_mul());

        if (it.do_g2s_store() && it.slm_bufs() == 1) {
            iter_stmt = iter_stmt.append(funcs::barrier());
            iter_stmt = iter_stmt.append(g2s_store);
            iter_stmt = iter_stmt.append(funcs::barrier());
        }

        if (it.do_prefetch()) iter_stmt = iter_stmt.append(prefetch);

        if (it.do_mul()) {
            for (size_t i = 0; i < mul.size(); i++) {
                iter_stmt = iter_stmt.append(g2r_load[i]);
                iter_stmt = iter_stmt.append(s2r_load[i]);
                iter_stmt = iter_stmt.append(mul[i]);
            }
        }
        iter_stmt = slm_sync_mgr_.before_S(
                iter_stmt, it.do_mul(), it.is_last_mul(), it.iter);

        if (it.do_g2s_store() && it.slm_bufs() >= 2) {
            iter_stmt = iter_stmt.append(g2s_store);
        }

        iter_stmt = slm_sync_mgr_.after_S(iter_stmt, it.is_last_mul(), it.iter);

        if (assign_sbids(cfg_))
            iter_stmt = sbid_assigner_t(sbid_mgr).assign(iter_stmt);

        iter_stmt = inject_local_let(iter_stmt, lets, it.linear_id);

        return iter_stmt;
    }

    stmt_t sub_gmem_bufs(const stmt_t &stmt, const compute_iterator_t &it,
            bool is_read) const {
        if (it.slm_bufs() == 0) return stmt;
        if (is_read && !it.do_g2s_store()) return stmt;
        if (!is_read && !it.do_g2s_load()) return stmt;

        int buf_idx = (is_read ? it.gmem_read_buf_index()
                               : it.gmem_write_buf_index());
        if (buf_idx == 0) return stmt;

        auto ret = stmt;
        for (auto &b : g2s_reg_bufs_) {
            ret = substitute(ret, b.buf, b.buf[buf_idx * b.size]);
        }
        return ret;
    }

    stmt_t sub_slm_bufs(const stmt_t &stmt, const compute_iterator_t &it,
            bool is_read) const {
        if (it.slm_bufs() <= 1) return stmt;
        if (is_read && !it.do_mul()) return stmt;
        if (!is_read && !it.do_g2s_store()) return stmt;

        int upd = (is_read ? it.slm_read_offset_update()
                           : it.slm_write_offset_update());

        auto stmt_vec = flatten_statements(stmt);

        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            auto *call = s.as_ptr<func_call_t>();
            if (!call) continue;
            auto *func = call->func.as_ptr<send_t>();
            if (!func) continue;

            auto &send = call->func.as<send_t>();
            auto &args = call->args;
            auto &mem_buf = send_t::arg_mem_buf(args);
            auto &header_buf = send_t::arg_mem_off(args);

            // This is not send to SLM, skip.
            if (!send.is_slm()) continue;

            // May have signed offset.
            auto store_obj = send.create_offset_store(
                    header_buf, mem_buf, upd, /*is_signed_offset=*/true);
            auto &store = store_obj.as<store_t>();
            expr_t old_value
                    = load_t::make(send.address_type(), store.buf, store.off);
            auto post_inc_store = store_t::make(
                    store.buf, store.off, old_value + store.value);
            ret = substitute(ret, s, stmt_seq_t::make(s, post_inc_store), 1);
        }

        return ret;
    }

    static stmt_t sub_fma_acc_with_zero(const stmt_t &stmt) {
        auto stmt_vec = flatten_statements(stmt);

        object_eq_set_t<expr_t> seen_dst;
        stmt_t ret = stmt;
        for (auto &s : stmt_vec) {
            if (is_zero_points_call(s)) continue;
            if (is_func_call<dpas_t>(s) && !dpas_t::is_dp4a_call(s)) {
                auto &call = s.as<func_call_t>();

                auto &dst = dpas_t::arg_dst(s);
                auto src0 = expr_t(0); // Will be translated to null register.
                auto &src1 = dpas_t::arg_src1(s);
                auto &src2 = dpas_t::arg_src2(s);

                if (!seen_dst.insert(dst).second) continue;

                auto new_call = func_call_t::make(
                        call.func, {dst, src0, src1, src2}, call.attr);
                ret = substitute(ret, s, new_call, 1);
            } else if (is_func_call<mad_t>(s)) {
                auto &call = s.as<func_call_t>();

                auto &dst = mad_t::arg_dst(s);
                auto src0 = expr_t(0); // Will be translated to null register.
                auto &src1 = mad_t::arg_src1(s);
                auto &src2 = mad_t::arg_src2(s);

                if (!seen_dst.insert(dst).second) continue;

                auto new_call = func_call_t::make(
                        call.func, {dst, src0, src1, src2}, call.attr);
                ret = substitute(ret, s, new_call, 1);
            }
        }
        return ret;
    }

    static bool is_zero_points_call(const stmt_t &s) {
        auto is_zp_var = [&](const expr_t &e) {
            auto &base = get_base(e);
            auto &name = base.as<var_t>().name;
            return name.find("zp_") == 0;
        };
        if (is_func_call<dpas_t>(s)) {
            auto &src1 = dpas_t::arg_src1(s);
            auto &src2 = dpas_t::arg_src2(s);
            return is_zp_var(src1) || is_zp_var(src2);
        }
        if (is_func_call<mad_t>(s)) {
            auto &src1 = mad_t::arg_src1(s);
            auto &src2 = mad_t::arg_src2(s);
            return is_zp_var(src1) || is_zp_var(src2);
        }
        return false;
    }

    // Returns memory buffers if is_mem is true and register buffers otherwise.
    static object_set_t<expr_t> find_send_buffers(
            const stmt_t &s, bool is_mem) {
        object_set_t<expr_t> ret;
        auto calls = find_objects<func_call_t>(s);
        for (auto &_c : calls) {
            auto &c = _c.as<func_call_t>();
            if (!c.func.is<send_t>()) continue;
            auto &buf = (is_mem ? send_t::arg_mem_buf(_c)
                                : send_t::arg_reg_buf(_c));
            ret.insert(buf.as<ptr_t>().base);
        }
        return ret;
    }

    static stmt_t inject_local_let(const stmt_t &_s,
            const std::vector<stmt_t> &enclosed_lets, int id) {
        auto s = _s;

        // Inject let statements from the innermost loop.
        for (auto &_let : enclosed_lets) {
            auto &let = _let.as<let_t>();
            s = let_t::make(let.var, let.value, s);
        }

        // Substitute variables to avoid clashing.
        auto lets = find_objects<let_t>(s);
        for (auto &_let : lets) {
            auto &let = _let.as<let_t>();
            auto &var = let.var.as<var_t>();
            auto local_var = var_t::make(
                    var.type, var.name + "_" + std::to_string(id));
            s = substitute(s, let.var, local_var);
        }
        return s;
    }

    static stmt_t remove_post_inc_stores(const stmt_t &_s) {
        auto stores = find_objects<store_t>(_s);
        auto s = _s;
        for (auto &_store : stores) {
            auto &store = _store.as<store_t>();
            if (!contains_object(store.value, store.buf)) continue;
            s = substitute(s, store, stmt_t());
        }
        return s;
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    int ab_slm_size_;

    stmt_t root_;
    alloc_manager_t alloc_mgr_;
    compute_step_t step_;
    compute_loop_nest_t loop_nest_;
    compute_params_t params_;
    slm_sync_manager_t slm_sync_mgr_;

    std::vector<buffer_info_t> g2s_reg_bufs_; // For SLM buffering.
    bool fuse_zero_out_with_fma_ = false;
};

stmt_t inject_unrolling(const stmt_t &s, ir_context_t &ir_ctx,
        const conv_config_t &cfg, int ab_slm_size) {
    trace_start();
    auto ret = unrolling_injector_t(s, cfg, ir_ctx, ab_slm_size).inject();
    trace_pass("inject_unrolling", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

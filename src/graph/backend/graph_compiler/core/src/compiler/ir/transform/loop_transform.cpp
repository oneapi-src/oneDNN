/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#include <unordered_map>

#include <algorithm>
#include <atomic>
#include <limits>
#include <utility>
#include "../builder.hpp"
#include "../ir_comparer.hpp"
#include "../pass/ir_copy_internal.hpp"
#include "../visitor.hpp"
#include "auto_cast.hpp"
#include "constant_fold.hpp"
#include "tensor_shrink.hpp"
#include <util/utils.hpp>

SC_MODULE(ir.loop_transform)
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

int64_t get_const_as_int(const constant_c &c) {
    assert(!c.get()->is_vector());
    switch (get_type_category(c->dtype_)) {
        case CATE_INT: return c->value_[0].s64; break;
        case CATE_UINT: return c->value_[0].u64; break;
        default:
            COMPILE_ASSERT(0, "Bad type to get int from const: " << c);
            return 0;
    }
}

int64_t get_expr_as_int(const expr_c &e) {
    constant_c c = e.checked_as<constant_c>();
    return get_const_as_int(c);
}

static bool is_constant_for(for_loop_node_t *loop) {
    return loop->iter_begin_.isa<constant>() && loop->iter_end_.isa<constant>()
            && loop->step_.isa<constant>();
}

static void get_constant_from_for_loop(for_loop_node_t *_loop, int64_t &min,
        int64_t &max, int64_t &step, bool check_step = true) {
    COMPILE_ASSERT(is_constant_for(_loop),
            "Only support constant for loops for for-loop-transforms: "
                    << _loop->node_ptr_from_this());
    step = get_const_as_int(_loop->step_.as<constant>());
    if (check_step) {
        COMPILE_ASSERT(step == 1,
                "for-loop-transforms only support step=1: "
                        << _loop->node_ptr_from_this());
    }
    min = get_const_as_int(_loop->iter_begin_.as<constant>());
    max = get_const_as_int(_loop->iter_end_.as<constant>());
    COMPILE_ASSERT(max >= min,
            "for-loop-transforms: the begin should be less than or eq to end: "
                    << _loop->node_ptr_from_this());
}

////////////////////////
// New loop transform starts here
////////////////////////

class var_inplace_replacer_t : public ir_inplace_visitor_t {
public:
    var_inplace_replacer_t(std::unordered_map<var_node *, expr> *remap)
        : remap_(remap) {}
    std::unordered_map<var_node *, expr> *remap_;
    using ir_inplace_visitor_t::dispatch_impl;
    using ir_inplace_visitor_t::visit_impl;
    expr visit_impl(var v) override {
        auto itr = remap_->find(v.get());
        if (itr != remap_->end()) {
            changed_ = true;
            return itr->second;
        }
        return v;
    }

    // sync with loop transform
    void transform_shrink_info(const expr &v) {
        if (v->attr_
                && v->attr_->has_key(tensor_shrinker_attrs::should_shrink)) {
            auto &shrink_info = v->attr_->get<tensor_shrinker_t::shrink_info_t>(
                    tensor_shrinker_attrs::should_shrink);
            std::vector<expr> ret;
            ir_inplace_visitor_t::dispatch_expr_vector(shrink_info.base_, ret);
            ir_inplace_visitor_t::dispatch_expr_vector(shrink_info.shape_, ret);
        }
    }

    expr visit_impl(tensor v) override {
        ir_inplace_visitor_t::visit_impl(v);
        transform_shrink_info(v);
        return v;
    }

    expr visit_impl(tensorptr v) override {
        ir_inplace_visitor_t::visit_impl(v);
        transform_shrink_info(v);
        return v;
    }
};

bool for_loop_node_t::isvalid() const {
    return var_.defined();
}

for_loop make_const_for(var v, int64_t min, int64_t max, stmt &&body_) {
    auto ret = make_stmt<for_loop_node_t>(std::move(v),
            make_expr<constant_node>(min, v->dtype_),
            make_expr<constant_node>(max, v->dtype_),
            make_expr<constant_node>(int64_t(1), v->dtype_), std::move(body_),
            true, for_type::NORMAL);
    // set parent node
    add_parent_node(ret->body_, ret);
    return ret;
}

// copies the IR and check if there is any static var
class ir_copier_with_unroll_check_t : public ir_copier_impl_t {
    using ir_copier_impl_t::dispatch;
    using ir_copier_impl_t::ir_copier_impl_t;

    void view(define_c v) override {
        COMPILE_ASSERT(v->linkage_ == linkage::local,
                "Only allow local variables in unroll, got: " << v);
        replace_map_[v->var_] = expr();
        ir_copier_impl_t::view(std::move(v));
    }

    void view(for_loop_c v) override {
        // Replace var defined by a for_loop
        replace_map_[v->var_] = expr();
        ir_copier_impl_t::view(std::move(v));
    }
};

static int find_ths_and_then_remove(for_loop_node_t *ths, const stmts &parent) {
    for (auto v = parent->seq_.begin(); v != parent->seq_.end(); v++) {
        if ((*v).get() == ths) {
            int pos = static_cast<int>(v - parent->seq_.begin());
            parent->seq_.erase(v);
            return pos;
        }
    }
    COMPILE_ASSERT(false, "Cannot find the axis in the parent");
    return -1;
}

void for_loop_node_t::unroll(uint64_t factor, const stmt &parent) {
    COMPILE_ASSERT(isvalid(), "Transforming an invalid for-loop");
    if (factor == 1) { return; }
    auto this_reserve = node_ptr_from_this();
    bool remove = false;
    bool has_remainder = true;
    int loop_idx = -1;
    int64_t min, max, step;
    bool is_const = is_constant_for(this);
    if (factor == 0) {
        COMPILE_ASSERT(parent.defined() && parent.isa<stmts>(),
                "parent is not defined or is not stmts.");
        COMPILE_ASSERT(is_const, "Need const for-loop to fully unroll");
        loop_idx = find_ths_and_then_remove(this, parent.static_as<stmts>());
        remove = true;
    }
    if (is_const) {
        get_constant_from_for_loop(this, min, max, step, false);
        int64_t loop_len = max - min;
        if (factor == 0) { factor = utils::divide_and_ceil(loop_len, step); }
        has_remainder = loop_len % (factor * step) != 0
                || loop_len < ((int64_t)factor * step);
    }
    // make new variables
    var oldvar = var_.checked_as<var>();
    var newvar = var_->remake().static_as<var>();
    expr step_x_factor = is_const ? (step * factor) : step_ * factor;
    newvar->name_ += "_u";

    // make the new unrolled body
    stmts seq = make_stmt<stmts_node_t>(std::vector<stmt>());
    // make the indexing variable that can be shared by all unrolled stmts
    expr newidx = oldvar->remake();
    // a loop var when reserves loop, a constant node when unrolls all.
    expr replace_value;
    if (!remove) {
        seq->seq_.emplace_back(builder::make_var_tensor_def_unattached(
                newidx, linkage::local, newvar * step_x_factor + iter_begin_));
        replace_value = newidx;
    } else {
        replace_value = make_expr<constant_node>(
                get_const_as_int(iter_begin_.static_as<constant>()),
                oldvar->dtype_);
    }
    // unroll by the factor
    for (uint64_t i = 0; i < factor; i++) {
        std::unordered_map<expr_c, expr> replace_map;
        ir_copier_with_unroll_check_t copier(replace_map, false);
        replace_map[oldvar] = replace_value + (is_const ? i * step : i * step_);
        seq->seq_.emplace_back(copier.copy(body_));
    }

    // the new loop length, constant folded and pure expr versions
    if (!remove) {
        uint64_t c_newloop_len = is_const ? (max - min) / (step * factor) : 0;
        expr newloop_len = is_const
                ? c_newloop_len
                : ((iter_end_ - iter_begin_) / (step_ * factor));
        // if has remainder, should put the seq into an if-else
        if (has_remainder) {
            // make the remainder loop, first make the lower bound of the loop
            // var
            expr begins = is_const
                    ? (c_newloop_len * (step * factor) + min)
                    : (newloop_len * (step_ * factor) + iter_begin_);
            auto remainder = builder::make_for_loop_unattached(oldvar, begins,
                    iter_end_, step_, body_, incremental_,
                    kind_ != for_type::PARALLEL ? kind_ : for_type::NORMAL);
            // put the remainder loop and the unrolled body into the if-else
            stmts seq_remainder = make_stmt<stmts_node_t>(std::vector<stmt>());
            seq_remainder->seq_.emplace_back(std::move(remainder));
            auto if_check
                    = builder::make_if_else_unattached(newvar < newloop_len,
                            std::move(seq), std::move(seq_remainder));
            seq = make_stmt<stmts_node_t>(
                    std::vector<stmt> {std::move(if_check)});
        }
        // update the loop vars
        body_ = std::move(seq);

        iter_begin_ = 0;
        if (has_remainder) {
            iter_end_ = is_const ? (c_newloop_len + 1) : (newloop_len + 1);
        } else {
            iter_end_ = is_const ? (c_newloop_len) : (newloop_len);
        }
        step_ = 1;
        var_ = newvar;
    } else {
        assert(loop_idx != -1);
        auto &pseq = parent.static_as<stmts>()->seq_;
        pseq.insert(
                pseq.begin() + loop_idx, seq->seq_.begin(), seq->seq_.end());
    }
}

static for_loop build_inner_for(for_loop_node_t *ths, int64_t min,
        int64_t block, node_ptr_map *node_remap) {
    // make new variables
    var varptr = ths->var_.checked_as<var>();
    // remake a new var to replace old one. Call "remake" to avoid infinite
    // recursion in var_inplace_replacer after fuse():
    // cause: in fuse(), v => v / 123, we use v / 123 multiple times
    // in split, we inplace change (v * 123) => (v * 321 + 1) / 123, but (v *
    // 123) is used many times, we will recursively replace (v * 321 + 1) / 123
    // => ((v * 321 + 1) * 321 + 1) / 123
    // thus we need to remake v
    ths->var_ = ths->var_->remake();
    std::string &varname = ths->var_.static_as<var>()->name_;
    std::string oldname = std::move(varname);
    varname = oldname + "_0outer";

    var vin = make_expr<var_node>(ths->var_->dtype_, oldname + "_0inner");

    // old iter variable is mapped to outer * block + inner
    expr remapped
            = ths->var_ * make_expr<constant_node>(block, ths->var_->dtype_)
            + vin;
    if (min != 0) {
        remapped = remapped + make_expr<constant_node>(min, ths->var_->dtype_);
    }
    std::unordered_map<var_node *, expr> remap = {{varptr.get(), remapped}};
    if (node_remap) {
        node_remap->insert(std::make_pair(varptr.impl, remapped.impl));
    }
    var_inplace_replacer_t pass(&remap);
    pass.dispatch_impl(ths->body_);

    // make inner loop
    auto inner_for = make_const_for(vin, 0, block, std::move(ths->body_));
    return inner_for;
}

for_loop for_loop_node_t::split(int64_t block, node_ptr_map *node_remap) {
    COMPILE_ASSERT(isvalid(), "Transforming an invalid for-loop");
    int64_t min, max, step;
    get_constant_from_for_loop(this, min, max, step);
    int64_t loop_len = max - min;
    COMPILE_ASSERT(loop_len % block == 0 && loop_len >= block,
            "The loop length "
                    << loop_len
                    << " should be divisible of and larger than the block size "
                    << block);
    int64_t outer_len = loop_len / block;

    // set outer loop
    iter_begin_ = make_expr<constant_node>(int64_t(0), var_->dtype_);
    iter_end_ = make_expr<constant_node>(outer_len, var_->dtype_);

    auto inner_for = build_inner_for(this, min, block, node_remap);

    body_ = make_stmt<stmts_node_t>(std::vector<stmt>({inner_for}));
    return inner_for;
}

for_loop for_loop_node_t::split_on_num_threads(
        int64_t num_groups, node_ptr_map *node_remap) {
    COMPILE_ASSERT(isvalid(), "Transforming an invalid for-loop");
    int64_t min, max, step;
    get_constant_from_for_loop(this, min, max, step);
    COMPILE_ASSERT(
            min == 0 && step == 1, "Only support begin is 0 and step is 1")
    COMPILE_ASSERT(max == num_threads_, "Only support num_iters == num_threads")
    int64_t ori_num_threads = num_threads_;
    COMPILE_ASSERT(ori_num_threads % num_groups == 0,
            "The num_threads " << ori_num_threads
                               << " should be divisible by num_groups "
                               << num_groups);
    num_threads_ = ori_num_threads / num_groups;
    iter_end_ = make_expr<constant_node>(uint64_t(num_threads_), var_->dtype_);

    auto inner_for = build_inner_for(this, min, num_groups, node_remap);
    inner_for->num_threads_ = num_groups;
    inner_for->kind_ = for_type::PARALLEL;
    if (attr_ && attr_->has_key(stmt_attr_key::parallel_loop_balanced)) {
        inner_for->attr()[stmt_attr_key::parallel_loop_balanced]
                = attr()[stmt_attr_key::parallel_loop_balanced];
    }
    body_ = make_stmt<stmts_node_t>(std::vector<stmt>({inner_for}));
    return inner_for;
}

for_loop get_inner_for_loop(const for_loop_node_t *f) {
    const for_loop_node_t *cur = f;
    if (cur->body_.isa<stmts>()) {
        auto stmtlist = cur->body_.static_as<stmts>();
        // if it is a basic block with one statement and it is a for-loop
        if (stmtlist->seq_.size() == 1
                && stmtlist->seq_.at(0).isa<for_loop>()) {
            return stmtlist->seq_.at(0).static_as<for_loop>();
        }
    } else if (cur->body_.isa<for_loop>()) {
        return cur->body_.static_as<for_loop>();
    }
    return for_loop();
}

for_loop get_last_loop_in_body(const stmt &body) {
    if (body.isa<stmts>()) {
        auto stmtlist = body.static_as<stmts>();
        if (!stmtlist->seq_.empty()) {
            stmt last_stmt;
            for (int64_t i = stmtlist->seq_.size() - 1; i >= 0; --i) {
                last_stmt = stmtlist->seq_[i];
                if (!stmtlist->seq_[i].isa<stmts>()
                        || !stmtlist->seq_[i].static_as<stmts>()->seq_.empty())
                    break;
            }
            if (last_stmt.isa<for_loop>()) {
                return last_stmt.static_as<for_loop>();
            } else {
                return get_last_loop_in_body(last_stmt);
            }
        }
    } else if (body.isa<for_loop>()) {
        return body.static_as<for_loop>();
    }
    return for_loop();
}

for_loop for_loop_node_t::fuse(const for_loop &ax, node_ptr_map *node_remap) {
    COMPILE_ASSERT(ax->isvalid(), "Transforming an invalid for-loop: ax");
    COMPILE_ASSERT(isvalid(), "Transforming an invalid for-loop: this");
    if (!get_inner_for_loop(this).ptr_same(ax)) {
        SC_MODULE_INFO << "We can only fuse the next inner loop";
        return this->node_ptr_from_this().static_as<for_loop>();
    }
    static std::atomic<int> fuse_count(0);
    expr min1, max1, step1;
    expr min2, max2, step2;
    auto get_expr_from_for_loop
            = [](for_loop_node_t *_loop, expr &min, expr &max, expr &step) {
                  min = _loop->iter_begin_;
                  max = _loop->iter_end_;
                  step = _loop->step_;
              };
    get_expr_from_for_loop(this, min1, max1, step1);
    get_expr_from_for_loop(ax.get(), min2, max2, step2);
    expr loop_len1 = max1 - min1;
    expr loop_len2 = max2 - min2;
    expr outer_len = loop_len1 * loop_len2;
    var var1 = var_.checked_as<var>(), var2 = ax->var_.checked_as<var>();
    COMPILE_ASSERT(var_->dtype_ == ax->var_->dtype_,
            "The fused for loop variables should have the same types, got "
                    << var_->dtype_ << " and " << var_->dtype_);
    // make new variables
    var vout = make_expr<var_node>(var_->dtype_,
            std::string("fused_0") + var1->name_ + "__" + var2->name_ + "_"
                    + std::to_string(fuse_count++));

    std::unordered_map<var_node *, expr> var_remap;
    // old iter variable is mapped to vout / loop_len2 and vout % loop_len2
    expr outer = vout / loop_len2;
    outer = outer + min1;
    outer = do_cast_and_fold(outer);
    var_remap.insert(std::make_pair(var1.get(), outer));
    if (node_remap) {
        node_remap->insert(std::make_pair(var1.impl, outer.impl));
    }
    expr inner = vout % loop_len2;
    inner = inner + min2;
    inner = do_cast_and_fold(inner).remove_const();

    var_remap.insert(std::make_pair(var2.get(), inner));
    if (node_remap) {
        node_remap->insert(std::make_pair(var2.impl, inner.impl));
    }
    var_inplace_replacer_t pass(&var_remap);
    auto newbody = pass.dispatch_impl(ax->body_);

    var_ = vout;
    iter_begin_ = make_expr<constant_node>(int64_t(0), var1->dtype_);
    iter_end_ = do_cast_and_fold(outer_len).remove_const();
    step_ = make_expr<constant_node>(int64_t(1), var1->dtype_);

    // redirect parent node
    add_parent_node(newbody, node_ptr_from_this());

    body_ = std::move(newbody);

    ax->var_ = expr(); // invalidate ax
    return for_loop(shared_from_this());
}

class loop_replacer_t : public ir_inplace_visitor_t {
public:
    using ir_inplace_visitor_t::dispatch_impl;
    using ir_inplace_visitor_t::visit_impl;
    for_loop_node_t *old_;
    for_loop new_;
    bool replaced = false;
    loop_replacer_t(for_loop_node_t *old, for_loop new_)
        : old_(old), new_(std::move(new_)) {}
    stmt visit_impl(for_loop v) override {
        if (v.get() == old_) {
            replaced = true;
            return new_;
        }
        return v;
    }

    expr dispatch_impl(expr e) override {
        // no need to look into expr
        return e;
    }
};

class loop_parallel_replacer_t : public ir_inplace_visitor_t {
private:
    bool ignore_nested_parallel_;

public:
    loop_parallel_replacer_t(bool ignore_nested_parallel = false)
        : ignore_nested_parallel_(ignore_nested_parallel) {}
    using ir_inplace_visitor_t::dispatch_impl;
    using ir_inplace_visitor_t::visit_impl;
    stmt visit_impl(for_loop v) override {
        dispatch_impl(v->body_);
        if (v->kind_ == for_type::PARALLEL) {
            if (v->num_threads_ > 0) {
                if (!ignore_nested_parallel_) {
                    v->kind_ = for_type::NORMAL;
                    v->num_threads_ = 0;
                }
            } else {
                v->kind_ = for_type::NORMAL;
            }
        }
        return v;
    }
    expr dispatch_impl(expr e) override { return e; }
};

void for_loop_node_t::reorder(stmt parent, std::vector<for_loop> &&ax) {
    COMPILE_ASSERT(!ax.empty(), "The number of axises to reorder should > 0");
    for_loop cur = for_loop(shared_from_this());
    stmt inner_body;
    for (unsigned i = 0; i < ax.size(); i++) {
        COMPILE_ASSERT(cur.defined(),
                "Bad number of axises to reorder. Got "
                        << ax.size() << " to reorder, but only have " << i
                        << " nested for-loops");
        COMPILE_ASSERT(cur->isvalid(), "Transforming an invalid for-loop");
        inner_body = cur->body_;
        COMPILE_ASSERT(std::find_if(ax.begin(), ax.end(),
                               [cur](for_loop &v) { return v.ptr_same(cur); })
                        != ax.end(),
                "Cannot find axis " << cur->var_
                                    << " in the given axises to reorder");
        cur = get_inner_for_loop(cur.get());
    }

    // redirect parent node
    add_parent_node(inner_body, ax.back());

    ax.back()->body_ = std::move(inner_body);
    cur = ax.back();
    if (ax.size() > 1) {
        for (int64_t i = ax.size() - 2; i >= 0; i--) {
            ax.at(i)->body_ = make_stmt<stmts_node_t>(
                    std::vector<stmt>({std::move(cur)}));
            cur = ax.at(i);
        }
    }
    loop_replacer_t replacer(this, ax.front());
    replacer.dispatch_impl(std::move(parent));
    COMPILE_ASSERT(replacer.replaced,
            "Cannot find the for-loop to replace in the parent stmt");
}

void flatten_stmt_and_append(const stmt &s, std::vector<stmt> &out) {
    if (s.isa<stmts>()) {
        for (auto &v : s.static_as<stmts>()->seq_) {
            out.push_back(v);
        }
    } else {
        out.push_back(s);
    }
}

static void check_loop_for_merge(for_loop_node_t *ths, for_loop_node_t *ax) {
    COMPILE_ASSERT(ths->isvalid() && ax->isvalid(),
            "Invalid for-loop. It has been fused or merged");
    COMPILE_ASSERT(ax != ths, "The axis to merge should not be \'this\'");
}

static bool is_loop_range_same(for_loop_node_t *ths, for_loop_node_t *ax) {
    ir_comparer ircmp(false, true, true);
    ircmp.set_expr_mapping(ths->var_, ax->var_);
    return ths->incremental_ == ax->incremental_
            && ths->iter_begin_->equals(ax->iter_begin_, ircmp)
            && ths->iter_end_->equals(ax->iter_end_, ircmp)
            && ths->step_->equals(ax->step_, ircmp);
}

static void find_ths_and_ax_then_remove(
        for_loop_node_t *ths, const stmt &parent, for_loop_node_t *ax) {
    COMPILE_ASSERT(parent.isa<stmts>(), "The parent should be an stmts_node_t");
    auto s = parent.static_as<stmts>();
    constexpr size_t invalid = std::numeric_limits<size_t>::max();
    size_t this_in_parent = invalid, ax_in_parent = invalid;
    for (size_t i = 0; i < s->seq_.size(); i++) {
        auto &v = s->seq_.at(i);
        if (v.get() == ths) {
            this_in_parent = i;
        } else if (v.get() == ax) {
            ax_in_parent = i;
        }
    }
    COMPILE_ASSERT(this_in_parent != invalid && ax_in_parent != invalid,
            "Cannot find the axises in the parent");
    s->seq_.erase(s->seq_.begin() + ax_in_parent); // remove ax from parent
}

static int *get_unroll_factor_attr(const for_loop_node_t *ths) {
    if (ths->attr_) {
        return ths->attr_->get_or_null<int>(stmt_attr_key::unroll_loop);
    }
    return nullptr;
}

static void do_merge(
        for_loop_node_t *ths, const stmt &parent, const for_loop &ax) {
    // now replace ax's variable with this->var_
    std::unordered_map<var_node *, expr> remap;
    remap.insert(std::make_pair(ax->var_.checked_as<var>().get(), ths->var_));
    var_inplace_replacer_t pass(&remap);
    auto axbody = pass.dispatch_impl(ax->body_);
    ax->var_ = expr(); // invalidate ax
    if (parent.defined()) {
        find_ths_and_ax_then_remove(ths, parent, ax.get());
    }
    // merge the bodies
    std::vector<stmt> newbody;
    flatten_stmt_and_append(ths->body_, newbody);
    flatten_stmt_and_append(axbody, newbody);
    ths->body_ = make_stmt<stmts_node_t>(std::move(newbody));
    if (auto unroll_ax = get_unroll_factor_attr(ax.get())) {
        auto unroll_ths = get_unroll_factor_attr(ths);
        if (!unroll_ths) {
            ths->attr()[stmt_attr_key::unroll_loop] = *unroll_ax;
        } else {
            COMPILE_ASSERT(*unroll_ax == *unroll_ths,
                    "Different unroll factors when merging the loops: "
                            << *unroll_ths << "v.s." << *unroll_ax);
        }
    }
}

for_loop for_loop_node_t::merge(const stmt &parent, const for_loop &ax) {
    check_loop_for_merge(this, ax.get());
    COMPILE_ASSERT(is_loop_range_same(this, ax.get()),
            "The ranges of the merged for-loops should be the same");
    do_merge(this, parent, ax);
    return node_ptr_from_this().as<for_loop>();
}

for_loop for_loop_node_t::merge(stmt parent, for_loop ax, unsigned num_inner) {
    for_loop_node_t *ths = this;
    for_loop axis = std::move(ax);
    for (unsigned i = 0; i < num_inner; i++) {
        COMPILE_ASSERT(ths && axis.defined(),
                "Merging " << num_inner << " inner loops, but have only " << i
                           << " loops in the IR");
        for_loop_node_t *next_ths = get_inner_for_loop(ths).get();
        for_loop next_ax = get_inner_for_loop(axis.get());
        ths->merge(parent, axis);
        parent = ths->body_;
        ths = next_ths;
        axis = next_ax;
    }
    return node_ptr_from_this().static_as<for_loop>();
}

int for_loop_node_t::merge_all(stmt parent, for_loop ax) {
    for_loop_node_t *ths = this;
    for_loop axis = std::move(ax);
    int num_loops = 0;
    for (;;) {
        // if there is no inner loops, break
        if (!ths || !axis.defined()) { break; }
        for_loop_node_t *next_ths = get_inner_for_loop(ths).get();
        for_loop next_ax = get_inner_for_loop(axis.get());

        // check if the loops are valid
        check_loop_for_merge(ths, axis.get());
        // if the loops are not mergable, break
        if (!is_loop_range_same(ths, axis.get())) { break; }
        do_merge(ths, parent, axis);
        num_loops++;
        parent = ths->body_;
        ths = next_ths;
        axis = next_ax;
    }
    return num_loops;
}

void for_loop_node_t::parallel_merge(const stmt &parent, const for_loop &ax) {
    COMPILE_ASSERT(isvalid(), "Invalid loop");
    COMPILE_ASSERT(ax->isvalid(), "Invalid loop");
    COMPILE_ASSERT(step_.isa<constant>()
                    && get_const_as_int(step_.static_as<constant>()) == 1,
            "the step of this should be 1");
    COMPILE_ASSERT(ax->step_.isa<constant>()
                    && get_const_as_int(ax->step_.static_as<constant>()) == 1,
            "the step of ax should be 1");
    find_ths_and_ax_then_remove(this, parent, ax.get());
    auto body1 = std::move(body_);
    stmts body2 = ax->body_.isa<stmts>()
            ? ax->body_.static_as<stmts>()
            : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(ax->body_)});
    body2->seq_.insert(body2->seq_.begin(),
            builder::make_var_tensor_def_unattached(ax->var_, linkage::local,
                    var_ - iter_end_ + ax->iter_begin_));
    auto if_else
            = builder::make_if_else_unattached(var_ < iter_end_, body1, body2);
    body_ = builder::make_stmts_unattached({if_else});
    // set parent node
    add_parent_node(body1, if_else);
    add_parent_node(body2, if_else);
    add_parent_node(if_else, body_);
    add_parent_node(body_, node_ptr_from_this());
    iter_end_ = iter_end_ + ax->iter_end_ - ax->iter_begin_;
    ax->var_ = expr();
}

void remove_parallel(stmt body, bool ignore_nested_parallel) {
    loop_parallel_replacer_t replacer(ignore_nested_parallel);
    replacer.dispatch_impl(std::move(body));
}

void remove_parallel(func_t body, bool ignore_nested_parallel) {
    loop_parallel_replacer_t replacer(ignore_nested_parallel);
    replacer.dispatch_impl(std::move(body));
}

std::vector<for_loop> collect_loops(const stmt &body) {
    std::vector<for_loop> ret;
    if (body.isa<stmts>()) {
        for (auto &smt : body.static_as<stmts>()->seq_) {
            if (smt.isa<for_loop>()) {
                ret.push_back(smt.static_as<for_loop>());
            }
        }
    } else {
        if (body.isa<for_loop>()) { ret.push_back(body.static_as<for_loop>()); }
    }
    return ret;
}

std::vector<for_loop> collect_nested_loops(stmt body) {
    std::vector<for_loop> ret;
    auto cur = std::move(body);
    bool outer_loop = true;
    while (true) {
        if (cur.isa<stmts>()) {
            auto stmts_cur = cur.static_as<stmts>();
            bool continue_flag = false;
            for (unsigned i = 0; i < stmts_cur->seq_.size(); i++) {
                auto smt = stmts_cur->seq_[i];
                if (smt.isa<for_loop>()) {
                    ret.push_back(smt.static_as<for_loop>());
                    cur = ret.back()->body_;
                    continue_flag = true;
                    break;
                } else if (!ret.empty()) {
                    outer_loop = false;
                    break;
                }
            }
            if (continue_flag && outer_loop) { continue; }
        } else {
            if (cur.isa<for_loop>()) {
                ret.push_back(cur.static_as<for_loop>());
                cur = ret.back()->body_;
                continue;
            }
        }
        break;
    }
    return ret;
}

static size_t collect_all_loops_helper(
        std::vector<for_loop> &ret, const stmt &body) {
    size_t collected = 0;
    if (body.isa<stmts>()) {
        for (auto &smt : body.static_as<stmts>()->seq_) {
            collected += collect_all_loops_helper(ret, smt);
        }
    } else if (body.isa<for_loop>()) {
        auto loop = body.static_as<for_loop>();
        ret.push_back(loop);
        collected++;
        collected += collect_all_loops_helper(ret, loop->body_);
    } else if (body.isa<if_else>()) {
        auto cond = body.static_as<if_else>();
        if (cond->then_case_.defined()) {
            collected += collect_all_loops_helper(ret, cond->then_case_);
        }
        if (cond->else_case_.defined()) {
            collected += collect_all_loops_helper(ret, cond->else_case_);
        }
    }
    return collected;
}

std::vector<for_loop> collect_all_loops(const stmt &body) {
    std::vector<for_loop> ret;
    collect_all_loops_helper(ret, body);
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

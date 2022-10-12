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

#include "gpu/jit/pass/shuffle_splitter.hpp"

#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class shuffle_splitter_t : public ir_mutator_t {
public:
    static expr_t add(const expr_t &e, const expr_t &ee) {
        if (e.is_empty()) {
            return ee;
        } else if (ee.is_empty()) {
            return e;
        } else {
            return e + ee;
        }
    };

    object_t _mutate(const binary_op_t &obj) override {
        if (obj.op_kind != op_kind_t::_add) return ir_mutator_t::_mutate(obj);

        // Aggregate bcast expressions together
        auto new_obj = ir_mutator_t::_mutate(obj);
        auto args = split_by_add(new_obj, obj.type.elems());
        if (args.size() <= 1) return new_obj;

        std::vector<expr_t> bcasts;
        std::vector<expr_t> non_bcasts;
        for (auto &a : args) {
            if (a.type().elems() != obj.type.elems()) {
                bcasts.push_back(a);
            } else {
                non_bcasts.push_back(a);
            }
        }

        if (bcasts.size() <= 1) return new_obj;

        int elems = obj.type.elems();
        expr_t e = shuffle_t::make_broadcast(make_add(bcasts), elems);
        if (!non_bcasts.empty()) e = add(e, make_add(non_bcasts));

        ir_assert(!e.is_empty());
        return std::move(e);
    }

    object_t _mutate(const shuffle_t &obj) override {
        object_t new_obj = ir_mutator_t::_mutate(obj);
        if (obj.is_broadcast() || !new_obj.is<shuffle_t>()) return new_obj;

        auto &o = new_obj.as<shuffle_t>();

        // Split shuffle to bcast(expr) + vector(exprs) + vector(constants). Use
        // existing vector(constants) to improve the effect of common
        // subexpression elimnation.

        expr_t vec_bcast;
        std::vector<expr_t> vec_const;
        std::vector<expr_t> vec_off;

        std::vector<object_eq_map_t<expr_t, int>> args;
        bool can_split = false;
        const expr_t zero = cast(0, o.type.scalar());

        for (auto &v : o.vec) {
            // Only supports integer arithmetic
            if (!v.type().is_int()) return new_obj;
            auto v_args = split_by_add(v, v.type().elems());
            if (v_args.size() > 1) can_split = true;
            expr_t e_const = zero;
            args.emplace_back();
            for (auto &a : v_args) {
                if (is_const(a)) {
                    e_const += a;
                } else {
                    args.back()[a] += 1;
                }
            }
            vec_const.push_back(const_fold(e_const));
        }

        if (!can_split) return new_obj;

        // Multiset Intersection
        auto intersect = [](object_eq_map_t<expr_t, int> &a,
                                 object_eq_map_t<expr_t, int> &b) {
            object_eq_map_t<expr_t, int> c;
            for (auto &kv : a) {
                auto &key = kv.first;
                int rep_a = kv.second;
                int rep_b = b[key];
                int rep_c = std::min(rep_a, rep_b);
                if (rep_c > 0) c[key] = rep_c;
            }
            return c;
        };
        // Multiset Difference
        auto difference = [](object_eq_map_t<expr_t, int> &a,
                                  object_eq_map_t<expr_t, int> &b) {
            object_eq_map_t<expr_t, int> c;
            for (auto &kv : a) {
                auto key = kv.first;
                int rep_a = kv.second;
                int rep_b = b[key];
                int rep_c = rep_a - rep_b;
                if (rep_c > 0) c[key] = rep_c;
            }
            return c;
        };

        auto is_empty_or_fill = [&](std::vector<expr_t> &vec) {
            for (auto &c : vec) {
                if (!c.is_empty() && !c.is_equal(zero)) { return false; }
                if (c.is_empty()) c = zero;
            }
            return true;
        };

        auto is_bcast = [](const std::vector<expr_t> &vec) {
            for (auto &c : vec) {
                if (!c.is_equal(vec[0])) { return false; }
            }
            return true;
        };

        auto get_bcast_difference = [](expr_t expr_a, expr_t expr_b) {
            if (!expr_a.is<shuffle_t>() || !expr_b.is<shuffle_t>())
                return expr_t();

            auto &a = expr_a.as<shuffle_t>();
            auto &b = expr_b.as<shuffle_t>();
            if (a.idx.size() != b.idx.size()) return expr_t();
            if (a.vec.size() != b.vec.size()) return expr_t();

            for (size_t i = 0; i < a.idx.size(); i++) {
                if (a.idx[i] != b.idx[i]) return expr_t();
            }

            if (a.vec.size() <= 0) return expr_t();
            expr_t offset = const_fold(a.vec[0] - b.vec[0]);
            for (size_t i = 0; i < a.vec.size(); i++) {
                expr_t new_offset = const_fold(a.vec[i] - b.vec[i]);
                if (!offset.is_equal(new_offset)) return expr_t();
            }
            return offset;
        };

        auto base_args = args[0];
        for (int i = 1; i < (int)args.size(); i++) {
            base_args = intersect(base_args, args[i]);
        }

        vec_bcast = make_add(base_args);
        for (auto &a : args)
            vec_off.push_back(make_add(difference(a, base_args)));

        bool is_bcast_empty = base_args.size() == 0;
        bool is_consts_empty = is_empty_or_fill(vec_const);
        bool is_consts_bcast = is_bcast(vec_const);
        bool is_off_empty = is_empty_or_fill(vec_off);

        expr_t const_shuffle;
        if (!is_consts_empty) {
            const_shuffle = shuffle_t::make(vec_const, o.idx);
            if (!is_consts_bcast) {
                expr_t offset;
                for (auto &k : const_shuffles_) {
                    offset = get_bcast_difference(const_shuffle, k);
                    if (!offset.is_empty()) {
                        vec_bcast = add(vec_bcast, offset);
                        const_shuffle = k;
                        is_consts_bcast
                                = is_bcast(const_shuffle.as<shuffle_t>().vec);
                        break;
                    }
                }

                if (offset.is_empty()) {
                    const_shuffles_.emplace(const_shuffle);
                }
            }

            if (is_consts_bcast) {
                const_shuffle = shuffle_t::make_broadcast(
                        const_shuffle.as<shuffle_t>().vec[0], o.type.elems());
            }
        }

        expr_t e;
        if (!is_bcast_empty)
            e = add(e, shuffle_t::make_broadcast(vec_bcast, o.type.elems()));
        if (!is_off_empty) e = add(e, shuffle_t::make(vec_off, o.idx));
        e = add(e, const_shuffle);

        return std::move(e);
    }

private:
    object_eq_set_t<expr_t> const_shuffles_;
    static std::vector<expr_t> split_by_add(const expr_t &e, int elems) {
        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle && shuffle->is_broadcast() && shuffle->elems() == elems) {
            return split_by_add(shuffle->vec[0], elems);
        }
        auto *op = e.as_ptr<binary_op_t>();
        if (!op || op->op_kind != op_kind_t::_add) return {e};
        auto a_args = split_by_add(op->a, elems);
        auto b_args = split_by_add(op->b, elems);
        std::vector<expr_t> args;
        args.insert(args.end(), a_args.begin(), a_args.end());
        args.insert(args.end(), b_args.begin(), b_args.end());
        return args;
    }

    static expr_t make_add(const std::vector<expr_t> &args) {
        if (args.empty()) return 0;
        expr_t e = args[0];
        for (int i = 1; i < (int)args.size(); i++)
            e = e + args[i];
        return e;
    }
    static expr_t make_add(const object_eq_map_t<expr_t, int> &args) {
        if (args.empty()) return 0;
        expr_t e;
        for (auto &kv : args)
            if (kv.second == 0)
                continue;
            else if (kv.second == 1)
                e = add(e, kv.first);
            else
                e = add(e, kv.second * kv.first);
        return e;
    }
};

stmt_t split_shuffle(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = shuffle_splitter_t().mutate(s);
    trace_pass("split_shuffle", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

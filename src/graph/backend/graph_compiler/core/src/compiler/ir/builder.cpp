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

#include "builder.hpp"

#include <util/utils.hpp>

#include <algorithm>
#include <atomic>
#include <utility>
#include "passlet/passlet.hpp"
#include "passlet/structural_analysis.hpp"
#include "sc_expr.hpp"
#include "sc_function.hpp"
#include "sc_stmt.hpp"
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
static void merge_attrs(std::unique_ptr<any_map_t> &mergeto,
        const std::unique_ptr<any_map_t> &mergefrom) {
    if (mergeto) {
        auto &ths_attr_map = mergefrom->as_map();
        mergeto->as_map().insert(ths_attr_map.begin(), ths_attr_map.end());
    } else {
        mergeto = utils::make_unique<any_map_t>(*mergefrom);
    }
}

static void copy_temp_data(const node_base *src, node_base *dst) {
    if (src->temp_data_) {
        auto &data = *src->temp_data_;
        if (!data.vtable()) { return; }
        if (!data.vtable()->copy_assigner_ && !data.vtable()->copy_ctor_) {
            return;
        }
        dst->temp_data() = data;
    }
}

expr copy_attr(const expr_base &ths, expr &&newexpr) {
    if (ths.attr_) { merge_attrs(newexpr->attr_, ths.attr_); }
    copy_temp_data(&ths, newexpr.get());
    return std::move(newexpr);
}

stmt copy_attr(const stmt_base_t &ths, stmt &&newstmt) {
    if (ths.attr_) { merge_attrs(newstmt->attr_, ths.attr_); }
    copy_temp_data(&ths, newstmt.get());
    return std::move(newstmt);
}

func_t copy_attr(const func_base &ths, func_t &&newfunc) {
    if (ths.attr_) { merge_attrs(newfunc->attr_, ths.attr_); }
    copy_temp_data(&ths, newfunc.get());
    return std::move(newfunc);
}

static std::vector<expr> vector_remove_const(const std::vector<expr_c> &v) {
    std::vector<expr> ret;
    ret.reserve(v.size());
    for (auto &i : v) {
        ret.emplace_back(i.remove_const());
    }
    return ret;
}

// add ret to parent node of s.
void add_parent_node(const stmt &s, const stmt &ret) {
    s->attr()["builder.parent_node"] = passlet::structural_result_t {ret, s};
}

passlet::structural_result_t *get_parent_struct(const stmt &v) {
    if (!v->attr_) return nullptr;
    return v->attr_->get_or_null<passlet::structural_result_t>(
            "builder.parent_node");
}

// This function can find the parent node in IR, if the node has no parent
// node, return stmt().
stmt get_parent_node(const stmt &node) {
    auto stru_res = get_parent_struct(node);
    if (!stru_res) return stmt();
    return stru_res->get_parent_node();
}

stmt get_common_parent_node(const stmt &node1, const stmt &node2) {
    auto stru_res1 = get_parent_struct(node1),
         stru_res2 = get_parent_struct(node2);
    if (!stru_res1 || !stru_res2) return stmt();
    return stmt {const_cast<stmt_base_t *>(
            stru_res1->find_shared_parent(
                    *stru_res2,
                    [](passlet::passlet_t *ths, const node_base *v)
                            -> passlet::structural_result_t * {
                        if (!v->attr_) return nullptr;
                        return v->attr_
                                ->get_or_null<passlet::structural_result_t>(
                                        "builder.parent_node");
                    },
                    true, true))
                         ->shared_from_this()};
}

expr &get_base_tensor(expr &buffer) {
    expr *tsr = &buffer;
    while (!tsr->isa<tensor>()) {
        COMPILE_ASSERT(
                tsr->isa<tensorptr>(), "Only tensor or tensorptr is accepted")
        auto base = tsr->static_as<tensorptr>()->base_;
        COMPILE_ASSERT(base.isa<indexing>(),
                "tensor_ptr base should be indexing, but got: " << base);
        tsr = &(base.static_as<indexing>()->ptr_);
    }
    return *tsr;
}

tensor get_real_tensor(const expr &buffer) {
    auto buf = buffer;
    return get_base_tensor(buf).checked_as<tensor>();
}

void set_base_tensor(expr &tptr, const expr &tsr) {
    get_base_tensor(tptr) = tsr;
}

namespace builder {

static thread_local builder_impl_t *cur_builder = nullptr;

builder_impl_t *get_current_builder() {
    return cur_builder;
}

void set_current_builder(builder_impl_t *b) {
    cur_builder = b;
}

expr make_constant(float val) {
    return make_expr<constant_node>(val);
};

expr make_constant(int32_t val) {
    return make_expr<constant_node>(static_cast<int64_t>(val));
};

expr make_constant(uint64_t val) {
    return make_expr<constant_node>(val);
};

expr make_constant(const std::vector<union_val> &val, sc_data_type_t dtype) {
    return make_expr<constant_node>(val, dtype);
}

expr make_var(sc_data_type_t type, const std::string &name) {
    return make_expr<var_node>(type, name);
};

expr make_cast(sc_data_type_t type, const expr_c &in) {
    return make_expr<cast_node>(type, in.remove_const());
}

func_t make_func(const std::string &name, const std::vector<expr> &params,
        stmt body, sc_data_type_t ret_type) {
    return func_t(new func_base(name, params, std::move(body), ret_type));
}

func_t make_func(const std::string &name, const std::vector<expr_c> &params,
        const stmt_c &body, sc_data_type_t ret_type) {
    return make_func(
            name, vector_remove_const(params), body.remove_const(), ret_type);
}

expr builder_impl_t::make_str(const std::string &str) {
    static std::atomic<int> counter = {0};
    expr ret = make_tensor("_str_0_" + std::to_string(++counter),
            std::vector<expr> {expr(str.size() + 1)}, datatypes::u8);
    push_var_tensor_def(ret, linkage::local);
    for (size_t i = 0; i < str.size(); i++) {
        push_assign(make_indexing(ret, expr(i)),
                make_expr<constant_node>(uint64_t(str.at(i)), datatypes::u8));
    }
    push_assign(make_indexing(ret, expr(str.size())),
            make_expr<constant_node>(uint64_t(0), datatypes::u8));
    return ret;
}

expr tensor_ptr(const expr &tens, const std::vector<expr> &idx,
        const std::vector<expr> &shape, bool is_slice) {
    COMPILE_ASSERT(tens.isa<tensor>() || tens.isa<tensorptr>(),
            "tensor_ptr only accepts a tensor or tensorptr, got: " << tens);
    const std::vector<expr> *real_shape;
    if (is_slice && shape.empty()) {
        if (tens.isa<tensor>()) {
            real_shape = &tens.static_as<tensor>()->dims_;
        } else {
            real_shape = &tens.static_as<tensorptr>()->shape_;
        }
    } else {
        real_shape = &shape;
    }
    return make_expr<tensorptr_node>(
            make_expr<indexing_node>(tens, idx, expr()), *real_shape, is_slice);
}

expr tensor_ptr(const expr_c &tens, const std::vector<expr_c> &idx,
        const std::vector<expr_c> &shape, bool is_slice) {
    return tensor_ptr(tens.remove_const(), vector_remove_const(idx),
            vector_remove_const(shape), is_slice);
}

expr tensor_ptr(const expr &tensor, std::initializer_list<expr> idx,
        std::initializer_list<expr> shape, bool is_slice) {
    return tensor_ptr(
            tensor, std::vector<expr>(idx), std::vector<expr>(shape), is_slice);
}

expr remake_binary(const expr_c &l, const expr_c &r, const expr_c &original) {
    if (original.isa<intrin_call>()) {
        auto orig_type = original.static_as<intrin_call>()->type_;
        switch (orig_type) {
            case intrin_type::min:
            case intrin_type::max:
            case intrin_type::int_and:
            case intrin_type::int_or:
            case intrin_type::int_xor:
            case intrin_type::shl:
            case intrin_type::shr:
                return copy_attr(*original,
                        make_expr<intrin_call_node>(orig_type,
                                std::vector<expr> {
                                        l.remove_const(), r.remove_const()},
                                *original.static_as<intrin_call>()
                                         ->intrin_attrs_));
            default: assert(0 && "Bad op for remake_binary");
        }
    }
    switch (original->node_type_) {
        case sc_expr_type::add: return copy_attr(*original, make_add(l, r));
        case sc_expr_type::sub: return copy_attr(*original, make_sub(l, r));
        case sc_expr_type::mul: return copy_attr(*original, make_mul(l, r));
        case sc_expr_type::div: return copy_attr(*original, make_div(l, r));
        case sc_expr_type::mod: return copy_attr(*original, make_mod(l, r));
        case sc_expr_type::cmp_eq:
            return copy_attr(*original, make_cmp_eq(l, r));
        case sc_expr_type::cmp_ne:
            return copy_attr(*original, make_cmp_ne(l, r));
        case sc_expr_type::cmp_lt:
            return copy_attr(*original, make_cmp_lt(l, r));
        case sc_expr_type::cmp_le:
            return copy_attr(*original, make_cmp_le(l, r));
        case sc_expr_type::cmp_gt:
            return copy_attr(*original, make_cmp_gt(l, r));
        case sc_expr_type::cmp_ge:
            return copy_attr(*original, make_cmp_ge(l, r));
        case sc_expr_type::logic_and:
            return copy_attr(*original, make_logic_and(l, r));
        case sc_expr_type::logic_or:
            return copy_attr(*original, make_logic_or(l, r));
        default: assert(0 && "Bad op for remake_binary"); return expr();
    }
}

#define GEN_BINARY(name) \
    expr make_##name(const expr_c &left, const expr_c &right) { \
        return make_expr<name##_node>( \
                left.remove_const(), right.remove_const()); \
    };

expr make_func_addr(func_t v) {
    return make_expr<func_addr_node>(std::move(v));
}

expr make_phi(const std::vector<expr> &values, bool is_loop_phi) {
    return make_expr<ssa_phi_node>(values, is_loop_phi);
}

intrin_call remake_intrin_call(
        const intrin_call_c &v, const std::vector<expr> &newargs) {
    return make_expr<intrin_call_node>(v->type_, newargs, *v->intrin_attrs_);
}

low_level_intrin remake_low_level_intrin(
        const low_level_intrin_c &v, const std::vector<expr> &newargs) {
    low_level_intrin new_intrin = v->remake().static_as<low_level_intrin>();
    new_intrin->args_ = newargs;
    return new_intrin;
}

expr make_x86_intrin(x86_intrin_type::x86_intrin_type_t type,
        const std::vector<expr> &args, const any_map_t &attrs) {
    auto intrin = make_expr<low_level_intrin_node>(
            low_level_intrin_kind::x86_general, // x86 low level kind
            static_cast<int64_t>(type), // x86 intrin type
            args, attrs);
    get_x86_intrinsic_handler(intrin->type_).on_initialize(*intrin);
    return intrin;
}

expr make_reinterpret(const expr_c &v, sc_data_type_t dtype) {
    any_map_t attr;
    attr[intrin_attr::out_dtype] = dtype;
    auto ret = make_expr<intrin_call_node>(intrin_type::reinterpret,
            std::vector<expr> {v.remove_const()}, attr);
    return ret;
}

expr make_isnan(const expr_c &v) {
    auto ret = make_expr<intrin_call_node>(intrin_type::isnan,
            std::vector<expr> {v.remove_const()}, any_map_t());
    ret->dtype_ = sc_data_type_t::boolean(v->dtype_.lanes_);
    return ret;
}

expr make_saturated_cast(const expr_c &v, sc_data_type_t dtype) {
    any_map_t attr;
    attr[intrin_attr::out_dtype] = dtype;
    auto ret = make_expr<intrin_call_node>(intrin_type::saturated_cast,
            std::vector<expr> {v.remove_const()}, attr);
    return ret;
}

expr make_round_and_cast(const expr_c &v, sc_data_type_t dtype) {
    return make_expr<intrin_call_node>(intrin_type::round_and_cast,
            std::vector<expr> {v.remove_const()},
            any_map_t {{intrin_attr::out_dtype, dtype}});
}

expr make_min(const expr_c &left, const expr_c &right) {
    return make_expr<intrin_call_node>(intrin_type::min,
            std::vector<expr> {left.remove_const(), right.remove_const()},
            any_map_t());
}

expr make_max(const expr_c &left, const expr_c &right) {
    return make_expr<intrin_call_node>(intrin_type::max,
            std::vector<expr> {left.remove_const(), right.remove_const()},
            any_map_t());
}

expr make_int_and(const expr_c &left, const expr_c &right) {
    return make_expr<intrin_call_node>(intrin_type::int_and,
            std::vector<expr> {left.remove_const(), right.remove_const()},
            any_map_t());
}

expr make_int_or(const expr_c &left, const expr_c &right) {
    return make_expr<intrin_call_node>(intrin_type::int_or,
            std::vector<expr> {left.remove_const(), right.remove_const()},
            any_map_t());
}

expr make_int_xor(const expr_c &left, const expr_c &right) {
    return make_expr<intrin_call_node>(intrin_type::int_xor,
            std::vector<expr> {left.remove_const(), right.remove_const()},
            any_map_t());
}

expr make_shl(const expr_c &left, const expr_c &right) {
    return make_expr<intrin_call_node>(intrin_type::shl,
            std::vector<expr> {left.remove_const(), right.remove_const()},
            any_map_t());
}

expr make_shr(const expr_c &left, const expr_c &right) {
    return make_expr<intrin_call_node>(intrin_type::shr,
            std::vector<expr> {left.remove_const(), right.remove_const()},
            any_map_t());
}

expr make_abs(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::abs,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_round(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::round,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_floor(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::floor,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_ceil(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::ceil,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_exp(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::exp,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_log(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::log,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_erf(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::erf,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_sqrt(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::sqrt,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_rsqrt(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::rsqrt,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_reduce_add(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::reduce_add,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_reduce_mul(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::reduce_mul,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_reduce_max(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::reduce_max,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_reduce_min(const expr_c &v) {
    return make_expr<intrin_call_node>(intrin_type::reduce_min,
            std::vector<expr> {v.remove_const()}, any_map_t());
}

expr make_broadcast(const expr_c &v, int lanes) {
    return make_expr<intrin_call_node>(intrin_type::broadcast,
            std::vector<expr> {v.remove_const()}, any_map_t {{"lanes", lanes}});
}

expr make_fmadd(const expr_c &v_a, const expr_c &v_b, const expr_c &v_c) {
    return make_expr<intrin_call_node>(intrin_type::fmadd,
            std::vector<expr> {
                    v_a.remove_const(), v_b.remove_const(), v_c.remove_const()},
            any_map_t());
}

expr make_unpack_low(const expr_c &v_a, const expr_c &v_b, int elem_bits) {
    return make_expr<intrin_call_node>(intrin_type::unpack_low,
            std::vector<expr> {v_a.remove_const(), v_b.remove_const()},
            any_map_t {{"elem_bits", elem_bits}});
}

expr make_unpack_high(const expr_c &v_a, const expr_c &v_b, int elem_bits) {
    return make_expr<intrin_call_node>(intrin_type::unpack_high,
            std::vector<expr> {v_a.remove_const(), v_b.remove_const()},
            any_map_t {{"elem_bits", elem_bits}});
}

expr make_shuffle(const expr_c &v_a, const expr_c &v_b, const int &v_c,
        const int &type_bits) {
    return make_expr<intrin_call_node>(intrin_type::shuffle,
            std::vector<expr> {v_a.remove_const(), v_b.remove_const()},
            any_map_t {{"shuffle_imm", v_c}, {"type_bits", type_bits}});
}

expr make_permute(const expr_c &v_a, const expr_c &v_b, const int &v_c,
        const int &type_bits) {
    return make_expr<intrin_call_node>(intrin_type::permute,
            std::vector<expr> {v_a.remove_const(), v_b.remove_const()},
            any_map_t {{"permute_imm", v_c}, {"type_bits", type_bits}});
}

expr make_gather(const expr_c &addr, const expr_c &indices) {
    return make_expr<intrin_call_node>(intrin_type::gather,
            std::vector<expr> {addr.remove_const(), indices.remove_const()},
            any_map_t());
}

expr make_permutex2var(
        const expr_c &v_a, const expr_c &v_b, const expr_c &v_c) {
    return make_expr<intrin_call_node>(intrin_type::permutex2var,
            std::vector<expr> {
                    v_a.remove_const(), v_b.remove_const(), v_c.remove_const()},
            any_map_t());
}

expr make_permutexvar(const expr_c &idx, const expr_c &v, const int lanes) {
    return make_expr<intrin_call_node>(intrin_type::permutexvar,
            std::vector<expr> {idx.remove_const(), v.remove_const()},
            any_map_t {{"lanes", lanes}});
}

expr make_insert(const expr_c &v_a, const expr_c &v_b, const int imm) {
    return make_expr<intrin_call_node>(intrin_type::insert,
            std::vector<expr> {v_a.remove_const(), v_b.remove_const()},
            any_map_t {{"insert_imm", imm}});
}

expr make_extract(const expr_c &v_a, const int imm, const int lanes) {
    return make_expr<intrin_call_node>(intrin_type::extract,
            std::vector<expr> {v_a.remove_const()},
            any_map_t {{"extract_imm", imm}, {"lanes", lanes}});
}

expr make_read_struct(const expr_c &in, const std::string &struct_name,
        const int &field_name) {
    return make_expr<intrin_call_node>(intrin_type::read_struct,
            std::vector<expr> {in.remove_const()},
            any_map_t {{intrin_attr::struct_name, struct_name},
                    {intrin_attr::struct_field, field_name}});
}

expr make_write_struct(const expr_c &in, const expr_c &field,
        const std::string &struct_name, const int &field_name) {
    return make_expr<intrin_call_node>(intrin_type::write_struct,
            std::vector<expr> {in.remove_const(), field.remove_const()},
            any_map_t {{intrin_attr::struct_name, struct_name},
                    {intrin_attr::struct_field, field_name}});
}

expr make_get_group_id(uint64_t par_for_level_id) {
    return make_expr<intrin_call_node>(intrin_type::get_group_id,
            std::vector<expr> {par_for_level_id}, any_map_t());
}

expr make_get_group_thread_id(int par_for_level_id) {
    return make_expr<intrin_call_node>(intrin_type::get_group_thread_id,
            std::vector<expr> {par_for_level_id}, any_map_t());
}

GEN_BINARY(add);
GEN_BINARY(sub);
GEN_BINARY(mul);
GEN_BINARY(div);
GEN_BINARY(mod);

GEN_BINARY(cmp_eq);
GEN_BINARY(cmp_ne);
GEN_BINARY(cmp_lt);
GEN_BINARY(cmp_le);
GEN_BINARY(cmp_gt);
GEN_BINARY(cmp_ge);

GEN_BINARY(logic_and);
GEN_BINARY(logic_or);

expr make_logic_not(const expr_c &in) {
    return make_expr<logic_not_node>(in.remove_const());
}

expr make_select(const expr_c &cond, const expr_c &l, const expr_c &r) {
    return make_expr<select_node>(
            cond.remove_const(), l.remove_const(), r.remove_const());
}

expr make_clip(
        const expr_c &in, const expr_c &clip_min, const expr_c &clip_max) {
    expr ge_clip_min = builder::make_min(in, clip_max);
    return builder::make_max(ge_clip_min, clip_min);
}

expr make_indexing(const expr &ptr, const std::vector<expr> &idx,
        uint16_t length, const expr &mask) {
    return make_expr<indexing_node>(ptr, idx, length, mask);
}

expr make_indexing(const expr_c &ptr, const std::vector<expr_c> &idx,
        uint16_t length, const expr_c &mask) {
    return make_expr<indexing_node>(ptr.remove_const(),
            vector_remove_const(idx), length, mask.remove_const());
}

expr make_indexing(const expr &ptr, std::initializer_list<expr> idx,
        uint16_t length, const expr &mask) {
    return make_expr<indexing_node>(ptr, std::vector<expr>(idx), length, mask);
}

expr make_indexing(const expr_c &ptr, const expr_c &idx, uint16_t length,
        const expr_c &mask) {
    return make_expr<indexing_node>(ptr.remove_const(),
            std::vector<expr> {idx.remove_const()}, length,
            mask.remove_const());
}

expr make_call(const func_t &func, const std::vector<expr> &args) {
    return make_expr<call_node>(func, args);
}

expr make_call(const func_c &func, const std::vector<expr_c> &args) {
    return make_call(std::const_pointer_cast<func_base>(func),
            vector_remove_const(args));
}

expr remake_call(
        const func_t &func, const std::vector<expr> &args, const call_c &old) {
    return copy_attr(*old,
            make_expr<call_node>(func, args,
                    std::vector<call_node::parallel_attr_t> {old->para_attr_}));
}

expr remake_call(const func_c &func, const std::vector<expr_c> &args,
        const call_c &old) {
    return remake_call(std::const_pointer_cast<func_base>(func),
            vector_remove_const(args), old);
}

expr make_tensor(const std::string &name, const std::vector<expr> &dims,
        sc_data_type_t dtype, address_space addrspace,
        const std::shared_ptr<static_data_t> &init_value,
        const std::vector<expr> &strides) {
    return make_expr<tensor_node>(
            dtype, name, dims, addrspace, init_value, strides);
}

expr make_tensor(const std::string &name, const std::vector<expr_c> &dims,
        sc_data_type_t dtype, address_space addrspace,
        const std::shared_ptr<static_data_t> &init_value,
        const std::vector<expr_c> &strides) {
    return make_expr<tensor_node>(dtype, name, vector_remove_const(dims),
            addrspace, init_value, vector_remove_const(strides));
}

expr make_tensor(const std::string &name, std::initializer_list<expr> dims,
        sc_data_type_t dtype, address_space addrspace,
        const std::shared_ptr<static_data_t> &init_value,
        std::initializer_list<expr> strides) {
    return make_tensor(name, std::vector<expr>(dims), dtype, addrspace,
            init_value, std::vector<expr>(strides));
}

expr make_stensor(const std::string &name, const std::vector<expr> &dims,
        const std::vector<expr> &strides, sc_data_type_t dtype,
        address_space addrspace,
        const std::shared_ptr<static_data_t> &init_value) {
    COMPILE_ASSERT(strides.size() == dims.size(),
            "Dims and strides shall have same length.");
    return make_tensor(name, dims, dtype, addrspace, init_value, strides);
}

expr make_stensor(const std::string &name, const std::vector<expr_c> &dims,
        const std::vector<expr_c> &strides, sc_data_type_t dtype,
        address_space addrspace,
        const std::shared_ptr<static_data_t> &init_value) {
    COMPILE_ASSERT(strides.size() == dims.size(),
            "Dims and strides shall have same length.");
    return make_tensor(name, dims, dtype, addrspace, init_value, strides);
}

void builder_impl_t::basic_block_t::emit(const stmt &stmt) {
    body.push_back(stmt);
}

stmt builder_impl_t::basic_block_t::get() {
    return make_stmt<stmts_node_t>(std::move(body));
}

builder_impl_t::basic_block_t &builder_impl_t::get_current_scope() {
    assert(!scopes.empty());
    return scopes.back();
}

void builder_impl_t::emit(const stmt &s) {
    COMPILE_ASSERT(!scopes.empty(),
            "Emitting to empty scope stack. You need to call "
            "push_scope() first");
    get_current_scope().emit(s);
}

stmt builder_impl_t::pop_scope() {
    auto ret = scopes.back().get();
    for (auto &s : ret.checked_as<stmts>()->seq_) {
        add_parent_node(s, ret);
    }
    add_parent_node(ret, stmt());
    scopes.pop_back();
    return ret;
}

void builder_impl_t::push_scope() {
    scopes.emplace_back();
}

stmts builder_impl_t::push_anchor() {
    auto s = make_stmt<stmts_node_t>(std::vector<stmt> {});
    emit(s);
    return s;
}

stmt make_assign_unattached(const expr_c &var, const expr_c &value) {
    return make_stmt<assign_node_t>(var.remove_const(), value.remove_const());
}

stmt make_stmts_unattached(const std::vector<stmt_c> &seq) {
    std::vector<stmt> s;
    s.reserve(seq.size());
    for (auto &v : seq) {
        s.emplace_back(v.remove_const());
    }
    return make_stmt<stmts_node_t>(std::move(s));
}

stmt make_if_else_unattached(const expr_c &condition, const stmt_c &then_case,
        const stmt_c &else_case) {
    return make_stmt<if_else_node_t>(condition.remove_const(),
            then_case.remove_const(), else_case.remove_const());
}

stmt make_evaluate_unattached(const expr_c &val) {
    return make_stmt<evaluate_node_t>(val.remove_const());
}

stmt make_returns_unattached(const expr_c &val) {
    return make_stmt<returns_node_t>(val.remove_const());
}

stmt make_var_tensor_def_unattached(
        const expr_c &var, linkage l, const expr_c &init) {
    return make_stmt<define_node_t>(var.remove_const(), l, init.remove_const());
}

stmt make_for_loop_unattached(const expr_c &var, const expr_c &iter_begin,
        const expr_c &iter_end, const expr_c &step, const stmt_c &body,
        bool incremental, for_type kind, int num_threads) {
    return make_stmt<for_loop_node_t>(var.remove_const(),
            iter_begin.remove_const(), iter_end.remove_const(),
            step.remove_const(), body.remove_const(), incremental, kind,
            num_threads);
}

stmt builder_impl_t::push_assign(const expr &var, const expr &value) {
    auto ret = make_stmt<assign_node_t>(var, value);
    emit(ret);
    return ret;
}

stmt builder_impl_t::push_if_else(const expr &condition_,
        const stmt &then_case_, const stmt &else_case_) {
    auto ret = make_stmt<if_else_node_t>(condition_, then_case_, else_case_);
    add_parent_node(then_case_, ret);
    if (else_case_.defined()) add_parent_node(else_case_, ret);
    emit(ret);
    return ret;
}

stmt builder_impl_t::push_evaluate(const expr &val) {
    auto ret = make_stmt<evaluate_node_t>(val);
    emit(ret);
    return ret;
}

stmt builder_impl_t::push_returns(const expr &val) {
    auto ret = make_stmt<returns_node_t>(val);
    emit(ret);
    return ret;
}

stmt builder_impl_t::push_var_tensor_def(
        const expr &var, linkage l, const expr &init) {
    auto ret = make_stmt<define_node_t>(var, l, init);
    emit(ret);
    return ret;
}

stmt builder_impl_t::push_for_loop(const expr &var_, const expr &iter_begin_,
        const expr &iter_end_, const expr &step_, const stmt &body_,
        bool incremental_, for_type kind, int num_threads) {
    auto ret = make_stmt<for_loop_node_t>(var_, iter_begin_, iter_end_, step_,
            body_, incremental_, kind, num_threads);
    add_parent_node(body_, ret);
    emit(ret);
    return ret;
}

stmt builder_impl_t::brgemm(const expr_c &x, const expr_c &w, const expr_c &y,
        const expr_c &blocks, const expr_c &M, const expr_c &N, const expr_c &K,
        const expr_c &ldx, const expr_c &ldw, const expr_c &ldy,
        const expr_c &x_block_stride, const expr_c &w_block_stride,
        const std::vector<expr> &postops_data, const expr_c &c_buf,
        const expr_c &bd_mask_idx, const brgemm_args::extra_args_t &extras) {
    auto args = std::vector<expr> {x.remove_const(), w.remove_const(),
            y.remove_const(), blocks.remove_const(), M.remove_const(),
            N.remove_const(), K.remove_const(), ldx.remove_const(),
            ldw.remove_const(), ldy.remove_const(),
            x_block_stride.remove_const(), w_block_stride.remove_const()};
    args.insert(args.end(), postops_data.begin(), postops_data.end());
    args.emplace_back(c_buf.remove_const());
    args.emplace_back(bd_mask_idx.remove_const());
    return push_evaluate(make_expr<intrin_call_node>(intrin_type::brgemm, args,
            any_map_t {{intrin_attr::brgemm_extras, extras}}));
}

stmt builder_impl_t::list_brgemm(const expr_c &x, const expr_c &w,
        const expr_c &y, const expr_c &blocks, const expr_c &M, const expr_c &N,
        const expr_c &K, const expr_c &ldx, const expr_c &ldw,
        const expr_c &ldy, const expr_c &x_block_stride,
        const expr_c &w_block_stride, const expr_c &len,
        const std::vector<expr> &postops_data, const expr_c &c_buf,
        const expr_c &bd_mask_idx, const brgemm_args::extra_args_t &extras) {
    auto args = std::vector<expr> {x.remove_const(), w.remove_const(),
            y.remove_const(), blocks.remove_const(), M.remove_const(),
            N.remove_const(), K.remove_const(), ldx.remove_const(),
            ldw.remove_const(), ldy.remove_const(),
            x_block_stride.remove_const(), w_block_stride.remove_const(),
            len.remove_const()};
    args.insert(args.end(), postops_data.begin(), postops_data.end());
    args.emplace_back(c_buf.remove_const());
    args.emplace_back(bd_mask_idx.remove_const());
    return push_evaluate(make_expr<intrin_call_node>(intrin_type::list_brgemm,
            args, any_map_t {{intrin_attr::brgemm_extras, extras}}));
}
} // namespace builder
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

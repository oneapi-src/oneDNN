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
#include <iostream>
#include <limits>
#include <string.h>
#include <utility>

#include "builder.hpp"
#include "intrinsics.hpp"
#include "ir_comparer.hpp"
#include "ir_utils.hpp"
#include "sc_expr.hpp"
#include "sc_function.hpp"
#include "ssa_data.hpp"
#include "visitable.hpp"
#include <compiler/dimensions.hpp>
#include <compiler/ir/pass/printer.hpp>
#include <util/any_map.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

any_t &node_base::temp_data() const {
    if (!temp_data_) {
        const_cast<node_base *>(this)->temp_data_ = utils::make_unique<any_t>();
    }
    return *temp_data_;
}

static any_t empty_any;

const any_t &node_base::get_temp_data() const {
    if (!temp_data_) { return empty_any; }
    return *temp_data_;
}

any_map_t &node_base::attr() {
    if (!attr_) { attr_ = utils::make_unique<any_map_t>(); }
    return *attr_;
}

ostream &operator<<(ostream &os, sc_expr_type val) {
    switch (val) {
#define HANDLE_CASE(X) \
    case sc_expr_type::X: os << "sc_expr_type::" #X; break;

        HANDLE_CASE(undef)
        HANDLE_CASE(constant)
        HANDLE_CASE(var)
        HANDLE_CASE(cast)
        HANDLE_CASE(add)
        HANDLE_CASE(sub)
        HANDLE_CASE(mul)
        HANDLE_CASE(div)
        HANDLE_CASE(mod)
        HANDLE_CASE(cmp_eq)
        HANDLE_CASE(cmp_ne)
        HANDLE_CASE(cmp_lt)
        HANDLE_CASE(cmp_le)
        HANDLE_CASE(cmp_gt)
        HANDLE_CASE(cmp_ge)
        HANDLE_CASE(logic_and)
        HANDLE_CASE(logic_or)
        HANDLE_CASE(logic_not)
        HANDLE_CASE(select)
        HANDLE_CASE(indexing)
        HANDLE_CASE(call)
        HANDLE_CASE(tensor)
        HANDLE_CASE(tensorptr)
        HANDLE_CASE(intrin_call)
        HANDLE_CASE(func_addr)
        HANDLE_CASE(ssa_phi)
#undef HANDLE_CASE
        default: os << "(unrecognized sc_expr_type value)"; break;
    }
    return os;
}

ostream &operator<<(ostream &os, intrin_type val) {
    switch (val) {
#define HANDLE_CASE(X) \
    case intrin_type::X: os << "intrin_type::" #X; break;

        HANDLE_CASE(min)
        HANDLE_CASE(max)
        HANDLE_CASE(abs)
        HANDLE_CASE(round)
        HANDLE_CASE(floor)
        HANDLE_CASE(ceil)
        HANDLE_CASE(exp)
        HANDLE_CASE(sqrt)
        HANDLE_CASE(rsqrt)
        HANDLE_CASE(reduce_add)
        HANDLE_CASE(reduce_mul)
        HANDLE_CASE(reduce_max)
        HANDLE_CASE(reduce_min)
        HANDLE_CASE(fmadd)
        HANDLE_CASE(unpack_low)
        HANDLE_CASE(unpack_high)
        HANDLE_CASE(shuffle)
        HANDLE_CASE(permute)
        HANDLE_CASE(int_and)
        HANDLE_CASE(int_or)
        HANDLE_CASE(int_xor)
        HANDLE_CASE(reinterpret)
        HANDLE_CASE(broadcast)
        HANDLE_CASE(isnan)
        HANDLE_CASE(shl)
        HANDLE_CASE(shr)
        HANDLE_CASE(permutex2var)
        HANDLE_CASE(permutexvar)
        HANDLE_CASE(insert)
        HANDLE_CASE(extract)
        HANDLE_CASE(load_const_mem)
        HANDLE_CASE(brgemm)
        HANDLE_CASE(list_brgemm)
        HANDLE_CASE(NUM_INTRINSICS)
#undef HANDLE_CASE
        default: os << "(unrecognized intrin_type value)"; break;
    }
    return os;
}

ostream &operator<<(ostream &os, x86_intrin_type::x86_intrin_type_t val) {
    switch (val) {
#define HANDLE_CASE(X) \
    case x86_intrin_type::X: os << "x86_intrin_type::" #X; break;
        HANDLE_CASE(avx_broadcast_idx)
        HANDLE_CASE(NUM_INTRINSICS)
#undef HANDLE_CASE
        default: os << "(unrecognized x86_intrin_type value)"; break;
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, linkage val) {
    switch (val) {
        case linkage::public_global: os << "linkage::public_global"; break;
        case linkage::private_global: os << "linkage::private_global"; break;
        case linkage::static_local: os << "linkage::static_local"; break;
        case linkage::local: os << "linkage::local"; break;
    }
    return os;
}

node_base::~node_base() = default;
expr_base::~expr_base() = default;

expr_base::expr_base() = default;
expr_base::expr_base(sc_data_type_t type) : dtype_(type) {}
expr_base::expr_base(sc_expr_type exp_type) : node_type_(exp_type) {}
expr_base::expr_base(sc_data_type_t type, sc_expr_type exp_type)
    : dtype_(type), node_type_(exp_type) {}

expr::node_ptr(float v) : parent(builder::make_constant(v)) {}
expr::node_ptr(int32_t v) : parent(builder::make_constant(v)) {}
expr::node_ptr(uint64_t v) : parent(builder::make_constant(v)) {}
expr::node_ptr(bool v)
    : node_ptr(make_expr<constant_node>(uint64_t(v), datatypes::boolean)) {}

expr_c::node_ptr(float v) : parent(builder::make_constant(v)) {}
expr_c::node_ptr(int32_t v) : parent(builder::make_constant(v)) {}
expr_c::node_ptr(uint64_t v) : parent(builder::make_constant(v)) {}
expr_c::node_ptr(bool v)
    : node_ptr(make_expr<constant_node>(uint64_t(v), datatypes::boolean)) {}

expr::lvalue_proxy_t::lvalue_proxy_t() : require_remake_(true) {}

expr::lvalue_proxy_t::lvalue_proxy_t(expr data, bool require_remake)
    : data_(std::move(data)), require_remake_(require_remake) {}

expr expr::lvalue_proxy_t::get() const {
    if (require_remake_) {
        return data_->remake();
    } else {
        return expr(data_);
    }
}

expr::lvalue_proxy_t::operator expr() const {
    return get();
}

expr::lvalue_proxy_t::operator expr_c() const {
    return get();
}

void expr::lvalue_proxy_t::operator=(const expr &other) const {
    builder::get_current_builder()->push_assign(get(), other);
}

void expr::lvalue_proxy_t::operator=(expr::lvalue_proxy_t &other) const {
    this->operator=(other.get());
}

expr::lvalue_proxy_t expr::lvalue_proxy_t::operator[](
        const std::vector<expr> &index) const {
    return expr::lvalue_proxy_t(builder::make_indexing(*this, index), true);
}

expr::lvalue_proxy_t expr::lvalue_proxy_t::operator[](expr index) const {
    return expr::lvalue_proxy_t(
            builder::make_indexing(*this, std::move(index)), true);
}

expr::lvalue_proxy_t expr::lvalue_proxy_t::operator[](
        const span_t &index) const {
    return expr::lvalue_proxy_t(get()[index], true);
}

expr::lvalue_proxy_t::lvalue_proxy_t(expr::lvalue_proxy_t &&other)
    : data_(std::move(other.data_)), require_remake_(other.require_remake_) {}

expr::lvalue_proxy_t::lvalue_proxy_t(const expr::lvalue_proxy_t &other)
        = default;

expr::lvalue_proxy_t expr::operator[](const std::vector<expr> &index) const {
    return expr::lvalue_proxy_t(builder::make_indexing(*this, index), true);
}

expr::lvalue_proxy_t expr::operator[](expr index) {
    return expr::lvalue_proxy_t(
            builder::make_indexing(*this, std::move(index)), true);
}

expr::lvalue_proxy_t expr::operator[](const span_t &index) const {
    std::vector<expr> idx;
    idx.reserve(index.index_.size());
    for (auto &i : index.index_) {
        idx.emplace_back(i);
    }

    return expr::lvalue_proxy_t(
            builder::make_indexing(*this, idx, index.length_, index.mask_),
            true);
}

void print_indents(ostream &os, int indent) {
    for (int i = 0; i < indent; i++) {
        os << "  ";
    }
}

ostream &operator<<(ostream &os, const expr_c &e) {
    return os << e.get();
}

ostream &operator<<(ostream &os, const expr_base *e) {
    e->to_string(os);
    return os;
}

void expr_base::to_string(ostream &os) const {
    ir_printer_t p {os};
    p.dispatch(node_ptr_from_this());
}

bool expr_base::equals(expr_c other) const {
    ir_comparer cmper;
    return this->equals(std::move(other), cmper);
}

expr constant_node::remake() const {
    return copy_attr(*this, make_expr<constant_node>(value_, dtype_));
}

#define ASCAST_OR_RETURN(v, other) \
    using self \
            = node_ptr<typename std::remove_reference<decltype(*this)>::type, \
                    expr_base>; \
    if (!(v).isa<self>()) { \
        return ctx.set_result(node_ptr_from_this(), v, false); \
    } \
    if ((v)->dtype_ != dtype_) { \
        return ctx.set_result(node_ptr_from_this(), v, false); \
    } \
    auto other = v.static_as<self>(); // NOLINT(bugprone-macro-parentheses)

#define DYNCAST_OR_RETURN(v, other) \
    using self \
            = node_ptr<typename std::remove_reference<decltype(*this)>::type, \
                    expr_base>; \
    if (!(v).instanceof <self>()) { \
        return ctx.set_result(node_ptr_from_this(), v, false); \
    } \
    if ((v)->dtype_ != dtype_) { \
        return ctx.set_result(node_ptr_from_this(), v, false); \
    } \
    auto other = v.static_as<self>(); // NOLINT(bugprone-macro-parentheses)

#define RETURN(val) return ctx.set_result(node_ptr_from_this(), v, (val));

bool constant_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    if (other->value_.size() != value_.size()) return false;
    sc_data_etype etype = dtype_.is_etype_pointer() ? sc_data_etype::POINTER
                                                    : dtype_.type_code_;
    switch (etype) {
        case sc_data_etype::F16:
        case sc_data_etype::BF16:
        case sc_data_etype::F32:
            for (unsigned i = 0; i < value_.size(); i++) {
                if (other->value_[i].f32 != value_[i].f32) RETURN(false);
            }
            RETURN(true);
        case sc_data_etype::POINTER:
        case sc_data_etype::S32:
        case sc_data_etype::U8:
        case sc_data_etype::U16:
        case sc_data_etype::U32:
        case sc_data_etype::S8:
        case sc_data_etype::INDEX:
        case sc_data_etype::BOOLEAN:
            for (unsigned i = 0; i < value_.size(); i++) {
                if (other->value_[i].s64 != value_[i].s64) RETURN(false);
            }
            RETURN(true);
        default: assert(0 && "Unknown type for const");
    }
    return false;
}

expr var_node::remake() const {
    return copy_attr(*this, builder::make_var(dtype_, name_));
}

bool var_node::equals(expr_c v, ir_comparer &ctx) const {
    if (ctx.cmp_var_ref_) {
        if (ctx.get_expr_mapping(node_ptr_from_this(), v)) { return true; }
        RETURN(v.get() == this);
    }
    ASCAST_OR_RETURN(v, other);
    bool name_checking_passed = !ctx.cmp_names_ || (name_ == other->name_);
    if (!name_checking_passed
            || !ctx.check_or_set_expr_mapping(node_ptr_from_this(), v)) {
        RETURN(false);
    }
    // all other checks are done in ASCAST_OR_RETURN
    return true;
}

expr cast_node::remake() const {
    return copy_attr(*this, builder::make_cast(dtype_, in_));
}

bool cast_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    return in_->equals(other->in_, ctx);
}

bool binary_node::equals(expr_c v, ir_comparer &ctx) const {
    DYNCAST_OR_RETURN(v, other);
    if (node_type_ != other->node_type_) { RETURN(false); }
    auto ret = l_->equals(other->l_, ctx) && r_->equals(other->r_, ctx);
    if (!ret && ctx.cmp_commutative_
            && (node_type_ == sc_expr_type::add
                    || node_type_ == sc_expr_type::mul)) {
        return l_->equals(other->r_, ctx) && r_->equals(other->l_, ctx);
    }
    return ret;
}

bool logic_node::equals(expr_c v, ir_comparer &ctx) const {
    DYNCAST_OR_RETURN(v, other);
    if (node_type_ != other->node_type_) { RETURN(false); }
    auto ret = l_->equals(other->l_, ctx) && r_->equals(other->r_, ctx);
    if (!ret && ctx.cmp_commutative_) {
        return l_->equals(other->r_, ctx) && r_->equals(other->l_, ctx);
    }
    return ret;
}

bool cmp_node::equals(expr_c v, ir_comparer &ctx) const {
    DYNCAST_OR_RETURN(v, other);
    if (node_type_ != other->node_type_) { RETURN(false); }
    auto ret = l_->equals(other->l_, ctx) && r_->equals(other->r_, ctx);
    if (!ret && ctx.cmp_commutative_
            && (node_type_ == sc_expr_type::cmp_eq
                    || node_type_ == sc_expr_type::cmp_ne)) {
        return l_->equals(other->r_, ctx) && r_->equals(other->l_, ctx);
    }
    return ret;
}

#define GEN_BINARY(CLASS, OP) \
    expr CLASS##_node::remake() const { \
        return copy_attr(*this, builder::make_##CLASS(l_, r_)); \
    }

expr add_node::remake() const {
    auto ret = builder::make_add(l_, r_);
    if (dtype_ != datatypes::undef) ret->dtype_ = dtype_;
    return copy_attr(*this, std::move(ret));
}

GEN_BINARY(sub, " - ")
GEN_BINARY(mul, " * ")
GEN_BINARY(div, " / ")
GEN_BINARY(mod, " % ")
GEN_BINARY(cmp_eq, " == ")
GEN_BINARY(cmp_lt, " < ")
GEN_BINARY(cmp_le, " <= ")
GEN_BINARY(cmp_gt, " > ")
GEN_BINARY(cmp_ge, " >= ")
GEN_BINARY(cmp_ne, " != ")
GEN_BINARY(logic_and, " && ")
GEN_BINARY(logic_or, " || ")

bool logic_not_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    return in_->equals(other->in_, ctx);
}

expr logic_not_node::remake() const {
    return copy_attr(*this, builder::make_logic_not(in_));
}

bool select_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    return cond_->equals(other->cond_, ctx) && l_->equals(other->l_, ctx)
            && r_->equals(other->r_, ctx);
}

expr select_node::remake() const {
    return copy_attr(*this, builder::make_select(cond_, l_, r_));
}

expr indexing_node::remake() const {
    return copy_attr(
            *this, builder::make_indexing(ptr_, idx_, dtype_.lanes_, mask_));
}

bool indexing_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    return ptr_->equals(other->ptr_, ctx)
            && ctx.set_result(node_ptr_from_this(), v,
                    ctx.expr_arr_equals(idx_, other->idx_))
            && ctx.check_equals_may_null(mask_, other->mask_);
}

static sc_data_type_t get_func_type(node_base *func) {
    if (auto f = dynamic_cast<func_base *>(func)) {
        return f->ret_type_;
    } else {
        auto e = dynamic_cast<expr_base *>(func);
        assert(e);
        return e->attr().get<func_t>("prototype")->ret_type_;
    }
}

call_node::call_node(const std::shared_ptr<node_base> &func,
        const std::vector<expr> &args, std::vector<parallel_attr_t> &&para_attr)
    : expr_base(get_func_type(func.get()), sc_expr_type::call)
    , func_(func)
    , args_(args)
    , para_attr_(std::move(para_attr)) {}

call_node::call_node(const expr &func, const std::vector<expr> &args)
    : expr_base(func->attr().get<func_t>("prototype")->ret_type_,
            sc_expr_type::call)
    , func_(func.impl)
    , args_(args) {}

call_node::call_node(const func_t &func, const std::vector<expr> &args,
        std::vector<parallel_attr_t> &&para_attr)
    : expr_base(func->ret_type_, sc_expr_type::call)
    , func_(func)
    , args_(args)
    , para_attr_(std::move(para_attr)) {}

call_node::parallel_attr_t::parallel_attr_t(expr begin_, expr end_, expr step_)
    : begin_(std::move(begin_))
    , end_(std::move(end_))
    , step_(std::move(step_)) {}

func_t call_node::get_prototype() const {
    func_t the_func = std::dynamic_pointer_cast<func_base>(func_);
    func_t proto_func;
    if (!the_func) {
        auto the_expr = std::dynamic_pointer_cast<expr_base>(func_);
        assert(the_expr);
        proto_func = the_expr->attr().get<func_t>("prototype");
    } else {
        proto_func = the_func;
    }
    return proto_func;
}

expr call_node::remake() const {
    return copy_attr(*this, make_expr<call_node>(func_, args_));
}

// for the callee, just check if pointer is same
bool call_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    auto shared = node_ptr_from_this();
    auto &ths_func = *func_;
    auto &other_func = *other->func_;
    if (typeid(ths_func) != typeid(other_func)) { return false; }
    if (auto e = dynamic_cast<expr_base *>(func_.get())) {
        // if is expr, compare the callee by equals()
        if (!e->equals(
                    expr_c(std::static_pointer_cast<expr_base>(other->func_)),
                    ctx)) {
            return false;
        }
    } else {
        if (ctx.cmp_callee_) {
            auto f = dynamic_cast<func_base *>(func_.get());
            assert(f);
            if (!f->equals(
                        std::static_pointer_cast<func_base>(other->func_), ctx))
                return false;
        } else {
            if (!ctx.set_result(shared, v, func_ == other->func_)) return false;
        }
    }
    if (para_attr_.size() != other->para_attr_.size()) { RETURN(false); }
    for (unsigned i = 0; i < para_attr_.size(); i++) {
        auto &p = para_attr_[i];
        auto &op = other->para_attr_[i];
        if (!p.begin_->equals(op.begin_, ctx) || !p.end_->equals(op.end_, ctx)
                || !p.step_->equals(op.step_, ctx)) {
            return false;
        }
    }
    RETURN(ctx.expr_arr_equals(args_, other->args_));
}

tensor_node::tensor_node(sc_data_type_t dtype, const std::string &name,
        const std::vector<expr> &dims, address_space address_space,
        const std::shared_ptr<static_data_t> &init_value,
        const std::vector<expr> &strides)
    : expr_base(
            sc_data_type_t::pointerof(dtype.type_code_), sc_expr_type::tensor)
    , elem_dtype_(dtype)
    , dims_(dims)
    , name_(name)
    , address_space_(address_space)
    , init_value_(init_value)
    , strides_(strides) {
    if (strides_.empty()) { strides_ = dims_to_dense_stride(dims_); }
}

void tensor_node::to_string_full(ir_printer_t &printer) {
    auto &os = printer.os_;
    printer.do_dispatch(node_ptr_from_this()) << ": [" << elem_dtype_ << " * ";
    if (!dims_.empty()) {
        for (unsigned i = 0; i < dims_.size() - 1; i++) {
            printer.do_dispatch(dims_.at(i)) << " * ";
        }
        printer.do_dispatch(dims_.back());
    }
    os << ']';
    if (address_space_ != address_space::automatic) {
        switch (address_space_) {
            case address_space::device: os << " device"; break;
            case address_space::host: os << " host"; break;
            default: assert(0); break;
        }
    }
    if (init_value_ == tensor_node::get_zero_tensor_initializer()) {
        os << "{zero_init}";
    } else if (init_value_ && init_value_->size_ == sizeof(union_val)) {
        union_val val = *reinterpret_cast<union_val *>(init_value_->data_);
        os << "{value:" << val.u64 << '}';
    }
}

const std::shared_ptr<static_data_t> &
tensor_node::get_zero_tensor_initializer() {
    static std::shared_ptr<static_data_t> ret
            = std::make_shared<static_data_t>(nullptr, 0);
    return ret;
}

std::shared_ptr<static_data_t> tensor_node::make_tensor_initializer(
        union_val val) {
    union_val theval;
    theval.u64 = 0;
    theval = val;
    return std::make_shared<static_data_t>(&theval, sizeof(val));
}

expr tensor_node::remake() const {
    return copy_attr(*this,
            builder::make_stensor(name_, dims_, strides_, elem_dtype_,
                    address_space_, init_value_));
}

// ignore the names
bool tensor_node::equals(expr_c v, ir_comparer &ctx) const {
    if (ctx.cmp_var_ref_) {
        if (ctx.get_expr_mapping(node_ptr_from_this(), v)) { return true; }
        RETURN(v.get() == this);
    }
    ASCAST_OR_RETURN(v, other);
    bool name_checking_passed = !ctx.cmp_names_ || (name_ == other->name_);
    if (!name_checking_passed || address_space_ != other->address_space_
            || dtype_ != other->dtype_ || elem_dtype_ != other->elem_dtype_
            || !ctx.check_or_set_expr_mapping(node_ptr_from_this(), v)) {
        RETURN(false);
    }
    if (init_value_) {
        if (!other->init_value_) { RETURN(false); }
        if (init_value_->size_ != other->init_value_->size_
                || memcmp(init_value_->data_, other->init_value_->data_,
                        other->init_value_->size_)) {
            RETURN(false);
        }
    } else {
        if (other->init_value_) { RETURN(false); }
    }
    RETURN(ctx.expr_arr_equals(dims_, other->dims_)
            && ctx.expr_arr_equals(strides_, other->strides_));
}

expr tensorptr_node::remake() const {
    return copy_attr(
            *this, make_expr<tensorptr_node>(base_, shape_, is_slice_));
}

bool tensorptr_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    if (other->is_slice_ != is_slice_) { RETURN(false); }
    if (!ctx.expr_arr_equals(shape_, other->shape_)) { RETURN(false); }
    return base_->equals(other->base_, ctx);
}

intrin_call_node::intrin_call_node(intrin_type intrin,
        const std::vector<expr> &args, const any_map_t &attrs)
    : expr_base(sc_expr_type::intrin_call)
    , type_(intrin)
    , args_(args)
    , intrin_attrs_(utils::make_unique<any_map_t>(attrs)) {
    get_intrinsic_handler(type_).on_initialize(*this);
}

expr intrin_call_node::remake() const {
    return copy_attr(
            *this, make_expr<intrin_call_node>(type_, args_, *intrin_attrs_));
}

bool intrin_call_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    if (type_ != other->type_) { RETURN(false); }
    if (type_ == intrin_type::brgemm || type_ == intrin_type::list_brgemm) {
        auto &extra_args = intrin_attrs_->get<brgemm_args::extra_args_t>(
                intrin_attr::brgemm_extras);
        auto &other_extra_args
                = other->intrin_attrs_->get<brgemm_args::extra_args_t>(
                        intrin_attr::brgemm_extras);
        if (extra_args != other_extra_args) { RETURN(false); }
    }
    RETURN(ctx.expr_arr_equals(args_, other->args_));
}

bool intrin_call_node::check_brgemm_arg_size(size_t expected_size) const {
    if (type_ != intrin_type::brgemm && type_ != intrin_type::list_brgemm) {
        return true;
    }

    return args_.size() == expected_size;
}

expr func_addr_node::remake() const {
    return copy_attr(*this, make_expr<func_addr_node>(func_));
}

bool func_addr_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    RETURN(func_ == other->func_);
}

ssa_phi_node::ssa_phi_node(const std::vector<expr> &values, bool is_loop_phi)
    : expr_base(type_code_), values_(values), is_loop_phi_(is_loop_phi) {
    COMPILE_ASSERT(!values_.empty(), "Phi node expects non-empty inputs");
    dtype_ = values_.begin()->get()->dtype_;
    for (auto &v : values_) {
        COMPILE_ASSERT(dtype_ == v->dtype_,
                "Phi node expects exprs with the same type, got "
                        << dtype_ << " v.s. " << v->dtype_);
    }
}

expr ssa_phi_node::remake() const {
    return copy_attr(*this, make_expr<ssa_phi_node>(values_, is_loop_phi_));
}

bool ssa_phi_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    RETURN(other->is_loop_phi_ == is_loop_phi_
            && ctx.expr_arr_equals(values_, other->values_));
}

low_level_intrin_node::low_level_intrin_node(low_level_intrin_kind kind,
        int64_t type, const std::vector<expr> &args, const any_map_t &attrs)
    : expr_base(sc_expr_type::low_level_intrin)
    , kind_(kind)
    , type_(type)
    , args_(args)
    , intrin_attrs_(utils::make_unique<any_map_t>(attrs)) {}

expr low_level_intrin_node::remake() const {
    auto ret = make_expr<low_level_intrin_node>(
            kind_, type_, args_, *intrin_attrs_);
    ret->dtype_ = dtype_;
    return copy_attr(*this, std::move(ret));
}

bool low_level_intrin_node::equals(expr_c v, ir_comparer &ctx) const {
    ASCAST_OR_RETURN(v, other);
    if (kind_ != other->kind_) { RETURN(false); }
    if (type_ != other->type_) { RETURN(false); }
    RETURN(ctx.expr_arr_equals(args_, other->args_));
}

const std::string &get_node_name(const expr &e) {
    tensor t = e.as<tensor>();
    if (t.get() != nullptr) { return t->name_; }

    var v = e.as<var>();
    if (v.get() != nullptr) { return v->name_; }

    COMPILE_ASSERT(
            false, "Not an expr_base subclass that has a 'name_' member.");
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

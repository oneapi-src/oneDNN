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

#include "codegen_c.hpp"
#include <algorithm>
#include <assert.h>
#include <memory>
#include <string.h>
#include <string>
#include <utility>
#include <vector>
#include "../ir/viewer.hpp"
#include "codegen_c_internal.hpp"
#include "precodegen_passes.hpp"
#include <compiler/ir/intrinsics.hpp>
#include <compiler/ir/pass/func_dependency.hpp>
#include <compiler/ir/pass/printer.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/jit/symbol_resolver.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>
#include <util/scoped_timer.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
static constexpr const char *wrapper_postfix = "_0wrapper";

static std::string get_closure_wrapper_name(const std::string &name) {
    return name + "_0closurewrapper";
}

static void print_cpp_etype(ostream &os, sc_data_etype t) {
    switch (t) {
        case sc_data_etype::UNDEF: assert(0 && "Met undef"); break;
        case sc_data_etype::F16: os << "_Float16"; break;
        case sc_data_etype::BF16: os << "uint16_t"; break;
        case sc_data_etype::U16: os << "uint16_t"; break;
        case sc_data_etype::F32: os << "float"; break;
        case sc_data_etype::S32: os << "int32_t"; break;
        case sc_data_etype::U32: os << "uint32_t"; break;
        case sc_data_etype::S8: os << "int8_t"; break;
        case sc_data_etype::U8: os << "uint8_t"; break;
        case sc_data_etype::INDEX: os << "uint64_t"; break;
        case sc_data_etype::POINTER: os << "void*"; break;
        case sc_data_etype::VOID_T: os << "void"; break;
        case sc_data_etype::BOOLEAN: os << "bool"; break;
        case sc_data_etype::GENERIC: os << "generic_val"; break;
        default:
            if (etypes::is_pointer(t)) {
                print_cpp_etype(os, etypes::get_pointer_element(t));
                os << '*';
                return;
            }
            assert(0 && "Unknown type");
    }
}

ostream &print_cpp_type(ostream &os, sc_data_type_t dtype) {
    assert(dtype.lanes_ == 1 && "Only scalar types allowed here");
    print_cpp_etype(os, dtype.type_code_);
    return os;
}

ostream &codegen_c_vis::print_cpp_var_def(const var &v) {
    print_type(v->dtype_);
    *os << ' ' << v->name_;
    return *os;
}

ostream &codegen_c_vis::print_tensor_def(const tensor &v) {
    if (v->attr_ && v->attr_->get_or_else("volatile", false)) {
        *os << "volatile ";
    }
    print_type(v->elem_dtype_);
    auto ali_info = alias_info::get_alias_info(*v);
    if (ali_info && !ali_info->has_no_alias()) {
        *os << "* ";
    } else {
        *os << "* __restrict__ ";
    }
    *os << v->name_;
    return *os;
}

ostream &codegen_c_vis::print_param(const expr &e) {
    if (e.isa<var>()) {
        print_cpp_var_def(e.static_as<var>());
    } else {
        print_tensor_def(e.checked_as<tensor>());
    }
    return *os;
}

ostream &codegen_c_vis::print_func_params(const func_c &v, bool with_type) {
    if (!v->params_.empty()) {
        for (size_t i = 0; i < v->params_.size() - 1; i++) {
            expr e = v->params_[i];
            if (with_type) {
                print_param(e);
            } else {
                if (e.isa<var>()) {
                    *os << e.static_as<var>()->name_;
                } else {
                    *os << e.checked_as<tensor>()->name_;
                }
            }
            *os << ", ";
        }
        expr e = v->params_.back();
        if (with_type) {
            print_param(e);
        } else {
            if (e.isa<var>()) {
                *os << e.static_as<var>()->name_;
            } else {
                *os << e.checked_as<tensor>()->name_;
            }
        }
    }
    return *os;
}

void codegen_c_vis::print_type(sc_data_type_t dtype) {
    if (dtype.lanes_ == 1) {
        print_cpp_type(*os, dtype);
    } else {
        switch (dtype) {
            case sc_data_type_t::bf16(4): *os << "vec_u16x4"; break;
            case sc_data_type_t::bf16(8): *os << "vec_u16x8"; break;
            case sc_data_type_t::bf16(16): *os << "vec_u16x16"; break;
            case sc_data_type_t::bf16(32): *os << "vec_u16x32"; break;

            case sc_data_type_t::f16(4): *os << "vec_f16x4"; break;
            case sc_data_type_t::f16(8): *os << "vec_f16x8"; break;
            case sc_data_type_t::f16(16): *os << "vec_f16x16"; break;
            case sc_data_type_t::f16(32): *os << "vec_f16x32"; break;

            case sc_data_type_t::f32(4): *os << "vec_f32x4"; break;
            case sc_data_type_t::f32(8): *os << "vec_f32x8"; break;
            case sc_data_type_t::f32(16): *os << "vec_f32x16"; break;

            case sc_data_type_t::s32(4): *os << "vec_s32x4"; break;
            case sc_data_type_t::s32(8): *os << "vec_s32x8"; break;
            case sc_data_type_t::s32(16): *os << "vec_s32x16"; break;

            case sc_data_type_t::u32(4): *os << "vec_u32x4"; break;
            case sc_data_type_t::u32(8): *os << "vec_u32x8"; break;
            case sc_data_type_t::u32(16): *os << "vec_u32x16"; break;

            case sc_data_type_t::index(2): *os << "vec_u64x2"; break;
            case sc_data_type_t::index(4): *os << "vec_u64x4"; break;
            case sc_data_type_t::index(8): *os << "vec_u64x8"; break;

            case sc_data_type_t::u16(8): *os << "vec_u16x8"; break;
            case sc_data_type_t::u16(16): *os << "vec_u16x16"; break;
            case sc_data_type_t::u16(32): *os << "vec_u16x32"; break;

            case sc_data_type_t::s8(8): *os << "vec_s8x8"; break;
            case sc_data_type_t::s8(16): *os << "vec_s8x16"; break;
            case sc_data_type_t::s8(32): *os << "vec_s8x32"; break;
            case sc_data_type_t::s8(64): *os << "vec_s8x64"; break;

            case sc_data_type_t::u8(8): *os << "vec_u8x8"; break;
            case sc_data_type_t::u8(16): *os << "vec_u8x16"; break;
            case sc_data_type_t::u8(32): *os << "vec_u8x32"; break;
            case sc_data_type_t::u8(64): *os << "vec_u8x64"; break;

            case sc_data_type_t::boolean(4): *os << "uint8_t"; break;
            case sc_data_type_t::boolean(8): *os << "uint8_t"; break;
            case sc_data_type_t::boolean(16): *os << "uint16_t"; break;
            case sc_data_type_t::boolean(32): *os << "uint32_t"; break;
            case sc_data_type_t::boolean(64): *os << "uint64_t"; break;

            default:
                COMPILE_ASSERT(
                        0, "Cannot generate vector type for C++: " << dtype);
                break;
        }
    }
}

codegen_c_vis::codegen_c_vis(ostream *os, bool prototype_only, bool is_static)
    : os(os), prototype_only(prototype_only), is_static(is_static) {}

stmt_c codegen_c_vis::dispatch(stmt_c v) {
    if (v->attr_) {
        if (auto comments
                = v->attr_->get_or_null<std::vector<std::string>>("comments")) {
            for (auto &str : *comments) {
                *os << "// " << str << "\n";
                print_indents(*os, indents);
            }
        }
    }
    return ir_visitor_t::dispatch(std::move(v));
}

static const std::string &get_func_name(const func_c &v) {
    if (v->attr_) {
        if (auto pname = v->attr_->get_or_null<std::string>(
                    "temp.replace_func_name")) {
            return *pname;
        }
    }
    return v->name_;
}

func_c codegen_c_vis::dispatch(func_c v) {
    if (utils::string_startswith(v->name_, "_should_inline_")) { return v; }
    if (prototype_only) { print_func_comments(v, *os); }
    bool is_symbol_in_runtime
            = !is_offline_ && default_external_symbol_resolve(v->name_);
    if (!is_symbol_in_runtime) {
        *os << (is_static ? "static " : "extern \"C\" ");
    }
    print_type(v->ret_type_);
    const std::string &real_name = get_func_name(v);
    if (is_symbol_in_runtime) {
        *os << " (*" << real_name << "_fptr" << ')' << '(';
    } else {
        *os << " " << real_name << '(';
    }
    print_func_params(v);
    *os << ") noexcept";
    if (prototype_only) {
        if (v->attr_) {
            if (v->attr_->get_or_else(function_attrs::pure, false)) {
                if (is_symbol_in_runtime) {
                    // create a wrapper to symbol fptr for compiler function opt
                    *os << ";\n";
                    print_type(v->ret_type_);
                    *os << " __" << real_name << wrapper_postfix << '(';
                    print_func_params(v);
                    *os << ") noexcept __attribute__((const));\n";

                    print_type(v->ret_type_);
                    *os << " __" << real_name << wrapper_postfix << '(';
                    print_func_params(v);
                    *os << ") noexcept { return " << real_name << "_fptr(";
                    print_func_params(v, false);
                    *os << "); }";
                } else {
                    *os << " __attribute__((const))";
                }
            }
            if (v->attr_->get_or_else(function_attrs::no_alias, false)) {
                *os << " __attribute__((returns_nonnull))  ";
                if (!is_offline_) { *os << "/*"; }
                *os << "__attribute__((malloc))";
                if (!is_offline_) { *os << "*/"; }
            }
        }
        std::string tensor_idx = " __attribute__((nonnull (";
        bool has_tensor = false;
        for (size_t i = 0; i < v->params_.size(); i++) {
            if (v->params_[i].isa<tensor>()) {
                if (has_tensor) { tensor_idx += ','; }
                has_tensor = true;
                tensor_idx += std::to_string(i + 1);
            }
        }
        if (has_tensor) { *os << tensor_idx << ")))"; }
    }
    if (!prototype_only && v->body_.defined()) {
        dispatch(v->body_);
    } else {
        *os << ";";
    }
    return std::const_pointer_cast<func_base>(v);
}

void codegen_c_vis::view(constant_c v) {
    if (v->is_vector()) {
        print_type(v->dtype_);
        v->to_string(*os);
    } else if (v->dtype_ != datatypes::pointer && v->dtype_.is_pointer()) {
        (*os) << '(' << '(';
        print_type(v->dtype_);
        (*os) << ')' << v->value_.at(0).u64 << ')';
    } else {
        if (v->dtype_.is_etype(sc_data_etype::F16)) {
            (*os) << "(_Float16)";
            v->to_string(*os);
        } else {
            v->to_string(*os);
        }
    }
}

void codegen_c_vis::view(var_c v) {
    *os << v->name_;
}

void codegen_c_vis::view(cast_c v) {
    // if it is T* => void* cast, C++ can auto cast it
    if (v->in_->dtype_.is_pointer() && v->dtype_ == datatypes::pointer) {
        dispatch(v->in_);
        return;
    }
    // cast to generic_val
    if (v->dtype_ == datatypes::generic) {
        assert(v->in_->dtype_.lanes_ == 1
                && "casting to generic vector is not implemented");
        // we have constructors defined. C++ will handle the conversions
        dispatch(v->in_);
        return;
    }
    // cast from generic_val
    if (v->in_->dtype_ == datatypes::generic) {
        assert(v->in_->dtype_.lanes_ == 1
                && "casting from generic vector is not implemented");
        if (v->dtype_.is_etype_pointer()) {
            // prints (float*)(XXX.v_ptr)
            *os << '(';
            print_cpp_etype(*os, v->dtype_.type_code_);
            *os << ")(";
            dispatch(v->in_);
            *os << ".v_ptr)";
        } else {
            // prints xxx.v_float
            dispatch(v->in_);
            *os << ".v_";
            print_cpp_etype(*os, v->dtype_.type_code_);
        }
        return;
    }

    if (v->dtype_.lanes_ == 1) {
        if (v->in_->dtype_.is_etype(sc_data_etype::F32)
                && v->dtype_.is_etype(sc_data_etype::BF16)) {
            *os << "tobf16(";
            dispatch(v->in_);
            *os << ')';
        } else if (v->in_->dtype_.is_etype(sc_data_etype::F16)
                && v->dtype_.is_etype(sc_data_etype::F32)) {
            *os << "(float)";
            dispatch(v->in_);
        } else if (v->in_->dtype_.is_etype(sc_data_etype::F32)
                && v->dtype_.is_etype(sc_data_etype::F16)) {
            *os << "(_Float16)";
            dispatch(v->in_);
        } else {
            *os << '(';
            print_cpp_type(*os, v->dtype_) << ')';
            dispatch(v->in_);
        }

    } else {
        COMPILE_ASSERT(v->dtype_.lanes_ == v->in_->dtype_.lanes_,
                "Vector cast should have same lanes. Got "
                        << v->dtype_ << " vs " << v->in_->dtype_);
        if (v->in_->dtype_.is_etype(sc_data_etype::F32)
                && v->dtype_.is_etype(sc_data_etype::BF16)) {
            *os << "tobf16(";
            dispatch(v->in_);
            *os << ')';
        } else {
            *os << '(';
            print_type(v->dtype_);
            *os << ")(";
            dispatch(v->in_);
            *os << ')';
        }
    }
}

#define GEN_BINARY(TYPE) \
    void codegen_c_vis::print_binary(const TYPE &v, const char *op) { \
        *os << '('; \
        dispatch(v->l_); \
        *os << op; \
        dispatch(v->r_); \
        *os << ')'; \
    }

GEN_BINARY(binary_c);
GEN_BINARY(logic_c);
GEN_BINARY(cmp_c);

void codegen_c_vis::trinary_func_codegen_c(
        const std::vector<expr> &args, const char *funcname) {
    COMPILE_ASSERT(args.size() == 3,
            "Invalid arg size: " << args.size() << ", should be 3");
    auto &os = *this->os;
    os << funcname << "(";
    dispatch(args[0]);
    os << ", ";
    dispatch(args[1]);
    os << ", ";
    dispatch(args[2]);
    os << ')';
}

void codegen_c_vis::binary_func_codegen_c(
        const std::vector<expr> &args, const char *funcname) {
    COMPILE_ASSERT(args.size() == 2,
            "Invalid arg size: " << args.size() << ", should be 3");
    auto &os = *this->os;
    os << funcname << '(';
    dispatch(args[0]);
    os << ", ";
    dispatch(args[1]);
    os << ')';
}

void codegen_c_vis::unary_func_codegen_c(
        const expr &arg, const char *funcname) {
    auto &os = *this->os;
    os << funcname << '(';
    dispatch(arg);
    os << ')';
}

void codegen_c_vis::view(func_addr_c v) {
    if (!is_offline_ && default_external_symbol_resolve(v->func_->name_)) {
        *os << "(void*)" << v->func_->name_ << "_fptr";
    } else {
        *os << "(void*)&" << v->func_->name_;
    }
}

static const char *prefetch_names[]
        = {"_MM_HINT_T0", "_MM_HINT_T1", "_MM_HINT_T2", "_MM_HINT_NTA"};

void codegen_c_vis::view(intrin_call_c v) {
    switch (v->type_) {
        case intrin_type::min: binary_func_codegen_c(v->args_, "sc_min"); break;
        case intrin_type::max: binary_func_codegen_c(v->args_, "sc_max"); break;
        case intrin_type::abs:
            unary_func_codegen_c(v->args_[0], "sc_abs");
            break;
        case intrin_type::round:
            unary_func_codegen_c(v->args_[0], "sc_round");
            break;
        case intrin_type::floor:
            unary_func_codegen_c(v->args_[0], "sc_floor");
            break;
        case intrin_type::ceil:
            unary_func_codegen_c(v->args_[0], "sc_ceil");
            break;
        case intrin_type::exp:
            unary_func_codegen_c(v->args_[0], "sc_exp");
            break;
        case intrin_type::sqrt:
            unary_func_codegen_c(v->args_[0], "sc_sqrt");
            break;
        case intrin_type::rsqrt:
            unary_func_codegen_c(v->args_[0], "sc_rsqrt");
            break;
        case intrin_type::reduce_add:
            unary_func_codegen_c(v->args_[0], "sc_reduce_add");
            break;
        case intrin_type::reduce_mul:
            unary_func_codegen_c(v->args_[0], "sc_reduce_mul");
            break;
        case intrin_type::reduce_max:
            unary_func_codegen_c(v->args_[0], "sc_reduce_max");
            break;
        case intrin_type::reduce_min:
            unary_func_codegen_c(v->args_[0], "sc_reduce_min");
            break;
        case intrin_type::fmadd:
            trinary_func_codegen_c(v->args_, "sc_fmadd");
            break;
        case intrin_type::unpack_low:
            *os << "sc_unpack_low_";
            print_type(v->dtype_);
            *os << "_";
            *os << v->intrin_attrs_->get<int>("elem_bits");
            *os << "bits";
            *os << "(";
            dispatch(v->args_[0]);
            *os << ", ";
            dispatch(v->args_[1]);
            *os << ')';
            break;
        case intrin_type::unpack_high:
            *os << "sc_unpack_high_";
            print_type(v->dtype_);
            *os << "_";
            *os << v->intrin_attrs_->get<int>("elem_bits");
            *os << "bits";
            *os << "(";
            dispatch(v->args_[0]);
            *os << ", ";
            dispatch(v->args_[1]);
            *os << ')';
            break;
        case intrin_type::shuffle:
            *os << "sc_shuffle_";
            print_type(v->dtype_);
            *os << "_";
            *os << v->intrin_attrs_->get<int>("type_bits");
            *os << "bits";
            *os << "(";
            dispatch(v->args_[0]);
            *os << ", ";
            dispatch(v->args_[1]);
            *os << ", ";
            *os << v->intrin_attrs_->get<int>("shuffle_imm");
            *os << ')';
            break;
        case intrin_type::permute:
            *os << "sc_permute_";
            print_type(v->dtype_);
            *os << "(";
            dispatch(v->args_[0]);
            *os << ", ";
            dispatch(v->args_[1]);
            *os << ", ";
            *os << v->intrin_attrs_->get<int>("permute_imm");
            *os << ')';
            break;
        case intrin_type::prefetch: {
            *os << "_mm_prefetch(";
            dispatch(v->args_[0]);
            auto locality = v->intrin_attrs_->get<int>("locality");
            COMPILE_ASSERT(locality <= 3 && locality >= 0,
                    "bad locality for prefetch");
            *os << ", " << prefetch_names[locality] << ')';
            break;
        }
        case intrin_type::gather:
            *os << "sc_gather(";
            dispatch(v->args_[0]);
            *os << ", ";
            dispatch(v->args_[1]);
            *os << ")";
            break;
        case intrin_type::broadcast:
            print_type(v->dtype_);
            *os << "(";
            dispatch(v->args_[0]);
            *os << ')';
            break;
        case intrin_type::reinterpret:
            *os << "sc_reinterpret<";
            print_type(v->dtype_);
            *os << ">(";
            dispatch(v->args_[0]);
            *os << ')';
            break;
        case intrin_type::isnan:
            *os << "sc_isnan(";
            dispatch(v->args_[0]);
            *os << ')';
            break;
        case intrin_type::saturated_cast:
            *os << "sc_saturated_cast<";
            print_type(v->dtype_);
            *os << ">(";
            dispatch(v->args_[0]);
            *os << ')';
            break;
        case intrin_type::round_and_cast:
            *os << "sc_round_and_cast<";
            print_type(v->dtype_);
            *os << ">(";
            dispatch(v->args_[0]);
            *os << ')';
            break;
        case intrin_type::int_and:
            *os << '(';
            dispatch(v->args_[0]);
            *os << " & ";
            dispatch(v->args_[1]);
            *os << ')';
            break;
        case intrin_type::int_or:
            *os << '(';
            dispatch(v->args_[0]);
            *os << " | ";
            dispatch(v->args_[1]);
            *os << ')';
            break;
        case intrin_type::int_xor:
            *os << '(';
            dispatch(v->args_[0]);
            *os << " ^ ";
            dispatch(v->args_[1]);
            *os << ')';
            break;
        case intrin_type::shl:
            *os << '(';
            dispatch(v->args_[0]);
            *os << " << ";
            dispatch(v->args_[1]);
            *os << ')';
            break;
        case intrin_type::shr:
            *os << '(';
            dispatch(v->args_[0]);
            *os << " >> ";
            dispatch(v->args_[1]);
            *os << ')';
            break;
        case intrin_type::permutex2var:
            trinary_func_codegen_c(v->args_, "sc_permutex2var");
            break;
        case intrin_type::permutexvar:
            if (v->args_[0].isa<constant_c>()) {
                // vpermq have two different invocation. If is imm, we need to
                // do use these part.
                int lanes = v->intrin_attrs_->get<int>("lanes");
                int elem_bits = utils::get_sizeof_etype(
                                        v->args_[1]->dtype_.type_code_)
                        * 8 * lanes;
                auto suffix = std::to_string(elem_bits) + "bits";
                *os << "sc_permutexvar_";
                print_type(v->args_[1]->dtype_);
                *os << "_" + suffix;
                *os << '(';
                dispatch(v->args_[0]);
                *os << ',';
                dispatch(v->args_[1]);
                *os << ')';
            } else {
                binary_func_codegen_c(v->args_, "sc_permutexvar");
            }
            break;
        case intrin_type::insert:
            *os << "sc_insert_";
            print_type(v->dtype_);
            *os << "(";
            dispatch(v->args_[0]);
            *os << ", ";
            dispatch(v->args_[1]);
            *os << ", ";
            *os << v->intrin_attrs_->get<int>("insert_imm");
            *os << ')';
            break;
        case intrin_type::extract:
            *os << "sc_extract_";
            print_type(v->args_[0]->dtype_);
            *os << "(";
            dispatch(v->args_[0]);
            *os << ", ";
            *os << v->intrin_attrs_->get<int>("extract_imm");
            *os << ')';
            break;
        default: assert(0 && "Unknown intrinsic!"); break;
    }
}

void codegen_c_vis::view(add_c v) {
    print_binary(v, " + ");
}
void codegen_c_vis::view(sub_c v) {
    print_binary(v, " - ");
}
void codegen_c_vis::view(mul_c v) {
    print_binary(v, " * ");
}
void codegen_c_vis::view(div_c v) {
    print_binary(v, " / ");
}
void codegen_c_vis::view(mod_c v) {
    print_binary(v, " % ");
}

void codegen_c_vis::view(cmp_eq_c v) {
    print_binary(v, " == ");
}
void codegen_c_vis::view(cmp_lt_c v) {
    print_binary(v, " < ");
}
void codegen_c_vis::view(cmp_le_c v) {
    print_binary(v, " <= ");
}
void codegen_c_vis::view(cmp_gt_c v) {
    print_binary(v, " > ");
}
void codegen_c_vis::view(cmp_ge_c v) {
    print_binary(v, " >= ");
}
void codegen_c_vis::view(cmp_ne_c v) {
    print_binary(v, " != ");
}

void codegen_c_vis::view(logic_and_c v) {
    print_binary(v, " && ");
}
void codegen_c_vis::view(logic_or_c v) {
    print_binary(v, " || ");
}

void codegen_c_vis::view(logic_not_c v) {
    *os << '!';
    dispatch(v->in_);
}

void codegen_c_vis::view(select_c v) {
    if (v->l_->dtype_.lanes_ > 1) {
        *os << "sc_select(";
        dispatch(v->cond_);
        *os << ", ";
        dispatch(v->l_);
        *os << ", ";
        dispatch(v->r_);
    } else {
        *os << "(";
        dispatch(v->cond_);
        *os << "?";
        dispatch(v->l_);
        *os << ":";
        dispatch(v->r_);
    }
    *os << ")";
}

void codegen_c_vis::view(indexing_c v) {
    if (v->dtype_.lanes_ > 1) {
        if (v->mask_.defined()) {
            print_type(v->dtype_);
            *os << "::mask_load(&";
            dispatch(v->ptr_);
            *os << '[';
            assert(v->idx_.size() == 1);
            dispatch(v->idx_.front());
            *os << "], ";
            dispatch(v->mask_);
            *os << ')';
        } else {
            print_type(v->dtype_);
            *os << "::load(&";
            dispatch(v->ptr_);
            *os << '[';
            assert(v->idx_.size() == 1);
            dispatch(v->idx_.front());
            *os << ']' << ')';
        }
    } else {
        dispatch(v->ptr_);
        *os << '[';
        assert(v->idx_.size() == 1);
        dispatch(v->idx_.front());
        *os << ']';
    }
}

void codegen_c_vis::view(tensorptr_c v) {
    *os << '&';
    dispatch(v->base_);
}

void codegen_c_vis::view(call_c v) {
    func_t the_func = std::dynamic_pointer_cast<func_base>(v->func_);
    expr the_expr;
    if (!the_func) {
        the_expr = expr(std::dynamic_pointer_cast<expr_base>(v->func_));
        assert(the_expr.defined());
    }
    if (!v->para_attr_.empty()) {
        assert(v->args_.size() == 1);
        assert(v->para_attr_.size() == 1);
        assert(the_func);
        *os << "sc_parallel_call_cpu(" << the_func->name_ << ", ";
        dispatch(v->para_attr_[0].begin_);
        *os << ", ";
        dispatch(v->para_attr_[0].end_);
        *os << ", ";
        dispatch(v->para_attr_[0].step_);
        *os << ", ";
        *os << v->args_[0] << ')';
    } else {
        if (the_func) {
            if (!is_offline_
                    && default_external_symbol_resolve(the_func->name_)) {
                // if set gcc attributes, call func wrapper instead
                if (!the_func->attr_
                        || !the_func->attr_->get_or_else(
                                function_attrs::pure, false)) {
                    *os << the_func->name_ << "_fptr";
                } else {
                    *os << "__" << the_func->name_ << wrapper_postfix;
                }
            } else {
                *os << the_func->name_;
            }
        } else {
            auto func = the_expr->attr().get_or_else("prototype", func_t());
            COMPILE_ASSERT(func, "Call node expects an expr with prototype");
            *os << '(' << '(';
            print_type(func->ret_type_);
            *os << "(*)(";
            if (!func->params_.empty()) {
                for (size_t i = 0; i < func->params_.size() - 1; i++) {
                    print_type(func->params_[i]->dtype_);
                    *os << ',' << ' ';
                }
                print_type(func->params_.back()->dtype_);
            }
            *os << ')' << ')';
            dispatch(the_expr);
            *os << ')';
        }
        *os << '(';
        if (!v->args_.empty()) {
            for (size_t i = 0; i < v->args_.size() - 1; i++) {
                expr e = v->args_.at(i);
                dispatch(e);
                *os << ", ";
            }
            dispatch(v->args_.back());
        }
        *os << ")";
    }
}

void codegen_c_vis::view(tensor_c v) {
    *os << v->name_;
}

void codegen_c_vis::view(assign_c v) {
    if (v->var_->dtype_.lanes_ > 1 && v->var_.isa<indexing>()) {
        if (v->var_.static_as<indexing>()->mask_.defined()) {
            print_type(v->var_->dtype_);
            *os << "::mask_store(";
            *os << "&";
            indexing dst = v->var_.static_as<indexing>();
            dispatch(dst->ptr_);
            *os << '[';
            dispatch(dst->idx_.front());
            *os << "], ";
            dispatch(dst->mask_);
            *os << ", ";
            dispatch(v->value_);
            *os << ");";
        } else {
            print_type(v->var_->dtype_);
            *os << "::store(";
            dispatch(v->value_);
            *os << ", &";
            indexing dst = v->var_.static_as<indexing>();
            dispatch(dst->ptr_);
            *os << '[';
            dispatch(dst->idx_.front());
            *os << "]);";
        }
    } else {
        dispatch(v->var_);
        *os << " = ";
        dispatch(v->value_);
        *os << ';';
    }
}

void codegen_c_vis::view(stmts_c v) {
    *os << "{\n";
    indents++;
    for (auto &s : v->seq_) {
        print_indents(*os, indents);
        dispatch(s);
        *os << "\n";
    }
    indents--;
    print_indents(*os, indents);
    *os << "}";
}

void codegen_c_vis::view(if_else_c v) {
    *os << "if (";
    dispatch(v->condition_);
    *os << ") ";
    dispatch(v->then_case_);
    if (v->else_case_.defined()) {
        *os << " else ";
        dispatch(v->else_case_);
    }
}

void codegen_c_vis::view(evaluate_c v) {
    dispatch(v->value_);
    *os << ';';
}

void codegen_c_vis::view(returns_c v) {
    *os << "return ";
    if (v->value_.defined()) { dispatch(v->value_); }
    *os << ';';
}

void codegen_c_vis::view(define_c v) {
    if (v->linkage_ == linkage::static_local
            || v->linkage_ == linkage::private_global) {
        *os << "static ";
    }
    if (v->var_.isa<var>()) {
        auto thevar = v->var_.static_as<var>();
        if (v->var_->attr_
                && v->var_->attr_->has_key(attr_keys::module_global_offset)) {
            // if it is a global variable that is lowered to local
            print_type(thevar->dtype_);
            *os << '&' << ' ' << thevar->name_;
            auto &offset
                    = v->var_->attr_->get_any(attr_keys::module_global_offset);
            *os << " = *(";
            print_type(thevar->dtype_);
            if (auto ptr = offset.get_or_null<void *>()) {
                *os << "*)(" << *ptr << ')';
            } else {
                *os << "*)(__module_data + " << offset.get<size_t>() << ')';
            }
        } else {
            print_cpp_var_def(thevar);
            if (v->init_.defined()) {
                *os << " = ";
                dispatch(v->init_);
            }
        }
    } else if (v->var_.isa<tensor>()) {
        tensor t = v->var_.static_as<tensor>();
        // if it is a view of the rescheduled buffer/ local tensor on heap
        if (v->init_.defined()) {
            print_cpp_type(*os, t->elem_dtype_) << "* " << t->name_ << " = (";
            print_cpp_type(*os, t->elem_dtype_) << "*)";
            dispatch(v->init_);
            *os << ';';
            return;
        }
        bool use_heap = false;
        // if the tensor is defined as global, do not alloc it on heap
        if (v->linkage_ != linkage::public_global
                && v->linkage_ != linkage::private_global) {
            if (t->dims_.front().isa<constant>()) {
                auto dim = t->dims_.front().static_as<constant>();
                auto sz = get_const_as_int(dim);
                if (sz >= 256) { use_heap = true; }
            } else {
                use_heap = true;
            }
        }
        if (use_heap) {
            const char *buffertype = "_buffer<";
            if (t->attr_ && t->attr_->has_key("is_thread_buffer")
                    && t->attr_->get<bool>("is_thread_buffer")) {
                buffertype = "_thread_buffer<";
            }
            *os << buffertype;
            print_cpp_type(*os, t->elem_dtype_)
                    << "> _buf_" << t->name_ << "(__stream, ";
            dispatch(t->dims_.front());
            *os << "); ";
            print_cpp_type(*os, t->elem_dtype_)
                    << "* " << t->name_ << " = _buf_" << t->name_ << ".buf";
        } else {
            // explicitly align tensor with cache line size, except that tensor
            // is a scalar or bytes size < 64.
            bool need_align = false;
            // check condition.
            if (t->dims_.size() == 1
                    && get_const_as_int(t->dims_[0].checked_as<constant>())
                            == 1) {
                // it is a scalar
            } else {
                size_t shape = 1;
                for (auto &d : t->dims_) {
                    shape *= get_const_as_int(d.checked_as<constant>());
                }
                size_t dtsize
                        = utils::get_sizeof_etype(t->elem_dtype_.type_code_);
                // check bytes size
                if (shape * dtsize > 64) need_align = true;
            }
            // cache line alignment
            if (need_align) *os << "alignas(64) ";

            print_cpp_type(*os, t->elem_dtype_) << ' ' << t->name_ << '[';
            dispatch(t->dims_.front());
            *os << ']';
        }
    } else {
        assert(0 && "Bad var type");
    }
    *os << ';';
}

void codegen_c_vis::view(for_loop_c v) {
    if (v->kind_ == for_type::PARALLEL) {
        *os << "parallel((";
        dispatch(v->iter_end_);
        *os << " - ";
        dispatch(v->iter_begin_);
        *os << ") / ";
        dispatch(v->step_);
        *os << ", [&](const int __ithr, const int __nthr) {\n";
        indents++;
        print_indents(*os, indents);
        auto itrv = v->var_.checked_as<var>();
        print_cpp_var_def(itrv);
        *os << " = ";
        dispatch(v->iter_begin_);
        *os << " + __ithr * ";
        dispatch(v->step_);
        *os << ";\n";
        print_indents(*os, indents);
        dispatch(v->body_);
        indents--;
        *os << "\n";
        print_indents(*os, indents);
        *os << "});";
    } else {
        *os << "for (";
        auto itrv = v->var_.checked_as<var>();
        print_cpp_var_def(itrv);
        *os << " = ";
        dispatch(v->iter_begin_);
        *os << "; " << itrv->name_ << " < ";
        dispatch(v->iter_end_);
        *os << "; " << itrv->name_ << " += ";
        dispatch(v->step_);
        *os << ") ";
        dispatch(v->body_);
    }
}

static void prepare_include(std::ostream *source) {
    *source << R"(#include <runtime/kernel_include/cpu_include.hpp>

)";
}

static bool is_main_func_wrapper(const func_c &f) {
    return f->attr_ && f->attr_->get_or_else(function_attrs::is_main, false)
            && utils::string_endswith(f->name_, wrapper_postfix);
}

static bool is_func_static(const func_c &f, bool is_offline) {
    auto attr = f->attr_.get();
    if (is_offline) {
        // if is offline mode, hide all symbols except the main entry
        if (is_main_func_wrapper(f)) { return false; }
        if (f->name_ == "__sc_init__" || f->name_ == "memset") { return false; }
        return !default_external_symbol_resolve(f->name_);
    }
    return attr && attr->get_or_else(function_attrs::private_, false);
}

void write_cpp_prototype(
        std::ostream *source_, const func_c &f, bool is_offline) {
    codegen_c_vis vis(source_, true, is_func_static(f, is_offline));
    vis.is_offline_ = is_offline;
    vis.dispatch(f);
    *source_ << '\n';
}

void write_cpp_generic_wrapper(
        std::ostream *source_, const func_c &f, bool is_parallel) {
    *source_ << "extern \"C\" void ";
    if (is_parallel) {
        *source_ << get_closure_wrapper_name(f->name_)
                 << "(int64_t i, generic_val* args) {\n  ";
    } else {
        *source_ << f->name_ << "_0wrapper(generic_val* args) {\n  ";
    }
    *source_ << f->name_ << '(';
    std::vector<expr>::const_iterator itr;
    if (is_parallel) {
        *source_ << "i";
        if (f->params_.size() > 1) { *source_ << ", "; }
        assert(!f->params_.empty());
        // skip the first parameter, as it is given by "i"
        itr = f->params_.begin() + 1;
    } else {
        itr = f->params_.begin();
    }

    int idx = 0;
    for (; itr != f->params_.end(); ++itr) {
        auto &arg = *itr;
        if (arg.isa<var>()) {
            auto v = arg.static_as<var>();
            *source_ << "args[" << idx << "].v_";
            if (v->dtype_.is_pointer()) {
                *source_ << "ptr";
            } else {
                print_cpp_type(*source_, v->dtype_);
            }
        } else {
            auto v = arg.checked_as<tensor>();
            *source_ << '(';
            print_cpp_type(*source_, v->elem_dtype_)
                    << "*)(args[" << idx << "].v_ptr)";
        }
        idx++;
        if (itr + 1 != f->params_.end()) { *source_ << ", "; }
    }
    *source_ << ");\n}\n\n";
}

static func_c do_generate_c(func_c f, std::ostream &source,
        const std::vector<define> &globals, bool gen_wrapper, bool is_static,
        bool is_offline) {
    codegen_c_vis vis(&source, false, is_static);
    vis.is_offline_ = is_offline;
    vis.dispatch(f);
    source << '\n' << '\n';
    return f;
}

void c_generator_pass_t::operator()(func_t f) {
    f = pre_passes_(ir_module_t::from_entry_func(context_, f))
                ->get_entry_func();
    do_generate_c(f, source_, {}, gen_wrapper_, false, false);
}

c_generator_pass_t::c_generator_pass_t(std::ostream &source,
        const context_ptr &ctx, bool gen_wrapper,
        c_generator_optional_out_t *optional_out)
    : source_(source)
    , context_(ctx)
    , gen_wrapper_(gen_wrapper)
    , pre_passes_ {get_default_precodegen_passes(ctx, gen_wrapper)}
    , optional_out_(optional_out) {
    prepare_include(&source_);
    if (optional_out_) { prepare_include(optional_out_->offline_source_); }
}

const_ir_module_ptr preprocess_module_and_make_decl(
        const const_ir_module_ptr &mod, module_pass_t &pre_passes,
        std::ostream &source, c_generator_optional_out_t *optout) {
    auto mod_cpy = run_precodegen_passes(pre_passes, mod);
    auto timer
            = SC_SCOPED_TIMER_INFO("pass.time.c_generator_pass.preprocess", "");
    // stage 1, find and print all function prototypes
    std::vector<func_t> dep;
    std::unordered_set<func_t> depset;
    func_dependency_finder_t finder(dep);
    for (auto &f : mod_cpy->get_contents()) {
        finder(f, depset);
    }
    for (auto &v : mod_cpy->get_module_vars()) {
        finder(v, depset);
    }
    for (auto &f : dep) {
        if (!f->attr_ || !f->attr_->has_key("device_func")) {
            write_cpp_prototype(&source, f, false);
        }
    }
    source << '\n';
    source << '\n';

    if (optout) {
        bool managed_thread_pool = mod_cpy->attr_.get<bool>(
                ir_module_t::attr_key_t::MANAGED_THREAD_POOL);
        if (managed_thread_pool) {
            (*optout->offline_source_)
                    << R"(#include <runtime/managed_thread_pool.hpp>
)";
        }

        (*optout->offline_source_) << R"(#include <omp.h>
#define sc_get_thread_id omp_get_thread_num
#define sc_parallel_call_cpu_with_env sc_parallel_call_cpu_with_env_impl
)";
        for (auto &f : dep) {
            if (!f->attr_ || !f->attr_->has_key("device_func")) {
                write_cpp_prototype(optout->offline_source_, f, true);
            }
        }
        *(optout->offline_source_) << '\n' << '\n';
    }
    return mod_cpy;
}

static void generate_dumped_source(const const_ir_module_ptr &mod,
        c_generator_optional_out_t *optional_out, bool gen_wrapper) {
    std::string module_name = "main_entry";
    for (auto &f : mod->get_contents()) {
        if (f->attr_ && f->attr_->get_or_else(function_attrs::is_main, false)
                && !utils::string_endswith(f->name_, wrapper_postfix)) {
            module_name = f->name_;
            break;
        }
    }

    bool managed_thread_pool = mod->attr_.get<bool>(
            ir_module_t::attr_key_t::MANAGED_THREAD_POOL);
    for (auto &f : mod->get_contents()) {
        if (f->name_ == "__sc_init__") {
            f->attr()["temp.replace_func_name"] = "sc_init_" + module_name;
            f->attr()["comments"]
                    = std::vector<std::string> {"Initialize the " + module_name,
                            "@param __stream the stream pointer, usually "
                            "get_default_stream()",
                            "@param __module_data the module global data"};
            break;
        }
    }
    auto &header_src = *optional_out->header_source_;
    auto &data_src = *optional_out->data_source_;
    auto &offline_src = *optional_out->offline_source_;

    auto &mod_data = mod->attr_.get<std::shared_ptr<statics_table_t>>(
            ir_module_t::attr_key_t::MODULE_DATA_BUFFERS);

    header_src << R"(#include <stdint.h>
#include <runtime/generic_val.hpp>
using generic_val = dnnl::impl::graph::gc::generic_val;

extern uint8_t )"
               << module_name << "_data[" << mod_data->data_.size_ << "];\n\n";
    const char *skip_func_name = "__sc_init___0wrapper";
    for (auto &f : mod->get_contents()) {
        if (f->name_ == skip_func_name) { continue; }
        bool is_static_func = is_func_static(f, true);
        if (!is_static_func) { write_cpp_prototype(&header_src, f, true); }

        // change the name of the generated func to insert managed thread pool
        // code
        if (managed_thread_pool && is_main_func_wrapper(f)) {
            f->attr()["temp.replace_func_name"] = f->name_ + "_impl";
        }

        do_generate_c(f, offline_src, mod->get_module_vars(), gen_wrapper,
                is_static_func, true);
    }
    if (managed_thread_pool) {
        offline_src
                << "extern \"C\" void " << module_name
                << R"(_0wrapper(void* __stream, int8_t* __restrict__ __module_data, generic_val* __restrict__ args) noexcept{
  gc::runtime::thread_manager::cur_mgr.run_main_function((gc::runtime::thread_manager::main_func_t))"
                << module_name
                << R"(_0wrapper_impl, (gc::runtime::stream_t *)__stream, __module_data, args);
})";
    }

    // generate module data

    memset((char *)mod_data->data_.data_ + mod_data->initialized_size_, 0,
            mod_data->data_.size_ - mod_data->initialized_size_);
    data_src << R"(#include <stdint.h>

alignas(64) uint8_t )"
             << module_name << "_data[" << mod_data->data_.size_ << "] = {";
    uint8_t *buffer = (uint8_t *)mod_data->data_.data_;
    for (size_t i = 0; i < mod_data->initialized_size_; i++) {
        data_src << (uint32_t)buffer[i] << ',';
    }
    data_src << "};\n";
}

const_ir_module_ptr c_generator_pass_t::operator()(const_ir_module_ptr mod) {
    // TODO(xxx): cfake_jit is created with default ctx, which can't pass
    // below assertion. assert(mod->ctx_ == context_);
    mod = preprocess_module_and_make_decl(
            mod, pre_passes_, source_, optional_out_);
    auto timer = SC_SCOPED_TIMER_INFO("pass.time.c_generator_pass.codegen", "");
    for (auto &f : mod->get_contents()) {
        do_generate_c(f, source_, mod->get_module_vars(), gen_wrapper_,
                is_func_static(f, false), false);
    }
    if (optional_out_) {
        generate_dumped_source(mod, optional_out_, gen_wrapper_);
    }
    return mod;
}

c_generator_pass_t create_c_generator(std::ostream &os, const context_ptr &ctx,
        bool gen_wrapper, c_generator_optional_out_t *optional_out) {
    return c_generator_pass_t(os, ctx, gen_wrapper, optional_out);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

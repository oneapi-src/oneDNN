/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "shared_include.hpp"
#include "util/fp16.hpp"

using namespace llvm;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

void codegen_llvm_vis_t::generate_bin_op(
        const expr_c &v, const expr_c &l, const expr_c &r) {
    auto cate = get_etype_category_nothrow(l->dtype_.type_code_);
    auto lhs = generate_expr(l);
    auto rhs = generate_expr(r);
#define HANDLE_BIN_OP2(scname, intname, fpname, case_other) \
    case sc_expr_type::scname: \
        switch (cate) { \
            case CATE_INT: \
            case CATE_UINT: \
                current_val_ = builder_.Create##intname(lhs, rhs); \
                break; \
            case CATE_FLOAT: \
                current_val_ = builder_.Create##fpname(lhs, rhs); \
                break; \
            case CATE_OTHER: { \
                case_other \
            } \
            default: \
                COMPILE_ASSERT(false, \
                        "Cannot generate binary op for this type: " << v); \
        } \
        break
#define HANDLE_POINTER_CMP(llvm_name) \
    if (l->dtype_.type_code_ == sc_data_etype::POINTER \
            && r->dtype_.type_code_ == sc_data_etype::POINTER) { \
        current_val_ = builder_.Create##llvm_name(lhs, rhs); \
        break; \
    }
#define HANDLE_POINTER_EQ HANDLE_POINTER_CMP(ICmpEQ)
#define HANDLE_POINTER_NE HANDLE_POINTER_CMP(ICmpNE)
#define HANDLE_BIN_OP(scname, intname) \
    HANDLE_BIN_OP2(scname, intname, F##intname, )
#define HANDLE_BIN_SIGNED_OP2(scname, intname1, intname2, fpname) \
    case sc_expr_type::scname: \
        switch (cate) { \
            case CATE_INT: \
                current_val_ \
                        = builder_.Create##intname1##S##intname2(lhs, rhs); \
                break; \
            case CATE_UINT: \
                current_val_ \
                        = builder_.Create##intname1##U##intname2(lhs, rhs); \
                break; \
            case CATE_FLOAT: \
                current_val_ = builder_.Create##fpname(lhs, rhs); \
                break; \
            default: \
                COMPILE_ASSERT(false, \
                        "Cannot generate binary op for this type: " << v); \
        } \
        break
#define HANDLE_BIN_SIGNED_OP(scname, intname) \
    HANDLE_BIN_SIGNED_OP2(scname, , intname, F##intname)
#define HANDLE_CMP_SIGNED_OP(scname, llvmname) \
    HANDLE_BIN_SIGNED_OP2(scname, ICmp, llvmname, FCmpO##llvmname)
    switch (v->node_type_) {
        HANDLE_BIN_OP(add, Add);
        HANDLE_BIN_OP(sub, Sub);
        HANDLE_BIN_OP(mul, Mul);
        HANDLE_BIN_SIGNED_OP(div, Div);
        HANDLE_BIN_SIGNED_OP(mod, Rem);
        HANDLE_BIN_OP2(cmp_eq, ICmpEQ, FCmpOEQ, HANDLE_POINTER_EQ);
        HANDLE_BIN_OP2(cmp_ne, ICmpNE, FCmpONE, HANDLE_POINTER_NE);
        HANDLE_CMP_SIGNED_OP(cmp_lt, LT);
        HANDLE_CMP_SIGNED_OP(cmp_le, LE);
        HANDLE_CMP_SIGNED_OP(cmp_gt, GT);
        HANDLE_CMP_SIGNED_OP(cmp_ge, GE);
        case sc_expr_type::logic_and:
            current_val_ = builder_.CreateAnd(lhs, rhs);
            break;
        case sc_expr_type::logic_or:
            current_val_ = builder_.CreateOr(lhs, rhs);
            break;
        default: assert(0);
    }
}

void codegen_llvm_vis_t::view(binary_c v) {
    generate_bin_op(v, v->l_, v->r_);
}
void codegen_llvm_vis_t::view(cmp_c v) {
    generate_bin_op(v, v->l_, v->r_);
}

void codegen_llvm_vis_t::view(logic_c v) {
    generate_bin_op(v, v->l_, v->r_);
}

void codegen_llvm_vis_t::view(constant_c v) {
    std::vector<Constant *> vals;
    vals.reserve(v->value_.size());
    auto cate = get_etype_category_nothrow(v->dtype_.type_code_);
    bool is_bf16 = v->dtype_.type_code_ == sc_data_etype::BF16;
    bool is_f16 = v->dtype_.type_code_ == sc_data_etype::F16;
    sc_data_type_t base_type = v->dtype_;
    base_type.lanes_ = 1;
    auto llvm_base_type = get_type(base_type);
    switch (cate) {
        case CATE_FLOAT: {
            for (auto &val : v->value_) {
                if (is_bf16) {
                    uint64_t val_u64 = bf16_t(val.f32).storage_;
                    vals.push_back(
                            ConstantInt::get(llvm_base_type, val_u64, false));
                } else if (is_f16) {
                    uint16_t val_u16 = fp16_t(val.f32).storage_;
                    vals.push_back(ConstantFP::get(llvm_base_type,
                            APFloat(APFloat::IEEEhalf(), APInt(16, val_u16))));
                } else {
                    vals.push_back(
                            ConstantFP::get(llvm_base_type, APFloat(val.f32)));
                }
            }
        } break;
        case CATE_UINT:
        case CATE_INT: {
            bool is_signed = cate == CATE_INT;
            for (auto &val : v->value_) {
                vals.push_back(
                        ConstantInt::get(llvm_base_type, val.u64, is_signed));
            }
        } break;
        default:
            COMPILE_ASSERT(v->dtype_.is_pointer(),
                    "Unexpected type for LLVM. Expecting pointer.");
            for (auto &val : v->value_) {
                vals.push_back(ConstantExpr::getIntToPtr(
                        builder_.getInt64(val.u64), llvm_base_type));
            }
            break;
    }
    if (vals.size() != v->dtype_.lanes_) {
        COMPILE_ASSERT(
                vals.size() == 1, "Bad constant node. Expecting 1 value");
        // broadcast value
        current_val_
                = builder_.CreateVectorSplat(v->dtype_.lanes_, vals.front());
    } else {
        if (vals.size() != 1) {
            current_val_ = ConstantVector::get(vals);
        } else {
            current_val_ = vals.front();
        }
    }
}

void codegen_llvm_vis_t::view(var_c v) {
    auto ret = get_defined_var_ptr(v);
    bool is_global_base = utils::string_startswith(v->name_, "__module_data");
    bool is_special_params = (is_global_base || v->name_ == "__stream");
    if (is_lvalue_mode_) {
        assert(!is_special_params);
        is_lvalue_mode_ = false;
        current_val_ = ret;
    } else {
        if (is_special_params) {
            current_val_ = ret;
        } else {
            current_val_ = builder_.CreateLoad(
                    get_type(v->dtype_), ret, v->name_ + "_v");
        }
    }
}

llvm::Type *codegen_llvm_vis_t::get_llvm_bf16_native_type(unsigned lanes) {
#if SC_LLVM_BACKEND > 15
    // llvm 16's bf16 casting intrinsic uses bf16 instead of i16
    return VectorType::get(builder_.getBFloatTy(), lanes, false);
#else
    return get_type(sc_data_type_t::bf16(lanes));
#endif
}

void codegen_llvm_vis_t::view(cast_c v) {
    auto cate_out = get_etype_category_nothrow(v->dtype_.type_code_);
    auto cate_in = get_etype_category_nothrow(v->in_->dtype_.type_code_);
    auto in_v = generate_expr(v->in_);
    auto outtype = get_type(v->dtype_);
    auto check_cate = [&v]() {
        COMPILE_ASSERT(
                v->dtype_ == datatypes::generic, "Unexpected outtype " << v);
    };
    if (v->in_->dtype_.is_etype(sc_data_etype::F32)
            && v->dtype_.is_etype(sc_data_etype::BF16)) {
#if SC_LLVM_BACKEND > 10
        switch (v->in_->dtype_.lanes_) {
            case 1: {
                Value *vec = builder_.CreateVectorSplat(4, in_v);
                vec = builder_.CreateIntrinsic(
                        Intrinsic::x86_avx512bf16_mask_cvtneps2bf16_128, {},
                        {vec, UndefValue::get(get_llvm_bf16_native_type(8)),
                                /*mask*/
                                builder_.CreateVectorSplat(
                                        4, builder_.getInt1(true))});
                current_val_ = builder_.CreateExtractElement(vec, UINT64_C(0));
            } break;
            case 4:
                current_val_ = builder_.CreateIntrinsic(
                        Intrinsic::x86_avx512bf16_mask_cvtneps2bf16_128, {},
                        {in_v, UndefValue::get(get_llvm_bf16_native_type(8)),
                                /*mask*/
                                builder_.CreateVectorSplat(
                                        4, builder_.getInt1(true))});
#if SC_LLVM_BACKEND == 11
                current_val_ = builder_.CreateShuffleVector(current_val_,
                        current_val_, ArrayRef<int>({0, 1, 2, 3}));
#else
                current_val_ = builder_.CreateShuffleVector(
                        current_val_, {0, 1, 2, 3});
#endif
                break;
            case 8:
                current_val_ = builder_.CreateIntrinsic(
                        Intrinsic::x86_avx512bf16_cvtneps2bf16_256, {}, {in_v});
                break;
            case 16:
                current_val_ = builder_.CreateIntrinsic(
                        Intrinsic::x86_avx512bf16_cvtneps2bf16_512, {}, {in_v});
                break;
            default:
                std::stringstream ss;
                ss << "Unsupport cast lanes " << v->in_->dtype_.lanes_;
                throw std::runtime_error(ss.str());
        }
#if SC_LLVM_BACKEND > 15
        // llvm 16's bf16 casting intrinsic returns bf16 instead of i16
        current_val_ = builder_.CreateBitCast(current_val_, outtype);
#endif
        return;
#else
        throw std::runtime_error("Current version of LLVM cannot handle bf16");
#endif
    }
    switch (cate_in) {
        case CATE_FLOAT: {
            switch (cate_out) {
                case CATE_FLOAT:
                    current_val_ = builder_.CreateFPCast(in_v, outtype);
                    break;
                case CATE_INT:
                    current_val_ = builder_.CreateFPToSI(in_v, outtype);
                    break;
                case CATE_UINT:
                    current_val_ = builder_.CreateFPToUI(in_v, outtype);
                    break;
                case CATE_OTHER: {
                    check_cate();
                    auto bits
                            = utils::get_sizeof_etype(v->in_->dtype_.type_code_)
                            * 8;
                    auto ret = builder_.CreateBitCast(
                            in_v, IntegerType::get(context_, bits));
                    current_val_ = builder_.CreateZExtOrBitCast(
                            ret, builder_.getInt64Ty());
                } break;
            }
        } break;
        case CATE_INT: {
            switch (cate_out) {
                case CATE_FLOAT:
                    current_val_ = builder_.CreateSIToFP(in_v, outtype);
                    break;
                case CATE_INT:
                case CATE_UINT:
                    current_val_ = builder_.CreateSExtOrTrunc(in_v, outtype);
                    break;
                case CATE_OTHER: {
                    check_cate();
                    current_val_ = builder_.CreateZExtOrBitCast(
                            in_v, builder_.getInt64Ty());
                } break;
            }
        } break;
        case CATE_UINT: {
            switch (cate_out) {
                case CATE_FLOAT:
                    current_val_ = builder_.CreateUIToFP(in_v, outtype);
                    break;
                case CATE_INT:
                case CATE_UINT:
                    current_val_ = builder_.CreateZExtOrTrunc(in_v, outtype);
                    break;
                case CATE_OTHER: {
                    check_cate();
                    current_val_ = builder_.CreateZExtOrBitCast(
                            in_v, builder_.getInt64Ty());
                } break;
            }
        } break;
        case CATE_OTHER:
            if (v->in_->dtype_ == datatypes::generic) {
                auto bits = module_->getDataLayout().getTypeAllocSizeInBits(
                        outtype);
                auto ret = builder_.CreateTruncOrBitCast(
                        in_v, IntegerType::get(context_, bits));
                switch (cate_out) {
                    case CATE_OTHER:
                        COMPILE_ASSERT(v->dtype_.is_pointer(),
                                "Unexpected out type " << v);
                        current_val_ = builder_.CreateIntToPtr(ret, outtype);
                        break;
                    case CATE_FLOAT:
                        current_val_ = builder_.CreateBitCast(ret, outtype);
                        break;
                    case CATE_INT:
                    case CATE_UINT: current_val_ = ret; break;
                }
            } else {
                COMPILE_ASSERT(v->in_->dtype_.is_pointer(),
                        "Unexpected in type " << v);
                if (v->dtype_.is_pointer()) {
                    // pointer to pointer
                    current_val_ = builder_.CreatePointerCast(in_v, outtype);
                } else {
                    // pointer to generic val
                    check_cate();
                    current_val_ = builder_.CreatePtrToInt(
                            in_v, builder_.getInt64Ty());
                }
            }
            break;
    }
}

void codegen_llvm_vis_t::view(logic_not_c v) {
    current_val_ = builder_.CreateNot(generate_expr(v->in_));
}
void codegen_llvm_vis_t::view(select_c v) {
    auto l = generate_expr(v->l_);
    auto r = generate_expr(v->r_);
    auto cond = convert_mask(v->cond_, v->l_->dtype_.lanes_ == 4);
    current_val_ = builder_.CreateSelect(cond, l, r);
}

void codegen_llvm_vis_t::view(indexing_c v) {
    bool is_lvalue_mode = is_lvalue_mode_;
    is_lvalue_mode_ = false;
    COMPILE_ASSERT(v->idx_.size() == 1, "Expecting 1D array: " << v);
    auto base = generate_expr(v->ptr_);
    assert(v->ptr_->dtype_.is_pointer());
    auto element_type = get_type(v->ptr_->dtype_.get_pointer_element());
    auto ptr = builder_.CreateGEP(
            element_type, base, generate_expr(v->idx_.front()));
    auto target_type = get_type(v->dtype_);
    if (target_type != element_type) {
        // allow pointer to pointer
        assert(v->dtype_ == datatypes::pointer
                || llvm::cast<VectorType>(*target_type).getElementType()
                        == element_type);
        ptr = builder_.CreatePointerCast(ptr, target_type->getPointerTo());
    }
    if (is_lvalue_mode) {
        current_val_ = ptr;
    } else {
        bool is_volatile = (v->ptr_->attr_
                && v->ptr_->attr_->get_or_else("volatile", false));
        if (v->mask_.defined()) {
            Value *mask;
            auto bit_len = utils::get_sizeof_type(v->mask_->dtype_) * 8;
            if (v->dtype_.lanes_ != bit_len) {
                COMPILE_ASSERT(v->dtype_.lanes_ == 4,
                        "Currently only 8bit -> 4bit is supported, but get "
                                << v->dtype_
                                << " lanes = " << v->dtype_.lanes_);
                mask = convert_mask(v->mask_, true);
            } else {
                mask = convert_mask(v->mask_);
            }
            auto znode = make_expr<constant_node>(0UL, v->dtype_);
            auto zero = generate_expr(znode);
#if SC_LLVM_BACKEND > 12
            current_val_
                    = set_alias(builder_.CreateMaskedLoad(get_type(v->dtype_),
                                        ptr, SC_LLVM_ALIGN(1), mask, zero),
                            v->ptr_);
#elif SC_LLVM_BACKEND >= 11
            current_val_ = set_alias(builder_.CreateMaskedLoad(
                                             ptr, SC_LLVM_ALIGN(1), mask, zero),
                    v->ptr_);
#else
            current_val_ = set_alias(
                    builder_.CreateMaskedLoad(ptr, 1, mask, zero), v->ptr_);
#endif
        } else {
            if (v->dtype_.lanes_ > 1) {
                current_val_ = set_alias(
                        builder_.CreateAlignedLoad(get_type(v->dtype_), ptr,
                                SC_LLVM_ALIGN(1), is_volatile),
                        v->ptr_);
            } else {
                current_val_
                        = set_alias(builder_.CreateLoad(get_type(v->dtype_),
                                            ptr, is_volatile),
                                v->ptr_);
            }
        }
    }
}
void codegen_llvm_vis_t::view(tensorptr_c v) {
    is_lvalue_mode_ = true;
    current_val_ = generate_expr(v->base_);
}

void codegen_llvm_vis_t::view(func_addr_c v) {
    current_val_ = builder_.CreatePointerCast(
            get_or_create_func(v->func_), builder_.getInt8PtrTy());
}
void codegen_llvm_vis_t::view(call_c v) {
    std::vector<Value *> args;
    auto the_func = std::dynamic_pointer_cast<func_base>(v->func_);
    Value *ll_func;
    FunctionType *ft = nullptr;
    if (the_func) {
        auto F = get_or_create_func(the_func);
        ll_func = F;
        ft = F->getFunctionType();
    } else {
        auto the_expr = std::dynamic_pointer_cast<expr_base>(v->func_);
        assert(the_expr);
        auto proto_func = the_expr->attr().get_or_else("prototype", func_t());
        COMPILE_ASSERT(proto_func, "Call node expects an expr with prototype");
        ft = create_func_type(proto_func);
        ll_func = generate_expr(expr_c(the_expr));
        ll_func = builder_.CreatePointerCast(ll_func, ft->getPointerTo());
    }
    for (size_t i = 0; i < v->args_.size(); i++) {
        auto &val = v->args_[i];
        auto ll_value = generate_expr(val);
        auto target_type = *(ft->param_begin() + i);
        if (ll_value->getType() != target_type) {
            COMPILE_ASSERT(target_type == builder_.getInt8PtrTy(),
                    "LLVM can only handle autocast to pointer");
            ll_value = builder_.CreatePointerCast(ll_value, target_type);
        }
        args.push_back(ll_value);
    }
    current_val_ = builder_.CreateCall(ft, ll_func, args);
}
void codegen_llvm_vis_t::view(tensor_c v) {
    current_val_ = get_defined_var_ptr(v);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

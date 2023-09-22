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

using namespace llvm;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

Value *codegen_llvm_vis_t::call_unary_llvm_intrin(const intrin_call_c &v,
        type_category cate, Intrinsic::ID id, bool must_fp) {
    assert(v->args_.size() == 1);
    auto inval = generate_expr(v->args_[0]);
    if (must_fp) {
        COMPILE_ASSERT(cate == CATE_FLOAT, "Bad type. Expecting float: " << v);
    }
    return builder_.CreateUnaryIntrinsic(id, inval);
}

Value *codegen_llvm_vis_t::call_binary_llvm_intrin(const intrin_call_c &v,
        type_category cate, Intrinsic::ID id, bool must_fp) {
    assert(v->args_.size() == 2);
    auto inval1 = generate_expr(v->args_[0]);
    auto inval2 = generate_expr(v->args_[1]);
    if (must_fp) {
        COMPILE_ASSERT(cate == CATE_FLOAT, "Bad type. Expecting float: " << v);
    }
    return builder_.CreateBinaryIntrinsic(id, inval1, inval2);
}

Value *codegen_llvm_vis_t::call_binary_llvm_normal(
        const intrin_call_c &v, llvm_binary_func op) {
    assert(v->args_.size() == 2);
    auto inval1 = generate_expr(v->args_[0]);
    auto inval2 = generate_expr(v->args_[1]);
    return (builder_.*op)(inval1, inval2, "");
}

Value *codegen_llvm_vis_t::make_int_min_max(
        const intrin_call_c &v, bool ismin, type_category cate) {
    assert(v->args_.size() == 2);
    auto v1 = generate_expr(v->args_[0]);
    auto v2 = generate_expr(v->args_[1]);
    return make_int_min_max(v1, v2, ismin, cate);
}

Value *codegen_llvm_vis_t::make_int_min_max(
        Value *v1, Value *v2, bool ismin, type_category cate) {
    // fix-me: use smax/smin for newer LLVM
    llvm::Value *(llvm::IRBuilder<>::*ptr)(
            llvm::Value * LHS, llvm::Value * RHS, const llvm::Twine &Name);
    if (ismin) {
        if (cate == CATE_INT) {
            ptr = &IRBuilder<>::CreateICmpSLE;
        } else {
            ptr = &IRBuilder<>::CreateICmpULE;
        }
    } else {
        if (cate == CATE_INT) {
            ptr = &IRBuilder<>::CreateICmpSGE;
        } else {
            ptr = &IRBuilder<>::CreateICmpUGE;
        }
    }
    return builder_.CreateSelect((builder_.*ptr)(v1, v2, ""), v1, v2);
}

Value *codegen_llvm_vis_t::do_lower_saturated_cast(const intrin_call_c &v) {
    COMPILE_ASSERT(ctx_->machine_.cpu_flags_.fAVX512F,
            "lowered saturated_cast needs AVX512F");
    assert(v->args_.size() == 1);
    auto inval1 = generate_expr(v->args_[0]);
    auto intype = v->args_[0]->dtype_;
    auto out_llvm_ty = get_type(v->dtype_);
    auto ths = this;
    // the fast path for AVX512
    auto pmovus_db_512 = [ths, out_llvm_ty](Value *v, bool issigned) {
        Intrinsic::ID id = issigned ? Intrinsic::x86_avx512_mask_pmovs_db_512
                                    : Intrinsic::x86_avx512_mask_pmovus_db_512;
        return ths->builder_.CreateIntrinsic(id, {},
                {v, UndefValue::get(out_llvm_ty),
                        ths->builder_.getInt16(0xffff)});
    };
    if (v->dtype_ == sc_data_type_t::s8(16)) {
        if (intype == sc_data_type_t::s32(16)) {
            return pmovus_db_512(inval1, true);
        }
    } else if (v->dtype_ == sc_data_type_t::u8(16)) {
        if (intype == sc_data_type_t::s32(16)) {
            return pmovus_db_512(inval1, false);
        }
    }
    COMPILE_ASSERT(false,
            "lowered saturated_cast cannot handle: "
                    << v << '(' << intype << "->" << v->dtype_ << ')');
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

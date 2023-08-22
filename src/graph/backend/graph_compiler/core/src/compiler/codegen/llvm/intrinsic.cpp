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
#include "util/utils.hpp"

using namespace llvm;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Implementing LLVM-x86 intrinsics
 * 1. first find the GCC/Clang built-in intrinsic name in
 * https://github.com/llvm/llvm-project/blob/main/clang/lib/Headers
 *    e.g. Goto definition of _mm512_cvtusepi32_epi8, you will get
 *    __builtin_ia32_pmovusdb512_mask
 * 2. Find the built-in function name in
 * https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsX86.td
 * 3. Now we have the intrinsic name in LLVM :
 *    x86_avx512_mask_pmovus_db_512
 * */
void codegen_llvm_vis_t::view(intrin_call_c v) {
    auto cate = get_etype_category_nothrow(v->dtype_.type_code_);

    switch (v->type_) {
        case intrin_type::reinterpret: {
            assert(v->args_.size() == 1);
            auto inval = generate_expr(v->args_[0]);
            auto outty = get_type(v->dtype_);
            if (outty->isPointerTy()) {
                auto src_cate = get_type_category_nothrow(v->args_[0]->dtype_);
                bool is_src_int = src_cate == CATE_INT || src_cate == CATE_UINT;
                if (is_src_int) {
                    current_val_ = builder_.CreateIntToPtr(inval, outty);
                } else {
                    current_val_ = builder_.CreatePointerCast(inval, outty);
                }
            } else if (inval->getType()->isPointerTy()) {
                auto dst_cate = get_type_category_nothrow(v->dtype_);
                bool is_dest_int
                        = dst_cate == CATE_INT || dst_cate == CATE_UINT;
                COMPILE_ASSERT(is_dest_int,
                        "Expecting pointer to int for reinterpret");
                current_val_ = builder_.CreatePtrToInt(inval, outty);
            } else {
                current_val_ = builder_.CreateBitCast(inval, outty);
            }
        } break;
        case intrin_type::abs: {
            assert(v->args_.size() == 1);
            auto inval = generate_expr(v->args_[0]);

            std::string intrin_name;
            llvm::raw_string_ostream os(intrin_name);
            switch (cate) {
                case CATE_FLOAT:
                    current_val_ = builder_.CreateUnaryIntrinsic(
                            Intrinsic::fabs, inval);
                    break;
                case CATE_INT: {
                    auto znode = make_expr<constant_node>(
                            0UL, v->args_[0]->dtype_);
                    auto zero = generate_expr(znode);
                    auto sign = builder_.CreateICmpSGT(inval, zero);
                    auto neg = builder_.CreateSub(zero, inval);
                    current_val_ = builder_.CreateSelect(sign, inval, neg);
                } break;
                default: assert(0); break;
            }
        } break;
        case intrin_type::sqrt: {
            current_val_
                    = call_unary_llvm_intrin(v, cate, Intrinsic::sqrt, true);
        } break;
        case intrin_type::rsqrt: {
            // todo: The precision of AVX2 intrinsic is ~0.000366, not meet
            // f32 precision judgements. Use fast-math pass to enable
            // avx2/sse intrinsic with low precision(bf16/int8)
            if (ctx_->machine_.cpu_flags_.fAVX512F && v->dtype_.lanes_ > 1) {
                auto inval = generate_expr(v->args_[0]);
                switch (v->dtype_.lanes_) {
                    case 4:
                        current_val_ = builder_.CreateIntrinsic(
                                Intrinsic::x86_avx512_rsqrt14_ps_128, {},
                                {inval, inval, builder_.getInt8(0xff)});

                        break;
                    case 8:
                        current_val_ = builder_.CreateIntrinsic(
                                Intrinsic::x86_avx512_rsqrt14_ps_256, {},
                                {inval, inval, builder_.getInt8(0xff)});

                        break;
                    case 16:
                        COMPILE_ASSERT(ctx_->machine_.cpu_flags_.fAVX512F,
                                "rsqrt of 16 floats needs AVX512");
                        current_val_ = builder_.CreateIntrinsic(
                                Intrinsic::x86_avx512_rsqrt14_ps_512, {},
                                {inval, inval, builder_.getInt16(0xffff)});
                        break;
                    default:
                        COMPILE_ASSERT(false,
                                "LLVM backend has not yet support "
                                "rsqrt with lanes = "
                                        << v->dtype_.lanes_);
                        break;
                }
            } else {
                current_val_ = call_unary_llvm_intrin(
                        v, cate, Intrinsic::sqrt, true);
                Value *ones
                        = ConstantFP::get(builder_.getFloatTy(), APFloat(1.0f));
                if (v->dtype_.lanes_ > 1) {
                    ones = builder_.CreateVectorSplat(v->dtype_.lanes_, ones);
                }
                current_val_ = builder_.CreateFDiv(ones, current_val_);
            }
        } break;
        case intrin_type::int_and: {
            current_val_
                    = call_binary_llvm_normal(v, &llvm::IRBuilder<>::CreateAnd);
        } break;
        case intrin_type::int_or: {
            current_val_
                    = call_binary_llvm_normal(v, &llvm::IRBuilder<>::CreateOr);
        } break;
        case intrin_type::int_xor: {
            current_val_
                    = call_binary_llvm_normal(v, &llvm::IRBuilder<>::CreateXor);
        } break;
        case intrin_type::round: {
            current_val_ = call_unary_llvm_intrin(
                    v, cate, Intrinsic::nearbyint, true);
        } break;
        case intrin_type::ceil: {
            current_val_
                    = call_unary_llvm_intrin(v, cate, Intrinsic::ceil, true);
        } break;
        case intrin_type::floor: {
            current_val_
                    = call_unary_llvm_intrin(v, cate, Intrinsic::floor, true);
        } break;
        case intrin_type::log: {
            current_val_
                    = call_unary_llvm_intrin(v, cate, Intrinsic::log, true);
        } break;
        case intrin_type::max: {
            if (cate == CATE_FLOAT) {
                current_val_ = call_binary_llvm_intrin(
                        v, cate, Intrinsic::maxnum, true);
            } else {
                current_val_ = make_int_min_max(v, false, cate);
            }
        } break;
        case intrin_type::min: {
            if (cate == CATE_FLOAT) {
                current_val_ = call_binary_llvm_intrin(
                        v, cate, Intrinsic::minnum, true);
            } else {
                current_val_ = make_int_min_max(v, true, cate);
            }
        } break;
        case intrin_type::shl: {
            assert(v->args_.size() == 2);
            auto inval1 = generate_expr(v->args_[0]);
            auto inval2 = generate_expr(v->args_[1]);
            COMPILE_ASSERT(cate == CATE_INT || cate == CATE_UINT,
                    "Bad type. Expecting int: " << v);
            current_val_ = builder_.CreateShl(inval1, inval2);
        } break;
        case intrin_type::shr: {
            assert(v->args_.size() == 2);
            auto inval1 = generate_expr(v->args_[0]);
            auto inval2 = generate_expr(v->args_[1]);
            if (cate == CATE_INT) {
                current_val_ = builder_.CreateAShr(inval1, inval2);
            } else {
                COMPILE_ASSERT(
                        cate == CATE_UINT, "Bad type. Expecting int: " << v);
                current_val_ = builder_.CreateLShr(inval1, inval2);
            }
        } break;
        case intrin_type::broadcast: {
            assert(v->args_.size() == 1);
            auto inval1 = generate_expr(v->args_[0]);
            auto lanes = v->dtype_.lanes_;
            auto in_lanes = v->args_[0]->dtype_.lanes_;
            if (lanes != 1) {
                if (in_lanes != 1) {
                    while (in_lanes < lanes) {
                        std::vector<shuffle_idx_t> array(in_lanes << 1);
                        for (uint16_t i = 0; i < (in_lanes << 1); i++) {
                            array[i] = i;
                        }
                        inval1 = builder_.CreateShuffleVector(
                                inval1, inval1, array);
                        in_lanes = in_lanes << 1;
                    }
                    current_val_ = inval1;
                } else {
                    current_val_ = builder_.CreateVectorSplat(lanes, inval1);
                }
            } else {
                current_val_ = inval1;
            }
        } break;
        case intrin_type::fmadd: {
            assert(v->args_.size() == 3);
            auto inval1 = generate_expr(v->args_[0]);
            auto inval2 = generate_expr(v->args_[1]);
            auto inval3 = generate_expr(v->args_[2]);
            auto ret = builder_.CreateIntrinsic(Intrinsic::fma,
                    {get_type(v->dtype_)}, {inval1, inval2, inval3});
            ret->setFastMathFlags(builder_.getFastMathFlags());
            current_val_ = ret;
        } break;
        case intrin_type::reduce_add:
        case intrin_type::reduce_mul:
        case intrin_type::reduce_max:
        case intrin_type::reduce_min: {
            Intrinsic::ID cur_intrinsic = Intrinsic::not_intrinsic;
#if SC_LLVM_BACKEND > 11
#define LLVM_INTRINSIC_EXP_V2(name) Intrinsic::vector_reduce_##name
#define LLVM_INTRINSIC_EXP LLVM_INTRINSIC_EXP_V2
#elif SC_LLVM_BACKEND > 8
#define LLVM_INTRINSIC_EXP_V2(name) \
    Intrinsic::experimental_vector_reduce_v2_##name
#define LLVM_INTRINSIC_EXP(name) Intrinsic::experimental_vector_reduce_##name
#else
#define LLVM_INTRINSIC_EXP_V2(name) Intrinsic::experimental_vector_reduce_##name
#define LLVM_INTRINSIC_EXP LLVM_INTRINSIC_EXP_V2
#endif
            if (v->type_ == intrin_type::reduce_add) {
                if (cate == CATE_FLOAT) {
                    cur_intrinsic = LLVM_INTRINSIC_EXP_V2(fadd);
                } else {
                    cur_intrinsic = LLVM_INTRINSIC_EXP(add);
                }
            } else if (v->type_ == intrin_type::reduce_mul) {
                if (cate == CATE_FLOAT) {
                    cur_intrinsic = LLVM_INTRINSIC_EXP_V2(fmul);
                } else {
                    cur_intrinsic = LLVM_INTRINSIC_EXP(mul);
                }
            } else if (v->type_ == intrin_type::reduce_max) {
                if (cate == CATE_FLOAT) {
                    cur_intrinsic = LLVM_INTRINSIC_EXP(fmax);
                } else if (cate == CATE_INT) {
                    cur_intrinsic = LLVM_INTRINSIC_EXP(smax);
                } else if (cate == CATE_UINT) {
                    cur_intrinsic = LLVM_INTRINSIC_EXP(umax);
                }
            } else if (v->type_ == intrin_type::reduce_min) {
                if (cate == CATE_FLOAT) {
                    cur_intrinsic = LLVM_INTRINSIC_EXP(fmin);
                } else if (cate == CATE_INT) {
                    cur_intrinsic = LLVM_INTRINSIC_EXP(smin);
                } else if (cate == CATE_UINT) {
                    cur_intrinsic = LLVM_INTRINSIC_EXP(umin);
                }
            }
            assert(v->args_.size() == 1);
            auto inval = generate_expr(v->args_[0]);
            if ((v->type_ == intrin_type::reduce_add
                        || v->type_ == intrin_type::reduce_mul)
                    && cate == CATE_FLOAT) {
                bool is_fp16 = v->args_[0]->dtype_.is_etype(sc_data_etype::F16);
                current_val_ = builder_.CreateIntrinsic(
#if SC_LLVM_BACKEND > 11
                        cur_intrinsic, {inval->getType()},
#elif SC_LLVM_BACKEND > 8
                        cur_intrinsic, {get_type(v->dtype_), inval->getType()},
#else
                        cur_intrinsic,
                        {get_type(v->dtype_), get_type(v->dtype_),
                                inval->getType()},
#endif
                        {ConstantFP::get(get_type(v->dtype_),
                                 is_fp16 ? APFloat(APFloat::IEEEhalf(),
                                         APInt(16, static_cast<uint16_t>(0)))
                                         : APFloat(0.0f)),
                                inval});
                llvm::cast<CallInst>(*current_val_)
                        .setFastMathFlags(builder_.getFastMathFlags());
            } else {
                current_val_ = builder_.CreateIntrinsic(
#if SC_LLVM_BACKEND >= 10
                        cur_intrinsic, {inval->getType()},
#else
                        cur_intrinsic, {get_type(v->dtype_), inval->getType()},
#endif
                        {inval});
            }
        } break;
        case intrin_type::saturated_cast: {
            current_val_ = do_lower_saturated_cast(v);
        } break;
        case intrin_type::round_and_cast: {
            assert(v->args_.size() == 1);
            auto inval1 = generate_expr(v->args_[0]);
            COMPILE_ASSERT(v->dtype_.type_code_ == sc_data_etype::S32
                            && v->args_[0]->dtype_.type_code_
                                    == sc_data_etype::F32,
                    "LLVM backend has not yet support round_and_cast "
                    "like "
                    "this: " << v);
            switch (v->dtype_.lanes_) {
                case 1:
                    current_val_ = builder_.CreateFPToSI(
                            builder_.CreateUnaryIntrinsic(
                                    Intrinsic::round, inval1),
                            builder_.getInt32Ty());
                    break;
                case 4:
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_sse2_cvtps2dq, {}, inval1);
                    break;
                case 8:
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx_cvt_ps2dq_256, {}, inval1);
                    break;
                case 16:
                    COMPILE_ASSERT(ctx_->machine_.cpu_flags_.fAVX512F,
                            "round_and_cast of 16 floats needs AVX512");
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx512_mask_cvtps2dq_512, {},
                            {inval1, UndefValue::get(get_type(v->dtype_)),
                                    /*mask*/ builder_.getInt16(0xffff),
                                    /*rounding mode =
                                       _MM_FROUND_CUR_DIRECTION 0x04*/
                                    builder_.getInt32(0x04)});
                    break;
                default:
                    COMPILE_ASSERT(false,
                            "LLVM backend has not yet support "
                            "round_and_cast with lanes = "
                                    << v->dtype_.lanes_);
                    break;
            }
        } break;
        case intrin_type::permutex2var: {
            assert(v->args_.size() == 3);
            auto inval1 = generate_expr(v->args_[0]);
            auto inval2 = generate_expr(v->args_[1]);
            auto inval3 = generate_expr(v->args_[2]);
            switch (v->args_[0]->dtype_.type_code_) {
                case sc_data_etype::F32:
                    switch (v->args_[0]->dtype_.lanes_) {
                        case 4:
                            current_val_ = builder_.CreateIntrinsic(
                                    Intrinsic::x86_avx512_vpermi2var_ps_128, {},
                                    {inval1, inval2, inval3});
                            break;
                        case 8:
                            current_val_ = builder_.CreateIntrinsic(
                                    Intrinsic::x86_avx512_vpermi2var_ps_256, {},
                                    {inval1, inval2, inval3});
                            break;
                        case 16:
                            current_val_ = builder_.CreateIntrinsic(
                                    Intrinsic::x86_avx512_vpermi2var_ps_512, {},
                                    {inval1, inval2, inval3});
                            break;
                        default:
                            COMPILE_ASSERT(false,
                                    "Unimplement lanes for permute2var: "
                                            << v->args_[0]->dtype_.lanes_);
                    }
                    break;
                case sc_data_etype::U8:
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx512_vpermi2var_qi_128, {},
                            {inval1, inval2, inval3});
                    break;
                default:
                    COMPILE_ASSERT(
                            false, "Unimplement datatype for permute2var");
            }
        } break;
        case intrin_type::permutexvar: {
            COMPILE_ASSERT(v->args_.size() == 2,
                    "need two args in permutexvar instructions.");
            auto inval1 = generate_expr(v->args_[0]);
            auto inval2 = generate_expr(v->args_[1]);
            int lanes = v->intrin_attrs_->get<int>("lanes");
            auto elem_bits
                    = utils::get_sizeof_etype(v->args_[1]->dtype_.type_code_)
                    * 8 * lanes;
            switch (elem_bits) {
                case 8: {
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx512_permvar_qi_512, {},
                            {inval2, inval1});
                } break;
                case 16: {
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx512_permvar_hi_512, {},
                            {inval2, inval1});
                } break;
                case 64: {
                    // Because there are two forms of invocation of the
                    // vpermq instruction. One is such as
                    // _mm256_permutexvar_epi64 (__m256i idx, __m256i a) and
                    // _mm256_permute4x64_epi64 (__m256i a, const int imm8). The
                    // corresponding code construction shufflevector of these
                    // two forms in llvm is different, and we are consistent
                    // with the source code of llvm.
                    if (v->args_[0].isa<constant_c>()) {
                        unsigned imm = get_expr_as_int(v->args_[0]);
                        assert(v->args_[0].isa<constant_c>() || elem_bits > 0);
                        bool is_avx512
                                = utils::get_sizeof_type(v->args_[1]->dtype_)
                                        * 8
                                == 512;
                        auto target_type = is_avx512
                                ? get_type(sc_data_type_t::index(8))
                                : get_type(sc_data_type_t::index(4));
                        unsigned numelts = is_avx512 ? 8 : 4;
                        auto outtype = get_type(v->args_[1]->dtype_);
                        auto tmp1 = builder_.CreateBitCast(inval2, target_type);
                        // These intrinsics operate on 256-bit lanes of four
                        // 64-bit elements.
                        std::vector<shuffle_idx_t> indices(numelts);
                        for (unsigned l = 0; l != numelts; l += 4)
                            for (unsigned i = 0; i != 4; ++i)
                                indices[l + i] = l + ((imm >> (2 * i)) & 0x3);

                        auto ret = builder_.CreateShuffleVector(
                                tmp1, indices, "perm");
                        current_val_ = builder_.CreateBitCast(ret, outtype);
                    } else {
                        COMPILE_ASSERT(false,
                                "Currently we do not need "
                                "_mm512_permutexvar_epi64 (__m512i idx, "
                                "__m512i a)");
                    }
                } break;
                default: {
                    COMPILE_ASSERT(false,
                            "Currently, we don't support "
                            "\"" << v->args_[1]->dtype_
                                 << "\"");
                } break;
            }
        } break;
        case intrin_type::unpack_high:
        case intrin_type::unpack_low: {
            COMPILE_ASSERT(v->args_.size() == 2,
                    "Expecting size of args = 2, but get " << v->args_.size());
            auto inval1 = generate_expr(v->args_[0]);
            auto inval2 = generate_expr(v->args_[1]);
            Value *tmp1;
            Value *tmp2;
            auto elem_bits = v->intrin_attrs_->get<int>("elem_bits");
            std::vector<shuffle_idx_t> hi_array, lo_array;
            const int type_bits = utils::get_sizeof_type(v->dtype_) * 8;
            // We should unpack according to the number of the avx data type
            // bits.
            switch (type_bits) {
                case 128: {
                    switch (elem_bits) {
                        case 8: {
                            auto target_type = get_type(sc_data_type_t::u8(16));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {8, 16 + 8, 9,
                                    16 + 9, 10, 16 + 10, 11, 16 + 11, 12,
                                    16 + 12, 13, 16 + 13, 14, 16 + 14, 15,
                                    16 + 15};
                            lo_array.resize(16);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 8; });
                        } break;
                        case 16: {
                            auto target_type = get_type(sc_data_type_t::u16(8));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {
                                    4, 8 + 4, 5, 8 + 5, 6, 8 + 6, 7, 8 + 7};
                            lo_array.resize(8);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 4; });
                        } break;
                        case 32: {
                            auto target_type = get_type(sc_data_type_t::u32(4));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {
                                    2, 4 + 2, 3, 4 + 3};
                            lo_array.resize(4);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 2; });
                        } break;
                        case 64: {
                            auto target_type
                                    = get_type(sc_data_type_t::index(2));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {1, 2 + 1};
                            lo_array.resize(2);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 1; });
                        } break;
                        default: {
                            COMPILE_ASSERT(false,
                                    "Only support 8, 16, 32 and 64 bits");
                        };
                    };
                } break;
                case 256: {
                    switch (elem_bits) {
                        case 8: {
                            auto target_type = get_type(sc_data_type_t::u8(32));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {8, 32 + 8, 9,
                                    32 + 9, 10, 32 + 10, 11, 32 + 11, 12,
                                    32 + 12, 13, 32 + 13, 14, 32 + 14, 15,
                                    32 + 15, 24, 32 + 24, 25, 32 + 25, 26,
                                    32 + 26, 27, 32 + 27, 28, 32 + 28, 29,
                                    32 + 29, 30, 32 + 30, 31, 32 + 31};
                            lo_array.resize(32);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 8; });
                        } break;
                        case 16: {
                            auto target_type
                                    = get_type(sc_data_type_t::u16(16));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {4, 16 + 4, 5,
                                    16 + 5, 6, 16 + 6, 7, 16 + 7, 12, 16 + 12,
                                    13, 16 + 13, 14, 16 + 14, 15, 16 + 15};
                            lo_array.resize(16);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 4; });
                        } break;
                        case 32: {
                            auto target_type = get_type(sc_data_type_t::u32(8));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {
                                    2, 8 + 2, 3, 8 + 3, 6, 8 + 6, 7, 8 + 7};
                            lo_array.resize(8);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 2; });
                        } break;
                        case 64: {
                            auto target_type
                                    = get_type(sc_data_type_t::index(4));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {
                                    1, 4 + 1, 3, 4 + 3};
                            lo_array.resize(4);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 1; });
                        } break;
                        default: {
                            COMPILE_ASSERT(false,
                                    "Only support 8, 16, 32 and 64 bits");
                        } break;
                    }
                } break;
                case 512: {
                    switch (elem_bits) {
                        case 8: {
                            auto target_type = get_type(sc_data_type_t::u8(64));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {8, 64 + 8, 9,
                                    64 + 9, 10, 64 + 10, 11, 64 + 11, 12,
                                    64 + 12, 13, 64 + 13, 14, 64 + 14, 15,
                                    64 + 15, 24, 64 + 24, 25, 64 + 25, 26,
                                    64 + 26, 27, 64 + 27, 28, 64 + 28, 29,
                                    64 + 29, 30, 64 + 30, 31, 64 + 31, 40,
                                    64 + 40, 41, 64 + 41, 42, 64 + 42, 43,
                                    64 + 43, 44, 64 + 44, 45, 64 + 45, 46,
                                    64 + 46, 47, 64 + 47, 56, 64 + 56, 57,
                                    64 + 57, 58, 64 + 58, 59, 64 + 59, 60,
                                    64 + 60, 61, 64 + 61, 62, 64 + 62, 63,
                                    64 + 63};
                            lo_array.resize(64);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 8; });
                        } break;
                        case 16: {
                            auto target_type
                                    = get_type(sc_data_type_t::u16(32));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {4, 32 + 4, 5,
                                    32 + 5, 6, 32 + 6, 7, 32 + 7, 12, 32 + 12,
                                    13, 32 + 13, 14, 32 + 14, 15, 32 + 15, 20,
                                    32 + 20, 21, 32 + 21, 22, 32 + 22, 23,
                                    32 + 23, 28, 32 + 28, 29, 32 + 29, 30,
                                    32 + 30, 31, 32 + 31};
                            lo_array.resize(32);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 4; });
                        } break;
                        case 32: {
                            auto target_type
                                    = get_type(sc_data_type_t::u32(16));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {2, 18, 3, 19,
                                    2 + 4, 18 + 4, 3 + 4, 19 + 4, 2 + 8, 18 + 8,
                                    3 + 8, 19 + 8, 2 + 12, 18 + 12, 3 + 12,
                                    19 + 12};
                            lo_array.resize(16);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 2; });
                        } break;
                        case 64: {
                            auto target_type
                                    = get_type(sc_data_type_t::index(8));
                            tmp1 = builder_.CreateBitCast(inval1, target_type);
                            tmp2 = builder_.CreateBitCast(inval2, target_type);
                            hi_array = std::vector<shuffle_idx_t> {1, 9, 1 + 2,
                                    9 + 2, 1 + 4, 9 + 4, 1 + 6, 9 + 6};
                            lo_array.resize(8);
                            std::transform(hi_array.begin(), hi_array.end(),
                                    lo_array.begin(),
                                    [](shuffle_idx_t x) { return x - 1; });
                        } break;
                        default: {
                            COMPILE_ASSERT(false,
                                    "Only support 8, 16, 32 and 64 bits");
                        } break;
                    }
                } break;
                default: {
                    COMPILE_ASSERT(false, "Invalid simd bits: " << type_bits);
                } break;
            }
            ArrayRef<shuffle_idx_t> arr = v->type_ == intrin_type::unpack_high
                    ? hi_array
                    : lo_array;
            auto res = builder_.CreateShuffleVector(tmp1, tmp2, arr);
            current_val_ = builder_.CreateBitCast(res, get_type(v->dtype_));
        } break;
        case intrin_type::shuffle: {
            COMPILE_ASSERT(v->args_.size() == 2,
                    "args size must be 2, but get " << v->args_.size());
            auto inval1 = generate_expr(v->args_[0]);
            auto inval2 = generate_expr(v->args_[1]);
            auto imm8 = v->intrin_attrs_->get<int>("shuffle_imm");
            auto type_bits = v->intrin_attrs_->get<int>("type_bits");
            switch (type_bits) {
                case 32: {
                    unsigned num_elts = v->args_[0]->dtype_.lanes_;
                    unsigned num_lanes
                            = utils::get_sizeof_type(v->args_[0]->dtype_) * 8
                            / 128;
                    unsigned num_lanes_elts = num_elts / num_lanes;

                    // Splat the 8-bits of immediate 4 times to help the
                    // loop wrap around.
                    imm8 = (imm8 & 0xff) * 0x01010101;

                    std::vector<shuffle_idx_t> indices(num_elts);
                    for (unsigned l = 0; l != num_elts; l += num_lanes_elts) {
                        for (unsigned i = 0; i != num_lanes_elts; ++i) {
                            unsigned index = imm8 % num_lanes_elts;
                            imm8 /= num_lanes_elts;
                            if (i >= (num_lanes_elts / 2)) index += num_elts;
                            indices[l + i] = l + index;
                        }
                    }

                    current_val_ = builder_.CreateShuffleVector(
                            inval1, inval2, indices, "shuf32");
                } break;
                case 128: {
                    unsigned num_elts = v->args_[0]->dtype_.lanes_;
                    unsigned num_lanes
                            = utils::get_sizeof_type(v->args_[0]->dtype_) * 8
                                    == 512
                            ? 4
                            : 2;
                    unsigned num_lanes_elts = num_elts / num_lanes;

                    std::vector<shuffle_idx_t> indices(num_elts);
                    for (unsigned l = 0; l != num_elts; l += num_lanes_elts) {
                        unsigned index = (imm8 % num_lanes) * num_lanes_elts;
                        imm8 /= num_lanes; // Discard the bits we just used.
                        if (l >= (num_elts / 2))
                            index += num_elts; // Switch to other source.
                        for (unsigned i = 0; i != num_lanes_elts; ++i) {
                            indices[l + i] = index + i;
                        }
                    }

                    current_val_ = builder_.CreateShuffleVector(
                            inval1, inval2, indices, "shuf128");
                } break;
                default: {
                    COMPILE_ASSERT(false,
                            "Curerntly only support type_bits 32 and 128, "
                            "but get "
                                    << type_bits);
                } break;
            };
        } break;
        case intrin_type::permute: {
            assert(v->args_.size() == 2);

            auto imm8 = v->intrin_attrs_->get<int>("permute_imm");
            auto val0 = generate_expr(v->args_[0]);
            auto val1 = generate_expr(v->args_[1]);

            unsigned numelts = v->args_[0]->dtype_.lanes_;

            // This takes a very simple approach since there are two
            // lanes and a shuffle can have 2 inputs. So we reserve the
            // first input for the first lane and the second input for
            // the second lane. This may result in duplicate sources,
            // but this can be dealt with in the backend.

            Value *outops[2];
            int indices[16];
            for (unsigned l = 0; l != 2; ++l) {
                // Determine the source for this lane.
                if (imm8 & (1 << ((l * 4) + 3)))
                    outops[l]
                            = llvm::ConstantAggregateZero::get(val0->getType());
                else if (imm8 & (1 << ((l * 4) + 1)))
                    outops[l] = val1;
                else
                    outops[l] = val0;

                for (unsigned i = 0; i != numelts / 2; ++i) {
                    // Start with ith element of the source for this
                    // lane.
                    unsigned idx = (l * numelts) + i;
                    // If bit 0 of the immediate half is set, switch to
                    // the high half of the source.
                    if (imm8 & (1 << (l * 4))) idx += numelts / 2;
                    indices[(l * (numelts / 2)) + i] = idx;
                }
            }

            current_val_ = builder_.CreateShuffleVector(outops[0], outops[1],
                    ArrayRef<shuffle_idx_t>(indices, numelts), "vperm");
        } break;
        case intrin_type::prefetch: {
            assert(v->args_.size() == 1);
            auto locality = v->intrin_attrs_->get<int>("locality");
            assert(locality <= 3 && locality >= 0
                    && "bad locality for prefetch");
            auto inval1 = generate_expr(v->args_[0]);
            current_val_ = builder_.CreateIntrinsic(Intrinsic::prefetch,
#if SC_LLVM_BACKEND > 8
                    {builder_.getInt8PtrTy()},
#else
                    {},
#endif
                    {builder_.CreatePointerCast(
                             inval1, builder_.getInt8PtrTy()),
                            /*rw*/ builder_.getInt32(0),
                            /*locality*/
                            builder_.getInt32(3 - locality),
                            /*type:i/d*/ builder_.getInt32(1)});
        } break;
        case intrin_type::gather: {
            assert(v->args_.size() == 2);
            COMPILE_ASSERT(v->dtype_.is_etype(sc_data_etype::F32),
                    "Expecting f32 for gather: " << v);
            auto inval1 = builder_.CreatePointerCast(
                    generate_expr(v->args_[0]), builder_.getInt8PtrTy());
            auto inval2 = generate_expr(v->args_[1]);
            auto full_mask = generate_expr(make_expr<constant_node>(
                    std::vector<union_val>(
                            v->dtype_.lanes_, UINT64_C(0xffffffff)),
                    v->dtype_));
            auto dummy_vec = generate_expr(make_expr<constant_node>(
                    std::vector<union_val>(v->dtype_.lanes_, UINT64_C(0)),
                    v->dtype_));
            switch (v->dtype_.lanes_) {
                case 1: {
                    auto indexing = make_expr<indexing_node>(v->args_[0],
                            std::vector<expr> {v->args_[1]}, expr());
                    current_val_ = generate_expr(indexing);
                    break;
                }
                case 4:
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx2_gather_d_ps, {},
                            {dummy_vec, inval1, inval2, full_mask,
                                    builder_.getInt8(4)});
                    break;
                case 8:
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx2_gather_d_ps_256, {},
                            {dummy_vec, inval1, inval2, full_mask,
                                    builder_.getInt8(4)});
                    break;
                case 16: {
                    COMPILE_ASSERT(ctx_->machine_.cpu_flags_.fAVX512F,
                            "gather of 16 floats needs AVX512");
                    current_val_ = builder_.CreateIntrinsic(
                            Intrinsic::x86_avx512_gather_dps_512, {},
                            {dummy_vec, inval1, inval2,
                                    builder_.getInt16(0xffff),
                                    builder_.getInt32(4)});
                    break;
                }
                default:
                    COMPILE_ASSERT(false,
                            "LLVM backend has not yet support "
                            "gather with lanes = "
                                    << v->dtype_.lanes_);
                    break;
            }
        } break;
        case intrin_type::insert: {
            auto val0 = generate_expr(v->args_[0]);
            auto val1 = generate_expr(v->args_[1]);
            unsigned index = v->intrin_attrs_->get<int>("insert_imm");
            const unsigned elem_bits
                    = utils::get_sizeof_type(v->args_[1]->dtype_) * 8;
            auto lanes = v->args_[0]->dtype_.lanes_;
            auto type_code = v->args_[0]->dtype_.type_code_;
            if (ctx_->machine_.cpu_flags_.fAVX512F && elem_bits >= 128) {
                // avx512 128bit insert can use this part.
                unsigned dst_num_elts = lanes, src_num_elts = lanes / 2;
                unsigned subvectors = dst_num_elts / src_num_elts;
                index &= subvectors - 1;
                index *= src_num_elts;
                std::vector<shuffle_idx_t> indices(dst_num_elts);
                for (unsigned i = 0; i != dst_num_elts; ++i) {
                    indices[i] = (i >= src_num_elts)
                            ? src_num_elts + (i % src_num_elts)
                            : i;
                }
                Value *op1
                        = builder_.CreateShuffleVector(val1, indices, "widen");
                for (unsigned i = 0; i != dst_num_elts; ++i) {
                    if (i >= index && i < (index + src_num_elts)) {
                        indices[i] = (i - index) + dst_num_elts;
                    } else {
                        indices[i] = i;
                    }
                }
                current_val_ = builder_.CreateShuffleVector(
                        val0, op1, indices, "insert");
            } else {
                assert(elem_bits <= 128);
                uint64_t idx = index;
                idx &= lanes - 1;
                current_val_ = builder_.CreateInsertElement(val0, val1, idx);
            }
        } break;
        case intrin_type::extract: {
            auto val0 = generate_expr(v->args_[0]);
            unsigned index = v->intrin_attrs_->get<int>("extract_imm");
            const unsigned elem_bits = utils::get_sizeof_type(v->dtype_) * 8;
            auto lanes = v->args_[0]->dtype_.lanes_;
            auto type_code = v->args_[0]->dtype_.type_code_;
            if (ctx_->machine_.cpu_flags_.fAVX512F && elem_bits >= 128) {
                // avx512 128bit extract can use this part.
                unsigned dst_num_elts
                        = index / utils::get_sizeof_etype(v->dtype_.as_etype()),
                        src_num_elts = lanes;
                unsigned subvectors = src_num_elts / dst_num_elts;
                index &= subvectors - 1;
                index *= src_num_elts;

                std::vector<shuffle_idx_t> indices(dst_num_elts);
                for (unsigned i = 0; i != dst_num_elts; ++i)
                    indices[i] = i + index;
                current_val_ = builder_.CreateShuffleVector(
                        val0, indices, "extract");
            } else {
                assert(elem_bits <= 128);
                uint64_t idx = (uint64_t)index;
                idx &= lanes - 1;
                current_val_ = builder_.CreateExtractElement(val0, idx);
            }
        } break;
        default: {
            std::stringstream ss;
            ss << "Intrinsics not implemented ";
            v->to_string(ss);
            throw std::runtime_error(ss.str());
        }
    }
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

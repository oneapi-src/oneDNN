/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_GEN_OPERATION_OPERATIONS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_GEN_OPERATION_OPERATIONS_HPP

#include "operand.hpp"

namespace sc {
namespace sc_xbyak {

//===========================================================================
// Special Instructions
//===========================================================================

void X86_MOV(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src);
void AVX_MOVSS(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src);
void AVX_MOVPS(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src);
void AVX_MOVPI8(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src);
void AVX_MOVPI16(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src);
void AVX_MOVPI32(Xbyak::CodeGenerator &gen, const operand &op_dst,
        const operand &op_src);

//===========================================================================
// Instruction Format Labels
// * R: x86 reg, sub-reg: 8/16/32/64
// * X: avx reg, sub-reg: x/y/z
// * K: avx512 mask reg, writemask: w/wz
// * T: amx tile reg
// * M: memory
// * I: immediate
//===========================================================================

/*
 * X86_64 Instruction Format
 */

#define X86_RM(GEN, INS, OP_1) \
    { \
        COMPILE_ASSERT(OP_1.is_reg() || OP_1.is_addr(), \
                "Invalid x86_" #INS << ": " << OP_1); \
        (GEN).INS(OP_1.get_operand()); \
    }

#define X86_R64_M(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_reg() && OP_2.is_addr(), \
                "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_reg64(), OP_2.get_addr()); \
    }

#define X86_R64_RM(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_reg() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_reg64(), OP_2.get_reg()); \
        } else if (OP_1.is_reg() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_reg64(), OP_2.get_addr()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define X86_RM_R8I(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_reg() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_reg8()); \
        } else if (OP_1.is_reg() && OP_2.is_imm()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_imm()); \
        } else if (OP_1.is_addr() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_reg8()); \
        } else if (OP_1.is_addr() && OP_2.is_imm()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define X86_RM_RMI(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_reg() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_reg()); \
        } else if (OP_1.is_reg() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_addr()); \
        } else if (OP_1.is_reg() && OP_2.is_imm()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_imm()); \
        } else if (OP_1.is_addr() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_reg()); \
        } else if (OP_1.is_addr() && OP_2.is_imm()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define X86_RM_RM(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_reg() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_reg()); \
        } else if (OP_1.is_reg() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_addr()); \
        } else if (OP_1.is_addr() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_reg()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define X86_R_RM(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_reg() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_reg()); \
        } else if (OP_1.is_reg() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_addr()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define X86_R_RM_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        if (OP_1.is_reg() && OP_2.is_reg() && OP_3.is_imm()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_reg(), OP_3.get_imm()); \
        } else if (OP_1.is_reg() && OP_2.is_addr() && OP_3.is_imm()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_addr(), OP_3.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2 \
                                        << ", " << OP_3); \
        } \
    }

/*
 * AVX Instruction Format
 */

#define AVX_R32_XM(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_reg() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_reg32(), OP_2.get_xyz()); \
        } else if (OP_1.is_reg() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_reg32(), OP_2.get_addr()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_R32M_Xx_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        if (OP_1.is_reg() && OP_2.is_xyz() && OP_3.is_imm()) { \
            (GEN).INS(OP_1.get_reg32(), OP_2.get_xmm(), OP_3.get_imm()); \
        } else if (OP_1.is_addr() && OP_2.is_xyz() && OP_3.is_imm()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_xmm(), OP_3.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_XxR32M_XxR32M(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_xyz() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_reg32()); \
        } else if (OP_1.is_xyz() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_addr()); \
        } else if (OP_1.is_reg() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_reg32(), OP_2.get_xmm()); \
        } else if (OP_1.is_addr() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_xmm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_XM_XM(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_xyz() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz()); \
        } else if (OP_1.is_xyz() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_addr()); \
        } else if (OP_1.is_addr() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_xyz()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_X_XM(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && (OP_2.is_xyz() || OP_2.is_addr()), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_xyz(), OP_2.get_operand()); \
    }

#define AVX_X_X_XM(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() \
                        && (OP_3.is_xyz() || OP_3.is_addr()), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz(), OP_3.get_operand()); \
    }

#define AVX_X_X_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz(), OP_3.get_imm()); \
    }

#define AVX_X_X_XI(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_xyz()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz(), OP_3.get_xyz()); \
        } else if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_imm()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz(), OP_3.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 \
                                        << ", " << OP_3); \
        } \
    }

#define AVX_X_X_XxI(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_xyz()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz(), OP_3.get_xmm()); \
        } else if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_imm()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz(), OP_3.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 \
                                        << ", " << OP_3); \
        } \
    }

#define AVX_XxM_Xy_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT((OP_1.is_xyz() || OP_1.is_addr()) && OP_2.is_xyz() \
                        && OP_3.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_operand(), OP_2.get_ymm(), OP_3.get_imm()); \
    }

#define AVX_X_X_XM_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() \
                        && (OP_3.is_xyz() || OP_4.is_addr()) && OP_4.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3 << ", " << OP_4); \
        (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

#define AVX_Xy_Xy_XM_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() \
                        && (OP_3.is_xyz() || OP_4.is_addr()) && OP_4.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3 << ", " << OP_4); \
        (GEN).INS(OP_1.get_ymm(), OP_2.get_ymm(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

/*
 * AVX512 Instruction Format
 */

#define AVX512_X_XxR32M(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_xyz() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_reg32()); \
        } else if (OP_1.is_xyz() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_addr()); \
        } else if (OP_1.is_xyz() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_xmm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX512_X_XxR16M(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_xyz() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_reg16()); \
        } else if (OP_1.is_xyz() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_addr()); \
        } else if (OP_1.is_xyz() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_xmm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX512_X_XxR8M(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_xyz() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_reg8()); \
        } else if (OP_1.is_xyz() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_addr()); \
        } else if (OP_1.is_xyz() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_xmm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX512_Z_Z_O_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() \
                        && (OP_3.is_xyz() || OP_3.is_addr()) && OP_4.is_imm(), \
                "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                       << OP_3 << ", " << OP_4); \
        (GEN).INS(OP_1.get_zmm(), OP_2.get_zmm(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

#define AVX512_XM_X(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_xyz() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz()); \
        } else if (OP_1.is_addr() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_xyz()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX512_X_X_XM(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() \
                        && (OP_3.is_xyz() || OP_3.is_addr()), \
                "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                       << OP_3); \
        (GEN).INS(OP_1.get_xyz(), OP_2.get_xyz(), OP_3.get_operand()); \
    }

#define AVX512_X_XM_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && (OP_2.is_xyz() || OP_2.is_addr()) \
                        && OP_3.is_imm(), \
                "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                       << OP_3); \
        (GEN).INS(OP_1.get_xyz(), OP_2.get_operand(), OP_3.get_imm()); \
    }

#define AVX512_XM_XM_Kwz(GEN, INS, OP_1, OP_2, K) \
    { \
        if (OP_1.is_xyz() && OP_2.is_xyz() && K.is_mask()) { \
            (GEN).INS(OP_1.get_xyz() | K.get_mask() | Xbyak::util::T_z, \
                    OP_2.get_xyz()); \
        } else if (OP_1.is_xyz() && OP_2.is_addr() && K.is_mask()) { \
            (GEN).INS(OP_1.get_xyz() | K.get_mask() | Xbyak::util::T_z, \
                    OP_2.get_addr()); \
        } else if (OP_1.is_addr() && OP_2.is_xyz() && K.is_mask()) { \
            (GEN).INS(OP_1.get_addr() | K.get_mask() | Xbyak::util::T_z, \
                    OP_2.get_xyz()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2 \
                                           << ", " << K); \
        } \
    }

#define AVX512_X_X_XM_Kw(GEN, INS, OP_1, OP_2, OP_3, K) \
    { \
        if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_xyz() && K.is_mask()) { \
            (GEN).INS(OP_1.get_xyz() | K.get_mask(), OP_2.get_xyz(), \
                    OP_3.get_xyz()); \
        } else if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_addr() \
                && K.is_mask()) { \
            (GEN).INS(OP_1.get_xyz() | K.get_mask(), OP_2.get_xyz(), \
                    OP_3.get_addr()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2 \
                                           << ", " << OP_3 << ", " << K); \
        } \
    }

#define AVX512_XyM_Xz_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        if ((OP_1.is_xyz() || OP_1.is_addr()) && OP_2.is_xyz() \
                && OP_3.is_imm()) { \
            (GEN).INS(OP_1.get_operand(), OP_2.get_zmm(), OP_3.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2 \
                                           << ", " << OP_3); \
        } \
    }

#define AVX512_K_X_XM_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_mask() && OP_2.is_xyz() \
                        && (OP_3.is_xyz() || OP_3.is_addr()) && OP_4.is_imm(), \
                "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                       << OP_3 << ", " << OP_4); \
        (GEN).INS(OP_1.get_mask(), OP_2.get_xyz(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

#define AVX512_KR32M_KR32M(GEN, INS, OP_1, OP_2) \
    { \
        if (OP_1.is_mask() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_mask(), OP_2.get_reg32()); \
        } else if (OP_1.is_mask() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_mask(), OP_2.get_addr()); \
        } else if (OP_1.is_addr() && OP_2.is_mask()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_mask()); \
        } else if (OP_1.is_reg() && OP_2.is_mask()) { \
            (GEN).INS(OP_1.get_reg32(), OP_2.get_mask()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx512_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AMX_M(GEN, INS, OP_1) \
    { \
        COMPILE_ASSERT(OP_1.is_addr(), "Invalid AMX_" #INS << ": " << OP_1); \
        (GEN).INS(OP_1.get_addr()); \
    }

#define AMX_T(GEN, INS, OP_1) \
    { \
        COMPILE_ASSERT(OP_1.is_tmm(), "Invalid AMX_" #INS << ": " << OP_1); \
        (GEN).INS(OP_1.get_tmm()); \
    }

#define AMX_T_T_T(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_tmm() && OP_2.is_tmm() && OP_3.is_tmm(), \
                "Invalid AMX_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_tmm(), OP_2.get_tmm(), OP_3.get_tmm()); \
    }

#define AMX_T_M(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_tmm() && OP_2.is_addr(), \
                "Invalid AMX_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_tmm(), OP_2.get_addr()); \
    }

#define AMX_M_T(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_addr() && OP_2.is_tmm(), \
                "Invalid AMX_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_addr(), OP_2.get_tmm()); \
    }

} // namespace sc_xbyak
} // namespace sc

#endif

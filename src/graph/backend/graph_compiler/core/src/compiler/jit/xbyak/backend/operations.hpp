/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_OPERATIONS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_BACKEND_OPERATIONS_HPP

#include "operand.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

//===========================================================================
// Instruction Format Labels
// * R: x86 reg, sub-reg: 8/16/32/64
// * X/Y/Z: avx reg
// * K: avx512 mask reg
// * T: amx tile reg
// * M: memory
// * I: immediate
//===========================================================================

/*
 * X86_64 Instruction Format
 */

#define X86_RM(GEN, INS, OP_1) \
    { \
        COMPILE_ASSERT(OP_1.is_r_m(), "Invalid x86_" #INS << ": " << OP_1); \
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
        COMPILE_ASSERT(OP_1.is_reg() && OP_2.is_r_m(), \
                "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_reg64(), OP_2.get_operand()); \
    }

#define X86_R32_RM(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_reg() && OP_2.is_r_m(), \
                "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_reg32(), OP_2.get_operand()); \
    }

#define X86_R_RM(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_reg() && OP_2.is_r_m(), \
                "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_reg(), OP_2.get_operand()); \
    }

#define X86_R_RM_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_reg() && OP_2.is_r_m() && OP_3.is_imm(), \
                "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_reg(), OP_2.get_operand(), OP_3.get_imm()); \
    }

#define X86_RM_R8I(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_r_m() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_operand(), OP_2.get_reg8()); \
        } else if (OP_1.is_r_m() && OP_2.is_imm()) { \
            (GEN).INS(OP_1.get_operand(), OP_2.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define X86_RM_RMI(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_reg() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_addr()); \
        } else if (OP_1.is_r_m() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_operand(), OP_2.get_reg()); \
        } else if (OP_1.is_r_m() && OP_2.is_imm()) { \
            (GEN).INS(OP_1.get_operand(), OP_2.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define X86_RM_RM(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_reg() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_reg(), OP_2.get_addr()); \
        } else if (OP_1.is_r_m() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_operand(), OP_2.get_reg()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define X86_R64_R64_R64(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        /*  */ if (OP_1.is_reg() && OP_2.is_reg() && OP_3.is_reg()) { \
            (GEN).INS(OP_1.get_reg64(), OP_2.get_reg64(), OP_3.get_reg64()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid x86_" #INS << ": " << OP_1 << ", " << OP_2 << "," \
                                        << OP_3); \
        } \
    }

/*
 * AVX Instruction Format
 */

#define AVX_R64_X(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_reg() && OP_2.is_xyz(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_reg64(), OP_2.get_xmm()); \
    }

#define AVX_R32_XM(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_reg() && OP_2.is_x_m(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_reg32(), OP_2.get_operand()); \
    }

#define AVX_R64_XM(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_reg() && OP_2.is_x_m(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_reg64(), OP_2.get_operand()); \
    }

#define AVX_RM_X_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_r_m() && OP_2.is_xyz() && OP_3.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_operand(), OP_2.get_xmm(), OP_3.get_imm()); \
    }

#define AVX_X_X_RM_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() \
                        && (OP_3.is_reg() || OP_3.is_addr()) && OP_4.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << "," \
                                    << OP_3 << "," << OP_4); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

#define AVX_Y_Y_YM_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() \
                        && (OP_3.is_xyz() || OP_3.is_addr()) && OP_4.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << "," \
                                    << OP_3 << "," << OP_4); \
        (GEN).INS(OP_1.get_ymm(), OP_2.get_ymm(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

#define AVX_X_R(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_reg(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_reg()); \
    }

#define AVX_X_M(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_addr(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_addr()); \
    }

#define AVX_Y_M(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_addr(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_ymm(), OP_2.get_addr()); \
    }

#define AVX_X_XM(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_x_m(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_operand()); \
    }

#define AVX_XM_X(GEN, INS, OP_1, OP_2) \
    { \
        COMPILE_ASSERT(OP_1.is_x_m() && OP_2.is_xyz(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        (GEN).INS(OP_1.get_operand(), OP_2.get_xmm()); \
    }

#define AVX_X_X_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_imm()); \
    }

#define AVX_X_X_RM(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_r_m(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_operand()); \
    }

#define AVX_X_XM_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_x_m() && OP_3.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_operand(), OP_3.get_imm()); \
    }

#define AVX_X_X_XM(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_x_m(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_operand()); \
    }

#define AVX_Y_Y_YM(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_x_m(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_ymm(), OP_2.get_ymm(), OP_3.get_operand()); \
    }

#define AVX_X_M_X(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_addr() && OP_3.is_xyz(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_addr(), OP_3.get_xmm()); \
    }

#define AVX_X_X_XM_X(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_x_m() \
                        && OP_4.is_xyz(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3 << ", " << OP_4); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_operand(), \
                OP_4.get_xmm()); \
    }

#define AVX_XM_Y_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_x_m() && OP_2.is_xyz() && OP_3.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_operand(), OP_2.get_ymm(), OP_3.get_imm()); \
    }

#define AVX_Y_YM_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_x_m() && OP_3.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_ymm(), OP_2.get_operand(), OP_3.get_imm()); \
    }

#define AVX_YM_Z_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_x_m() && OP_2.is_xyz() && OP_3.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_operand(), OP_2.get_zmm(), OP_3.get_imm()); \
    }

#define AVX_X_X_XM_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_x_m() \
                        && OP_4.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3 << ", " << OP_4); \
        (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

#define AVX_Y_Y_XM_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_x_m() \
                        && OP_4.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3 << ", " << OP_4); \
        (GEN).INS(OP_1.get_ymm(), OP_2.get_ymm(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

#define AVX_Z_Z_XM_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_x_m() \
                        && OP_4.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3 << ", " << OP_4); \
        (GEN).INS(OP_1.get_zmm(), OP_2.get_zmm(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

#define AVX_XM_Z_I(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        COMPILE_ASSERT(OP_1.is_x_m() && OP_2.is_xyz() && OP_3.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3); \
        (GEN).INS(OP_1.get_operand(), OP_2.get_zmm(), OP_3.get_imm()); \
    }

#define AVX_K_X_XM_I(GEN, INS, OP_1, OP_2, OP_3, OP_4) \
    { \
        COMPILE_ASSERT(OP_1.is_mask() && OP_2.is_xyz() && OP_3.is_x_m() \
                        && OP_4.is_imm(), \
                "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 << ", " \
                                    << OP_3 << ", " << OP_4); \
        (GEN).INS(OP_1.get_mask(), OP_2.get_xmm(), OP_3.get_operand(), \
                OP_4.get_imm()); \
    }

#define AVX_XM_XM(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_xyz() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm()); \
        } else if (OP_1.is_xyz() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_addr()); \
        } else if (OP_1.is_addr() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_xmm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_XM_X_XM(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        /*  */ if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_addr()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_addr()); \
        } else if (OP_1.is_addr() && OP_2.is_xyz() && OP_3.is_xyz()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_xmm(), OP_3.get_xmm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 \
                                        << ", " << OP_3); \
        } \
    }

#define AVX_X_X_XI(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        /*  */ if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_xyz()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_xmm()); \
        } else if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_imm()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 \
                                        << ", " << OP_3); \
        } \
    }

#define AVX_X_XM_XMI(GEN, INS, OP_1, OP_2, OP_3) \
    { \
        /*  */ if (OP_1.is_xyz() && OP_2.is_xyz() && OP_3.is_x_m()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm(), OP_3.get_operand()); \
        } else if (OP_1.is_xyz() && OP_2.is_x_m() && OP_3.is_imm()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_operand(), OP_3.get_imm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2 \
                                        << ", " << OP_3); \
        } \
    }

#define AVX_X_XMR32(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_xyz() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_reg32()); \
        } else if (OP_1.is_xyz() && OP_2.is_x_m()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_operand()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_X_XMR16(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_xyz() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_reg16()); \
        } else if (OP_1.is_xyz() && OP_2.is_x_m()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_operand()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_X_XMR8(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_xyz() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_reg8()); \
        } else if (OP_1.is_xyz() && OP_2.is_x_m()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_operand()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_XMR32_XMR32(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_xyz() && OP_2.is_reg()) { \
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

#define AVX_XMR64_XMR64(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_xyz() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_xmm()); \
        } else if (OP_1.is_xyz() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_reg64()); \
        } else if (OP_1.is_xyz() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_xmm(), OP_2.get_addr()); \
        } else if (OP_1.is_reg() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_reg64(), OP_2.get_xmm()); \
        } else if (OP_1.is_addr() && OP_2.is_xyz()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_xmm()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_KMR32_KMR32(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_mask() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_mask(), OP_2.get_reg32()); \
        } else if (OP_1.is_mask() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_mask(), OP_2.get_addr()); \
        } else if (OP_1.is_mask() && OP_2.is_mask()) { \
            (GEN).INS(OP_1.get_mask(), OP_2.get_mask()); \
        } else if (OP_1.is_addr() && OP_2.is_mask()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_mask()); \
        } else if (OP_1.is_reg() && OP_2.is_mask()) { \
            (GEN).INS(OP_1.get_reg32(), OP_2.get_mask()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

#define AVX_KMR64_KMR64(GEN, INS, OP_1, OP_2) \
    { \
        /*  */ if (OP_1.is_mask() && OP_2.is_reg()) { \
            (GEN).INS(OP_1.get_mask(), OP_2.get_reg64()); \
        } else if (OP_1.is_mask() && OP_2.is_addr()) { \
            (GEN).INS(OP_1.get_mask(), OP_2.get_addr()); \
        } else if (OP_1.is_mask() && OP_2.is_mask()) { \
            (GEN).INS(OP_1.get_mask(), OP_2.get_mask()); \
        } else if (OP_1.is_addr() && OP_2.is_mask()) { \
            (GEN).INS(OP_1.get_addr(), OP_2.get_mask()); \
        } else if (OP_1.is_reg() && OP_2.is_mask()) { \
            (GEN).INS(OP_1.get_reg64(), OP_2.get_mask()); \
        } else { \
            COMPILE_ASSERT(false, \
                    "Invalid avx_" #INS << ": " << OP_1 << ", " << OP_2); \
        } \
    }

/*
 * AMX Instruction Format
 */

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

} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

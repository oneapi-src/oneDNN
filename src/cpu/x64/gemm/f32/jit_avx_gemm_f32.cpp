/*******************************************************************************
* Copyright 2016-2024 Intel Corporation
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

#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/gemm/f32/gemm_utils_f32.hpp"
#include "cpu/gemm/f32/ref_gemm_f32.hpp"
#include "cpu/gemm/gemm_msan_unpoison.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/gemm/gemm_driver.hpp"

#include "cpu/x64/gemm/f32/jit_avx_gemm_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

#define CACHE_LINE_SIZE 64

#define STACKSIZE get_size_of_abi_save_regs()
#ifdef _WIN32
#define STACK_CAPACITY 8448
#else
// Roughly 4 4kB pages for kernel.
#define STACK_CAPACITY (4 * PAGE_4K)
#endif
#define SIZE 4
#define OFFSET 32
#define BASE_SHIFT 2
#define SECOND_FETCH 14

namespace avx_gemm_f32 {
using namespace gemm_utils;
using namespace Xbyak;

struct xbyak_gemm_t : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx_gemm_f32_xbyak_gemm)
    xbyak_gemm_t(char isTransA, char isTransB, float beta, bool hasBias = false)
        : jit_generator(jit_name())
        , isTransA(isTransA)
        , isTransB(isTransB)
        , hasBias(hasBias)
        , is_avx2(mayiuse(avx2))
        , UNROLL_M(is_avx2 ? 16 : 8)
        , UNROLL_N(6)
        , isBeta0(beta == 0.0)
        , isBetaN(!isBeta0 && beta != 1.0)
        , PREFETCHSIZEA(128)
        , PREFETCHSIZEB((!isTransB) ? -16 : 0)
        , STACK_K_CAPACITY((STACK_CAPACITY - 256) / (SIZE * UNROLL_M)) {}

    // Fused multiply add; may become one or two instructions
    void fma(bool useFma, const Ymm &reg0, const Ymm &reg1, const Ymm &reg2,
            bool overWrite = false) {
        if (useFma) {
            if (is_avx2) {
                vfmadd231ps(reg2, reg1, reg0);
            } else {
                assert(UNROLL_M == 8);
                auto tent_vreg = overWrite ? reg1 : ymm1;
                vmulps(tent_vreg, reg1, reg0);
                vaddps(reg2, reg2, tent_vreg);
            }
        } else {
            if (!overWrite) {
                vmulps(ymm15, reg1, reg0);
                vaddps(reg2, reg2, ymm15);
            } else {
                vmulps(reg1, reg1, reg0);
                vaddps(reg2, reg2, reg1);
            }
        }
    }

    // Inner kernel with k=8
    void innerkernel8(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const Ymm &reg00, const Ymm &reg01, const Ymm &reg02,
            const Ymm &reg03, const Ymm &reg04, const Ymm &reg05,
            const Ymm &reg06, const Ymm &reg07, const Ymm &reg08,
            const Ymm &reg09, const Ymm &reg10, const Ymm &reg11,
            const Ymm &reg12, const Ymm &reg13, const Ymm &reg14,
            const Ymm &reg15, const Ymm &reg16, const Ymm &reg17,
            const Ymm &reg18, const Ymm &reg19, const Ymm &reg20,
            const Ymm &reg21, const Ymm &reg22, const Ymm &reg23) {
        Ymm fmareg;

        if (!isDirect) {
            prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 0) * SIZE]);
        } else {
            prefetcht0(ptr[AO1 + LDA4]);
        }

        for (int i = 0; i < 8; i++) {
            if (isDirect) {
                if (isLoad1Unmasked) {
                    vmovups(ymm0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm0, VMASK, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(ymm1, VMASK,
                                ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    }
                }
                add(AO1, LDA);
            }

            if (!isTransB) {
                vbroadcastss(ymm2, ptr[BO1 + (i - OFFSET) * SIZE]);
            } else {
                vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
            }
            fmareg = (i % 2 == 0) ? reg00 : reg12;
            fma(useFma, ymm0, ymm2, fmareg);
            if (unroll_m >= 16) {
                fmareg = (i % 2 == 0) ? reg06 : reg18;
                fma(useFma, ymm1, ymm2, fmareg);
            }
            if (i == 0) {
                if (!isTransB) { prefetcht0(ptr[BO1 + PREFETCHSIZEB * SIZE]); }
            }
            if (unroll_n >= 2) {
                if (!isTransB) {
                    if (i == 1) {
                        prefetcht0(ptr[BO1 + LDB + PREFETCHSIZEB * SIZE]);
                    }
                    vbroadcastss(
                            ymm2, ptr[BO1 + LDB * 1 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg01 : reg13;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg07 : reg19;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (isCopy) {
                vmovups(ptr[LDA4 + (unroll_m * i + 0 * 8 - OFFSET) * SIZE],
                        ymm0);
                if (unroll_m >= 16) {
                    vmovups(ptr[LDA4 + (unroll_m * i + 1 * 8 - OFFSET) * SIZE],
                            ymm1);
                }
                if (i == 7) { sub(LDA4, -unroll_m * 8 * SIZE); }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    if (i == 2) {
                        prefetcht0(ptr[BO1 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                    }
                    vbroadcastss(
                            ymm2, ptr[BO1 + LDB * 2 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg02 : reg14;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg08 : reg20;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (i == 7) {
                if (!isTransB) { sub(BO1, -8 * SIZE); }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    if (i == 3) { prefetcht0(ptr[BO2 + PREFETCHSIZEB * SIZE]); }
                    vbroadcastss(ymm2, ptr[BO2 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg03 : reg15;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg09 : reg21;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    if (i == 4) {
                        prefetcht0(ptr[BO2 + LDB + PREFETCHSIZEB * SIZE]);
                    }
                    vbroadcastss(
                            ymm2, ptr[BO2 + LDB * 1 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg04 : reg16;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg10 : reg22;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    if (i == 5) {
                        prefetcht0(ptr[BO2 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                    }
                    vbroadcastss(
                            ymm2, ptr[BO2 + LDB * 2 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg05 : reg17;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg11 : reg23;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }
            if (isTransB) {
                prefetcht0(ptr[BO1 + BO2]);
                add(BO1, LDB);
            }

            if (i == 0) {
                if (unroll_m >= 4) {
                    if (!isDirect) {
                        prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 2 * 8) * SIZE]);
                    } else {
                        prefetcht0(ptr[AO1 + LDA4]);
                    }
                }
            }
            if (i == 1 || i == 2) {
                if (unroll_m >= 8) {
                    if (!isDirect) {
                        prefetcht0(ptr[AO1
                                + (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE]);
                    } else {
                        prefetcht0(ptr[AO1 + LDA4]);
                    }
                }
            }
            if (i == 3 || i == 4 || i == 5 || i == 6) {
                if (unroll_m >= 16) {
                    if (!isDirect) {
                        prefetcht0(ptr[AO1
                                + (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE]);
                    } else {
                        prefetcht0(ptr[AO1 + LDA4]);
                    }
                }
            }
            if (i == 7) {
                if (!isTransB) {
                    if (unroll_n >= 4) { sub(BO2, -8 * SIZE); }
                }
                if (!isTransA) {
                    prefetcht2(ptr[AA]);
                    lea(AA, ptr[AA + LDA]);
                }
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    vmovups(ymm0,
                            ptr[AO1
                                    + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                                            * SIZE]);
                } else {
                    vmaskmovps(ymm0, VMASK,
                            ptr[AO1
                                    + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                                            * SIZE]);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm1,
                                ptr[AO1
                                        + (unroll_m * (i + 1) + 1 * 8 - OFFSET)
                                                * SIZE]);
                    } else {
                        vmaskmovps(ymm1, VMASK,
                                ptr[AO1
                                        + (unroll_m * (i + 1) + 1 * 8 - OFFSET)
                                                * SIZE]);
                    }
                }
            }
        }

        if (!isDirect) { sub(AO1, -unroll_m * 8 * SIZE); }
        sub(LL, 1);
    }

    // Inner kernel with k=4
    void innerkernel4(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const Ymm &reg00, const Ymm &reg01, const Ymm &reg02,
            const Ymm &reg03, const Ymm &reg04, const Ymm &reg05,
            const Ymm &reg06, const Ymm &reg07, const Ymm &reg08,
            const Ymm &reg09, const Ymm &reg10, const Ymm &reg11,
            const Ymm &reg12, const Ymm &reg13, const Ymm &reg14,
            const Ymm &reg15, const Ymm &reg16, const Ymm &reg17,
            const Ymm &reg18, const Ymm &reg19, const Ymm &reg20,
            const Ymm &reg21, const Ymm &reg22, const Ymm &reg23) {
        Ymm fmareg;

        if (!isDirect) {
            prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 0) * SIZE]);
        } else {
            prefetcht0(ptr[AO1 + LDA4]);
        }

        for (int i = 0; i < 4; i++) {
            if (isDirect) {
                if (isLoad1Unmasked) {
                    vmovups(ymm0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm0, VMASK, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(ymm1, VMASK,
                                ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    }
                }
                add(AO1, LDA);
            }

            if (!isTransB) {
                vbroadcastss(ymm2, ptr[BO1 + (i - OFFSET) * SIZE]);
            } else {
                vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
            }
            fmareg = (i % 2 == 0) ? reg00 : reg12;
            fma(useFma, ymm0, ymm2, fmareg);
            if (unroll_m >= 16) {
                fmareg = (i % 2 == 0) ? reg06 : reg18;
                fma(useFma, ymm1, ymm2, fmareg);
            }
            if (i == 0) {
                if (!isTransB) { prefetcht0(ptr[BO1 + PREFETCHSIZEB * SIZE]); }
            }
            if (unroll_n >= 2) {
                if (!isTransB) {
                    if (i == 1) {
                        prefetcht0(ptr[BO1 + LDB + PREFETCHSIZEB * SIZE]);
                    }
                    vbroadcastss(
                            ymm2, ptr[BO1 + LDB * 1 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg01 : reg13;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg07 : reg19;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (isCopy) {
                vmovups(ptr[LDA4 + (unroll_m * i + 0 * 8 - OFFSET) * SIZE],
                        ymm0);
                if (unroll_m >= 16) {
                    vmovups(ptr[LDA4 + (unroll_m * i + 1 * 8 - OFFSET) * SIZE],
                            ymm1);
                }
                if (i == 3) { sub(LDA4, -unroll_m * 4 * SIZE); }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    if (i == 2) {
                        prefetcht0(ptr[BO1 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                    }
                    vbroadcastss(
                            ymm2, ptr[BO1 + LDB * 2 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg02 : reg14;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg08 : reg20;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (i == 7) {
                if (!isTransB) { sub(BO1, -8 * SIZE); }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    if (i == 3) { prefetcht0(ptr[BO2 + PREFETCHSIZEB * SIZE]); }
                    vbroadcastss(ymm2, ptr[BO2 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg03 : reg15;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg09 : reg21;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    if (i == 4) {
                        prefetcht0(ptr[BO2 + LDB + PREFETCHSIZEB * SIZE]);
                    }
                    vbroadcastss(
                            ymm2, ptr[BO2 + LDB * 1 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg04 : reg16;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg10 : reg22;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    if (i == 5) {
                        prefetcht0(ptr[BO2 + LDB * 2 + PREFETCHSIZEB * SIZE]);
                    }
                    vbroadcastss(
                            ymm2, ptr[BO2 + LDB * 2 + (i - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg05 : reg17;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg11 : reg23;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }
            if (isTransB) {
                prefetcht0(ptr[BO1 + BO2]);
                add(BO1, LDB);
            }

            if (i == 0) {
                if (unroll_m >= 4) {
                    if (!isDirect) {
                        prefetcht0(ptr[AO1 + (PREFETCHSIZEA + 2 * 8) * SIZE]);
                    } else {
                        prefetcht0(ptr[AO1 + LDA4]);
                    }
                }
            }
            if (i == 1 || i == 2) {
                if (unroll_m >= 8) {
                    if (!isDirect) {
                        prefetcht0(ptr[AO1
                                + (PREFETCHSIZEA + (2 + 2 * i) * 8) * SIZE]);
                    } else {
                        prefetcht0(ptr[AO1 + LDA4]);
                    }
                }
            }
            if (i == 3) {
                if (!isTransB) {
                    sub(BO1, -4 * SIZE);
                    if (unroll_n >= 4) { sub(BO2, -4 * SIZE); }
                }
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    vmovups(ymm0,
                            ptr[AO1
                                    + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                                            * SIZE]);
                } else {
                    vmaskmovps(ymm0, VMASK,
                            ptr[AO1
                                    + (unroll_m * (i + 1) + 0 * 8 - OFFSET)
                                            * SIZE]);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm1,
                                ptr[AO1
                                        + (unroll_m * (i + 1) + 1 * 8 - OFFSET)
                                                * SIZE]);
                    } else {
                        vmaskmovps(ymm1, VMASK,
                                ptr[AO1
                                        + (unroll_m * (i + 1) + 1 * 8 - OFFSET)
                                                * SIZE]);
                    }
                }
            }
        }

        if (!isDirect) { sub(AO1, -unroll_m * 4 * SIZE); }
    }

    // Inner kernel with k=2
    void innerkernel2(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const Ymm &reg00, const Ymm &reg01, const Ymm &reg02,
            const Ymm &reg03, const Ymm &reg04, const Ymm &reg05,
            const Ymm &reg06, const Ymm &reg07, const Ymm &reg08,
            const Ymm &reg09, const Ymm &reg10, const Ymm &reg11,
            const Ymm &reg12, const Ymm &reg13, const Ymm &reg14,
            const Ymm &reg15, const Ymm &reg16, const Ymm &reg17,
            const Ymm &reg18, const Ymm &reg19, const Ymm &reg20,
            const Ymm &reg21, const Ymm &reg22, const Ymm &reg23) {
        Ymm fmareg;

        for (int i = 0; i < 2; i++) {
            if (isDirect) {
                if (isLoad1Unmasked) {
                    vmovups(ymm0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm0, VMASK, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(ymm1, VMASK,
                                ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                    }
                }
                add(AO1, LDA);
            }

            vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
            fmareg = (i % 2 == 0) ? reg00 : reg12;
            fma(useFma, ymm0, ymm2, fmareg);
            if (unroll_m >= 16) {
                fmareg = (i % 2 == 0) ? reg06 : reg18;
                fma(useFma, ymm1, ymm2, fmareg);
            }
            if (unroll_n >= 2) {
                if (!isTransB) {
                    vbroadcastss(
                            ymm2, ptr[BO1 + LDB * 1 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (1 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg01 : reg13;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg07 : reg19;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (unroll_n >= 3) {
                if (!isTransB) {
                    vbroadcastss(
                            ymm2, ptr[BO1 + LDB * 2 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (2 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg02 : reg14;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg08 : reg20;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (unroll_n >= 4) {
                if (!isTransB) {
                    vbroadcastss(ymm2, ptr[BO2 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (3 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg03 : reg15;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg09 : reg21;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (unroll_n >= 5) {
                if (!isTransB) {
                    vbroadcastss(
                            ymm2, ptr[BO2 + LDB * 1 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (4 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg04 : reg16;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg10 : reg22;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (unroll_n >= 6) {
                if (!isTransB) {
                    vbroadcastss(
                            ymm2, ptr[BO2 + LDB * 2 + (0 - OFFSET) * SIZE]);
                } else {
                    vbroadcastss(ymm2, ptr[BO1 + (5 - OFFSET) * SIZE]);
                }
                fmareg = (i % 2 == 0) ? reg05 : reg17;
                fma(useFma, ymm0, ymm2, fmareg);
                if (unroll_m >= 16) {
                    fmareg = (i % 2 == 0) ? reg11 : reg23;
                    fma(useFma, ymm1, ymm2, fmareg);
                }
            }

            if (isCopy) {
                vmovups(ptr[LDA4 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE],
                        ymm0);
                if (unroll_m >= 16) {
                    vmovups(ptr[LDA4 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE],
                            ymm1);
                }
                sub(LDA4, -unroll_m * SIZE);
            }

            if (!isDirect) {
                if (isLoad1Unmasked) {
                    vmovups(ymm0,
                            ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm0, VMASK,
                            ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
                }
                if (unroll_m >= 16) {
                    if (isLoad2Unmasked) {
                        vmovups(ymm1,
                                ptr[AO1
                                        + (unroll_m * 1 + 1 * 8 - OFFSET)
                                                * SIZE]);
                    } else {
                        vmaskmovps(ymm1, VMASK,
                                ptr[AO1
                                        + (unroll_m * 1 + 1 * 8 - OFFSET)
                                                * SIZE]);
                    }
                }
                sub(AO1, -unroll_m * SIZE);
            }

            if (!isTransB) {
                sub(BO1, -SIZE);
                if (unroll_n >= 4) { sub(BO2, -SIZE); }
            } else {
                add(BO1, LDB);
            }
        }
    }

    // Inner kernel with k=1
    void innerkernel1(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const Ymm &reg00, const Ymm &reg01, const Ymm &reg02,
            const Ymm &reg03, const Ymm &reg04, const Ymm &reg05,
            const Ymm &reg06, const Ymm &reg07, const Ymm &reg08,
            const Ymm &reg09, const Ymm &reg10, const Ymm &reg11) {
        if (isDirect) {
            if (isLoad1Unmasked) {
                vmovups(ymm0, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
            } else {
                vmaskmovps(ymm0, VMASK, ptr[AO1 + (0 * 8 - OFFSET) * SIZE]);
            }
            if (unroll_m >= 16) {
                if (isLoad2Unmasked) {
                    vmovups(ymm1, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm1, VMASK, ptr[AO1 + (1 * 8 - OFFSET) * SIZE]);
                }
            }
            add(AO1, LDA);
        }

        vbroadcastss(ymm2, ptr[BO1 + (0 - OFFSET) * SIZE]);
        fma(useFma, ymm0, ymm2, reg00);
        if (unroll_m >= 16) { fma(useFma, ymm1, ymm2, reg06); }

        if (unroll_n >= 2) {
            if (!isTransB) {
                vbroadcastss(ymm2, ptr[BO1 + LDB * 1 + (0 - OFFSET) * SIZE]);
            } else {
                vbroadcastss(ymm2, ptr[BO1 + (1 - OFFSET) * SIZE]);
            }
            fma(useFma, ymm0, ymm2, reg01);
            if (unroll_m >= 16) { fma(useFma, ymm1, ymm2, reg07); }
        }

        if (unroll_n >= 3) {
            if (!isTransB) {
                vbroadcastss(ymm2, ptr[BO1 + LDB * 2 + (0 - OFFSET) * SIZE]);
            } else {
                vbroadcastss(ymm2, ptr[BO1 + (2 - OFFSET) * SIZE]);
            }
            fma(useFma, ymm0, ymm2, reg02);
            if (unroll_m >= 16) { fma(useFma, ymm1, ymm2, reg08); }
        }

        if (unroll_n >= 4) {
            if (!isTransB) {
                vbroadcastss(ymm2, ptr[BO2 + (0 - OFFSET) * SIZE]);
            } else {
                vbroadcastss(ymm2, ptr[BO1 + (3 - OFFSET) * SIZE]);
            }
            fma(useFma, ymm0, ymm2, reg03);
            if (unroll_m >= 16) { fma(useFma, ymm1, ymm2, reg09); }
        }

        if (unroll_n >= 5) {
            if (!isTransB) {
                vbroadcastss(ymm2, ptr[BO2 + LDB * 1 + (0 - OFFSET) * SIZE]);
            } else {
                vbroadcastss(ymm2, ptr[BO1 + (4 - OFFSET) * SIZE]);
            }
            fma(useFma, ymm0, ymm2, reg04);
            if (unroll_m >= 16) { fma(useFma, ymm1, ymm2, reg10); }
        }

        if (unroll_n >= 6) {
            if (!isTransB) {
                vbroadcastss(ymm2, ptr[BO2 + LDB * 2 + (0 - OFFSET) * SIZE]);
            } else {
                vbroadcastss(ymm2, ptr[BO1 + (5 - OFFSET) * SIZE]);
            }
            fma(useFma, ymm0, ymm2, reg05);
            if (unroll_m >= 16) { fma(useFma, ymm1, ymm2, reg11); }
        }

        if (isCopy) {
            vmovups(ptr[LDA4 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE], ymm0);
            if (unroll_m >= 16) {
                vmovups(ptr[LDA4 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE],
                        ymm1);
            }
            sub(LDA4, -unroll_m * SIZE);
        }

        if (!isDirect) {
            if (isLoad1Unmasked) {
                vmovups(ymm0,
                        ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
            } else {
                vmaskmovps(ymm0, VMASK,
                        ptr[AO1 + (unroll_m * 1 + 0 * 8 - OFFSET) * SIZE]);
            }
            if (unroll_m >= 16) {
                if (isLoad2Unmasked) {
                    vmovups(ymm1,
                            ptr[AO1 + (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm1, VMASK,
                            ptr[AO1 + (unroll_m * 1 + 1 * 8 - OFFSET) * SIZE]);
                }
            }
            sub(AO1, -unroll_m * SIZE);
        }

        if (!isTransB) {
            sub(BO1, -SIZE);
            if (unroll_n >= 4) { sub(BO2, -SIZE); }
        } else {
            add(BO1, LDB);
        }
    }

    // Main kernel; does prefetching and calls innerkernel{1,2,4,8} as
    // appropriate
    // After calculating results in registers, writes back to C matrix
    void kernel(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy, bool useFma,
            const Ymm &reg00, const Ymm &reg01, const Ymm &reg02,
            const Ymm &reg03, const Ymm &reg04, const Ymm &reg05,
            const Ymm &reg06, const Ymm &reg07, const Ymm &reg08,
            const Ymm &reg09, const Ymm &reg10, const Ymm &reg11,
            const Ymm &reg12, const Ymm &reg13, const Ymm &reg14,
            const Ymm &reg15, const Ymm &reg16, const Ymm &reg17,
            const Ymm &reg18, const Ymm &reg19, const Ymm &reg20,
            const Ymm &reg21, const Ymm &reg22, const Ymm &reg23) {
        if (!isDirect) {
            lea(AO1, ptr[rsp + 256 + OFFSET * SIZE]);
        } else {
            mov(AO1, A);
        }

        if (isCopy) {
            lea(LDA4, ptr[rsp + 256 + OFFSET * SIZE]);
        } else {
            lea(LDA4, ptr[LDA * 8 + (8 - 1 - OFFSET) * SIZE]);
        }

        if (isTransB) {
            lea(BO2, ptr[LDB * 4 + (8 - 1 - OFFSET) * SIZE]);
            lea(BO2, ptr[BO2 + LDB * 2]);
        }

        if (!isDirect) {
            if (isLoad1Unmasked) {
                vmovups(ymm0,
                        ptr[AO1 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE]);
            } else {
                vmaskmovps(ymm0, VMASK,
                        ptr[AO1 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE]);
            }
            if (unroll_m >= 16) {
                if (isLoad2Unmasked) {
                    vmovups(ymm1,
                            ptr[AO1 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm1, VMASK,
                            ptr[AO1 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE]);
                }
            }
        }

        for (int i = 4; i < 10; i++) {
            vxorps(Ymm(i), Ymm(i), Ymm(i));
            vxorps(Ymm(i + 6), Ymm(i + 6), Ymm(i + 6));
        }

        mov(LL, K);
        sar(LL, 3);

        std::vector<Label> labels(8);

        sub(LL, SECOND_FETCH);
        jle(labels[1], T_NEAR);
        align(16);

        L(labels[0]);
        innerkernel8(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);
        jg(labels[0], T_NEAR);
        align(16);

        L(labels[1]);
        prefetcht0(ptr[CO1 + (unroll_m - 1) * SIZE]);
        if (unroll_n >= 2) prefetcht0(ptr[CO1 + LDC + (unroll_m - 1) * SIZE]);
        if (unroll_n >= 3)
            prefetcht0(ptr[CO1 + LDC * 2 + (unroll_m - 1) * SIZE]);
        if (unroll_n >= 4) prefetcht0(ptr[CO2 + (unroll_m - 1) * SIZE]);
        if (unroll_n >= 5) prefetcht0(ptr[CO2 + LDC + (unroll_m - 1) * SIZE]);
        if (unroll_n >= 6)
            prefetcht0(ptr[CO2 + LDC * 2 + (unroll_m - 1) * SIZE]);

        add(LL, SECOND_FETCH);
        jle(labels[3], T_NEAR);
        align(16);

        L(labels[2]);
        innerkernel8(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);
        jg(labels[2], T_NEAR);
        align(16);

        L(labels[3]);
        test(K, 4);
        jle(labels[4], T_NEAR);
        innerkernel4(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);

        L(labels[4]);
        test(K, 2);
        jle(labels[5], T_NEAR);
        innerkernel2(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11, reg12, reg13,
                reg14, reg15, reg16, reg17, reg18, reg19, reg20, reg21, reg22,
                reg23);
        align(16);

        L(labels[5]);
        if (unroll_m == 16) {
            if (unroll_n <= 3) {
                vaddps(reg00, reg00, reg12);
                vaddps(reg01, reg01, reg13);
                vaddps(reg02, reg02, reg14);
                vaddps(reg06, reg06, reg18);
                vaddps(reg07, reg07, reg19);
                vaddps(reg08, reg08, reg20);
            }
        }

        if (unroll_m <= 8) {
            vaddps(reg00, reg00, reg12);
            vaddps(reg01, reg01, reg13);
            vaddps(reg02, reg02, reg14);
            vaddps(reg03, reg03, reg15);
            vaddps(reg04, reg04, reg16);
            vaddps(reg05, reg05, reg17);
        }

        test(K, 1);
        jle(labels[6], T_NEAR);
        innerkernel1(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, useFma, reg00, reg01, reg02, reg03, reg04,
                reg05, reg06, reg07, reg08, reg09, reg10, reg11);
        align(16);

        L(labels[6]);
        vbroadcastss(VALPHA, ALPHA);

        if (isBetaN) { vbroadcastss(VBETA, BETA); }

        // Write back the results; all beta and bias cases need to be
        // handled
        switch (unroll_n) {
            case 1: mov(rax, LDC); break;
            case 2: lea(rax, ptr[LDC * 2]); break;
            case 3: lea(rax, ptr[LDC + LDC * 2]); break;
            case 4: lea(rax, ptr[LDC + LDC * 4]); break;
            case 5:
                lea(rax, ptr[LDC * 4]);
                add(rax, LDC);
                break;
            case 6:
                lea(rax, ptr[LDC + LDC * 2]);
                add(rax, rax);
                break;
        }

        if (hasBias) {
            mov(BIAS1, BIAS);
            if (isLoad1Unmasked) {
                vmovups(VBIAS1, ptr[BIAS1 + 0 * SIZE]);
            } else {
                vmaskmovps(VBIAS1, VMASK, ptr[BIAS1 + 0 * SIZE]);
            }
        }

        for (int i = 0; i < unroll_n; i++) {
            vmulps(Ymm(i + 4), Ymm(i + 4), VALPHA);
            if (!isBeta0) {
                if (isLoad1Unmasked) {
                    switch (i) {
                        case 0: vmovups(ymm0, ptr[CO1 + 0 * SIZE]); break;
                        case 1: vmovups(ymm0, ptr[CO1 + LDC + 0 * SIZE]); break;
                        case 2:
                            vmovups(ymm0, ptr[CO1 + LDC * 2 + 0 * SIZE]);
                            break;
                        case 3: vmovups(ymm0, ptr[CO2 + 0 * SIZE]); break;
                        case 4: vmovups(ymm0, ptr[CO2 + LDC + 0 * SIZE]); break;
                        case 5:
                            vmovups(ymm0, ptr[CO2 + LDC * 2 + 0 * SIZE]);
                            break;
                    }
                } else {
                    switch (i) {
                        case 0:
                            vmaskmovps(ymm0, VMASK, ptr[CO1 + 0 * SIZE]);
                            break;
                        case 1:
                            vmaskmovps(ymm0, VMASK, ptr[CO1 + LDC + 0 * SIZE]);
                            break;
                        case 2:
                            vmaskmovps(
                                    ymm0, VMASK, ptr[CO1 + LDC * 2 + 0 * SIZE]);
                            break;
                        case 3:
                            vmaskmovps(ymm0, VMASK, ptr[CO2 + 0 * SIZE]);
                            break;
                        case 4:
                            vmaskmovps(ymm0, VMASK, ptr[CO2 + LDC + 0 * SIZE]);
                            break;
                        case 5:
                            vmaskmovps(
                                    ymm0, VMASK, ptr[CO2 + LDC * 2 + 0 * SIZE]);
                            break;
                    }
                }

                if (!isBetaN) {
                    vaddps(Ymm(i + 4), ymm0, Ymm(i + 4));
                } else {
                    fma(useFma, VBETA, ymm0, Ymm(i + 4), true);
                }
            }
            if (hasBias) { vaddps(Ymm(i + 4), VBIAS1, Ymm(i + 4)); }
            if (isLoad1Unmasked) {
                switch (i) {
                    case 0: vmovups(ptr[CO1 + 0 * SIZE], Ymm(i + 4)); break;
                    case 1:
                        vmovups(ptr[CO1 + LDC + 0 * SIZE], Ymm(i + 4));
                        break;
                    case 2:
                        vmovups(ptr[CO1 + LDC * 2 + 0 * SIZE], Ymm(i + 4));
                        break;
                    case 3: vmovups(ptr[CO2 + 0 * SIZE], Ymm(i + 4)); break;
                    case 4:
                        vmovups(ptr[CO2 + LDC + 0 * SIZE], Ymm(i + 4));
                        break;
                    case 5:
                        vmovups(ptr[CO2 + LDC * 2 + 0 * SIZE], Ymm(i + 4));
                        break;
                }
            } else {
                switch (i) {
                    case 0:
                        vmaskmovps(ptr[CO1 + 0 * SIZE], VMASK, Ymm(i + 4));
                        break;
                    case 1:
                        vmaskmovps(
                                ptr[CO1 + LDC + 0 * SIZE], VMASK, Ymm(i + 4));
                        break;
                    case 2:
                        vmaskmovps(ptr[CO1 + LDC * 2 + 0 * SIZE], VMASK,
                                Ymm(i + 4));
                        break;
                    case 3:
                        vmaskmovps(ptr[CO2 + 0 * SIZE], VMASK, Ymm(i + 4));
                        break;
                    case 4:
                        vmaskmovps(
                                ptr[CO2 + LDC + 0 * SIZE], VMASK, Ymm(i + 4));
                        break;
                    case 5:
                        vmaskmovps(ptr[CO2 + LDC * 2 + 0 * SIZE], VMASK,
                                Ymm(i + 4));
                        break;
                }
            }

            if (unroll_m >= 16) {
                // Re-use ymm4 (VBIAS2)
                if (i == 0) {
                    if (hasBias) {
                        if (isLoad1Unmasked) {
                            vmovups(VBIAS2, ptr[BIAS1 + 8 * SIZE]);
                        } else {
                            vmaskmovps(VBIAS2, VMASK, ptr[BIAS1 + 8 * SIZE]);
                        }
                    }
                }
                vmulps(Ymm(i + 10), Ymm(i + 10), VALPHA);
                if (!isBeta0) {
                    if (isLoad2Unmasked) {
                        switch (i) {
                            case 0: vmovups(ymm0, ptr[CO1 + 8 * SIZE]); break;
                            case 1:
                                vmovups(ymm0, ptr[CO1 + LDC + 8 * SIZE]);
                                break;
                            case 2:
                                vmovups(ymm0, ptr[CO1 + LDC * 2 + 8 * SIZE]);
                                break;
                            case 3: vmovups(ymm0, ptr[CO2 + 8 * SIZE]); break;
                            case 4:
                                vmovups(ymm0, ptr[CO2 + LDC + 8 * SIZE]);
                                break;
                            case 5:
                                vmovups(ymm0, ptr[CO2 + LDC * 2 + 8 * SIZE]);
                                break;
                        }
                    } else {
                        switch (i) {
                            case 0:
                                vmaskmovps(ymm0, VMASK, ptr[CO1 + 8 * SIZE]);
                                break;
                            case 1:
                                vmaskmovps(
                                        ymm0, VMASK, ptr[CO1 + LDC + 8 * SIZE]);
                                break;
                            case 2:
                                vmaskmovps(ymm0, VMASK,
                                        ptr[CO1 + LDC * 2 + 8 * SIZE]);
                                break;
                            case 3:
                                vmaskmovps(ymm0, VMASK, ptr[CO2 + 8 * SIZE]);
                                break;
                            case 4:
                                vmaskmovps(
                                        ymm0, VMASK, ptr[CO2 + LDC + 8 * SIZE]);
                                break;
                            case 5:
                                vmaskmovps(ymm0, VMASK,
                                        ptr[CO2 + LDC * 2 + 8 * SIZE]);
                                break;
                        }
                    }
                    if (!isBetaN) {
                        vaddps(Ymm(i + 10), ymm0, Ymm(i + 10));
                    } else {
                        fma(useFma, VBETA, ymm0, Ymm(i + 10), true);
                    }
                }
                if (hasBias) { vaddps(Ymm(i + 10), VBIAS2, Ymm(i + 10)); }
                if (isLoad2Unmasked) {
                    switch (i) {
                        case 0:
                            vmovups(ptr[CO1 + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 1:
                            vmovups(ptr[CO1 + LDC + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 2:
                            vmovups(ptr[CO1 + LDC * 2 + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 3:
                            vmovups(ptr[CO2 + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 4:
                            vmovups(ptr[CO2 + LDC + 8 * SIZE], Ymm(i + 10));
                            break;
                        case 5:
                            vmovups(ptr[CO2 + LDC * 2 + 8 * SIZE], Ymm(i + 10));
                            break;
                    }
                } else {
                    switch (i) {
                        case 0:
                            vmaskmovps(ptr[CO1 + 8 * SIZE], VMASK, Ymm(i + 10));
                            break;
                        case 1:
                            vmaskmovps(ptr[CO1 + LDC + 8 * SIZE], VMASK,
                                    Ymm(i + 10));
                            break;
                        case 2:
                            vmaskmovps(ptr[CO1 + LDC * 2 + 8 * SIZE], VMASK,
                                    Ymm(i + 10));
                            break;
                        case 3:
                            vmaskmovps(ptr[CO2 + 8 * SIZE], VMASK, Ymm(i + 10));
                            break;
                        case 4:
                            vmaskmovps(ptr[CO2 + LDC + 8 * SIZE], VMASK,
                                    Ymm(i + 10));
                            break;
                        case 5:
                            vmaskmovps(ptr[CO2 + LDC * 2 + 8 * SIZE], VMASK,
                                    Ymm(i + 10));
                            break;
                    }
                }
            }
            if (i == 2) add(CO1, rax);
        }
        if (unroll_n >= 4) { add(CO2, rax); }

        // Compute next address of B
        if (!isTransB) {
            lea(rax, ptr[K * SIZE]);
            switch (unroll_n) {
                case 1:
                    add(BO1, LDB);
                    add(BO2, LDB);
                    break;
                case 2:
                    lea(BO1, ptr[BO1 + LDB * 2]);
                    lea(BO2, ptr[BO2 + LDB * 2]);
                    break;
                case 3:
                    lea(BO1, ptr[BO1 + LDB3]);
                    lea(BO2, ptr[BO2 + LDB3]);
                    break;
                case 4:
                    lea(BO1, ptr[BO1 + LDB * 4]);
                    lea(BO2, ptr[BO2 + LDB * 4]);
                    break;
                case 5:
                    lea(BO1, ptr[BO1 + LDB * 4]);
                    add(BO1, LDB);
                    lea(BO2, ptr[BO2 + LDB * 4]);
                    add(BO2, LDB);
                    break;
                case 6:
                    lea(BO1, ptr[BO1 + LDB3 * 2]);
                    lea(BO2, ptr[BO2 + LDB3 * 2]);
                    break;
            }
            sub(BO1, rax);
            sub(BO2, rax);
        } else {
            mov(rax, LDB);
            imul(rax, K);
            sub(BO1, rax);
            add(BO1, unroll_n * SIZE);
        }
    }

    void kernel_16x6(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, true, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11,
                ymm12, ymm13, ymm14, ymm15, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9,
                ymm10, ymm11, ymm12, ymm13, ymm14, ymm15);
    }

    void kernel_16x5(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, true, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11,
                ymm12, ymm13, ymm14, ymm15, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9,
                ymm10, ymm11, ymm12, ymm13, ymm14, ymm15);
    }

    void kernel_16x4(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, true, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11,
                ymm12, ymm13, ymm14, ymm15, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9,
                ymm10, ymm11, ymm12, ymm13, ymm14, ymm15);
    }

    void kernel_16x3(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy,
            bool useFma = true) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, useFma, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10,
                ymm11, ymm12, ymm13, ymm14, ymm15, ymm7, ymm8, ymm9, ymm7, ymm8,
                ymm9, ymm13, ymm14, ymm15, ymm13, ymm14, ymm15);
    }

    void kernel_16x2(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_16x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    void kernel_16x1(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_16x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    void kernel_8x6(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy,
            bool useFma = true) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, useFma, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10,
                ymm11, ymm12, ymm13, ymm14, ymm15, ymm10, ymm11, ymm12, ymm13,
                ymm14, ymm15, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15);
    }

    void kernel_8x5(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x6(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy);
    }

    void kernel_8x4(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x6(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy);
    }

    void kernel_8x3(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy,
            bool useFma = true) {
        kernel(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked, isDirect,
                isCopy, useFma, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10,
                ymm11, ymm12, ymm13, ymm14, ymm15, ymm7, ymm8, ymm9, ymm7, ymm8,
                ymm9, ymm13, ymm14, ymm15, ymm13, ymm14, ymm15);
    }

    void kernel_8x2(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    void kernel_8x1(int unroll_m, int unroll_n, bool isLoad1Unmasked,
            bool isLoad2Unmasked, bool isDirect, bool isCopy) {
        kernel_8x3(unroll_m, unroll_n, isLoad1Unmasked, isLoad2Unmasked,
                isDirect, isCopy, false);
    }

    // Function for packing if needed
    void do_pack(int unroll_m, bool isLoad1Unmasked, bool isLoad2Unmasked) {
        std::vector<Label> labels(6);

        int regIdx;
        Reg64 reg;

        mov(BO1, A);
        lea(AO1, ptr[rsp + 256 + OFFSET * SIZE]);

        if (isTransA) {
            lea(BO2, ptr[BO1 + LDA * 4]);
            lea(CO1, ptr[LDA + LDA * 2]);
            vmovupd(ymm7, STRIDE);
        }

        mov(LL, K);
        sar(LL, 2);
        jle(labels[1], T_NEAR);
        align(16);

        L(labels[0]);
        if (!isTransA) {
            for (int i = 0; i < 4; i++) {
                regIdx = (i % 2 == 0) ? 4 : 6;
                if (isLoad1Unmasked) {
                    vmovups(Ymm(regIdx), ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(Ymm(regIdx), VMASK,
                            ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
                }
                if (unroll_m > 8) {
                    if (isLoad2Unmasked) {
                        vmovups(Ymm(regIdx + 1),
                                ptr[BO1 + (1 * 8 - OFFSET) * SIZE]);
                    } else {
                        vmaskmovps(Ymm(regIdx + 1), VMASK,
                                ptr[BO1 + (1 * 8 - OFFSET) * SIZE]);
                    }
                }
                add(BO1, LDA);

                vmovups(ptr[AO1 + (unroll_m * i + 0 * 8 - OFFSET) * SIZE],
                        Ymm(regIdx));
                if (unroll_m > 8) {
                    vmovups(ptr[AO1 + (unroll_m * i + 1 * 8 - OFFSET) * SIZE],
                            Ymm(regIdx + 1));
                }
            }

        } else {
            if (isLoad1Unmasked) {
                for (int i = 0; i < 2; i++) {
                    reg = (i % 2 == 0) ? BO1 : BO2;
                    vmovups(xmm0, ptr[reg + (0 * 8 - OFFSET) * SIZE]);
                    vmovups(xmm1, ptr[reg + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                    lea(BO2, ptr[reg + LDA * 2]);
                    vunpcklps(xmm4, xmm0, xmm1);
                    vunpckhps(xmm5, xmm0, xmm1);
                    vmovups(xmm0, ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                    vmovups(xmm1, ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                    lea(BO2, ptr[BO2 + LDA * 2]);
                    vunpcklps(xmm6, xmm0, xmm1);
                    vunpckhps(xmm2, xmm0, xmm1);

                    vunpcklpd(xmm0, xmm4, xmm6);
                    vunpckhpd(xmm1, xmm4, xmm6);
                    vmovups(ptr[AO1 + (unroll_m * 0 + i * 4 - OFFSET) * SIZE],
                            xmm0);
                    vmovups(ptr[AO1 + (unroll_m * 1 + i * 4 - OFFSET) * SIZE],
                            xmm1);
                    vunpcklpd(xmm0, xmm5, xmm2);
                    vunpckhpd(xmm1, xmm5, xmm2);
                    vmovups(ptr[AO1 + (unroll_m * 2 + i * 4 - OFFSET) * SIZE],
                            xmm0);
                    vmovups(ptr[AO1 + (unroll_m * 3 + i * 4 - OFFSET) * SIZE],
                            xmm1);
                }
            } else if (is_avx2) {
                for (int i = 0; i < 2; i++) {
                    vmovaps(xmm4, xmm3);
                    vgatherqps(xmm0,
                            ptr[BO1 + ymm7 + ((2 * i) - OFFSET) * SIZE], xmm4);
                    vmovaps(xmm4, xmm3);
                    vgatherqps(xmm1,
                            ptr[BO1 + ymm7 + ((2 * i + 1) - OFFSET) * SIZE],
                            xmm4);

                    vmovups(ptr[AO1
                                    + (unroll_m * (2 * i) + 0 * 4 - OFFSET)
                                            * SIZE],
                            xmm0);
                    vmovups(ptr[AO1
                                    + (unroll_m * (2 * i + 1) + 0 * 4 - OFFSET)
                                            * SIZE],
                            xmm1);
                }

                lea(BO2, ptr[BO1 + LDA * 4]);

                for (int i = 0; i < 2; i++) {
                    vextractf128(xmm4, ymm3, 1);
                    vgatherqps(xmm0,
                            ptr[BO2 + ymm7 + ((2 * i) - OFFSET) * SIZE], xmm4);
                    vextractf128(xmm4, ymm3, 1);
                    vgatherqps(xmm1,
                            ptr[BO2 + ymm7 + ((2 * i + 1) - OFFSET) * SIZE],
                            xmm4);

                    vmovups(ptr[AO1
                                    + (unroll_m * (2 * i) + 1 * 4 - OFFSET)
                                            * SIZE],
                            xmm0);
                    vmovups(ptr[AO1
                                    + (unroll_m * (2 * i + 1) + 1 * 4 - OFFSET)
                                            * SIZE],
                            xmm1);
                }

                lea(BO2, ptr[BO2 + LDA * 4]);
            } else {
                vxorps(xmm4, xmm4, xmm4);
                lea(BO2, ptr[BO1 + LDA * 4]);

                auto el_cp = [&](int section, int ld_step) {
                    RegExp src_addr = section == 0 ? BO1 : BO2;
                    if (ld_step == 1 || ld_step == 2)
                        src_addr = src_addr + LDA * ld_step;
                    else if (ld_step == 3)
                        src_addr = src_addr + CO1;
                    src_addr = src_addr - OFFSET * SIZE;

                    vmovups(Xmm(ld_step % 2), ptr[src_addr]);
                    RegExp dst_addr
                            = AO1 + (ld_step + section * 4 - OFFSET) * SIZE;
                    for (int off = 0; off < 4; ++off)
                        pextrd(ptr[dst_addr + unroll_m * off * SIZE],
                                Xmm(ld_step % 2), off);
                };

                el_cp(0, 0);
                cmp(M, 4 * 0 + 0 + 1);
                je(labels[4], T_NEAR);
                el_cp(0, 1);
                cmp(M, 4 * 0 + 1 + 1);
                je(labels[4], T_NEAR);
                el_cp(0, 2);
                cmp(M, 4 * 0 + 2 + 1);
                je(labels[4], T_NEAR);
                el_cp(0, 3);
                cmp(M, 4 * 0 + 3 + 1);
                je(labels[4], T_NEAR);
                el_cp(1, 0);
                cmp(M, 4 * 1 + 0 + 1);
                je(labels[4], T_NEAR);
                el_cp(1, 1);
                cmp(M, 4 * 1 + 1 + 1);
                je(labels[4], T_NEAR);
                el_cp(1, 2);
                L(labels[4]);

                lea(BO2, ptr[BO2 + LDA * 4]);
            }

            if (unroll_m >= 16) {
                assert(is_avx2);
                if (isLoad2Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        vmovups(xmm0, ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        vmovups(xmm1,
                                ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        lea(BO2, ptr[BO2 + LDA * 2]);
                        vunpcklps(xmm4, xmm0, xmm1);
                        vunpckhps(xmm5, xmm0, xmm1);
                        vmovups(xmm0, ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        vmovups(xmm1,
                                ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        if (i == 0) lea(BO2, ptr[BO2 + LDA * 2]);
                        vunpcklps(xmm6, xmm0, xmm1);
                        vunpckhps(xmm2, xmm0, xmm1);

                        vunpcklpd(xmm0, xmm4, xmm6);
                        vunpckhpd(xmm1, xmm4, xmm6);
                        vmovups(ptr[AO1
                                        + (unroll_m * 0 + (i + 2) * 4 - OFFSET)
                                                * SIZE],
                                xmm0);
                        vmovups(ptr[AO1
                                        + (unroll_m * 1 + (i + 2) * 4 - OFFSET)
                                                * SIZE],
                                xmm1);
                        vunpcklpd(xmm0, xmm5, xmm2);
                        vunpckhpd(xmm1, xmm5, xmm2);
                        vmovups(ptr[AO1
                                        + (unroll_m * 2 + (i + 2) * 4 - OFFSET)
                                                * SIZE],
                                xmm0);
                        vmovups(ptr[AO1
                                        + (unroll_m * 3 + (i + 2) * 4 - OFFSET)
                                                * SIZE],
                                xmm1);
                    }
                } else {
                    for (int i = 0; i < 2; i++) {
                        vmovaps(xmm4, xmm3);
                        vgatherqps(xmm0,
                                ptr[BO2 + ymm7 + ((2 * i) - OFFSET) * SIZE],
                                xmm4);
                        vmovaps(xmm4, xmm3);
                        vgatherqps(xmm1,
                                ptr[BO2 + ymm7 + ((2 * i + 1) - OFFSET) * SIZE],
                                xmm4);

                        vmovups(ptr[AO1
                                        + (unroll_m * (2 * i) + 2 * 4 - OFFSET)
                                                * SIZE],
                                xmm0);
                        vmovups(ptr[AO1
                                        + (unroll_m * (2 * i + 1) + 2 * 4
                                                  - OFFSET)
                                                * SIZE],
                                xmm1);
                    }

                    lea(BO2, ptr[BO2 + LDA * 4]);

                    for (int i = 0; i < 2; i++) {
                        vextractf128(xmm4, ymm3, 1);
                        vgatherqps(xmm0,
                                ptr[BO2 + ymm7 + ((2 * i) - OFFSET) * SIZE],
                                xmm4);
                        vextractf128(xmm4, ymm3, 1);
                        vgatherqps(xmm1,
                                ptr[BO2 + ymm7 + ((2 * i + 1) - OFFSET) * SIZE],
                                xmm4);

                        vmovups(ptr[AO1
                                        + (unroll_m * (2 * i) + 3 * 4 - OFFSET)
                                                * SIZE],
                                xmm0);
                        vmovups(ptr[AO1
                                        + (unroll_m * (2 * i + 1) + 3 * 4
                                                  - OFFSET)
                                                * SIZE],
                                xmm1);
                    }

                    lea(BO2, ptr[BO2 + LDA * 4]);
                }
            }
            add(BO1, (4 * SIZE));
        }

        add(AO1, unroll_m * 4 * SIZE);
        sub(LL, 1);
        jg(labels[0], T_NEAR);
        align(16);

        L(labels[1]);
        mov(LL, K);
        and_(LL, 3);
        jle(labels[3], T_NEAR);
        align(16);

        L(labels[2]);
        if (!isTransA) {
            if (isLoad1Unmasked) {
                vmovups(ymm4, ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
            } else {
                vmaskmovps(ymm4, VMASK, ptr[BO1 + (0 * 8 - OFFSET) * SIZE]);
            }
            if (unroll_m > 8) {
                if (isLoad2Unmasked) {
                    vmovups(ymm5, ptr[BO1 + (1 * 8 - OFFSET) * SIZE]);
                } else {
                    vmaskmovps(ymm5, VMASK, ptr[BO1 + (1 + 8 - OFFSET) * SIZE]);
                }
            }
            add(BO1, LDA);
            vmovups(ptr[AO1 + (unroll_m * 0 + 0 * 8 - OFFSET) * SIZE], ymm4);
            if (unroll_m > 8) {
                vmovups(ptr[AO1 + (unroll_m * 0 + 1 * 8 - OFFSET) * SIZE],
                        ymm5);
            }
        } else {
            if (isLoad1Unmasked) {
                for (int i = 0; i < 2; i++) {
                    reg = (i % 2 == 0) ? BO1 : BO2;
                    vmovss(Xmm(i + 1), ptr[reg + (0 * 8 - OFFSET) * SIZE]);
                    vmovss(xmm0, ptr[reg + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                    lea(BO2, ptr[reg + LDA * 2]);
                    vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                }
                vunpcklpd(xmm1, xmm1, xmm2);
                vmovups(ptr[AO1 + (unroll_m * 0 + 0 * 4 - OFFSET) * SIZE],
                        xmm1);

                for (int i = 0; i < 2; i++) {
                    vmovss(Xmm(i + 1), ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                    vmovss(xmm0, ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                    lea(BO2, ptr[BO2 + LDA * 2]);
                    vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                }
                vunpcklpd(xmm1, xmm1, xmm2);
                vmovups(ptr[AO1 + (unroll_m * 0 + 1 * 4 - OFFSET) * SIZE],
                        xmm1);
            } else if (is_avx2) {
                vmovaps(xmm4, xmm3);
                vgatherqps(
                        xmm1, ptr[BO1 + ymm7 + (0 * 8 - OFFSET) * SIZE], xmm4);
                lea(BO2, ptr[BO1 + LDA * 4]);
                vmovups(ptr[AO1 + (unroll_m * 0 + 0 * 4 - OFFSET) * SIZE],
                        xmm1);

                vextractf128(xmm4, ymm3, 1);
                vgatherqps(
                        xmm1, ptr[BO2 + ymm7 + (0 * 8 - OFFSET) * SIZE], xmm4);
                lea(BO2, ptr[BO2 + LDA * 4]);
                vmovups(ptr[AO1 + (unroll_m * 0 + 1 * 4 - OFFSET) * SIZE],
                        xmm1);
            } else {
                vxorps(xmm4, xmm4, xmm4);
                lea(BO2, ptr[BO1 + LDA * 4]);

                auto el_cp = [&](int section, int ld_step) {
                    RegExp src_addr = section == 0 ? BO1 : BO2;
                    if (ld_step == 1 || ld_step == 2)
                        src_addr = src_addr + LDA * ld_step;
                    else if (ld_step == 3)
                        src_addr = src_addr + CO1;
                    src_addr = src_addr - OFFSET * SIZE;

                    vmovss(xmm1, ptr[src_addr]);
                    RegExp dst_addr
                            = AO1 + (ld_step + section * 4 - OFFSET) * SIZE;
                    movss(ptr[dst_addr], xmm1);
                };

                el_cp(0, 0);
                cmp(M, 4 * 0 + 0 + 1);
                je(labels[5], T_NEAR);
                el_cp(0, 1);
                cmp(M, 4 * 0 + 1 + 1);
                je(labels[5], T_NEAR);
                el_cp(0, 2);
                cmp(M, 4 * 0 + 2 + 1);
                je(labels[5], T_NEAR);
                el_cp(0, 3);
                cmp(M, 4 * 0 + 3 + 1);
                je(labels[5], T_NEAR);
                el_cp(1, 0);
                cmp(M, 4 * 1 + 0 + 1);
                je(labels[5], T_NEAR);
                el_cp(1, 1);
                cmp(M, 4 * 1 + 1 + 1);
                je(labels[5], T_NEAR);
                el_cp(1, 2);
                L(labels[5]);

                lea(BO2, ptr[BO2 + LDA * 4]);
            }

            if (unroll_m >= 16) {
                assert(is_avx2);
                if (isLoad2Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        vmovss(Xmm(i + 1), ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        vmovss(xmm0,
                                ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        lea(BO2, ptr[BO2 + LDA * 2]);
                        vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                    }
                    vunpcklpd(xmm1, xmm1, xmm2);
                } else {
                    vmovaps(xmm4, xmm3);
                    vgatherqps(xmm1, ptr[BO2 + ymm7 + (0 * 8 - OFFSET) * SIZE],
                            xmm4);
                    lea(BO2, ptr[BO2 + LDA * 4]);
                }
                vmovups(ptr[AO1 + (unroll_m * 0 + 2 * 4 - OFFSET) * SIZE],
                        xmm1);

                if (isLoad2Unmasked) {
                    for (int i = 0; i < 2; i++) {
                        vmovss(Xmm(i + 1), ptr[BO2 + (0 * 8 - OFFSET) * SIZE]);
                        vmovss(xmm0,
                                ptr[BO2 + LDA * 1 + (0 * 8 - OFFSET) * SIZE]);
                        lea(BO2, ptr[BO2 + LDA * 2]);
                        vunpcklps(Xmm(i + 1), Xmm(i + 1), Xmm(0));
                    }
                    vunpcklpd(xmm1, xmm1, xmm2);
                } else {
                    vextractf128(xmm4, ymm3, 1);
                    vgatherqps(xmm1, ptr[BO2 + ymm7 + (0 * 8 - OFFSET) * SIZE],
                            xmm4);
                }
                vmovups(ptr[AO1 + (unroll_m * 0 + 3 * 4 - OFFSET) * SIZE],
                        xmm1);
            }
            add(BO1, SIZE);
        }

        add(AO1, unroll_m * SIZE);
        sub(LL, 1);
        jg(labels[2], T_NEAR);
        align(16);

        L(labels[3]);
    }

    // High-level subroutine; does packing if needed, then splits C matrix.
    // Operates on chunks of 16 rows, 6 columns at a time (handling tail
    // cases appropriately).
    // Masking is used for tail cases where M is not divisible by 8.
    void subloop(int unroll_m, bool isLoad1Unmasked, bool isLoad2Unmasked) {
        std::vector<Label> labels(15);

        if (isTransA) { do_pack(unroll_m, isLoad1Unmasked, isLoad2Unmasked); }

        mov(CO1, C);
        lea(CO2, ptr[CO1 + LDC * 2]);
        add(CO2, LDC);
        add(C, unroll_m * SIZE);
        mov(BO1, B);
        if (!isTransB) { lea(BO2, qword[B + LDB3]); }

        if (!isTransA) {
            lea(AA, ptr[A + (unroll_m * 2 - 1 - OFFSET) * SIZE]);
            cmp(M, UNROLL_M);
            jg(labels[13], T_NEAR);

            mov(AA, ORIG_A);
            lea(AA, ptr[AA + (unroll_m - 1 - OFFSET) * SIZE]);
            L(labels[13]);
        }

        mov(LL, N);
        mov(I, LL);
        if (!isTransA) {
            // If N is too small, skip copy operation
            cmp(LL, UNROLL_N * 3);
            jle(labels[7], T_NEAR);

            // If A is not aligned to cache line
            cmp(FLAG, 0);
            je(labels[7], T_NEAR);
        } else {
            cmp(LL, UNROLL_N);
            jl(labels[1], T_NEAR);
        }
        align(16);

        if (!isTransA) {
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, true, true);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        true, true);
            }
        } else {
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, false, false);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        false, false);
            }
        }

        sub(I, UNROLL_N);
        cmp(I, UNROLL_N);
        jl(labels[1], T_NEAR);
        align(16);

        L(labels[0]);
        if (unroll_m == 16) {
            kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                    false, false);
        } else {
            kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                    false, false);
        }
        sub(I, UNROLL_N);
        cmp(I, UNROLL_N);
        jge(labels[0], T_NEAR);
        align(16);

        L(labels[1]);
        cmp(I, 1);
        jne(labels[2], T_NEAR);
        if (unroll_m == 16) {
            kernel_16x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        jmp(labels[14], T_NEAR);
        align(16);

        L(labels[2]);
        cmp(I, 2);
        jne(labels[3], T_NEAR);
        if (unroll_m == 16) {
            kernel_16x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        jmp(labels[14], T_NEAR);
        align(16);

        L(labels[3]);
        cmp(I, 3);
        jne(labels[4], T_NEAR);
        if (unroll_m == 16) {
            kernel_16x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        jmp(labels[14], T_NEAR);
        align(16);

        L(labels[4]);
        cmp(I, 4);
        jne(labels[5], T_NEAR);
        if (unroll_m == 16) {
            kernel_16x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        jmp(labels[14], T_NEAR);
        align(16);

        L(labels[5]);
        cmp(I, 5);
        jne(labels[14], T_NEAR);
        if (unroll_m == 16) {
            kernel_16x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        } else {
            kernel_8x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, false,
                    false);
        }
        jmp(labels[14], T_NEAR);
        align(16);

        if (!isTransA) {
            L(labels[7]);
            cmp(I, UNROLL_N);
            jl(labels[6], T_NEAR);
            align(16);

            L(labels[8]);
            if (unroll_m == 16) {
                kernel_16x6(unroll_m, UNROLL_N, isLoad1Unmasked,
                        isLoad2Unmasked, true, false);
            } else {
                kernel_8x6(unroll_m, UNROLL_N, isLoad1Unmasked, isLoad2Unmasked,
                        true, false);
            }
            sub(I, UNROLL_N);
            cmp(I, UNROLL_N);
            jge(labels[8], T_NEAR);
            align(16);

            L(labels[6]);
            cmp(I, 1);
            jne(labels[9], T_NEAR);
            if (unroll_m == 16) {
                kernel_16x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x1(unroll_m, 1, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            jmp(labels[14], T_NEAR);
            align(16);

            L(labels[9]);
            cmp(I, 2);
            jne(labels[10], T_NEAR);
            if (unroll_m == 16) {
                kernel_16x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x2(unroll_m, 2, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            jmp(labels[14], T_NEAR);
            align(16);

            L(labels[10]);
            cmp(I, 3);
            jne(labels[11], T_NEAR);
            if (unroll_m == 16) {
                kernel_16x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x3(unroll_m, 3, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            jmp(labels[14], T_NEAR);
            align(16);

            L(labels[11]);
            cmp(I, 4);
            jne(labels[12], T_NEAR);
            if (unroll_m == 16) {
                kernel_16x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x4(unroll_m, 4, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            jmp(labels[14], T_NEAR);
            align(16);

            L(labels[12]);
            cmp(I, 5);
            jne(labels[14], T_NEAR);
            if (unroll_m == 16) {
                kernel_16x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            } else {
                kernel_8x5(unroll_m, 5, isLoad1Unmasked, isLoad2Unmasked, true,
                        false);
            }
            align(16);
        }

        L(labels[14]);
        // Compute address for A
        if (!isTransA) {
            add(A, unroll_m * SIZE);
        } else {
            mov(rax, LDA);
            imul(rax, rax, unroll_m);
            add(A, rax);
        }

        // Compute next address of BIAS
        if (hasBias) { add(BIAS, unroll_m * SIZE); }
    }

    void generate() override ATTRIBUTE_OPTIMIZE {
        assert(IMPLICATION(!is_avx2, mayiuse(avx)));

        preamble();

        Label buffer_in_ws, buffer_allocated;

        // Get the registers
        mov(B, ARG_B);
        mov(LDB, ARG_LDB);
        mov(r15, ARG_BETA);
        mov(r12, ARG_C);
        if (hasBias) mov(r10, ARG_BIAS);
        mov(LDC, ARG_LDC);
        mov(rbp, rsp);

        vmovss(xmm0, ptr[ARG_ALPHA]);
        vmovss(xmm1, ptr[r15]);

#ifdef _WIN32
        mov(A, ARG_A);
        mov(LDA, ARG_LDA);
#endif

        cmp(K, STACK_K_CAPACITY);
        jg(buffer_in_ws, T_NEAR);

        // Create buffer and align to 4kB page
        lea(rax, ptr[K * SIZE]);
        sal(rax, math::ilog2q(UNROLL_M));
        add(rax, 256);
        sub(rsp, rax);
        and_(rsp, -PAGE_4K);
        jmp(buffer_allocated, T_NEAR);

        L(buffer_in_ws);
        mov(rsp, ARG_WS);

        L(buffer_allocated);

        mov(ORIG_SP, rbp);
        mov(M, ARG_M);
        mov(N, ARG_N);
        mov(C, r12);
        if (hasBias) mov(BIAS, r10);
        vmovss(ALPHA, xmm0);
        vmovss(BETA, xmm1);
        sub(A, -OFFSET * SIZE);
        sub(B, -OFFSET * SIZE);
        mov(ORIG_A, A);
        sal(LDA, BASE_SHIFT);
        sal(LDB, BASE_SHIFT);
        sal(LDC, BASE_SHIFT);
        lea(LDB3, ptr[LDB + LDB * 2]);

        for (int i = 0; i < 8; i++) {
            mov(dword[rsp + 88 + i * 4], i);
        }

        if (isTransA && is_avx2) {
            movq(xmm0, LDA);
            vpbroadcastq(ymm1, xmm0);
            vinsertf128(ymm0, ymm0, xmm0, 1);
            vpermilpd(ymm0, ymm0, 5);
            vpaddq(ymm1, ymm1, ymm1);
            vperm2f128(ymm1, ymm1, ymm1, 8);
            vpaddq(ymm0, ymm0, ymm1);
            vmovups(STRIDE, ymm0);
        }

        // Check A alignment and leading dimension; take copy-based path as
        // needed
        mov(rax, LDA);
        or_(rax, A);
        and_(rax, 0x1f);
        mov(FLAG, rax);
        std::vector<Label> labels(5);

        cmp(M, UNROLL_M);
        jl(labels[0], T_NEAR);
        align(16);

        L(labels[1]);
        subloop(UNROLL_M, true, true);
        sub(M, UNROLL_M);
        cmp(M, UNROLL_M);
        jge(labels[1], T_NEAR);
        align(16);

        L(labels[0]);
        cmp(M, 0);
        jle(labels[4], T_NEAR);

        if (UNROLL_M > 8) {
            cmp(M, 8);
            jle(labels[2], T_NEAR);

            sub(M, 8);
            vbroadcastss(VMASK, M);
            vpcmpgtd(VMASK, VMASK, MASK);

            subloop(16, true, false);
            jmp(labels[4], T_NEAR);
            align(16);

            L(labels[2]);
            cmp(M, 8);
            jne(labels[3], T_NEAR);
            subloop(8, true, true);
            jmp(labels[4], T_NEAR);
        }

        align(16);

        L(labels[3]);
        vbroadcastss(VMASK, M);
        if (is_avx2) {
            vpcmpgtd(VMASK, VMASK, MASK);
        } else {
            auto xmask = Xmm(VMASK.getIdx());
            auto xmm_tmp = xmm4;

            vextractf128(xmm_tmp, VMASK, 1);
            vpcmpgtd(xmask, xmask, MASK);
            vpcmpgtd(xmm_tmp, xmm_tmp, dword[rsp + 88 + 4 * 4]); // MASK + 4
            vinsertf128(VMASK, VMASK, xmm_tmp, 1);
        }
        subloop(8, false, false);
        align(16);

        L(labels[4]);
        // Restore original stack
        mov(rsp, ORIG_SP);

        vzeroupper();
        postamble();
    }

public:
    int unroll_m() const { return UNROLL_M; }
    dim_t stack_k_capacity() const { return STACK_K_CAPACITY; }

private:
    const char isTransA;
    const char isTransB;
    const bool hasBias;
    const bool is_avx2;
    const int UNROLL_M;
    const int UNROLL_N;
    const bool isBeta0;
    const bool isBetaN;
    const int PREFETCHSIZEA;
    const int PREFETCHSIZEB;
    const dim_t STACK_K_CAPACITY;

    // Register allocation (for convenience)
    const Reg64 ARG_M = abi_param1;
    const Reg64 ARG_N = abi_param2;
    const Reg64 K = abi_param3;
    const Reg64 ARG_ALPHA = abi_param4;
#ifdef _WIN32
    const Address ARG_A = ptr[rsp + OFFSET_SHADOWSPACE + STACKSIZE];
    const Address ARG_LDA
            = qword[rsp + OFFSET_SHADOWSPACE + sizeof(float *) + STACKSIZE];
    const int stackOffset = OFFSET_SHADOWSPACE + sizeof(float *) + STACKSIZE;
    const Reg64 A = rsi;
    const Reg64 LDA = rdi;
#else
    const Reg64 ARG_A = r8;
    const Reg64 ARG_LDA = r9;
    const int stackOffset = STACKSIZE;
    const Reg64 A = ARG_A;
    const Reg64 LDA = ARG_LDA;
#endif
    const Address ARG_B = ptr[rsp + 8 + stackOffset];
    const Address ARG_LDB = ptr[rsp + 16 + stackOffset];
    const Address ARG_BETA = ptr[rsp + 24 + stackOffset];
    const Address ARG_C = ptr[rsp + 32 + stackOffset];
    const Address ARG_LDC = ptr[rsp + 40 + stackOffset];
    const Address ARG_BIAS = ptr[rsp + 48 + stackOffset];
    const Address ARG_WS = ptr[rsp + 56 + stackOffset];

    const Reg64 B = r11;
    const Reg64 LDB = rbx;
    const Reg64 LDC = r13;
    const Reg64 LL = rax;
    const Reg64 AO1 = abi_param2;
    const Reg64 BO1 = abi_param4;
    const Reg64 BO2 = rbp;
    const Reg64 CO1 = r14;
    const Reg64 CO2 = r15;
    const Reg64 LDB3 = r10;
    const Reg64 LDA4 = abi_param1;
    const Reg64 AA = r12;
    const Reg64 BIAS1 = abi_param1;

    const Address M = qword[rsp + 0];
    const Address N = qword[rsp + 8];
    const Address FLAG = qword[rsp + 16];
    const Address I = qword[rsp + 24];
    const Address C = qword[rsp + 32];
    const Address BIAS = qword[rsp + 40];
    const Address ALPHA = qword[rsp + 48];
    const Address BETA = qword[rsp + 64];
    const Address ORIG_A = qword[rsp + 80];
    const Address MASK = dword[rsp + 88];
    const Address STRIDE = qword[rsp + 120];
    const Address ORIG_SP = qword[rsp + 152];

    const Ymm VALPHA = ymm1;
    const Ymm VBETA = ymm2;
    const Ymm VMASK = ymm3;
    const Ymm VBIAS1 = ymm2;
    const Ymm VBIAS2 = ymm4;
};

xbyak_gemm_t *get_xbyak_gemm(
        bool isTransA, bool isTransB, float beta, bool hasBias) {
    auto beta_idx = [](float beta) {
        return (beta == 0.0) ? 0 : (beta == 1.0 ? 1 : 2);
    };

    // Kernel table [isTransA][isTransB][hasBias][beta (0, 1, other)]
    static maybe_unique_ptr<xbyak_gemm_t> kernel_table[2][2][2][3];
    static std::once_flag initialized;
    static std::atomic<dnnl_status_t> st(dnnl_success);
    std::call_once(initialized, [&] {
        for (bool isTransA : {false, true})
            for (bool isTransB : {false, true})
                for (bool hasBias : {false, true})
                    for (float beta : {0.0f, 1.0f, 2.0f}) {
                        // nocopy sgemm with bias for beta != 0.0 is not supported
                        if (hasBias && beta != 0.0) continue;
                        auto &kern = kernel_table[isTransA][isTransB][hasBias]
                                                 [beta_idx(beta)];

                        kern.reset(new xbyak_gemm_t(
                                isTransA, isTransB, beta, hasBias));
                        if (kern->create_kernel() != dnnl_success) {
                            st = dnnl_runtime_error;
                            return;
                        }
                    }
    });

    return (st == dnnl_success)
            ? kernel_table[isTransA][isTransB][hasBias][beta_idx(beta)].get()
            : nullptr;
}

dnnl_status_t sgemm_nocopy_driver(const char *transa, const char *transb,
        dim_t m, dim_t n, dim_t k, const float *alpha, const float *a,
        dim_t lda, const float *b, dim_t ldb, const float *beta, float *c,
        dim_t ldc, const float *bias) {

    bool isTransA = (*transa == 'T' || *transa == 't');
    bool isTransB = (*transb == 'T' || *transb == 't');

    dim_t Bm, sizeM, Bn, sizeN, Bk, sizeK;

    dim_t i, j;

    if ((m <= 0) || (n <= 0)) return dnnl_success;

    if ((k <= 0) || (alpha[0] == 0.)) {

        if (beta[0] == 0.) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[i + j * ldc] = 0.0;
        } else if (beta[0] != 1.) {
            for (j = 0; j < n; j++)
                for (i = 0; i < m; i++)
                    c[i + j * ldc] *= beta[0];
        }

        return dnnl_success;
    }

    assert(IMPLICATION(bias != nullptr, *beta == 0.0));

    // XXX: this happens on every thread...
    bool hasBias = (bias != nullptr);
    auto ker_bn = get_xbyak_gemm(isTransA, isTransB, *beta, hasBias);
    auto ker_b1 = get_xbyak_gemm(isTransA, isTransB, 1.0, false);
    auto ker_b0 = get_xbyak_gemm(isTransA, isTransB, 0.0, false);
    if (utils::any_null(ker_bn, ker_b1, ker_b0)) return dnnl_runtime_error;

    dim_t BM = 4032;
    dim_t BN = isTransA ? 96 : 48;
    dim_t BK = isTransB ? 96 : 256;

    float *ws = nullptr;
    bool use_heap_mem = BK > ker_b1->stack_k_capacity();
    if (use_heap_mem) {
        // Kernel uses sizeK * unroll_m + 64 + unroll_m elements as workspace.
        const dim_t um = ker_b1->unroll_m();
        const dim_t max_sizeK = BK;
        const size_t ws_size = sizeof *ws * (max_sizeK * um + 64 + um);

        ws = (float *)malloc(ws_size, PAGE_4K);
        if (!ws) return dnnl_out_of_memory;
    }

    const float *curA, *curB, *curBias = nullptr;
    float *curC;

    for (Bk = 0; Bk < k; Bk += sizeK) {
        sizeK = k - Bk;
        if (sizeK >= BK * 2)
            sizeK = BK;
        else {
            if (sizeK > BK) sizeK = (sizeK + 1) / 2;
        }

        for (Bm = 0; Bm < m; Bm += sizeM) {
            sizeM = m - Bm;
            if (sizeM >= BM * 2)
                sizeM = BM;
            else {
                if (sizeM > BM + BM / 2) sizeM = (sizeM + 1) / 2;
            }

            for (Bn = 0; Bn < n; Bn += sizeN) {
                sizeN = n - Bn;
                if (sizeN >= BN * 2)
                    sizeN = BN;
                else {
                    if (sizeN > BN + BN / 2) sizeN = (sizeN + 1) / 2;
                }

                if (!isTransA) {
                    curA = a + Bm + Bk * lda;
                } else {
                    curA = a + Bk + Bm * lda;
                }
                if (!isTransB) {
                    curB = b + Bk + Bn * ldb;
                } else {
                    curB = b + Bn + Bk * ldb;
                }
                curC = c + Bm + (size_t)Bn * ldc;
                if (bias != nullptr) {
                    if (Bk == 0) {
                        curBias = bias + Bm;
                    } else {
                        curBias = nullptr;
                    }
                }
                if (Bk == 0) {
                    if (*beta == 0.0 && bias == nullptr)
                        (*ker_b0)(sizeM, sizeN, sizeK, alpha, curA, lda, curB,
                                ldb, beta, curC, ldc, curBias, ws);
                    else
                        (*ker_bn)(sizeM, sizeN, sizeK, alpha, curA, lda, curB,
                                ldb, beta, curC, ldc, curBias, ws);
                } else {
                    (*ker_b1)(sizeM, sizeN, sizeK, alpha, curA, lda, curB, ldb,
                            beta, curC, ldc, curBias, ws);
                }
            }
        }
    }

    free(ws);
    msan_unpoison_matrix(c, m, n, ldc, sizeof(*c));

    return dnnl_success;
}

} // namespace avx_gemm_f32

dnnl_status_t jit_avx_gemm_f32(int nthrs, const char *transa,
        const char *transb, const dim_t *p_m, const dim_t *p_n,
        const dim_t *p_k, const float *p_alpha, const float *A,
        const dim_t *p_lda, const float *B, const dim_t *p_ldb,
        const float *p_beta, float *C, const dim_t *p_ldc, const float *bias) {

    using namespace dnnl::impl::utils;
    using namespace avx_gemm_f32;
    using namespace gemm_utils;

    if (*p_beta != 0 && bias)
        return ref_gemm(transa, transb, p_m, p_n, p_k, p_alpha, A, p_lda, B,
                p_lda, p_beta, C, p_ldc, bias);

    int nthr_max = dnnl_get_current_num_threads();
    int nthr_to_use = nstl::min(nthrs, nthr_max);

    dim_t m = *p_m;
    dim_t n = *p_n;
    dim_t k = *p_k;
    dim_t lda = *p_lda;
    dim_t ldb = *p_ldb;
    dim_t ldc = *p_ldc;
    float beta = *p_beta;
    dim_t MB, NB, KB;

    int nthr_m = 1, nthr_n = 1, nthr_k = 1, nthr_mn = 1;

    // Determine threading partitioning
    calc_nthr_nocopy_avx(
            m, n, k, nthr_to_use, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    assert(IMPLICATION(!dnnl_thr_syncable(), nthr_k == 1));

    nthr_to_use = nthr_m * nthr_n * nthr_k;

    nthr_mn = nthr_m * nthr_n;

    unsigned char *ompstatus_ = nullptr;
    unsigned char volatile *ompstatus = nullptr;

    float *c_buffers = nullptr;

    if (nthr_k > 1) {
        ompstatus_ = (unsigned char *)malloc(
                nthr_to_use * CACHE_LINE_SIZE, CACHE_LINE_SIZE);
        if (!ompstatus_) return dnnl_out_of_memory;

        ompstatus = (unsigned char volatile *)ompstatus_;
        assert(ompstatus);

        for (int i = 0; i < nthr_to_use; i++)
            ompstatus[i * CACHE_LINE_SIZE] = 0;

        c_buffers = (float *)malloc(
                sizeof(*c_buffers) * nthr_m * nthr_n * (nthr_k - 1) * MB * NB,
                PAGE_4K);
        if (!c_buffers) {
            free(ompstatus_);
            return dnnl_out_of_memory;
        }
    }

    if (nthr_to_use == 1) {
        auto status = sgemm_nocopy_driver(transa, transb, m, n, k, p_alpha, A,
                lda, B, ldb, p_beta, C, ldc, bias);
        return status;
    }

    // Always use the maximum number of threads to avoid OMP overhead that can
    // occur due to change thread counts.
    int nthr_spawn = dnnl_thr_syncable() ? nthr_max : nthr_to_use;

    std::atomic<dnnl_status_t> st(dnnl_success);
    parallel(nthr_spawn, [&](int ithr, int nthr) {
        assert(nthr_spawn == nthr);
        MAYBE_UNUSED(nthr);

        int ithr_m, ithr_n, ithr_k, ithr_mn;
        dim_t m_from, m_to, myM;
        dim_t n_from, n_to, myN;
        dim_t k_from, k_to, myK;
        int cbase, ibase;
        const float *myA, *myB, *myBias = nullptr;
        float *myC = C, myBeta;
        dim_t ld = ldc;

        int sum_later = (nthr < nthr_m * nthr_n * nthr_k);

        if (ithr < nthr_m * nthr_n * nthr_k) {

            ithr_mn = ithr % nthr_mn;
            ithr_m = ithr_mn % nthr_m;
            ithr_n = ithr_mn / nthr_m;
            ithr_k = ithr / nthr_mn;

            /* swap ithr_k for performance improvement */
            if (ithr_k == 0)
                ithr_k = nthr_k - 1;
            else if (ithr_k == nthr_k - 1)
                ithr_k = 0;

            m_from = MB * (ithr_m);
            m_to = MB * (ithr_m + 1);
            if (m_to > m) m_to = m;
            myM = m_to - m_from;

            n_from = NB * (ithr_n);
            n_to = NB * (ithr_n + 1);
            if (n_to > n) n_to = n;
            myN = n_to - n_from;

            k_from = KB * (ithr_k);
            k_to = KB * (ithr_k + 1);
            if (k_to > k) k_to = k;
            myK = k_to - k_from;

            cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);
            ibase = (ithr_m + nthr_m * ithr_n) * nthr_k;

            if ((myM > 0) && (myN > 0)) {

                if (*transa == 'N' || *transa == 'n') {
                    myA = &(A[m_from + k_from * lda]);
                } else {
                    myA = &(A[k_from + m_from * lda]);
                }
                if (*transb == 'N' || *transb == 'n') {
                    myB = &(B[k_from + n_from * ldb]);
                } else {
                    myB = &(B[n_from + k_from * ldb]);
                }
                if (ithr_k == 0) {
                    myC = &(C[m_from + n_from * ldc]);
                    myBeta = beta;
                    ld = ldc;
                    if (bias) myBias = &(bias[m_from]);
                } else {
                    myC = c_buffers + MB * NB * (cbase + ithr_k - 1);
                    myBeta = 0.0;
                    ld = MB;
                    myBias = nullptr;
                }

                dnnl_status_t st_thr = sgemm_nocopy_driver(transa, transb, myM,
                        myN, myK, p_alpha, myA, lda, myB, ldb, &myBeta, myC, ld,
                        myBias);
                if (st_thr != dnnl_success) {
                    st = st_thr;
                    return;
                }

                if (nthr_k > 1 && !sum_later)
                    ompstatus[(ibase + ithr_k) * CACHE_LINE_SIZE] = 1;
            }

            if (nthr_k > 1 && !sum_later) {

                // sum matrices partitioned along K dimension
                dim_t n1, n2;

                partition_unit_diff(ithr_k, nthr_k, myN, &n1, &n2);

                if (ithr_k > 0) {

                    myC = c_buffers + MB * NB * (cbase + ithr_k - 1) + n1 * MB;
                    /* need to wait until main thread finishes */
                    while (ompstatus[ibase * CACHE_LINE_SIZE] != 1) {};

                    /* my cache is hot */
                    sum_two_matrices(myM, n2, myC, MB,
                            &C[m_from + (n_from + n1) * ldc], ldc);
                }

                for (int ik = 1; ik < nthr_k; ++ik) {
                    if (ik != ithr_k) {

                        myC = c_buffers + MB * NB * (cbase + ik - 1) + n1 * MB;

                        while (ompstatus[(ibase + ik) * CACHE_LINE_SIZE] != 1) {
                        };

                        sum_two_matrices(myM, n2, myC, MB,
                                &C[m_from + (n_from + n1) * ldc], ldc);
                    }
                }
            }
        }
    });

    if (st != dnnl_success) {
        free(ompstatus_);
        free(c_buffers);
        return st;
    }

    // handle C summation later
    if (nthr_k > 1 && ompstatus[0] == 0) {

        parallel(nthr_spawn, [&](int ithr, int nthr) {
            assert(nthr_spawn == nthr);
            MAYBE_UNUSED(nthr);

            int ithr_m, ithr_n, ithr_k, ithr_mn;
            dim_t m_from, m_to, myM;
            dim_t n_from, n_to, myN;
            int cbase;
            float *myC = C;

            if (ithr < nthr_m * nthr_n * nthr_k) {

                ithr_mn = ithr % nthr_mn;
                ithr_m = ithr_mn % nthr_m;
                ithr_n = ithr_mn / nthr_m;
                ithr_k = ithr / nthr_mn;

                /* swap ithr_k for performance improvement */
                if (ithr_k == 0)
                    ithr_k = nthr_k - 1;
                else if (ithr_k == nthr_k - 1)
                    ithr_k = 0;

                m_from = MB * (ithr_m);
                m_to = MB * (ithr_m + 1);
                if (m_to > m) m_to = m;
                myM = m_to - m_from;

                n_from = NB * (ithr_n);
                n_to = NB * (ithr_n + 1);
                if (n_to > n) n_to = n;
                myN = n_to - n_from;

                cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

                if (nthr_k > 1) {
                    // sum matrices partitioned along K dimension
                    dim_t n1, n2;

                    partition_unit_diff(ithr_k, nthr_k, myN, &n1, &n2);

                    if (ithr_k > 0) {

                        myC = c_buffers + MB * NB * (cbase + ithr_k - 1)
                                + n1 * MB;

                        /* my cache is hot */
                        sum_two_matrices(myM, n2, myC, MB,
                                &C[m_from + (n_from + n1) * ldc], ldc);
                    }

                    for (int ik = 1; ik < nthr_k; ++ik) {
                        if (ik != ithr_k) {

                            myC = c_buffers + MB * NB * (cbase + ik - 1)
                                    + n1 * MB;

                            sum_two_matrices(myM, n2, myC, MB,
                                    &C[m_from + (n_from + n1) * ldc], ldc);
                        }
                    }
                }
            }
        });
    }

    free(c_buffers);
    free(ompstatus_);

    return dnnl_success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

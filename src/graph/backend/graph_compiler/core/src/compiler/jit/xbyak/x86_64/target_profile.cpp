/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include <util/utils.hpp>

#include "registers.hpp"
#include "target_profile.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {
namespace x86_64 {

target_profile_t get_target_profile(
        const runtime::target_machine_t &target_machine) {
    // target profile
    target_profile_t profile(target_machine);

    // Allocatable general purpose 64bit regs
    profile.alloc_gp_regs_ = {
            regs::rax,
            regs::rcx,
            regs::rdx,
            regs::rbx,
            // regs::rsp, needs to point to the head of the stack
            // regs::rbp, needs to point to start of stack frame
            regs::rsi,
            regs::rdi,
            regs::r8,
            regs::r9,
            regs::r10,
            regs::r11,
            regs::r12,
            regs::r13,
            regs::r14,
            regs::r15,
    };

    // Allocatable SIMD regs
    profile.alloc_xmm_regs_ = {
            regs::xmm0,
            regs::xmm1,
            regs::xmm2,
            regs::xmm3,
            regs::xmm4,
            regs::xmm5,
            regs::xmm6,
            regs::xmm7,
            regs::xmm8,
            regs::xmm9,
            regs::xmm10,
            regs::xmm11,
            regs::xmm12,
            regs::xmm13,
            regs::xmm14,
            regs::xmm15,
    };
    profile.alloc_xmm_vex_regs_ = profile.alloc_xmm_regs_;

    // AVX-512 extended regs
    if (target_machine.cpu_flags_.fAVX512F) {
        auto &xmm_regs = profile.alloc_xmm_regs_;
        xmm_regs.insert(xmm_regs.end(),
                {
                        regs::xmm16,
                        regs::xmm17,
                        regs::xmm18,
                        regs::xmm19,
                        regs::xmm20,
                        regs::xmm21,
                        regs::xmm22,
                        regs::xmm23,
                        regs::xmm24,
                        regs::xmm25,
                        regs::xmm26,
                        regs::xmm27,
                        regs::xmm28,
                        regs::xmm29,
                        regs::xmm30,
                        regs::xmm31,
                });
        // Allocatable mask regs
        profile.alloc_mask_regs_ = {
                regs::k1,
                regs::k2,
                regs::k3,
                regs::k4,
                regs::k5,
                regs::k6,
                regs::k7,
        };
    }

    // Allocatable AMX tile regs
    if (target_machine.cpu_flags_.fAVX512AMXTILE) {
        profile.alloc_tile_regs_ = {
                regs::tmm0,
                regs::tmm1,
                regs::tmm2,
                regs::tmm3,
                regs::tmm4,
                regs::tmm5,
                regs::tmm6,
                regs::tmm7,
        };
    }

    // Function call abi info
#ifdef __linux__
    profile.call_convention_ = call_convention::system_v;
    profile.shadow_space_bytes_ = 0;
    profile.red_zone_bytes_ = 0;
    // Function return regs
    profile.func_return_gp_reg_ = regs::rax;
    profile.func_return_xmm_reg_ = regs::xmm0;
    // Function call arg regs
    profile.func_arg_gp_regs_ = {
            regs::rdi, // 5
            regs::rsi, // 4
            regs::rdx, // 2
            regs::rcx, // 1
            regs::r8, // 6
            regs::r9, // 7
    };
    profile.func_arg_xmm_regs_ = {
            regs::xmm0,
            regs::xmm1,
            regs::xmm2,
            regs::xmm3,
            regs::xmm4,
            regs::xmm5,
            regs::xmm6,
            regs::xmm7,
    };
    // Function call save regs
    profile.caller_saved_gp_regs_ = {
            regs::rax,
            regs::rcx,
            regs::rdx,
            regs::rsi,
            regs::rdi,
            regs::r8,
            regs::r9,
            regs::r10,
            regs::r11,
    };
    profile.callee_saved_gp_regs_ = {
            regs::rbx,
            regs::r12,
            regs::r13,
            regs::r14,
            regs::r15,
    };
    profile.callee_saved_xmm_regs_ = {
            // non
    };
#elif _WIN32
    profile.call_convention_ = call_convention::microsoft;
    profile.shadow_space_bytes_ = 32;
    profile.red_zone_bytes_ = 0;
    // Function return regs
    profile.func_return_gp_reg_ = regs::rax;
    profile.func_return_xmm_reg_ = regs::xmm0;
    // Function call arg regs
    profile.func_arg_gp_regs_ = {
            regs::rcx, // 1
            regs::rdx, // 2
            regs::r8, // 6
            regs::r9, // 7
    };
    profile.func_arg_xmm_regs_ = {
            regs::xmm0,
            regs::xmm1,
            regs::xmm2,
            regs::xmm3,
    };
    // Function call save regs
    profile.caller_saved_gp_regs_ = {
            regs::rax,
            regs::rcx,
            regs::rdx,
            regs::r8,
            regs::r9,
            regs::r10,
            regs::r11,
    };
    profile.callee_saved_gp_regs_ = {
            regs::rbx,
            regs::rdi,
            regs::rsi,
            regs::r12,
            regs::r13,
            regs::r14,
            regs::r15,
    };
    profile.callee_saved_xmm_regs_ = {
            regs::xmm6,
            regs::xmm7,
            regs::xmm8,
            regs::xmm9,
            regs::xmm10,
            regs::xmm11,
            regs::xmm12,
            regs::xmm13,
            regs::xmm14,
            regs::xmm15,
    };
#else
    COMPILE_ASSERT(false, "Not supported");
#endif

    return profile;
}

} // namespace x86_64
} // namespace xbyak
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

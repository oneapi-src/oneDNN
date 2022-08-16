/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include <immintrin.h>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <runtime/config.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/target_machine.hpp>
#include <unordered_map>
#include <util/string_utils.hpp>
#include <util/utils.hpp>
#ifdef _MSC_VER
//  Windows
#include <intrin.h>
#define cpuid(info, x, y) __cpuidex(info, x, y)
#else
// #include <cpuid.h>
static void cpuid(int info[4], int InfoType, int v2) {
    __cpuid_count(InfoType, v2, info[0], info[1], info[2], info[3]);
}
#endif

#ifndef _WIN32
extern char **environ;
#endif

SC_MODULE(target)
namespace sc {
using namespace env_key;
namespace runtime {
target_machine_t::target_machine_t(
        type device_type, std::unique_ptr<machine_flags_t> device_flags)
    : device_type_(device_type), device_flags_(std::move(device_flags)) {}

target_machine_t::target_machine_t(const target_machine_t &other)
    : device_type_(other.device_type_), cpu_flags_(other.cpu_flags_) {}

target_machine_t &target_machine_t::operator=(const target_machine_t &other) {
    device_type_ = other.device_type_;
    cpu_flags_ = other.cpu_flags_;
    return *this;
}

const machine_flags_t &target_machine_t::get_device_flags() const {
    if (device_type_ == type::cpu) { return cpu_flags_; }
    return *device_flags_;
}

static int get_xcr0() {
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0); /* min VS2010 SP1 compiler is required */
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    return xcr0;
}

// sets max_simd_bits
void target_machine_t::set_simd_length_and_max_cpu_threads(cpu_flags_t &flg) {
    if (flg.fAVX512F)
        flg.max_simd_bits = 512;
    else if (flg.fAVX2 || flg.fAVX)
        flg.max_simd_bits = 256;
    else if (flg.fSSE)
        flg.max_simd_bits = 128;
    else
        flg.max_simd_bits = 0;
}

uint32_t machine_flags_t::get_max_vector_lanes(sc_data_etype etype) const {
    return max_simd_bits / 8 / utils::get_sizeof_etype(etype);
}

target_machine_t &get_runtime_target_machine() {
    static target_machine_t rtm = get_native_target_machine();
    return rtm;
}

void set_runtime_target_machine(const target_machine_t &in) {
    auto &rtm = get_runtime_target_machine();
    rtm = in;
}

target_machine_t get_native_target_machine() {
    target_machine_t tm(target_machine_t::type::cpu, nullptr);
    int xcr0 = get_xcr0();
    int info[4];
    cpuid(info, 0, 0);
    int nIds = info[0];

    cpuid(info, 0x80000000, 0);
    unsigned nExIds = info[0];

    //  Detect Features
    if (nIds >= 0x00000001) {
        cpuid(info, 0x00000001, 0);
        tm.cpu_flags_.fMMX = (info[3] & ((int)1 << 23)) != 0;
        tm.cpu_flags_.fSSE = (info[3] & ((int)1 << 25)) != 0;
        tm.cpu_flags_.fSSE2 = (info[3] & ((int)1 << 26)) != 0;
        tm.cpu_flags_.fSSE3 = (info[2] & ((int)1 << 0)) != 0;

        tm.cpu_flags_.fSSSE3 = (info[2] & ((int)1 << 9)) != 0;
        tm.cpu_flags_.fSSE41 = (info[2] & ((int)1 << 19)) != 0;
        tm.cpu_flags_.fSSE42 = (info[2] & ((int)1 << 20)) != 0;
        tm.cpu_flags_.fAES = (info[2] & ((int)1 << 25)) != 0;

        tm.cpu_flags_.fAVX
                = ((xcr0 & 0x6) == 0x6) && (info[2] & ((int)1 << 28)) != 0;
        tm.cpu_flags_.fFMA3 = (info[2] & ((int)1 << 12)) != 0;

        tm.cpu_flags_.fRDRAND = (info[2] & ((int)1 << 30)) != 0;
    }
    if (nIds >= 0x00000007) {
        cpuid(info, 0x00000007, 0);
        tm.cpu_flags_.fAVX2
                = tm.cpu_flags_.fAVX && (info[1] & ((int)1 << 5)) != 0;

        tm.cpu_flags_.fBMI1 = (info[1] & ((int)1 << 3)) != 0;
        tm.cpu_flags_.fBMI2 = (info[1] & ((int)1 << 8)) != 0;
        tm.cpu_flags_.fADX = (info[1] & ((int)1 << 19)) != 0;
        tm.cpu_flags_.fSHA = (info[1] & ((int)1 << 29)) != 0;
        tm.cpu_flags_.fPREFETCHWT1 = (info[2] & ((int)1 << 0)) != 0;

        tm.cpu_flags_.fAVX512F
                = ((xcr0 & 0xe6) == 0xe6) && (info[1] & ((int)1 << 16)) != 0;
        bool hasavx512 = tm.cpu_flags_.fAVX512F;
        tm.cpu_flags_.fAVX512CD = hasavx512 && (info[1] & ((int)1 << 28)) != 0;
        tm.cpu_flags_.fAVX512PF = hasavx512 && (info[1] & ((int)1 << 26)) != 0;
        tm.cpu_flags_.fAVX512ER = hasavx512 && (info[1] & ((int)1 << 27)) != 0;
        tm.cpu_flags_.fAVX512VL = hasavx512 && (info[1] & ((int)1 << 31)) != 0;
        tm.cpu_flags_.fAVX512BW = hasavx512 && (info[1] & ((int)1 << 30)) != 0;
        tm.cpu_flags_.fAVX512DQ = hasavx512 && (info[1] & ((int)1 << 17)) != 0;
        tm.cpu_flags_.fAVX512IFMA
                = hasavx512 && (info[1] & ((int)1 << 21)) != 0;
        tm.cpu_flags_.fAVX512VNNI
                = hasavx512 && (info[2] & ((int)1 << 11)) != 0;
        tm.cpu_flags_.fAVX512AMXBF16
                = hasavx512 && (info[3] & ((int)1 << 22)) != 0;
        tm.cpu_flags_.fAVX512AMXTILE
                = hasavx512 && (info[3] & ((int)1 << 24)) != 0;
        tm.cpu_flags_.fAVX512AMXINT8
                = hasavx512 && (info[3] & ((int)1 << 25)) != 0;
        tm.cpu_flags_.fAVX512VBMI = hasavx512 && (info[2] & ((int)1 << 1)) != 0;

        if (hasavx512) {
            int info2[4];
            cpuid(info2, 0x00000007, 1);
            tm.cpu_flags_.fAVX512BF16 = (info2[0] & ((int)1 << 5)) != 0;
        }
    }
    if (nExIds >= 0x80000001) {
        cpuid(info, 0x80000001, 0);
        tm.cpu_flags_.fx64 = (info[3] & ((int)1 << 29)) != 0;
        tm.cpu_flags_.fABM = (info[2] & ((int)1 << 5)) != 0;
        tm.cpu_flags_.fSSE4a = (info[2] & ((int)1 << 6)) != 0;
        tm.cpu_flags_.fFMA4 = (info[2] & ((int)1 << 16)) != 0;
        tm.cpu_flags_.fXOP = (info[2] & ((int)1 << 11)) != 0;
    }
    target_machine_t::set_simd_length_and_max_cpu_threads(tm.cpu_flags_);
    return tm;
}
} // namespace runtime
} // namespace sc

/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CONFIG_TARGET_MACHINE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CONFIG_TARGET_MACHINE_HPP
#include <memory>
#include <utility>
#include <compiler/ir/sc_data_type.hpp>
#include <util/def.hpp>
namespace sc {

enum class jit_kind {
    cfake = 0,
    llvm,
};

struct machine_flags_t {
    unsigned int max_simd_bits;
    SC_INTERNAL_API uint32_t get_max_vector_lanes(sc_data_etype etype) const;
};

struct cpu_flags_t : public machine_flags_t {
    //  Misc.
    bool fMMX = false;
    bool fx64 = false;
    bool fABM = false; // Advanced Bit Manipulation
    bool fRDRAND = false;
    bool fBMI1 = false;
    bool fBMI2 = false;
    bool fADX = false;
    // what about fPREFETCHW?
    bool fPREFETCHWT1 = false;

    //  SIMD: 128-bit
    bool fSSE = false;
    bool fSSE2 = false;
    bool fSSE3 = false;
    bool fSSSE3 = false;
    bool fSSE41 = false;
    bool fSSE42 = false;
    bool fSSE4a = false;
    bool fAES = false;
    bool fSHA = false;

    //  SIMD: 256-bit
    bool fAVX = false;
    bool fXOP = false;
    bool fFMA3 = false;
    bool fFMA4 = false;
    bool fAVX2 = false;

    //  SIMD: 512-bit
    bool fAVX512F = false; //  AVX512 Foundation
    bool fAVX512CD = false; //  AVX512 Conflict Detection
    bool fAVX512PF = false; //  AVX512 Prefetch
    bool fAVX512ER = false; //  AVX512 Exponential + Reciprocal
    bool fAVX512VL = false; //  AVX512 Vector Length Extensions
    bool fAVX512BW = false; //  AVX512 Byte + Word
    bool fAVX512DQ = false; //  AVX512 Doubleword + Quadword
    bool fAVX512IFMA = false; //  AVX512 Integer 52-bit Fused Multiply-Add
    bool fAVX512VNNI = false; //  AVX512 Vector Neural Network Instructions
    bool fAVX512AMXBF16 = false; // AVX512 Advanced Matrix Extension for bf16
    bool fAVX512AMXTILE = false; // AVX512 Advanced Matrix Extension for tile
    bool fAVX512AMXINT8 = false; // AVX512 Advanced Matrix Extension for int8
    bool fAVX512VBMI = false; //  AVX512 Vector Byte Manipulation Instructions
    bool fAVX512BF16 = false; //  AVX512 BF16 Instructions
};

struct SC_INTERNAL_API target_machine_t {
    enum class type {
        cpu,
    } device_type_;
    cpu_flags_t cpu_flags_;
    target_machine_t(
            type device_type, std::unique_ptr<machine_flags_t> device_flags);
    // if is device, returns device_flags, else, return cpu_flags_t
    const machine_flags_t &get_device_flags() const;
    target_machine_t(const target_machine_t &other);
    target_machine_t(target_machine_t &&other) = default;
    target_machine_t &operator=(target_machine_t &&other) {
        device_type_ = other.device_type_;
        cpu_flags_ = other.cpu_flags_;
        device_flags_ = std::move(other.device_flags_);
        return *this;
    }

    static void set_simd_length_and_max_cpu_threads(cpu_flags_t &tm);

private:
    std::unique_ptr<machine_flags_t> device_flags_;
};

SC_INTERNAL_API target_machine_t get_native_target_machine();

} // namespace sc
#endif

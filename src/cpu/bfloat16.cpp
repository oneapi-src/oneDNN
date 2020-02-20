/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <array>
#include <memory>
#include "bfloat16.hpp"
#include "bit_cast.hpp"
#include "cpu_isa_traits.hpp"
#include "jit_avx512_core_bf16cvt.hpp"

namespace dnnl {
namespace impl {

using namespace cpu::bf16_support;

bfloat16_t &bfloat16_t::operator=(float f) {
    if (cpu::mayiuse(cpu::cpu_isa_t::avx512_core)) {
        jit_call_t p;
        p.inp = (void *)&f;
        p.out = (void *)this;
        static const cpu::jit_avx512_core_cvt_ps_to_bf16_t cvt_one_ps_to_bf16(
                1);
        cvt_one_ps_to_bf16.jit_ker(&p);
    } else {
        auto iraw = utils::bit_cast<std::array<uint16_t, 2>>(f);
        switch (std::fpclassify(f)) {
            case FP_SUBNORMAL:
            case FP_ZERO:
                // sign preserving zero (denormal go to zero)
                raw_bits_ = iraw[1];
                raw_bits_ &= 0x8000;
                break;
            case FP_INFINITE: raw_bits_ = iraw[1]; break;
            case FP_NAN:
                // truncate and set MSB of the mantissa force QNAN
                raw_bits_ = iraw[1];
                raw_bits_ |= 1 << 6;
                break;
            case FP_NORMAL:
                // round to nearest even and truncate
                const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
                const uint32_t int_raw
                        = utils::bit_cast<uint32_t>(f) + rounding_bias;
                iraw = utils::bit_cast<std::array<uint16_t, 2>>(int_raw);
                raw_bits_ = iraw[1];
                break;
        }
    }
    return *this;
}

bfloat16_t::operator float() const {
    std::array<uint16_t, 2> iraw = {{0, raw_bits_}};
    return utils::bit_cast<float>(iraw);
}

void cvt_float_to_bfloat16(bfloat16_t *out, const float *inp, size_t size) {
    if (cpu::mayiuse(cpu::cpu_isa_t::avx512_core)) {
        jit_call_t p_;
        p_.inp = (void *)inp;
        p_.out = (void *)out;
        p_.size = size;
        static const cpu::jit_avx512_core_cvt_ps_to_bf16_t cvt_ps_to_bf16;
        cvt_ps_to_bf16.jit_ker(&p_);
    } else {
        for (size_t i = 0; i < size; ++i)
            out[i] = inp[i];
    }
}

void cvt_bfloat16_to_float(float *out, const bfloat16_t *inp, size_t size) {
    if (cpu::mayiuse(cpu::cpu_isa_t::avx512_core)) {
        jit_call_t p_;
        p_.inp = (void *)inp;
        p_.out = (void *)out;
        p_.size = size;
        static const cpu::jit_avx512_core_cvt_bf16_to_ps_t cvt_bf16_to_ps;
        cvt_bf16_to_ps.jit_ker(&p_);
    } else {
        for (size_t i = 0; i < size; ++i)
            out[i] = inp[i];
    }
}

void add_floats_and_cvt_to_bfloat16(
        bfloat16_t *out, const float *inp0, const float *inp1, size_t size) {
    if (cpu::mayiuse(cpu::cpu_isa_t::avx512_core)) {
        jit_call_t p_;
        p_.inp = (void *)inp0;
        p_.add = (void *)inp1;
        p_.out = (void *)out;
        p_.size = size;
        static const cpu::jit_avx512_core_add_cvt_ps_to_bf16_t
                add_cvt_ps_to_bf16;
        add_cvt_ps_to_bf16.jit_ker(&p_);
    } else {
        for (size_t i = 0; i < size; ++i)
            out[i] = inp0[i] + inp1[i];
    }
}

} // namespace impl
} // namespace dnnl

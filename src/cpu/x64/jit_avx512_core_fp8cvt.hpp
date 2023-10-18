/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_FP8CVT_HPP
#define CPU_X64_JIT_AVX512_CORE_FP8CVT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct fp8_emulation_base_t {
    fp8_emulation_base_t(jit_generator *host) : host_(host) {}

    virtual ~fp8_emulation_base_t() = default;

    // Must be called from host kernel after postamble to populate lookup table.
    virtual void prepare_table() = 0;

    virtual void vcvt_f8_to_f16(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in)
            = 0;
    virtual void vcvt_f8_to_f32(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in)
            = 0;
    virtual void vcvt_f16_to_f8(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in)
            = 0;
    virtual void vcvt_f32_to_f8(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in)
            = 0;

protected:
    jit_generator *const host_;
};

struct fp8_emulation_e5m2_t : public fp8_emulation_base_t {
    fp8_emulation_e5m2_t(jit_generator *host, const Xbyak::Xmm &xmm_aux1,
            const Xbyak::Xmm &xmm_aux2, const Xbyak::Opmask kmask_aux_,
            const Xbyak::Reg64 reg64_aux)
        : fp8_emulation_base_t(host)
        , xmm_aux1_(xmm_aux1)
        , xmm_aux2_(xmm_aux2)
        , kmask_aux_(kmask_aux_)
        , reg64_aux_(reg64_aux) {}

    void prepare_table() override;

    void vcvt_f8_to_f16(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) override;
    void vcvt_f8_to_f32(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) override;
    void vcvt_f16_to_f8(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) override;
    void vcvt_f32_to_f8(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) override;

private:
    Xbyak::Label label_table_to_f8_;
    const Xbyak::Xmm xmm_aux1_;
    const Xbyak::Xmm xmm_aux2_;
    const Xbyak::Opmask kmask_aux_;
    const Xbyak::Reg64 reg64_aux_;
};

struct fp8_emulation_e4m3_t : public fp8_emulation_base_t {
    fp8_emulation_e4m3_t(jit_generator *host, const Xbyak::Xmm &xmm_aux1,
            const Xbyak::Xmm &xmm_aux2, const Xbyak::Xmm &xmm_aux3,
            const Xbyak::Reg64 reg64_aux)
        : fp8_emulation_base_t(host)
        , xmm_aux1_(xmm_aux1)
        , xmm_aux2_(xmm_aux2)
        , xmm_aux3_(xmm_aux3)
        , reg64_aux_(reg64_aux) {}

    void prepare_table() override;

    void vcvt_f8_to_f16(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) override;
    void vcvt_f8_to_f32(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) override;
    void vcvt_f16_to_f8(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) override;
    void vcvt_f32_to_f8(
            const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) override;

private:
    // Load table values from 128 consecutive bytes at given address.
    // Input Zmm register holds table lookup indices.
    // Must use full Zmm registers to properly load all table values.
    void tabulate(const data_type_t dt, const Xbyak::Zmm &zmm_out,
            const Xbyak::Zmm &zmm_in, const Xbyak::Address &addr);

    Xbyak::Label label_table_from_f8_;
    Xbyak::Label label_table_to_f8_;
    const Xbyak::Xmm xmm_aux1_;
    const Xbyak::Xmm xmm_aux2_;
    const Xbyak::Xmm xmm_aux3_;
    const Xbyak::Reg64 reg64_aux_;
};

enum f32_convert_mode_t {
    f8_e5m2_to_f16,
    f8_e4m3_to_f16,
    f8_e5m2_to_f32,
    f8_e4m3_to_f32,
    f16_to_f8_e5m2,
    f16_to_f8_e4m3,
    f32_to_f8_e5m2,
    f32_to_f8_e4m3,
    f16_to_f32,
    f32_to_f16,
};

struct jit_cvt_fp8_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_cvt_fp8_t)

    jit_cvt_fp8_t(f32_convert_mode_t mode);

private:
    const Xbyak::Xmm xmm_out = xmm0;
    const Xbyak::Xmm xmm_inp = xmm1;
    const Xbyak::Xmm xmm_aux1 = xmm2;
    const Xbyak::Xmm xmm_aux2 = xmm3;
    const Xbyak::Xmm xmm_aux3 = xmm4;
    const Xbyak::Opmask kmask_aux = k1;
    const Xbyak::Reg64 reg64_aux = abi_not_param1;
    const Xbyak::Reg64 reg64_out = abi_param1;
    const Xbyak::Reg64 reg64_inp = abi_param2;
    void generate() override;
    std::unique_ptr<fp8_emulation_base_t> fp8_emu_;
    f32_convert_mode_t mode_;
};

bool try_cvt_f8_e5m2_to_f32(float *out, const float8_e5m2_t *inp);
bool try_cvt_f8_e4m3_to_f32(float *out, const float8_e4m3_t *inp);
bool try_cvt_f8_e5m2_to_f16(float16_t *out, const float8_e5m2_t *inp);
bool try_cvt_f8_e4m3_to_f16(float16_t *out, const float8_e4m3_t *inp);
bool try_cvt_f16_to_f8_e5m2(float8_e5m2_t *out, const float16_t *inp);
bool try_cvt_f16_to_f8_e4m3(float8_e4m3_t *out, const float16_t *inp);
bool try_cvt_f32_to_f8_e5m2(float8_e5m2_t *out, const float *inp);
bool try_cvt_f32_to_f8_e4m3(float8_e4m3_t *out, const float *inp);
bool try_cvt_f16_to_f32(float *out, const float16_t *inp);
bool try_cvt_f32_to_f16(float16_t *out, const float *inp);

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

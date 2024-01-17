/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_SRC_QUANTIZATION_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_SRC_QUANTIZATION_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/x64/jit_brgemm_primitive_conf.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct src_quantization_compile_params_t {
    size_t ic_quant_block;
    data_type_t src_dt;
    data_type_t qsrc_dt;
};

struct src_quantization_runtime_params_t {
    const void *src_ptr;
    const void *qsrc_ptr;
    const void *src_scales_ptr;
    size_t ic_size;
};

struct jit_src_quantization_kernel_t {
    void operator()(const src_quantization_runtime_params_t *args) { assert(ker_);
        ker_(args);
    }

    jit_src_quantization_kernel_t(const src_quantization_compile_params_t& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_src_quantization_kernel_t() {}
protected:
    void (*ker_)(const src_quantization_runtime_params_t *);

    src_quantization_compile_params_t jcp_;
};

template <cpu_isa_t isa>
struct jit_brgemm_src_quantization_kernel_t : public jit_src_quantization_kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_src_quantization_kernel_t)

    jit_brgemm_src_quantization_kernel_t(const src_quantization_compile_params_t& jcp)
        : jit_src_quantization_kernel_t(jcp), jit_generator(jit_name()) {
        vec_size = cpu_isa_traits<isa>::vlen / sizeof(float);

        create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename utils::conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    static constexpr int n_vregs = cpu_isa_traits<isa>::n_vregs;

    void generate() override;
    void load_src(Vmm vmm_load, const Xbyak::Address& addr);

    Vmm vmm_src() {
        return Vmm(0);
    }

    Vmm vmm_max() {
        return Vmm(1);
    }

    Vmm vmm_sign_bit_mask() {
        return Vmm(2);
    }

    Vmm vmm_aux() {
        return Vmm(3);
    }

    Vmm vmm_int8_max() {
        return Vmm(4);
    }

    Vmm vmm_qscale() {
        return Vmm(5);
    }

    Vmm vmm_one() {
        return Vmm(6);
    }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_qsrc = r9;
    Xbyak::Reg64 reg_src_scales = r10;
    Xbyak::Reg64 reg_ic_size = r11;
    Xbyak::Reg64 reg_tmp = r12;

    size_t vec_size;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_JIT_BRGEMM_SRC_QUANTIZATION_KERNEL_HPP
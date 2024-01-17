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

#ifndef CPU_X64_JIT_BRGEMM_WEIGHTS_DECOMPRESSION_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_WEIGHTS_DECOMPRESSION_KERNEL_HPP

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

struct weights_decompression_compile_params_t {
    bool with_scales;
    bool with_zero_points;
    bool broadcast_scales;
    bool broadcast_zero_points;
    size_t oc_size;
    size_t ic_internal_size;
    data_type_t weights_dt;
    data_type_t decomp_buffer_dt;
    data_type_t zero_points_dt;
};

struct weights_decompression_runtime_params_t {
    const void *weights_ptr;
    const void *decomp_buffer_ptr;
    const void *scales_ptr;
    const void *zero_points_ptr;
    size_t ic_size;
};

struct jit_weights_decompression_kernel_t {
    void operator()(const weights_decompression_runtime_params_t *args) { assert(ker_);
        ker_(args);
    }

    jit_weights_decompression_kernel_t(const weights_decompression_compile_params_t& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_weights_decompression_kernel_t() {}
protected:
    void (*ker_)(const weights_decompression_runtime_params_t *);

    weights_decompression_compile_params_t jcp_;
};

template <cpu_isa_t isa>
struct jit_brgemm_weights_decompression_kernel_t : public jit_weights_decompression_kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_weights_decompression_kernel_t)

    jit_brgemm_weights_decompression_kernel_t(const weights_decompression_compile_params_t& jcp)
        : jit_weights_decompression_kernel_t(jcp), jit_generator(jit_name()) {
        vec_size = cpu_isa_traits<isa>::vlen / sizeof(float);

        create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename utils::conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    static constexpr int n_vregs = cpu_isa_traits<isa>::n_vregs;

    void generate() override;
    void init_decomp_params(std::function<Vmm(int)> vmm_params, Xbyak::Reg64 reg_params, bool broadcast_values, data_type_t element_type);
    void load_weights(Vmm vmm_load, const Xbyak::Address& addr, int ic);
    void store_weights(const Xbyak::Address& addr, Vmm vmm_store);

    Vmm vmm_scales(int ocb) {
        return Vmm(unroll_factor + ocb);
    }

    Vmm vmm_zero_points(int ocb) {
        return Vmm(2 * unroll_factor + ocb);
    }

    Vmm vmm_weights(int ocb) {
        assert(ocb < unroll_factor);
        return Vmm(ocb);
    }

    Vmm vmm_mask(int ic) {
        return Vmm(n_vregs - ic - 2);
    }

    Vmm vmm_tmp(int idx) {
        return Vmm(n_vregs - idx - 1);
    }

    Vmm vmm_lookup() { return vmm_tmp(0); }
    Vmm vmm_lookup_low() { return vmm_tmp(0); }
    Vmm vmm_lookup_high() { return vmm_tmp(1); }
    Vmm vmm_mask8() { return vmm_tmp(2); }
    Vmm vmm_mask7() { return vmm_tmp(3); }

    Xbyak::Reg64 reg_weights = r8;
    Xbyak::Reg64 reg_decomp_buffer = r9;
    Xbyak::Reg64 reg_scales = r10;
    Xbyak::Reg64 reg_zero_points = r11;
    Xbyak::Reg64 reg_ic_size = r12;
    Xbyak::Reg64 reg_tmp = r13;

    size_t vec_size;

    static constexpr int unroll_factor = 4;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_X64_JIT_BRGEMM_WEIGHTS_DECOMPRESSION_KERNEL_HPP
/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_LRN_KERNEL_HPP
#define CPU_X64_JIT_UNI_LRN_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct bf16_emulation_t;
struct jit_args_fwd_t {
    const void *src;
    void *dst, *scratch;
};

struct jit_args_bwd_t {
    const float *src, *diff_dst, *scratch;
    float *diff_src;
};

struct nchw8c_across {
    /*  version:
    *  -1: channels 0..7,
    *   1: channels C-8 .. C-1,
    *   0: other channels
    *   3: channels only for this kernel(without prev and next)
    */
    int H, W, version;
    nchw8c_across(int h, int w, int v) : H(h), W(w), version(v) {}
};

struct within_config {
    int H, W, size;
    within_config(int h, int w, int s) : H(h), W(w), size(s) {}
};

struct nchw_across {
    int C, HW, tail;
    nchw_across(int c, int hw, int t) : C(c), HW(hw), tail(t) {}
};

struct nhwc_across {
    int C;
    nhwc_across(int c) : C(c) {}
};

template <cpu_isa_t isa, data_type_t d_type>
struct jit_uni_lrn_fwd_kernel : public jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lrn_fwd_kernel)

    jit_uni_lrn_fwd_kernel(const within_config &J, float A, float K,
            prop_kind_t pk, void *code_ptr = nullptr,
            size_t code_size = 4 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_fwd_kernel(const nchw8c_across &J, float A, float K,
            prop_kind_t pk, void *code_ptr = nullptr,
            size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_fwd_kernel(const nhwc_across &J, float A, float K,
            prop_kind_t pk, void *code_ptr = nullptr,
            size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    jit_uni_lrn_fwd_kernel(const nchw_across &J, float A, float K,
            prop_kind_t pk, void *code_ptr = nullptr,
            size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    ~jit_uni_lrn_fwd_kernel();
    void operator()(jit_args_fwd_t *arg) { ker(arg); }
    void (*ker)(jit_args_fwd_t *);

    static constexpr int VECTOR_LENGTH = isa == avx512_common ? 16 : 8;

private:
    using Vmm = typename utils::conditional<isa == avx2, Xbyak::Ymm,
            Xbyak::Zmm>::type;
    void load_data(Vmm reg, const Xbyak::Address p);
    void store_data(const Xbyak::Address p, Vmm reg);
    void load_constant(float constant, Vmm v_constant, Xbyak::Xmm x_constant);
    void within_body_reg_blocked(int loop_count, int max_reg_block, int hoff,
            int Hoff, int woff, int Woff, int stride, prop_kind_t pk);
    void within_body(int hoff, int Hoff, int woff, int Woff, int stride,
            prop_kind_t pk, int reg_block = 1, int single_pixel_offset = 0);
    void nchw_body(int tail, int HW, prop_kind_t pk, Xbyak::Ymm ymask,
            Xbyak::Ymm ya, Xbyak::Ymm yb, Xbyak::Ymm yc, Xbyak::Ymm yd,
            Xbyak::Ymm ye, Xbyak::Ymm ysum);
    void nchw_body_sse41(int tail, int HW, prop_kind_t pk, Xbyak::Xmm xe_lo,
            Xbyak::Xmm xe_hi, Xbyak::Xmm xsum_lo, Xbyak::Xmm xsum_hi);
    void nchw_tail_sse41(int tail, Xbyak::Reg64 reg_dst, Xbyak::Xmm xtail_lo,
            Xbyak::Xmm xtail_hi);
    void move_data_pointers(int pixel_count, prop_kind_t pk);

    const Xbyak::Reg64 src_ = rax;
    const Xbyak::Reg64 dst_ = r8;
    const Xbyak::Reg64 scratch_ = rdx;
    const Xbyak::Reg64 imm_addr64_ = rbx;
    const Xbyak::Reg64 store_addr_ = rbp;

    const Xbyak::Xmm xalpha_ = xmm0;
    const Xbyak::Xmm xk_ = xmm1;
    const Xbyak::Ymm yk_ = ymm1;
    const Vmm valpha_ = Vmm(0);
    const Vmm vk_ = Vmm(1);
    const Xbyak::Reg64 h_ = r9;
    const Xbyak::Reg64 w_ = r10;

    const Xbyak::Zmm bf16_emu_reserv_1_ = Xbyak::Zmm(28);
    const Xbyak::Zmm bf16_emu_reserv_2_ = Xbyak::Zmm(29);
    const Xbyak::Reg64 bf16_emu_scratch_ = rax;
    const Xbyak::Zmm bf16_emu_reserv_3_ = Xbyak::Zmm(30);
    const Xbyak::Zmm bf16_emu_reserv_4_ = Xbyak::Zmm(31);
    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    float alpha_;
    float k_;
    int tempIdx_ = 0;
    int reg_block_idx_ = 0;
    static constexpr int stack_space_needed_ = 11 * 4 * sizeof(float) + 16;
    static constexpr int single_pixel_offset_
            = VECTOR_LENGTH * sizeof(typename prec_traits<d_type>::type);
};

template <cpu_isa_t isa>
struct jit_uni_lrn_bwd_kernel_f32 : public jit_generator {
    Xbyak::Reg64 src_ = rax;
    Xbyak::Reg64 diffsrc = r8;
    Xbyak::Reg64 diffdst = r9;
    Xbyak::Reg64 workspace = rdx;
    Xbyak::Reg64 imm_addr64_ = rsi;

    Xbyak::Xmm xnalphabeta = xmm0;
    Xbyak::Ymm ynalphabeta = ymm0;

    float nalphabeta;

    int use_h_parallelizm;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_lrn_bwd_kernel_f32)

    jit_uni_lrn_bwd_kernel_f32(const struct nchw8c_across &J, float A, float B,
            int use_h_parallel, void *code_ptr = nullptr,
            size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);

    void operator()(jit_args_bwd_t *arg) { ker(arg); }
    void (*ker)(jit_args_bwd_t *);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

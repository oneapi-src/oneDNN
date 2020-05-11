/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_BASE_HPP
#define CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_BASE_HPP

#include <memory>
#include "common/c_types_map.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using acc_data_t = float;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;
using namespace Xbyak;
using namespace Xbyak::util;

template <data_type_t d_type>
class jit_avx512_common_lrn_kernel_bwd_t : public jit_generator {
public:
    jit_avx512_common_lrn_kernel_bwd_t(
            float alpha, float beta, void *code_ptr, size_t code_size);

    using data_t = typename prec_traits<d_type>::type;

    struct jit_args_bwd_t {
        jit_args_bwd_t();
        const data_t *src, *diff_dst, *ws0, *ws1;
        data_t *diff_src;
        static const int32_t mask[20];
        const int32_t *mask_ptr;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_bwd);
    void (*ker)(jit_args_bwd_t *);
    void operator()(jit_args_bwd_t *arg) { ker(arg); }

protected:
    static inline Zmm zreg(int irb, int i) { return Zmm(irb * 7 + i); };
    static inline Ymm yreg(int irb, int i) { return Ymm(irb * 7 + i); };
    static inline Xmm xreg(int irb, int i) { return Xmm(irb * 7 + i); };

    void store_data(bool nt, const Address addr, Zmm zr);
    void load_data(Xmm reg, const Address p);

    Reg64 src_ = rax;
    Reg64 diffsrc_ = r8;
    Reg64 diffdst_ = r9;
    Reg64 workspace0_ = rdx;
    Reg64 workspace1_ = rsi;
    Reg64 imm_addr64_ = rbx;
    Reg64 param_ = abi_param1;
    Zmm znalphabeta_ = zmm0;
    Xmm xnalphabeta_ = xmm0;

    Zmm bf16_emu_reserv_1_ = Zmm(28);
    Zmm bf16_emu_reserv_2_ = Zmm(29);
    Reg64 bf16_emu_scratch_ = rax;
    Zmm bf16_emu_reserv_3_ = Zmm(30);
    Zmm bf16_emu_reserv_4_ = Zmm(31);

    static constexpr int z_tmp_ = 7;

    static constexpr int zdiffdst_ = 5;
    static constexpr int zdiffsrc_ = 6;
    static constexpr int zsrc_ = 1;

    static constexpr int za_ = 1;
    static constexpr int zb_ = 2;
    static constexpr int zd_ = 3;
    static constexpr int ze_ = 4;
    static constexpr int zws0_ = 2;

    float nalphabeta_;
    const bool emulateBfloat_;
    const int reg_block_;
    static constexpr int vlen_ = d_type == bf16 ? 32 : 64;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

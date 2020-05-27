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

#ifndef CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_BLOCKED_HPP
#define CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_BLOCKED_HPP

#include "common/c_types_map.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/lrn/jit_avx512_common_lrn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;
using namespace Xbyak;
using namespace Xbyak::util;

template <data_type_t d_type>
class jit_avx512_common_lrn_kernel_bwd_blocked_t : public jit_generator {

public:
    using data_t = typename prec_traits<d_type>::type;

    struct jit_args_bwd_t {
        const data_t *src, *diff_dst, *ws0, *ws1;
        data_t *diff_src;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_bwd_blocked_t)

    void (*ker)(jit_args_bwd_t *);
    void operator()(jit_args_bwd_t *arg) { ker(arg); }

    jit_avx512_common_lrn_kernel_bwd_blocked_t(const struct nChw16c_across_t &J,
            float A, float B, int use_h_parallel, void *code_ptr = nullptr,
            size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);
    ~jit_avx512_common_lrn_kernel_bwd_blocked_t() { delete bf16_emu_; }

private:
    void compute_loop(int loop_size_param, int prefetchL1, int prefetchL2);

    int xmm_size, zmm_size, buffer_block, buffer_nest_offset, src_prev_offset,
            vlen, reg_block;
    int HW, W;
    across_version version;

    Reg64 src = rax;
    Reg64 diffsrc = r8;
    Reg64 diffdst = r9;
    Reg64 workspace0 = rdx;
    Reg64 workspace1 = rsi;
    Reg64 imm_addr64 = rbx;
    Reg64 param = abi_param1;
    Zmm znalphabeta = zmm0;
    Xmm xnalphabeta = xmm0;

    Reg64 t = rsp;
    Reg64 hw = r10;
    Zmm bf16_emu_reserv_1 = Zmm(28);
    Zmm bf16_emu_reserv_2 = Zmm(29);
    Reg64 bf16_emu_scratch = rax;
    Zmm bf16_emu_reserv_3 = Zmm(30);
    Zmm bf16_emu_reserv_4 = Zmm(31);

    const int xws1_prev = 1;
    const int xdiffdst_prev = 2;
    const int zws1 = 1;

    const int zsrc = 1;
    const int zdiffdst = 5;
    const int zdiffsrc = 6;

    const int xws1_next = 1;
    const int xdiffdst_next = 3;

    const int za = 1;
    const int zb = 2;
    const int zd = 3;
    const int ze = 4;
    const int zws0 = 2;

    float nalphabeta;

    int use_h_parallelism;
    bf16_emulation_t *bf16_emu_;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

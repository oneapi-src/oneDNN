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

#include <numeric>
#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_base.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

template <data_type_t d_type>
const int32_t
        jit_avx512_common_lrn_kernel_bwd_t<d_type>::jit_args_bwd_t::mask[48]
        = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0};

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_bwd_t<d_type>::jit_args_bwd_t::jit_args_bwd_t()
    : src(nullptr)
    , diff_dst(nullptr)
    , ws0(nullptr)
    , ws1(nullptr)
    , diff_src(nullptr)
    , mask_ptr(&mask[16]) {}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_t<d_type>::load_data(
        Xmm reg, const Address p) {
    if (d_type == bf16) {
        this->vpmovzxwd(reg, p);
        this->vpslld(reg, reg, 0x10);
    } else
        this->vmovups(reg, p);
};

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_t<d_type>::store_data(
        bool nt, const Address addr, Zmm zr) {
    if (d_type == bf16) {
        Ymm yr = Ymm(zr.getIdx());
        if (mayiuse(avx512_core_bf16))
            vcvtneps2bf16(yr, zr);
        else
            bf16_emu_->vcvtneps2bf16(yr, zr);
        vmovdqu16(addr, yr);
    } else if (nt)
        uni_vmovntps(addr, zr);
    else
        uni_vmovups(addr, zr);
}

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_bwd_t<d_type>::jit_avx512_common_lrn_kernel_bwd_t(
        float alpha, float beta, int local_size, void *code_ptr,
        size_t code_size)
    : jit_generator(code_ptr, code_size)
    , local_size_ {local_size - !(local_size % 2)}
    , z_prev_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(), 3);
        return v;
    }()}
    , z_next_ {[this]() {
        std::vector<int> v(this->local_size_ / 2);
        std::iota(v.begin(), v.end(), 3 + this->local_size_ / 2);
        return v;
    }()}
    , nalphabeta_(-2 * alpha * beta)
    , emulateBfloat_(d_type == bf16 && !mayiuse(avx512_core_bf16))
    , reg_block_ {(d_type == bf16 && !mayiuse(avx512_core_bf16) ? 26 : 30)
              / (std::max(this->local_size_ + 2, 7))} {

    const int regs_used_per_block = std::max(this->local_size_ + 2, 7);
    zreg = [regs_used_per_block](int irb, int i) {
        return Zmm(irb * regs_used_per_block + i);
    };
    yreg = [regs_used_per_block](int irb, int i) {
        return Ymm(irb * regs_used_per_block + i);
    };
    xreg = [regs_used_per_block](int irb, int i) {
        return Xmm(irb * regs_used_per_block + i);
    };

    if (emulateBfloat_) {
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                bf16_emu_reserv_1_, bf16_emu_reserv_2_, bf16_emu_reserv_3_,
                bf16_emu_scratch_, bf16_emu_reserv_4_);
        bf16_emu_->init_vcvtneps2bf16();
    }
}

template class jit_avx512_common_lrn_kernel_bwd_t<f32>;
template class jit_avx512_common_lrn_kernel_bwd_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

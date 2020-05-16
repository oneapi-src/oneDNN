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

#include "cpu/x64/lrn/jit_avx512_common_lrn_fwd_base.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

static constexpr int acc_size = sizeof(acc_data_t);

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_t<d_type>::load_data(
        Xmm reg, const Address p, bool from_stack) {
    this->vmovups(reg, p);
};

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_t<d_type>::load_tail(int tail_value,
        Reg64 src, int src_mem_offset, int dst_stack_offset,
        int tmp_load_to_stack_idx_tail) {

    const auto load_tail_simd = [&](Xmm tmp_reg, int vlen) {
        this->load_data(tmp_reg, this->EVEX_compress_addr(src, src_mem_offset));
        this->vmovups(this->EVEX_compress_addr(rsp, dst_stack_offset), tmp_reg);
        dst_stack_offset += vlen * acc_size;
        src_mem_offset += vlen * acc_size;
        tail_value -= vlen;
    };

    if (tail_value >= 8)
        load_tail_simd(this->yreg(0, tmp_load_to_stack_idx_tail), 8);
    if (tail_value >= 4)
        load_tail_simd(this->xreg(0, tmp_load_to_stack_idx_tail), 4);

    for (int i = 0; i < tail_value; ++i) {

            this->vmovss(this->xreg(0, tmp_load_to_stack_idx_tail),
                    this->EVEX_compress_addr(src, src_mem_offset));

        this->vmovss(ptr[rsp + dst_stack_offset],
                this->xreg(0, tmp_load_to_stack_idx_tail));

        dst_stack_offset += acc_size;
        src_mem_offset += acc_size;
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_t<d_type>::store_data(
        const Address addr, Zmm zr, Ymm yr) {
    this->vmovups(addr, zr);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_t<d_type>::store_tail(int tail_value,
        Zmm src, Reg64 dst, int dst_mem_offset, int tmp_stack_offset,
        int tmp_idx) {

    this->store_data(this->EVEX_compress_addr(rsp, tmp_stack_offset), src,
            this->yreg(0, tmp_idx));

    const auto store_tail_simd = [&](Xmm tmp_reg, int vlen) {
        this->vmovups(tmp_reg, this->EVEX_compress_addr(rsp, tmp_stack_offset));
        this->vmovups(this->EVEX_compress_addr(dst, dst_mem_offset), tmp_reg);
        tmp_stack_offset += vlen * acc_size;
        dst_mem_offset += vlen * acc_size;
        tail_value -= vlen;
    };

    if (tail_value >= 8) store_tail_simd(this->yreg(0, tmp_idx), 8);
    if (tail_value >= 4) store_tail_simd(this->xreg(0, tmp_idx), 4);

    for (int i = 0; i < tail_value;
            ++i, tmp_stack_offset += acc_size, dst_mem_offset += acc_size) {
        this->vmovss(this->xreg(0, tmp_idx),
                this->EVEX_compress_addr(rsp, tmp_stack_offset));
        this->vmovss(this->EVEX_compress_addr(dst, dst_mem_offset),
                this->xreg(0, tmp_idx));
    }
}

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_fwd_t<d_type>::jit_avx512_common_lrn_kernel_fwd_t(
        prop_kind_t prop_kind, float alpha, float k, void *code_ptr,
        size_t code_size)
    : jit_generator(code_ptr, code_size)
    , pk_(prop_kind)
    , alpha_(alpha)
    , k_(k)
    , reg_block_(4) {}

template <data_type_t d_type>
constexpr int
        jit_avx512_common_lrn_kernel_fwd_t<d_type>::jit_args_fwd_t::mask[20];

template class jit_avx512_common_lrn_kernel_fwd_t<f32>;
template class jit_avx512_common_lrn_kernel_fwd_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx2_relu_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_relu_call_s, field)

#define ymm_ns Ymm(15)
#define ymm_zero Ymm(14)

#define ymm_src Ymm(0)
#define xmm_src Xmm(0)
#define ymm_dst Ymm(1)
#define xmm_dst Xmm(1)
#define ymm_mask Ymm(2)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

void jit_avx2_relu_kernel_f32::step(bool vectorize, size_t shift) {
    if (vectorize) {
        vmovups(ymm_src, ptr[reg_src + shift]);
    } else {
        movss(xmm_src, ptr[reg_src + shift]);
    }

    vmulps(ymm_dst, ymm_src, ymm_ns);
    vcmpgtps(ymm_mask, ymm_src, ymm_zero);
    vblendvps(ymm_dst, ymm_dst, ymm_src, ymm_mask);

    if (vectorize) {
        vmovups(ptr[reg_dst + shift], ymm_dst);
    } else {
        movss(ptr[reg_dst + shift], xmm_dst);
    }
}

void jit_avx2_relu_kernel_f32::generate() {
    using Xbyak::Ymm;
    this->preamble();

    mov(reg_src, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[this->param1 + GET_OFF(dst)]);
    mov(imm_addr64, reinterpret_cast<size_t>(&jrp.negative_slope));

    vbroadcastss(ymm_ns, ptr[imm_addr64]);

    vxorps(ymm_zero, ymm_zero, ymm_zero);

    const size_t vector_shift = jrp.vector_length * sizeof(float);
    mov(reg_main_loop_iterator, ptr[this->param1 + GET_OFF(main_loop_iters)]);

    L(".relu_main_loop");
    cmp(reg_main_loop_iterator, 0);
    je(".relu_remainder", T_NEAR);
    for (size_t uf = 0; uf < jrp.unroll_factor; uf++) {
        step(true, uf * vector_shift);
    }
    add(reg_src, jrp.unroll_factor * vector_shift);
    add(reg_dst, jrp.unroll_factor * vector_shift);
    dec(reg_main_loop_iterator);
    jmp(".relu_main_loop", T_NEAR);

    L(".relu_remainder");

    mov(reg_remainder, ptr[this->param1 + GET_OFF(process_remainder)]);
    cmp(reg_remainder, 0);
    je(".relu_end", T_NEAR);
    const size_t vectorized_remainder_size =
        jrp.remainder_size - jrp.remainder_main_loop_iters * jrp.unrolled_size;
    const size_t remainder_vectors =
        vectorized_remainder_size / jrp.vector_length;
    for (size_t uf = 0; uf < remainder_vectors; uf++) {
        step(true, uf * vector_shift);
    }

    add(reg_src, remainder_vectors * vector_shift);
    add(reg_dst, remainder_vectors * vector_shift);
    for (size_t uf = 0; uf < jrp.remainder_size % jrp.vector_length; uf++) {
        step(false, uf * sizeof(float));
    }

    L(".relu_end");
    this->postamble();
}

status_t jit_avx2_relu_kernel_f32::init_conf(jit_relu_conf_t &jrp,
        const relu_desc_t &rd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &dst_d, float negative_slope)
{
    jrp.vector_length = 8;
    jrp.unroll_factor = 4;
    jrp.jit_n_runs = 256; // TODO: use runtime info about available threads

    jrp.n_elems = src_d.nelems();

    jrp.unrolled_size = jrp.vector_length * jrp.unroll_factor;
    jrp.main_loop_iters = jrp.n_elems / (jrp.unrolled_size * jrp.jit_n_runs);

    jrp.block_size = jrp.unrolled_size * jrp.main_loop_iters;

    jrp.remainder_size = jrp.block_size == 0
        ? jrp.n_elems : jrp.n_elems % jrp.block_size;
    jrp.remainder_main_loop_iters = jrp.remainder_size / jrp.unrolled_size;

    jrp.negative_slope = rd.negative_slope;

    return status::success;
}

inline Xbyak::Address jit_avx2_relu_kernel_f32::get_address(
        Xbyak::Reg64 base, int offset) {
    using Xbyak::Ymm;
    return YWORD[base + offset];
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

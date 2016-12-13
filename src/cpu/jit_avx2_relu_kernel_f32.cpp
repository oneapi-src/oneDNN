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

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

void jit_avx2_relu_kernel_f32::step(bool vectorize, int shift) {
    if (vectorize) {
        vmovups(ymm_from, ptr[reg_from + shift]);
        if (jrp.isBackward)
            vmovups(ymm_for_comparison, ptr[reg_for_comparison + shift]);
    } else {
        movss(xmm_from, ptr[reg_from + shift]);
        if (jrp.isBackward)
            movss(xmm_for_comparison, ptr[reg_for_comparison + shift]);
    }

    vmulps(ymm_to, ymm_from, ymm_ns);
    vcmpgtps(ymm_mask, ymm_for_comparison, ymm_zero);
    vblendvps(ymm_to, ymm_to, ymm_from, ymm_mask);

    if (vectorize) {
        vmovups(ptr[reg_to + shift], ymm_to);
    } else {
        movss(ptr[reg_to + shift], xmm_to);
    }
}

void jit_avx2_relu_kernel_f32::setupRegisters() {
    reg_from = rax;
    reg_for_comparison = jrp.isBackward ? rbx : reg_from;
    reg_to = rcx;
    reg_main_loop_iterator = rdx;
    reg_remainder = rdi;
    imm_addr64 = rsi;

    ymm_ns = reg_ymm(15);
    ymm_zero = reg_ymm(14);
    ymm_mask = reg_ymm(13);

    ymm_from = reg_ymm(0);
    xmm_from = reg_xmm(0);
    ymm_for_comparison = jrp.isBackward ? reg_ymm(1) : ymm_from;
    xmm_for_comparison = jrp.isBackward ? reg_xmm(1) : xmm_from;
    ymm_to = reg_ymm(2);
    xmm_to = reg_xmm(2);
}

void jit_avx2_relu_kernel_f32::generate() {
    using Xbyak::Ymm;
    this->preamble();

    setupRegisters();

    mov(reg_for_comparison, ptr[this->param1 + GET_OFF(for_comparison)]);
    mov(reg_from, ptr[this->param1 + GET_OFF(from)]);
    mov(reg_to, ptr[this->param1 + GET_OFF(to)]);
    mov(imm_addr64, reinterpret_cast<size_t>(&jrp.negative_slope));

    vbroadcastss(ymm_ns, ptr[imm_addr64]);

    vxorps(ymm_zero, ymm_zero, ymm_zero);

    const int vector_shift = jrp.vector_length * sizeof(float);
    mov(reg_main_loop_iterator, ptr[this->param1 + GET_OFF(main_loop_iters)]);

    L(".relu_main_loop");
    cmp(reg_main_loop_iterator, 0);
    je(".relu_remainder", T_NEAR);
    for (int uf = 0; uf < jrp.unroll_factor; uf++) {
        step(true, uf * vector_shift);
    }
    add(reg_from, jrp.unroll_factor * vector_shift);
    add(reg_to, jrp.unroll_factor * vector_shift);
    if (jrp.isBackward)
        add(reg_for_comparison, jrp.unroll_factor * vector_shift);
    dec(reg_main_loop_iterator);
    jmp(".relu_main_loop", T_NEAR);

    L(".relu_remainder");

    mov(reg_remainder, ptr[this->param1 + GET_OFF(process_remainder)]);
    cmp(reg_remainder, 0);
    je(".relu_end", T_NEAR);
    const int vectorized_remainder_size =
        jrp.remainder_size - jrp.remainder_main_loop_iters * jrp.unrolled_size;
    const int remainder_vectors =
        vectorized_remainder_size / jrp.vector_length;
    for (int uf = 0; uf < remainder_vectors; uf++) {
        step(true, uf * vector_shift);
    }

    add(reg_from, remainder_vectors * vector_shift);
    add(reg_to, remainder_vectors * vector_shift);
    if (jrp.isBackward)
        add(reg_for_comparison, remainder_vectors * vector_shift);
    for (int uf = 0; uf < jrp.remainder_size % jrp.vector_length; uf++) {
        step(false, uf * sizeof(float));
    }

    L(".relu_end");
    this->postamble();
}

status_t jit_avx2_relu_kernel_f32::init_conf(jit_relu_conf_t &jrp,
        const relu_desc_t &rd, const memory_desc_wrapper &data_d,
        bool isBackward)
{
    jrp.vector_length = 8;
    jrp.unroll_factor = 4;
    jrp.jit_n_runs = 256; // TODO: use runtime info about available threads

    jrp.n_elems = data_d.nelems();

    jrp.unrolled_size = jrp.vector_length * jrp.unroll_factor;
    jrp.main_loop_iters = jrp.n_elems / (jrp.unrolled_size * jrp.jit_n_runs);

    jrp.block_size = jrp.unrolled_size * jrp.main_loop_iters;

    jrp.remainder_size = jrp.block_size == 0
        ? jrp.n_elems : jrp.n_elems % jrp.block_size;
    jrp.remainder_main_loop_iters = jrp.remainder_size / jrp.unrolled_size;

    jrp.negative_slope = rd.negative_slope;
    jrp.isBackward = isBackward;

    return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

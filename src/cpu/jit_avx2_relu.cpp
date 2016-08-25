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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx2_relu.hpp"
#include "jit_generator.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

enum { VECTOR_LENGTH = 8, UNROLLING_FACTOR = 4, JIT_N_RUNS = 1024 };
typedef struct {
    const float *src;
    float *dst;
} jit_args_t;

template <impl::precision_t prec>
struct jit_avx2_relu<prec>::xbyak_relu : public jit_generator {
    Xbyak::Reg64 src = rax;
    Xbyak::Reg64 dst = r8;
    Xbyak::Reg64 main_loop_iterator = r9;
    Xbyak::Reg64 imm_addr64 = rbx;

    Xbyak::Ymm yns = ymm15;
    Xbyak::Ymm yzero = ymm14;
    Xbyak::Ymm ysrc = ymm0;
    Xbyak::Ymm ydst = ymm1;
    Xbyak::Ymm ymask = ymm2;

    Xbyak::Xmm xsrc = xmm0;
    Xbyak::Xmm xdst = xmm1;

    xbyak_relu(
        float *run_time_ptr_negative_slope,
        uint32_t compile_time_main_loop_iterations,
        size_t compile_time_reminder,
        void *code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
    {
        this->preamble();

        mov(src, ptr[this->param1 + 0]);
        mov(dst, ptr[this->param1 + 8]);

        mov(imm_addr64, reinterpret_cast<size_t>(run_time_ptr_negative_slope));
        vbroadcastss(yns, ptr[imm_addr64]);

        vxorps(yzero, yzero, yzero);

        auto ker = [&](bool is_vectorized, size_t shift) {
            if (is_vectorized) {
                vmovups(ysrc, ptr[src + shift]);
            } else {
                movss(xsrc, ptr[src + shift]);
                vzeroupper();
            }

            vmulps(ydst, ysrc, yns);
            vcmpgtps(ymask, ysrc, yzero);
            vblendvps(ydst, ydst, ysrc, ymask);

            if (is_vectorized) {
                vmovups(ptr[dst + shift], ydst);
            } else {
                movss(ptr[dst + shift], xdst);
            }
        };

        const size_t vector_shift = VECTOR_LENGTH * sizeof(prec);
        if (compile_time_main_loop_iterations != 0) {
            mov(main_loop_iterator, compile_time_main_loop_iterations); 

            L(".relu_main_loop");
            for (size_t uf = 0; uf < UNROLLING_FACTOR; uf++) {
                ker(true, uf * vector_shift);
            }
            add(src, UNROLLING_FACTOR * vector_shift);
            add(dst, UNROLLING_FACTOR * vector_shift);
            dec(main_loop_iterator);
            cmp(main_loop_iterator, 0);
            jne(".relu_main_loop", T_NEAR);
        }

        const size_t reminder_vectors = compile_time_reminder / VECTOR_LENGTH;
        for (size_t uf = 0; uf < reminder_vectors; uf++) {
            ker(true, uf * vector_shift);
        }

        add(src, reminder_vectors * vector_shift);
        add(dst, reminder_vectors * vector_shift);
        for (size_t uf = 0; uf < compile_time_reminder % VECTOR_LENGTH; uf++) {
            ker(false, uf * sizeof(prec));
        }

        this->postamble();
    }
};

template <impl::precision_t prec>
jit_avx2_relu<prec>::jit_avx2_relu(const relu_primitive_desc_t &rpd,
    const primitive_at_t *inputs, const primitive *outputs[])
    : relu<jit_avx2_relu<prec>>(rpd, inputs, outputs)
    , jit_negative_slope(rpd.relu_desc.negative_slope) {

    const memory_desc_wrapper src_d(this->_rpd.src_primitive_desc.memory_desc);

    const size_t n_elems = src_d.nelems();

    size_t jit_iterations =
        n_elems / UNROLLING_FACTOR / VECTOR_LENGTH / JIT_N_RUNS;
    jit_iterations = jit_iterations < 1 ? 1 : jit_iterations;

    chunk_size = VECTOR_LENGTH * UNROLLING_FACTOR * jit_iterations;
    n_chunks = n_elems / chunk_size;
    n_reminder_elems = n_elems % chunk_size;

    const size_t reminder_loop_iterations =
        n_reminder_elems / VECTOR_LENGTH / UNROLLING_FACTOR;
    const size_t jit_reminder = n_reminder_elems -
        reminder_loop_iterations * VECTOR_LENGTH * UNROLLING_FACTOR;

    typedef void (*kernel_t)(const void *);
    this->jit_relu = new xbyak_relu(&this->jit_negative_slope,
            jit_iterations, 0);
    this->ker_main = reinterpret_cast<kernel_t>(
            const_cast<uint8_t*>(this->jit_relu->getCode()));

    this->jit_relu_reminder = new xbyak_relu(&this->jit_negative_slope,
            reminder_loop_iterations, jit_reminder);
    this->ker_reminder = reinterpret_cast<kernel_t>(
            const_cast<uint8_t*>(this->jit_relu_reminder->getCode()));
}

template <impl::precision_t prec>
jit_avx2_relu<prec>::~jit_avx2_relu() {
    delete this->jit_relu;
    delete this->jit_relu_reminder;
}

template <impl::precision_t prec>
status_t jit_avx2_relu<prec>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(
            this->input()[0].primitive->output()[
            this->input()[0].output_index]->memory_const());
    auto dst = reinterpret_cast<data_t *>(this->output()[0]->memory());

    const memory_desc_wrapper src_d(this->_rpd.src_primitive_desc.memory_desc);
    const memory_desc_wrapper dst_d(this->_rpd.dst_primitive_desc.memory_desc);

    src += src_d.blocking_desc().offset_padding;
    dst += dst_d.blocking_desc().offset_padding;

#   pragma omp parallel for
    for (size_t c = 0; c < n_chunks + 1; ++c) {
        jit_args_t args;
        args.src = &src[c * chunk_size];
        args.dst = &dst[c * chunk_size];
        if (c != n_chunks) {
            ker_main(&args);
        } else if (n_reminder_elems != 0) {
            ker_reminder(&args);
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t jit_avx2_relu<prec>::set_default_parameters(relu_desc_t &relu_d) {
    if (relu_d.src_desc.format == any)
        CHECK(types::set_default_format<prec>(relu_d.src_desc, nChw8c));
    if (relu_d.dst_desc.format == any)
        CHECK(types::set_default_format<prec>(relu_d.dst_desc, nChw8c));
    return status::success;
}

template <impl::precision_t prec>
status_t jit_avx2_relu<prec>::constraint(const relu_desc_t &relu_d) {
    const memory_desc_wrapper src_d(relu_d.src_desc);
    const memory_desc_wrapper dst_d(relu_d.dst_desc);

    bool args_ok = true
        && relu_d.prop_kind == prop_kind::forward
        && src_d.similar_to(dst_d)
        && src_d.is_dense();

    return args_ok ? success : unimplemented;
}

template <impl::precision_t prec>
const primitive_impl jit_avx2_relu<prec>::implementation = {
    jit_avx2_relu<prec>::create
};

template class jit_avx2_relu<precision::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

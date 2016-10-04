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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx2_relu.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

enum { VECTOR_LENGTH = 8, UNROLLING_FACTOR = 4, JIT_N_RUNS = 1024 };
struct jit_args_t {
    const float *src;
    const float *dst;
};

struct jit_avx2_relu_fwd_t::xbyak_relu: public jit_generator {
    xbyak_relu(float *run_time_ptr_negative_slope,
            int compile_time_main_loop_iterations,
            size_t compile_time_reminder, void *code_ptr = nullptr,
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

        const size_t vector_shift = VECTOR_LENGTH * sizeof(float);
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
            ker(false, uf * sizeof(float));
        }

        this->postamble();

        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(
                    this->getCode()));
    }

    void operator()(const jit_args_t *args) { (*ker_)(args); }

private:
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

    void (*ker_)(const jit_args_t *args);
};

jit_avx2_relu_fwd_t::jit_avx2_relu_fwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {
    const memory_desc_wrapper data_d(conf_.src_pd());
    n_elems_ = data_d.nelems();

    const size_t step = VECTOR_LENGTH * UNROLLING_FACTOR;
    const size_t jit_iters = nstl::max<size_t>(1,
            n_elems_ / (step * JIT_N_RUNS));

    chunk_size_ = step * jit_iters;

    const size_t n_rem_elems = n_elems_ % chunk_size_;
    const size_t rem_loop_iters = n_rem_elems / step;
    const size_t jit_reminder = n_rem_elems - rem_loop_iters * step;

    negative_slope_ = conf_.desc()->negative_slope;
    ker_ = new xbyak_relu(&negative_slope_, jit_iters, 0);
    ker_rem_ = new xbyak_relu(&negative_slope_, rem_loop_iters, jit_reminder);
}

jit_avx2_relu_fwd_t::~jit_avx2_relu_fwd_t() {
    delete ker_;
    if (ker_rem_) delete ker_rem_;
}

void jit_avx2_relu_fwd_t::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());

    src += data_d.blocking_desc().offset_padding;
    dst += data_d.blocking_desc().offset_padding;

    const int n_chunks = n_elems_ / chunk_size_;
    const int n_reminder_elems = n_elems_ % chunk_size_;

#   pragma omp parallel for schedule(static)
    for (int n = 0; n < n_chunks + 1; ++n) {
        jit_args_t args;
        args.src = &src[n * chunk_size_];
        args.dst = &dst[n * chunk_size_];
        if (n != n_chunks) {
            (*ker_)(&args);
        } else if (n_reminder_elems != 0) {
            (*ker_rem_)(&args);
        }
    }
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

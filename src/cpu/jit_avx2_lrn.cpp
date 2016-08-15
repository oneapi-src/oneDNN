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

#include "mkl_dnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx2_lrn.hpp"
#include "type_helpers.hpp"

#if 1
#define XBYAK64 1
#define XBYAK_NO_OP_NAMES 1
#include "xbyak.h"
#include "utils_xbyak.hpp"

class xbyak_lrn : public Xbyak::CodeGenerator
{
public:
    Xbyak::Reg64 args = Xbyak::util::cdecl_param1;
    Xbyak::Reg64 src = rax;
    Xbyak::Reg64 dst = r8;
    Xbyak::Reg64 scratch = rdx;
    Xbyak::Reg64 hw = r9;
    Xbyak::Reg64 t = rsp;

    Xbyak::Ymm yalpha = ymm0;
    Xbyak::Ymm ysrc_prev = ymm1;
    Xbyak::Ymm ysrc = ymm2;
    Xbyak::Ymm ysrc_next = ymm3;
    Xbyak::Ymm ya = ymm4;
    Xbyak::Ymm yb = ymm5;
    Xbyak::Ymm yc = ymm2;
    Xbyak::Ymm yd = ymm6;
    Xbyak::Ymm ye = ymm7;
    Xbyak::Ymm yone = ymm8;
    Xbyak::Ymm ysum = ymm9;
    Xbyak::Ymm ysum2 = ymm10;
    Xbyak::Ymm ydst = ymm11;
    Xbyak::Ymm ybase = ymm12;

    void preamble()
    {
        using Xbyak::util::reg_to_preserve;
        size_t nregs = sizeof(reg_to_preserve) / sizeof(reg_to_preserve[0]);
        for (size_t i = 0; i < nregs; ++i) push(Xbyak::Reg64(reg_to_preserve[i]));
    }
    void postamble()
    {
        using Xbyak::util::reg_to_preserve;
        size_t nregs = sizeof(reg_to_preserve) / sizeof(reg_to_preserve[0]);
        for (size_t i = 0; i < nregs; ++i) pop(Xbyak::Reg64(reg_to_preserve[nregs - 1 - i]));
        ret();
    }

    xbyak_lrn(
        float *run_time_ptr_alpha,
        float *run_time_ptr_one,
        uint32_t compile_time_HW,
        int version, // -1 channels 0..7, 1 channels C-8 .. C-1, 0 -- other channels
        void* code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        :
        Xbyak::CodeGenerator(code_size, code_ptr)
    {
            if (compile_time_HW == 0)
            {
                ret();
                return;
            }
            preamble();

            mov(src, ptr[args + 0]);
            mov(dst, ptr[args + 8]);
            mov(scratch, ptr[args + 16]);
            sub(t, 96);
            vbroadcastss(yalpha, ptr[run_time_ptr_alpha]);
            vbroadcastss(yone, ptr[run_time_ptr_one]);
            if (version == -1)
            {
                vxorps(ysrc_prev, ysrc_prev, ysrc_prev);
                vmovups(ptr[t + 0], ysrc_prev);
            }
            if (version == +1)
            {
                vxorps(ysrc_next, ysrc_next, ysrc_next);
                vmovups(ptr[t + 64], ysrc_next);
            }

            mov(hw, compile_time_HW);
            L(".lrn_loop");

            if (version != -1) vmovups(ysrc_prev, ptr[src - compile_time_HW * 32]);
            vmovups(ysrc, ptr[src]);
            if (version != +1) vmovups(ysrc_next, ptr[src + compile_time_HW * 32]);

            if (version != -1) vmovups(ptr[t + 0], ysrc_prev);
            vmovups(ptr[t + 32], ysrc);
            if (version != +1) vmovups(ptr[t + 64], ysrc_next);

            vmovups(ya, ptr[t + 32 - 8]);
            vmovups(yb, ptr[t + 32 - 4]);
            vmovups(yd, ptr[t + 32 + 4]);
            vmovups(ye, ptr[t + 32 + 8]);
            vmulps(ysum, yc, yc);
            vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya*ya
            vfmadd231ps(ysum, yb, yb);
            vfmadd231ps(ysum, yd, yd);
            vfmadd231ps(ysum, ye, ye);

            vfmadd132ps(ysum, yone, yalpha); // ysum <- ysum*yalpha+yone
            vmovaps(ybase, ysum);
            vmulps(ysum2, ysum, ysum);
            vmulps(ysum, ysum, ysum2); // ysum = ybase^3;
            vsqrtps(ysum, ysum);
            vsqrtps(ysum, ysum); // ysum = ybase^0.75
            vdivps(ydst, ysrc, ysum); // ydst = ysrc / ysum
            vmulps(ysum, ysum, ybase); // ysum = ybase ^ 1.75 -- for back prop
            vmovups(ptr[dst], ydst);
            vmovups(ptr[scratch], ysum);

            add(src, 32);
            add(dst, 32);
            add(scratch, 32);
            dec(hw);
            cmp(hw, 0);
            jne(".lrn_loop", T_NEAR);

            add(t, 96);
            postamble();
            return;
        }
};
#endif

#define CHECK(f) do { \
    status_t status = f; \
    if (status != success) return status; \
} while(0)

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::alg_kind;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::memory_format;
using namespace mkl_dnn::impl::primitive_kind;

enum { VECTOR_LENGTH = 8 };
typedef struct {
    const float *src;
    float *dst, *scratch;
} jit_args_t;

template <impl::precision_t prec>
status_t jit_avx2_lrn<prec>::execute_forward() {
    const data_t *src =
        reinterpret_cast<const data_t *>(this->input()[0].primitive->output()[this->input()[0].output_index]->memory_const());
    data_t *scratch =
        reinterpret_cast<data_t *>(this->input()[1].primitive->output()[this->input()[1].output_index]->memory());
    data_t *dst =
        reinterpret_cast<data_t *>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_ppd.src_primitive_desc.memory_desc),
        scratch_d(this->_ppd.scratch_primitive_desc.memory_desc),
        dst_d(this->_ppd.dst_primitive_desc.memory_desc);

    const uint32_t C = src_d.dims()[1];
    const uint32_t HW = src_d.dims()[2]*src_d.dims()[3];

    const uint32_t N = src_d.dims()[0];
#   pragma omp parallel for collapse(2)
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t c8 = 0; c8 < C / VECTOR_LENGTH; ++c8) {
            jit_args_t args;
            args.src     = &src    [n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.dst     = &dst    [n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.scratch = &scratch[n*HW*C + c8 * HW * VECTOR_LENGTH];
            if (c8 == 0)
                ker_hw8_first(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                ker_hw8_last(&args);
            else
                ker_hw8(&args);
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t jit_avx2_lrn<prec>::execute_backward_data() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t jit_avx2_lrn<prec>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkl_dnn::impl::engine &engine) {
    if (op_desc._kind != primitive_kind::lrn)
        return invalid_arguments;
    auto lrn_d = op_desc.lrn;

    // TODO: f32 ?
    if (lrn_d.prop_kind != forward)
        return unimplemented;

    if (lrn_d.alg_kind != lrn_across_channels)
        return unimplemented;

    if (lrn_d.src_desc.tensor_desc.ndims != 4)
        return unimplemented;

    // 0 is mini-batch, 1 is channel, 2 is height, 3 is width
    if (lrn_d.src_desc.tensor_desc.dims[1] % VECTOR_LENGTH != 0)
        return unimplemented;

    if (lrn_d.src_desc.tensor_desc.dims[1] < 2*VECTOR_LENGTH)
        return unimplemented;

    if (lrn_d.beta != 0.75)
        return unimplemented;

    if (lrn_d.local_size != 5)
        return unimplemented;

    /* memory descriptors check and fill-in */
    if (lrn_d.src_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&lrn_d.src_desc,
        &lrn_d.src_desc.tensor_desc, f32, nChw8c));
    if (lrn_d.dst_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&lrn_d.dst_desc,
        &lrn_d.dst_desc.tensor_desc, f32, nChw8c));

    CHECK(lrn_d.src_desc.format == nChw8c ? mkl_dnn_success : mkl_dnn_try_again);
    CHECK(lrn_d.dst_desc.format == nChw8c ? mkl_dnn_success : mkl_dnn_try_again);

    memory_desc_t scratch_desc;
    CHECK(mkl_dnn_memory_desc_init(&scratch_desc,
        &lrn_d.dst_desc.tensor_desc, f32, lrn_d.dst_desc.format));

    /* memory primitive descriptors check */
    memory_primitive_desc_t src_pd, scratch_pd, dst_pd;
    CHECK(mkl_dnn_memory_primitive_desc_init(&src_pd,
        &lrn_d.src_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&dst_pd,
        &lrn_d.dst_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&scratch_pd,
        &scratch_desc, &engine));

    /* final stage */
    lrn_primitive_desc_t ppd;
    for (size_t x = 0; x < sizeof(ppd); ++x) reinterpret_cast<char*>(&ppd)[x] = '\0';
    ppd.base.primitive_kind = lrn;
    ppd.base.engine = &engine;
    ppd.base.implementation = reinterpret_cast<const void*>(&implementation);
    ppd.lrn_desc = lrn_d;
    ppd.src_primitive_desc = src_pd;
    ppd.scratch_primitive_desc = scratch_pd;
    ppd.dst_primitive_desc = dst_pd;

    // if (!lrn_primitive_desc_is_ok(ppd)) return invalid; // ???

    primitive_desc->lrn = ppd;

    return success;
}

template <impl::precision_t prec>
status_t jit_avx2_lrn<prec>::create(primitive **primitive,
        const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const mkl_dnn::impl::primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == lrn);

    auto& ppd = primitive_desc->lrn;
    // TODO: some checks here.

    *primitive = new jit_avx2_lrn(ppd, inputs, outputs);
    return primitive ? success : out_of_memory;
}

template <impl::precision_t prec>
const primitive_impl jit_avx2_lrn<prec>::implementation = {
    jit_avx2_lrn::create, /* .primitive_create */
};

template class jit_avx2_lrn<f32>;

template <impl::precision_t prec>
jit_avx2_lrn<prec>::jit_avx2_lrn(const lrn_primitive_desc_t &ppd,
    const primitive_at_t *inputs, const primitive *outputs[])
    : primitive(ppd, const_cast<impl::engine*>(ppd.base.engine), not_ready)
    , _ppd(_primitive_desc.lrn)
    , jit_alpha(ppd.lrn_desc.alpha/ppd.lrn_desc.local_size)
    , jit_one(1.0)
{
    _input.push_back(inputs[0]);
    _input.push_back(inputs[1]);
    _output.push_back(outputs[0]);
    uint32_t H = ppd.src_primitive_desc.memory_desc.tensor_desc.dims[2];
    uint32_t W = ppd.src_primitive_desc.memory_desc.tensor_desc.dims[3];

    typedef void(*kernel_t)(const void*);
    this->jit_lrn = new xbyak_lrn(&this->jit_alpha, &this->jit_one, H*W, 0);
    this->ker_hw8 = reinterpret_cast<kernel_t>(this->jit_lrn->getCode());

    this->jit_lrn_first = new xbyak_lrn(&this->jit_alpha, &this->jit_one, H*W, -1);
    this->ker_hw8_first = reinterpret_cast<kernel_t>(this->jit_lrn_first->getCode());

    this->jit_lrn_last = new xbyak_lrn(&this->jit_alpha, &this->jit_one, H*W, +1);
    this->ker_hw8_last = reinterpret_cast<kernel_t>(this->jit_lrn_last->getCode());
}

template <impl::precision_t prec>
jit_avx2_lrn<prec>::~jit_avx2_lrn()
{
    delete this->jit_lrn;
    delete this->jit_lrn_first;
    delete this->jit_lrn_last;
}


}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

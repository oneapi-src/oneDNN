/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "cpu/aarch64/cpu_barrier.hpp"

#define CGA64 CodeGeneratorAArch64
namespace xa = Xbyak::Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace simple_barrier {

using namespace Xbyak::Xbyak_aarch64;

#define push64(reg); \
        CGA64::sub(CGA64::sp, CGA64::sp, 8); \
        CGA64::str(reg, xa::ptr(CGA64::sp));

#define pop64(reg); \
        CGA64::ldr(reg, xa::ptr(CGA64::sp)); \
        CGA64::add(CGA64::sp, CGA64::sp, 8);

void jit_t::generate( xa::XReg reg_ctx, xa::XReg reg_nthr ) {
#define BAR_CTR_OFF offsetof(ctx_t, ctr)
#define BAR_SENSE_OFF offsetof(ctx_t, sense)

    xa::LabelAArch64 barrier_exit_label, barrier_exit_restore_label, spin_label;

    CGA64::cmp(reg_nthr, 1);
    CGA64::b(xa::EQ, barrier_exit_label); //jbe(barrier_exit_label);

    push64(reg_tmp);
    push64(reg_tmp_imm);
    push64(reg_tmp_ofs);

    /* take and save current sense */
    CGA64::ldr(reg_tmp, xa::ptr(reg_ctx, static_cast<int32_t>(BAR_SENSE_OFF)));
    push64(reg_tmp);
    CGA64::mov(reg_tmp, 1);


    if(BAR_CTR_OFF == 0){
        CGA64::ldaddal(reg_tmp, reg_tmp, xa::ptr(reg_ctx));
    }else{
        CGA64::add_imm(reg_tmp_ofs, reg_ctx, BAR_CTR_OFF, reg_tmp_imm);
        CGA64::ldaddal(reg_tmp, reg_tmp, xa::ptr(reg_tmp_ofs));
    }
    CGA64::add(reg_tmp, reg_tmp, 1);
    CGA64::cmp(reg_tmp, reg_nthr);
    pop64(reg_tmp); /* restore previous sense */
    CGA64::b(xa::NE, spin_label); //jne(spin_label);

    /* the last thread {{{ */
    if(BAR_CTR_OFF == 0){
        CGA64::mov(reg_tmp_imm, 0);
        CGA64::str(reg_tmp_imm, xa::ptr(reg_ctx));
    }else{
        CGA64::add_imm(reg_tmp_ofs, reg_ctx, BAR_CTR_OFF, reg_tmp_imm);
        CGA64::mov(reg_tmp_imm, 0);
        CGA64::str(reg_tmp_imm, xa::ptr(reg_tmp_ofs));
    }

    // notify waiting threads
    CGA64::mvn(reg_tmp, reg_tmp); //not_(reg_tmp);
    if( BAR_SENSE_OFF == 0 ){
        CGA64::str(reg_tmp, xa::ptr(reg_ctx));
    }else{
        CGA64::add_imm(reg_tmp_ofs, reg_ctx, BAR_SENSE_OFF, reg_tmp_imm);
        CGA64::str(reg_tmp, xa::ptr(reg_tmp_ofs));
    }
    CGA64::b(barrier_exit_restore_label);
    /* }}} the last thread */

    CGA64::L_aarch64(spin_label);
    CGA64::yield();
    if( BAR_SENSE_OFF == 0 ){
        CGA64::ldr(reg_tmp_imm, xa::ptr(reg_ctx));
    }else{
        CGA64::add_imm(reg_tmp_ofs, reg_ctx, BAR_SENSE_OFF, reg_tmp_imm);
        CGA64::ldr(reg_tmp_imm, xa::ptr(reg_tmp_ofs));
    }
    CGA64::cmp(reg_tmp, reg_tmp_imm);
    CGA64::b(xa::EQ, spin_label); //je(spin_label);

    CGA64::L_aarch64(barrier_exit_restore_label);
    pop64(reg_tmp_ofs);
    pop64(reg_tmp_imm);
    pop64(reg_tmp);

    CGA64::L_aarch64(barrier_exit_label);
#undef BAR_CTR_OFF
#undef BAR_SENSE_OFF
}

void barrier(ctx_t *ctx, int nthr) {
    static jit_t j; /* XXX: constructed on load ... */
    j.barrier(ctx, nthr); // barrier
}

} // namespace simple_barrier

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_BRGEMM_CONV_TRANS_KERNEL_HPP
#define CPU_AARCH64_JIT_BRGEMM_CONV_TRANS_KERNEL_HPP

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace jit_sve_core_brgemm_conv_trans_kernel {
struct jit_brgemm_conv_trans_kernel_call_s {
    const void *src;
    const void *dst;
    size_t owb;
    size_t ic;
    size_t t_pad;
    size_t h_count;
    size_t b_pad;
};

struct jit_sve_core_brgemm_conv_trans_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sve_core_brgemm_conv_trans_kernel_t)

    jit_sve_core_brgemm_conv_trans_kernel_t(const jit_brgemm_conv_conf_t &ajcp);

    static int dst_w(const jit_brgemm_conv_conf_t &ajcp, int out_w);

protected:
    jit_brgemm_conv_conf_t jcp;
    dim_t inp_dsz;
    dim_t ic_block_sz;
    dim_t iw_size, dst_w_block, dst_stride;
    dim_t dst_h_offset, dst_w_offset;
    dim_t VL, n_vec, n_tail_vec;

    const XReg inp_ptr = x15;
    const XReg dst_ptr = x14;

    const XReg aux_inp_ptr = x13;
    const XReg aux_dst_ptr = x12;

    const XReg reg_hc = x10;

    const XReg reg_ic = x9;

    const XReg reg_owb = x2; //rdx;

    const XReg kh_over = x8;
    const XReg reg_t_pad = x0; //rax;
    const XReg reg_b_pad = x3; //rbx;

    const XReg reg_tmp = x6; //rsi;

    const PReg ktail_mask = PReg(2);
    const PReg kblock_tail_mask = PReg(3);

    const ZReg zmm_tmp = ZReg(0);
    const ZReg zmm_zero = ZReg(1);

    void load(const ZReg &x, const AdrNoOfs &addr);
    void load(const ZReg &x, const PReg &p, const AdrNoOfs &addr);

    void store(const AdrNoOfs &addr, const ZReg &x);
    void store(const AdrNoOfs &addr, const PReg &p, const ZReg &x);

    void zero_ic_block(bool is_ic_tail, dim_t dst_off);
    void copy_ic_block(
            bool is_ic_tail, dim_t inp_off, dim_t dst_off, bool do_load);
    void generate() override;
    void copy_ow_block(bool is_ic_tail);
    void copy_ow_block_body(int lpad, int ow_len, int iw_len, bool is_ic_tail);

    int inp_w(int out_w) const;
    int inp_w(int out_w, int kw) const;
    int inp_w_start(int owb) const;
};

struct jit_sve_core_brgemm_conv_rtus_kernel_t
    : jit_sve_core_brgemm_conv_trans_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sve_core_brgemm_conv_rtus_kernel_t)

    jit_sve_core_brgemm_conv_rtus_kernel_t(const jit_brgemm_conv_conf_t &ajcp);

private:
    void generate() override;
};

} // namespace jit_sve_core_brgemm_conv_trans_kernel

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

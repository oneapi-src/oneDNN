/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
* Copyright 2022-2023 FUJITSU LIMITED
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
#ifndef CPU_AARCH64_JIT_TRANSPOSE_UTILS_HPP
#define CPU_AARCH64_JIT_TRANSPOSE_UTILS_HPP

#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct jit_trans_src_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;
        const void *src_prf;
        const void *tr_src_prf;
        int ch_work;

        /* 1st conv 4fma: backward by weights */
        int nthr_oc_b; /* number of threads process given src image */
        int tr_src_ih_start, tr_src_ih_end; /* thread's transposition bounds */
        simple_barrier::ctx_t *tr_src_bctx; /* transposition synchronization */
    };

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;

    jit_trans_src_t(const jit_conv_conf_t *conf) : conf_(conf) {}
    virtual ~jit_trans_src_t() {}

    const jit_conv_conf_t *conf_;
};

struct jit_src_transpose_s {
    size_t size;
    const void *src;
    const void *tr_src;
    const void *src_prf;
    const void *tr_src_prf;
};

struct jit_trans_dst_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;
        const void *src_prf;
        const void *tr_src_prf;
        int ch_work;

        /* 1st conv 4fma: backward by weights */
        int nthr_oc_b; /* number of threads process given src image */
        int tr_src_ih_start, tr_src_ih_end; /* thread's transposition bounds */
        simple_barrier::ctx_t *tr_src_bctx; /* transposition synchronization */
    };

    jit_trans_dst_t(const jit_conv_conf_t *conf) : conf_(conf) {}
    virtual ~jit_trans_dst_t() {}

    virtual void operator()(ctx_t *ctx) = 0;
    virtual status_t create_kernel() = 0;
    const jit_conv_conf_t *conf_;
};

struct jit_transpose4x16_src_t {
    int src_pf0_distance;
    int tr_src_pf0_distance;
    bool src_pf1;
    bool tr_src_pf1;
};

struct jit_transpose4x16_src : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_transpose4x16_src)

    jit_transpose4x16_src(const jit_1x1_conv_conf_t *aparams,
            jit_transpose4x16_src_t *tparams_)
        : params(aparams), tparams(tparams_) {}

    const jit_1x1_conv_conf_t *params;
    const jit_transpose4x16_src_t *tparams;

    static const int transpose_size = 4;

private:
    static const int typesize = sizeof(float);

    int src_stride = 0, tr_src_stride = 0;

    Xbyak_aarch64::XReg imm_addr64 = x3;

    Xbyak_aarch64::PReg kF0 = p1;
    Xbyak_aarch64::PReg kCC = p2;
    Xbyak_aarch64::PReg k33 = p3;
    Xbyak_aarch64::PReg kFFFF = p4;

    Xbyak_aarch64::ZReg vidx01 = z31;
    Xbyak_aarch64::ZReg vidx10 = z30;
    Xbyak_aarch64::ZReg vidx1 = z29;
    Xbyak_aarch64::ZReg vidxP = z28;

    Xbyak_aarch64::XReg reg_src = x8;
    Xbyak_aarch64::XReg reg_tr_src = x9;
    Xbyak_aarch64::XReg reg_src_prf = x10;
    Xbyak_aarch64::XReg reg_tr_src_prf = x11;
    Xbyak_aarch64::XReg reg_loop = x12;
    Xbyak_aarch64::XReg reg_tr_src_tmp = x13;
    Xbyak_aarch64::WReg regw_tmp = w14;

    void transpose_block(int ur, int nrows);
    void transpose(int nrows);
    void generate() override;
};

struct jit_diff_wei_trans_to_vnni_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_diff_wei_trans_to_vnni_t)

    jit_diff_wei_trans_to_vnni_t(const int &kd, const int &kh, const int &kw,
            const int &ic_block, const int &oc_block)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, sve_512)
        , kd_(kd)
        , kh_(kh)
        , kw_(kw)
        , ic_block_(ic_block)
        , oc_block_(oc_block) {}

    ~jit_diff_wei_trans_to_vnni_t() {}

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    const int kd_, kh_, kw_;
    const int ic_block_, oc_block_;

private:
    void generate() override;
};

jit_trans_src_t *create_trans_src(const jit_conv_conf_t *conf);
jit_trans_dst_t *create_trans_dst(const jit_conv_conf_t *conf);

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

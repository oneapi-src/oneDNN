/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef CPU_JIT_UNI_LAYER_NORMALIZATION_KERNELS_HPP
#define CPU_JIT_UNI_LAYER_NORMALIZATION_KERNELS_HPP

#include "cpu_layer_normalization_pd.hpp"
#include "jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

class statistics_kernel_t : jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_uni_layer_normalization_fwd_t::statistics_kernel);
    statistics_kernel_t(const layer_normalization_pd_t *pd)
        : C_(pd->norm_axis()), ker_(nullptr) {
        if (mayiuse(avx2)) { generate(); }
    }
    ~statistics_kernel_t() {}

    void operator()(const float *src, float *mean, float *var) {
        if (ker_) {
            ker_args args;
            args.src = src;
            args.mean = mean;
            args.var = var;
            ker_(&args);
        } else {
            float v_mean = 0;
            PRAGMA_OMP_SIMD(reduction(+ : v_mean))
            for (dim_t c = 0; c < C_; ++c) {
                v_mean += src[c];
            }
            v_mean /= C_;

            float v_variance = 0;
            PRAGMA_OMP_SIMD(reduction(+ : v_variance))
            for (dim_t c = 0; c < C_; ++c) {
                auto m = src[c] - v_mean;
                v_variance += m * m;
            }
            v_variance /= C_;

            *mean = v_mean;
            *var = v_variance;
        }
    }

private:
    int C_;
    int unroll_factor_ = 8;
    int simd_w_ = 8;

    struct ker_args {
        const float *src;
        float *mean;
        float *var;
    };
    void (*ker_)(const ker_args *args);

    void generate() {
        using namespace Xbyak;

        preamble();
#define PARAM_OFF(x) offsetof(ker_args, x)
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_var, ptr[reg_param + PARAM_OFF(var)]);
#undef PARAM_OFF

        // compute mean
        compute([=](Ymm ymm_dst) { vaddps(ymm_dst, ymm_dst, ymm_src); });
        movss(ptr[reg_mean], Xmm(0));

        //compute var
        vbroadcastss(ymm_mean, Xmm(0));
        compute([=](Ymm ymm_dst) {
            vsubps(ymm_src, ymm_mean, ymm_src);
            vfmadd231ps(ymm_dst, ymm_src, ymm_src);
        });
        movss(ptr[reg_var], Xmm(0));

        postamble();

        ker_ = getCode<decltype(ker_)>();
    }

    void load_src(Xbyak::Ymm &ymm_src, int nelems, size_t offt = 0) {
        if (nelems == 1)
            movss(Xbyak::Xmm(ymm_src.getIdx()), dword[reg_src + offt]);
        else if (nelems == simd_w_)
            vmovups(ymm_src, yword[reg_src + offt]);
        else
            assert(!"unsupported nelems for load src");
    }

    template <typename F>
    void compute(F op) {
        using namespace Xbyak;

        const int C_vecs = C_ / simd_w_;

        vpxor(Ymm(0), Ymm(0), Ymm(0));
        if (C_vecs > 0) {
            const int unroll = C_vecs >= unroll_factor_ ? unroll_factor_ : 1;
            assert(math::is_pow2(unroll));

            for (int i = 1; i < unroll; i++)
                vpxor(Ymm(i), Ymm(i), Ymm(i));

            // unrolled loop
            for (int i = 0; i < C_vecs / unroll; i++)
                for (int j = 0; j < unroll; j++) {
                    load_src(ymm_src, simd_w_,
                            (i * unroll + j) * simd_w_ * sizeof(float));
                    op(Ymm(j));
                }

            // unrolled loop reduction
            int n = unroll;
            while (n > 1) {
                for (int j = 0; j < n / 2; j++)
                    vaddps(Ymm(j), Ymm(j), Ymm(j + n / 2));
                n = n / 2;
            }

            // unrolled loop remainder
            for (int i = utils::rnd_dn(C_vecs, unroll); i < C_vecs; i++) {
                load_src(ymm_src, simd_w_, i * simd_w_ * sizeof(float));
                op(Ymm(0));
            }

            // vector reduction
            Xmm xmm_high = Xmm(1);
            vextractf128(xmm_high, Ymm(0), 1);
            vaddps(Xmm(0), xmm_high, Xmm(0));
            vhaddps(Xmm(0), Xmm(0), Xmm(0));
            vhaddps(Xmm(0), Xmm(0), Xmm(0));
        }

        // vector remainder
        for (int i = utils::rnd_dn(C_, simd_w_); i < C_; i++) {
            load_src(ymm_src, 1, i * sizeof(float));
            op(Ymm(0));
        }

        // scale
        Xmm xmm_tmp = Xmm(ymm_src.getIdx());
        mov(reg_tmp, float2int(C_));
        movq(xmm_tmp, reg_tmp);
        divss(Xmm(0), xmm_tmp);
    };

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_src = rdx;
    Xbyak::Reg64 reg_mean = rbx;
    Xbyak::Reg64 reg_var = rbp;
    Xbyak::Reg64 reg_tmp = rax;

    // vector registers 0 .. unroll_factor_ are reseved for unrolling
    Xbyak::Ymm ymm_src = Xbyak::Ymm(14);
    Xbyak::Ymm ymm_mean = Xbyak::Ymm(15);
};

class data_kernel_t : jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_uni_layer_normalization_fwd_t::data_kernel);
    data_kernel_t(const layer_normalization_pd_t *pd)
        : C_(pd->norm_axis())
        , use_scaleshift_(pd->use_scaleshift())
        , eps_(pd->desc()->layer_norm_epsilon)
        , ker_(nullptr) {
        if (mayiuse(avx2)) { generate(); }
    }
    ~data_kernel_t() {}
    void operator()(const float *src, float *dst, const float *ss,
            const float *mean, const float *var) {
        if (ker_) {
            ker_args args;
            args.src = src;
            args.dst = dst;
            args.ss = ss;
            args.mean = mean;
            float inv_sqrtvar = 1. / sqrtf(*var + eps_);
            args.inv_sqrtvar = &inv_sqrtvar;
            ker_(&args);
        } else {
            float inv_sqrtvar = 1. / sqrtf(*var + eps_);
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; ++c) {
                const float sm = (use_scaleshift_ ? ss[c] : 1.0f) * inv_sqrtvar;
                const float sv = use_scaleshift_ ? ss[C_ + c] : 0;
                dst[c] = sm * (src[c] - *mean) + sv;
            }
        }
    }

private:
    int C_;
    bool use_scaleshift_;
    const float eps_;
    int simd_w_ = 8;

    struct ker_args {
        const float *src;
        float *dst;
        const float *ss;
        const float *mean;
        const float *inv_sqrtvar;
    };
    void (*ker_)(const ker_args *args);

    void load(Xbyak::Ymm &ymm_src, Xbyak::Reg64 reg_src, int nelems,
            size_t offt) {
        if (nelems == 1)
            movss(Xbyak::Xmm(ymm_src.getIdx()), dword[reg_src + offt]);
        else if (nelems == simd_w_)
            vmovups(ymm_src, yword[reg_src + offt]);
        else
            assert(!"unsupported nelems");
    }

    void store(Xbyak::Ymm &ymm_dst, Xbyak::Reg64 reg_dst, int nelems,
            size_t offt) {
        if (nelems == 1)
            movss(dword[reg_dst + offt], Xbyak::Xmm(ymm_dst.getIdx()));
        else if (nelems == simd_w_)
            vmovups(yword[reg_dst + offt], ymm_dst);
        else
            assert(!"unsupported nelems");
    }

    void generate() {
        using namespace Xbyak;

        preamble();
#define PARAM_OFF(x) offsetof(ker_args, x)
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        mov(reg_ss, ptr[reg_param + PARAM_OFF(ss)]);

        Xmm xmm_tmp = Xmm(ymm_tmp.getIdx());
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(mean)]);
        movss(xmm_tmp, dword[reg_tmp]);
        vbroadcastss(ymm_mean, xmm_tmp);

        mov(reg_tmp, ptr[reg_param + PARAM_OFF(inv_sqrtvar)]);
        movss(xmm_tmp, dword[reg_tmp]);
        vbroadcastss(ymm_inv_sqrtvar, xmm_tmp);
#undef PARAM_OFF

        const int C_vecs = C_ / simd_w_;

        auto op = [=](int nelems, size_t offt) {
            if (use_scaleshift_) {
                load(ymm_gamma, reg_ss, nelems, offt);
                load(ymm_beta, reg_ss, nelems, offt + C_ * sizeof(float));
            }
            load(ymm_data, reg_src, nelems, offt);
            vsubps(ymm_data, ymm_data, ymm_mean);
            vmulps(ymm_data, ymm_data, ymm_inv_sqrtvar);
            if (use_scaleshift_) vfmadd213ps(ymm_data, ymm_gamma, ymm_beta);
            store(ymm_data, reg_dst, nelems, offt);
        };

        for (int i = 0; i < C_vecs; i++)
            op(simd_w_, i * simd_w_ * sizeof(float));

        for (int i = utils::rnd_dn(C_, simd_w_); i < C_; i++)
            op(1, i * sizeof(float));

        postamble();

        ker_ = getCode<decltype(ker_)>();
    }

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_src = rdx;
    Xbyak::Reg64 reg_dst = rax;
    Xbyak::Reg64 reg_ss = r9;
    Xbyak::Reg64 reg_tmp = r8;

    Xbyak::Ymm ymm_inv_sqrtvar = Xbyak::Ymm(10);
    Xbyak::Ymm ymm_data = Xbyak::Ymm(11);
    Xbyak::Ymm ymm_gamma = Xbyak::Ymm(12);
    Xbyak::Ymm ymm_beta = Xbyak::Ymm(13);
    Xbyak::Ymm ymm_tmp = Xbyak::Ymm(14);
    Xbyak::Ymm ymm_mean = Xbyak::Ymm(15);
};

class diff_ss_kernel_t : jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_uni_layer_normalization_fwd_t::diff_dst_kernel);
    diff_ss_kernel_t(const layer_normalization_pd_t *pd)
        : C_(pd->norm_axis())
        , eps_(pd->desc()->layer_norm_epsilon)
        , ker_(nullptr) {
        if (mayiuse(avx2)) { generate(); }
    }
    ~diff_ss_kernel_t() {}
    void operator()(const float *src, const float *diff_dst, float *diff_gamma,
            float *diff_beta, const float *mean, const float *var) {
        if (ker_) {
            ker_args args;
            args.src = src;
            args.diff_dst = diff_dst;
            args.diff_gamma = diff_gamma;
            args.diff_beta = diff_beta;
            args.mean = mean;
            float inv_sqrtvar = 1. / sqrtf(*var + eps_);
            args.inv_sqrtvar = &inv_sqrtvar;
            ker_(&args);
        } else {
            float inv_sqrtvar = 1. / sqrtf(*var + eps_);
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; c++) {
                float dd = diff_dst[c];
                diff_gamma[c] += (src[c] - *mean) * dd * inv_sqrtvar;
                diff_beta[c] += dd;
            }
        }
    }

private:
    int C_;
    const float eps_;
    int simd_w_ = 8;

    struct ker_args {
        const float *src;
        const float *diff_dst;
        float *diff_gamma;
        float *diff_beta;
        const float *mean;
        const float *inv_sqrtvar;
    };
    void (*ker_)(const ker_args *args);

    void load(Xbyak::Ymm &ymm_src, Xbyak::Reg64 reg_src, int nelems,
            size_t offt) {
        if (nelems == 1)
            movss(Xbyak::Xmm(ymm_src.getIdx()), dword[reg_src + offt]);
        else if (nelems == simd_w_)
            vmovups(ymm_src, yword[reg_src + offt]);
        else
            assert(!"unsupported nelems");
    }

    void store(Xbyak::Ymm &ymm_dst, Xbyak::Reg64 reg_dst, int nelems,
            size_t offt) {
        if (nelems == 1)
            movss(dword[reg_dst + offt], Xbyak::Xmm(ymm_dst.getIdx()));
        else if (nelems == simd_w_)
            vmovups(yword[reg_dst + offt], ymm_dst);
        else
            assert(!"unsupported nelems");
    }

    void generate() {
        using namespace Xbyak;

        preamble();
#define PARAM_OFF(x) offsetof(ker_args, x)
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_diff_dst, ptr[reg_param + PARAM_OFF(diff_dst)]);
        mov(reg_diff_gamma, ptr[reg_param + PARAM_OFF(diff_gamma)]);
        mov(reg_diff_beta, ptr[reg_param + PARAM_OFF(diff_beta)]);

        mov(reg_tmp, ptr[reg_param + PARAM_OFF(mean)]);
        movss(xmm_tmp, dword[reg_tmp]);
        vbroadcastss(ymm_mean, xmm_tmp);

        mov(reg_tmp, ptr[reg_param + PARAM_OFF(inv_sqrtvar)]);
        movss(xmm_tmp, dword[reg_tmp]);
        vbroadcastss(ymm_inv_sqrtvar, xmm_tmp);
#undef PARAM_OFF

        const int C_vecs = C_ / simd_w_;
        auto op = [=](int nelems, size_t offt) {
            load(ymm_ddst, reg_diff_dst, nelems, offt);
            load(ymm_dbeta, reg_diff_beta, nelems, offt);
            load(ymm_dgamma, reg_diff_gamma, nelems, offt);
            load(ymm_src, reg_src, nelems, offt);
            vaddps(ymm_dbeta, ymm_dbeta, ymm_ddst);
            vsubps(ymm_src, ymm_src, ymm_mean);
            vmulps(ymm_src, ymm_src, ymm_inv_sqrtvar);
            vfmadd231ps(ymm_dgamma, ymm_src, ymm_ddst);
            store(ymm_dbeta, reg_diff_beta, nelems, offt);
            store(ymm_dgamma, reg_diff_gamma, nelems, offt);
        };

        for (int i = 0; i < C_vecs; i++)
            op(simd_w_, i * simd_w_ * sizeof(float));

        for (int i = utils::rnd_dn(C_, simd_w_); i < C_; i++)
            op(1, i * sizeof(float));

        postamble();

        ker_ = getCode<decltype(ker_)>();
    }

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_src = rdx;
    Xbyak::Reg64 reg_diff_dst = rax;
    Xbyak::Reg64 reg_tmp = r10;
    Xbyak::Reg64 reg_diff_gamma = r9;
    Xbyak::Reg64 reg_diff_beta = r8;

    Xbyak::Xmm xmm_tmp = Xbyak::Xmm(9);

    Xbyak::Ymm ymm_inv_sqrtvar = Xbyak::Ymm(10);
    Xbyak::Ymm ymm_ddst = Xbyak::Ymm(11);
    Xbyak::Ymm ymm_dgamma = Xbyak::Ymm(12);
    Xbyak::Ymm ymm_dbeta = Xbyak::Ymm(13);
    Xbyak::Ymm ymm_src = Xbyak::Ymm(14);
    Xbyak::Ymm ymm_mean = Xbyak::Ymm(15);
};

class diff_data_kernel_t : jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_uni_layer_normalization_fwd_t::diff_data_kernel);
    diff_data_kernel_t(const layer_normalization_pd_t *pd)
        : C_(pd->norm_axis())
        , eps_(pd->desc()->layer_norm_epsilon)
        , calculate_diff_stats_(!pd->use_global_stats())
        , use_scaleshift_(pd->use_scaleshift())
        , ker_(nullptr) {
        if (mayiuse(avx2)) { generate(); }
    }
    ~diff_data_kernel_t() {}
    void operator()(const float *src, const float *diff_dst, float *diff_src,
            const float *ss, const float *mean, const float *var) {
        if (ker_) {
            ker_args args;
            args.src = src;
            args.diff_dst = diff_dst;
            args.diff_src = diff_src;
            args.ss = ss;
            args.mean = mean;
            float inv_sqrtvar = 1.f / sqrtf(*var + eps_);
            args.inv_sqrtvar = &inv_sqrtvar;
            ker_(&args);
        } else {
            float inv_sqrtvar = 1.f / sqrtf(*var + eps_);
            float dd_gamma = 0, dd_gamma_x = 0;
            if (calculate_diff_stats_) {
                PRAGMA_OMP_SIMD(reduction(+ : dd_gamma, dd_gamma_x))
                for (dim_t c = 0; c < C_; c++) {
                    float gamma = use_scaleshift_ ? ss[c] : 1;
                    dd_gamma += diff_dst[c] * gamma;
                    dd_gamma_x += diff_dst[c] * gamma * (src[c] - *mean);
                }
                dd_gamma_x *= inv_sqrtvar;
            }
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; c++) {
                float gamma = use_scaleshift_ ? ss[c] : 1;
                float v_diff_src = diff_dst[c] * gamma;
                if (calculate_diff_stats_)
                    v_diff_src -= dd_gamma / C_
                            + (src[c] - *mean) * dd_gamma_x * inv_sqrtvar / C_;
                v_diff_src *= inv_sqrtvar;
                diff_src[c] = v_diff_src;
            }
        }
    }

private:
    int C_;
    const float eps_;
    bool calculate_diff_stats_;
    bool use_scaleshift_;
    int simd_w_ = 8;

    struct ker_args {
        const float *src;
        const float *diff_dst;
        float *diff_src;
        const float *ss;
        const float *mean;
        const float *inv_sqrtvar;
    };
    void (*ker_)(const ker_args *args);

    void load(Xbyak::Ymm &ymm_src, Xbyak::Reg64 reg_src, int nelems,
            size_t offt) {
        if (nelems == 1)
            movss(Xbyak::Xmm(ymm_src.getIdx()), dword[reg_src + offt]);
        else if (nelems == simd_w_)
            vmovups(ymm_src, yword[reg_src + offt]);
        else
            assert(!"unsupported nelems");
    }

    void store(Xbyak::Ymm &ymm_dst, Xbyak::Reg64 reg_dst, int nelems,
            size_t offt) {
        if (nelems == 1)
            movss(dword[reg_dst + offt], Xbyak::Xmm(ymm_dst.getIdx()));
        else if (nelems == simd_w_)
            vmovups(yword[reg_dst + offt], ymm_dst);
        else
            assert(!"unsupported nelems");
    }

    void generate() {
        using namespace Xbyak;

        preamble();
#define PARAM_OFF(x) offsetof(ker_args, x)
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_diff_dst, ptr[reg_param + PARAM_OFF(diff_dst)]);
        mov(reg_diff_src, ptr[reg_param + PARAM_OFF(diff_src)]);
        mov(reg_gamma, ptr[reg_param + PARAM_OFF(ss)]);

        if (calculate_diff_stats_) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(mean)]);
            movss(xmm_tmp, dword[reg_tmp]);
            vbroadcastss(ymm_mean, xmm_tmp);
        }

        mov(reg_tmp, ptr[reg_param + PARAM_OFF(inv_sqrtvar)]);
        movss(xmm_tmp, dword[reg_tmp]);
        vbroadcastss(ymm_inv_sqrtvar, xmm_tmp);
#undef PARAM_OFF

        mov(reg_tmp, float2int(C_));
        movq(xmm_tmp, reg_tmp);
        uni_vbroadcastss(ymm_C, xmm_tmp);

        const int C_vecs = C_ / simd_w_;

        auto compute_dd_gammas = [=](int nelems, size_t offt) {
            Ymm ymm_ddst = ymm_dsrc;
            load(ymm_ddst, reg_diff_dst, nelems, offt);
            if (use_scaleshift_) {
                load(ymm_gamma, reg_gamma, nelems, offt);
                vmulps(ymm_ddst, ymm_ddst, ymm_gamma);
            }
            load(ymm_src, reg_src, nelems, offt);
            vaddps(ymm_dd_gamma, ymm_dd_gamma, ymm_ddst);
            vsubps(ymm_src, ymm_src, ymm_mean);
            vfmadd231ps(ymm_dd_gamma_x, ymm_ddst, ymm_src);
        };

        auto reduce = [=](Ymm ymm_vec) {
            vextractf128(xmm_tmp, ymm_vec, 1);
            Xmm xmm_vec = Xmm(ymm_vec.getIdx());
            vaddps(xmm_vec, xmm_tmp, xmm_vec);
            vhaddps(xmm_vec, xmm_vec, xmm_vec);
            vhaddps(xmm_vec, xmm_vec, xmm_vec);
        };

        auto compute_diff_src = [=](int nelems, size_t offt) {
            load(ymm_dsrc, reg_diff_dst, nelems, offt);
            if (use_scaleshift_) {
                load(ymm_gamma, reg_gamma, nelems, offt);
                vmulps(ymm_dsrc, ymm_dsrc, ymm_gamma);
            }
            if (calculate_diff_stats_) {
                load(ymm_src, reg_src, nelems, offt);
                vsubps(ymm_src, ymm_src, ymm_mean);
                vmulps(ymm_src, ymm_src, ymm_inv_sqrtvar);
                vfmadd213ps(ymm_src, ymm_dd_gamma_x, ymm_dd_gamma);
                vdivps(ymm_src, ymm_src, ymm_C);
                vsubps(ymm_dsrc, ymm_dsrc, ymm_src);
            }
            vmulps(ymm_dsrc, ymm_dsrc, ymm_inv_sqrtvar);
            store(ymm_dsrc, reg_diff_src, nelems, offt);
        };

        if (calculate_diff_stats_) {
            vpxor(ymm_dd_gamma, ymm_dd_gamma, ymm_dd_gamma);
            vpxor(ymm_dd_gamma_x, ymm_dd_gamma_x, ymm_dd_gamma_x);

            for (int i = 0; i < C_vecs; i++)
                compute_dd_gammas(simd_w_, i * simd_w_ * sizeof(float));

            reduce(ymm_dd_gamma);
            reduce(ymm_dd_gamma_x);

            for (int i = utils::rnd_dn(C_, simd_w_); i < C_; i++)
                compute_dd_gammas(1, i * sizeof(float));

            vmulps(ymm_dd_gamma_x, ymm_dd_gamma_x, ymm_inv_sqrtvar);
            Xmm xmm_dd_gamma = Xmm(ymm_dd_gamma.getIdx());
            vbroadcastss(ymm_dd_gamma, xmm_dd_gamma);
            Xmm xmm_dd_gamma_x = Xmm(ymm_dd_gamma_x.getIdx());
            vbroadcastss(ymm_dd_gamma_x, xmm_dd_gamma_x);
        }

        for (int i = 0; i < C_vecs; i++)
            compute_diff_src(simd_w_, i * simd_w_ * sizeof(float));

        for (int i = utils::rnd_dn(C_, simd_w_); i < C_; i++)
            compute_diff_src(1, i * sizeof(float));

        postamble();

        ker_ = getCode<decltype(ker_)>();
    }

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_src = rdx;
    Xbyak::Reg64 reg_diff_src = rax;
    Xbyak::Reg64 reg_diff_dst = rbx;
    Xbyak::Reg64 reg_gamma = r11;
    Xbyak::Reg64 reg_tmp = r10;
    Xbyak::Reg64 reg_dd_gamma = r9;
    Xbyak::Reg64 reg_dd_gamma_x = r8;

    Xbyak::Xmm xmm_tmp = Xbyak::Xmm(7);

    Xbyak::Ymm ymm_C = Xbyak::Ymm(8);
    Xbyak::Ymm ymm_gamma = Xbyak::Ymm(9);
    Xbyak::Ymm ymm_inv_sqrtvar = Xbyak::Ymm(10);
    Xbyak::Ymm ymm_dsrc = Xbyak::Ymm(11);
    Xbyak::Ymm ymm_dd_gamma_x = Xbyak::Ymm(12);
    Xbyak::Ymm ymm_dd_gamma = Xbyak::Ymm(13);
    Xbyak::Ymm ymm_src = Xbyak::Ymm(14);
    Xbyak::Ymm ymm_mean = Xbyak::Ymm(15);
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

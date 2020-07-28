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

#include "cpu/x64/brgemm/brgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;

void execute_brgemm_kernel(const brgemm_desc_t *brg, int bs,
        const void **addr_A, const void **addr_B, void *ptr_C, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = 0;
    brgemm_p.N = bs;
    (*brg->kernel_)(&brgemm_p);
}

void execute_brgemm_kernel(const brgemm_desc_t *brg, int bs, const void *addr_A,
        const dim_t *offs_A, const void *addr_B, const dim_t *offs_B,
        void *ptr_C, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.offset_A = offs_A;
    brgemm_p.offset_B = offs_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = 0;
    brgemm_p.N = bs;
    (*brg->kernel_)(&brgemm_p);
}

void execute_brgemm_kernel(const brgemm_desc_t *brg, int bs, const void *addr_A,
        const void *addr_B, void *ptr_C, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = 0;
    brgemm_p.N = bs;
    (*brg->kernel_)(&brgemm_p);
}

void execute_brgemm_kernel_postops(const brgemm_desc_t *brg, int bs,
        const void **addr_A, const void **addr_B, void *ptr_C, void *ptr_D,
        const void *bias, const float *scales, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_D;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = bias;
    brgemm_p.ptr_scales = scales;
    brgemm_p.do_post_ops = 1;
    brgemm_p.N = bs;
    (*brg->kernel_)(&brgemm_p);
}

void execute_brgemm_kernel_postops(const brgemm_desc_t *brg, int bs,
        const void *addr_A, const dim_t *offs_A, const void *addr_B,
        const dim_t *offs_B, void *ptr_C, void *ptr_D, const void *bias,
        const float *scales, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.offset_A = offs_A;
    brgemm_p.offset_B = offs_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_D;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = bias;
    brgemm_p.ptr_scales = scales;
    brgemm_p.do_post_ops = 1;
    brgemm_p.N = bs;
    (*brg->kernel_)(&brgemm_p);
}

void execute_brgemm_kernel_postops(const brgemm_desc_t *brg, int bs,
        const void *addr_A, const void *addr_B, void *ptr_C, void *ptr_D,
        const void *bias, const float *scales, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_D;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = bias;
    brgemm_p.ptr_scales = scales;
    brgemm_p.do_post_ops = 1;
    brgemm_p.N = bs;
    (*brg->kernel_)(&brgemm_p);
}

status_t create_brgemm_descriptor(brgemm_desc_t **brg_,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, bool transA, bool transB,
        brgemm_layout_t layout, float alpha, float beta, dim_t LDA, dim_t LDB,
        dim_t LDC, dim_t M, dim_t N, dim_t K, const brgemm_strides_t *strides) {
    /*
    Matrices are in row-major layouts:
    m - number of rows of the matrix op(A) and number of rows of the matrix C
    n - number of columns of the matrix op(B) and number of columns of the matrix C
    k - number of columns of the matrix op(A) and number of rows of the matrix op(B)

    A: lda * m
    B: ldb * k
    C: ldc * m

    LDA - lda must be at least max(1, k);
    LDB - ldb must be at least max(1, n);
    LDC - ldc must be at least max(1, n);
    */
    if (transA || transB) return status::unimplemented;
    if (layout == brgemm_col_major) return status::unimplemented;

    *brg_ = new brgemm_desc_t();
    if (*brg_ == nullptr) return status::out_of_memory;
    auto brg = *brg_;

    auto &cfg = brg->cfg;

    cfg.dt_a = dt_a;
    cfg.dt_b = dt_b;

    cfg.is_int8 = (cfg.dt_a == data_type::u8 && cfg.dt_b == data_type::s8);
    cfg.is_bf16 = (cfg.dt_a == data_type::bf16 && cfg.dt_b == data_type::bf16);
    cfg.is_f32 = (cfg.dt_a == data_type::f32 && cfg.dt_b == data_type::f32);

    if (!cfg.is_int8 && !cfg.is_bf16 && !cfg.is_f32)
        return status::unimplemented;
    if (cfg.is_f32 && !mayiuse(avx512_core)) return status::unimplemented;
    if (cfg.is_bf16 && (!mayiuse(avx512_core_bf16) && !mayiuse(amx_bf16)))
        return status::unimplemented;
    if (cfg.is_int8 && (!mayiuse(avx512_core_vnni) && !mayiuse(amx_int8)))
        return status::unimplemented;

    cfg.dt_c = (cfg.is_int8) ? data_type::s32 : data_type::f32;
    cfg.dt_d = cfg.dt_c;
    cfg.dt_bias = cfg.dt_c;

    cfg.is_int8_amx = cfg.is_int8 && mayiuse(amx_int8);
    cfg.is_bf16_amx = cfg.is_bf16 && mayiuse(amx_bf16);

    cfg.LDA = (int)LDA;
    cfg.LDB = (int)LDB;
    cfg.LDC = (int)LDC;
    cfg.LDD = (int)LDC;

    cfg.M = (int)M;
    cfg.N = (int)N;
    cfg.K = (int)K;

    if (cfg.M <= 0 || cfg.N <= 0 || cfg.K <= 0 || cfg.LDA < nstl::max(1, cfg.K)
            || cfg.LDB < nstl::max(1, cfg.N) || cfg.LDC < nstl::max(1, cfg.N))
        return status::invalid_arguments;

    cfg.with_bias = false;
    cfg.with_eltwise = false;
    cfg.sum_scale = 0;
    cfg.with_scales = false;

    cfg.beta = beta;
    cfg.alpha = alpha;

    cfg.typesize_A = types::data_type_size(cfg.dt_a);
    cfg.typesize_B = types::data_type_size(cfg.dt_b);
    cfg.typesize_C = types::data_type_size(cfg.dt_c);
    cfg.typesize_D = types::data_type_size(cfg.dt_d);
    cfg.type = type;

    cfg.m_block2 = 0;
    cfg.mb2 = 0;
    cfg.mb2_tail = 0;

    cfg.n_step = cfg.k_step = 4 / cfg.typesize_A;

    if (!cfg.is_int8_amx && !cfg.is_bf16_amx) {
        cfg.n_block = 16;
        cfg.nb = cfg.N / cfg.n_block;
        cfg.nb_tail = cfg.N % cfg.n_block;

        cfg.n_block2 = 4; // (M < 9) ? 2 : 4 | TODO - fix this for INT8
        cfg.nb2 = cfg.nb / cfg.n_block2;
        cfg.nb2_tail = cfg.nb % cfg.n_block2;

        if (cfg.nb2 == 0) cfg.n_block2 = nstl::max(1, cfg.nb2_tail);
        cfg.embd_bcst = !cfg.is_int8 && !cfg.is_bf16
                && (cfg.nb2_tail <= 1 && cfg.nb2 == 0);

        int n_block = (cfg.nb2 != 0) ? cfg.n_block2 : cfg.nb2_tail;
        int max_regs
                = (cfg.embd_bcst ? 28
                                 : ((cfg.beta == 1.f || cfg.beta == 0.f) ? 30
                                                                         : 29))
                / (n_block + 1);

        int min_block = 6;

        cfg.m_block = 1;
        for (int m_block = max_regs; m_block >= min_block; m_block--) {
            if (cfg.M % m_block == 0) {
                cfg.m_block = m_block;
                break;
            }
        }
        if (cfg.m_block == 1) {
            cfg.m_block = nstl::min(max_regs, cfg.M);
            int m_tail = cfg.M % max_regs;
            for (int i = max_regs; i >= min_block; i--) {
                int i_tail = cfg.M % i;
                if (i_tail > m_tail || i_tail == 0) {
                    cfg.m_block = i;
                    m_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
        cfg.mb = cfg.M / cfg.m_block;
        cfg.mb_tail = cfg.M % cfg.m_block;

        cfg.k_block = 16 / cfg.typesize_A;
        cfg.kb = cfg.K / cfg.k_block;
        cfg.kb_tail = cfg.K % cfg.k_block;
    } else {
        // Blocking configuration for AMX
        const int max_width = 16, min_width = 1;
        for (int m_block = max_width; m_block >= min_width; m_block--) {
            if (cfg.M % m_block == 0) {
                cfg.m_block = m_block;
                break;
            }
        }
        if (cfg.m_block == 1) {
            cfg.m_block = nstl::min(max_width, cfg.M);
            cfg.mb_tail = cfg.M % max_width;
            for (int i = max_width; i >= min_width; i--) {
                int i_tail = cfg.M % i;
                if (i_tail > cfg.mb_tail || i_tail == 0) {
                    cfg.m_block = i;
                    cfg.mb_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
        cfg.mb = cfg.M / cfg.m_block;
        cfg.mb_tail = cfg.M % cfg.m_block;

        cfg.m_block2 = (cfg.mb > 2) ? 2 : 1;
        cfg.mb2 = cfg.mb / cfg.m_block2;
        cfg.mb2_tail = (cfg.m_block2 == 1) ? cfg.mb : cfg.mb % cfg.m_block2;

        cfg.n_block = 16;
        cfg.nb = cfg.N / cfg.n_block;
        cfg.nb_tail = cfg.N % cfg.n_block;

        cfg.n_block2 = (cfg.nb > 0 && cfg.nb % 2 == 0) ? 2 : 1;
        cfg.nb2 = cfg.nb / cfg.n_block2;
        cfg.nb2_tail = cfg.nb % cfg.n_block2;

        cfg.k_block = cfg.is_bf16_amx ? 32 : 64;
        cfg.kb = cfg.K / cfg.k_block;
        cfg.kb_tail = cfg.K % cfg.k_block;

        // Remove these guard in the future:
        if ((cfg.kb > 0 && cfg.kb_tail) || (cfg.nb > 0 && cfg.nb_tail))
            return status::unimplemented;
        if (cfg.kb_tail % ((cfg.is_bf16_amx) ? 2 : 4))
            return status::unimplemented;
    }

    if (strides != nullptr) {
        cfg.stride_a = strides->stride_a;
        cfg.stride_b = strides->stride_b;
    } else {
        cfg.stride_a = cfg.stride_b = 0;
    }

    // sum may be implemented by beta, so we define cfg.with_sum value
    // in convolution init_conf
    cfg.with_sum = false;

    return status::success;
}

status_t brgemm_descriptor_add_postops(brgemm_desc_t *brg,
        const primitive_attr_t *attr, impl::data_type_t dt_d, int LDD,
        impl::data_type_t dt_bias) {
    auto &cfg = brg->cfg;

    // TODO: Add AMX support
    if (cfg.is_int8_amx || cfg.is_bf16_amx) return status::unimplemented;

    cfg.attr = attr;

    cfg.with_bias = (dt_bias == data_type::undef) ? false : true;
    cfg.dt_bias = dt_bias;
    cfg.typesize_bias = (dt_bias == data_type::undef)
            ? 0
            : types::data_type_size(cfg.dt_bias);

    cfg.LDD = LDD;

    if ((cfg.dt_a == data_type::u8 && cfg.dt_b == data_type::s8)
            && (!one_of(dt_d, data_type::u8, data_type::s8, data_type::s32,
                    data_type::f32))
            && (!one_of(dt_bias, data_type::undef, data_type::u8, data_type::s8,
                    data_type::s32, data_type::f32)))
        return status::unimplemented;
    if ((cfg.dt_a == data_type::bf16 && cfg.dt_b == data_type::bf16)
            && (!one_of(dt_d, data_type::bf16, data_type::f32))
            && (!one_of(dt_bias, data_type::undef, data_type::bf16,
                    data_type::f32)))
        return status::unimplemented;
    if ((cfg.dt_a == data_type::f32 && cfg.dt_b == data_type::f32)
            && (!one_of(dt_d, data_type::f32))
            && (!one_of(dt_bias, data_type::undef, data_type::f32)))
        return status::unimplemented;

    cfg.dt_d = dt_d;
    cfg.typesize_D = types::data_type_size(cfg.dt_d);

    const auto &p = cfg.attr->post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    cfg.sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 0;

    const int eltwise_ind = p.find(primitive_kind::eltwise);
    cfg.with_eltwise = eltwise_ind != -1;
    if (cfg.with_eltwise) cfg.eltwise = p.entry_[eltwise_ind].eltwise;

    if (cfg.is_int8) {
        const auto &oscales = cfg.attr->output_scales_;
        cfg.is_oc_scale = oscales.mask_ == 1 << 1;
        cfg.with_scales = true;
    }

    return status::success;
}

status_t create_brgemm_kernel(brgemm_desc_t *brg) {
    if (brg->kernel_ == nullptr) {
        brg->kernel_ = new jit_brgemm_kernel_t(brg->cfg);
        if (brg->kernel_ == nullptr)
            return status::out_of_memory;
        else
            return brg->kernel_->create_kernel();
    } else {
        return status::not_required;
    }
}

void destroy_brgemm_descriptor(brgemm_desc_t *brg) {
    delete brg;
}

status_t brgemm_init_tiles(const brgemm_desc_t *brg, char palette[64]) {
    constexpr int max_palette_size_in_bytes = 64;
    auto &cfg = brg->cfg;

    if (!cfg.is_int8_amx || !cfg.is_bf16_amx) return status::unimplemented;

    int n_block = (!cfg.nb && cfg.nb_tail) ? cfg.nb_tail : cfg.n_block;
    int k_block = (!cfg.kb && cfg.kb_tail) ? cfg.kb_tail : cfg.k_block;

    auto cfg_tiles = [=](palette_config_t *buff, int Ac, int n_block) {
        char *_tc = (char *)buff;
        for (int i = 0; i < max_palette_size_in_bytes; i++)
            _tc[i] = 0;

        int Ar = cfg.m_block;
        int Br = (cfg.typesize_C != 0) ? Ac / cfg.typesize_C : 0;
        int Cr = cfg.m_block;

        int k_step = 4 / cfg.typesize_A;

        int Bc = n_block * cfg.typesize_B * k_step;
        int Cc = n_block * cfg.typesize_C;

        for (int m = 0; m < brgemm_amx::max_m_block2; m++)
            tc_configure_tile(buff, brgemm_amx::get_A_tensor(m), Ar, Ac);
        for (int n = 0; n < brgemm_amx::max_n_block2; n++)
            tc_configure_tile(buff, brgemm_amx::get_B_tensor(n), Br, Bc);

        for (int m = 0; m < brgemm_amx::max_m_block2; m++)
            for (int n = 0; n < brgemm_amx::max_n_block2; n++)
                tc_configure_tile(buff, brgemm_amx::get_C_tensor(m, n), Cr, Cc);

        buff->palette_id = amx::get_max_palette();
    };

    int Ac = cfg.typesize_A * k_block;
    cfg_tiles((palette_config_t *)(palette), Ac, n_block);

    return status::success;
    // TODO: add tail processing support
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

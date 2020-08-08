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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/cpu_barrier.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;

void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
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
    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const dim_t *offs_A, const void *addr_B,
        const dim_t *offs_B, void *ptr_C, void *scratch) {
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
    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const void *addr_B, void *ptr_C, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = 0;
    brgemm_p.N = bs;
    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, int bs,
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
    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, int bs,
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
    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, int bs,
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
    (*brg_kernel)(&brgemm_p);
}

status_t brgemm_desc_init(brgemm_t *brg, brgemm_batch_kind_t type,
        impl::data_type_t dt_a, impl::data_type_t dt_b, bool transA,
        bool transB, brgemm_layout_t layout, float alpha, float beta, dim_t LDA,
        dim_t LDB, dim_t LDC, dim_t M, dim_t N, dim_t K,
        const brgemm_strides_t *strides) {
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
    if (brg == nullptr) return status::invalid_arguments;

    if (transA || transB) return status::unimplemented;
    if (layout == brgemm_col_major) return status::unimplemented;

    brg->dt_a = dt_a;
    brg->dt_b = dt_b;

    brg->is_int8 = (brg->dt_a == data_type::u8 && brg->dt_b == data_type::s8);
    brg->is_bf16
            = (brg->dt_a == data_type::bf16 && brg->dt_b == data_type::bf16);
    brg->is_f32 = (brg->dt_a == data_type::f32 && brg->dt_b == data_type::f32);

    if (!brg->is_int8 && !brg->is_bf16 && !brg->is_f32)
        return status::unimplemented;
    if (brg->is_f32 && !mayiuse(avx512_core)) return status::unimplemented;
    if (brg->is_bf16 && (!mayiuse(avx512_core_bf16) && !mayiuse(amx_bf16)))
        return status::unimplemented;
    if (brg->is_int8 && (!mayiuse(avx512_core_vnni) && !mayiuse(amx_int8)))
        return status::unimplemented;

    brg->dt_c = (brg->is_int8) ? data_type::s32 : data_type::f32;
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->is_int8_amx = brg->is_int8 && mayiuse(amx_int8);
    brg->is_bf16_amx = brg->is_bf16 && mayiuse(amx_bf16);

    brg->LDA = (int)LDA;
    brg->LDB = (int)LDB;
    brg->LDC = (int)LDC;
    brg->LDD = (int)LDC;

    brg->M = (int)M;
    brg->N = (int)N;
    brg->K = (int)K;

    if (brg->M <= 0 || brg->N <= 0 || brg->K <= 0
            || brg->LDA < nstl::max(1, brg->K)
            || brg->LDB < nstl::max(1, brg->N)
            || brg->LDC < nstl::max(1, brg->N))
        return status::invalid_arguments;

    brg->with_bias = false;
    brg->with_eltwise = false;
    brg->sum_scale = 0;
    brg->with_scales = false;

    brg->beta = beta;
    brg->alpha = alpha;

    brg->typesize_A = types::data_type_size(brg->dt_a);
    brg->typesize_B = types::data_type_size(brg->dt_b);
    brg->typesize_C = types::data_type_size(brg->dt_c);
    brg->typesize_D = types::data_type_size(brg->dt_d);
    brg->type = type;

    brg->m_block2 = 0;
    brg->mb2 = 0;
    brg->mb2_tail = 0;

    brg->n_step = brg->k_step = 4 / brg->typesize_A;

    if (!brg->is_int8_amx && !brg->is_bf16_amx) {
        brg->n_block = 16;
        brg->nb = brg->N / brg->n_block;
        brg->nb_tail = brg->N % brg->n_block;

        brg->n_block2 = 4; // (M < 9) ? 2 : 4 | TODO - fix this for INT8
        brg->nb2 = brg->nb / brg->n_block2;
        brg->nb2_tail = brg->nb % brg->n_block2;

        if (brg->nb2 == 0) brg->n_block2 = nstl::max(1, brg->nb2_tail);
        brg->embd_bcst = !brg->is_int8 && !brg->is_bf16
                && (brg->nb2_tail <= 1 && brg->nb2 == 0);

        int n_block = (brg->nb2 != 0) ? brg->n_block2 : brg->nb2_tail;
        int max_regs = (brg->embd_bcst ? 28
                                       : ((brg->beta == 1.f || brg->beta == 0.f)
                                                       ? 30
                                                       : 29))
                / (n_block + 1);

        int min_block = 6;

        brg->m_block = 1;
        for (int m_block = max_regs; m_block >= min_block; m_block--) {
            if (brg->M % m_block == 0) {
                brg->m_block = m_block;
                break;
            }
        }
        if (brg->m_block == 1) {
            brg->m_block = nstl::min(max_regs, brg->M);
            int m_tail = brg->M % max_regs;
            for (int i = max_regs; i >= min_block; i--) {
                int i_tail = brg->M % i;
                if (i_tail > m_tail || i_tail == 0) {
                    brg->m_block = i;
                    m_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
        brg->mb = brg->M / brg->m_block;
        brg->mb_tail = brg->M % brg->m_block;

        brg->k_block = 16 / brg->typesize_A;
        brg->kb = brg->K / brg->k_block;
        brg->kb_tail = brg->K % brg->k_block;
    } else {
        // Blocking configuration for AMX
        const int max_width = 16, min_width = 1;
        for (int m_block = max_width; m_block >= min_width; m_block--) {
            if (brg->M % m_block == 0) {
                brg->m_block = m_block;
                break;
            }
        }
        if (brg->m_block == 1) {
            brg->m_block = nstl::min(max_width, brg->M);
            brg->mb_tail = brg->M % max_width;
            for (int i = max_width; i >= min_width; i--) {
                int i_tail = brg->M % i;
                if (i_tail > brg->mb_tail || i_tail == 0) {
                    brg->m_block = i;
                    brg->mb_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
        brg->mb = brg->M / brg->m_block;
        brg->mb_tail = brg->M % brg->m_block;

        brg->m_block2 = (brg->mb > 2) ? 2 : 1;
        brg->mb2 = brg->mb / brg->m_block2;
        brg->mb2_tail
                = (brg->m_block2 == 1) ? brg->mb : brg->mb % brg->m_block2;

        brg->n_block = 16;
        brg->nb = brg->N / brg->n_block;
        brg->nb_tail = brg->N % brg->n_block;

        brg->n_block2 = (brg->nb > 0 && brg->nb % 2 == 0) ? 2 : 1;
        brg->nb2 = brg->nb / brg->n_block2;
        brg->nb2_tail = brg->nb % brg->n_block2;

        brg->k_block = brg->is_bf16_amx ? 32 : 64;
        brg->kb = brg->K / brg->k_block;
        brg->kb_tail = brg->K % brg->k_block;

        // Remove these guard in the future:
        if ((brg->kb > 0 && brg->kb_tail) || (brg->nb > 0 && brg->nb_tail))
            return status::unimplemented;
        if (brg->kb_tail % ((brg->is_bf16_amx) ? 2 : 4))
            return status::unimplemented;
    }

    if (strides != nullptr) {
        brg->stride_a = strides->stride_a;
        brg->stride_b = strides->stride_b;
    } else {
        brg->stride_a = brg->stride_b = 0;
    }

    // sum may be implemented by beta, so we define brg->with_sum value
    // in convolution init_conf
    brg->with_sum = false;

    return status::success;
}

status_t brgemm_desc_add_postops(brgemm_t *brg, const primitive_attr_t *attr,
        impl::data_type_t dt_d, int LDD, impl::data_type_t dt_bias) {
    if (brg == nullptr) return status::invalid_arguments;

    // TODO: Add AMX support
    if (brg->is_int8_amx || brg->is_bf16_amx) return status::unimplemented;

    brg->attr = attr;

    brg->with_bias = (dt_bias == data_type::undef) ? false : true;
    brg->dt_bias = dt_bias;
    brg->typesize_bias = (dt_bias == data_type::undef)
            ? 0
            : types::data_type_size(brg->dt_bias);

    brg->LDD = LDD;

    if ((brg->dt_a == data_type::u8 && brg->dt_b == data_type::s8)
            && (!one_of(dt_d, data_type::u8, data_type::s8, data_type::s32,
                    data_type::f32))
            && (!one_of(dt_bias, data_type::undef, data_type::u8, data_type::s8,
                    data_type::s32, data_type::f32)))
        return status::unimplemented;
    if ((brg->dt_a == data_type::bf16 && brg->dt_b == data_type::bf16)
            && (!one_of(dt_d, data_type::bf16, data_type::f32))
            && (!one_of(dt_bias, data_type::undef, data_type::bf16,
                    data_type::f32)))
        return status::unimplemented;
    if ((brg->dt_a == data_type::f32 && brg->dt_b == data_type::f32)
            && (!one_of(dt_d, data_type::f32))
            && (!one_of(dt_bias, data_type::undef, data_type::f32)))
        return status::unimplemented;

    brg->dt_d = dt_d;
    brg->typesize_D = types::data_type_size(brg->dt_d);

    const auto &p = brg->attr->post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    brg->sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 0;

    const int eltwise_ind = p.find(primitive_kind::eltwise);
    brg->with_eltwise = eltwise_ind != -1;
    if (brg->with_eltwise) brg->eltwise = p.entry_[eltwise_ind].eltwise;

    if (brg->is_int8) {
        const auto &oscales = brg->attr->output_scales_;
        brg->is_oc_scale = oscales.mask_ == 1 << 1;
        brg->with_scales = true;
    }

    return status::success;
}

status_t brgemm_kernel_create(
        brgemm_kernel_t **brg_kernel, const brgemm_t &brg) {
    CHECK(safe_ptr_assign<brgemm_kernel_t>(
            *brg_kernel, new brgemm_kernel_t(brg)));
    return (*brg_kernel)->create_kernel();
}

void brgemm_kernel_destroy(brgemm_kernel_t *brg_kernel) {
    delete brg_kernel;
}

status_t brgemm_init_tiles(const brgemm_t &brg, char palette[64]) {
    constexpr int max_palette_size_in_bytes = 64;

    if (!(brg.is_int8_amx || brg.is_bf16_amx)) return status::unimplemented;

    int n_block = (!brg.nb && brg.nb_tail) ? brg.nb_tail : brg.n_block;
    int k_block = (!brg.kb && brg.kb_tail) ? brg.kb_tail : brg.k_block;

    auto cfg_tiles = [=](palette_config_t *buff, int Ac, int n_block) {
        char *_tc = (char *)buff;
        for (int i = 0; i < max_palette_size_in_bytes; i++)
            _tc[i] = 0;

        int Ar = brg.m_block;
        int Br = (brg.typesize_C != 0) ? Ac / brg.typesize_C : 0;
        int Cr = brg.m_block;

        int k_step = 4 / brg.typesize_A;

        int Bc = n_block * brg.typesize_B * k_step;
        int Cc = n_block * brg.typesize_C;

        for (int m = 0; m < brgemm_amx::max_m_block2; m++)
            tc_configure_tile(buff, brgemm_amx::get_A_tensor(m), Ar, Ac);
        for (int n = 0; n < brgemm_amx::max_n_block2; n++)
            tc_configure_tile(buff, brgemm_amx::get_B_tensor(n), Br, Bc);

        for (int m = 0; m < brgemm_amx::max_m_block2; m++)
            for (int n = 0; n < brgemm_amx::max_n_block2; n++)
                tc_configure_tile(buff, brgemm_amx::get_C_tensor(m, n), Cr, Cc);

        buff->palette_id = amx::get_max_palette();
    };

    int Ac = brg.typesize_A * k_block;
    cfg_tiles((palette_config_t *)(palette), Ac, n_block);

    return status::success;
    // TODO: add tail processing support
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

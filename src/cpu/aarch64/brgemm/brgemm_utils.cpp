/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
* Copyright 2023 FUJITSU LIMITED
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

#include "cpu/aarch64/brgemm/brgemm_utils.hpp"
#include "cpu/aarch64/brgemm/jit_brdgmm_kernel.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;

enum {
    decomposition_2x2 = 101,
    decomposition_3x1_3,
    decomposition_3x1_2,
    undefined,
};

impl::data_type_t get_accum_datatype(brgemm_t *brg) {
    // this assert should check if 'init_kernel_datatype()' was previously
    // called.
    assert(brg->is_int8 || brg->is_bf16 || brg->is_f32 || brg->is_f16);
    return brg->is_int8 ? data_type::s32 : data_type::f32;
}

void init_kernel_datatype(
        brgemm_t *brg, impl::data_type_t dt_a, impl::data_type_t dt_b) {
    assert(dt_a != data_type::undef && dt_b != data_type::undef);
    brg->is_int8 = utils::one_of(dt_a, data_type::u8, data_type::s8)
            && utils::one_of(dt_b, data_type::u8, data_type::s8);
    brg->is_bf16 = (dt_a == data_type::bf16) && (dt_b == data_type::bf16);
    brg->is_f32 = (dt_a == data_type::f32) && (dt_b == data_type::f32);
    brg->is_f16 = utils::one_of(data_type::f16, dt_a, dt_b);
    assert(brg->is_int8 || brg->is_bf16 || brg->is_f32 || brg->is_f16);
}

void init_common_conf(brgemm_t *brg, brgemm_batch_kind_t type, float alpha,
        float beta, const brgemm_strides_t *strides) {
    brg->beta = beta;
    brg->alpha = alpha;
    brg->type = type;
    brg->with_bias = false;
    brg->with_eltwise = false;
    brg->with_sum = false;
    brg->with_weights_scale_adjust = false;
    brg->sum_scale = 0;
    brg->sum_zp = 0;
    brg->with_scales = false;

    if (strides != nullptr) {
        brg->stride_a = strides->stride_a;
        brg->stride_b = strides->stride_b;
    } else {
        brg->stride_a = brg->stride_b = 0;
    }
}

namespace brgemm_utils {

bool can_dispatch_uker(const brgemm_t *brg) {
    return false;
}
void maybe_try_bf32(brgemm_t *brg) {
    //
}

void set_isa_impl(brgemm_t *brg) {
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) &&
                // maybe IMPLICATION(brg->isa_user != isa_undef,
                //  is_superset(brg->isa_user, isa)), but the API is not clear.
                one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_bf32) {
        assert(!"unsupported case");
    } else if (brg->is_f32) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(sve_512), sve_512,
                is_isa_ok(sve_256), sve_256);
    } else if (brg->is_bf16) {
        assert(!"unsupported case");
    } else if (brg->is_f16) {
        assert(!"unsupported case");
    } else if (brg->is_int8) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(sve_512), sve_512,
                is_isa_ok(sve_256), sve_256);
    }
}

void set_brg_vmm(brgemm_t *brg) {

    brg->is_zmm = mayiuse(sve_512) && is_superset(brg->isa_impl, sve_512);
    brg->is_ymm = !brg->is_zmm && mayiuse(sve_256)
            && is_superset(brg->isa_impl, sve_256);
}

int calculate_ldb_params(brgemm_t *brg, const int try_ld_block2) {
    brg->ld_block2 = try_ld_block2;
    brg->ldb2 = brg->ldb / brg->ld_block2;
    brg->ldb2_tail = brg->ldb % brg->ld_block2;

    if (brg->ldb2 == 0) brg->ld_block2 = nstl::max(1, brg->ldb2_tail);
    brg->embd_bcst = brg->is_f32
            && (brg->ldb2_tail <= 1 && brg->ldb2 == 0)
            /*only sve512 or more can bcast*/
            && is_superset(brg->isa_impl, sve_512);

    const int adj_ld_block2
            = (brg->ldb2 != 0) ? brg->ld_block2 : brg->ldb2_tail;
    return nstl::max(1, adj_ld_block2);
}

int calculate_max_bcast_block(brgemm_t *brg, const int adj_ld_block2) {

    constexpr int max_bcst_regs = 1;
    const bool req_compensation = brg->req_s8s8_compensation
            || brg->zp_type_a != brgemm_broadcast_t::none;
    const bool req_zp_a_comp_pads
            = (brg->req_cal_comp_pads || brg->brgattr.max_top_vpad > 0
                      || brg->brgattr.max_bottom_vpad > 0)
            && brg->zp_type_a != brgemm_broadcast_t::none;
    const int beta_regs = !one_of(brg->beta, 1.f, 0.f);

    const int max_isa_regs = 32;
    // note: the 'adj_ld_block2' already removes the necessary registers
    // for 'embd_bcst'
    auto max_reg_count = max_isa_regs - max_bcst_regs - beta_regs
            - req_compensation - req_zp_a_comp_pads;
    if (req_zp_a_comp_pads)
        max_reg_count
                = nstl::min(max_reg_count, max_isa_regs - max_bcst_regs - 5);

    int max_bcast_block = max_reg_count - adj_ld_block2;

    if (brg->is_bf16_emu) {
        // in theory, vmm bf16_emu register indices overlap with other vmm
        // registers related to 'max_bcast_block'
        assert(is_superset(brg->isa_impl, sve_512));
        constexpr int bf16_emu_reg_count = 28;
        max_bcast_block = nstl::min(max_bcast_block, bf16_emu_reg_count);
    }

    if (brg->is_int8 && !brg->has_int8_vnni) max_bcast_block -= 2;

    max_bcast_block /= adj_ld_block2;

    return max_bcast_block;
}

inline size_t data_type_vnni_granularity(data_type_t data_type) {
    using namespace data_type;
    switch (data_type) {
        case f32:
        case s32: return size_t(1);
        case f16:
        case bf16: return size_t(2);
        case s8:
        case u8: return size_t(4);
        case data_type::undef:
        default: assert(!"unknown data_type");
    }
    return size_t(0); /* should not be reachable */
}
status_t brgemm_blocking(brgemm_t *brg) {

    set_isa_impl(brg);
    if (brg->isa_impl == isa_undef) return status::unimplemented;
    set_brg_vmm(brg);
    if (!(brg->is_zmm || brg->is_ymm)) return status::unimplemented;

    const int simd_w = is_superset(brg->isa_impl, sve_512) ? 16 : 8;
    brg->ld_block = simd_w;
    brg->ldb = brg->load_dim / brg->ld_block;
    brg->ldb_tail = brg->load_dim % brg->ld_block;

    int adj_ld_block2 = calculate_ldb_params(brg, 4);
    int max_bcast_block = calculate_max_bcast_block(brg, adj_ld_block2);

    // reduce 'ld_block2' to allow a larger 'bd_block'
    const int max_vpad = nstl::max(
            brg->brgattr.max_top_vpad, brg->brgattr.max_bottom_vpad);
    if (is_superset(brg->isa_impl, sve_256) && max_bcast_block < max_vpad) {
        adj_ld_block2 = calculate_ldb_params(brg, 2);
        max_bcast_block = calculate_max_bcast_block(brg, adj_ld_block2);
    }

    const int min_block = 1;
    float best_bd_block_eff = 0.f;
    brg->bd_block = 1;
    for (int bd_block = max_bcast_block; bd_block >= min_block; bd_block--) {
        const auto bd_block_disb = static_cast<float>(brg->bcast_dim)
                / rnd_up(brg->bcast_dim, bd_block);
        const auto brgemm_microkernel_eff
                = (static_cast<float>(adj_ld_block2) * bd_block)
                / (((adj_ld_block2) + bd_block) * max_bcast_block);
        const auto bd_block_eff = bd_block_disb * brgemm_microkernel_eff;

        float block_foot_print = static_cast<float>(brg->typesize_A)
                * (bd_block * brg->reduce_dim);
        if (block_foot_print <= static_cast<float>(
                    platform::get_per_core_cache_size(1))
                && (bd_block_eff > best_bd_block_eff)) {
            brg->bd_block = bd_block;
            best_bd_block_eff = bd_block_eff;
        }
    }
    brg->bdb = brg->bcast_dim / brg->bd_block;
    brg->bdb_tail = brg->bcast_dim % brg->bd_block;

    const int rd_unroll = 4;
    const int vnni_granularity = data_type_vnni_granularity(brg->dt_a);
    brg->rd_block = rd_unroll * vnni_granularity;
    brg->rdb = brg->reduce_dim / brg->rd_block;
    brg->rdb_tail = brg->reduce_dim % brg->rd_block;

    brg->is_M_tail = false;
    return status::success;
}

status_t brdgmm_blocking(brgemm_t *brg) {

    if (brg->isa_impl == isa_undef) return status::unimplemented;

    const int requires_permute_dst_vmm = brg->isa_impl == sve_512
            && jit_brdgmm_kernel_base_t::is_fast_vnni_int8(*brg);
    const int max_vregs = 32;
    const int aux_vregs
            = nstl::max(brg->is_bf16_emu * 4, 2) + requires_permute_dst_vmm;
    const int max_acc_vmms = max_vregs - aux_vregs;
    const int simd_w = isa_max_vlen(brg->isa_impl) / brg->typesize_C;

    auto &M = brg->bcast_dim;
    auto &N = brg->load_dim;

    // In current implementation of dgmm, there is no reduce dim.
    auto &m_block1 = brg->bd_block;
    auto &nb_m_block1 = brg->bdb;
    auto &m_block1_tail = brg->bdb_tail;
    auto &m_block2 = brg->bd_block2;
    auto &nb_m_block2 = brg->bdb2;
    auto &m_block2_tail = brg->bdb2_tail;

    auto &n_block1 = brg->ld_block;
    auto &nb_n_block1 = brg->ldb;
    auto &n_block1_tail = brg->ldb_tail;
    auto &n_block2 = brg->ld_block2;
    auto &nb_n_block2 = brg->ldb2;
    auto &n_block2_tail = brg->ldb2_tail;

    // begin blocking
    const int n_block1_num_steps = 1;
    n_block1 = n_block1_num_steps * simd_w;
    nb_n_block1 = div_up(N, n_block1);
    n_block1_tail = N % n_block1;

    const int max_n_block2_vmms = 4;
    const int max_n_block2 = max_n_block2_vmms / n_block1_num_steps;
    n_block2 = nstl::min(max_n_block2, nb_n_block1);
    nb_n_block2 = div_up(nb_n_block1, n_block2);
    n_block2_tail = nb_n_block1 % n_block2;

    m_block1 = 1;
    nb_m_block1 = M / m_block1;
    m_block1_tail = M % m_block1;
    m_block2 = nstl::min(
            nb_m_block1, max_acc_vmms / (n_block2 * n_block1_num_steps));
    nb_m_block2 = div_up(nb_m_block1, m_block2);
    m_block2_tail = nb_m_block1 % m_block2;

    return status::success;
}

void init_brgemm_conf(brgemm_t *brg, cpu_isa_t isa, brgemm_batch_kind_t type,
        impl::data_type_t dt_a, impl::data_type_t dt_b, brgemm_layout_t layout,
        float alpha, float beta, dim_t LDA, dim_t LDB, dim_t LDC, dim_t M,
        dim_t N, dim_t K, const brgemm_strides_t *strides, bool is_bf32) {

    init_common_conf(brg, type, alpha, beta, strides);

    brg->layout = layout;

    brg->dt_a = brg->is_row_major() ? dt_a : dt_b;
    brg->dt_b = brg->is_row_major() ? dt_b : dt_a;
    init_kernel_datatype(brg, brg->dt_a, brg->dt_b);

    brg->dt_c = get_accum_datatype(brg);
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->typesize_A = types::data_type_size(brg->dt_a);
    brg->typesize_B = types::data_type_size(brg->dt_b);
    brg->typesize_C = types::data_type_size(brg->dt_c);
    brg->typesize_D = types::data_type_size(brg->dt_d);

    brg->isa_user = isa;
    set_isa_impl(brg);
    brg->is_bf32 = false;

    brg->has_int8_vnni = true;

    set_brg_vmm(brg); // TODO: Investigate if it is really needed here.
    brg->req_s8s8_compensation = brg->is_int8 && brg->dt_a == data_type::s8;

    brg->LDA = (brg->is_row_major()) ? static_cast<int>(LDA)
                                     : static_cast<int>(LDB);
    brg->LDB = (brg->is_row_major()) ? static_cast<int>(LDB)
                                     : static_cast<int>(LDA);
    brg->LDC = static_cast<int>(LDC);
    brg->LDD = static_cast<int>(LDC);

    brg->bcast_dim
            = (brg->is_row_major()) ? static_cast<int>(M) : static_cast<int>(N);
    brg->load_dim
            = (brg->is_row_major()) ? static_cast<int>(N) : static_cast<int>(M);
    brg->reduce_dim = static_cast<int>(K);

    brg->bd_block2 = 0;
    brg->bdb2 = 0;
    brg->bdb2_tail = 0;

    const bool is_b_in_vnni_format = false;
    brg->ld_step
            = is_b_in_vnni_format ? data_type_vnni_granularity(brg->dt_b) : 1;

    const bool has_no_vnni_compute_instruction = false;
    brg->rd_step = has_no_vnni_compute_instruction
            ? 1
            : data_type_vnni_granularity(brg->dt_b);
}

void init_brdgmm_conf(brgemm_t *brg, cpu_isa_t isa, brgemm_batch_kind_t type,
        impl::data_type_t dt_a, impl::data_type_t dt_b, brgemm_layout_t layout,
        float alpha, float beta, dim_t LDA, dim_t LDC, dim_t M, dim_t N,
        const brgemm_strides_t *strides) {

    init_common_conf(brg, type, alpha, beta, strides);

    brg->layout = layout;

    brg->dt_a = dt_a;
    brg->dt_b = dt_b;
    init_kernel_datatype(brg, brg->dt_a, brg->dt_b);

    brg->dt_c = get_accum_datatype(brg);
    brg->dt_d = brg->dt_c;
    brg->dt_bias = brg->dt_c;

    brg->typesize_A = types::data_type_size(brg->dt_a);
    brg->typesize_B = types::data_type_size(brg->dt_b);
    brg->typesize_C = types::data_type_size(brg->dt_c);
    brg->typesize_D = types::data_type_size(brg->dt_d);

    brg->isa_user = isa;
    auto is_isa_ok = [&](cpu_isa_t isa) {
        return mayiuse(isa) && one_of(brg->isa_user, isa_undef, isa);
    };

    if (brg->is_f32) {
        brg->isa_impl = utils::map(true, isa_undef, is_isa_ok(sve_512), sve_512,
                is_isa_ok(sve_256), sve_256);
    }

    brg->is_dgmm = true;

    brg->LDA = static_cast<int>(LDA);
    brg->LDC = static_cast<int>(LDC);
    brg->LDD = static_cast<int>(LDC);

    brg->bcast_dim = M;
    brg->load_dim = N;
}

} // namespace brgemm_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include <unordered_set>

#include "common/dnnl_thread.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

// TODO add a method to print brgemm conf info
#define VCONDCHECK_BG(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, brgemm_matmul, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

#define VCHECK_BG(f, msg, ...) \
    VCHECK(primitive, create, dispatch, brgemm_matmul, f, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::cpu::matmul;

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace data_type;
using namespace format_tag;

int get_default_n_block(format_tag_t matrix_b_tag) {
    // Note: consider using weights mem_descriptor 'inner_blks' to
    // return B's inner block for non-default cases.
    switch (matrix_b_tag) {
        case aCB16b64c:
        case aCB16b64c2b:
        case aCB16b64c4b:
        case BA16a64b4a:
        case BA16a64b2a:
        case BA16a64b: return 64;
        case aCB16b48c:
        case aCB16b48c2b:
        case aCB16b48c4b:
        case BA16a48b:
        case BA16a48b2a:
        case BA16a48b4a: return 48;
        case aCB16b32c:
        case aCB16b32c2b:
        case aCB16b32c4b:
        case BA16a32b:
        case BA16a32b2a:
        case BA16a32b4a: return 32;
        case aCB16b16c:
        case aCB16b16c2b:
        case aCB16b16c4b:
        case BA16a16b:
        case BA16a16b2a:
        case BA16a16b4a: return 16;
        default: return 64;
    }
}

// TODO: add support of post-ops with multiple binary and eltwise execution
bool post_ops_ok(brgemm_matmul_conf_t &bgmmc, const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d,
        bool limit_bcast_strategies_set = false) {
    using namespace injector;

    const auto &post_ops = attr.post_ops_;
    const auto ndims = dst_d.ndims();

    bool is_binary_po_per_oc_sp_bcast {};
    bool is_binary_po_channel_bcast {};
    bool is_binary_po_per_mb_w_bcast {};
    bool is_binary_po_per_w_bcast {};
    std::tie(is_binary_po_per_oc_sp_bcast, is_binary_po_channel_bcast,
            is_binary_po_per_mb_w_bcast, is_binary_po_per_w_bcast)
            = binary_injector_utils::bcast_strategies_present_tup(
                    post_ops.entry_, dst_d,
                    broadcasting_strategy_t::per_oc_spatial,
                    broadcasting_strategy_t::per_mb_spatial,
                    broadcasting_strategy_t::per_mb_w,
                    broadcasting_strategy_t::per_w);
    const bool supported_binary_bcast
            = IMPLICATION(is_binary_po_per_oc_sp_bcast, ndims < 4)
            && IMPLICATION(
                    is_binary_po_channel_bcast, utils::one_of(ndims, 3, 4))
            && IMPLICATION(
                    is_binary_po_per_mb_w_bcast, utils::one_of(ndims, 3, 4))
            && IMPLICATION(
                    is_binary_po_per_w_bcast, utils::one_of(ndims, 3, 4));
    const bcast_set_t default_bcast_set = {broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::per_mb_spatial,
            broadcasting_strategy_t::per_mb_w, broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::no_broadcast};
    const bcast_set_t limited_bcast_set = {broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::no_broadcast};
    const bcast_set_t bcast_set = limit_bcast_strategies_set
            ? limited_bcast_set
            : default_bcast_set;
    return supported_binary_bcast
            && injector::post_ops_ok(post_ops_ok_args_t(get_max_cpu_isa(),
                    {sum, eltwise, binary}, post_ops, &dst_d,
                    false /*sum_at_pos_0_only*/,
                    false /*sum_requires_scale_one*/,
                    false /*sum_requires_zp_zero*/,
                    true /*sum_requires_same_params*/, bcast_set));
}

status_t check_isa_with_datatype(
        const cpu_isa_t isa, const brgemm_matmul_conf_utils_t &bm_conf_utils) {
    const bool ok = IMPLICATION(bm_conf_utils.is_f32(),
                            isa == avx512_core || bm_conf_utils.is_bf32())
            && IMPLICATION(bm_conf_utils.is_int8(),
                    one_of(isa, avx512_core_amx, avx512_core_vnni, avx512_core,
                            avx2_vnni_2, avx2_vnni))
            && IMPLICATION(bm_conf_utils.is_bf16(),
                    one_of(isa, avx512_core_amx, avx512_core_bf16, avx2_vnni_2))
            && IMPLICATION(bm_conf_utils.is_f16(),
                    one_of(isa, avx512_core_amx_fp16, avx512_core_fp16,
                            avx2_vnni_2))
            && IMPLICATION(bm_conf_utils.is_int8_with_bf16_dst(),
                    is_superset(isa, avx512_core) || isa == avx2_vnni_2);
    return ok ? status::success : status::unimplemented;
}

brgemm_matmul_conf_utils_t::brgemm_matmul_conf_utils_t(
        brgemm_matmul_conf_t &bgmmc, const cpu_isa_t isa,
        const primitive_attr_t &attr, bool A_any_layout, bool B_any_layout,
        bool C_any_layout, bool bias_any_layout)
    : bgmmc(bgmmc)
    , f32_dt(utils::everyone_is(f32, bgmmc.src_dt, bgmmc.wei_dt, bgmmc.dst_dt))
    , bf16_dt(utils::everyone_is(bf16, bgmmc.src_dt, bgmmc.wei_dt)
              && one_of(bgmmc.dst_dt, bf16, f32))
    , f16_dt(utils::everyone_is(f16, bgmmc.src_dt, bgmmc.wei_dt)
              && one_of(bgmmc.dst_dt, f16, f32))
    , int8_dt(utils::one_of(bgmmc.src_dt, u8, s8) && bgmmc.wei_dt == s8
              && one_of(bgmmc.dst_dt, u8, s8, s32, f32, bf16))
    , bf32_dt(f32_dt && attr.fpmath_mode_ == fpmath_mode::bf16
              && isa == avx512_core_amx)
    , A_any_layout(A_any_layout)
    , B_any_layout(B_any_layout)
    , C_any_layout(C_any_layout)
    , bias_any_layout(bias_any_layout)
    , plain_tensor_layout_tag(utils::pick(bgmmc.ndims - 2, ab, abc, abcd, abcde,
              abcdef, abcdefg, abcdefgh, abcdefghi, abcdefghij, abcdefghijk,
              abcdefghijkl))
    , transposed_tensor_layout_tag(utils::pick(bgmmc.ndims - 2, ba, acb, abdc,
              abced, abcdfe, abcdegf, abcdefhg, abcdefgih, abcdefghji,
              abcdefghikj, abcdefghijlk))
    , blocked_64n_B_layout_tag(pick_blocked_B_layout(64))
    , blocked_48n_B_layout_tag(pick_blocked_B_layout(48))
    , blocked_32n_B_layout_tag(pick_blocked_B_layout(32))
    , blocked_16n_B_layout_tag(pick_blocked_B_layout(16))
    , blocked_B_layouts_allowed(!utils::one_of(format_tag::undef,
              blocked_64n_B_layout_tag, blocked_48n_B_layout_tag,
              blocked_32n_B_layout_tag, blocked_16n_B_layout_tag))
    , n_blk_fixed((!B_any_layout) && blocked_B_layouts_allowed)
    , isa_(isa) {
    assert(int8_dt || bf16_dt || f16_dt || f32_dt || bf32_dt);
}

status_t brgemm_matmul_conf_utils_t::set_or_check_B_tag(
        memory_desc_t &B_md, bool init_n_tag) const {

    if (B_any_layout) {
        const int default_n_block = init_n_tag
                ? get_default_n_block(format_tag::undef)
                : bgmmc.N_blk;
        bgmmc.wei_tag = blocked_B_layouts_allowed
                ? this->pick_blocked_B_layout(default_n_block)
                : plain_tensor_layout_tag;
        if (format_tag::undef == bgmmc.wei_tag) return status::unimplemented;

        VCHECK_BG(memory_desc_init_by_tag(B_md, bgmmc.wei_tag),
                VERBOSE_UNSUPPORTED_TAG);
        const int dmax = nstl::min(bgmmc.ndims, 3);
        const memory_desc_wrapper B_d(&B_md);
        for (int d = 0; d < dmax; d++) {
            int dim = bgmmc.ndims - 1 - d;
            bgmmc.B_strides[d]
                    = bgmmc.b_dt_sz * B_d.blocking_desc().strides[dim];
        }
    } else {
        bgmmc.wei_tag = blocked_B_layouts_allowed
                ? memory_desc_matches_one_of_tag(B_md, plain_tensor_layout_tag,
                        transposed_tensor_layout_tag, blocked_64n_B_layout_tag,
                        blocked_48n_B_layout_tag, blocked_32n_B_layout_tag,
                        blocked_16n_B_layout_tag)
                : memory_desc_matches_one_of_tag(B_md, plain_tensor_layout_tag,
                        transposed_tensor_layout_tag, acbd, adbc);

        if (format_tag::undef == bgmmc.wei_tag) return status::unimplemented;
    }

    return status::success;
}

status_t brgemm_matmul_conf_utils_t::update_and_check_B_tag(
        memory_desc_t &B_md, int n_blk_size) const {

    if (n_blk_fixed && n_blk_size != bgmmc.wei_n_blk)
        return status::unimplemented;

    if (!(B_any_layout && blocked_B_layouts_allowed)) return status::success;

    return set_or_check_B_tag(B_md, false);
}

status_t brgemm_matmul_conf_utils_t::set_or_check_tags(memory_desc_t &A_md,
        memory_desc_t &C_md, memory_desc_t &bias_md) const {
    if (A_any_layout) {
        const format_tag_t desired_A_tag = plain_tensor_layout_tag;
        VCHECK_BG(memory_desc_init_by_tag(A_md, desired_A_tag),
                VERBOSE_UNSUPPORTED_TAG);
        bgmmc.src_tag = desired_A_tag;
    } else {
        const bool xf16_avx2_vnni_2 = (this->is_bf16() || this->is_f16())
                && bgmmc.isa == avx2_vnni_2;
        const bool can_treat_transposed_A_as_plain = bgmmc.M == 1;
        bgmmc.src_tag = (this->is_bf16() || this->is_f32() || this->is_bf32()
                                || this->is_f16())
                        && !xf16_avx2_vnni_2
                ? memory_desc_matches_one_of_tag(A_md, plain_tensor_layout_tag,
                        transposed_tensor_layout_tag, acbd, adbc)
                // Enable support of int8 problems with formally transposed A
                // layout which can be treated as plain.
                // TODO: remove this extra code path after transposed A is
                // supported for int8
                : (this->is_int8() && can_treat_transposed_A_as_plain)
                ? memory_desc_matches_one_of_tag(A_md, plain_tensor_layout_tag,
                        transposed_tensor_layout_tag, acbd)
                : memory_desc_matches_one_of_tag(
                        A_md, plain_tensor_layout_tag, acbd);
    }

    if (C_any_layout) {
        const format_tag_t desired_C_tag = plain_tensor_layout_tag;
        VCHECK_BG(memory_desc_init_by_tag(C_md, desired_C_tag),
                VERBOSE_UNSUPPORTED_TAG);
        bgmmc.dst_tag = desired_C_tag;
    } else {
        const memory_desc_wrapper C_mdw(C_md);
        // If one of dims is `1` then `ba` is identical to `ab`.
        format_tag_t allowed_transposed_tensor_layout_tag
                = C_mdw.ndims() == 2 && C_mdw.count_non_unit_dims(1)
                ? ba
                : plain_tensor_layout_tag;
        bgmmc.dst_tag
                = memory_desc_matches_one_of_tag(C_md, plain_tensor_layout_tag,
                        allowed_transposed_tensor_layout_tag, acbd);
    }

    if (one_of(format_tag::undef, bgmmc.src_tag, bgmmc.dst_tag))
        return status::unimplemented;

    if (bgmmc.with_bias && bias_any_layout)
        VCHECK_BG(memory_desc_init_by_tag(bias_md, plain_tensor_layout_tag),
                VERBOSE_UNSUPPORTED_TAG);

    return status::success;
}

status_t brgemm_matmul_conf_utils_t::set_B_flags(memory_desc_t &B_md) const {

    memory_desc_t want_B_md = B_md;
    // Set bits for all dimensions except k dimension
    const int compensation_mask
            = ((1 << bgmmc.ndims) - 1 - (1 << (bgmmc.ndims - 2)));
    if (bgmmc.s8s8_compensation_required && bgmmc.blocked_B) {
        want_B_md.extra.flags |= memory_extra_flags::compensation_conv_s8s8;
        want_B_md.extra.compensation_mask = compensation_mask;
    }
    if (bgmmc.src_zp_type != brgemm_broadcast_t::none && bgmmc.blocked_B) {
        want_B_md.extra.flags
                |= memory_extra_flags::compensation_conv_asymmetric_src;
        want_B_md.extra.asymm_compensation_mask = compensation_mask;
    }

    if (B_any_layout) {
        B_md = want_B_md;
        return status::success;
    }

    return B_md == want_B_md ? status::success : status::unimplemented;
}

format_tag_t brgemm_matmul_conf_utils_t::pick_blocked_B_layout(
        int n_blk) const {

    if (bgmmc.ndims > 3) return format_tag::undef;
    if (this->is_int8()) switch (n_blk) {
            case 64: return bgmmc.ndims == 3 ? aCB16b64c4b : BA16a64b4a;
            case 48: return bgmmc.ndims == 3 ? aCB16b48c4b : BA16a48b4a;
            case 32: return bgmmc.ndims == 3 ? aCB16b32c4b : BA16a32b4a;
            case 16: return bgmmc.ndims == 3 ? aCB16b16c4b : BA16a16b4a;
            default: return format_tag::undef;
        }

    if (this->is_bf16() || (this->is_f16() && bgmmc.isa != avx512_core_fp16))
        switch (n_blk) {
            case 64: return bgmmc.ndims == 3 ? aCB16b64c2b : BA16a64b2a;
            case 48: return bgmmc.ndims == 3 ? aCB16b48c2b : BA16a48b2a;
            case 32: return bgmmc.ndims == 3 ? aCB16b32c2b : BA16a32b2a;
            case 16: return bgmmc.ndims == 3 ? aCB16b16c2b : BA16a16b2a;
            default: return format_tag::undef;
        }
    // Note: bf32 assumes f32 blocking
    if (this->is_f32() || this->is_bf32() || this->is_f16()) switch (n_blk) {
            case 64: return bgmmc.ndims == 3 ? aCB16b64c : BA16a64b;
            case 48: return bgmmc.ndims == 3 ? aCB16b48c : BA16a48b;
            case 32: return bgmmc.ndims == 3 ? aCB16b32c : BA16a32b;
            case 16: return bgmmc.ndims == 3 ? aCB16b16c : BA16a16b;
            default: return format_tag::undef;
        }
    return format_tag::undef;
}

brgemm_broadcast_t get_zp_type(const primitive_attr_t &attr, int arg) {
    return attr.zero_points_.has_default_values(arg)
            ? brgemm_broadcast_t::none
            : brgemm_broadcast_t::per_tensor;
}

struct matmul_amx_blocking_params_t : public brgemm_matmul_conf_t {
    matmul_amx_blocking_params_t()
        : nthr_k_(0)
        , nthr_mnb_(0)
        , nthr_(0)
        , n_blk_(0)
        , n_chunk_size_(0)
        , n_chunk_elems_(0)
        , m_blk_(0)
        , m_chunk_size_(0)
        , m_chunk_elems_(0)
        , k_blk_(0)
        , k_chunk_size_(0)
        , k_chunk_elems_(0)
        , current_lda_(0)
        , need_buf_c_(false)
        , blocking_chunk_mem_size_(0)
        , efficiency_score_(0.0f) {}

    matmul_amx_blocking_params_t(const brgemm_matmul_conf_t &bgmmc)
        : brgemm_matmul_conf_t(bgmmc)
        , nthr_k_(nstl::max(nthr_k, 1))
        , nthr_mnb_(nthr / nthr_k_)
        , nthr_(nthr_mnb_ * nthr_k_)
        , n_blk_(N_blk)
        , n_chunk_size_(N_chunk_size)
        , n_chunk_elems_(n_blk_ * n_chunk_size_)
        , m_blk_(M_blk)
        , m_chunk_size_(M_chunk_size)
        , m_chunk_elems_(m_blk_ * m_chunk_size_)
        , k_blk_(K_blk)
        , k_chunk_size_(brgemm_batch_size)
        , k_chunk_elems_(k_blk_ * k_chunk_size_)
        , current_lda_(LDA)
        , need_buf_c_(use_buffer_c)
        , blocking_chunk_mem_size_(0)
        , efficiency_score_(0.0f) {}

    void set_blocking_parameters(int nthr_k, int n_blk, int n_chunk_size,
            int m_blk, int m_chunk_size);
    void update_configuration(brgemm_matmul_conf_t &bgmmc) const;
    float get_blocking_scores() const { return efficiency_score_; }

    static size_t L2_threshold();

private:
    // num threads for parallelism wrt k dimension
    int nthr_k_;
    // num threads for parallelism wrt m, n and batch dimensions
    int nthr_mnb_;
    int nthr_;
    dim_t n_blk_, n_chunk_size_, n_chunk_elems_;
    dim_t m_blk_, m_chunk_size_, m_chunk_elems_;
    dim_t k_blk_, k_chunk_size_, k_chunk_elems_;

    dim_t current_lda_;
    bool need_buf_c_;
    size_t blocking_chunk_mem_size_;
    float efficiency_score_;

    void update_k_blocking_dependent_params();
    dim_t get_actual_lda();
    bool is_buffer_c_required();
    size_t calculate_chunk_memory_size();
    float get_thread_balance_scores();
    float get_copied_data_reusage_scores();
    float get_L2_utilization_scores() const;
    float calculate_blocking_scores();
};

struct matmul_avx512_blocking_params_t {
    struct matmul_params_t {

        matmul_params_t(int m, int n, int k, int od)
            : M(m), N(n), K(k), batch(od) {}

        const int M;
        const int N;
        const int K;
        const int batch;
    };

    matmul_avx512_blocking_params_t(const matmul_params_t &m, const int nthr)
        : mp(m)
        , m_chunks(1)
        , m_blk(1)
        , m_tail(0)
        , n_chunks(1)
        , n_blk(1)
        , n_tail(0)
        , batch_size(1)
        , k_blk(1)
        , k_tail(0)
        , nthr_k(1)
        , nthr(nthr) {}

    matmul_avx512_blocking_params_t &operator=(
            const matmul_avx512_blocking_params_t &brgemm_params) {
        m_chunks = brgemm_params.m_chunks;
        m_blk = brgemm_params.m_blk;
        m_tail = brgemm_params.m_tail;
        n_chunks = brgemm_params.n_chunks;
        n_blk = brgemm_params.n_blk;
        n_tail = brgemm_params.n_tail;
        batch_size = brgemm_params.batch_size;
        k_blk = brgemm_params.k_blk;
        k_tail = brgemm_params.k_tail;
        nthr_k = brgemm_params.nthr_k;
        return *this;
    }

    const matmul_params_t &mp;
    int m_chunks, m_blk, m_tail;
    int n_chunks, n_blk, n_tail;
    int batch_size, k_blk, k_tail;
    int nthr_k;
    const int nthr;

    void update_params(int m_chunks_, int m_blk_, int n_chunks_, int n_blk_,
            int batch_size_, int k_blk_, int nthr_k_) {
        m_chunks = m_chunks_;
        m_blk = m_blk_;
        m_tail = mp.M % m_blk;
        n_chunks = n_chunks_;
        n_blk = n_blk_;
        n_tail = mp.N % n_blk;
        batch_size = batch_size_;
        k_blk = k_blk_;
        k_tail = mp.K % k_blk;
        nthr_k = nthr_k_;
    }

    float calculate_spatial_disbalance(size_t work, size_t thread_block) const {
        size_t mod = work % thread_block;
        size_t scalar = work < thread_block
                ? thread_block - mod
                : nstl::min(thread_block - mod, mod);
        return static_cast<float>(scalar) / thread_block;
    }

    float get_imbalance() const {
        const size_t cur_nthr = nthr / nthr_k;

        size_t parallel_work = get_parallel_work();
        const float parallel_work_disb
                = calculate_spatial_disbalance(parallel_work, cur_nthr);

        int m_work = (m_blk * div_up(mp.M, m_blk)) % mp.M;
        const float m_blk_disbalance = static_cast<float>(m_work) / mp.M;

        int num_n_blk = div_up(mp.N, n_blk);
        int par_n_chunks = div_up(num_n_blk, n_chunks);
        const float n_chunk_disbalance
                = (static_cast<float>(par_n_chunks) * n_chunks - num_n_blk)
                / num_n_blk;

        const float disbalance_nthr_k
                = calculate_spatial_disbalance(mp.K, nthr_k * k_blk);

        const float thread_allocation_disb
                = (cur_nthr * nthr_k) != static_cast<size_t>(nthr)
                ? (static_cast<float>(nthr) - cur_nthr * nthr_k) / nthr
                : 0;

        const float score
                = (parallel_work_disb + m_blk_disbalance + n_chunk_disbalance
                          + thread_allocation_disb + disbalance_nthr_k)
                / 5;

        return score;
    }

    size_t get_parallel_work() const {
        int m_elems = div_up(mp.M, m_blk * m_chunks);
        int n_elems = div_up(mp.N, n_blk * n_chunks);
        return static_cast<size_t>(m_elems) * n_elems * mp.batch;
    }

    inline dim_t get_actual_lda(bool use_buffer_a, dim_t a_dt_sz) const {
        if (!use_buffer_a) return mp.K;

        constexpr int bytes_in_cacheline = 64;
        const int elems_in_cacheline = bytes_in_cacheline / a_dt_sz;
        dim_t lda = rnd_up(k_blk, elems_in_cacheline);
        const bool is_big_pow_2 = lda >= 512 && math::is_pow2(lda);
        if (is_big_pow_2) lda += elems_in_cacheline;
        return lda;
    }

    inline bool is_buffer_c_required(
            dim_t acc_dt, dim_t dst_dt, bool with_sum) const {
        const size_t k_chunk_elems = k_blk * batch_size;
        if (nthr_k > 1 && static_cast<size_t>(mp.K) > k_chunk_elems)
            return true;

        return ((acc_dt != dst_dt || with_sum)
                && (static_cast<size_t>(mp.K) > k_chunk_elems
                        || mp.K % k_blk > 0));
    }

    void update_configuration(brgemm_matmul_conf_t &bgmmc) const {
        bgmmc.M_blk = m_blk;
        bgmmc.M_chunk_size = m_chunks;
        bgmmc.N_blk = n_blk;
        bgmmc.N_chunk_size = n_chunks;

        bgmmc.K_blk = rnd_up(k_blk, bgmmc.required_k_granularity);
        bgmmc.brgemm_batch_size = batch_size;

        bgmmc.nthr_k = nthr_k;

        bgmmc.use_buffer_c = is_buffer_c_required(
                bgmmc.acc_dt, bgmmc.dst_dt, bgmmc.with_sum);
        bgmmc.LDA = (bgmmc.src_tag == acbd && !bgmmc.use_buffer_a
                        ? bgmmc.A_strides[1] / bgmmc.a_dt_sz
                        : get_actual_lda(bgmmc.use_buffer_a, bgmmc.tr_a_dt_sz));
    }
};

size_t matmul_amx_blocking_params_t::L2_threshold() {
    return 3 * platform::get_per_core_cache_size(2) / 4;
}

void compute_blocking_heuristic_amx(const brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils,
        matmul_amx_blocking_params_t &best_blocking) {

    matmul_amx_blocking_params_t current_blocking(bgmmc);

    const int min_k_per_thread = 1024;
    const int max_k_parallel_work
            = div_up(static_cast<int>(bgmmc.K), min_k_per_thread);
    const bool is_amx_xf16 = bgmmc.is_amx
            && (bm_conf_utils.is_bf16() || bm_conf_utils.is_f16()
                    || bm_conf_utils.is_bf32());
    const bool is_amx_int8 = bgmmc.is_amx && bm_conf_utils.is_int8();

    const bool runtime_dims
            = bgmmc.is_runtime_M || bgmmc.is_runtime_N || bgmmc.is_runtime_K;
    const int max_nthr_k = !runtime_dims && is_amx_xf16 && bgmmc.batch == 1
            ? nstl::min(saturate(1, 7, bgmmc.nthr / 8), max_k_parallel_work)
            : 1;
    int iter = 0;
    const int runtime_M_chunk = bgmmc.lda_big_pow2() ? 2 : 4;
    for (int nthr_k = 1; nthr_k <= max_nthr_k; nthr_k++) {
        int num_M_blk = bgmmc.is_runtime_M ? 1 : div_up(bgmmc.M, bgmmc.M_blk);
        int num_N_blk = div_up(bgmmc.N, bgmmc.N_blk);
        int k_parallel_work = nstl::min(max_k_parallel_work, nthr_k);
        int num_parallel_work
                = bgmmc.batch * num_M_blk * num_N_blk * k_parallel_work;
        const bool a_lot_of_parallel_work_lvl2
                = num_parallel_work > 16 * bgmmc.nthr;
        const bool low_parallelism
                = static_cast<float>(num_parallel_work) < 1.5f * bgmmc.nthr;
        const bool maybe_low_blocking
                = is_amx_int8 && bm_conf_utils.maybe_low_brg_blocking();
        const int min_M_blk = !bgmmc.is_runtime_M
                        && (maybe_low_blocking || low_parallelism)
                        && bgmmc.M_blk > 32
                ? div_up(bgmmc.M_blk, 2)
                : bgmmc.M_blk;
        const int min_N_blk = low_parallelism && is_amx_xf16
                        && !bm_conf_utils.check_n_blk_fixed()
                        && bgmmc.N_blk > 32 && !runtime_dims
                ? 32
                : bgmmc.N_blk;
        const int desired_M_chunk = bgmmc.is_runtime_M
                ? runtime_M_chunk
                : nstl::min(4, num_M_blk);
        const int desired_N_chunk
                = nstl::min(a_lot_of_parallel_work_lvl2 ? 6 : 4, num_N_blk);

        std::unordered_set<int> mblk_candidates;
        for (int m_blk = bgmmc.M_blk; m_blk >= min_M_blk;
                m_blk = m_blk > 1 ? div_up(m_blk, 2) : m_blk - 1) {
            if (IMPLICATION(maybe_low_blocking, m_blk != bgmmc.M_blk))
                mblk_candidates.insert(m_blk);
        }

        if (!bgmmc.is_runtime_M && bgmmc.M > 16) {
            // Add multiple of 16 M block sizes for consideration
            const int mul16_m_blk_max
                    = nstl::min(rnd_dn(static_cast<int>(bgmmc.M), 16), 64);
            const int mul16_m_blk_min = rnd_up(min_M_blk, 16);
            for (int m_blk = mul16_m_blk_max; m_blk >= mul16_m_blk_min;
                    m_blk -= 16) {
                mblk_candidates.insert(m_blk);
            }
        }

        for_(int n_blk = bgmmc.N_blk; n_blk >= min_N_blk; n_blk -= 16)
        for_(int m_blk : mblk_candidates)
        for_(int n_ch_sz = desired_N_chunk; n_ch_sz >= 1; n_ch_sz--)
        for (int m_ch_sz = desired_M_chunk; m_ch_sz >= 1; m_ch_sz--, iter++) {
            current_blocking.set_blocking_parameters(
                    nthr_k, n_blk, n_ch_sz, m_blk, m_ch_sz);
            if (current_blocking.get_blocking_scores()
                    > best_blocking.get_blocking_scores())
                best_blocking = current_blocking;
        }
    }
}

float compute_blocking_heuristic_avx512(brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils,
        const matmul_avx512_blocking_params_t::matmul_params_t &matmul,
        matmul_avx512_blocking_params_t &best_blocking) {

    const int nthr = bgmmc.nthr;

    const int max_m_blk = nstl::min(256, matmul.M);
    int min_m_blk = nstl::min(32, matmul.M);

    int n_blk = bgmmc.N_blk;
    const int n_chunks = div_up(matmul.N, n_blk);
    const int max_n_chunks = bgmmc.use_buffer_a ? 16 : 1;
    const int n_chunks_start = nstl::min(max_n_chunks, div_up(matmul.N, n_blk));

    // Note: do not extend K_blk for 'bwd_w' cases
    const bool use_extended_k_blk = matmul.K > 1024
            && (!bm_conf_utils.check_is_transposed(bgmmc.src_tag));
    int default_k_blk = use_extended_k_blk ? 1024 : 512;
    int k_blk = nstl::min(matmul.K, default_k_blk);
    int start_nthr_k = 1;

    // for cases with low parallel work, reduce 'min_m_blk' to
    // increase potential parallelization balance.
    const size_t max_parallel = matmul.batch * n_chunks;
    const bool low_parallel_work = static_cast<size_t>(nthr) > max_parallel;
    if (low_parallel_work) {

        min_m_blk = nstl::min(matmul.M, 16);

        // 2nd level tuning for low parallel work cases:
        bool bwd_w_low_spatial_work
                = bm_conf_utils.check_is_transposed(bgmmc.src_tag)
                && matmul.M <= 512;
        bool low_spatial_work = matmul.M <= 40;
        if (low_spatial_work || bwd_w_low_spatial_work) {

            // Reduce n_blk size to increase parallel space
            // note: over reduction of n_blk size on 2d shapes when n_chunks == 1
            // showed significant performance degradation
            if (!bm_conf_utils.check_n_blk_fixed()
                    && IMPLICATION(n_chunks == 1, bgmmc.batch_ndims > 0))
                n_blk = nstl::min(matmul.N, 32);

            // force to plain B (wei) in small spatial size for FWD:
            // note: this showed significant performance gain in WnD shapes
            bool is_FWD = !(bm_conf_utils.check_is_transposed(bgmmc.wei_tag)
                    || bm_conf_utils.check_is_transposed(bgmmc.src_tag));
            if (bgmmc.use_buffer_b && is_FWD) {
                bgmmc.use_buffer_b = bm_conf_utils.use_buffer_b(false);
            }
        }

        // Parallelize across K for shapes with big 'K' dimension
        bool bwd_w_par_k_blk = bgmmc.batch == 1
                && bm_conf_utils.check_is_transposed(bgmmc.src_tag)
                && IMPLICATION(bm_conf_utils.is_bf16(), math::is_pow2(matmul.K))
                && matmul.K >= 2048;
        if (bwd_w_par_k_blk) {
            start_nthr_k = nstl::min(nthr, 4);
            assert(k_blk == nstl::min(matmul.K, 512));
        }
    }

    float best_imbalance = 1.f; // reduce
    for_(int nthr_k = start_nthr_k; nthr_k >= 1; --nthr_k)
    for_(int n_chunk_size = n_chunks_start; n_chunk_size >= 1; --n_chunk_size)
    for (int m_blk = max_m_blk; m_blk >= min_m_blk; --m_blk) {

        matmul_avx512_blocking_params_t cur_params(matmul, nthr);
        cur_params.update_params(
                1, m_blk, n_chunk_size, n_blk, 1, k_blk, nthr_k);

        float cur_imbalance = cur_params.get_imbalance();
        if (cur_imbalance < best_imbalance) {
            best_imbalance = cur_imbalance;
            best_blocking = cur_params;
        }
    }
    return best_imbalance;
}

float compute_blocking_heuristic_avx2(brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils,
        const matmul_avx512_blocking_params_t::matmul_params_t &matmul,
        matmul_avx512_blocking_params_t &best_blocking) {

    const int nthr = bgmmc.nthr;

    const int max_m_blk = nstl::min(/*64*/ 256, matmul.M);
    int min_m_blk = nstl::min(32, matmul.M); // max_m_blk

    int n_blk = bgmmc.N_blk;
    const int n_chunks = div_up(matmul.N, n_blk);
    const int max_n_chunks = bgmmc.use_buffer_a ? 16 : 1;
    const int n_chunks_start = nstl::min(max_n_chunks, n_chunks);

    int default_k_blk = 1024;
    int k_blk = nstl::min(matmul.K, default_k_blk);
    int start_nthr_k = 1;

    // for cases with low parallel work, reduce 'min_m_blk' to
    // increase potential parallelization balance.
    const size_t max_parallel = matmul.batch * n_chunks;
    const bool low_parallel_work = static_cast<size_t>(nthr) > max_parallel;
    if (low_parallel_work) {

        min_m_blk = nstl::min(matmul.M, 16);

        bool low_spatial_work = matmul.M <= 40;
        if (low_spatial_work) {

            // Reduce n_blk size to increase parallel space
            // note: over reduction of n_blk size on 2d shapes when n_chunks == 1
            // showed significant performance degradation
            if (!bm_conf_utils.check_n_blk_fixed()
                    && IMPLICATION(n_chunks == 1, bgmmc.batch_ndims > 0))
                n_blk = nstl::min(matmul.N, 32);
        }
    }

    float best_imbalance = 1.f; // reduce
    for_(int nthr_k = start_nthr_k; nthr_k >= 1; --nthr_k)
    for_(int n_chunk_size = n_chunks_start; n_chunk_size >= 1; --n_chunk_size)
    for (int m_blk = max_m_blk; m_blk >= min_m_blk; --m_blk) {

        matmul_avx512_blocking_params_t cur_params(matmul, nthr);
        cur_params.update_params(
                1, m_blk, n_chunk_size, n_blk, 1, k_blk, nthr_k);

        float cur_imbalance = cur_params.get_imbalance();
        if (cur_imbalance < best_imbalance) {
            best_imbalance = cur_imbalance;
            best_blocking = cur_params;
        }
    }
    return best_imbalance;
}

status_t compute_blocking_heuristic(brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils) {

    bgmmc.N_blk = nstl::min(static_cast<dim_t>(bgmmc.wei_n_blk), bgmmc.N);

    bgmmc.M_chunk_size = bgmmc.N_chunk_size = 1;

    if (bgmmc.is_amx) {

        // Configure matrix sizes
        if (bgmmc.is_runtime_M) {
            bgmmc.M_blk = 64; // use fixed block size for runtime M case
        } else {
            auto get_block_candidate = [&]() -> dim_t {
                // for AMX prefer block sizes which utilize at least 13 tile
                // rows
                const dim_t tile_rows_min = 13;
                const dim_t tile_rows_max = 16;
                const dim_t scale_rows_min = 2;
                const dim_t scale_rows_max = 4;

                for_(dim_t r = tile_rows_max; r >= tile_rows_min; r--)
                for (dim_t s = scale_rows_max; s >= scale_rows_min; s--) {
                    const dim_t m_blk = s * r;
                    if (bgmmc.M % m_blk == 0) return m_blk;
                }

                const dim_t max_M = scale_rows_max * tile_rows_max;
                return nstl::min(bgmmc.M, max_M);
            };
            bgmmc.M_blk = get_block_candidate();
        }

        // AMX BRGEMM kernel requires (K_brgemm % 64 == 0 || K_brgemm < 64)
        // for K_brgemm reduction value to avoid AMX tiles re-configuration.
        // To satisfy this condition K_tail value is fixed to K % wei_k_blk here.
        const bool fixed_K_tail_size
                = bgmmc.K % bgmmc.wei_k_blk > 0 && bgmmc.K > bgmmc.wei_k_blk;
        bgmmc.K_blk = bgmmc.K < bgmmc.wei_k_blk
                ? rnd_up(bgmmc.K, bgmmc.required_k_granularity)
                : fixed_K_tail_size ? bgmmc.wei_k_blk
                                    : bgmmc.K;
        bgmmc.brgemm_batch_size
                = nstl::max(bgmmc.K / bgmmc.K_blk, static_cast<dim_t>(1));

        matmul_amx_blocking_params_t best_blocking(bgmmc);

        compute_blocking_heuristic_amx(bgmmc, bm_conf_utils, best_blocking);

        VCONDCHECK_BG(best_blocking.get_blocking_scores() != 0.0f,
                VERBOSE_BLOCKING_FAIL);

        best_blocking.update_configuration(bgmmc);

    } else if (is_superset(bm_conf_utils.get_isa(), avx512_core)) {
        // TODO:
        // *) adjust K_BLK using 'rnd_up(bgmmc.K, bgmmc.required_k_granularity)'
        //    for non-f32 datatypes.
        // *) optimize param search complexity

        // Approach for selecting ideal 'blocking parameters':
        // M_blk:
        // - main param for having parallel_work optimally distributed.
        // - 'br_block' is a BRGeMM uKernel parameter derived from 'M_Blk',
        // however, there is no measured performance impact from small
        // variations in 'br_block' size.
        //
        // M_Chunks:
        // - no noticeable performance impact i.e. 'M_blk = M_Chunks * M_Blk';
        // with M_Chunks > 1', brgemm has the same performance results. Instead,
        // choose a larger 'M_blk'.
        //
        // N_blk:
        // - ideally 64 (from 'get_default_n_block()').
        // - can be reduced to 32 to improve performance for some shapes, as
        //  well as increasing parallelization search space.
        //
        // N_Chunks:
        // - No different as long as thread/work balance is the same.
        // - Note: for A_Transposed cases using A_buffer (i.e. bwd-w): select
        // a higher count to increase performance -better for transposed data
        // reuse.
        //
        // K_blk:
        // - block size variation '512 <= K_blk < 1024' has negligible
        // performance difference. However, Some cases benefit from higher
        // block size.
        // - can parallelize if not enough work; notice: requires reduction!
        //
        // Batch_Size:
        // - unused.

        const matmul_avx512_blocking_params_t::matmul_params_t matmul(
                bgmmc.M, bgmmc.N, bgmmc.K, bgmmc.batch);

        matmul_avx512_blocking_params_t best_blocking(matmul, bgmmc.nthr);

        const float best_imbalance = compute_blocking_heuristic_avx512(
                bgmmc, bm_conf_utils, matmul, best_blocking);

        if (best_imbalance == 1.f) return status::unimplemented;

        best_blocking.update_configuration(bgmmc);
    } else {
        assert(one_of(bm_conf_utils.get_isa(), avx2_vnni, avx2_vnni_2));

        const matmul_avx512_blocking_params_t::matmul_params_t matmul(
                bgmmc.M, bgmmc.N, bgmmc.K, bgmmc.batch);

        matmul_avx512_blocking_params_t best_blocking(matmul, bgmmc.nthr);

        const float best_imbalance = compute_blocking_heuristic_avx2(
                bgmmc, bm_conf_utils, matmul, best_blocking);

        if (best_imbalance == 1.f) return status::unimplemented;

        best_blocking.update_configuration(bgmmc);
    }

    return status::success;
}

status_t init_brgemm_matmul_conf(cpu_isa_t isa, brgemm_matmul_conf_t &bgmmc,
        const matmul_desc_t &mmd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);

    bgmmc = zero<decltype(bgmmc)>();
    bgmmc.isa = isa;
    bgmmc.nthr = dnnl_get_max_threads();
    bgmmc.brg_type = brgemm_addr;

    bgmmc.src_dt = src_d.data_type();
    bgmmc.dst_dt = dst_d.data_type();
    bgmmc.wei_dt = weights_d.data_type();

    bgmmc.with_bias = mmd.bias_desc.format_kind != format_kind::undef;
    bgmmc.bia_dt = bgmmc.with_bias ? mmd.bias_desc.data_type : data_type::undef;
    bgmmc.s8s8_compensation_required = bgmmc.src_dt == s8 && !isa_has_s8s8(isa);
    bgmmc.ndims = dst_d.ndims();

    brgemm_matmul_conf_utils_t bm_conf_utils(bgmmc, isa, attr,
            src_d.format_kind() == format_kind::any,
            weights_d.format_kind() == format_kind::any,
            dst_d.format_kind() == format_kind::any,
            bias_md.format_kind == format_kind::any);

    VCHECK_BG(check_isa_with_datatype(isa, bm_conf_utils),
            VERBOSE_ISA_DT_MISMATCH);

    bgmmc.is_amx = is_superset(isa, avx512_core_amx);
    bgmmc.a_dt_sz = bgmmc.tr_a_dt_sz = types::data_type_size(bgmmc.src_dt);
    bgmmc.b_dt_sz = bgmmc.tr_b_dt_sz = types::data_type_size(bgmmc.wei_dt);

    bgmmc.is_bf32 = bm_conf_utils.is_bf32();

    // Make BRGeMM compute MatMul as if it were in bfloat16, while down-convert
    // happens during copy-buffer computations
    if (bgmmc.is_bf32) {
        bgmmc.src_dt = bf16;
        bgmmc.wei_dt = bf16;
        bgmmc.tr_a_dt_sz = types::data_type_size(bf16);
        bgmmc.tr_b_dt_sz = types::data_type_size(bf16);
    } else if (bm_conf_utils.is_f16() && bgmmc.isa == avx512_core_fp16) {
        // Similar to bf32, convert input data before compute
        bgmmc.src_dt = f32;
        bgmmc.wei_dt = f32;
        bgmmc.tr_a_dt_sz = types::data_type_size(f32);
        bgmmc.tr_b_dt_sz = types::data_type_size(f32);
    }

    bgmmc.acc_dt = bm_conf_utils.is_int8() ? s32 : f32;

    bgmmc.c_dt_sz = types::data_type_size(bgmmc.dst_dt);
    bgmmc.acc_dt_sz = types::data_type_size(bgmmc.acc_dt);
    if (bgmmc.with_bias) bgmmc.bias_dt_sz = types::data_type_size(bgmmc.bia_dt);

    const auto &src_scales = attr.scales_.get(DNNL_ARG_SRC);
    const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
    bgmmc.with_scales = !src_scales.has_default_values()
            || !wei_scales.has_default_values();
    if (bgmmc.with_scales) {
        bgmmc.is_oscale_per_n = wei_scales.mask_ == 1 << (bgmmc.ndims - 1);

        // only common and per-oc-channel scales are supported
        VCONDCHECK_BG(wei_scales.mask_ == 0 || bgmmc.is_oscale_per_n,
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    const auto &dst_scales = attr.scales_.get(DNNL_ARG_DST);
    bgmmc.with_dst_scales = !dst_scales.has_default_values();
    // only common scales are supported
    if (bgmmc.with_dst_scales && dst_scales.mask_ != 0)
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    bgmmc.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    bgmmc.with_eltwise = eltwise_ind != -1;
    const int binary_ind = p.find(primitive_kind::binary);
    const int prelu_ind = p.find(primitive_kind::prelu);
    bgmmc.with_binary = !everyone_is(-1, binary_ind, prelu_ind);
    VCONDCHECK_BG(post_ops_ok(bgmmc, attr, dst_d), VERBOSE_UNSUPPORTED_POSTOP);

    bgmmc.src_zp_type = get_zp_type(attr, DNNL_ARG_SRC);
    bgmmc.wei_zp_type = get_zp_type(attr, DNNL_ARG_WEIGHTS);
    bgmmc.dst_zp_type = get_zp_type(attr, DNNL_ARG_DST);

    VCONDCHECK_BG(
            IMPLICATION(!bm_conf_utils.is_int8(),
                    everyone_is(brgemm_broadcast_t::none, bgmmc.src_zp_type,
                            bgmmc.wei_zp_type, bgmmc.dst_zp_type)),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    matmul_helper_t helper(src_d, weights_d, dst_d);

    bgmmc.batch_ndims = bgmmc.ndims - 2;
    bgmmc.M = helper.M();
    bgmmc.N = helper.N();
    bgmmc.K = helper.K();
    bgmmc.batch = helper.batch();
    bgmmc.is_runtime_M = is_runtime_value(bgmmc.M);
    bgmmc.is_runtime_N = is_runtime_value(bgmmc.N);
    bgmmc.is_runtime_K = is_runtime_value(bgmmc.K);

    // runtime value for M dimension is only supported
    if (is_runtime_value(bgmmc.batch) || bgmmc.is_runtime_N
            || bgmmc.is_runtime_K)
        return status::unimplemented;

    // Runtime value for M dimension is supported for 2d AMX int8/bfloat16
    // problems only.
    const bool runtime_M_supported = bgmmc.is_amx && bgmmc.ndims == 2
            && one_of(true, bm_conf_utils.is_int8(), bm_conf_utils.is_bf16());
    if (bgmmc.is_runtime_M && !runtime_M_supported)
        return status::unimplemented;

    bgmmc.batch_without_first_dim
            = bgmmc.batch_ndims > 1 ? helper.batch() / dst_d.dims()[0] : 0;

    bgmmc.bcast_A_desc.set_params(
            src_d.dims(), dst_d.dims(), bgmmc.batch_ndims, bgmmc.batch);
    bgmmc.bcast_B_desc.set_params(
            weights_d.dims(), dst_d.dims(), bgmmc.batch_ndims, bgmmc.batch);

    // required granularity for k dimension
    bgmmc.required_k_granularity
            = bgmmc.is_amx ? data_type_vnni_granularity(bgmmc.wei_dt) : 1;
    VCONDCHECK_BG(bgmmc.required_k_granularity > 0, VERBOSE_BLOCKING_FAIL);
    bgmmc.wei_k_blk = data_type_vnni_simd_elems<avx512_core>(bgmmc.wei_dt);

    VCHECK_BG(bm_conf_utils.set_or_check_tags(src_md, dst_md, bias_md),
            VERBOSE_UNSUPPORTED_TAG);
    VCHECK_BG(bm_conf_utils.set_or_check_B_tag(weights_md),
            VERBOSE_UNSUPPORTED_TAG);

    bgmmc.req_wei_vnni_downconvert = bm_conf_utils.wei_down_convert_to_vnni();

    VCHECK_BG(attr.set_default_formats(&dst_md), VERBOSE_UNSUPPORTED_TAG);

    bgmmc.wei_n_blk = get_default_n_block(bgmmc.wei_tag);

    bgmmc.blocked_B = bm_conf_utils.get_blocked_B();
    bgmmc.use_buffer_b = bm_conf_utils.use_buffer_b();

    const bool transposed_A = bm_conf_utils.check_is_transposed(bgmmc.src_tag);
    // if M == 1 we can still treat formally transposed A as plain
    // and avoid copy routine creation/execution
    const bool treat_transposed_A_as_plain = transposed_A && bgmmc.M == 1;
    bgmmc.transposed_A = ((transposed_A && !treat_transposed_A_as_plain)
            || bgmmc.src_tag == adbc);
    // For batched problems with plain A and C and fully broadcasted across B
    // we can merge all the batch dimensions into M if broadcast strategies
    // set is limited for binary post-ops
    const bool plain_A_layout = bm_conf_utils.check_is_plain(bgmmc.src_tag)
            || treat_transposed_A_as_plain;
    const bool merge_batch_dims_into_M = bgmmc.batch > 1
            && bgmmc.bcast_B_desc.bcast_across_all_batch_dims
            && bm_conf_utils.check_is_plain(bgmmc.dst_tag) && plain_A_layout
            && post_ops_ok(
                    bgmmc, attr, dst_d, true /* limit_bcast_strategies_set */);
    if (merge_batch_dims_into_M) {
        bgmmc.M *= bgmmc.batch;
        bgmmc.batch = 1;
    }

    // runtime A stride wrt M dimension is not acceptable
    if (is_runtime_value(helper.get_a_stride(bgmmc.ndims - 2)))
        return status::unimplemented;

    // runtime A stride wrt K dimension is acceptable for transpose A and
    // runtime M case only
    const bool stride_A_wrt_K_dim_ok = IMPLICATION(
            is_runtime_value(helper.get_a_stride(bgmmc.ndims - 1)),
            bgmmc.transposed_A && bgmmc.is_runtime_M);
    if (!stride_A_wrt_K_dim_ok) return status::unimplemented;

    // runtime A strides wrt batch dimensions are acceptable for runtime M case
    // only
    for (int b = 0; b < bgmmc.batch_ndims; b++) {
        if (!IMPLICATION(is_runtime_value(helper.get_a_stride(b)),
                    bgmmc.is_runtime_M))
            return status::unimplemented;
    }

    const bool lda_is_big_2pow
            = (bm_conf_utils.is_bf16()
                      || (bgmmc.is_amx && bm_conf_utils.is_f16()))
            && (bgmmc.isa != avx2_vnni_2) // no perf study yet.
            && bgmmc.lda_big_pow2() && bgmmc.M >= 1024;
    const bool is_copy_a_required
            = (bgmmc.is_amx
                      && ((bgmmc.K % bgmmc.required_k_granularity != 0)
                              || bm_conf_utils.is_bf32()))
            || (bm_conf_utils.is_f16() && isa == avx512_core_fp16)
            || bgmmc.wei_zp_type != brgemm_broadcast_t::none
            || bgmmc.transposed_A || lda_is_big_2pow;
    bgmmc.use_buffer_a = is_copy_a_required;

    // Supported computation with copy only part of A related to K_tail if
    // is_copy_a_required == true, but the current performance measurements
    // show worse performance for it in comparison with copy whole A approach
    // (especially for big K sizes).
    bgmmc.use_buffer_a_tail_only = false;

    const int dmax = nstl::min(bgmmc.ndims, 3);
    for (int d = 0; d < dmax; d++) {
        int dim = bgmmc.ndims - 1 - d;
        bgmmc.A_strides[d] = bgmmc.a_dt_sz * src_d.blocking_desc().strides[dim];
        bgmmc.B_strides[d]
                = bgmmc.b_dt_sz * weights_d.blocking_desc().strides[dim];
        bgmmc.C_strides[d] = bgmmc.c_dt_sz * dst_d.blocking_desc().strides[dim];
    }

    // We need to correct A_strides if batched dimensions are merged in M and
    // A layout is formally transposed but could be treated as plain
    if (merge_batch_dims_into_M
            && (src_d.matches_tag(acbd) || treat_transposed_A_as_plain)) {
        bgmmc.A_strides[1] = bgmmc.A_strides[2];
    }

    // We need to correct C_strides if batched dimensions are merged in M and
    // C layout is formally transposed but could be treated as plain
    if (merge_batch_dims_into_M && dst_d.matches_tag(acbd)) {
        bgmmc.C_strides[1] = bgmmc.C_strides[2];
    }

    // BF32 'Hint' Heuristic:
    // Under the following conditions, F32 through AVX512_CORE performs better
    // than using BF32 arithmetic.
    VCONDCHECK_BG(!(bgmmc.is_bf32 && (bgmmc.M < 8)
                          && ((bgmmc.wei_tag == abcd)
                                  || bm_conf_utils.is_any_B_layout())),
            VERBOSE_UNSUPPORTED_FPMATH_MODE);

    // Heuristic tries to optimize the following parameters:
    // - M_blk, M_Chunk
    // - N_blk, N_Chunk
    // - K_blk, batch_size
    // - nthr_K
    VCHECK_BG(compute_blocking_heuristic(bgmmc, bm_conf_utils),
            VERBOSE_BLOCKING_FAIL);

    if (bgmmc.wei_n_blk > bgmmc.N_blk
            && IMPLICATION(
                    bgmmc.N == bgmmc.N_blk, bgmmc.N >= bgmmc.wei_n_blk)) {
        bgmmc.wei_n_blk = bgmmc.N_blk;
        VCHECK_BG(bm_conf_utils.update_and_check_B_tag(
                          weights_md, bgmmc.wei_n_blk),
                VERBOSE_UNSUPPORTED_TAG);

        bgmmc.req_wei_vnni_downconvert
                = bm_conf_utils.wei_down_convert_to_vnni();
    }

    VCHECK_BG(bm_conf_utils.set_B_flags(weights_md), VERBOSE_BLOCKING_FAIL);

    bgmmc.M_tail = bgmmc.is_runtime_M ? 0 : bgmmc.M % bgmmc.M_blk;
    bgmmc.N_tail = bgmmc.N % bgmmc.N_blk;
    bgmmc.K_tail = bgmmc.K > bgmmc.K_blk
            ? rnd_up(bgmmc.K % bgmmc.K_blk, bgmmc.required_k_granularity)
            : 0;

    bgmmc.LDB = bm_conf_utils.get_actual_LDB();
    bgmmc.LDD = bgmmc.dst_tag == acbd ? dst_d.blocking_desc().strides[2]
                                      : bgmmc.N;
    bgmmc.LDC
            = bgmmc.use_buffer_c && bgmmc.nthr_k <= 1 ? bgmmc.N_blk : bgmmc.LDD;

    init_aux_values(bgmmc, src_d, weights_d, dst_d);

    // Dispatch small shapes to VNNI for better performance
    const bool runtime_dims
            = bgmmc.is_runtime_M || bgmmc.is_runtime_N || bgmmc.is_runtime_K;

    bool is_small_shapes = bgmmc.is_amx && !runtime_dims;

    // Disable 'small_shape' heuristic for amx_fp16 until it is validated with
    // performance measurements.
    is_small_shapes = is_small_shapes && (bgmmc.isa != avx512_core_amx_fp16);

    if (bm_conf_utils.is_bf16() || bm_conf_utils.is_f16()) {
        // empirical observation for performance breakpoint between amx and vnni bf16/f16
        const dim_t buffer_a_chunk_sz_limit = 126;
        is_small_shapes = is_small_shapes
                && bgmmc.buffer_a_chunk_sz <= buffer_a_chunk_sz_limit;
    } else {
        is_small_shapes = is_small_shapes && bgmmc.ndims < 3
                && ((bgmmc.M == 1 && bgmmc.K == 256)
                        || (bgmmc.M <= 32 && bgmmc.M * bgmmc.N <= 256)
                        || bgmmc.K <= 16);
    }
    VCONDCHECK_BG(!is_small_shapes, VERBOSE_SMALL_SHAPES);

    return status::success;
}

void init_aux_values(brgemm_matmul_conf_t &bgmmc,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &wei_d,
        const memory_desc_wrapper &dst_d) {

    bgmmc.M_chunk_elems = bgmmc.M_blk * bgmmc.M_chunk_size;
    bgmmc.N_chunk_elems = bgmmc.N_blk * bgmmc.N_chunk_size;
    bgmmc.K_chunk_elems = bgmmc.K_blk * bgmmc.brgemm_batch_size;
    bgmmc.M_chunks = div_up(bgmmc.M, bgmmc.M_chunk_elems);
    bgmmc.N_chunks = div_up(bgmmc.N, bgmmc.N_chunk_elems);
    bgmmc.K_chunks = div_up(bgmmc.K, bgmmc.K_chunk_elems);
    bgmmc.num_M_blocks = div_up(bgmmc.M, bgmmc.M_blk);
    bgmmc.num_N_blocks = div_up(bgmmc.N, bgmmc.N_blk);
    const int last_chunck_batch_size
            = (nstl::max(bgmmc.K, bgmmc.K_blk)
                      - (bgmmc.K_chunks - 1) * bgmmc.K_chunk_elems)
            / bgmmc.K_blk;
    bgmmc.brgemm_batch_tail_size
            = last_chunck_batch_size % bgmmc.brgemm_batch_size;

    bgmmc.buffer_c_chunk_sz = bgmmc.acc_dt_sz * bgmmc.LDC
            * (bgmmc.nthr_k > 1 ? bgmmc.M : bgmmc.M_blk);
    bgmmc.buffer_c_per_thread_sz = bgmmc.buffer_c_chunk_sz
            * (bgmmc.nthr_k > 1 ? 1 : bgmmc.M_chunk_size * bgmmc.N_chunk_size);

    bgmmc.buffer_a_chunk_sz = bgmmc.tr_a_dt_sz * bgmmc.M_blk
            * (bgmmc.use_buffer_a_tail_only ? bgmmc.wei_k_blk : bgmmc.LDA);
    bgmmc.buffer_a_chunk_shift_along_m = bgmmc.buffer_a_chunk_sz
            * (bgmmc.use_buffer_a_tail_only ? 1 : bgmmc.brgemm_batch_size);
    bgmmc.buffer_a_per_thread_sz
            = bgmmc.buffer_a_chunk_shift_along_m * bgmmc.M_chunk_size;

    bgmmc.buffer_b_chunk_sz = bgmmc.tr_b_dt_sz * bgmmc.LDB
            * rnd_up(bgmmc.K_blk, bgmmc.wei_k_blk);
    bgmmc.buffer_b_per_thread_sz
            = bgmmc.buffer_b_chunk_sz * bgmmc.brgemm_batch_size;

    bgmmc.s8s8_comp_ithr_str
            = bgmmc.use_buffer_b ? bgmmc.wei_n_blk * bgmmc.N_chunk_size : 0;
    bgmmc.s8s8_comp_b_str = bgmmc.use_buffer_b
            ? 0
            : div_up(bgmmc.N, bgmmc.wei_n_blk) * bgmmc.wei_n_blk;
    bgmmc.s8s8_comp_n_str = bgmmc.wei_n_blk;

    bgmmc.A_ptr_shift_b = 0;
    bgmmc.copy_A_src_stride
            = bgmmc.a_dt_sz * (bgmmc.transposed_A ? bgmmc.M : bgmmc.K);

    // If src have dimensions equal to 1, multiple tags can be matched so
    // we need to make sure:
    // - A_ptr_shift_b is set for acbd and adbc even if bgmmc.src_tag is abcd
    // - Plain md that matches acbd or adbc does not dispatch into their codepath
    if (src_d.matches_one_of_tag(acbd, adbc)) {
        if (src_d.matches_one_of_tag(abcd, abdc) == format_tag::undef) {
            const dim_t factor = bgmmc.src_dt == f32 ? 2 : 1;
            const dim_t src_stride = src_d.matches_tag(acbd)
                    ? bgmmc.A_strides[1]
                    : bgmmc.A_strides[0];
            bgmmc.copy_A_src_stride
                    = nstl::min(src_d.blocking_desc().strides[0],
                              src_stride / factor)
                    * factor;
        }

        const dim_t bcast_shift_b = src_d.matches_tag(acbd) ? bgmmc.K : bgmmc.M;
        bgmmc.A_ptr_shift_b
                = (bgmmc.bcast_A_desc.bcast_mask == 2
                                  ? bcast_shift_b
                                  : src_d.blocking_desc().strides[0])
                * bgmmc.a_dt_sz;
    }

    bgmmc.B_ptr_shift_b = 0;
    bgmmc.copy_B_wei_stride = 0;

    // If weights have dimensions equal to 1, multiple tags can be matched so
    // we need to make sure:
    // - B_ptr_shift_b is set for acbd and adbc even if bgmmc.wei_tag is abcd
    // - Plain md that matches acbd or adbc does not dispatch into their codepath
    if (wei_d.matches_one_of_tag(acbd, adbc) != format_tag::undef) {
        if (wei_d.matches_one_of_tag(abcd, abdc) == format_tag::undef) {
            const dim_t factor = bgmmc.wei_dt == f32 ? 2 : 1;
            const dim_t wei_stride = wei_d.matches_tag(acbd)
                    ? bgmmc.B_strides[1]
                    : bgmmc.B_strides[0];
            bgmmc.copy_B_wei_stride
                    = nstl::min(wei_d.blocking_desc().strides[0],
                              wei_stride / factor)
                    * factor;
        }

        const dim_t bcast_shift_b = wei_d.matches_tag(acbd) ? bgmmc.N : bgmmc.K;
        bgmmc.B_ptr_shift_b
                = (bgmmc.bcast_B_desc.bcast_mask == 2
                                  ? bcast_shift_b
                                  : wei_d.blocking_desc().strides[0])
                * bgmmc.b_dt_sz;
    }

    bgmmc.C_ptr_shift_b = dst_d.matches_one_of_tag(acbd)
            ? dst_d.blocking_desc().strides[0] * bgmmc.c_dt_sz
            : 0;

    bgmmc.has_zero_point_a = bgmmc.src_zp_type != brgemm_broadcast_t::none;
    bgmmc.has_zero_point_b = bgmmc.wei_zp_type != brgemm_broadcast_t::none;
    bgmmc.has_zero_point_c = bgmmc.dst_zp_type != brgemm_broadcast_t::none;
    bgmmc.post_ops_applicable = one_of(true, bgmmc.with_sum, bgmmc.with_bias,
            bgmmc.with_scales, bgmmc.with_eltwise, bgmmc.with_binary,
            bgmmc.acc_dt != bgmmc.dst_dt, bgmmc.s8s8_compensation_required,
            bgmmc.has_zero_point_a, bgmmc.has_zero_point_b,
            bgmmc.has_zero_point_c, bgmmc.with_dst_scales);

    bgmmc.zp_a_comp_shift_n = bgmmc.wei_n_blk;
    bgmmc.zp_a_comp_elems_per_thr
            = bgmmc.N_chunk_size * bgmmc.zp_a_comp_shift_n;

    const int s32_elems_in_cacheline = 16;
    bgmmc.zp_b_comp_result_shift_m = bgmmc.M_blk;
    bgmmc.zp_b_comp_buffer_start
            = bgmmc.M_chunk_size * bgmmc.zp_b_comp_result_shift_m;
    bgmmc.zp_b_comp_buffer_shift_m = s32_elems_in_cacheline * bgmmc.M_blk;
    bgmmc.zp_b_comp_elems_per_thr = bgmmc.M_chunk_size
            * (bgmmc.zp_b_comp_result_shift_m + bgmmc.zp_b_comp_buffer_shift_m);

    bgmmc.brgemm_batch_element_per_thr_sz = 16 * bgmmc.brgemm_batch_size;
}

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const brgemm_matmul_conf_t &bgmmc) {
    const size_t default_data_align = sizeof(char);
    if (bgmmc.brg_type == brgemm_addr)
        scratchpad.book(key_brgemm_primitive_batch,
                static_cast<size_t>(bgmmc.nthr)
                        * bgmmc.brgemm_batch_element_per_thr_sz,
                sizeof(brgemm_batch_element_t), 64);

    if (bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only)
        scratchpad.book(key_brgemm_primitive_buffer_a,
                bgmmc.nthr * bgmmc.buffer_a_per_thread_sz, default_data_align);

    if (bgmmc.use_buffer_b) {
        scratchpad.book(key_brgemm_primitive_buffer_b,
                bgmmc.nthr * bgmmc.buffer_b_per_thread_sz, default_data_align);

        if (bgmmc.s8s8_compensation_required && (!bgmmc.blocked_B))
            scratchpad.book(key_brgemm_primitive_buffer_comp,
                    bgmmc.nthr * bgmmc.s8s8_comp_ithr_str,
                    types::data_type_size(f32));
    }

    if (bgmmc.use_buffer_c)
        scratchpad.book(key_brgemm_primitive_buffer,
                bgmmc.nthr * bgmmc.buffer_c_per_thread_sz, default_data_align);

    if (bgmmc.has_zero_point_a) {
        const auto num_elems = bgmmc.nthr * bgmmc.zp_a_comp_elems_per_thr;
        scratchpad.book(key_brgemm_primitive_zp_comp_a, num_elems,
                types::data_type_size(s32));
    }

    if (bgmmc.has_zero_point_b)
        scratchpad.book(key_brgemm_primitive_zp_comp_b,
                bgmmc.nthr * bgmmc.zp_b_comp_elems_per_thr,
                types::data_type_size(s32));

    if (is_superset(bgmmc.isa, avx512_core_amx))
        scratchpad.book(key_conv_amx_tile_buffer,
                static_cast<size_t>(bgmmc.nthr) * bgmmc.wsp_tile_per_thr_bytes,
                default_data_align);
    if (bgmmc.is_runtime_M)
        scratchpad.book(key_brgemm_primitive_buffer_d,
                bgmmc.LDD * bgmmc.M_blk * bgmmc.M_chunk_size * bgmmc.c_dt_sz,
                default_data_align);
}

void matmul_amx_blocking_params_t::update_k_blocking_dependent_params() {
    k_chunk_elems_ = k_blk_ * k_chunk_size_;
    current_lda_ = get_actual_lda();
    need_buf_c_ = is_buffer_c_required();
}

void matmul_amx_blocking_params_t::set_blocking_parameters(
        int nthr_k, int n_blk, int n_chunk_size, int m_blk, int m_chunk_size) {
    nthr_k_ = nstl::max(1, nthr_k);
    nthr_mnb_ = nthr / nthr_k_;
    nthr_ = nthr_mnb_ * nthr_k_;
    n_blk_ = n_blk;
    n_chunk_size_ = n_chunk_size;
    m_blk_ = m_blk;
    m_chunk_size_ = m_chunk_size;
    if (one_of(0, n_blk_, n_chunk_size_, m_blk_, m_chunk_size_)) {
        k_blk_ = k_chunk_size_ = k_chunk_elems_ = 0;
        efficiency_score_ = 0.0f;
        return;
    }

    n_chunk_elems_ = n_blk_ * n_chunk_size_;
    m_chunk_elems_ = m_blk_ * m_chunk_size_;

    if (K < wei_k_blk) {
        k_blk_ = is_amx ? rnd_up(K, required_k_granularity) : K;
        k_chunk_size_ = 1;
    } else {
        dim_t k_per_thr = div_up(K, nthr_k_);
        k_blk_ = nstl::min(rnd_up(k_per_thr, required_k_granularity),
                static_cast<dim_t>(wei_k_blk));
        const dim_t num_k_blk = div_up(K, k_blk_);
        const dim_t num_k_blk_per_thread = div_up(num_k_blk, nthr_k_);
        k_chunk_size_ = num_k_blk_per_thread;

        auto chunk_sz = calculate_chunk_memory_size();
        const dim_t div_min = chunk_sz / L2_threshold();
        const dim_t div_max = div_up(chunk_sz, L2_threshold());
        // for big pow2 lda prefer to increase area of linear memory access
        const dim_t adjust_k_divisor_threshold = lda_big_pow2() ? 2 : 0;
        // adjust k blocking values to fit into L2 cache
        if (div_min > adjust_k_divisor_threshold && k_chunk_size_ > 1) {
            const auto kc1
                    = nstl::max(k_chunk_size_ / div_min, static_cast<dim_t>(1));
            const auto kc2 = div_up(k_chunk_size_, div_max);
            const auto tail1 = num_k_blk_per_thread % kc1;
            const auto tail2 = num_k_blk_per_thread % kc2;
            // prefer adjusted chunk size with more equal work distribution
            // across iterations
            k_chunk_size_ = IMPLICATION(tail1 == 0 || tail2 < tail1, tail2 == 0)
                    ? kc2
                    : kc1;
        }

        const dim_t current_k_tail = K % k_blk_;
        if (current_k_tail == 0 && K % (k_blk_ * k_chunk_size_) == 0) {
            k_blk_ *= k_chunk_size_;
            k_chunk_size_ = 1;
        } else if (nthr_k_ == 1
                && K == k_blk_ * k_chunk_size_ + current_k_tail) {
            k_blk_ *= k_chunk_size_;
            k_chunk_size_ = 2;
        }
    }

    blocking_chunk_mem_size_ = calculate_chunk_memory_size();

    efficiency_score_ = calculate_blocking_scores();
}

// returns score for current blocking parameters' values in range [0, 1]
// for parallel work over threads distribution score. Maximum scores - when
// all threads have the same work amount w/o tails
float matmul_amx_blocking_params_t::get_thread_balance_scores() {
    // Ignore M sizes in thread balance computation as actual M size is unknown
    if (is_runtime_M) return (float)N / rnd_up(N, n_chunk_elems_);

    dim_t num_M_chunks = div_up(M, m_chunk_elems_);
    dim_t num_N_chunks = div_up(N, n_chunk_elems_);
    float mnb_parallel_score = batch * ((float)M / m_chunk_elems_)
            * ((float)N / n_chunk_elems_)
            / rnd_up(batch * num_M_chunks * num_N_chunks, nthr_mnb_)
            * nthr_mnb_;
    float k_parallel_score = 1.0f;
    if (nthr_k_ > 1) {
        dim_t num_K_chunks = div_up(K, k_chunk_elems_);
        const float parallel_reduction_penalty = 0.8f;
        k_parallel_score = parallel_reduction_penalty
                * ((float)K / k_chunk_elems_) / rnd_up(num_K_chunks, nthr_k_)
                * nthr_k_;
    }

    return mnb_parallel_score * k_parallel_score / nthr;
}

// returns score for current blocking parameters' values in range [0, 1]
// for copied data reusage
float matmul_amx_blocking_params_t::get_copied_data_reusage_scores() {
    const dim_t effective_m_chunk_sz = 64 * 4;
    const dim_t desired_M_chunk_size = is_runtime_M
            ? effective_m_chunk_sz
            : nstl::min(M, effective_m_chunk_sz);
    const dim_t effective_n_chunk_sz = 64 * (use_buffer_a ? 4 : 1);
    const dim_t desired_N_chunk_size = nstl::min(N, effective_n_chunk_sz);
    const float coef_M = nstl::min(
            static_cast<float>(m_chunk_elems_) / desired_M_chunk_size, 1.0f);
    const float coef_N = nstl::min(
            static_cast<float>(n_chunk_elems_) / desired_N_chunk_size, 1.0f);
    return 0.5f * (coef_M + coef_N);
}

// returns score for current blocking parameters' values in range [0, 1]
// for L2 utilization
float matmul_amx_blocking_params_t::get_L2_utilization_scores() const {
    const float relative_difference_with_L2
            = fabsf((float)L2_threshold() - blocking_chunk_mem_size_)
            / nstl::max(L2_threshold(), blocking_chunk_mem_size_);
    return 1.0f - relative_difference_with_L2;
}

// returns score for current blocking parameters' values in range [0, 1]
// consists of 3 parts with its own weights:
// 	1) parallel work over threads distribution score
// 	2) L2 utilization score
// 	3) copied data re-usage score
float matmul_amx_blocking_params_t::calculate_blocking_scores() {
    if (one_of(0, n_blk_, n_chunk_size_, m_blk_, m_chunk_size_, k_blk_,
                k_chunk_size_))
        return 0.0f;

    const float nthr_coeff = nstl::min(nthr, 100);
    const float reusage_factor = 1.0f;
    // for runtume M the actual size is unknown, use independent on num_threads
    // balance factors
    const float balance_factor
            = is_runtime_M ? 1.0f : (nthr_coeff - 1.0f) / nthr_coeff;
    const float cache_utilization_factor
            = is_runtime_M ? 1.0f : 1.0f / nthr_coeff;

    float scores = cache_utilization_factor * get_L2_utilization_scores()
            + reusage_factor * get_copied_data_reusage_scores();
    if (balance_factor > 0.0f)
        scores += balance_factor * get_thread_balance_scores();
    return scores
            / (reusage_factor + balance_factor + cache_utilization_factor);
}

void matmul_amx_blocking_params_t::update_configuration(
        brgemm_matmul_conf_t &bgmmc) const {
    bgmmc.nthr_k = nthr_k_;
    bgmmc.M_blk = m_blk_;
    bgmmc.M_chunk_size = m_chunk_size_;
    bgmmc.N_blk = n_blk_;
    bgmmc.N_chunk_size = n_chunk_size_;

    bgmmc.K_blk = k_blk_;
    bgmmc.brgemm_batch_size = k_chunk_size_;

    bgmmc.use_buffer_c = need_buf_c_;
    bgmmc.LDA = current_lda_;
}

dim_t matmul_amx_blocking_params_t::get_actual_lda() {
    if (!use_buffer_a) return src_tag == acbd ? A_strides[1] / a_dt_sz : K;

    constexpr int bytes_in_cacheline = 64;
    const int elems_in_cacheline = bytes_in_cacheline / a_dt_sz;
    dim_t lda = rnd_up(k_blk_, elems_in_cacheline);
    const bool is_big_2_pow = lda >= 512 && math::is_pow2(lda);
    if (is_big_2_pow) lda += elems_in_cacheline;
    return lda;
}

bool matmul_amx_blocking_params_t::is_buffer_c_required() {
    if (nthr_k_ > 1 && K > k_chunk_elems_) return true;

    return ((acc_dt != dst_dt || with_sum)
            && (K > k_chunk_elems_ || K % k_blk_ > 0));
}

size_t matmul_amx_blocking_params_t::calculate_chunk_memory_size() {
    update_k_blocking_dependent_params();

    size_t A_chunk_sz = a_dt_sz * k_chunk_elems_ * m_chunk_elems_;
    size_t A_buf_sz = use_buffer_a
            ? tr_a_dt_sz * current_lda_ * k_chunk_size_ * m_chunk_elems_
            : 0;
    size_t B_chunk_sz = b_dt_sz * k_chunk_elems_ * n_chunk_elems_;
    size_t B_buf_sz = use_buffer_b ? tr_b_dt_sz * n_blk_ * k_chunk_elems_ : 0;
    size_t C_chunk_sz = c_dt_sz * m_chunk_elems_ * n_chunk_elems_;
    size_t C_buf_sz
            = need_buf_c_ ? acc_dt_sz * m_chunk_elems_ * n_chunk_elems_ : 0;
    return A_chunk_sz + A_buf_sz + B_chunk_sz + B_buf_sz + C_chunk_sz
            + C_buf_sz;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

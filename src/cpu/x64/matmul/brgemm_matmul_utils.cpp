/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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
#include "cpu/binary_injector_utils.hpp"
#include "cpu/matmul/gemm_based_common.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/matmul/amx_blocking_heuristics.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"
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

int get_n_block_from_tag(format_tag_t matrix_b_tag) {
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
        case aCB2b24c:
        case BA8a24b: return 24;
        case aCB16b16c:
        case aCB16b16c2b:
        case aCB16b16c4b:
        case BA16a16b:
        case BA16a16b2a:
        case BA16a16b4a: return 16;
        case aCB2b8c:
        case BA8a8b: return 8;
        default: return 0;
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
    bool is_binary_po_per_oc_d_bcast {};
    bool is_binary_po_channel_bcast {};
    bool is_binary_po_per_mb_bcast {};
    bool is_binary_po_per_mb_w_bcast {};
    bool is_binary_po_per_w_bcast {};
    bool is_binary_po_batch_bcast {};
    std::tie(is_binary_po_per_oc_sp_bcast, is_binary_po_per_oc_d_bcast,
            is_binary_po_channel_bcast, is_binary_po_per_mb_bcast,
            is_binary_po_per_mb_w_bcast, is_binary_po_per_w_bcast,
            is_binary_po_batch_bcast)
            = binary_injector_utils::bcast_strategies_present_tup(
                    post_ops.entry_, dst_d,
                    broadcasting_strategy_t::per_oc_spatial,
                    broadcasting_strategy_t::per_oc_d,
                    broadcasting_strategy_t::per_mb,
                    broadcasting_strategy_t::per_mb_spatial,
                    broadcasting_strategy_t::per_mb_w,
                    broadcasting_strategy_t::per_w,
                    broadcasting_strategy_t::batch);
    const bool supported_binary_bcast
            = IMPLICATION(is_binary_po_per_oc_sp_bcast, ndims < 4)
            && IMPLICATION(is_binary_po_per_oc_d_bcast, ndims == 4)
            && IMPLICATION(
                    is_binary_po_channel_bcast, utils::one_of(ndims, 3, 4))
            && IMPLICATION(
                    is_binary_po_per_mb_w_bcast, utils::one_of(ndims, 3, 4))
            && IMPLICATION(is_binary_po_per_w_bcast, utils::one_of(ndims, 3, 4))
            && IMPLICATION(
                    is_binary_po_per_mb_bcast, utils::one_of(ndims, 3, 4))
            && IMPLICATION(
                    is_binary_po_batch_bcast, utils::one_of(ndims, 3, 4));
    const bcast_set_t default_bcast_set = {broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::per_oc_d, broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::per_mb,
            broadcasting_strategy_t::per_mb_spatial,
            broadcasting_strategy_t::per_mb_w, broadcasting_strategy_t::per_w,
            broadcasting_strategy_t::batch,
            broadcasting_strategy_t::no_broadcast};
    const bcast_set_t limited_bcast_set = {broadcasting_strategy_t::scalar,
            broadcasting_strategy_t::no_broadcast};
    const bcast_set_t bcast_set
            = limit_bcast_strategies_set || bgmmc.is_runtime_N
            ? limited_bcast_set
            : default_bcast_set;
    // binary post-ops are disabled for runtime N due to issues in
    // injector::post_ops_ok() with identification of per_oc strategy
    // for some cases
    std::vector<post_op_type> accepted_post_ops;
    accepted_post_ops.push_back(sum);
    accepted_post_ops.push_back(eltwise);
    if (!bgmmc.is_runtime_N) accepted_post_ops.push_back(binary);
    return supported_binary_bcast
            && injector::post_ops_ok(
                    post_ops_ok_args_t(get_max_cpu_isa(), accepted_post_ops,
                            post_ops, &dst_d, false /*sum_at_pos_0_only*/,
                            false /*sum_requires_scale_one*/,
                            false /*sum_requires_zp_zero*/,
                            true /*sum_requires_same_params*/, bcast_set));
}

// Trivial: The motivation is to compute batch offset for a memory
// (src or wei or dst) with minimal overhead, by using
// `batch_offset = b * b_stride`.
// This is possible when the batch layout is contiguous.
bool is_batch_layout_trivial(
        const memory_desc_wrapper &mdw, const dim_t batch) {
    const int ndims = mdw.ndims();
    if (ndims <= 3) return true;

    const auto &strides = mdw.strides();
    const int batch_start_idx = ndims - 3;
    dim_t cur_stride = strides[batch_start_idx];
    dim_t min_batch_stride = cur_stride;
    dim_t max_batch_stride = cur_stride;
    for (int d = batch_start_idx - 1; d >= 0; --d) {
        cur_stride = strides[d];
        min_batch_stride = nstl::min(min_batch_stride, cur_stride);
        max_batch_stride = nstl::max(max_batch_stride, cur_stride);
    }
    if (min_batch_stride == 0) return false;
    return max_batch_stride / min_batch_stride == batch;
}

status_t check_isa_with_datatype(
        const cpu_isa_t isa, const brgemm_matmul_conf_utils_t &bm_conf_utils) {
    const bool ok
            = IMPLICATION(bm_conf_utils.is_f32(),
                      one_of(isa, avx512_core, avx2) || bm_conf_utils.is_bf32())
            && IMPLICATION(bm_conf_utils.is_int8(),
                    is_superset(isa, avx512_core)
                            || is_superset(isa, avx2_vnni))
            && IMPLICATION(bm_conf_utils.is_bf16(),
                    one_of(isa, avx512_core_amx, avx512_core_bf16, avx2_vnni_2))
            && IMPLICATION(bm_conf_utils.is_f16(),
                    one_of(isa, avx512_core_amx_fp16, avx512_core_fp16,
                            avx2_vnni_2))
            // `avx512_core_amx_fp16` is not supported for plain upconversion
            // as HW supports native compute.
            && IMPLICATION(bm_conf_utils.is_f32_f16(),
                    one_of(isa, avx512_core_fp16, avx2_vnni_2, avx512_core,
                            avx2))
            // `avx512_core_amx` is not supported for plain upconversion as HW
            // supports native compute.
            && IMPLICATION(bm_conf_utils.is_f32_bf16(),
                    one_of(isa, avx512_core_bf16, avx2_vnni_2, avx512_core,
                            avx2))
            && IMPLICATION(bm_conf_utils.is_int8_with_bf16_dst(),
                    is_superset(isa, avx512_core) || isa == avx2_vnni_2)
            && IMPLICATION(bm_conf_utils.is_bf16_with_int_wei(),
                    is_superset(isa, avx512_core_bf16))
            && IMPLICATION(bm_conf_utils.is_f16_with_int_wei(),
                    one_of(isa, avx512_core_amx_fp16, avx512_core_fp16))
            && IMPLICATION(bm_conf_utils.is_f8(),
                    is_superset(isa, avx512_core_amx_fp16));
    return ok ? status::success : status::unimplemented;
}

status_t check_datatype_cfg(const brgemm_matmul_conf_utils_t &bm_conf_utils) {
    const bool ok
            = one_of(true, bm_conf_utils.is_f32(), bm_conf_utils.is_bf16(),
                      bm_conf_utils.is_f16(), bm_conf_utils.is_f32_f16(),
                      bm_conf_utils.is_f32_bf16(), bm_conf_utils.is_bf32(),
                      bm_conf_utils.is_f8(), bm_conf_utils.is_int8(),
                      bm_conf_utils.is_bf16_with_int_wei(),
                      bm_conf_utils.is_f16_with_int_wei())
            && IMPLICATION(bm_conf_utils.is_bf16_with_int_wei()
                            || bm_conf_utils.is_f16_with_int_wei(),
                    bm_conf_utils.with_weights_decompression());
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
    , f8_dt(one_of(bgmmc.src_dt, f8_e5m2, f8_e4m3)
              && one_of(bgmmc.wei_dt, f8_e5m2, f8_e4m3)
              && one_of(bgmmc.dst_dt, f16, f32, bf16, f8_e5m2, f8_e4m3))
    , int8_dt(utils::one_of(bgmmc.src_dt, u8, s8) && bgmmc.wei_dt == s8
              && one_of(bgmmc.dst_dt, u8, s8, s32, f32, bf16))
    , bf32_dt(f32_dt
              && one_of(attr.fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::any)
              && isa == avx512_core_amx)
    , weights_decompression_support(one_of(bgmmc.wei_dt, u8, s8, u4, s4)
              && one_of(attr.fpmath_.mode_, fpmath_mode::bf16, fpmath_mode::f16,
                      fpmath_mode::any)
              && IMPLICATION(attr.fpmath_.mode_ == fpmath_mode::f16,
                      bgmmc.src_dt == f16)
              && IMPLICATION(attr.fpmath_.mode_ == fpmath_mode::bf16,
                      bgmmc.src_dt == bf16)
              && attr.fpmath_.apply_to_int_)
    , bf16_with_int_wei_dt(weights_decompression_support && bgmmc.src_dt == bf16
              && one_of(bgmmc.dst_dt, bf16, f32))
    // Keep this var separate from f16_dt to not slip f16:f16 on avx512_core and
    // avx2 as there's no kernel for such combination.
    , f32_f16_dt(bgmmc.src_dt == f32 && bgmmc.wei_dt == f16
              && one_of(bgmmc.dst_dt, f16, f32))
    // Keep this var separate from bf16_dt to not slip bf16:bf16 on avx512_core
    // and avx2 as there's no kernel for such combination.
    , f32_bf16_dt(bgmmc.src_dt == f32 && bgmmc.wei_dt == bf16
              && one_of(bgmmc.dst_dt, bf16, f32))
    , f16_with_int_wei_dt(weights_decompression_support && bgmmc.src_dt == f16
              && one_of(bgmmc.dst_dt, f16, f32))
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
    , blocked_24n_B_layout_tag(pick_blocked_B_layout(24))
    , blocked_16n_B_layout_tag(pick_blocked_B_layout(16))
    , blocked_8n_B_layout_tag(pick_blocked_B_layout(8))
    , blocked_B_layouts_allowed(IMPLICATION(is_f32(),
                                        !utils::one_of(format_tag::undef,
                                                blocked_64n_B_layout_tag,
                                                blocked_48n_B_layout_tag,
                                                blocked_32n_B_layout_tag,
                                                blocked_24n_B_layout_tag,
                                                blocked_16n_B_layout_tag,
                                                blocked_8n_B_layout_tag))
              && IMPLICATION(!is_f32(),
                      !utils::one_of(format_tag::undef,
                              blocked_64n_B_layout_tag,
                              blocked_48n_B_layout_tag,
                              blocked_32n_B_layout_tag,
                              blocked_16n_B_layout_tag)))
    , n_blk_fixed((!B_any_layout) && blocked_B_layouts_allowed)
    , isa_(isa) {}

int brgemm_matmul_conf_utils_t::get_default_n_block(
        format_tag_t matrix_b_tag) const {
    const int n_blk = get_n_block_from_tag(matrix_b_tag);
    if (n_blk > 0) return n_blk;

    const int simd_w = isa_max_vlen(isa_) / sizeof(float);

    if (matmul_amx_blocking_params_macro_t::is_supported(bgmmc, *this)) {
        return 32;
    }

    return is_superset(isa_, avx512_core) || !f32_dt
            ? 64
            : nstl::min<int>(24, rnd_up(bgmmc.N, simd_w));
}

status_t brgemm_matmul_conf_utils_t::set_or_check_B_tag(memory_desc_t &B_md,
        const matmul_helper_t &helper, bool init_n_tag) const {
    const memory_desc_wrapper B_d(&B_md);
    if (B_any_layout) {
        const int default_n_block = init_n_tag
                ? get_default_n_block(format_tag::undef)
                : bgmmc.N_blk;
        bgmmc.wei_tag = blocked_B_layouts_allowed && !bgmmc.is_runtime_N
                        && !bgmmc.is_int4_weights
                ? this->pick_blocked_B_layout(default_n_block)
                : plain_tensor_layout_tag;
        VCONDCHECK_BG(
                format_tag::undef != bgmmc.wei_tag, VERBOSE_UNSUPPORTED_TAG)

        VCHECK_BG(memory_desc_init_by_tag(B_md, bgmmc.wei_tag),
                VERBOSE_UNSUPPORTED_TAG);
        const int dmax = nstl::min(bgmmc.ndims, 3);
        for (int d = 0; d < dmax; d++) {
            int dim = bgmmc.ndims - 1 - d;
            bgmmc.B_strides[d]
                    = bgmmc.b_dt_sz * B_d.blocking_desc().strides[dim];
        }
    } else {
        bgmmc.wei_tag = blocked_B_layouts_allowed && !bgmmc.is_runtime_N
                        && !bgmmc.is_int4_weights
                ? memory_desc_matches_one_of_tag(B_md, plain_tensor_layout_tag,
                        transposed_tensor_layout_tag, blocked_64n_B_layout_tag,
                        blocked_48n_B_layout_tag, blocked_32n_B_layout_tag,
                        blocked_16n_B_layout_tag)
                : memory_desc_matches_one_of_tag(B_md, plain_tensor_layout_tag,
                        transposed_tensor_layout_tag, acbd, adbc);

        if (bgmmc.wei_tag == format_tag::undef) {
            if (gemm_based::check_gemm_input_format(B_md)) {
                // Note: Here we batch layout may not be accurately represented
                // by the wei_tag string, due to all the permutations of the
                // batch. Only the gemm dimensions "n, k" are accurately
                // represented in the string representing transposed or not.
                bgmmc.wei_tag = helper.transB() == 'N'
                        ? plain_tensor_layout_tag
                        : transposed_tensor_layout_tag;
            }
        }
        VCONDCHECK_BG(
                format_tag::undef != bgmmc.wei_tag, VERBOSE_UNSUPPORTED_TAG)
    }
    return status::success;
}

status_t brgemm_matmul_conf_utils_t::update_and_check_B_tag(memory_desc_t &B_md,
        int n_blk_size, const matmul_helper_t &helper) const {
    if (n_blk_fixed && n_blk_size != bgmmc.wei_n_blk)
        return status::unimplemented;

    if (!(B_any_layout && blocked_B_layouts_allowed)) return status::success;

    return set_or_check_B_tag(B_md, helper, false);
}

status_t brgemm_matmul_conf_utils_t::set_or_check_tags(memory_desc_t &A_md,
        memory_desc_t &C_md, memory_desc_t &bias_md,
        const matmul_helper_t &helper) const {
    if (A_any_layout) {
        const format_tag_t desired_A_tag = plain_tensor_layout_tag;
        VCHECK_BG(memory_desc_init_by_tag(A_md, desired_A_tag),
                VERBOSE_UNSUPPORTED_TAG);
        bgmmc.src_tag = desired_A_tag;
    } else {
        const bool xf16_avx2_vnni_2 = (this->is_bf16() || this->is_f16())
                && bgmmc.isa == avx2_vnni_2;
        const bool is_int8_avx512_core
                = this->is_int8() && is_superset(bgmmc.isa, avx512_core);
        const bool is_adbc_allowed
                = (this->is_bf16() || this->is_f32() || this->is_bf32()
                          || this->is_f16() || this->is_f32_f16()
                          || this->is_f32_bf16() || this->is_bf16_with_int_wei()
                          || this->is_f16_with_int_wei())
                && !xf16_avx2_vnni_2;
        bgmmc.src_tag = is_adbc_allowed
                ? memory_desc_matches_one_of_tag(A_md, plain_tensor_layout_tag,
                        transposed_tensor_layout_tag, acbd, adbc)
                : is_int8_avx512_core
                ? memory_desc_matches_one_of_tag(A_md, plain_tensor_layout_tag,
                        transposed_tensor_layout_tag, acbd)
                : memory_desc_matches_one_of_tag(
                        A_md, plain_tensor_layout_tag, acbd);
        if (bgmmc.src_tag == format_tag::undef
                || (memory_desc_matches_tag(A_md, transposed_tensor_layout_tag)
                        && memory_desc_matches_tag(
                                A_md, plain_tensor_layout_tag)
                        && IMPLICATION(
                                !is_adbc_allowed, is_int8_avx512_core))) {
            if (gemm_based::check_gemm_input_format(A_md)) {
                // Note: Here we batch layout may not be accurately represented
                // by the wei_tag string, due to all the permutations of the
                // batch. Only the gemm dimensions "m, k" are accurately
                // represented in the string representing transposed or not.
                bgmmc.src_tag = helper.transA() == 'N'
                        ? plain_tensor_layout_tag
                        : transposed_tensor_layout_tag;
            }
            if (!IMPLICATION(bgmmc.src_tag == transposed_tensor_layout_tag,
                        is_adbc_allowed || is_int8_avx512_core))
                bgmmc.src_tag = format_tag::undef;
        }
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
        if (bgmmc.dst_tag == format_tag::undef) {
            if (gemm_based::check_gemm_output_format(C_md)) {
                // Note: Here we batch layout may not be accurately represented
                // by the wei_tag string, due to all the permutations of the
                // batch. Only the gemm dimensions "m, n" are accurately
                // represented in the string.
                bgmmc.dst_tag = plain_tensor_layout_tag;
            }
        }
    }

    VCONDCHECK_BG(!one_of(format_tag::undef, bgmmc.src_tag, bgmmc.dst_tag),
            VERBOSE_UNSUPPORTED_TAG)

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
    if (this->is_int8() || this->is_f8()) switch (n_blk) {
            case 64: return bgmmc.ndims == 3 ? aCB16b64c4b : BA16a64b4a;
            case 48: return bgmmc.ndims == 3 ? aCB16b48c4b : BA16a48b4a;
            case 32: return bgmmc.ndims == 3 ? aCB16b32c4b : BA16a32b4a;
            case 16: return bgmmc.ndims == 3 ? aCB16b16c4b : BA16a16b4a;
            default: return format_tag::undef;
        }

    if (this->is_bf16() || this->is_bf16_with_int_wei()
            || ((this->is_f16() || this->is_f32_f16() || this->is_f32_bf16()
                        || this->is_f16_with_int_wei())
                    && (is_superset(bgmmc.isa, avx512_core_amx)
                            || is_superset(bgmmc.isa, avx2_vnni_2))))
        switch (n_blk) {
            case 64: return bgmmc.ndims == 3 ? aCB16b64c2b : BA16a64b2a;
            case 48: return bgmmc.ndims == 3 ? aCB16b48c2b : BA16a48b2a;
            case 32: return bgmmc.ndims == 3 ? aCB16b32c2b : BA16a32b2a;
            case 16: return bgmmc.ndims == 3 ? aCB16b16c2b : BA16a16b2a;
            default: return format_tag::undef;
        }
    // Note: bf32 assumes f32 blocking
    if (this->is_f32() || this->is_bf32() || this->is_f16()
            || this->is_f32_f16() || this->is_f32_bf16()
            || this->is_f16_with_int_wei())
        switch (n_blk) {
            case 64: return bgmmc.ndims == 3 ? aCB16b64c : BA16a64b;
            case 48: return bgmmc.ndims == 3 ? aCB16b48c : BA16a48b;
            case 32: return bgmmc.ndims == 3 ? aCB16b32c : BA16a32b;
            case 24: return bgmmc.ndims == 3 ? aCB8b24c : BA8a24b;
            case 16:
                return bgmmc.wei_k_blk == 8
                        ? (bgmmc.ndims == 3 ? aCB8b16c : BA8a16b)
                        : (bgmmc.ndims == 3 ? aCB16b16c : BA16a16b);
            case 8: return bgmmc.ndims == 3 ? aCB8b8c : BA8a8b;
            default: return format_tag::undef;
        }
    return format_tag::undef;
}

brgemm_broadcast_t get_zp_type(const primitive_attr_t &attr, int arg) {
    return attr.zero_points_.has_default_values(arg)
            ? brgemm_broadcast_t::none
            : brgemm_broadcast_t::per_tensor;
}

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
        bgmmc.LDA = bgmmc.adjust_a_strides || bgmmc.use_buffer_a
                        || bgmmc.treat_A_as_plain
                ? get_actual_lda(bgmmc.use_buffer_a, bgmmc.tr_a_dt_sz)
                : bgmmc.A_strides[1] / bgmmc.a_dt_sz;
    }
};

float compute_blocking_heuristic_avx512(brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils,
        const matmul_avx512_blocking_params_t::matmul_params_t &matmul,
        matmul_avx512_blocking_params_t &best_blocking) {
    const int nthr = bgmmc.nthr;

    const int max_m_blk = nstl::min(256, matmul.M);
    int min_m_blk = nstl::min(32, matmul.M);

    dim_t min_m_chunks = div_up(matmul.M, max_m_blk);

    int n_blk = bgmmc.N_blk;
    const int n_chunks = div_up(matmul.N, n_blk);
    const int max_n_chunks = bgmmc.use_buffer_a ? 16 : 1;
    const int n_chunks_start = nstl::min(max_n_chunks, n_chunks);

    // Note: do not extend K_blk for 'bwd_w' cases
    const bool use_extended_k_blk = matmul.K > 1024
            && (!bm_conf_utils.check_is_transposed(bgmmc.src_tag));
    int default_k_blk = use_extended_k_blk ? 1024 : 512;
    int k_blk = nstl::min(matmul.K, default_k_blk);
    int start_nthr_k = 1;
    int last_nthr_k = 1;

    // for cases with low parallel work, reduce 'min_m_blk' to
    // increase potential parallelization balance.
    const dim_t max_parallel = static_cast<dim_t>(matmul.batch) * n_chunks;
    const dim_t max_bmn_parallel = max_parallel * min_m_chunks;
    const bool low_parallel_work = nthr > max_parallel;
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
                && !bm_conf_utils.is_int8()
                && IMPLICATION(bm_conf_utils.is_bf16(), math::is_pow2(matmul.K))
                && matmul.K >= 2048;
        if (bwd_w_par_k_blk) {
            start_nthr_k = nstl::min(nthr, 4);
            assert(k_blk == nstl::min(matmul.K, 512));
        }

        // Enable k-partitioning for huge k and small m/n dimensions.
        bool is_huge_k = matmul.K >= 20000;
        bool is_small_mn = matmul.M <= 512 && matmul.N <= 512;
        bool use_k_partitioning = is_huge_k && is_small_mn;

        // TODO: expand to other data types.
        use_k_partitioning = use_k_partitioning && bm_conf_utils.is_f32();

        if (use_k_partitioning) {
            auto least_prime_factor = [](int n) {
                assert(n > 0);
                if (n == 1) return 1;
                for (int factor = 2; factor < n; factor++)
                    if (n % factor == 0) return factor;
                return n;
            };

            int nthr_bmn = max_div(max_bmn_parallel, nthr);
            int nthr_k = nstl::max(nthr / nthr_bmn, 1);
            int nthr_remainder = nthr % nthr_bmn;

            // Choose number of threads in k-dim to allow a larger block size
            // for m-dim.
            while (nthr_k <= nthr_remainder) {
                nthr_bmn /= least_prime_factor(nthr_bmn);
                nthr_k = nstl::max(nthr / nthr_bmn, 1);
                nthr_remainder = nthr % nthr_bmn;
            }

            // Reduce number of threads in k-dim to balanced work.
            dim_t k_chunks = div_up(matmul.K, k_blk);
            while (k_chunks <= 5 * nthr_k && nthr_k > 1)
                nthr_k /= least_prime_factor(nthr_k);

            // Fix number of threads for k-dim.
            start_nthr_k = nthr_k;
            last_nthr_k = nthr_k;
        }
    }

    // Use large m-blocking if possible.
    const bool is_huge_n = matmul.N >= 20000;
    const bool large_bmn_parallelism = max_bmn_parallel > 10 * nthr;
    const bool has_m_tail = matmul.M % max_m_blk != 0;
    const bool use_k_partitioning = start_nthr_k > 1;
    bool use_large_m_blk = is_huge_n && large_bmn_parallelism && !has_m_tail;
    use_large_m_blk &= !use_k_partitioning;

    // TODO: Expand to other data types.
    use_large_m_blk = use_large_m_blk && bm_conf_utils.is_f32();

    if (use_large_m_blk) min_m_blk = max_m_blk;

    matmul_avx512_blocking_params_t cur_params(matmul, nthr);
    float best_imbalance = 1.f; // reduce
    for (int nthr_k = start_nthr_k; nthr_k >= last_nthr_k; --nthr_k) {
        bool found_best_blocking = false;
        for_(int n_chunk_size = n_chunks_start; n_chunk_size >= 1;
                --n_chunk_size)
        for (int m_blk = max_m_blk; m_blk >= min_m_blk; --m_blk) {
            cur_params.update_params(
                    1, m_blk, n_chunk_size, n_blk, 1, k_blk, nthr_k);

            float cur_imbalance = cur_params.get_imbalance();

            const int m_chunk_size = 1;
            int m_chunks = div_up(bgmmc.M, m_blk * m_chunk_size);
            int n_chunks = div_up(bgmmc.N, n_blk * n_chunk_size);
            int work_amount = bgmmc.batch * m_chunks * n_chunks;

            int nthr_bmn = nthr / nthr_k;
            bool skip_config = work_amount < nthr_bmn * 3
                    && work_amount % nthr_bmn != 0 && start_nthr_k == 1;
            if (skip_config) continue;

            if (cur_imbalance < best_imbalance) {
                best_imbalance = cur_imbalance;
                best_blocking = cur_params;
                found_best_blocking = true;
            }
        }

        if (!found_best_blocking) {
            cur_params.update_params(1, min_m_blk, 1, n_blk, 1, k_blk, nthr_k);

            float cur_imbalance = cur_params.get_imbalance();
            if (cur_imbalance < best_imbalance) {
                best_imbalance = cur_imbalance;
                best_blocking = cur_params;
            }
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

float compute_blocking_heuristic_avx2_f32(brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils,
        const matmul_avx512_blocking_params_t::matmul_params_t &matmul,
        matmul_avx512_blocking_params_t &best_blocking) {
    float best_imbalance = 1.f; // reduce

    const int nthr = bgmmc.nthr;

    dim_t max_m_blk = nstl::min(256, matmul.M);
    dim_t min_m_blk = max_m_blk;

    int n_blk = bgmmc.N_blk;
    const int n_chunks = div_up(matmul.N, n_blk);
    const int max_n_chunks = bgmmc.use_buffer_a ? 16 : 1;
    const int n_chunks_start = nstl::min(max_n_chunks, n_chunks);

    int default_k_blk = 1024;
    int k_blk = nstl::min(matmul.K, default_k_blk);
    int start_nthr_k = 1;

    // for cases with low parallel work, reduce 'min_m_blk' to
    // increase potential parallelization balance.
    size_t max_parallel = matmul.batch * n_chunks;
    const float req_additional_parallel = nthr / max_parallel;
    if (req_additional_parallel > 1) {
        min_m_blk = saturate<int>(
                16, max_m_blk, matmul.M / req_additional_parallel);
        max_parallel *= div_up(matmul.M, min_m_blk);
    } else if (bm_conf_utils.check_is_transposed(bgmmc.src_tag)
            && matmul.K >= 4096) {
        min_m_blk = nstl::max(16, matmul.M / 4);
        default_k_blk = 192;
    }

    bool low_parallel_work = max_parallel % nthr != 0
            && (static_cast<float>(max_parallel) / nthr) < 2;
    if (low_parallel_work) {
        // Reduce n_blk size to increase parallel space
        // note: over reduction of n_blk size on 2d shapes when n_chunks == 1
        // showed significant performance degradation
        if (!bm_conf_utils.check_n_blk_fixed()
                && IMPLICATION(n_chunks == 1, bgmmc.batch_ndims > 0)) {
            n_blk = nstl::min(matmul.N, 16);
        }
    }

    max_m_blk = nstl::max(max_m_blk, min_m_blk);
    for_(int nthr_k = start_nthr_k; nthr_k >= 1; --nthr_k)
    for_(int n_chunk_size = n_chunks_start; n_chunk_size >= 1; --n_chunk_size)
    for (int m_blk = min_m_blk; m_blk <= max_m_blk; m_blk += 4) {
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
    bgmmc.N_blk = bgmmc.wei_n_blk;
    if (!bgmmc.is_runtime_N) bgmmc.N_blk = nstl::min(bgmmc.N_blk, bgmmc.N);

    bgmmc.M_chunk_size = bgmmc.N_chunk_size = bgmmc.K_chunk_size = 1;

    bool prefer_copy_a
            = one_of(true, bm_conf_utils.is_f32() && bgmmc.isa == avx2,
                      bm_conf_utils.is_bf16(),
                      bm_conf_utils.is_bf16_with_int_wei(),
                      (bgmmc.is_amx
                              && (bm_conf_utils.is_f16()
                                      || bm_conf_utils.is_f16_with_int_wei())))
            && (bgmmc.isa != avx2_vnni_2) // no perf study yet.
            && bgmmc.lda_big_pow2() && bgmmc.M >= 1024;

    // Avoid copying A for small N gives better performance.
    // TODO: Expand for other precisions and cases.

    if (bgmmc.is_amx && bm_conf_utils.is_int8())
        prefer_copy_a &= bgmmc.N >= 256;

    if (bgmmc.is_amx) {
        if (matmul_amx_blocking_params_macro_t::is_supported(
                    bgmmc, bm_conf_utils)) {
            //grid heuristic is possible best blocking is set
            matmul_amx_blocking_params_macro_t best_blocking(bgmmc);
            matmul_amx_blocking_params_macro_t::find_best_blocking(
                    bgmmc, bm_conf_utils, best_blocking);

            if (best_blocking.get_blocking_scores() != 0.0f) {
                best_blocking.update_configuration(bgmmc);
                return status::success;
            }
        }
        bgmmc.use_buffer_a |= prefer_copy_a;

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

        matmul_amx_blocking_params_micro_t best_blocking(bgmmc);

        matmul_amx_blocking_params_micro_t::find_best_blocking(
                bgmmc, bm_conf_utils, best_blocking);

        VCONDCHECK_BG(best_blocking.get_blocking_scores() != 0.0f,
                VERBOSE_BLOCKING_FAIL, "");

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
        bgmmc.use_buffer_a |= prefer_copy_a;
        const matmul_avx512_blocking_params_t::matmul_params_t matmul(
                bgmmc.M, bgmmc.N, bgmmc.K, bgmmc.batch);

        matmul_avx512_blocking_params_t best_blocking(matmul, bgmmc.nthr);

        const float best_imbalance = compute_blocking_heuristic_avx512(
                bgmmc, bm_conf_utils, matmul, best_blocking);

        VCONDCHECK_BG(best_imbalance != 1.f, VERBOSE_BLOCKING_FAIL, "")

        best_blocking.update_configuration(bgmmc);
    } else {
        bgmmc.use_buffer_a |= prefer_copy_a;
        VCONDCHECK_BG(is_superset(bm_conf_utils.get_isa(), avx2),
                VERBOSE_UNSUPPORTED_ISA)
        const bool is_f32 = bm_conf_utils.is_f32() && bgmmc.isa == avx2;

        const matmul_avx512_blocking_params_t::matmul_params_t matmul(
                bgmmc.M, bgmmc.N, bgmmc.K, bgmmc.batch);

        matmul_avx512_blocking_params_t best_blocking(matmul, bgmmc.nthr);

        const float best_imbalance = is_f32
                ? compute_blocking_heuristic_avx2_f32(
                        bgmmc, bm_conf_utils, matmul, best_blocking)
                : compute_blocking_heuristic_avx2(
                        bgmmc, bm_conf_utils, matmul, best_blocking);

        VCONDCHECK_BG(best_imbalance != 1.f, VERBOSE_BLOCKING_FAIL, "")

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
    bgmmc.orig_src_dt = src_d.data_type();
    bgmmc.dst_dt = dst_d.data_type();
    bgmmc.wei_dt = weights_d.data_type();
    bgmmc.orig_wei_dt = weights_d.data_type();

    bgmmc.with_reduce = mmd.reduce_desc.format_kind != format_kind::undef;
    bgmmc.reduce_dt
            = bgmmc.with_reduce ? mmd.reduce_desc.data_type : data_type::undef;
    bgmmc.reduce_kind = mmd.reduce_kind;

    bgmmc.with_bias = mmd.bias_desc.format_kind != format_kind::undef;
    bgmmc.bia_dt = bgmmc.with_bias ? mmd.bias_desc.data_type : data_type::undef;
    bgmmc.s8s8_compensation_required = bgmmc.src_dt == s8 && !isa_has_s8s8(isa);
    bgmmc.ndims = dst_d.ndims();

    const bool is_wei_any = weights_d.format_kind() == format_kind::any
            || weights_d.is_sparse_packed_desc();
    brgemm_matmul_conf_utils_t bm_conf_utils(bgmmc, isa, attr,
            src_d.format_kind() == format_kind::any, is_wei_any,
            dst_d.format_kind() == format_kind::any,
            bias_md.format_kind == format_kind::any);

    VCHECK_BG(check_datatype_cfg(bm_conf_utils), VERBOSE_UNSUPPORTED_DT_CFG);
    VCHECK_BG(check_isa_with_datatype(isa, bm_conf_utils),
            VERBOSE_ISA_DT_MISMATCH);

    bgmmc.is_amx = is_superset(isa, avx512_core_amx);
    bgmmc.a_dt_sz = bgmmc.tr_a_dt_sz = types::data_type_size(bgmmc.src_dt);
    bgmmc.b_dt_sz = bgmmc.tr_b_dt_sz = types::data_type_size(bgmmc.wei_dt);

    bgmmc.packed_sparse_weights = weights_d.is_sparse_packed_desc();
    if (bgmmc.packed_sparse_weights) {
        VCONDCHECK_BG(bgmmc.is_amx, VERBOSE_ISA_SPARSE_ENCODING_MISMATCH);
        VCONDCHECK_BG(bgmmc.wei_dt == s8, VERBOSE_UNSUPPORTED_DT);
    }
    bgmmc.is_bf32 = bm_conf_utils.is_bf32();
    bgmmc.is_bf16_with_int_wei = bm_conf_utils.is_bf16_with_int_wei();
    bgmmc.is_f16_with_int_wei = bm_conf_utils.is_f16_with_int_wei();
    bgmmc.is_f32_f16 = bm_conf_utils.is_f32_f16();
    bgmmc.is_f32_bf16 = bm_conf_utils.is_f32_bf16();
    bgmmc.with_wei_decompression = bm_conf_utils.with_weights_decompression();
    bgmmc.is_int4_weights = one_of(bgmmc.wei_dt, data_type::s4, data_type::u4);

    // Make BRGeMM compute MatMul as if it were in bfloat16, while down-convert
    // happens during copy-buffer computations
    if (bgmmc.is_bf32 || bgmmc.is_bf16_with_int_wei) {
        bgmmc.src_dt = bf16;
        bgmmc.wei_dt = bf16;
        bgmmc.tr_a_dt_sz = types::data_type_size(bf16);
        bgmmc.tr_b_dt_sz = types::data_type_size(bf16);
    } else if ((bm_conf_utils.is_f16() || bgmmc.is_f16_with_int_wei)
            && bgmmc.isa == avx512_core_fp16) {
        // Similar to bf32, convert input data before compute
        bgmmc.src_dt = f32;
        bgmmc.wei_dt = f32;
        bgmmc.tr_a_dt_sz = types::data_type_size(f32);
        bgmmc.tr_b_dt_sz = types::data_type_size(f32);
    } else if ((bm_conf_utils.is_f32_f16() || bm_conf_utils.is_f32_bf16())
            && is_superset(bgmmc.isa, avx2)) {
        // Note 1: Keep this branch separately from f16 one to have different
        // ISA conditions (f16 includes f16:f32 and f16:f16 combinations). Same
        // applies for bf16 (which includes bf16:bf16).
        // Note 2: If `use_buffer_b()` is false, let the kernel perform the
        // conversion. Otherwise, make the copy_b routine handle the conversion
        // and set kernel data types to f32.
        // Note 3: Since `use_buffer_b()` depends on `bgmmc.wei_tag`, which is
        // set later in the code due to its dependencies, the update of data
        // types to f32 happens below in ANCHOR: `CONVERT_F32_XF16_DATA_TYPES`.
    } else if (bgmmc.is_f16_with_int_wei && bgmmc.isa != avx512_core_fp16) {
        bgmmc.src_dt = f16;
        bgmmc.wei_dt = f16;
        bgmmc.tr_a_dt_sz = types::data_type_size(f16);
        bgmmc.tr_b_dt_sz = types::data_type_size(f16);
    }

    bgmmc.acc_dt = bm_conf_utils.is_int8() ? s32 : f32;

    bgmmc.c_dt_sz = types::data_type_size(bgmmc.dst_dt);
    bgmmc.acc_dt_sz = types::data_type_size(bgmmc.acc_dt);
    if (bgmmc.with_bias) bgmmc.bias_dt_sz = types::data_type_size(bgmmc.bia_dt);
    if (bgmmc.with_reduce)
        bgmmc.reduce_dt_sz = types::data_type_size(bgmmc.reduce_dt);

    const auto &src_scales = attr.scales_.get(DNNL_ARG_SRC);
    const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
    const bool has_wei_scales = !wei_scales.has_default_values();
    bgmmc.with_scales = !src_scales.has_default_values() || has_wei_scales;
    if (has_wei_scales) {
        const auto wei_qmask_N = 1 << (bgmmc.ndims - 1);
        const auto wei_qmask_K = 1 << (bgmmc.ndims - 2);
        bgmmc.is_oscale_per_k = wei_scales.get_mask() & wei_qmask_K;
        bgmmc.is_oscale_per_n = wei_scales.get_mask() & wei_qmask_N;
        bgmmc.apply_scales_in_buffer_b = bgmmc.is_oscale_per_k
                && bgmmc.with_wei_decompression && bgmmc.N * bgmmc.K != 1;

        // only common and per-oc-channel scales are supported
        // only per-ic-channel scales is supprted with weight decompression
        VCONDCHECK_BG(wei_scales.get_mask() == 0 || bgmmc.is_oscale_per_n
                        || IMPLICATION(bgmmc.is_oscale_per_k,
                                bgmmc.with_wei_decompression),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    const auto &dst_scales = attr.scales_.get(DNNL_ARG_DST);
    bgmmc.with_dst_scales = !dst_scales.has_default_values();
    // only common scales are supported
    VCONDCHECK_BG(!(bgmmc.with_dst_scales && dst_scales.get_mask() > 0),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    const auto &p = attr.post_ops_;
    bgmmc.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    bgmmc.with_eltwise = eltwise_ind != -1;
    const int binary_ind = p.find(primitive_kind::binary);
    const int prelu_ind = p.find(primitive_kind::prelu);
    bgmmc.with_binary = !everyone_is(-1, binary_ind, prelu_ind);

    bgmmc.src_zp_type = get_zp_type(attr, DNNL_ARG_SRC);
    bgmmc.wei_zp_type = get_zp_type(attr, DNNL_ARG_WEIGHTS);
    bgmmc.dst_zp_type = get_zp_type(attr, DNNL_ARG_DST);

    VCONDCHECK_BG(
            IMPLICATION(!(bm_conf_utils.is_int8()
                                || bm_conf_utils.with_weights_decompression()),
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

    VCHECK_BG(bm_conf_utils.set_or_check_tags(src_md, dst_md, bias_md, helper),
            VERBOSE_UNSUPPORTED_TAG);
    VCHECK_BG(attr.set_default_formats(&dst_md), VERBOSE_UNSUPPORTED_TAG);
    VCONDCHECK_BG(post_ops_ok(bgmmc, attr, dst_d), VERBOSE_UNSUPPORTED_POSTOP);

    // runtime values for M/N dimensions are only supported
    VCONDCHECK_BG((!(is_runtime_value(bgmmc.batch) || bgmmc.is_runtime_K)),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED)
    // Single runtime dimension is only supported for now
    VCONDCHECK_BG(!(bgmmc.is_runtime_M && bgmmc.is_runtime_N),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED)
    // Runtime value for M dimension is supported for 2d AMX int8/bfloat16
    // problems only.
    const bool runtime_M_supported = bgmmc.is_amx && bgmmc.ndims == 2
            && one_of(true, bm_conf_utils.is_int8(), bm_conf_utils.is_bf16());
    VCONDCHECK_BG(!(bgmmc.is_runtime_M && !runtime_M_supported),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED)

    // Runtime N value is supported for 2d AMX int8/bfloat16 problems only.
    const bool runtime_N_supported = bgmmc.is_amx && bgmmc.ndims == 2
            && one_of(true, bm_conf_utils.is_int8(), bm_conf_utils.is_bf16());
    VCONDCHECK_BG(!(bgmmc.is_runtime_N && !runtime_N_supported),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED)

    bgmmc.batch_without_first_dim
            = bgmmc.batch_ndims > 1 ? helper.batch() / dst_d.dims()[0] : 0;

    bgmmc.bcast_A_desc.set_params(
            src_d.dims(), dst_d.dims(), bgmmc.batch_ndims, bgmmc.batch);
    bgmmc.bcast_B_desc.set_params(
            weights_d.dims(), dst_d.dims(), bgmmc.batch_ndims, bgmmc.batch);

    // required granularity for k dimension
    bgmmc.required_k_granularity
            = bgmmc.is_amx ? data_type_vnni_granularity(bgmmc.wei_dt) : 1;

    VCONDCHECK_BG(bgmmc.required_k_granularity > 0, VERBOSE_BLOCKING_FAIL, "");

    bgmmc.wei_k_blk = data_type_vnni_simd_elems(bgmmc.wei_dt, bgmmc.isa);

    VCHECK_BG(bm_conf_utils.set_or_check_B_tag(weights_md, helper),
            VERBOSE_UNSUPPORTED_TAG);

    bgmmc.req_wei_vnni_downconvert = bm_conf_utils.wei_down_convert_to_vnni();

    VCHECK_BG(attr.set_default_formats(&dst_md), VERBOSE_UNSUPPORTED_TAG);

    bgmmc.wei_n_blk = bm_conf_utils.get_default_n_block(bgmmc.wei_tag);

    bgmmc.blocked_B = bm_conf_utils.get_blocked_B();
    bgmmc.transposed_B = bm_conf_utils.check_is_transposed(bgmmc.wei_tag)
            || bgmmc.wei_tag == adbc;
    bgmmc.use_buffer_b = bm_conf_utils.use_buffer_b();
    bgmmc.req_transpose_scales = bgmmc.apply_scales_in_buffer_b
            && bgmmc.is_oscale_per_k && bgmmc.is_oscale_per_n
            && bgmmc.transposed_B;

    if ((bm_conf_utils.is_f32_f16() || bm_conf_utils.is_f32_bf16())
            && is_superset(bgmmc.isa, avx2) && bm_conf_utils.use_buffer_b()) {
        // ANCHOR: `CONVERT_F32_XF16_DATA_TYPES`
        bgmmc.src_dt = f32;
        bgmmc.wei_dt = f32;
        bgmmc.tr_a_dt_sz = types::data_type_size(f32);
        bgmmc.tr_b_dt_sz = types::data_type_size(f32);
    }

    // int4 weights decompression only supports plain and transpose layouts
    // TODO: enable int4 reorder and extend support to blocked weights
    // layout when needed
    if (bgmmc.with_wei_decompression && bgmmc.is_int4_weights)
        VCONDCHECK_BG(bm_conf_utils.check_is_plain(bgmmc.wei_tag)
                        || bm_conf_utils.check_is_transposed(bgmmc.wei_tag),
                VERBOSE_UNSUPPORTED_TAG);

    const bool transposed_A = bm_conf_utils.check_is_transposed(bgmmc.src_tag);
    // When M == 1 MatMul always considers A to be non-transposed even if A md
    // was created using "ba" tag.
    bgmmc.treat_A_as_plain = bgmmc.M == 1;
    bgmmc.transposed_A = ((transposed_A && !bgmmc.treat_A_as_plain)
            || bgmmc.src_tag == adbc);
    // For batched problems with plain A and C and fully broadcasted across B
    // we can merge all the batch dimensions into M if broadcast strategies
    // set is limited for binary post-ops
    const bool plain_A_layout = bm_conf_utils.check_is_plain(bgmmc.src_tag)
            || bgmmc.treat_A_as_plain;
    const bool merge_batch_dims_into_M = bgmmc.batch > 1
            && bgmmc.bcast_B_desc.bcast_across_all_batch_dims && plain_A_layout
            && helper.is_src_dst_layout_batch_fusable()
            && post_ops_ok(
                    bgmmc, attr, dst_d, true /* limit_bcast_strategies_set */);
    if (merge_batch_dims_into_M) {
        bgmmc.M *= bgmmc.batch;
        bgmmc.batch = 1;
    }

    // runtime A stride wrt M dimension is not acceptable
    VCONDCHECK_BG(!is_runtime_value(helper.get_a_stride(bgmmc.ndims - 2)),
            VERBOSE_UNSUPPORTED_MEM_STRIDE);

    // runtime A stride wrt K dimension is acceptable for transpose A and
    // runtime M case only
    const bool stride_A_wrt_K_dim_ok = IMPLICATION(
            is_runtime_value(helper.get_a_stride(bgmmc.ndims - 1)),
            bgmmc.transposed_A && bgmmc.is_runtime_M);
    VCONDCHECK_BG(stride_A_wrt_K_dim_ok, VERBOSE_UNSUPPORTED_MEM_STRIDE);

    // runtime A strides wrt batch dimensions are acceptable for runtime M case
    // only
    for (int b = 0; b < bgmmc.batch_ndims; b++) {
        VCONDCHECK_BG(IMPLICATION(is_runtime_value(helper.get_a_stride(b)),
                              bgmmc.is_runtime_M),
                VERBOSE_UNSUPPORTED_MEM_STRIDE);
    }

    // runtime B stride wrt N dimension is not acceptable
    VCONDCHECK_BG(!is_runtime_value(helper.get_b_stride(bgmmc.ndims - 1)),
            VERBOSE_UNSUPPORTED_MEM_STRIDE);

    // runtime B stride wrt K dimension is acceptable for non-transposed B and
    // runtime N case only
    const bool stride_B_wrt_K_dim_ok = IMPLICATION(
            is_runtime_value(helper.get_b_stride(bgmmc.ndims - 2)),
            !bgmmc.transposed_B && bgmmc.is_runtime_N);
    VCONDCHECK_BG(stride_B_wrt_K_dim_ok, VERBOSE_UNSUPPORTED_MEM_STRIDE);

    // runtime B strides wrt batch dimensions are acceptable for runtime N case
    // only
    for (int b = 0; b < bgmmc.batch_ndims; b++) {
        VCONDCHECK_BG(IMPLICATION(is_runtime_value(helper.get_b_stride(b)),
                              bgmmc.is_runtime_N),
                VERBOSE_UNSUPPORTED_MEM_STRIDE);
    }

    // runtime C stride wrt N dimension is not acceptable
    VCONDCHECK_BG(!is_runtime_value(helper.get_c_stride(bgmmc.ndims - 1)),
            VERBOSE_UNSUPPORTED_MEM_STRIDE);

    // runtime C stride wrt M dimension is acceptable for runtime N case only
    const bool stride_C_wrt_K_dim_ok = IMPLICATION(
            is_runtime_value(helper.get_c_stride(bgmmc.ndims - 2)),
            bgmmc.is_runtime_N);
    VCONDCHECK_BG(stride_C_wrt_K_dim_ok, VERBOSE_UNSUPPORTED_MEM_STRIDE);

    // runtime C strides wrt batch dimensions are acceptable for runtime N case
    // only
    for (int b = 0; b < bgmmc.batch_ndims; b++) {
        VCONDCHECK_BG(IMPLICATION(is_runtime_value(helper.get_c_stride(b)),
                              bgmmc.is_runtime_N),
                VERBOSE_UNSUPPORTED_MEM_STRIDE);
    }

    const bool is_copy_a_required = (bgmmc.is_amx && bm_conf_utils.is_bf32())
            || ((bm_conf_utils.is_f16() || bm_conf_utils.is_f16_with_int_wei())
                    && isa == avx512_core_fp16)
            || (bgmmc.wei_zp_type != brgemm_broadcast_t::none
                    && !bm_conf_utils.with_weights_decompression())
            || bgmmc.transposed_A;

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
    bgmmc.adjust_a_strides = merge_batch_dims_into_M
            && (src_d.matches_tag(acbd) || bgmmc.treat_A_as_plain);
    if (bgmmc.adjust_a_strides) bgmmc.A_strides[1] = bgmmc.A_strides[2];

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
            VERBOSE_BLOCKING_FAIL, "");

    if (bgmmc.wei_n_blk > bgmmc.N_blk
            && IMPLICATION(
                    bgmmc.N == bgmmc.N_blk, bgmmc.N >= bgmmc.wei_n_blk)) {
        assert(!bgmmc.is_runtime_N
                && "N_blk should not be adjusted for runtime N");
        bgmmc.wei_n_blk = bgmmc.N_blk;
        VCHECK_BG(bm_conf_utils.update_and_check_B_tag(
                          weights_md, bgmmc.wei_n_blk, helper),
                VERBOSE_UNSUPPORTED_TAG);

        bgmmc.req_wei_vnni_downconvert
                = bm_conf_utils.wei_down_convert_to_vnni();
    }

    VCHECK_BG(bm_conf_utils.set_B_flags(weights_md), VERBOSE_BLOCKING_FAIL, "");

    bgmmc.M_tail = bgmmc.is_runtime_M ? 0 : bgmmc.M % bgmmc.M_blk;
    bgmmc.N_tail = bgmmc.is_runtime_N ? 0 : bgmmc.N % bgmmc.N_blk;
    bgmmc.K_tail = bgmmc.K > bgmmc.K_blk
            ? (bgmmc.extendable_k ? bgmmc.K % bgmmc.K_blk
                                  : rnd_up(bgmmc.K % bgmmc.K_blk,
                                          bgmmc.required_k_granularity))
            : 0;

    bgmmc.LDB = bm_conf_utils.get_actual_LDB();
    bgmmc.LDD = dst_d.ndims() == 2 && dst_d.count_non_unit_dims(1)
            ? bgmmc.N
            : dst_d.blocking_desc().strides[bgmmc.ndims - 2];
    bgmmc.LDC = bgmmc.use_buffer_c && bgmmc.nthr_k <= 1
            ? (bgmmc.is_amx ? nstl::min((dim_t)32, bgmmc.N_blk) : bgmmc.N_blk)
                    * (bgmmc.is_runtime_N ? bgmmc.N_chunk_size : 1)
            : bgmmc.LDD;

    bgmmc.is_src_batch_layout_trivial
            = is_batch_layout_trivial(src_d, bgmmc.batch);
    bgmmc.is_wei_batch_layout_trivial
            = is_batch_layout_trivial(weights_d, bgmmc.batch);
    bgmmc.is_dst_batch_layout_trivial
            = is_batch_layout_trivial(dst_d, bgmmc.batch);
    init_aux_values(bgmmc, src_d, weights_d, dst_d);

    bgmmc.use_buffer_reduce
            = (bgmmc.reduce_dt != data_type::f32) || (bgmmc.nthr_k > 1);

    const dim_t max_a_stride = bgmmc.M_blk
            * (bgmmc.use_buffer_a ? bgmmc.copy_A_src_stride
                                  : bgmmc.LDA * bgmmc.a_dt_sz);
    const dim_t max_b_stride = bgmmc.K_blk
            * (bgmmc.use_buffer_b ? bgmmc.copy_B_wei_stride
                                  : bgmmc.LDB * bgmmc.b_dt_sz);
    const dim_t max_c_stride = bgmmc.M_blk * bgmmc.LDC * bgmmc.c_dt_sz;
    const dim_t max_d_stride = bgmmc.M_blk * bgmmc.LDD * bgmmc.acc_dt_sz;

    const dim_t max_supported_stride = std::numeric_limits<int32_t>::max();

    VCONDCHECK_BG(max_a_stride <= max_supported_stride,
            VERBOSE_UNSUPPORTED_FEATURE,
            "src stride > INT32_MAX is not supported");
    VCONDCHECK_BG(max_b_stride <= max_supported_stride,
            VERBOSE_UNSUPPORTED_FEATURE,
            "weights stride > INT32_MAX is not supported");
    VCONDCHECK_BG(std::max(max_c_stride, max_d_stride) <= max_supported_stride,
            VERBOSE_UNSUPPORTED_FEATURE,
            "dst stride > INT32_MAX is not supported");

    // When is_wei_batch_layout_trivial is true, we only support that
    // batch offset can be divided by 2
    if (bgmmc.is_int4_weights) {
        VCONDCHECK_BG(IMPLICATION(bgmmc.is_wei_batch_layout_trivial
                                      && bgmmc.batch > 1,
                              bgmmc.B_strides[2] % 2 == 0),
                VERBOSE_BAD_DIM);
    }

    // Dispatch small shapes to VNNI for better performance
    const bool runtime_dims
            = bgmmc.is_runtime_M || bgmmc.is_runtime_N || bgmmc.is_runtime_K;

    bool is_small_shapes = bgmmc.is_amx && !runtime_dims;

    // Disable 'small_shape' heuristic for amx_fp16 until it is validated with
    // performance measurements.
    is_small_shapes = is_small_shapes && (bgmmc.isa != avx512_core_amx_fp16);

    if (bm_conf_utils.is_bf16() || bm_conf_utils.is_f16()
            || bm_conf_utils.is_f32_f16() || bm_conf_utils.is_f32_bf16()
            || bm_conf_utils.is_bf16_with_int_wei()
            || bm_conf_utils.is_f16_with_int_wei()) {
        // empirical observation for performance breakpoint between amx and vnni
        // bf16/f16
        const dim_t buffer_a_chunk_sz_limit = 126;
        is_small_shapes = is_small_shapes
                && bgmmc.buffer_a_gb_stride <= buffer_a_chunk_sz_limit;
    } else {
        is_small_shapes = is_small_shapes && bgmmc.ndims < 3
                && ((bgmmc.M == 1 && bgmmc.K == 256)
                        || (bgmmc.M <= 32 && bgmmc.M * bgmmc.N <= 256)
                        || bgmmc.K <= 16);
    }
    // This is the only implementation that support the packed_sparse_weights
    // case therefore there is no fallback for it.
    is_small_shapes = is_small_shapes && !bgmmc.packed_sparse_weights;
    VCONDCHECK_BG(!is_small_shapes, VERBOSE_SMALL_SHAPES);

    return status::success;
}

status_t init_conf(brgemm_matmul_conf_t &conf, dim_t batch, dim_t M, dim_t K,
        dim_t N, dim_t in_ld, dim_t n_blk, data_type_t in_type,
        data_type_t out_type, format_tag_t in_tag) {
    if (n_blk <= 0 && M <= 0) return status::invalid_arguments;

    const auto vnni_granularity = data_type_vnni_granularity(out_type);
    if (vnni_granularity <= 0) return status::invalid_arguments;

    // Zero initialize the `conf` to avoid access to 'garbage' in members.
    conf = brgemm_matmul_conf_t();

    const bool is_bf16_with_int_wei = out_type == data_type::bf16
            && utils::one_of(in_type, data_type::s8, data_type::u8,
                    data_type::s4, data_type::u4);
    const bool with_wei_decompression = in_type != out_type
            && utils::one_of(in_type, data_type::s8, data_type::u8,
                    data_type::s4, data_type::u4);

    const bool is_copyB = N > 0;
    conf.isa = get_max_cpu_isa(); // Just use the best ISA possible.
    conf.is_bf32 = false;
    conf.batch = batch;
    conf.src_dt = conf.wei_dt = out_type;
    conf.orig_src_dt = conf.orig_wei_dt = in_type;
    // Note: will need to change `tr_a_dt_sz` for copyA in cases where src_dt != dst_dt
    conf.a_dt_sz = conf.tr_a_dt_sz = types::data_type_size(conf.src_dt);
    conf.N = N;
    conf.M = M;
    conf.K = K;
    const dim_t copyA_K_blk = isa_num_vregs(conf.isa) / 2;
    const dim_t copyB_K_blk = 16 * vnni_granularity;
    conf.K_blk = is_copyB ? copyB_K_blk : copyA_K_blk;
    conf.K_tail = conf.K % conf.K_blk;
    if (!is_copyB) {
        // Note: current implementation always calls the transposed kernel.
        conf.transposed_A = true;
        conf.M_blk = (dim_t)isa_max_vlen(conf.isa) / conf.a_dt_sz;
        conf.M_tail = conf.M % conf.M_blk;
        conf.copy_A_src_stride = in_ld * conf.a_dt_sz;
        // setting LDA parameter required for plain transpose
        conf.LDA = conf.K;

        // jit_brgemm_matmul_copy_a_tranposed_impl_t::dst_stride
        dim_t dst_stride = conf.LDA * conf.tr_a_dt_sz;

        dim_t max_src_encode_stride = conf.K_blk * conf.copy_A_src_stride;
        dim_t max_dst_encode_stride = conf.M_blk * dst_stride;

        // Cannot encode EVEX compressed addresses
        VCONDCHECK_BG(std::max(max_src_encode_stride, max_dst_encode_stride)
                        <= std::numeric_limits<int32_t>::max(),
                VERBOSE_UNSUPPORTED_MEM_STRIDE);

    } else {
        conf.blocked_B = !utils::one_of(in_tag, ab, ba, abc, acb);
        conf.transposed_B = utils::one_of(in_tag, ba, acb);
        conf.is_bf16_with_int_wei = is_bf16_with_int_wei;
        conf.with_wei_decompression = with_wei_decompression;
        conf.wei_tag = in_tag;
        conf.wei_n_blk = conf.N_blk = conf.LDB = n_blk;
        conf.N_tail = conf.N % conf.N_blk;
        conf.b_dt_sz = types::data_type_size(in_type);
        conf.tr_b_dt_sz = types::data_type_size(conf.wei_dt);
        conf.copy_B_wei_stride = in_ld * conf.b_dt_sz;
        conf.N_chunk_elems = conf.N; // To match seems unneeded assert.
        conf.s8s8_comp_b_str = utils::rnd_up(conf.N, conf.wei_n_blk);
        conf.s8s8_comp_n_str = conf.wei_n_blk;

        dim_t max_wei_encode_off = conf.K_blk * conf.copy_B_wei_stride
                + conf.wei_n_blk * conf.b_dt_sz;
        dim_t max_dst_encode_off
                = (conf.K_blk * conf.LDB + conf.wei_n_blk) * conf.tr_b_dt_sz;

        // Cannot encode EVEX compressed addresses
        VCONDCHECK_BG(std::max(max_wei_encode_off, max_dst_encode_off)
                        <= std::numeric_limits<int32_t>::max(),
                VERBOSE_UNSUPPORTED_MEM_STRIDE);
    }

    // The following members are different from the upper level `init_conf()`
    // call from the reorder implementation due to lacking a memory descriptor
    // to tip on compensation.
    // TODO: re-consider an interface change to enable these members.
    conf.s8s8_compensation_required = false;
    conf.src_zp_type = brgemm_broadcast_t::none;
    conf.has_zero_point_a = false;
    conf.has_zero_point_b = false;

    return status::success;
}

void init_aux_values(brgemm_matmul_conf_t &bgmmc,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &wei_d,
        const memory_desc_wrapper &dst_d) {
    bgmmc.M_chunk_elems = bgmmc.M_blk * bgmmc.M_chunk_size;
    bgmmc.N_chunk_elems = bgmmc.N_blk * bgmmc.N_chunk_size;
    bgmmc.K_chunk_elems
            = bgmmc.K_blk * bgmmc.K_chunk_size * bgmmc.brgemm_batch_size;
    bgmmc.M_chunks = div_up(bgmmc.M, bgmmc.M_chunk_elems);
    bgmmc.N_chunks = div_up(bgmmc.N, bgmmc.N_chunk_elems);
    bgmmc.K_chunks = div_up(bgmmc.K, bgmmc.K_chunk_elems);
    bgmmc.num_M_blocks = div_up(bgmmc.M, bgmmc.M_blk);
    bgmmc.num_N_blocks = div_up(bgmmc.N, bgmmc.N_blk);
    bgmmc.num_K_blocks = div_up(bgmmc.K, bgmmc.K_blk * bgmmc.brgemm_batch_size);

    const int last_chunck_batch_size
            = (nstl::max(bgmmc.K, bgmmc.K_blk)
                      - (bgmmc.K_chunks - 1) * bgmmc.K_chunk_elems)
            / bgmmc.K_blk;

    bgmmc.brgemm_batch_tail_size
            = last_chunck_batch_size % bgmmc.brgemm_batch_size;

    if (!bgmmc.is_runtime_N && bgmmc.is_amx && bgmmc.nthr_k == 1) {
        bgmmc.buffer_c_chunk_sz = rnd_up(bgmmc.N_blk, bgmmc.LDC) * bgmmc.M_blk
                * bgmmc.acc_dt_sz;
    } else {
        bgmmc.buffer_c_chunk_sz = bgmmc.acc_dt_sz
                * (bgmmc.is_runtime_N ? bgmmc.N_blk : bgmmc.LDC)
                * (bgmmc.nthr_k > 1 ? bgmmc.M : bgmmc.M_blk);
    }

    if (bgmmc.nthr_k > 1) {
        // c size == M * N (for reduction)
        bgmmc.buffer_c_per_thread_sz = bgmmc.buffer_c_chunk_sz;

    } else if (!bgmmc.is_runtime_N && !bgmmc.is_runtime_M
            && bgmmc.K_chunk_elems >= bgmmc.K) {
        // c size == BRGEMM size
        bgmmc.buffer_c_per_thread_sz = bgmmc.buffer_c_chunk_sz;

    } else {
        // c size == chunk size
        bgmmc.buffer_c_per_thread_sz = bgmmc.buffer_c_chunk_sz
                * bgmmc.M_chunk_size * bgmmc.N_chunk_size;
    }

    bgmmc.buffer_a_gb_stride = bgmmc.tr_a_dt_sz * bgmmc.M_blk
            * (bgmmc.use_buffer_a_tail_only ? bgmmc.wei_k_blk : bgmmc.LDA);

    bgmmc.buffer_a_k_stride
            = bgmmc.buffer_a_gb_stride * bgmmc.brgemm_batch_size;

    bgmmc.buffer_a_m_stride = bgmmc.buffer_a_k_stride * bgmmc.K_chunk_size;

    bgmmc.buffer_a_per_thread_sz = bgmmc.buffer_a_m_stride * bgmmc.M_chunk_size;

    bgmmc.buffer_b_gb_stride = bgmmc.tr_b_dt_sz * bgmmc.LDB * bgmmc.K_blk;
    bgmmc.buffer_b_k_brg_stride
            = bgmmc.buffer_b_gb_stride * bgmmc.brgemm_batch_size;

    bgmmc.buffer_b_n_blk_stride = bgmmc.tr_b_dt_sz
            * ((bgmmc.N_blk / bgmmc.LDB) * bgmmc.LDB2
                    + (bgmmc.N_blk % bgmmc.LDB)
                            * data_type_vnni_granularity(bgmmc.wei_dt));

    bgmmc.buffer_b_chunk_sz = bgmmc.tr_b_dt_sz * rnd_up(bgmmc.N_blk, bgmmc.LDB)
            * rnd_up(bgmmc.K_chunk_elems, bgmmc.wei_k_blk);

    bgmmc.buffer_b_per_thread_sz
            = bgmmc.buffer_b_chunk_sz * bgmmc.brgemm_batch_size;

    bgmmc.buffer_reduce_per_thread_sz = 0;
    if (bgmmc.reduce_kind == matmul_reduce_kind::src) {
        assert(bgmmc.acc_dt == f32);
        bgmmc.buffer_reduce_per_thread_sz = bgmmc.M * bgmmc.acc_dt_sz;
    }

    bgmmc.s8s8_comp_ithr_str
            = bgmmc.use_buffer_b ? bgmmc.wei_n_blk * bgmmc.N_chunk_size : 0;
    bgmmc.s8s8_comp_b_str = bgmmc.use_buffer_b
            ? 0
            : div_up(bgmmc.N, bgmmc.wei_n_blk) * bgmmc.wei_n_blk;
    bgmmc.s8s8_comp_n_str = bgmmc.wei_n_blk;

    bgmmc.A_ptr_shift_b = 0;
    bgmmc.copy_A_src_stride = bgmmc.a_dt_sz
            * src_d.strides()[bgmmc.ndims - 2 + bgmmc.transposed_A];

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
            const dim_t copy_A_src_stride = src_d.matches_tag(dabc)
                            && bgmmc.K * bgmmc.batch
                                    == src_d.blocking_desc().strides[0]
                    ? src_d.blocking_desc().strides[0]
                    : src_d.blocking_desc().strides[0] * bgmmc.K;
            bgmmc.copy_A_src_stride
                    = nstl::min(copy_A_src_stride, src_stride / factor)
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
    // - Plain md that matches transposed tag does not dispatch into its codepath
    if (wei_d.matches_one_of_tag(acbd, adbc) != format_tag::undef) {
        const dim_t bcast_shift_b = wei_d.matches_tag(acbd) ? bgmmc.N : bgmmc.K;
        bgmmc.B_ptr_shift_b
                = (bgmmc.bcast_B_desc.bcast_mask == 2
                                  ? bcast_shift_b
                                  : wei_d.blocking_desc().strides[0])
                * bgmmc.b_dt_sz;
    }
    if (wei_d.matches_one_of_tag(acbd, adbc) != format_tag::undef
            && wei_d.matches_one_of_tag(abcd, abdc) == format_tag::undef) {
        const dim_t factor = bgmmc.wei_dt == f32 ? 2 : 1;
        const dim_t wei_stride = wei_d.matches_tag(acbd) ? bgmmc.B_strides[1]
                                                         : bgmmc.B_strides[0];
        bgmmc.copy_B_wei_stride = nstl::min(wei_d.blocking_desc().strides[0],
                                          wei_stride / factor)
                * factor;
    } else if (bgmmc.transposed_B) {
        bgmmc.copy_B_wei_stride
                = (wei_d.strides()[bgmmc.ndims - 1] * bgmmc.b_dt_sz);
    } else if (bgmmc.is_runtime_N) {
        bgmmc.copy_B_wei_stride = bgmmc.N;
    } else if (bgmmc.blocked_B) {
        bgmmc.copy_B_wei_stride = (bgmmc.LDB * bgmmc.b_dt_sz);
    } else {
        bgmmc.copy_B_wei_stride
                = (wei_d.strides()[bgmmc.ndims - 2] * bgmmc.b_dt_sz);
    }

    bgmmc.C_ptr_shift_b = dst_d.matches_one_of_tag(acbd)
            ? dst_d.blocking_desc().strides[0] * bgmmc.c_dt_sz
            : 0;

    bgmmc.has_zero_point_a = bgmmc.src_zp_type != brgemm_broadcast_t::none;
    bgmmc.has_zero_point_b = bgmmc.wei_zp_type != brgemm_broadcast_t::none;
    bgmmc.has_zero_point_c = bgmmc.dst_zp_type != brgemm_broadcast_t::none;
    bgmmc.post_ops_applicable = one_of(true, bgmmc.with_sum, bgmmc.with_bias,
            bgmmc.with_scales && !bgmmc.apply_scales_in_buffer_b,
            bgmmc.with_eltwise, bgmmc.with_binary, bgmmc.acc_dt != bgmmc.dst_dt,
            bgmmc.s8s8_compensation_required, bgmmc.has_zero_point_a,
            bgmmc.has_zero_point_b, bgmmc.has_zero_point_c,
            bgmmc.with_dst_scales);

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

    if (bgmmc.use_buffer_reduce) {
        const bool is_reduce_f32 = bgmmc.reduce_dt == f32;
        scratchpad.book(key_brgemm_primitive_buffer_reduce,
                (bgmmc.nthr_k - is_reduce_f32)
                        * bgmmc.buffer_reduce_per_thread_sz,
                default_data_align);
    }

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
    if (bgmmc.is_runtime_M || bgmmc.is_runtime_N)
        scratchpad.book(key_brgemm_primitive_buffer_d,
                bgmmc.M_blk * bgmmc.N_blk * bgmmc.c_dt_sz * bgmmc.nthr,
                default_data_align);
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

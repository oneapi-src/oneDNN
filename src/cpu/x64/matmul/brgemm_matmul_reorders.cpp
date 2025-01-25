/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "common/dnnl_thread.hpp"

#include "common/reorder_pd.hpp"
#include "cpu/x64/matmul/brgemm_matmul_reorders.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace format_tag;

format_tag_t get_blocked_otag(const memory_desc_t &dst_md) {

    const memory_desc_wrapper od(dst_md);
    const auto vnni_granularity = data_type_vnni_granularity(od.data_type());

    format_tag_t otag = format_tag::undef;
    switch (vnni_granularity) {
        case 4:
            otag = od.matches_one_of_tag(aCB16b64c4b, BA16a64b4a, aCB16b48c4b,
                    BA16a48b4a, aCB16b32c4b, BA16a32b4a, aCB16b16c4b,
                    BA16a16b4a);
            break;
        case 2:
            otag = od.matches_one_of_tag(aCB16b64c2b, BA16a64b2a, aCB16b48c2b,
                    BA16a48b2a, aCB16b32c2b, BA16a32b2a, aCB16b16c2b,
                    BA16a16b2a);
            break;
        case 1:
            otag = od.matches_one_of_tag(aCB16b64c, BA16a64b, aCB16b48c,
                    BA16a48b, aCB16b32c, BA16a32b, aCB16b16c, BA16a16b);
            break;
        default: otag = format_tag::undef;
    }
    return otag;
}

// If two shapes are plain transposes of each other, then
// src.strides / dst.strides (or dst.strides / src.strides) will look like:
// index:   0, 1, 2, ..., l-1, l,   l+1, ..., l+m-1, l+m, l+m+1, ..., l+m+n
// src/dst: 1, 1, 1, ..., 1,   M,   M,   ..., M,     1/K, 1/K,   ..., 1/K
// dst/src: 1, 1, 1, ..., 1,   1/M, 1/M, ..., 1/M,   K,   K,     ..., K
//
// where the first k dimensions are batch dimensions, M is stride of
// first dimesions squashed, K is stride of the second dimension squashed.
//
// therefore M = M, K = K, batch = product of first k dimensions.
//
// Note: Assuming src strides are sorted and dst and dims are sorted
// in the same order. This done first thing in the
// function.
status_t calculate_plain_transpose_blocks(dim_t &batch, dim_t &M, dim_t &K,
        const memory_desc_t &src_md, const memory_desc_t &dst_md) {

    // drop all unit dims as they they will break the calculations. Removing
    // unit dims will not change the physical memory reorder problem.
    dims_t non_unit_dims {};
    dim_t non_unit_dim = 0;
    for (dim_t i = 0; i < src_md.ndims; i++) {
        if (src_md.dims[i] == 1) continue;
        non_unit_dims[non_unit_dim++] = src_md.dims[i];
    }

    memory_desc_t src_md_reduced, dst_md_reduced;
    memory_desc_reshape(src_md_reduced, src_md, non_unit_dim, non_unit_dims);
    memory_desc_reshape(dst_md_reduced, dst_md, non_unit_dim, non_unit_dims);

    const memory_desc_wrapper id(src_md_reduced), od(dst_md_reduced);

    dims_t sort_src_indices {};
    dims_t sort_dst_indices {};
    for (dim_t i = 0; i < id.ndims(); i++) {
        sort_src_indices[i] = i;
        sort_dst_indices[i] = i;
    }
    std::sort(sort_src_indices, sort_src_indices + id.ndims(),
            [id](int a, int b) { return id.strides()[a] > id.strides()[b]; });
    std::sort(sort_dst_indices, sort_dst_indices + od.ndims(),
            [od](int a, int b) { return od.strides()[a] > od.strides()[b]; });
    // make sure physical layout is dense and there is no magical
    // padding.
    for (dim_t i = id.ndims() - 1; i > 0; i--)
        VDISPATCH_REORDER_IC((id.strides()[sort_src_indices[i]]
                                             * id.dims()[sort_src_indices[i]]
                                     == id.strides()[sort_src_indices[i - 1]])
                        && (od.strides()[sort_dst_indices[i]]
                                        * od.dims()[sort_dst_indices[i]]
                                == od.strides()[sort_dst_indices[i - 1]]),
                VERBOSE_UNSUPPORTED_MEM_STRIDE);
    // sort the arrays by src strides.
    dims_t src_sorted_strides {};
    dims_t dst_sorted_strides {};
    dims_t sorted_dims {};
    for (dim_t i = 0; i < id.ndims(); i++) {
        src_sorted_strides[i] = id.strides()[sort_src_indices[i]];
        dst_sorted_strides[i] = od.strides()[sort_src_indices[i]];
        sorted_dims[i] = id.dims()[sort_src_indices[i]];
    }

    // find first unmatching stride
    dim_t l_idx = -1;
    for (dim_t s_idx = 0; s_idx < id.ndims(); s_idx++) {
        if (src_sorted_strides[s_idx] != dst_sorted_strides[s_idx]) {
            l_idx = s_idx;
            break;
        }
    }
    // if all strides are the same, then this is a direct copy
    VDISPATCH_REORDER_IC(l_idx != -1, VERBOSE_UNSUPPORTED_MEM_STRIDE);
    // batch is product of first k dimensions.
    batch = 1;
    for (dim_t d_idx = 0; d_idx < l_idx; d_idx++)
        batch *= sorted_dims[d_idx];
    // create the src/dst and dst/src arrays.
    dims_t src_over_dst {}, dst_over_src {};
    for (dim_t s_idx = l_idx; s_idx < id.ndims(); s_idx++) {
        src_over_dst[s_idx]
                = src_sorted_strides[s_idx] / dst_sorted_strides[s_idx];
        dst_over_src[s_idx]
                = dst_sorted_strides[s_idx] / src_sorted_strides[s_idx];
    }

    // check if src and dst are transposes of each other

    // find first umatching stride by division
    dim_t lm_idx = -1;
    int prev_M_src = src_over_dst[l_idx],
        prev_1_over_M_dst = dst_over_src[l_idx];
    // here we are checking to make sure all indexes are the same in src/dst
    // and dst/src arrays (comparing with prev_h_src and prev_h_dst). If its
    // different, then both src/dst and dst/src arrays should be different. In
    // which case we have found the l+m index (lm_idx). If one is different
    // while the other is not then this is not a plain transpose (return
    // status::unimplemented).
    //                             ^^^^^^^^^^^^^^^^^^^^
    // index:   0, 1, 2, ..., l-1, l,   l+1, ..., l+m-1, l+m, l+m+1, ..., l+m+n
    // src/dst: 1, 1, 1, ..., 1,   M,   M,   ..., M,     1/K, 1/K,   ..., 1/K
    // dst/src: 1, 1, 1, ..., 1,   1/M, 1/M, ..., 1/M,   K,   K,     ..., K
    for (dim_t s_idx = l_idx + 1; s_idx < id.ndims(); s_idx++) {
        const bool cond_ok
                = (src_over_dst[s_idx] == prev_M_src
                          && dst_over_src[s_idx] == prev_1_over_M_dst)
                || (src_over_dst[s_idx] != prev_M_src
                        && dst_over_src[s_idx] != prev_1_over_M_dst);
        VDISPATCH_REORDER_IC(cond_ok, VERBOSE_UNSUPPORTED_MEM_STRIDE);
        if (src_over_dst[s_idx] != prev_M_src) {
            lm_idx = s_idx;
            break;
        }
    }
    // This means all strides are the same. In this case its a direct copy.
    // Only case this might be possible is unit strides after batch.
    VDISPATCH_REORDER_IC(lm_idx != -1, VERBOSE_UNSUPPORTED_MEM_STRIDE);

    int prev_1_over_K_src = src_over_dst[lm_idx],
        prev_K_dst = dst_over_src[lm_idx];
    // Here we make sure the last strides divs are the same. If not, then this
    // means that one of the l+m, l+m+1, ..., l+m+n indexes is not in the same
    // order as in src. For example in src, looking at memory view, it can be
    // 1, 2, 3, 4, 5, 6, 7, 8 while in dst it is 1, 2, 3, 6, 8, 7, 4, 5 (1,2 are
    // batch dims). While the global ordering is the transposed (4,5 come after
    // 6, 7, 9 indicating possible transpose) the local ordering is different
    // (8 and 7 swapped meaning 6,7,8 squashed is not the same as 6,8,7
    // squashed).
    //                                                   ^^^^^^^^^^^^^^^^^^^^^^
    // index:   0, 1, 2, ..., l-1, l,   l+1, ..., l+m-1, l+m, l+m+1, ..., l+m+n
    // src/dst: 1, 1, 1, ..., 1,   M,   M,   ..., M,     1/K, 1/K,   ..., 1/K
    // dst/src: 1, 1, 1, ..., 1,   1/M, 1/M, ..., 1/M,   K,   K,     ..., K
    for (dim_t s_idx = lm_idx + 1; s_idx < id.ndims(); s_idx++) {
        const bool cond_ok = dst_over_src[s_idx] == prev_K_dst
                && src_over_dst[s_idx] == prev_1_over_K_src;
        VDISPATCH_REORDER_IC(cond_ok, VERBOSE_UNSUPPORTED_MEM_STRIDE);
    }

    // set block sizes with contiguous dimensions squashed.
    M = prev_M_src;
    K = prev_K_dst;

    return status::success;
}

// This function initializes all required fields in the conf object to generate
// copy_b kernel.
// This particular call relies on memory descriptors and used in this
// implementation. The sub-call is reduced to simplified objects and used in
// BRGeMM public API implementation for copy routines.
//
// Note: this version has some extra definitions that are available in memory
// descriptors only.
status_t init_conf(matmul::brgemm_matmul_conf_t &conf,
        const memory_desc_t &src_md, const memory_desc_t &dst_md) {
    const memory_desc_wrapper id(src_md), od(dst_md);
    const int ndims = id.ndims();
    const auto &dims = id.dims();
    const auto type_i = id.data_type();
    const auto type_o = od.data_type();

    const bool is_plain = id.is_plain() && od.is_plain();
    dim_t M, K, batch, in_ld, N;
    format_tag_t itag = format_tag::undef, otag = format_tag::undef;
    if (is_plain) {
        CHECK(calculate_plain_transpose_blocks(batch, M, K, src_md, dst_md));
        N = 0;
        in_ld = M;
        // The heuristic value is empirical
        const bool is_small_shape = batch * M * K < 49152;
        VDISPATCH_REORDER_IC(!is_small_shape, VERBOSE_SMALL_SHAPES);
    } else {
        batch = ndims > 2 ? dims[ndims - 3] : 1;
        M = 0;
        K = dims[ndims - 2];
        N = dims[ndims - 1];
        in_ld = ndims >= 2 ? id.strides()[ndims - 2] : 1;
        const bool int4_ok = IMPLICATION(
                utils::one_of(type_i, data_type::s4, data_type::u4),
                N % 2 == 0);
        VDISPATCH_REORDER_IC(int4_ok, VERBOSE_BAD_DIM, "N", ndims - 1);
        const bool is_bf16_with_int_wei = type_o == data_type::bf16
                && utils::one_of(type_i, data_type::s8, data_type::u8,
                        data_type::s4, data_type::u4);

        otag = get_blocked_otag(dst_md);
        // TODO: enable for itag = {ba, acb}
        itag = id.matches_one_of_tag(
                ab, abc, is_bf16_with_int_wei ? otag : format_tag::undef);
        VDISPATCH_REORDER_IC(!utils::one_of(format_tag::undef, itag, otag),
                VERBOSE_UNSUPPORTED_TAG);
    }

    CHECK(matmul::init_conf(conf, batch, M, K, N, in_ld,
            is_plain ? 0 : matmul::get_n_block_from_tag(otag), type_i, type_o,
            itag));

    conf.s8s8_compensation_required
            = od.extra().flags & memory_extra_flags::compensation_conv_s8s8;
    const bool req_asymmetric_comp = od.extra().flags
            & memory_extra_flags::compensation_conv_asymmetric_src;
    conf.src_zp_type = req_asymmetric_comp ? brgemm_broadcast_t::per_tensor
                                           : brgemm_broadcast_t::none;
    conf.has_zero_point_a = conf.src_zp_type != brgemm_broadcast_t::none;

    return status::success;
}

status_t brgemm_matmul_copy_reorder_t::pd_t::init(
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine) {
    using namespace status;

    CHECK(cpu_reorder_pd_t::init(engine, src_engine, dst_engine));

    const memory_desc_wrapper id(src_md_), od(dst_md_);
    const int ndims = id.ndims();

    const auto type_i = id.data_type();
    const auto type_o = od.data_type();

    // TODO: enable support for type_i != type_o cases
    const bool is_int_weights = utils::one_of(
            type_i, data_type::s8, data_type::u8, data_type::s4, data_type::u4);
    const bool is_plain = id.is_plain() && od.is_plain();
    const bool dt_ok
            = IMPLICATION(type_i == type_o,
                      utils::one_of(type_o, data_type::s8, data_type::bf16,
                              data_type::f16, data_type::f32))
            && IMPLICATION(type_i != type_o,
                    utils::one_of(type_o, data_type::f32, data_type::f16,
                            data_type::bf16)
                            && is_int_weights)
            // Plain transpose is supported only for f32. Consider
            // adding other datatypes for potential performance improvement.
            && IMPLICATION(is_plain,
                    type_i == data_type::f32 && type_o == data_type::f32);
    VDISPATCH_REORDER_IC(dt_ok, VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_REORDER_IC(
            id.is_dense(), VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    // plain transpose reorder works for all shapes.
    VDISPATCH_REORDER_IC(is_plain ? ndims >= 2 : utils::one_of(ndims, 2, 3),
            VERBOSE_BAD_NDIMS, "src", ndims);

    // plain transpose does not support postops
    VDISPATCH_REORDER_IC(IMPLICATION(is_plain, attr()->post_ops_.len() == 0),
            VERBOSE_UNSUPPORTED_POSTOP);

    const bool is_f16 = utils::one_of(data_type::f16, type_i, type_o);
    const bool is_s8s8 = type_i == data_type::s8 && type_o == data_type::s8;
    const bool is_bf16_with_int_wei
            = type_o == data_type::bf16 && is_int_weights;
    const bool isa_ok
            = IMPLICATION(is_bf16_with_int_wei, mayiuse(avx512_core_bf16))
            && IMPLICATION(is_f16, mayiuse(avx512_core_fp16))
            && IMPLICATION(!is_f16, mayiuse(avx512_core))
            && IMPLICATION(is_s8s8, mayiuse(avx512_core_vnni));
    VDISPATCH_REORDER_IC(isa_ok, VERBOSE_UNSUPPORTED_ISA);

    const bool has_adj_scale
            = od.extra().flags & memory_extra_flags::scale_adjust;
    VDISPATCH_REORDER_IC(
            !has_adj_scale, VERBOSE_UNSUPPORTED_MD_FLAG, "dst:scale_adjust");

    VDISPATCH_REORDER_IC(
            attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    VDISPATCH_REORDER_IC(
            od.is_blocking_desc(), VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    VDISPATCH_REORDER_IC(
            !od.has_runtime_dims_or_strides(), VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    VDISPATCH_REORDER_IC(!od.has_zero_dim(), VERBOSE_BAD_DIM, "dst", 0);

    CHECK(init_conf(matmul_conf_for_reorder_, src_md_, dst_md_));

    auto mask_ok = [&](bool check, int mask) {
        return IMPLICATION(
                check, mask == (1 << ndims) - 1 - (1 << (ndims - 2)));
    };

    const bool req_asymmetric_comp = od.extra().flags
            & memory_extra_flags::compensation_conv_asymmetric_src;
    const bool comp_masks_ok = true
            && mask_ok(matmul_conf_for_reorder_.s8s8_compensation_required,
                    od.extra().compensation_mask)
            && mask_ok(req_asymmetric_comp, od.extra().asymm_compensation_mask);
    if (!comp_masks_ok) return invalid_arguments;

    init_scratchpad();

    return status::success;
}

status_t brgemm_matmul_copy_reorder_t::pd_t::create(reorder_pd_t **reorder_pd,
        engine_t *engine, const primitive_attr_t *attr, engine_t *src_engine,
        const memory_desc_t *src_md, engine_t *dst_engine,
        const memory_desc_t *dst_md) {
    using namespace status;

    VDISPATCH_REORDER_IC(impl::is_dense_format_kind({src_md, dst_md}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    auto _pd = make_unique_pd<pd_t>(
            attr, src_engine->kind(), src_md, dst_engine->kind(), dst_md);
    if (_pd == nullptr) return out_of_memory;
    CHECK(_pd->init(engine, src_engine, dst_engine));
    CHECK(_pd->init_scratchpad_md());
    return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd.release());
}

status_t brgemm_matmul_copy_reorder_t::execute_body(
        const exec_ctx_t &ctx) const {
    using namespace utils;

    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const memory_desc_wrapper &src_d = pd()->src_md();
    const memory_desc_wrapper &dst_d = pd()->dst_md();
    const auto sdt_sz = types::data_type_size(src_d.data_type());
    const auto type_o = dst_d.data_type();
    const auto ddt_sz = types::data_type_size(type_o);
    const auto src_typesz_scale
            = utils::one_of(src_d.data_type(), data_type::s4, data_type::u4)
            ? 2
            : 1;

    const auto &kernel_conf = pd()->matmul_conf_for_reorder_;
    const size_t comp_offset_bytes
            = dst_d.size() - dst_d.additional_buffer_size();
    const size_t s8s8_comp_size_bytes = kernel_conf.s8s8_compensation_required
            ? dst_d.additional_buffer_size(
                    memory_extra_flags::compensation_conv_s8s8)
            : 0;
    const size_t zp_comp_offset_bytes
            = comp_offset_bytes + s8s8_comp_size_bytes;
    int32_t *cp = kernel_conf.s8s8_compensation_required
            ? reinterpret_cast<int32_t *>(dst + comp_offset_bytes)
            : nullptr;
    int32_t *zp = kernel_conf.has_zero_point_a
            ? reinterpret_cast<int32_t *>(dst + zp_comp_offset_bytes)
            : nullptr;

    const int ndims = src_d.ndims();
    if (kernel_conf.N <= 0) {
        parallel_nd(kernel_conf.batch,
                utils::div_up(kernel_conf.K, kernel_conf.K_blk),
                utils::div_up(kernel_conf.M, kernel_conf.M_blk),
                [&](const dim_t batch, const dim_t k_blk, const dim_t m_blk) {
                    auto ker_exec_ctx
                            = matmul::jit_brgemm_matmul_copy_a_t::ctx_t();
                    ker_exec_ctx.current_K_blk
                            = kernel_conf.K_blk * (k_blk + 1) > kernel_conf.K
                            ? kernel_conf.K % kernel_conf.K_blk
                            : kernel_conf.K_blk;
                    ker_exec_ctx.current_M_blk
                            = kernel_conf.M_blk * (m_blk + 1) > kernel_conf.M
                            ? kernel_conf.M % kernel_conf.M_blk
                            : kernel_conf.M_blk;
                    ker_exec_ctx.src = (void *)(src
                            + m_blk * kernel_conf.M_blk * kernel_conf.a_dt_sz
                            + k_blk * kernel_conf.K_blk * kernel_conf.M
                                    * kernel_conf.a_dt_sz
                            + batch * kernel_conf.K * kernel_conf.M
                                    * kernel_conf.a_dt_sz
                            + src_d.offset0() * kernel_conf.a_dt_sz);
                    ker_exec_ctx.tr_src = (void *)(dst
                            + m_blk * kernel_conf.M_blk * kernel_conf.K
                                    * kernel_conf.tr_a_dt_sz
                            + k_blk * kernel_conf.K_blk * kernel_conf.tr_a_dt_sz
                            + batch * kernel_conf.K * kernel_conf.M
                                    * kernel_conf.tr_a_dt_sz
                            + dst_d.offset0() * kernel_conf.tr_a_dt_sz);
                    (*a_kernel_)(&ker_exec_ctx);
                });

    } else {

#define get_blk_off(md, dt_sz, batch, d0, d1) \
    (ndims == 3 ? (dt_sz) * (md).blk_off((batch), (d0), (d1)) \
                : (dt_sz) * (md).blk_off((d0), (d1)))

        parallel_nd(kernel_conf.batch, div_up(kernel_conf.N, kernel_conf.N_blk),
                [&](dim_t batch, dim_t n_blk_idx) {
                    const auto n = n_blk_idx * kernel_conf.N_blk;
                    const bool is_N_tail
                            = (kernel_conf.N - n) < kernel_conf.N_blk;
                    auto ker_exec_ctx
                            = matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
                    ker_exec_ctx.current_N_blk = is_N_tail ? kernel_conf.N_tail
                                                           : kernel_conf.N_blk;

                    const auto comp_offset = batch * kernel_conf.s8s8_comp_b_str
                            + n_blk_idx * kernel_conf.s8s8_comp_n_str;

                    ker_exec_ctx.zp_a_compensation_ptr
                            = kernel_conf.has_zero_point_a
                            ? (void *)&zp[comp_offset]
                            : nullptr;
                    ker_exec_ctx.compensation_ptr
                            = kernel_conf.s8s8_compensation_required
                            ? (void *)&cp[comp_offset]
                            : nullptr;

                    // required to compute zp compensation
                    int tmp_neg_a_zp_val = -1;
                    ker_exec_ctx.zp_a_neg_value_ptr = &tmp_neg_a_zp_val;

                    int k_blk_idx = 0;
                    for (; k_blk_idx < kernel_conf.K / kernel_conf.K_blk;
                            k_blk_idx++) {
                        const auto k = k_blk_idx * kernel_conf.K_blk;
                        const auto src_offset = !kernel_conf.blocked_B
                                ? get_blk_off(src_d, sdt_sz, batch, k, n)
                                : get_blk_off(src_d, sdt_sz, batch, k_blk_idx,
                                        n_blk_idx);
                        ker_exec_ctx.src
                                = (void *)&src[src_offset / src_typesz_scale];
                        ker_exec_ctx.tr_src = (void *)&dst[get_blk_off(
                                dst_d, ddt_sz, batch, k_blk_idx, n_blk_idx)];
                        ker_exec_ctx.current_K_start = k;
                        ker_exec_ctx.current_K_iters = kernel_conf.K_blk;
                        (*b_kernel_)(&ker_exec_ctx);
                    }
                    if (kernel_conf.K_tail > 0) {
                        const auto k = k_blk_idx * kernel_conf.K_blk;
                        const auto src_offset = !kernel_conf.blocked_B
                                ? get_blk_off(src_d, sdt_sz, batch, k, n)
                                : get_blk_off(src_d, sdt_sz, batch, k_blk_idx,
                                        n_blk_idx);
                        ker_exec_ctx.src
                                = (void *)&src[src_offset / src_typesz_scale];
                        const auto dst_offset = get_blk_off(
                                dst_d, ddt_sz, batch, k_blk_idx, n_blk_idx);
                        ker_exec_ctx.tr_src = (void *)&dst[dst_offset];
                        ker_exec_ctx.current_K_start = k;
                        ker_exec_ctx.current_K_iters = kernel_conf.K_tail;
                        (*b_kernel_)(&ker_exec_ctx);
                        const auto vnni_granularity
                                = data_type_vnni_granularity(type_o);
                        const auto dst_zero_out_offset
                                = rnd_up(kernel_conf.K_tail, vnni_granularity)
                                * kernel_conf.N_blk * ddt_sz;
                        const auto elems_to_zero
                                = rnd_dn(kernel_conf.K_blk - kernel_conf.K_tail,
                                          vnni_granularity)
                                * kernel_conf.N_blk * ddt_sz;
                        array_set(&dst[dst_offset + dst_zero_out_offset], 0,
                                elems_to_zero);
                    }
                });

#undef get_blk_off
    }
    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

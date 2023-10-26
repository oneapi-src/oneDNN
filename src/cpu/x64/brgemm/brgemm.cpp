/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
#include "cpu/x64/brgemm/brgemm_utils.hpp"

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;
using namespace brgemm_utils;

void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const brgemm_batch_element_t *batch, void *ptr_C, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = batch;
    brgemm_p.ptr_A = nullptr;
    brgemm_p.ptr_B = nullptr;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = 0;
    brgemm_p.do_apply_comp = 0;
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = bs;

    assert(brg_kernel);

    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const void *addr_B,
        const brgemm_batch_element_t *batch, void *ptr_C, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = batch;
    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_C;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = nullptr;
    brgemm_p.do_post_ops = 0;
    brgemm_p.do_apply_comp = 0;
    brgemm_p.skip_accm = 0;
    brgemm_p.BS = bs;
    assert(brg_kernel);
    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, int bs,
        const brgemm_batch_element_t *batch, void *ptr_C, void *ptr_D,
        const brgemm_post_ops_data_t &post_ops_data, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = batch;
    brgemm_p.ptr_A = nullptr;
    brgemm_p.ptr_B = nullptr;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_D;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = post_ops_data.bias;
    brgemm_p.ptr_scales = post_ops_data.scales;
    brgemm_p.do_post_ops
            = post_ops_data.do_only_comp || post_ops_data.do_only_zp_a_val ? 0
                                                                           : 1;
    brgemm_p.do_apply_comp = post_ops_data.do_only_zp_a_val ? 0 : 1;
    brgemm_p.skip_accm = post_ops_data.skip_accumulation ? 1 : 0;
    brgemm_p.BS = bs;
    brgemm_p.zp_a_val = post_ops_data.zp_a_val;
    brgemm_p.post_ops_binary_rhs_arg_vec = post_ops_data.binary_post_ops_rhs;
    brgemm_p.oc_logical_off = post_ops_data.oc_logical_off;
    brgemm_p.dst_row_logical_off = post_ops_data.dst_row_logical_off;
    brgemm_p.data_C_ptr_ = post_ops_data.data_C_ptr_;
    brgemm_p.first_mb_matrix_addr_off = post_ops_data.first_mb_matrix_addr_off;
    brgemm_p.a_zp_compensations = post_ops_data.a_zp_compensations;
    brgemm_p.b_zp_compensations = post_ops_data.b_zp_compensations;
    brgemm_p.c_zp_values = post_ops_data.c_zp_values;
    brgemm_p.ptr_dst_scales = post_ops_data.dst_scales;
    assert(brg_kernel);
    (*brg_kernel)(&brgemm_p);
}

void brgemm_kernel_execute_postops(const brgemm_kernel_t *brg_kernel, int bs,
        const void *addr_A, const void *addr_B,
        const brgemm_batch_element_t *batch, void *ptr_C, void *ptr_D,
        const brgemm_post_ops_data_t &post_ops_data, void *scratch) {
    brgemm_kernel_params_t brgemm_p;

    brgemm_p.batch = batch;
    brgemm_p.ptr_A = addr_A;
    brgemm_p.ptr_B = addr_B;
    brgemm_p.ptr_C = ptr_C;
    brgemm_p.ptr_D = ptr_D;
    brgemm_p.ptr_buf = scratch;
    brgemm_p.ptr_bias = post_ops_data.bias;
    brgemm_p.ptr_scales = post_ops_data.scales;
    brgemm_p.do_post_ops
            = post_ops_data.do_only_comp || post_ops_data.do_only_zp_a_val ? 0
                                                                           : 1;
    brgemm_p.do_apply_comp = post_ops_data.do_only_zp_a_val ? 0 : 1;
    brgemm_p.skip_accm = post_ops_data.skip_accumulation ? 1 : 0;
    brgemm_p.BS = bs;
    brgemm_p.zp_a_val = post_ops_data.zp_a_val;
    brgemm_p.post_ops_binary_rhs_arg_vec = post_ops_data.binary_post_ops_rhs;
    brgemm_p.oc_logical_off = post_ops_data.oc_logical_off;
    brgemm_p.dst_row_logical_off = post_ops_data.dst_row_logical_off;
    brgemm_p.data_C_ptr_ = post_ops_data.data_C_ptr_;
    brgemm_p.first_mb_matrix_addr_off = post_ops_data.first_mb_matrix_addr_off;
    brgemm_p.a_zp_compensations = post_ops_data.a_zp_compensations;
    brgemm_p.b_zp_compensations = post_ops_data.b_zp_compensations;
    brgemm_p.c_zp_values = post_ops_data.c_zp_values;
    brgemm_p.ptr_dst_scales = post_ops_data.dst_scales;
    assert(brg_kernel);
    (*brg_kernel)(&brgemm_p);
}

status_t brgemm_desc_init(brgemm_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, bool transA, bool transB,
        brgemm_layout_t layout, float alpha, float beta, dim_t LDA, dim_t LDB,
        dim_t LDC, dim_t M, dim_t N, dim_t K, const brgemm_strides_t *strides) {
    /*
    m - number of rows of the matrix op(A) and number of rows of the matrix C
    n - number of columns of the matrix op(B) and number of columns of the matrix C
    k - number of columns of the matrix op(A) and number of rows of the matrix op(B)

    Matrices are in row-major layouts:
        A: lda * m, LDA - lda must be at least max(1, k)
        B: ldb * k, LDB - ldb must be at least max(1, n)
        C: ldc * m, LDC - ldc must be at least max(1, n)

    Matrices are in column-major layouts:
        A: lda * k, LDA - lda must be at least max(1, m)
        B: ldb * n, LDB - ldb must be at least max(1, k)
        C: ldc * n, LDC - ldc must be at least max(1, m)
    */
    if (brg == nullptr) return status::invalid_arguments;
    if (transA || transB) return status::unimplemented;

    brgemm_utils::init_brgemm_conf(brg, isa, type, dt_a, dt_b, layout, alpha,
            beta, LDA, LDB, LDC, M, N, K, strides);

    if (M <= 0 || N <= 0 || K <= 0) return status::invalid_arguments;

    if (utils::everyone_is(
                false, brg->is_int8, brg->is_bf16, brg->is_f32, brg->is_f16))
        return status::unimplemented;

    // Only avx512_core_amx kernel supports u8 weights.
    if (!IMPLICATION(brg->dt_b == u8, brg->isa_impl == avx512_core_amx))
        return status::unimplemented;

    CHECK(brgemm_blocking(brg));

    // avx2_vnni_2 kernel with xf16 data type requires blocked weights.
    if (brg->isa_impl == avx2_vnni_2 && brg->is_xf16()
            && brg->LDB % brg->ld_block > 0)
        return status::unimplemented;

    return status::success;
}

status_t brdgmm_desc_init(brgemm_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, bool transA, brgemm_layout_t layout,
        float alpha, float beta, dim_t LDA, dim_t LDC, dim_t M, dim_t N,
        const brgemm_strides_t *strides) {

    if (brg == nullptr) return status::invalid_arguments;
    if (transA || layout != brgemm_row_major || alpha != 1.0f || beta != 0.f)
        return status::unimplemented;

    brgemm_utils::init_brdgmm_conf(brg, isa, type, dt_a, dt_b, layout, alpha,
            beta, LDA, LDC, M, N, strides);

    const bool ldx_check = (LDA < N || LDC < N);
    if (ldx_check) return status::invalid_arguments;

    if (utils::everyone_is(
                false, brg->is_int8, brg->is_bf16, brg->is_f32, brg->is_f16))
        return status::unimplemented;

    CHECK(brdgmm_blocking(brg));

    return status::success;
}

status_t brgemm_desc_set_postops(brgemm_t *brg, const primitive_attr_t *attr,
        const memory_desc_t *dst_md, int LDD, impl::data_type_t dt_bias) {
    if (!brg || !dst_md) return status::invalid_arguments;

    brg->attr = attr;
    brg->dst_md = dst_md;

    brg->with_bias = (dt_bias == data_type::undef) ? false : true;
    brg->dt_bias = dt_bias;
    brg->typesize_bias = (dt_bias == data_type::undef)
            ? 0
            : types::data_type_size(brg->dt_bias);

    brg->LDD = LDD;
    const auto dt_d = dst_md->data_type;

    // check that bias and output data type are supported by isa
    if (!IMPLICATION(one_of(data_type::bf16, dt_bias, dt_d),
                is_superset(brg->isa_impl, avx512_core)
                        || is_superset(brg->isa_impl, avx2_vnni_2)))
        return status::unimplemented;
    if (!IMPLICATION(one_of(data_type::f16, dt_bias, dt_d),
                is_superset(brg->isa_impl, avx512_core_fp16)
                        || is_superset(brg->isa_impl, avx2_vnni_2)))
        return status::unimplemented;
    // check that combination of data types is allowed
    if ((brg->dt_a == data_type::u8 && brg->dt_b == data_type::s8)
            && (!one_of(dt_d, data_type::u8, data_type::s8, data_type::s32,
                    data_type::f32, data_type::bf16))
            && (!one_of(dt_bias, data_type::undef, data_type::u8, data_type::s8,
                    data_type::s32, data_type::f32, data_type::bf16)))
        return status::unimplemented;
    if ((brg->dt_a == data_type::bf16 && brg->dt_b == data_type::bf16)
            && (!one_of(dt_d, data_type::bf16, data_type::f32))
            && (!one_of(dt_bias, data_type::undef, data_type::bf16,
                    data_type::f32)))
        return status::unimplemented;
    if ((brg->dt_a == data_type::f32 && brg->dt_b == data_type::f32)
            && (!one_of(dt_d, data_type::f32))
            && (!one_of(dt_bias, data_type::undef, data_type::f32, dt_d)))
        return status::unimplemented;
    if (!IMPLICATION(brg->is_f16,
                one_of(dt_d, data_type::f32, data_type::f16)
                        && one_of(dt_bias, data_type::undef, data_type::f32,
                                data_type::f16)))
        return status::unimplemented;

    brg->dt_d = dt_d;
    brg->typesize_D = types::data_type_size(brg->dt_d);

    if (!IMPLICATION(brg->is_int8 && brg->dt_d == bf16,
                is_superset(brg->isa_impl, avx512_core)
                        || brg->isa_impl == avx2_vnni_2))
        return status::unimplemented;

    if (brg->is_int8 && brg->dt_d == bf16)
        brg->is_bf16_emu
                = !(mayiuse(avx512_core_bf16) || brg->isa_impl == avx2_vnni_2);

    // Rerun blocking heuristic due to reduced zmm register count
    if (brg->is_bf16_emu && brg->is_dgmm) CHECK(brdgmm_blocking(brg));

    if (!brg->attr) return status::success;

    using namespace injector;

    const auto &post_ops = brg->attr->post_ops_;
    const memory_desc_wrapper dst_d(dst_md);

    const auto binary_ind = post_ops.find(primitive_kind::binary);
    const auto prelu_ind = post_ops.find(primitive_kind::prelu);
    brg->with_binary = !everyone_is(-1, binary_ind, prelu_ind);

    // NOTE: Using brg->isa_impl here is a bit dangerous as it can change before
    //       kernel creation, so there is no gaurantee that the isa checked here
    //       matches the isa used at kernel creation time. For now this can only
    //       happen for bf32, where isa during this check is avx512_core and isa
    //       at kernel creation time is avx512_core_amx_bf16. It just so happens
    //       that the behavior of `post_ops_ok` is identical for those two isas,
    //       but there is no guarantee that will always be the case.
    if ((brg->with_binary && !dst_md)
            || !injector::post_ops_ok(
                    post_ops_ok_args_t(brg->isa_impl, {sum, eltwise, binary},
                            post_ops, &dst_d, false /*sum_at_pos_0_only*/,
                            false /*sum_requires_scale_one*/,
                            false /*sum_requires_zp_zero*/,
                            true /*sum_requires_same_params*/,
                            {broadcasting_strategy_t::per_oc,
                                    broadcasting_strategy_t::scalar,
                                    broadcasting_strategy_t::per_mb_spatial,
                                    broadcasting_strategy_t::per_mb_w,
                                    broadcasting_strategy_t::per_w,
                                    broadcasting_strategy_t::no_broadcast})))
        return status::unimplemented;

    const auto sum_idx = post_ops.find(primitive_kind::sum);
    const bool with_sum = sum_idx != -1;
    brg->with_sum = with_sum;
    brg->sum_scale = with_sum ? post_ops.entry_[sum_idx].sum.scale : 0;
    brg->sum_zp = with_sum ? post_ops.entry_[sum_idx].sum.zero_point : 0;
    const auto sum_dt
            = with_sum ? post_ops.entry_[sum_idx].sum.dt : data_type::undef;
    brg->sum_dt = sum_dt != data_type::undef ? sum_dt : dt_d;

    const auto eltwise_ind = post_ops.find(primitive_kind::eltwise);
    brg->with_eltwise = eltwise_ind != -1;

    const auto &src_scales = attr->scales_.get(DNNL_ARG_SRC);
    const auto &wei_scales = attr->scales_.get(DNNL_ARG_WEIGHTS);
    brg->with_scales = !src_scales.has_default_values()
            || !wei_scales.has_default_values()
            || brg->with_weights_scale_adjust;
    if (brg->with_scales) {
        // Note. the current version supports only two different output scale
        // types:
        //     1) common (mask_ = 0)
        //     2) per_n_dim_scale - broadcast across n dimension;
        //        for convolution and inner product promitives it corresponds
        //        to "per_oc" mask_ = 1 << 1; for matmul - to
        //        mask_ = (1 << (ndims - 1))), where ndims is number of
        //        dimensions for original matmul problem
        // So if wei_scales.mask_ != 0 (not common) it's assumed here that scale
        // type is per_n_dim_scale and driver which calls brgemm kernel checked
        // that mask has correct value for this case
        brg->is_oc_scale = wei_scales.mask_ != 0;
    }

    const auto &dst_scales = attr->scales_.get(DNNL_ARG_DST);
    brg->with_dst_scales = !dst_scales.has_default_values();
    const bool scales_ok = src_scales.mask_ == 0 && dst_scales.mask_ == 0
            && attr->scales_.has_default_values(
                    {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST});
    if (!scales_ok) return status::unimplemented;

    auto init_zp_type
            = [&](brgemm_broadcast_t &zp_type, int mem_arg) -> status_t {
        auto zero_points = attr->zero_points_;

        // common zero point type is supported for now
        if (!zero_points.common(mem_arg)) return status::unimplemented;

        zp_type = zero_points.has_default_values(mem_arg)
                ? brgemm_broadcast_t::none
                : brgemm_broadcast_t::per_tensor;
        return status::success;
    };

    init_zp_type(brg->zp_type_a, DNNL_ARG_SRC);
    init_zp_type(brg->zp_type_b, DNNL_ARG_WEIGHTS);
    init_zp_type(brg->zp_type_c, DNNL_ARG_DST);

    // src zero points require additional register in brgemm & brdgmm kernel
    const bool is_zp_src = brg->zp_type_a != brgemm_broadcast_t::none;
    if (brg->is_dgmm) {
        const bool sum_needs_vmm = (!is_superset(brg->isa_impl, avx512_core))
                && brg->with_sum && brg->sum_scale != 1.f;
        if (is_zp_src || sum_needs_vmm) CHECK(brdgmm_blocking(brg));
    } else if (is_zp_src || brg->is_bf16_emu)
        CHECK(brgemm_blocking(brg));

    return status::success;
}

status_t brgemm_desc_set_attr(brgemm_t *brg, const brgemm_attr_t &brgattr) {
    if (brg == nullptr) return status::invalid_arguments;

    // negative padding is not supported
    if (brgattr.max_top_vpad < 0 || brgattr.max_bottom_vpad < 0)
        return status::unimplemented;

    if (!brg->is_dgmm) {
        // virtual padding size is restricted by MAX_VPAD value
        if (brgattr.max_top_vpad > brgemm_t::MAX_VPAD
                || brgattr.max_bottom_vpad > brgemm_t::MAX_VPAD)
            return status::unimplemented;
    }

    // virtual padding is supported for "brgemm_row_major" layout
    // TODO: remove this restriction
    if ((brgattr.max_top_vpad > 0 || brgattr.max_bottom_vpad > 0)
            && brg->layout != brgemm_row_major)
        return status::unimplemented;

    brg->brgattr = brgattr;

    if (brgattr.fpmath_mode != fpmath_mode::strict) maybe_try_bf32(brg);

    const int max_vpad = nstl::max(brgattr.max_top_vpad,
            brgattr.max_bottom_vpad); // these should be equal
    bool hint_blocking_set
            = (brgattr.hint_bd_block != 0 || brgattr.hint_bd_block2 != 0
                    || brgattr.hint_ld_block != 0 || brgattr.hint_ld_block2 != 0
                    || brgattr.hint_load_nt_A != brgemm_hint_nt_undef
                    || brgattr.hint_load_nt_B != brgemm_hint_nt_undef
                    || brgattr.bs_group > 1);
    if (brgattr.use_uker || brg->is_bf16_tmm || hint_blocking_set
            || brgattr.bd_mask_level
            || brgattr.fpmath_mode != fpmath_mode::strict || max_vpad > 0) {
        if (brg->is_dgmm)
            CHECK(brdgmm_blocking(brg));
        else
            CHECK(brgemm_blocking(brg));
    }

    if (!brg->is_dgmm) {
        // virtual padding is restricted by bd_block size due to
        // brgemm_kernel implementation. TODO: remove this restriction
        const int min_bd_block
                = brg->bdb_tail > 0 ? brg->bdb_tail : brg->bd_block;
        if ((max_vpad > min_bd_block)) return status::unimplemented;
    }

    brg->LDA2 = (brgattr.LDA2 != 0) ? brgattr.LDA2 : brg->LDA;
    brg->LDB2 = (brgattr.LDB2 != 0) ? brgattr.LDB2 : brg->LDB;
    brg->LDC2_M = (brgattr.LDC2_M != 0) ? brgattr.LDC2_M : brg->LDC;
    brg->LDC2_N = (brgattr.LDC2_N != 0) ? brgattr.LDC2_N : brg->ld_block;

    brg->is_blocked = (brg->LDA2 != brg->LDA || brg->LDB2 != brg->LDB
            || brg->LDC2_M != brg->LDC || brg->LDC2_N != brg->ld_block);

    if (!IMPLICATION(brg->is_blocked, brg->layout = brgemm_row_major))
        return status::invalid_arguments;

    // virtual padding is not supported for "amx"
    if ((brgattr.max_top_vpad > 0 || brgattr.max_bottom_vpad > 0)
            && (brg->is_tmm))
        return status::unimplemented;

    brg->prfA = brgattr.hint_prfA;
    brg->prfB = brgattr.hint_prfB;
    brg->prfC = brgattr.hint_prfC;
    if (brgattr.hint_innermost_loop != brgemm_innermost_undef)
        brg->innermost_loop = brgattr.hint_innermost_loop;

    if (brgattr.hint_prefetching == brgemm_kernel_prefetching_t::brgemm_prf0
            && brg->prfC.dist0 < 0)
        brg->prfC.dist0 = 0;
    if (brgattr.hint_prefetching == brgemm_kernel_prefetching_t::brgemm_prf1
            && brg->prfC.dist1 < 0)
        brg->prfC.dist1 = 0;
    if (brgattr.hint_prefetching == brgemm_kernel_prefetching_t::brgemm_prf2
            && brg->prfC.dist2 < 0)
        brg->prfC.dist2 = 0;

    return status::success;
}

status_t brgemm_kernel_create(
        brgemm_kernel_t **brg_kernel, const brgemm_t &brg) {
    if (!brg_kernel) return status::invalid_arguments;
    *brg_kernel = nullptr;

    if (brg.is_dgmm) {
        if (brg.type == brgemm_static_offs) return status::unimplemented;
#define CASE(isa) \
    case isa: \
        CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel, \
                new brdgmm_kernel_t<isa, typename cpu_isa_traits<isa>::Vmm>( \
                        brg))); \
        break
        switch (brg.isa_impl) {
            CASE(avx512_core_fp16);
            CASE(avx512_core_bf16);
            CASE(avx512_core_vnni);
            CASE(avx512_core);
            CASE(avx2_vnni_2);
            CASE(avx2_vnni);
            CASE(avx2);
            default: return status::unimplemented;
        }
#undef CASE
    } else if (can_dispatch_uker(&brg)) {
        CHECK(safe_ptr_assign<brgemm_kernel_t>(
                *brg_kernel, new brgemm_amx_uker_t(brg)));
    } else {
        if (brg.type == brgemm_static_offs) return status::unimplemented;
        if (brg.is_tmm) {
            if (brg.is_f16_tmm) {
                CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel,
                        new brgemm_kernel_common_t<avx512_core_amx_fp16,
                                Xbyak::Tmm>(brg)));
            } else {
                CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel,
                        new brgemm_kernel_common_t<avx512_core_amx, Xbyak::Tmm>(
                                brg)));
            }
        } else if (brg.is_zmm) {
            // isa specific instantiations are required because
            // post-ops require template isa param.
            if (brg.isa_impl == avx512_core_fp16) {
                CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel,
                        new brgemm_kernel_common_t<avx512_core_fp16,
                                Xbyak::Zmm>(brg)));
            } else if (brg.isa_impl == avx512_core_bf16) {
                CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel,
                        new brgemm_kernel_common_t<avx512_core_bf16,
                                Xbyak::Zmm>(brg)));
            } else if (brg.isa_impl == avx512_core_vnni) {
                CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel,
                        new brgemm_kernel_common_t<avx512_core_vnni,
                                Xbyak::Zmm>(brg)));
            } else {
                CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel,
                        new brgemm_kernel_common_t<avx512_core, Xbyak::Zmm>(
                                brg)));
            }
        } else if (brg.is_ymm) {
            if (brg.isa_impl == avx2) {
                CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel,
                        new brgemm_kernel_common_t<avx2, Xbyak::Ymm>(brg)));
            } else if (brg.isa_impl == avx2_vnni) {
                CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel,
                        new brgemm_kernel_common_t<avx2_vnni, Xbyak::Ymm>(
                                brg)));
            } else if (brg.isa_impl == avx2_vnni_2) {
                CHECK(safe_ptr_assign<brgemm_kernel_t>(*brg_kernel,
                        new brgemm_kernel_common_t<avx2_vnni_2, Xbyak::Ymm>(
                                brg)));
            }
        }
    }
    if (!(*brg_kernel)) return status::unimplemented;
    status_t st = (*brg_kernel)->create_kernel();
    if (st != status::success) {
        // `brg_kernel` points to a pointer to kernel class created by `new`.
        // If kernel creation failed, release this resource before returning.
        delete *brg_kernel;
        return st;
    }
    return status::success;
}

status_t brgemm_kernel_destroy(brgemm_kernel_t *brg_kernel) {
    delete brg_kernel;
    return status::success;
}

status_t brgemm_init_tiles(const brgemm_t &brg, char palette[64]) {
    constexpr int max_palette_size_in_bytes = 64;

    if (!brg.is_tmm) return status::unimplemented;

    //TODO: Add support of tail processing by reduction dimension
    auto rd_block = (!brg.rdb && brg.rdb_tail) ? brg.rdb_tail : brg.rd_block;
    if (brg.is_bf32) rd_block = utils::rnd_up(rd_block, 2 /*vnni_granularity*/);

    palette_config_t *buff = (palette_config_t *)(palette);

    char *_tc = (char *)(buff);
    for (int i = 0; i < max_palette_size_in_bytes; i++)
        _tc[i] = 0;

    const int typesize_A = brg.is_bf32 ? sizeof(bfloat16_t) : brg.typesize_A;
    const int typesize_B = brg.is_bf32 ? sizeof(bfloat16_t) : brg.typesize_B;

    const int rd_step = 4 / typesize_A;

    const auto Ac = typesize_A * rd_block;

    const auto Br = (brg.typesize_C != 0) ? Ac / brg.typesize_C : 0;

    if (brg.get_num_A_tiles() + brg.get_num_B_tiles() + brg.get_num_C_tiles()
            > brgemm_t::AMX_TILES_NUM) {
        assert(!"brgemm internal error: invalid blocking");
        return status::unimplemented;
    }

    // Due to interleaving tileload/tmul we don't support blocking 1x6 and 6x1
    //TODO: update gemm_microkernel_amx to support such blocking
    if (brg.get_bd_block2() >= 6 || brg.get_num_C_tiles() >= 6)
        return status::unimplemented;

    for (int m = 0; m < brg.get_num_A_tiles(); m++) {
        const bool is_bd_tail
                = (brg.bdb_tail && m == (brg.get_num_A_tiles() - 1));
        const auto A_tensor = brg.get_A_tensor(m, is_bd_tail);
        const auto Ar = is_bd_tail ? brg.bdb_tail : brg.bd_block;
        tc_configure_tile(buff, A_tensor, Ar, Ac);
    }

    for (int n = 0; n < brg.get_num_B_tiles(); n++) {
        const bool is_ld_tail
                = (brg.ldb_tail && n == (brg.get_num_B_tiles() - 1));
        const auto B_tensor = brg.get_B_tensor(n, is_ld_tail);
        const auto Bc = (is_ld_tail ? brg.ldb_tail : brg.ld_block) * typesize_B
                * rd_step;
        tc_configure_tile(buff, B_tensor, Br, Bc);
    }

    for (int m = 0; m < brg.get_bd_block2(); m++) {
        const bool is_bd_tail
                = (brg.bdb_tail && m == (brg.get_bd_block2() - 1));
        const auto Cr = is_bd_tail ? brg.bdb_tail : brg.bd_block;
        for (int n = 0; n < brg.get_ld_block2(); n++) {
            const bool is_ld_tail
                    = (brg.ldb_tail && n == (brg.get_ld_block2() - 1));
            const auto Cc = (is_ld_tail ? brg.ldb_tail : brg.ld_block)
                    * brg.typesize_C;
            const auto C_tensor
                    = brg.get_C_tensor(m, n, is_bd_tail, is_ld_tail);
            tc_configure_tile(buff, C_tensor, Cr, Cc);
        }
    }

    buff->palette_id = amx::get_target_palette();

    return status::success;
}

namespace {
template <typename T>
static inline int sign(T v) {
    return (v > 0) ? 1 : ((v < 0) ? -1 : 0);
}

int brgemm_cmp(const brgemm_t &lhs, const brgemm_t &rhs) {
    // The macro CMP_BRGEMM_FIELD is designed to compare numerical parameters.
    // Float parameters must not be NaN
#define CMP_BRGEMM_FIELD(x) \
    if ((lhs.x) != (rhs.x)) return sign((lhs.x) - (rhs.x))

    // This function compares brgemm_t objects within a single brgemm primitive.
    // Comparison of objects from different primitives is not guaranteed due to
    // dependencies of brgemm descriptor on a primitive attributes.

    // Compare all non-pointer parameters of brgemm_t except derived
    CMP_BRGEMM_FIELD(bcast_dim);
    CMP_BRGEMM_FIELD(load_dim);
    CMP_BRGEMM_FIELD(reduce_dim);
    CMP_BRGEMM_FIELD(LDA);
    CMP_BRGEMM_FIELD(LDB);
    CMP_BRGEMM_FIELD(LDC);
    CMP_BRGEMM_FIELD(LDD);
    CMP_BRGEMM_FIELD(isa_user);
    CMP_BRGEMM_FIELD(isa_impl);
    CMP_BRGEMM_FIELD(alpha);
    CMP_BRGEMM_FIELD(beta);
    CMP_BRGEMM_FIELD(dt_a);
    CMP_BRGEMM_FIELD(dt_b);
    CMP_BRGEMM_FIELD(dt_c);
    CMP_BRGEMM_FIELD(dt_d);
    CMP_BRGEMM_FIELD(dt_bias);
    CMP_BRGEMM_FIELD(stride_a);
    CMP_BRGEMM_FIELD(stride_b);
    CMP_BRGEMM_FIELD(layout);
    CMP_BRGEMM_FIELD(type);
    CMP_BRGEMM_FIELD(is_dgmm);
    CMP_BRGEMM_FIELD(with_sum);
    CMP_BRGEMM_FIELD(req_cal_comp_pads);

    CMP_BRGEMM_FIELD(sum_scale);
    CMP_BRGEMM_FIELD(sum_zp);
    CMP_BRGEMM_FIELD(sum_dt);
    CMP_BRGEMM_FIELD(with_eltwise);
    CMP_BRGEMM_FIELD(with_binary);
    CMP_BRGEMM_FIELD(with_scales);

    CMP_BRGEMM_FIELD(zp_type_a);
    CMP_BRGEMM_FIELD(zp_type_b);
    CMP_BRGEMM_FIELD(zp_type_c);

    CMP_BRGEMM_FIELD(is_oc_scale);
    CMP_BRGEMM_FIELD(with_dst_scales);

    // Compare all non-pointer parameters of brgemm_attr_t except derived
    CMP_BRGEMM_FIELD(brgattr.max_bs);
    CMP_BRGEMM_FIELD(brgattr.max_top_vpad);
    CMP_BRGEMM_FIELD(brgattr.max_bottom_vpad);
    CMP_BRGEMM_FIELD(brgattr.max_top_bpad);
    CMP_BRGEMM_FIELD(brgattr.max_bottom_bpad);
    CMP_BRGEMM_FIELD(brgattr.hint_expected_A_size);
    CMP_BRGEMM_FIELD(brgattr.hint_expected_B_size);
    CMP_BRGEMM_FIELD(brgattr.hint_expected_C_size);
    CMP_BRGEMM_FIELD(brgattr.hint_innermost_loop);
    CMP_BRGEMM_FIELD(brgattr.hint_loop_order);
    CMP_BRGEMM_FIELD(brgattr.hint_prefetching);
    CMP_BRGEMM_FIELD(brgattr.hint_prfA.dist1);
    CMP_BRGEMM_FIELD(brgattr.hint_prfA.dist2);
    CMP_BRGEMM_FIELD(brgattr.hint_prfB.dist1);
    CMP_BRGEMM_FIELD(brgattr.hint_prfB.dist2);
    CMP_BRGEMM_FIELD(brgattr.hint_prfC.dist1);
    CMP_BRGEMM_FIELD(brgattr.hint_prfC.dist2);
    CMP_BRGEMM_FIELD(brgattr.wary_tail_read);
    CMP_BRGEMM_FIELD(brgattr.generate_skip_accumulation);
    CMP_BRGEMM_FIELD(brgattr.bd_mask_level);
    CMP_BRGEMM_FIELD(brgattr.use_uker);
    CMP_BRGEMM_FIELD(brgattr.use_interleave_stores);
    CMP_BRGEMM_FIELD(brgattr.fpmath_mode);
    CMP_BRGEMM_FIELD(brgattr.LDA2);
    CMP_BRGEMM_FIELD(brgattr.LDB2);
    CMP_BRGEMM_FIELD(brgattr.LDC2_M);
    CMP_BRGEMM_FIELD(brgattr.LDC2_N);
    CMP_BRGEMM_FIELD(brgattr.var_bs);
    CMP_BRGEMM_FIELD(brgattr.postops_only);
    CMP_BRGEMM_FIELD(brgattr.bs_group);

    CMP_BRGEMM_FIELD(brgattr.hint_bd_block);
    CMP_BRGEMM_FIELD(brgattr.hint_ld_block);
    CMP_BRGEMM_FIELD(brgattr.hint_bd_block2);
    CMP_BRGEMM_FIELD(brgattr.hint_ld_block2);
    CMP_BRGEMM_FIELD(brgattr.hint_ununroll_bd_loop);

    CMP_BRGEMM_FIELD(brgattr.hint_load_nt_A);
    CMP_BRGEMM_FIELD(brgattr.hint_load_nt_B);
    CMP_BRGEMM_FIELD(brgattr.K_koef);

    if (lhs.brgattr.bd_mask_level > 0)
        for (int i = 0; i < lhs.bcast_dim; i++) {
            CMP_BRGEMM_FIELD(brgattr.bd_mask[i]);
        }

    if (lhs.type == brgemm_static_offs)
        for (int i = 0; i < lhs.brgattr.max_bs; i++) {
            CMP_BRGEMM_FIELD(brgattr.static_offsets[i].offset.A);
            CMP_BRGEMM_FIELD(brgattr.static_offsets[i].offset.B);
        }

#undef CMP_BRGEMM_FIELD
    return 0;
}
} // namespace

bool brgemm_t::operator==(const brgemm_t &rhs) const {
    return (brgemm_cmp(*this, rhs) == 0);
}

bool brgemm_t::operator<(const brgemm_t &rhs) const {
    return (brgemm_cmp(*this, rhs) < 0);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl_ukernel.h"

#include "cpu/ukernel/c_types_map.hpp"

#if DNNL_X64
#include "cpu/x64/ukernel/brgemm.hpp"
#endif

#ifdef DNNL_EXPERIMENTAL_UKERNEL

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::ukernel;

status_t dnnl_brgemm_create(brgemm_t **brgemm, dim_t M, dim_t N, dim_t K,
        dim_t batch_size, dim_t lda, dim_t ldb, dim_t ldc, data_type_t a_dt,
        data_type_t b_dt, data_type_t c_dt) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_create(
            brgemm, M, N, K, batch_size, lda, ldb, ldc, a_dt, b_dt, c_dt);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_set_add_C(brgemm_t *brgemm, int add_C) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_set_add_C(brgemm, add_C);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_set_post_ops(brgemm_t *brgemm, dim_t ldd, data_type_t d_dt,
        const post_ops_t *post_ops) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_set_post_ops(brgemm, ldd, d_dt, post_ops);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_set_A_scales(brgemm_t *brgemm, int a_scale_mask) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_set_A_scales(brgemm, a_scale_mask);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_set_B_scales(brgemm_t *brgemm, int b_scale_mask) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_set_B_scales(brgemm, b_scale_mask);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_set_D_scales(brgemm_t *brgemm, int d_scale_mask) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_set_D_scales(brgemm, d_scale_mask);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_finalize(brgemm_t *brgemm) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_finalize(brgemm);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_get_B_pack_type(
        pack_type_t *pack_type, data_type_t dt_a, data_type_t dt_b) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_get_B_pack_type(pack_type, dt_a, dt_b);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_get_scratchpad_size(const brgemm_t *brgemm, size_t *size) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_get_scratchpad_size(brgemm, size);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_is_execute_postops_valid(
        const brgemm_t *brgemm, int *valid) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_is_execute_postops_valid(brgemm, valid);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_set_hw_context(const brgemm_t *brgemm) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_set_hw_context(brgemm);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_release_hw_context() {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_release_hw_context();
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_generate(brgemm_t *brgemm) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_generate(brgemm);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_execute(const brgemm_t *brgemm, const void *A_ptr,
        const void *B_ptr, const dim_t *A_B_offsets, void *C_ptr,
        void *scratchpad_ptr) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_execute(
            brgemm, A_ptr, B_ptr, A_B_offsets, C_ptr, scratchpad_ptr);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_execute_postops(const brgemm_t *brgemm, const void *A_ptr,
        const void *B_ptr, const dim_t *A_B_offsets, const void *C_ptr,
        void *D_ptr, void *scratchpad_ptr, const attr_params_t *attr_params) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_execute_postops(brgemm, A_ptr, B_ptr,
            A_B_offsets, C_ptr, D_ptr, scratchpad_ptr, attr_params);
#endif
    return status::unimplemented;
}

status_t dnnl_brgemm_destroy(brgemm_t *brgemm) {
#if DNNL_X64
    return x64::ukernel::dnnl_brgemm_destroy(brgemm);
#endif
    return status::unimplemented;
}

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

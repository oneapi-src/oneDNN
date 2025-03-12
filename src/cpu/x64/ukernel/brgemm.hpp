/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef CPU_X64_UKERNEL_BRGEMM_HPP
#define CPU_X64_UKERNEL_BRGEMM_HPP

#include "cpu/ukernel/c_types_map.hpp"

#include "cpu/x64/brgemm/brgemm_types.hpp"

#include "cpu/x64/ukernel/attr_params.hpp"

#ifdef DNNL_EXPERIMENTAL_UKERNEL

struct dnnl_brgemm : public dnnl::impl::c_compatible {
    dnnl_brgemm(dnnl::impl::dim_t M, dnnl::impl::dim_t N, dnnl::impl::dim_t K,
            dnnl::impl::dim_t batch_size, dnnl::impl::dim_t lda,
            dnnl::impl::dim_t ldb, dnnl::impl::dim_t ldc,
            dnnl::impl::data_type_t a_dt, dnnl::impl::data_type_t b_dt,
            dnnl::impl::data_type_t c_dt)
        : M_(M)
        , N_(N)
        , K_(K)
        , batch_size_(batch_size)
        , lda_(lda)
        , ldb_(ldb)
        , ldc_(ldc)
        , ldd_(ldc) // User may overwrite with set_post_ops().
        , a_dt_(a_dt)
        , b_dt_(b_dt)
        , c_dt_(c_dt)
        , d_dt_(c_dt) // User may overwrite with set_post_ops().
        , beta_(0.f) // User may overwrite with set_add_C().
        , brgemm_kernel_(nullptr) {}

    ~dnnl_brgemm();

    dnnl::impl::status_t set_add_C(int add_C);

    dnnl::impl::status_t set_post_ops(dnnl::impl::dim_t ldd,
            dnnl::impl::data_type_t d_dt,
            const dnnl::impl::post_ops_t *post_ops);

    dnnl::impl::status_t set_scales(int mask, int arg);

    dnnl::impl::status_t finalize();

    static dnnl::impl::status_t get_B_pack_type(
            dnnl::impl::cpu::ukernel::pack_type_t *pack_type,
            dnnl::impl::data_type_t dt_a, dnnl::impl::data_type_t dt_b);

    size_t get_scratchpad_size() const;

    bool is_execute_postops_valid() const;

    dnnl::impl::status_t set_hw_context() const;

    dnnl::impl::status_t generate();

    dnnl::impl::status_t execute(const void *A_ptr, const void *B_ptr,
            const dnnl::impl::dim_t *A_B_offsets, void *C_ptr,
            void *scratchpad_ptr) const;
    dnnl::impl::status_t execute(const void *A_ptr, const void *B_ptr,
            const dnnl::impl::dim_t *A_B_offsets, const void *C_ptr,
            void *D_ptr, void *scratchpad_ptr,
            const dnnl::impl::cpu::ukernel::attr_params_t *attr_params) const;

private:
    // User's inputs.
    dnnl::impl::dim_t M_, N_, K_, batch_size_;
    dnnl::impl::dim_t lda_, ldb_, ldc_, ldd_;
    dnnl::impl::data_type_t a_dt_, b_dt_, c_dt_, d_dt_;
    float beta_;
    // A copy of attributes to avoid dependency on user's attributes lifetime.
    dnnl::impl::primitive_attr_t attr_;

    // A main kernel.
    dnnl::impl::cpu::x64::brgemm_desc_t brgemm_desc_;
    dnnl::impl::cpu::x64::brgemm_kernel_t *brgemm_kernel_;

    // Creates a `verbose_info_` string once during `generate()` call, and calls
    // it during execute(). This is done to avoid string re-creation.
    dnnl::impl::status_t create_verbose_info();
    std::string verbose_info_;
};

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ukernel {

status_t dnnl_brgemm_create(dnnl_brgemm **brgemm, dim_t M, dim_t N, dim_t K,
        dim_t batch_size, dim_t lda, dim_t ldb, dim_t ldc, data_type_t a_dt,
        data_type_t b_dt, data_type_t c_dt);

status_t dnnl_brgemm_set_add_C(dnnl_brgemm *brgemm, int add_C);

status_t dnnl_brgemm_set_post_ops(dnnl_brgemm *brgemm, dim_t ldd,
        data_type_t d_dt, const post_ops_t *post_ops);

status_t dnnl_brgemm_set_A_scales(dnnl_brgemm *brgemm, int a_scale_mask);

status_t dnnl_brgemm_set_B_scales(dnnl_brgemm *brgemm, int b_scale_mask);

status_t dnnl_brgemm_set_D_scales(dnnl_brgemm *brgemm, int d_scale_mask);

status_t dnnl_brgemm_finalize(dnnl_brgemm *brgemm);

status_t dnnl_brgemm_get_B_pack_type(
        dnnl::impl::cpu::ukernel::pack_type_t *pack_type, data_type_t dt_a,
        data_type_t dt_b);

status_t dnnl_brgemm_get_scratchpad_size(
        const dnnl_brgemm *brgemm, size_t *size);

status_t dnnl_brgemm_is_execute_postops_valid(
        const dnnl_brgemm *brgemm, int *valid);

status_t dnnl_brgemm_set_hw_context(const dnnl_brgemm *brgemm);

status_t dnnl_brgemm_release_hw_context();

status_t dnnl_brgemm_generate(dnnl_brgemm *brgemm);

status_t dnnl_brgemm_execute(const dnnl_brgemm *brgemm, const void *A_ptr,
        const void *B_ptr, const dim_t *A_B_offsets, void *C_ptr,
        void *scratchpad_ptr);

status_t dnnl_brgemm_execute_postops(const dnnl_brgemm *brgemm,
        const void *A_ptr, const void *B_ptr, const dim_t *A_B_offsets,
        const void *C_ptr, void *D_ptr, void *scratchpad_ptr,
        const dnnl_ukernel_attr_params *attr_params);

status_t dnnl_brgemm_destroy(dnnl_brgemm *brgemm);

} // namespace ukernel
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

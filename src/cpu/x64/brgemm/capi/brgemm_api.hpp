/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_CAPI_BRGEMM_API_HPP
#define CPU_X64_BRGEMM_CAPI_BRGEMM_API_HPP

#include <memory>

#include "cpu/x64/matmul/brgemm_matmul_copy_utils.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

#include "cpu/x64/brgemm/brgemm_types.hpp"

#ifdef DNNL_EXPERIMENTAL_UKERNEL

// A section identical to c_map_types.hpp but just for brgemm ukernel so far.
namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using pack_type_t = dnnl_pack_type_t;
namespace pack_type {
const pack_type_t undef = dnnl_pack_type_undef;
const pack_type_t no_trans = dnnl_pack_type_no_trans;
const pack_type_t trans = dnnl_pack_type_trans;
const pack_type_t pack32 = dnnl_pack_type_pack32;
} // namespace pack_type

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

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
            const dnnl::impl::primitive_attr_t *attr);

    dnnl::impl::status_t finalize();

    dnnl::impl::cpu::x64::pack_type_t get_B_pack_type() const;

    size_t get_scratchpad_size() const;

    dnnl::impl::status_t set_hw_context() const;

    dnnl::impl::status_t generate();

    dnnl::impl::status_t execute(const void *A_ptr, const void *B_ptr,
            const dnnl::impl::dim_t *A_B_offsets, void *C_ptr,
            void *scratchpad_ptr) const;
    dnnl::impl::status_t execute(const void *A_ptr, const void *B_ptr,
            const dnnl::impl::dim_t *A_B_offsets, const void *C_ptr,
            void *D_ptr, void *scratchpad_ptr, const void *binary_po_ptr) const;

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
};

struct dnnl_brgemm_pack_B : public dnnl::impl::c_compatible {
    // Ctor that follows a call to initialize matmul conf struct.
    dnnl_brgemm_pack_B(dnnl::impl::dim_t K, dnnl::impl::dim_t N,
            dnnl::impl::dim_t in_ld, dnnl::impl::dim_t out_ld,
            dnnl::impl::data_type_t in_dt, dnnl::impl::data_type_t out_dt);

    // Generates a copy_b kernel.
    dnnl::impl::status_t generate();

    // Executes a copy_b kernel.
    dnnl::impl::status_t execute(const void *src, void *dst) const;

private:
    // User's inputs.
    dnnl::impl::dim_t K_, N_;
    dnnl::impl::dim_t in_ld_, out_ld_;
    dnnl::impl::data_type_t in_dt_, out_dt_;

    // A pack_B kernel.
    dnnl::impl::cpu::x64::matmul::brgemm_matmul_conf_t bmc_;
    // unique_ptr is required by API that generates a kernel.
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>
            kernel_;
};

#endif

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

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

#include "oneapi/dnnl/dnnl_ukernel.h"

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/verbose.hpp"

#include "cpu/x64/amx_tile_configure.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"

#include "cpu/x64/brgemm/capi/brgemm_api.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::status;
using namespace dnnl::impl::cpu::x64;

using brgemm_t = dnnl_brgemm;
using brgemm_pack_B_t = dnnl_brgemm_pack_B;

#define VCHECK_BRGEMM(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, brgemm, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__)

#define VCHECK_BRGEMM_STATUS(status, cond, msg, ...) \
    VCONDCHECK(primitive, create, check, brgemm, (cond), (status), msg, \
            ##__VA_ARGS__)

status_t dnnl_brgemm_create(brgemm_t **brgemm, dim_t M, dim_t N, dim_t K,
        dim_t batch_size, dim_t lda, dim_t ldb, dim_t ldc, dim_t ldd,
        data_type_t a_dt, data_type_t b_dt, data_type_t c_dt, data_type_t d_dt,
        float alpha, float beta, const primitive_attr_t *attr) {
    if (brgemm == nullptr) return invalid_arguments;

    auto _brgemm = new brgemm_t();
    auto &brgemm_desc = _brgemm->brgemm_desc_;

    brgemm_batch_kind_t batch_kind = brgemm_batch_kind_t::brgemm_offs;

    auto status = brgemm_desc_init(&brgemm_desc, cpu_isa_t::isa_undef,
            batch_kind, a_dt, b_dt, /* transA = */ false, /* trans_B = */ false,
            brgemm_row_major, alpha, beta, lda, ldb, ldc, M, N, K,
            /* strides = */ nullptr);
    if (status != status::success) {
        delete _brgemm;
        VCHECK_BRGEMM_STATUS(status, false, "brgemm_desc_init failed");
    }

    memory_desc_t D_md;
    dims_t dims {M, N};
    dims_t strides {ldc, 1};
    status = memory_desc_init_by_strides(
            D_md, /* ndims = */ 2, dims, d_dt, strides);
    if (status != status::success) {
        delete _brgemm;
        VCHECK_BRGEMM_STATUS(status, false, "D_md creation failed");
    }

    status = brgemm_desc_set_postops(
            &brgemm_desc, attr, &D_md, ldd, data_type::undef);
    if (status != status::success) {
        delete _brgemm;
        VCHECK_BRGEMM_STATUS(status, false, "brgemm_desc_set_postops failed");
    }

    VCHECK_BRGEMM(batch_size > 0, "batch size is non-positive");
    brgemm_attr_t brgemm_attr;
    brgemm_attr.max_bs = batch_size;
    if (mayiuse(avx512_core_amx)) {
        brgemm_attr.use_uker = true;
        brgemm_attr.use_interleave_stores = true;
        brgemm_attr.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf0;
    }

    status = brgemm_desc_set_attr(&brgemm_desc, brgemm_attr);
    if (status != status::success) {
        delete _brgemm;
        VCHECK_BRGEMM_STATUS(status, false, "brgemm_desc_set_attr failed");
    }

    // Note: API can't take a compensation buffer externally. Users must add
    // compensation on their own as a binary post-op.
    brgemm_desc.req_s8s8_compensation = false;

    *brgemm = _brgemm;
    return status::success;
}

status_t dnnl_brgemm_get_scratchpad_size(const brgemm_t *brgemm, size_t *size) {
    if (utils::any_null(brgemm, size)) return invalid_arguments;

    *size = brgemm->brgemm_desc_.get_wsp_buffer_size();
    return status::success;
}

status_t dnnl_brgemm_set_hw_context(const brgemm_t *brgemm) {
    if (brgemm == nullptr) return invalid_arguments;

    const auto &brgemm_desc = brgemm->brgemm_desc_;
    char palette[AMX_PALETTE_SIZE] = {};
    auto status = brgemm_init_tiles(brgemm_desc, palette);
    if (status == status::success) {
        status = amx_tile_lazy_configure(palette);
        VCHECK_BRGEMM_STATUS(
                status, status == status::success, "amx_tile_configure failed");
    }

    return status::success;
}

status_t dnnl_brgemm_release_hw_context() {
    if (mayiuse(avx512_core_amx)) {
        VCHECK_BRGEMM(amx_tile_release() == status::success,
                "amx_tile_release failed");
    }

    return status::success;
}

status_t dnnl_brgemm_generate(brgemm_t *brgemm) {
    if (brgemm == nullptr) return invalid_arguments;

    // Re-generation won't take any effect.
    if (brgemm->brgemm_kernel_ != nullptr) return status::success;

    auto &brgemm_desc = brgemm->brgemm_desc_;
    auto status = brgemm_kernel_create(&brgemm->brgemm_kernel_, brgemm_desc);
    VCHECK_BRGEMM_STATUS(
            status, status == status::success, "brgemm_kernel_create failed");

    return status::success;
}

status_t dnnl_brgemm_execute(const brgemm_t *brgemm, const void *A_ptr,
        const void *B_ptr, const dim_t *A_B_offsets, void *C_ptr,
        void *scratchpad_ptr) {
    const auto &brgemm_desc = brgemm->brgemm_desc_;
    const auto &brgemm_kernel = brgemm->brgemm_kernel_;

    const auto batch_size = brgemm_desc.brgattr.max_bs;

    std::vector<brgemm_batch_element_t> v_batch_element(batch_size);
    for (int i = 0; i < batch_size; i++) {
        v_batch_element[i].offset.A = A_B_offsets[2 * i];
        v_batch_element[i].offset.B = A_B_offsets[2 * i + 1];
    }

    brgemm_kernel_execute(brgemm_kernel, batch_size, A_ptr, B_ptr,
            v_batch_element.data(), C_ptr, scratchpad_ptr,
            /* dynamic_values = */ nullptr);
    return status::success;
}

status_t dnnl_brgemm_execute_postops(const brgemm_t *brgemm, const void *A_ptr,
        const void *B_ptr, const dim_t *A_B_offsets, const void *C_ptr,
        void *D_ptr, void *scratchpad_ptr, const void *binary_po_ptr) {
    const auto &brgemm_desc = brgemm->brgemm_desc_;
    const auto &brgemm_kernel = brgemm->brgemm_kernel_;

    const auto batch_size = brgemm_desc.brgattr.max_bs;

    std::vector<brgemm_batch_element_t> v_batch_element(batch_size);
    for (int i = 0; i < batch_size; i++) {
        v_batch_element[i].offset.A = A_B_offsets[2 * i];
        v_batch_element[i].offset.B = A_B_offsets[2 * i + 1];
    }

    brgemm_post_ops_data_t post_ops_data;
    post_ops_data.data_C_ptr_ = reinterpret_cast<const char *>(C_ptr);
    // This member expect a pointer to a vector of pointers to binary_po args.
    post_ops_data.binary_post_ops_rhs = &binary_po_ptr;

    bool use_D_as_C = false;
    if (brgemm_desc.dt_c == brgemm_desc.dt_d
            && brgemm_desc.attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::fpmath_mode))
        use_D_as_C = true;

    if (use_D_as_C) C_ptr = D_ptr;

    brgemm_kernel_execute_postops(brgemm_kernel, batch_size, A_ptr, B_ptr,
            v_batch_element.data(), const_cast<void *>(C_ptr), D_ptr,
            post_ops_data, scratchpad_ptr,
            /* dynamic_values = */ nullptr);
    return status::success;
}

status_t dnnl_brgemm_destroy(brgemm_t *brgemm) {
    brgemm_kernel_destroy(brgemm->brgemm_kernel_);
    delete brgemm;

    return status::success;
}

dnnl_brgemm_pack_B::dnnl_brgemm_pack_B(dim_t K, dim_t N, dim_t in_ld,
        dim_t out_ld, data_type_t in_dt, data_type_t out_dt) {
    // So far, only `ab` input format (dense or strided) is supported.
    assert(in_ld >= N);
    // Only special N_blk sizes are supported by matmul copy routines. Rest
    // will crash.
    assert(utils::one_of(out_ld, 16, 32, 48, 64));

    auto status = matmul::init_conf(
            bmc_, /* batch = */ 1, K, N, out_ld, in_dt, out_dt, format_tag::ab);
    assert(status == status::success);
    if (status != status::success) return;
}

bool brgemm_pack_B_t::need_pack() const {
    // TODO: move on unified method from the library.
    return bmc_.orig_wei_dt != data_type::f32
            && bmc_.orig_wei_dt != data_type::f16;
}

void brgemm_pack_B_t::generate() {
    // Re-generation won't take any effect.
    if (kernel_ != nullptr) return;

    auto status = matmul::create_brgemm_matmul_copy_b(kernel_, &bmc_);
    assert(status == status::success);
    if (status != status::success) return;
}

void brgemm_pack_B_t::execute(const void *src, void *dst) const {
    const uint8_t *src_ptr = reinterpret_cast<const uint8_t *>(src);
    uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(dst);

    const auto &kernel_conf = bmc_;
    const dim_t n_blks = utils::div_up(kernel_conf.N, kernel_conf.N_blk);
    const dim_t k_blks = utils::div_up(kernel_conf.K, kernel_conf.K_blk);
    const auto blk_size = kernel_conf.K_blk * kernel_conf.N_blk;

    const auto i_dt_sz = kernel_conf.b_dt_sz;
    const auto o_dt_sz = kernel_conf.a_dt_sz;

    for (dim_t n_blk_idx = 0; n_blk_idx < n_blks; n_blk_idx++) {
        const auto n = n_blk_idx * kernel_conf.N_blk;
        const bool is_N_tail = (kernel_conf.N - n) < kernel_conf.N_blk;
        auto ker_exec_ctx = matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
        ker_exec_ctx.current_N_blk
                = is_N_tail ? kernel_conf.N_tail : kernel_conf.N_blk;

        int k_blk_idx = 0;
        for (; k_blk_idx < kernel_conf.K / kernel_conf.K_blk; k_blk_idx++) {
            const auto k = k_blk_idx * kernel_conf.K_blk;
            assert(kernel_conf.wei_tag == format_tag::ab);
            // Since only `ab` is supported so far, hard code the stride.
            const auto src_offset = i_dt_sz * (k * kernel_conf.N + n);
            const auto dst_offset
                    = o_dt_sz * (k_blk_idx * blk_size + n_blk_idx * k_blks);
            ker_exec_ctx.src = &src_ptr[src_offset];
            ker_exec_ctx.tr_src = &dst_ptr[dst_offset];
            ker_exec_ctx.current_K_start = k;
            ker_exec_ctx.current_K_iters = kernel_conf.K_blk;
            (*kernel_)(&ker_exec_ctx);
        }
        if (kernel_conf.K_tail > 0) {
            const auto k = k_blk_idx * kernel_conf.K_blk;
            assert(kernel_conf.wei_tag == format_tag::ab);
            // Since only `ab` is supported so far, hard code the stride.
            const auto src_offset = i_dt_sz * (k * kernel_conf.N + n);
            const auto dst_offset
                    = o_dt_sz * (k_blk_idx * blk_size + n_blk_idx * k_blks);
            ker_exec_ctx.src = &src_ptr[src_offset];
            ker_exec_ctx.tr_src = &dst_ptr[dst_offset];
            ker_exec_ctx.current_K_start = k;
            ker_exec_ctx.current_K_iters = kernel_conf.K_tail;
            (*kernel_)(&ker_exec_ctx);
        }
    }
}

status_t dnnl_brgemm_pack_B_create(brgemm_pack_B_t **brgemm_pack_B, dim_t K,
        dim_t N, dim_t in_ld, dim_t out_ld, data_type_t in_dt,
        data_type_t out_dt) {
    if (brgemm_pack_B == nullptr) return status::invalid_arguments;

    *brgemm_pack_B = new brgemm_pack_B_t(K, N, in_ld, out_ld, in_dt, out_dt);
    return status::success;
}

status_t dnnl_brgemm_pack_B_need_pack(
        const brgemm_pack_B_t *brgemm_pack_B, int *need_pack) {
    if (utils::any_null(brgemm_pack_B, need_pack))
        return status::invalid_arguments;

    *need_pack = brgemm_pack_B->need_pack();
    return status::success;
}

status_t dnnl_brgemm_pack_B_generate(brgemm_pack_B_t *brgemm_pack_B) {
    if (brgemm_pack_B == nullptr) return status::invalid_arguments;

    brgemm_pack_B->generate();
    return status::success;
}

status_t dnnl_brgemm_pack_B_execute(const brgemm_pack_B_t *brgemm_pack_B,
        const void *in_ptr, void *out_ptr) {
    if (utils::any_null(brgemm_pack_B, in_ptr, out_ptr))
        return status::invalid_arguments;

    brgemm_pack_B->execute(in_ptr, out_ptr);
    return status::success;
}

status_t dnnl_brgemm_pack_B_destroy(brgemm_pack_B_t *brgemm_pack_B) {
    delete brgemm_pack_B;
    return status::success;
}

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

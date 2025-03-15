/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
* Copyright 2025 Arm Ltd. and affiliates
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
#include "cpu/aarch64/capi/capi.hpp"
#include "oneapi/dnnl/dnnl_ukernel.h"

#ifdef DNNL_EXPERIMENTAL_UKERNEL
#ifdef DNNL_AARCH64_USE_KAI
#include "cpu/aarch64/kleidiai/kai.hpp"
#endif
using namespace dnnl::impl;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::status;
using namespace dnnl::impl::cpu::aarch64;
using namespace dnnl::impl::cpu::aarch64::capi;

using brgemm_t = dnnl_brgemm;
using transform_t = dnnl_transform;

status_t attr_params_t::set_post_ops_args(const void **post_ops_args) {
    post_ops_args_ = post_ops_args;
    return status::success;
}

status_t attr_params_t::set_scales(const void *scales, int arg) {
    switch (arg) {
        case DNNL_ARG_SRC: a_scales_ = scales; break;
        case DNNL_ARG_WEIGHTS: b_scales_ = scales; break;
        case DNNL_ARG_DST: d_scales_ = scales; break;
        default: assert(!"unsupported arg");
    }
    return status::success;
}

const void *attr_params_t::get_scales(int arg) const {
    switch (arg) {
        case DNNL_ARG_SRC: return a_scales_;
        case DNNL_ARG_WEIGHTS: return b_scales_;
        case DNNL_ARG_DST: return d_scales_;
        default: assert(!"unsupported arg");
    }
    return nullptr;
}

status_t attr_params_t::set_bias(const void *bias) {
    bias_ = bias;
    return status::success;
}

const void *attr_params_t::get_bias() const {
    return bias_;
}

dnnl_transform::dnnl_transform(dim_t K, dim_t N, pack_type_t in_pack_type,
        dim_t in_ld, dim_t out_ld, data_type_t in_dt, data_type_t out_dt)
    : K_(K)
    , N_(N)
    , in_pack_type_(in_pack_type)
    , in_ld_(in_ld)
    , out_ld_(out_ld)
    , in_dt_(in_dt)
    , out_dt_(out_dt) {
    if (!utils::one_of(in_pack_type_, pack_type::kai_pack_int4,
                pack_type::kai_pack_f32)) {
        // Check for a valid in_ld depending on a pack type.
        assert(in_pack_type == pack_type::no_trans
                        ? IMPLICATION(K_ > 1, in_ld_ >= N_)
                        : in_ld_ >= K_);
        // Only special N_blk sizes are supported by matmul copy routines. Rest
        // will crash.
        assert(utils::one_of(out_ld_, 16, 32, 48, 64));

        const auto in_tag = in_pack_type == pack_type::trans ? format_tag::ba
                                                             : format_tag::ab;
        auto status = matmul::init_conf(bmc_, /* batch = */ 1, /* M = */ 0, K_,
                N_, in_ld_, out_ld_, in_dt_, out_dt_, in_tag);
        assert(status == status::success);
        if (status != status::success) return;

        if (in_pack_type == pack_type::trans) {
            strides_[0] = 1;
            strides_[1] = in_ld_;
        } else if (in_pack_type == pack_type::no_trans) {
            strides_[0] = in_ld_;
            strides_[1] = 1;
        } else {
            assert(!"Unsupported pack type");
        }
    }
}

status_t transform_t::generate() {
    if (!utils::one_of(in_pack_type_, pack_type::kai_pack_int4,
                pack_type::kai_pack_f32)) {
        // Re-generation won't take any effect.
        if (pack_B_kernel_ != nullptr) return status::success;

        CHECK(matmul::create_brgemm_matmul_copy_b(pack_B_kernel_, &bmc_));

        // Generate a verbose info string at the point where configuration is done.
        if (get_verbose(verbose_t::exec_profile, component_t::ukernel)) {
            CHECK(create_verbose_info());
        }
    }
    return status::success;
}

size_t transform_t::get_output_size() const {
#ifdef DNNL_AARCH64_USE_KAI
    if (in_pack_type_ == pack_type::kai_pack_int4) {
        brgemm_desc_t desc;
        if (in_dt_ == dnnl::impl::data_type_t::dnnl_f32) {
            if ((N_ % desc.BL == 0) && (desc.BL < N_))
                return kai_f32_qa8dxp_qs4c32p_kernel_packet_t(
                        kai_ukernel_id::f32_qai8dxp_qsi4c32p_gemv)
                        .get_lhs_packed_size(K_ /*M*/, N_ /*K*/);
            else
                return kai_f32_qa8dxp_qs4cxp_kernel_packet_t(
                        kai_ukernel_id::f32_qai8dxp_qsi4cxp_gemv)
                        .get_lhs_packed_size(K_ /*M*/, N_ /*K*/);

        } else if (in_dt_ == dnnl::impl::data_type_t::dnnl_s4) {
            if ((K_ % desc.BL == 0) && (desc.BL < K_))
                return kai_f32_qa8dxp_qs4c32p_kernel_packet_t(
                        kai_ukernel_id::f32_qai8dxp_qsi4c32p_gemv)
                        .get_rhs_packed_size(N_ /*N*/, K_ /*K*/);
            else
                return kai_f32_qa8dxp_qs4cxp_kernel_packet_t(
                        kai_ukernel_id::f32_qai8dxp_qsi4cxp_gemv)
                        .get_rhs_packed_size(N_ /*N*/, K_ /*K*/);
        }
    } else if (in_pack_type_ == pack_type::kai_pack_f32) {
        if (in_dt_ == dnnl::impl::data_type_t::dnnl_f32) {
            return kai_f32_f32_f32p_kernel_packet_t(
                    kai_ukernel_id::f32_f32_f32p)
                    .get_rhs_packed_size(K_ /*M*/, N_ /*K*/);
        }
    }
#endif
    return 0;
}

status_t transform_t::execute(const attr_params_t *attr_params,
        const void *in_ptr, void *out_packed_ptr) const {
    brgemm_desc_t desc;
    status_t status = status::success;
#ifdef DNNL_AARCH64_USE_KAI
    if (in_pack_type_ == pack_type::kai_pack_int4) {
        if (in_dt_ == dnnl::impl::data_type_t::dnnl_f32) {
            desc.M = K_;
            desc.K = N_;
            if ((N_ % desc.BL == 0) && (desc.BL < N_))
                kai_f32_qa8dxp_qs4c32p_kernel_packet_t(desc).run_lhs_quant_pack(
                        reinterpret_cast<const float *>(in_ptr),
                        out_packed_ptr);
            else
                kai_f32_qa8dxp_qs4cxp_kernel_packet_t(desc).run_lhs_quant_pack(
                        reinterpret_cast<const float *>(in_ptr),
                        out_packed_ptr);
            status = status::success;
        } else if (in_dt_ == dnnl::impl::data_type_t::dnnl_s4) {
            desc.K = K_;
            desc.N = N_;
            if ((K_ % desc.BL == 0) && (desc.BL < K_))
                kai_f32_qa8dxp_qs4c32p_kernel_packet_t(desc).run_rhs_pack(
                        reinterpret_cast<const uint8_t *>(in_ptr),
                        reinterpret_cast<const float *>(
                                attr_params->get_scales(DNNL_ARG_WEIGHTS)),
                        reinterpret_cast<const float *>(
                                attr_params->get_bias()),
                        reinterpret_cast<uint8_t *>(out_packed_ptr));
            else
                kai_f32_qa8dxp_qs4cxp_kernel_packet_t(desc).run_rhs_pack(
                        reinterpret_cast<const uint8_t *>(in_ptr),
                        reinterpret_cast<const float *>(
                                attr_params->get_scales(DNNL_ARG_WEIGHTS)),
                        reinterpret_cast<const float *>(
                                attr_params->get_bias()),
                        reinterpret_cast<uint8_t *>(out_packed_ptr));
        }
    } else if (in_pack_type_ == pack_type::kai_pack_f32) {
        desc.K = K_;
        desc.N = N_;
        kai_f32_f32_f32p_kernel_packet_t(desc).run_rhs_pack(
                reinterpret_cast<const float *>(in_ptr),
                reinterpret_cast<const float *>(attr_params->get_bias()),
                reinterpret_cast<float *>(out_packed_ptr));
    }
#endif
    return status;
}

status_t transform_t::execute(const void *src, void *dst) const {
    double start_ms = 0;
    if (get_verbose(verbose_t::exec_profile, component_t::ukernel))
        start_ms = get_msec();

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
            const auto src_offset
                    = i_dt_sz * (k * strides_[0] + n * strides_[1]);
            const auto dst_offset
                    = o_dt_sz * (k_blk_idx * blk_size + n_blk_idx * k_blks);
            ker_exec_ctx.src = &src_ptr[src_offset];
            ker_exec_ctx.tr_src = &dst_ptr[dst_offset];
            ker_exec_ctx.current_K_start = k;
            ker_exec_ctx.current_K_iters = kernel_conf.K_blk;
            (*pack_B_kernel_)(&ker_exec_ctx);
        }
        if (kernel_conf.K_tail > 0) {
            const auto k = k_blk_idx * kernel_conf.K_blk;
            const auto src_offset
                    = i_dt_sz * (k * strides_[0] + n * strides_[1]);
            const auto dst_offset
                    = o_dt_sz * (k_blk_idx * blk_size + n_blk_idx * k_blks);
            ker_exec_ctx.src = &src_ptr[src_offset];
            ker_exec_ctx.tr_src = &dst_ptr[dst_offset];
            ker_exec_ctx.current_K_start = k;
            ker_exec_ctx.current_K_iters = kernel_conf.K_tail;
            (*pack_B_kernel_)(&ker_exec_ctx);
        }
    }

    if (get_verbose(verbose_t::exec_profile, component_t::ukernel)) {
        double duration_ms = get_msec() - start_ms;

        std::stringstream ss;
        ss << "cpu,transform,pack_B,undef," << verbose_info_;
        VPROF(start_ms, ukernel, exec, VERBOSE_profile, ss.str().c_str(),
                duration_ms);
    }
    return status::success;
}

status_t transform_t::create_verbose_info() {
#if defined(DISABLE_VERBOSE)
    return status::success;
#endif

    std::stringstream ss;

    memory_desc_t src_md;
    const dims_t dims = {K_, N_};
    CHECK(memory_desc_init_by_strides(src_md, 2, dims, in_dt_, strides_));

    memory_desc_t dst_md;
    const dims_t dst_strides = {out_ld_, 1};
    CHECK(memory_desc_init_by_strides(dst_md, 2, dims, out_dt_, dst_strides));

    ss << md2fmt_str("src", &src_md, format_kind::undef) << " ";
    ss << md2fmt_str("dst", &dst_md, format_kind::undef);
    ss << ",,," << md2dim_str(&src_md);

    verbose_info_ = ss.str();
    return status::success;
}

////////////////
// Public API //
////////////////

/////////////////////////
// Attribute arguments //
/////////////////////////

status_t dnnl_ukernel_attr_params_create(attr_params_t **attr_params) {
    *attr_params = new attr_params_t();
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_post_ops_args(
        attr_params_t *attr_params, const void **post_ops_args) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_post_ops_args(post_ops_args));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_A_scales(
        attr_params_t *attr_params, const void *a_scales) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_scales(a_scales, DNNL_ARG_SRC));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_B_scales(
        attr_params_t *attr_params, const void *b_scales) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_scales(b_scales, DNNL_ARG_WEIGHTS));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_D_scales(
        attr_params_t *attr_params, const void *d_scales) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_scales(d_scales, DNNL_ARG_DST));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_bias(
        attr_params_t *attr_params, const void *bias) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_bias(bias));
    return status::success;
}

status_t dnnl_ukernel_attr_params_destroy(attr_params_t *attr_params) {
    delete attr_params;
    return status::success;
}

///////////////
// Transform //
///////////////
status_t dnnl_transform_create(transform_t **transform, dim_t K, dim_t N,
        pack_type_t in_pack_type, dim_t in_ld, dim_t out_ld, data_type_t in_dt,
        data_type_t out_dt) {
    if (transform == nullptr) return status::invalid_arguments;

    *transform
            = new transform_t(K, N, in_pack_type, in_ld, out_ld, in_dt, out_dt);
    return status::success;
}

status_t dnnl_transform_generate(transform_t *transform) {
    if (transform == nullptr) return status::invalid_arguments;

    CHECK(transform->generate());
    return status::success;
}

status_t dnnl_transform_get_output_size(transform_t *transform, size_t *size) {
    if (transform == nullptr) return status::invalid_arguments;

    *size = transform->get_output_size();
    return status::success;
}

status_t dnnl_transform_execute_attr_params(const transform_t *transform,
        const attr_params_t *attr_params, const void *in_ptr, void *out_ptr) {

    if (utils::any_null(transform, in_ptr, out_ptr))
        return status::invalid_arguments;

    CHECK(transform->execute(attr_params, in_ptr, out_ptr));
    return status::success;
}

status_t dnnl_transform_execute(
        const transform_t *transform, const void *in_ptr, void *out_ptr) {
    if (utils::any_null(transform, in_ptr, out_ptr))
        return status::invalid_arguments;

    CHECK(transform->execute(in_ptr, out_ptr));
    return status::success;
}

status_t dnnl_transform_destroy(transform_t *transform) {
    delete transform;
    return status::success;
}

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

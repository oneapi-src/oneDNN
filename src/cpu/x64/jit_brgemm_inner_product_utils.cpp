/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_brgemm_inner_product_utils.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace prop_kind;
using namespace data_type;

namespace brgemm_inner_product_utils {

// TODO: add support of post-ops with multiple binary and eltwise execution
bool post_ops_ok(
        jit_brgemm_primitive_conf_t &jbgp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    switch (p.len()) {
        case 0: return true;
        case 1: return is_eltwise(0) || p.contain(sum, 0);
        case 2:
            return (p.contain(sum, 0) && is_eltwise(1))
                    || (one_of(jbgp.src_dt, u8, s8) && p.contain(sum, 1)
                            && is_eltwise(0));
        default: return false;
    }

    return false;
}

status_t init_ip_conf_fwd(
        jit_brgemm_primitive_conf_t &jbgp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;
    jbgp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jbgp.with_eltwise = eltwise_ind != -1;
    if (jbgp.with_eltwise) jbgp.eltwise = p.entry_[eltwise_ind].eltwise;
    if (!post_ops_ok(jbgp, attr)) return status::unimplemented;
    if (jbgp.with_scales) {
        const auto &oscales = attr.output_scales_;
        jbgp.is_oc_scale = oscales.mask_ == 1 << 1;

        // only common and per-oc-channel scales are supported
        const bool oscales_ok = one_of(oscales.mask_, 0, 1 << 1);
        if (!oscales_ok) return status::unimplemented;
    }

    jbgp.use_buffer = IMPLICATION(jbgp.dst_dt == jbgp.acc_dt, jbgp.with_sum);

    jbgp.ic_block = jbgp.simd_w;
    if (jbgp.oc >= 4 * jbgp.simd_w) {
        jbgp.oc_block = 4 * jbgp.simd_w;
    } else if (jbgp.oc >= 2 * jbgp.simd_w) {
        jbgp.oc_block = 2 * jbgp.simd_w;
    } else {
        jbgp.oc_block = jbgp.simd_w;
    }

    jbgp.nb_ic = utils::div_up(jbgp.ic, jbgp.ic_block);
    jbgp.nb_oc = utils::div_up(jbgp.oc, jbgp.oc_block);
    jbgp.os = jbgp.mb;

    // Configure matrix sizes
    const int max_M = 64, min_M = 6;
    jbgp.os_block = 1;
    for (int m_ = max_M; m_ >= min_M; m_--) {
        if (jbgp.os % m_ == 0) {
            jbgp.os_block = m_;
            break;
        }
    }
    if (jbgp.os_block == 1) jbgp.os_block = nstl::min(jbgp.os, max_M);
    jbgp.nb_os = utils::div_up(jbgp.os, jbgp.os_block);
    jbgp.nb_os_blocking = 1;
    jbgp.M = jbgp.os_block;
    jbgp.M_tail = jbgp.os % jbgp.os_block;

    jbgp.K = jbgp.ic_block;
    jbgp.N = jbgp.oc_block;
    jbgp.N_tail = jbgp.oc % jbgp.oc_block;
    jbgp.K_tail = jbgp.ic % jbgp.ic_block;

    jbgp.LDA = jbgp.ic_without_padding;
    jbgp.LDB = jbgp.N;
    jbgp.LDC = (jbgp.use_buffer) ? jbgp.N : jbgp.oc_without_padding;
    jbgp.LDD = jbgp.oc_without_padding;

    jbgp.nb_ic_blocking = 1;
    for (int bl = 64; bl >= 1; bl--)
        if (jbgp.nb_ic % bl == 0) {
            jbgp.nb_ic_blocking = bl;
            break;
        }

    jbgp.gemm_batch_size = jbgp.nb_ic_blocking;

    return status::success;
}

status_t init_ip_conf_bwd_d(jit_brgemm_primitive_conf_t &jbgp) {
    jbgp.use_buffer_b = true;
    jbgp.use_buffer = jbgp.src_dt != jbgp.acc_dt;

    jbgp.ic_block = jbgp.simd_w;
    if (jbgp.ic >= 4 * jbgp.simd_w)
        jbgp.ic_block = 4 * jbgp.simd_w;
    else if (jbgp.ic >= 2 * jbgp.simd_w)
        jbgp.ic_block = 2 * jbgp.simd_w;
    jbgp.oc_block = jbgp.simd_w;

    jbgp.nb_ic = utils::div_up(jbgp.ic, jbgp.ic_block);
    jbgp.nb_oc = utils::div_up(jbgp.oc, jbgp.oc_block);
    jbgp.os = jbgp.mb;

    // Configure matrix sizes
    const int max_M = 64, min_M = 6;
    jbgp.os_block = 1;
    for (int m_ = max_M; m_ >= min_M; m_--) {
        if (jbgp.os % m_ == 0) {
            jbgp.os_block = m_;
            break;
        }
    }
    if (jbgp.os_block == 1) jbgp.os_block = nstl::min(jbgp.os, max_M);
    jbgp.nb_os = utils::div_up(jbgp.os, jbgp.os_block);
    jbgp.nb_os_blocking = 1;
    jbgp.M = jbgp.os_block;
    jbgp.M_tail = jbgp.os % jbgp.os_block;

    jbgp.K = jbgp.oc_block;
    jbgp.N = jbgp.ic_block;
    jbgp.N_tail = jbgp.ic % jbgp.ic_block;
    jbgp.K_tail = jbgp.oc % jbgp.oc_block;

    jbgp.LDA = jbgp.oc_without_padding;
    jbgp.LDB = jbgp.N;
    jbgp.LDC = (jbgp.use_buffer) ? jbgp.N : jbgp.ic_without_padding;
    jbgp.LDD = jbgp.ic_without_padding;

    jbgp.nb_oc_blocking = 1;
    for (int bl = 64; bl >= 1; bl--)
        if (jbgp.nb_oc % bl == 0) {
            jbgp.nb_oc_blocking = bl;
            break;
        }

    jbgp.gemm_batch_size = jbgp.nb_oc_blocking;

    return status::success;
}

status_t init_ip_conf(jit_brgemm_primitive_conf_t &jbgp,
        const inner_product_desc_t &ipd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, int nthreads) {

    using namespace prop_kind;
    if (!mayiuse(avx512_common)) return status::unimplemented;

    int ndims = src_d.ndims();
    int dst_ndims = dst_d.ndims();
    if (dst_ndims != 2) return status::unimplemented;

    jbgp = zero<decltype(jbgp)>();
    jbgp.ndims = ndims;
    jbgp.prop_kind = ipd.prop_kind;
    jbgp.ngroups = 1;
    jbgp.mb = src_d.dims()[0];
    jbgp.oc_without_padding = dst_d.dims()[1];
    jbgp.oc = jbgp.oc_without_padding;
    jbgp.ic_without_padding = src_d.dims()[1];
    jbgp.ic = jbgp.ic_without_padding;
    jbgp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jbgp.ih = (ndims < 4) ? 1 : src_d.dims()[ndims - 2];
    jbgp.iw = (ndims < 3) ? 1 : src_d.dims()[ndims - 1];
    jbgp.od = jbgp.oh = jbgp.ow = 1;
    jbgp.kd = (ndims == 5) ? weights_d.dims()[2] : 1;
    jbgp.kh = (ndims < 4) ? 1 : weights_d.dims()[ndims - 2];
    jbgp.kw = (ndims < 3) ? 1 : weights_d.dims()[ndims - 1];
    jbgp.stride_d = jbgp.stride_h = jbgp.stride_w = 1;

    if (!utils::everyone_is(1, jbgp.ow, jbgp.oh, jbgp.od))
        return status::unimplemented;
    if (jbgp.kw != jbgp.iw || jbgp.kh != jbgp.ih || jbgp.kd != jbgp.id)
        return status::unimplemented;
    if (!utils::everyone_is(1, jbgp.kw, jbgp.kh, jbgp.kd))
        return status::unimplemented;
    if (ndims != 2) return status::unimplemented;

    const int full_simd_w = 16;
    jbgp.simd_w = full_simd_w;

    jbgp.with_bias
            = pick_by_prop_kind(jbgp.prop_kind, ipd.bias_desc.format_kind,
                      format_kind::undef, ipd.diff_bias_desc.format_kind)
            != format_kind::undef;

    jbgp.src_dt = src_d.data_type();
    jbgp.dst_dt = dst_d.data_type();
    jbgp.wei_dt = weights_d.data_type();
    jbgp.bia_dt = jbgp.with_bias
            ? pick_by_prop_kind(jbgp.prop_kind, ipd.bias_desc.data_type,
                    data_type::undef, ipd.diff_bias_desc.data_type)
            : data_type::undef;

    if (!IMPLICATION(jbgp.wei_dt == s8, mayiuse(avx512_core_vnni)))
        return status::unimplemented;
    if (!IMPLICATION(jbgp.wei_dt == bf16, mayiuse(avx512_core_bf16)))
        return status::unimplemented;

    if (one_of(jbgp.src_dt, u8, s8)) {
        jbgp.acc_dt = s32;
        jbgp.with_scales = true;
    } else if (one_of(jbgp.src_dt, f32, bf16)) {
        jbgp.acc_dt = f32;
    } else
        return status::unimplemented;

    format_tag_t dat_tag = nc;
    format_tag_t wei_tag
            = get_brgemm_ip_weights_tag((dim_t)jbgp.oc, jbgp.wei_dt);
    if (wei_tag == format_tag::undef) return status::unimplemented;

    jbgp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jbgp.dst_tag = dst_d.matches_one_of_tag(dat_tag);
    jbgp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    if (jbgp.src_tag != dat_tag || jbgp.dst_tag != dat_tag
            || jbgp.wei_tag != wei_tag)
        return status::unimplemented;

    jbgp.brg_type = brgemm_addr;
    jbgp.nthr = nthreads;

    switch (jbgp.prop_kind) {
        case forward_training:
        case forward_inference: return init_ip_conf_fwd(jbgp, attr);
        case backward_data: return init_ip_conf_bwd_d(jbgp);
        default: assert(!"invalid prop_kind"); return invalid_arguments;
    }
}

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_primitive_conf_t &jbgp) {
    size_t sc_size = sizeof(void *);
    size_t n_elems = jbgp.nthr * 16 * jbgp.gemm_batch_size;
    if (jbgp.brg_type == brgemm_addr) {
        scratchpad.book(key_brgemm_primitive_addr_a, n_elems, sc_size, 64);
        scratchpad.book(key_brgemm_primitive_addr_b, n_elems, sc_size, 64);
    }
    if (jbgp.use_buffer) {
        scratchpad.book(key_brgemm_primitive_buffer,
                jbgp.nthr * jbgp.LDC * jbgp.M,
                types::data_type_size(jbgp.acc_dt));
    }

    if (jbgp.use_buffer_b && jbgp.prop_kind == dnnl_backward_data) {
        int size_B = jbgp.LDB * rnd_up(jbgp.K, 2);
#ifndef BRGEMM_IP_BWD_D_GLOBAL_B_TRANSPOSE
        scratchpad.book(key_brgemm_primitive_buffer_b,
                (dim_t)jbgp.nthr * jbgp.gemm_batch_size * size_B,
                types::data_type_size(jbgp.wei_dt));
#else
        scratchpad.book(key_brgemm_primitive_buffer_b,
                (dim_t)jbgp.nb_oc * jbgp.nb_ic * size_B,
                types::data_type_size(jbgp.wei_dt));
#endif
    }
}

} // namespace brgemm_inner_product_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

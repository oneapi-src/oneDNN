/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/ref_deconvolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

void ref_deconvolution_fwd_t::compute_fwd_bias(
        float *dst, const float *bias) const {
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int OD = pd()->OD();
    const int OC = pd()->OC() / G;
    const int ndims = pd()->desc()->src_desc.ndims;

    parallel_nd(MB, G, OC, OD, OH, OW,
            [&](int mb, int g, int oc, int od, int oh, int ow) {
                auto b = bias[g * OC + oc];
                switch (ndims) {
                    case 5:
                        dst[dst_d.off(mb, g * OC + oc, od, oh, ow)] += b;
                        break;
                    case 4: dst[dst_d.off(mb, g * OC + oc, oh, ow)] += b; break;
                    case 3: dst[dst_d.off(mb, g * OC + oc, ow)] += b; break;
                    default: assert(!"invalid dimension size");
                }
            });
}

template <data_type_t dst_type, data_type_t bia_type>
void ref_deconvolution_fwd_t::compute_fwd_bias_ncdhw(
        typename prec_traits<dst_type>::type *dst,
        const typename prec_traits<bia_type>::type *bias) const {
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int SP = pd()->OW() * pd()->OH() * pd()->OD();

    parallel_nd(MB, OC, [&](int mb, int oc) {
        PRAGMA_OMP_SIMD()
        for (int sp = 0; sp < SP; ++sp) {
            auto offset = (size_t)(mb * OC + oc) * SP + sp;
            dst[offset] += bias[oc];
        }
    });
}

template <data_type_t dst_type, data_type_t bia_type, int blksize>
void ref_deconvolution_fwd_t::compute_fwd_bias_nCdhwXc(
        typename prec_traits<dst_type>::type *dst,
        const typename prec_traits<bia_type>::type *bias) const {
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int SP = pd()->OW() * pd()->OH() * pd()->OD();

    const ptrdiff_t stride_mb = dst_d.blocking_desc().strides[0];

    parallel_nd(MB, utils::div_up(OC, blksize), SP,
            [&](int mb, int oc_blk, int sp) {
                int oc = oc_blk * blksize;
                auto offset = mb * stride_mb + oc * SP + sp * blksize;
                const int blk = nstl::min(blksize, OC - oc);

                PRAGMA_OMP_SIMD()
                for (int i = 0; i < blk; ++i)
                    dst[offset + i] += bias[oc + i];
            });
}

template <data_type_t dst_type, data_type_t bia_type>
void ref_deconvolution_fwd_t::compute_bias(const exec_ctx_t &ctx) const {
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<bia_type>::type bia_data_t;

    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    auto bias = CTX_IN_MEM(const bia_data_t *, DNNL_ARG_BIAS);

    using namespace format_tag;
    switch (pd()->dst_tag_) {
        case ncdhw:
        case nchw:
        case ncw: compute_fwd_bias_ncdhw<dst_type, bia_type>(dst, bias); break;
        case nCdhw8c:
        case nChw8c:
        case nCw8c:
            assert(!utils::one_of(data_type::bf16, dst_type, bia_type));
            compute_fwd_bias_nCdhwXc<dst_type, bia_type, 8>(dst, bias);
            break;
        case nCdhw16c:
        case nChw16c:
        case nCw16c:
            compute_fwd_bias_nCdhwXc<dst_type, bia_type, 16>(dst, bias);
            break;
        default:
            assert(!utils::one_of(data_type::bf16, dst_type, bia_type));
            compute_fwd_bias((float *)(dst), (const float *)(bias));
            break;
    }
}

void ref_deconvolution_bwd_weights_t::compute_bwd_bias(
        float *diff_bias, const float *diff_dst) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int OC = pd()->OC() / G;
    const int OD = pd()->OD();
    const int ndims = pd()->desc()->src_desc.ndims;

    parallel_nd(G, OC, [&](int g, int oc) {
        float db = 0;
        for (int mb = 0; mb < MB; ++mb) {
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        switch (ndims) {
                            case 5:
                                db += diff_dst[diff_dst_d.off(
                                        mb, g * OC + oc, od, oh, ow)];
                                break;
                            case 4:
                                db += diff_dst[diff_dst_d.off(
                                        mb, g * OC + oc, oh, ow)];
                                break;
                            case 3:
                                db += diff_dst[diff_dst_d.off(
                                        mb, g * OC + oc, ow)];
                                break;
                            default: assert(!"invalid dimension size");
                        }
                    }
                }
            }
        }
        diff_bias[g * OC + oc] = db;
    });
}

template <data_type_t dbia_type, data_type_t ddst_type>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_ncdhw(
        typename prec_traits<dbia_type>::type *diff_bias,
        const typename prec_traits<ddst_type>::type *diff_dst) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const int OC = pd()->OC();
    const int MB = pd()->MB();
    const int SP = pd()->OH() * pd()->OW() * pd()->OD();

    parallel_nd(OC, [&](int oc) {
        float db = 0;
        for (int mb = 0; mb < MB; ++mb) {
            PRAGMA_OMP_SIMD(reduction(+ : db))
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = (size_t)(mb * OC + oc) * SP + sp;
                db += diff_dst[offset];
            }
        }
        diff_bias[oc] = db;
    });
}

template <data_type_t dbia_type, data_type_t ddst_type, int blksize>
void ref_deconvolution_bwd_weights_t::compute_bwd_bias_nCdhwXc(
        typename prec_traits<dbia_type>::type *diff_bias,
        const typename prec_traits<ddst_type>::type *diff_dst) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    const int OC = pd()->OC();
    const int MB = pd()->MB();
    const int SP = pd()->OH() * pd()->OW() * pd()->OD();

    const ptrdiff_t stride_mb = diff_dst_d.blocking_desc().strides[0];

    parallel_nd(utils::div_up(OC, blksize), [&](int ocb) {
        float db[blksize] = {0};

        for (int mb = 0; mb < MB; ++mb) {
            for (int sp = 0; sp < SP; ++sp) {
                auto offset = mb * stride_mb + (ocb * SP + sp) * blksize;

                PRAGMA_OMP_SIMD()
                for (int i = 0; i < blksize; ++i)
                    db[i] += diff_dst[offset + i];
            }
        }

        const int blk = nstl::min(blksize, OC - ocb * blksize);

        PRAGMA_OMP_SIMD()
        for (int i = 0; i < blk; ++i)
            diff_bias[ocb * blksize + i] = db[i];
    });
}

template <data_type_t dbia_type, data_type_t ddst_type>
void ref_deconvolution_bwd_weights_t::compute_bias(
        const exec_ctx_t &ctx) const {
    typedef typename prec_traits<dbia_type>::type dbia_data_t;
    typedef typename prec_traits<ddst_type>::type ddst_data_t;

    auto diff_bias = CTX_OUT_MEM(dbia_data_t *, DNNL_ARG_DIFF_BIAS);
    auto diff_dst = CTX_IN_MEM(const ddst_data_t *, DNNL_ARG_DIFF_DST);

    using namespace format_tag;
    switch (pd()->dst_tag_) {
        case ncdhw:
        case nchw:
        case ncw:
            compute_bwd_bias_ncdhw<dbia_type, ddst_type>(diff_bias, diff_dst);
            break;
        case nCdhw8c:
        case nChw8c:
        case nCw8c:
            assert(!utils::one_of(data_type::bf16, dbia_type, ddst_type));
            compute_bwd_bias_nCdhwXc<dbia_type, ddst_type, 8>(
                    diff_bias, diff_dst);
            break;
        case nCdhw16c:
        case nChw16c:
        case nCw16c:
            compute_bwd_bias_nCdhwXc<dbia_type, ddst_type, 16>(
                    diff_bias, diff_dst);
            break;
        default:
            assert(!utils::one_of(data_type::bf16, dbia_type, ddst_type));
            compute_bwd_bias((float *)diff_bias, (const float *)diff_dst);
            break;
    }
};

using namespace data_type;

template void ref_deconvolution_fwd_t::compute_bias<f32, f32>(
        const exec_ctx_t &ctx) const;
template void ref_deconvolution_fwd_t::compute_bias<f32, bf16>(
        const exec_ctx_t &ctx) const;
template void ref_deconvolution_fwd_t::compute_bias<bf16, f32>(
        const exec_ctx_t &ctx) const;
template void ref_deconvolution_fwd_t::compute_bias<bf16, bf16>(
        const exec_ctx_t &ctx) const;

template void ref_deconvolution_bwd_weights_t::compute_bias<f32, f32>(
        const exec_ctx_t &ctx) const;
template void ref_deconvolution_bwd_weights_t::compute_bias<f32, bf16>(
        const exec_ctx_t &ctx) const;
template void ref_deconvolution_bwd_weights_t::compute_bias<bf16, bf16>(
        const exec_ctx_t &ctx) const;
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

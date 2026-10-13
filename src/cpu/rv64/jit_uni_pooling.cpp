/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2023 KNS Group LLC (YADRO)
* Copyright 2026 ZTE Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#include <algorithm>
#include <cfloat>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/rv64/jit_uni_pooling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

// ====================== native plain-layout drivers =======================
// Retained rv64-native drivers for the nspc and ncsp (plain) layouts (vectorize
// along OW or strided C; no plain<->blocked transpose). Cover forward inference
// and training and the native backward gather.
namespace {

template <cpu_isa_t isa, data_type_t d_type>
using ncsp_kernel_t = jit_uni_pool_ncsp_kernel_t<isa, d_type>;

struct init_scale_t {
    float init_val;
    float scale_val;
};

init_scale_t compute_init_scale(alg_kind_t alg, int id_start, int id_end,
        int ih_start, int ih_end, int iw_start, int iw_end, dim_t kerD,
        dim_t kerH, dim_t kerW) {
    if (alg == alg_kind::pooling_max) {
        return {-FLT_MAX, 0.0f};
    } else if (alg == alg_kind::pooling_avg_include_padding) {
        return {0.0f, 1.0f / (float)(kerD * kerH * kerW)};
    } else {
        const int count = std::max(0, id_end - id_start)
                * std::max(0, ih_end - ih_start)
                * std::max(0, iw_end - iw_start);
        return {0.0f, (count > 0) ? 1.0f / (float)count : 0.0f};
    }
}

template <typename data_t>
data_t empty_window_value(const jit_pool_conf_t &jpp) {
    if (jpp.alg == alg_kind::pooling_max) {
        float v = (float)nstl::numeric_limits<data_t>::lowest();
        if (jpp.with_relu) {
            if (jpp.relu_alpha == 0.0f)
                v = v < 0.0f ? 0.0f : v;
            else if (v <= 0.0f)
                v *= jpp.relu_alpha;
        }
        return (data_t)v;
    }
    return (data_t)0;
}

template <typename data_t>
jit_uni_pool_ncsp_args_t make_pooling_params(alg_kind_t alg,
        const data_t *src_mb_base, data_t *dst, dim_t mb, dim_t channels,
        dim_t od, dim_t oh, dim_t ow, dim_t outD, dim_t outH, dim_t outW,
        dim_t inD, dim_t inH, dim_t inW, dim_t kerD, dim_t kerH, dim_t kerW,
        dim_t strideD, dim_t strideH, dim_t strideW, dim_t padFront,
        dim_t padTop, dim_t padLeft, dim_t inW_stride, dim_t inD_stride,
        bool with_relu, float relu_alpha) {

    const size_t dst_spatial_base
            = ((((size_t)mb * outD + od) * outH + oh) * outW + ow) * channels;

    const int od_offset = (int)(od * strideD - padFront);
    const int oh_offset = (int)(oh * strideH - padTop);
    const int ow_offset = (int)(ow * strideW - padLeft);

    const int id_start = std::max(od_offset, 0);
    const int ih_start = std::max(oh_offset, 0);
    const int iw_start = std::max(ow_offset, 0);
    const int id_end = std::min(od_offset + (int)kerD, (int)inD);
    const int ih_end = std::min(oh_offset + (int)kerH, (int)inH);
    const int iw_end = std::min(ow_offset + (int)kerW, (int)inW);

    auto iv = compute_init_scale(alg, id_start, id_end, ih_start, ih_end,
            iw_start, iw_end, kerD, kerH, kerW);

    jit_uni_pool_ncsp_args_t p;
    p.src = src_mb_base;
    p.dst = &dst[dst_spatial_base];
    p.channels = channels;
    p.id_start = id_start;
    p.ih_start = ih_start;
    p.iw_start = iw_start;
    p.id_end = id_end;
    p.ih_end = ih_end;
    p.iw_end = iw_end;
    p.inW_stride = inW_stride;
    p.inD_stride = inD_stride;
    p.w_spatial_byte_stride = channels * sizeof(data_t);
    p.init_val = iv.init_val;
    p.scale_val = iv.scale_val;
    p.relu_alpha = relu_alpha;
    p.with_relu = with_relu;
    p.src_vec_byte_stride = (dim_t)sizeof(data_t);
    p.dst_vec_byte_stride = (dim_t)sizeof(data_t);
    return p;
}

// Shared byte offset of the first active lane for the in-kernel binary injector
// (indirect mode). 0 for scalar/per-oc (the rhs origin is already the first
// element); for full-dst it is this output position's channel-0 element offset,
// scaled by sizeof(f32) since src1 is always f32 (the dst may be f16).
template <typename data_t>
dim_t binary_off0(
        const jit_pool_conf_t &jpp, const data_t *dst, const void *p_dst) {
    // Only a full-dst src1 needs a per-position base offset; scalar/per-oc read
    // from a fixed origin. The broadcast category is classified once by the pd
    // gate (jpp.binary_bcast; none when there is no fused binary).
    if (jpp.binary_bcast != pool_binary_bcast_t::full_dst) return 0;
    const dim_t elem_off = static_cast<const data_t *>(p_dst) - dst;
    return elem_off * (dim_t)sizeof(float);
}

// Vectorize along C (strided) for a single ncsp output column.
template <cpu_isa_t isa, data_type_t d_type>
void pool_column_vec_c(const ncsp_kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, dim_t mb, dim_t od, dim_t oh,
        dim_t ow, const void *po_rhs) {
    using data_t = typename prec_traits_t<d_type>::type;
    const dim_t channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padF = jpp.f_pad, padT = jpp.t_pad, padL = jpp.l_pad;

    const size_t spatial_size = (size_t)inD * inH * inW;
    const size_t dst_spatial_size = (size_t)outD * outH * outW;

    const int id_start = std::max((int)(od * strideD - padF), 0);
    const int id_end = std::min((int)(od * strideD - padF + kerD), (int)inD);
    const int ih_start = std::max((int)(oh * strideH - padT), 0);
    const int ih_end = std::min((int)(oh * strideH - padT + kerH), (int)inH);
    const int iw_start = std::max((int)(ow * strideW - padL), 0);
    const int iw_end = std::min((int)(ow * strideW - padL + kerW), (int)inW);

    const dim_t kd_cnt = std::max(0, id_end - id_start);
    const dim_t kh_cnt = std::max(0, ih_end - ih_start);
    const dim_t kw_cnt = std::max(0, iw_end - iw_start);

    data_t *dst_base = dst + (size_t)mb * channels * dst_spatial_size
            + od * outH * outW + oh * outW + ow;

    const bool empty = (kd_cnt == 0 || kh_cnt == 0 || kw_cnt == 0);
    if (empty && !jpp.fuse_eltwise && !jpp.fuse_binary) {
        const data_t ev = empty_window_value<data_t>(jpp);
        for (dim_t c = 0; c < channels; c++)
            dst_base[(size_t)c * dst_spatial_size] = ev;
        return;
    }

    auto iv = compute_init_scale(jpp.alg, id_start, id_end, ih_start, ih_end,
            iw_start, iw_end, kerD, kerH, kerW);

    const data_t *src_base = empty ? src + (size_t)mb * channels * spatial_size
                                   : src + (size_t)mb * channels * spatial_size
                    + (size_t)id_start * inH * inW + (size_t)ih_start * inW
                    + iw_start;

    jit_uni_pool_ncsp_args_t p;
    p.src = src_base;
    p.dst = dst_base;
    p.channels = channels;
    p.id_start = 0;
    p.ih_start = 0;
    p.iw_start = 0;
    p.id_end = kd_cnt;
    p.ih_end = kh_cnt;
    p.iw_end = kw_cnt;
    p.inW_stride = inW;
    p.inD_stride = inH * inW;
    p.w_spatial_byte_stride = (dim_t)sizeof(data_t);
    p.init_val = iv.init_val;
    p.scale_val = iv.scale_val;
    p.relu_alpha = jpp.relu_alpha;
    p.with_relu = jpp.with_relu;
    p.src_vec_byte_stride = (dim_t)(spatial_size * sizeof(data_t));
    p.dst_vec_byte_stride = (dim_t)(dst_spatial_size * sizeof(data_t));
    p.post_op_rhs = po_rhs;
    p.post_op_off0 = binary_off0(jpp, dst, p.dst);
    kernel(&p);
}

template <cpu_isa_t isa, data_type_t d_type>
void pooling_ncsp(const ncsp_kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, const void *po_rhs) {
    using data_t = typename prec_traits_t<d_type>::type;
    const dim_t batch = jpp.mb, channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padF = jpp.f_pad, padT = jpp.t_pad, padL = jpp.l_pad;
    const alg_kind_t alg = jpp.alg;
    const bool with_relu = jpp.with_relu;
    const float relu_alpha = jpp.relu_alpha;

    const size_t spatial_size = (size_t)inD * inH * inW;
    const size_t dst_spatial_size = (size_t)outD * outH * outW;

    const dim_t int_lo = jpp.fuse_binary
            ? 0
            : std::max((padL + strideW - 1) / strideW, (dim_t)0);
    const dim_t int_hi = jpp.fuse_binary
            ? 0
            : ((inW + padL >= kerW)
                              ? std::min(std::max((inW - kerW + padL) / strideW
                                                         + 1,
                                                 int_lo),
                                        outW)
                              : int_lo);

    if (int_lo < int_hi) {
        // Output rows whose H window lies fully inside the input share the
        // window shape (and avg scale), so the kernel pools a block of them
        // per call (n_pos rows, base pointers advanced per row). Rows in the
        // H padding are processed one per call. Blocks are sized to keep at
        // least ~8 work items per thread.
        const dim_t oh_lo = std::min(
                std::max((padT + strideH - 1) / strideH, (dim_t)0), outH);
        const dim_t oh_hi = (inH + padT >= kerH)
                ? std::min(std::max((inH - kerH + padT) / strideH + 1, oh_lo),
                          outH)
                : oh_lo;
        const dim_t n_int_rows = oh_hi - oh_lo;
        const dim_t n_planes = batch * channels * outD;
        const dim_t min_work = 8 * (dim_t)dnnl_get_max_threads();
        const dim_t oh_blk = n_int_rows > 0
                ? utils::div_up(n_int_rows,
                          std::min(n_int_rows,
                                  utils::div_up(min_work, n_planes)))
                : 1;
        const dim_t n_int_blks = utils::div_up(n_int_rows, oh_blk);
        const dim_t n_row_units = oh_lo + n_int_blks + (outH - oh_hi);

        parallel_nd(batch, channels, outD, n_row_units,
                [&](dim_t mb, dim_t c, dim_t od, dim_t u) {
            dim_t oh, n_rows;
            if (u < oh_lo) {
                oh = u;
                n_rows = 1;
            } else if (u < oh_lo + n_int_blks) {
                oh = oh_lo + (u - oh_lo) * oh_blk;
                n_rows = std::min(oh_blk, oh_hi - oh);
            } else {
                oh = oh_hi + (u - oh_lo - n_int_blks);
                n_rows = 1;
            }

            const data_t *src_ch
                    = src + ((size_t)mb * channels + c) * spatial_size;
            data_t *dst_oh = dst
                    + ((size_t)mb * channels + c) * dst_spatial_size
                    + od * outH * outW + oh * outW;

            const int id_start = std::max((int)(od * strideD - padF), 0);
            const int id_end
                    = std::min((int)(od * strideD - padF + kerD), (int)inD);
            const int ih_start = std::max((int)(oh * strideH - padT), 0);
            const int ih_end
                    = std::min((int)(oh * strideH - padT + kerH), (int)inH);

            if ((id_start >= id_end || ih_start >= ih_end) && !jpp.fuse_eltwise
                    && !jpp.fuse_binary) {
                const data_t ev = empty_window_value<data_t>(jpp);
                for (dim_t r = 0; r < n_rows; r++)
                    for (dim_t ow = int_lo; ow < int_hi; ow++)
                        dst_oh[r * outW + ow] = ev;
                return;
            }

            auto iv = compute_init_scale(alg, id_start, id_end, ih_start,
                    ih_end, 0, (int)kerW, kerD, kerH, kerW);

            jit_uni_pool_ncsp_args_t p;
            p.src = src_ch + (int_lo * strideW - padL);
            p.dst = dst_oh + int_lo;
            p.channels = int_hi - int_lo;
            p.id_start = id_start;
            p.ih_start = ih_start;
            p.iw_start = 0;
            p.id_end = id_end;
            p.ih_end = ih_end;
            p.iw_end = kerW;
            p.inW_stride = inW;
            p.inD_stride = inH * inW;
            p.w_spatial_byte_stride = (dim_t)sizeof(data_t);
            p.init_val = iv.init_val;
            p.scale_val = iv.scale_val;
            p.relu_alpha = relu_alpha;
            p.with_relu = with_relu;
            p.src_vec_byte_stride = strideW * (dim_t)sizeof(data_t);
            p.dst_vec_byte_stride = (dim_t)sizeof(data_t);
            p.n_pos = n_rows;
            p.pos_src_byte_stride = strideH * inW * (dim_t)sizeof(data_t);
            p.pos_dst_byte_stride = outW * (dim_t)sizeof(data_t);
            kernel(&p);
        });
    }

    const dim_t lo = std::min(int_lo, outW);
    const dim_t hi = (int_lo < int_hi) ? int_hi : lo;
    const dim_t n_bnd = lo + (outW - hi);
    if (n_bnd > 0) {
        parallel_nd(batch, outD, outH, n_bnd,
                [&](dim_t mb, dim_t od, dim_t oh, dim_t k) {
            const dim_t ow = (k < lo) ? k : hi + (k - lo);
            pool_column_vec_c<isa, d_type>(
                    kernel, jpp, src, dst, mb, od, oh, ow, po_rhs);
        });
    }
}

template <cpu_isa_t isa, data_type_t d_type>
void pooling_ncsp_ow1(const ncsp_kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, const void *po_rhs) {
    parallel_nd(jpp.mb, jpp.od, jpp.oh, [&](dim_t mb, dim_t od, dim_t oh) {
        pool_column_vec_c<isa, d_type>(
                kernel, jpp, src, dst, mb, od, oh, 0, po_rhs);
    });
}

// nspc (f32): vectorize along C. Interior columns handled by the shape-baked
// ur_w-reuse kernel (full ur_w blocks); boundary columns and the sub-ur_w
// interior remainder fall back to one vectorize-C call per column.
template <cpu_isa_t isa, data_type_t d_type>
void pooling_nspc(const ncsp_kernel_t<isa, d_type> &kernel,
        const jit_uni_pool_interior_kernel_t<isa, d_type> &ikernel,
        const jit_pool_conf_t &jpp, const float *src, float *dst,
        const void *po_rhs) {
    const dim_t batch = jpp.mb, channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padFront = jpp.f_pad, padTop = jpp.t_pad, padLeft = jpp.l_pad;
    const alg_kind_t alg = jpp.alg;
    const bool with_relu = jpp.with_relu;
    const float relu_alpha = jpp.relu_alpha;
    const dim_t ur_w = jpp.ur_w;

    const dim_t inW_stride = inW * channels;
    const dim_t inD_stride = inH * inW * channels;

    const dim_t int_lo = std::max((padLeft + strideW - 1) / strideW, (dim_t)0);
    const dim_t int_hi = (inW + padLeft >= kerW)
            ? std::min(std::max((inW - kerW + padLeft) / strideW + 1, int_lo),
                      outW)
            : int_lo;

    parallel_nd(batch, outD, outH, [&](dim_t mb, dim_t od, dim_t oh) {
        const size_t mb_offset = (size_t)mb * (size_t)inD * inD_stride;
        const float *src_mb_base = &src[mb_offset];

        auto call_one = [&](dim_t ow) {
            auto p = make_pooling_params(alg, src_mb_base, dst, mb, channels,
                    od, oh, ow, outD, outH, outW, inD, inH, inW, kerD, kerH,
                    kerW, strideD, strideH, strideW, padFront, padTop, padLeft,
                    inW_stride, inD_stride, with_relu, relu_alpha);
            if ((p.id_start >= p.id_end || p.ih_start >= p.ih_end
                        || p.iw_start >= p.iw_end)
                    && !jpp.fuse_eltwise && !jpp.fuse_binary) {
                const float ev = empty_window_value<float>(jpp);
                float *d = static_cast<float *>(p.dst);
                for (dim_t c = 0; c < channels; c++)
                    d[c] = ev;
                return;
            }
            p.post_op_rhs = po_rhs;
            p.post_op_off0 = binary_off0(jpp, dst, p.dst);
            kernel(&p);
        };

        // Whole (od,oh) row in D/H padding must not run the count-based interior
        // kernel; such rows fall through to the per-column (bge-clamped) path.
        const int od_off = (int)(od * strideD - padFront);
        const int oh_off = (int)(oh * strideH - padTop);
        const int id_start = std::max(od_off, 0);
        const int ih_start = std::max(oh_off, 0);
        const int id_end = std::min(od_off + (int)kerD, (int)inD);
        const int ih_end = std::min(oh_off + (int)kerH, (int)inH);
        const dim_t kd_count = std::max(0, id_end - id_start);
        const dim_t kh_count = std::max(0, ih_end - ih_start);
        const bool row_in_pad = (kd_count == 0 || kh_count == 0);

        const dim_t n_int = (int_lo < int_hi) ? (int_hi - int_lo) : 0;
        // Binary post-ops run only on the channel-vec single-position path.
        const dim_t n_blocks
                = (row_in_pad || jpp.fuse_binary) ? 0 : (n_int / ur_w);
        const dim_t baked_hi = int_lo + n_blocks * ur_w;

        const dim_t lo = std::min(int_lo, outW);
        for (dim_t ow = 0; ow < lo; ++ow)
            call_one(ow);

        if (n_blocks > 0) {
            const int iw_start = (int)(int_lo * strideW - padLeft);
            jit_uni_pool_interior_args_t ip;
            ip.src = src_mb_base + (size_t)id_start * inD_stride
                    + (size_t)ih_start * inW_stride
                    + (size_t)iw_start * channels;
            ip.dst = dst
                    + (((((size_t)mb * outD + od) * outH + oh) * outW + int_lo)
                            * channels);
            ip.channels = channels;
            ip.kh_count = kh_count;
            ip.kd_count = kd_count;
            ip.n_blocks = n_blocks;
            ip.w_stride = (dim_t)channels * sizeof(float);
            ip.inW_stride = inW_stride * (dim_t)sizeof(float);
            ip.inD_stride = inD_stride * (dim_t)sizeof(float);
            const dim_t cnt = kd_count * kh_count * (dim_t)kerW;
            ip.scale_val = (cnt > 0) ? 1.0f / (float)cnt : 0.0f;
            ikernel(&ip);
        }

        for (dim_t ow = baked_hi; ow < int_hi; ++ow)
            call_one(ow);
        for (dim_t ow = int_hi; ow < outW; ++ow)
            call_one(ow);
    });
}

// nspc without the shape-baked interior kernel (f16): one row-unrolled call per
// interior OW run + one call per boundary column.
template <cpu_isa_t isa, data_type_t d_type>
void pooling_nspc_generic(const ncsp_kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, const void *po_rhs) {
    using data_t = typename prec_traits_t<d_type>::type;
    const dim_t batch = jpp.mb, channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padFront = jpp.f_pad, padTop = jpp.t_pad, padLeft = jpp.l_pad;
    const alg_kind_t alg = jpp.alg;
    const bool with_relu = jpp.with_relu;
    const float relu_alpha = jpp.relu_alpha;

    const dim_t inW_stride = inW * channels;
    const dim_t inD_stride = inH * inW * channels;

    // A fused binary is applied on the single-position path only (generate_f16's
    // binary path handles one output column), so collapse the n_pos-batched
    // interior range and route every column through call_one.
    const dim_t int_lo = jpp.fuse_binary
            ? 0
            : std::max((padLeft + strideW - 1) / strideW, (dim_t)0);
    const dim_t int_hi = jpp.fuse_binary ? 0
            : (inW + padLeft >= kerW)
            ? std::min(std::max((inW - kerW + padLeft) / strideW + 1, int_lo),
                      outW)
            : int_lo;

    parallel_nd(batch, outD, outH, [&](dim_t mb, dim_t od, dim_t oh) {
        const size_t mb_offset = (size_t)mb * (size_t)inD * inD_stride;
        const data_t *src_mb_base = &src[mb_offset];

        auto call_one = [&](dim_t ow) {
            auto p = make_pooling_params(alg, src_mb_base, dst, mb, channels,
                    od, oh, ow, outD, outH, outW, inD, inH, inW, kerD, kerH,
                    kerW, strideD, strideH, strideW, padFront, padTop, padLeft,
                    inW_stride, inD_stride, with_relu, relu_alpha);
            if ((p.id_start >= p.id_end || p.ih_start >= p.ih_end
                        || p.iw_start >= p.iw_end)
                    && !jpp.fuse_eltwise && !jpp.fuse_binary) {
                const data_t ev = empty_window_value<data_t>(jpp);
                data_t *d = static_cast<data_t *>(p.dst);
                for (dim_t c = 0; c < channels; c++)
                    d[c] = ev;
                return;
            }
            p.post_op_rhs = po_rhs;
            p.post_op_off0 = binary_off0(jpp, dst, p.dst);
            kernel(&p);
        };

        const dim_t lo = std::min(int_lo, outW);
        for (dim_t ow = 0; ow < lo; ++ow)
            call_one(ow);

        if (int_lo < int_hi) {
            auto p = make_pooling_params(alg, src_mb_base, dst, mb, channels,
                    od, oh, int_lo, outD, outH, outW, inD, inH, inW, kerD, kerH,
                    kerW, strideD, strideH, strideW, padFront, padTop, padLeft,
                    inW_stride, inD_stride, with_relu, relu_alpha);
            p.n_pos = int_hi - int_lo;
            p.pos_src_byte_stride
                    = (dim_t)strideW * channels * (dim_t)sizeof(data_t);
            p.pos_dst_byte_stride = (dim_t)channels * (dim_t)sizeof(data_t);
            kernel(&p);
        }

        const dim_t hi = (int_lo < int_hi) ? int_hi : lo;
        for (dim_t ow = hi; ow < outW; ++ow)
            call_one(ow);
    });
}

// nspc max forward-training: one per-output channel-vec call that tracks the
// argmax into the workspace (contiguous over channels). A fused post-op chain is
// applied to the pooled value before the store (po_rhs = binary origin array).
template <cpu_isa_t isa, data_type_t d_type>
void pooling_train_max_nspc(const ncsp_kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, char *indices,
        const void *po_rhs) {
    using data_t = typename prec_traits_t<d_type>::type;
    const dim_t batch = jpp.mb, channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padF = jpp.f_pad, padT = jpp.t_pad, padL = jpp.l_pad;
    const dim_t inW_stride = inW * channels;
    const dim_t inD_stride = inH * inW * channels;
    const dim_t KHKW = kerH * kerW;
    const size_t ind_sz = types::data_type_size(jpp.ind_dt);

    parallel_nd(batch, outD, outH, outW,
            [&](dim_t mb, dim_t od, dim_t oh, dim_t ow) {
        const data_t *src_mb_base = &src[(size_t)mb * inD * inD_stride];
        auto p = make_pooling_params(alg_kind::pooling_max, src_mb_base, dst,
                mb, channels, od, oh, ow, outD, outH, outW, inD, inH, inW, kerD,
                kerH, kerW, strideD, strideH, strideW, padF, padT, padL,
                inW_stride, inD_stride, /*with_relu=*/false,
                /*relu_alpha=*/0.f);
        // Full-kernel-relative index of the first (clamped) window element, plus
        // the per-row/plane skips of the clamped-off positions.
        const dim_t kd_base = std::max<dim_t>(0, padF - od * strideD);
        const dim_t kh_base = std::max<dim_t>(0, padT - oh * strideH);
        const dim_t kw_base = std::max<dim_t>(0, padL - ow * strideW);
        p.pos_base = kd_base * KHKW + kh_base * kerW + kw_base;
        p.pos_ih_step = kerW - (p.iw_end - p.iw_start);
        p.pos_id_step = KHKW - (p.ih_end - p.ih_start) * kerW;
        const size_t out_flat
                = ((((size_t)mb * outD + od) * outH + oh) * outW + ow)
                * channels;
        p.indices = indices + out_flat * ind_sz;
        p.ws_vec_byte_stride = (dim_t)ind_sz; // nspc: contiguous over channels
        p.post_op_rhs = po_rhs;
        p.post_op_off0 = binary_off0(jpp, dst, p.dst);
        kernel(&p);
    });
}

// ncsp max forward-training: one per-output channel-vec call (vectorize along C,
// strided) that tracks the argmax into the strided ncsp workspace. Mirrors the
// nspc variant but the src/dst/ws channel elements are dst_spatial apart.
template <cpu_isa_t isa, data_type_t d_type>
void pooling_train_max_ncsp(const ncsp_kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        const typename prec_traits_t<d_type>::type *src,
        typename prec_traits_t<d_type>::type *dst, char *indices,
        const void *po_rhs) {
    using data_t = typename prec_traits_t<d_type>::type;
    const dim_t batch = jpp.mb, channels = jpp.c;
    const dim_t outD = jpp.od, outH = jpp.oh, outW = jpp.ow;
    const dim_t inD = jpp.id, inH = jpp.ih, inW = jpp.iw;
    const dim_t kerD = jpp.kd, kerH = jpp.kh, kerW = jpp.kw;
    const dim_t strideD = jpp.stride_d, strideH = jpp.stride_h,
                strideW = jpp.stride_w;
    const dim_t padF = jpp.f_pad, padT = jpp.t_pad, padL = jpp.l_pad;
    const dim_t KHKW = kerH * kerW;
    const size_t ind_sz = types::data_type_size(jpp.ind_dt);
    const size_t spatial_size = (size_t)inD * inH * inW;
    const size_t dst_spatial_size = (size_t)outD * outH * outW;

    parallel_nd(batch, outD, outH, outW,
            [&](dim_t mb, dim_t od, dim_t oh, dim_t ow) {
        const int id_start = std::max((int)(od * strideD - padF), 0);
        const int id_end
                = std::min((int)(od * strideD - padF + kerD), (int)inD);
        const int ih_start = std::max((int)(oh * strideH - padT), 0);
        const int ih_end
                = std::min((int)(oh * strideH - padT + kerH), (int)inH);
        const int iw_start = std::max((int)(ow * strideW - padL), 0);
        const int iw_end
                = std::min((int)(ow * strideW - padL + kerW), (int)inW);
        const dim_t kd_cnt = std::max(0, id_end - id_start);
        const dim_t kh_cnt = std::max(0, ih_end - ih_start);
        const dim_t kw_cnt = std::max(0, iw_end - iw_start);
        const bool empty = (kd_cnt == 0 || kh_cnt == 0 || kw_cnt == 0);

        const data_t *src_base = empty
                ? src + (size_t)mb * channels * spatial_size
                : src + (size_t)mb * channels * spatial_size
                        + (size_t)id_start * inH * inW + (size_t)ih_start * inW
                        + iw_start;

        jit_uni_pool_ncsp_args_t p;
        p.src = src_base;
        p.dst = dst + (size_t)mb * channels * dst_spatial_size
                + od * outH * outW + oh * outW + ow;
        p.channels = channels;
        p.id_start = 0;
        p.ih_start = 0;
        p.iw_start = 0;
        p.id_end = kd_cnt;
        p.ih_end = kh_cnt;
        p.iw_end = kw_cnt;
        p.inW_stride = inW;
        p.inD_stride = inH * inW;
        p.w_spatial_byte_stride = (dim_t)sizeof(data_t);
        p.init_val = -FLT_MAX;
        p.scale_val = 0.0f;
        p.relu_alpha = 0.f;
        p.with_relu = false;
        p.src_vec_byte_stride = (dim_t)(spatial_size * sizeof(data_t));
        p.dst_vec_byte_stride = (dim_t)(dst_spatial_size * sizeof(data_t));
        // Full-kernel-relative index of the first (clamped) window element and
        // the per-row/plane skips of the clamped-off positions.
        const dim_t kd_base = std::max<dim_t>(0, padF - od * strideD);
        const dim_t kh_base = std::max<dim_t>(0, padT - oh * strideH);
        const dim_t kw_base = std::max<dim_t>(0, padL - ow * strideW);
        p.pos_base = kd_base * KHKW + kh_base * kerW + kw_base;
        p.pos_ih_step = kerW - kw_cnt;
        p.pos_id_step = KHKW - kh_cnt * kerW;
        const size_t out_flat
                = od * outH * outW + oh * outW + ow; // channel-0 spatial offset
        p.indices = indices
                + ((size_t)mb * channels * dst_spatial_size + out_flat)
                        * ind_sz;
        p.ws_vec_byte_stride
                = (dim_t)(dst_spatial_size * ind_sz); // ncsp strided
        p.post_op_rhs = po_rhs;
        p.post_op_off0 = binary_off0(jpp, dst, p.dst);
        kernel(&p);
    });
}

// Compile-time selection between the f32 shape-baked interior path and the
// generic nspc path (f16); a struct partial specialization keeps the f32-only
// baked call from being instantiated for f16.
template <cpu_isa_t isa, data_type_t d_type>
struct nspc_exec {
    static void run(const ncsp_kernel_t<isa, d_type> &k,
            const jit_uni_pool_interior_kernel_t<isa, d_type> *ik,
            const jit_pool_conf_t &jpp,
            const typename prec_traits_t<d_type>::type *src,
            typename prec_traits_t<d_type>::type *dst, const void *po_rhs) {
        UNUSED(ik);
        pooling_nspc_generic<isa, d_type>(k, jpp, src, dst, po_rhs);
    }
};

template <cpu_isa_t isa>
struct nspc_exec<isa, data_type::f32> {
    static void run(const ncsp_kernel_t<isa, data_type::f32> &k,
            const jit_uni_pool_interior_kernel_t<isa, data_type::f32> *ik,
            const jit_pool_conf_t &jpp, const float *src, float *dst,
            const void *po_rhs) {
        pooling_nspc<isa, data_type::f32>(k, *ik, jpp, src, dst, po_rhs);
    }
};

// Native gather backward (nspc/ncsp). Parallelises over input positions; for each
// input enumerates the covering outputs (see jit_uni_pool_bwd_contrib_t) and
// calls the kernel, which accumulates their diff_dst rows into diff_src and
// stores it once. max reads the argmax workspace; avg computes 1/num_summands per
// output (uniform include- vs exclude-padding). No data race: each input's
// diff_src is written by exactly one task.
template <cpu_isa_t isa, data_type_t d_type>
void pooling_bwd(const jit_uni_pool_bwd_kernel_t<isa, d_type> &kernel,
        const jit_pool_conf_t &jpp,
        typename prec_traits_t<d_type>::type *diff_src,
        const typename prec_traits_t<d_type>::type *diff_dst, const char *ws) {
    using data_t = typename prec_traits_t<d_type>::type;
    const dim_t MB = jpp.mb, C = jpp.c;
    const dim_t OD = jpp.od, OH = jpp.oh, OW = jpp.ow;
    const dim_t ID = jpp.id, IH = jpp.ih, IW = jpp.iw;
    const dim_t KD = jpp.kd, KH = jpp.kh, KW = jpp.kw;
    const dim_t SD = jpp.stride_d, SH = jpp.stride_h, SW = jpp.stride_w;
    const dim_t padF = jpp.f_pad, padT = jpp.t_pad, padL = jpp.l_pad;
    const alg_kind_t alg = jpp.alg;
    const bool is_max = alg == alg_kind::pooling_max;
    const bool nspc = jpp.tag_kind == jit_pool_tag_kind_t::nspc;
    const size_t ind_sz = is_max ? types::data_type_size(jpp.ind_dt) : 0;

    const size_t in_spatial = (size_t)ID * IH * IW;
    const size_t out_spatial = (size_t)OD * OH * OW;
    const dim_t KHKW = KH * KW;

    const dim_t src_vec = nspc ? (dim_t)sizeof(data_t)
                               : (dim_t)(in_spatial * sizeof(data_t));
    const dim_t dst_vec = nspc ? (dim_t)sizeof(data_t)
                               : (dim_t)(out_spatial * sizeof(data_t));
    const dim_t ws_vec
            = nspc ? (dim_t)ind_sz : (dim_t)(out_spatial * (dim_t)ind_sz);

    parallel_nd(MB, ID, IH, IW, [&](dim_t mb, dim_t id, dim_t ih, dim_t iw) {
        // Channel-0 element offset of this input position.
        const size_t src_off = nspc
                ? ((((size_t)mb * ID + id) * IH + ih) * IW + iw) * C
                : (size_t)mb * C * in_spatial + ((size_t)id * IH + ih) * IW
                        + iw;

        // Loose covering-output ranges (as in the reference); the kd/kh/kw checks
        // below drop the positions whose window does not actually cover (id,ih,iw).
        const dim_t od_l = std::max((id + padF - KD + 1) / SD, (dim_t)0);
        const dim_t oh_l = std::max((ih + padT - KH + 1) / SH, (dim_t)0);
        const dim_t ow_l = std::max((iw + padL - KW + 1) / SW, (dim_t)0);
        const dim_t od_r = std::min((id + padF) / SD + 1, OD);
        const dim_t oh_r = std::min((ih + padT) / SH + 1, OH);
        const dim_t ow_r = std::min((iw + padL) / SW + 1, OW);

        jit_uni_pool_bwd_contrib_t
                contribs[jit_uni_pool_bwd_kernel_t<isa, d_type>::max_contrib];
        int cnt = 0;
        for (dim_t od = od_l; od < od_r; ++od) {
            const dim_t kd = id - od * SD + padF;
            if (kd < 0 || kd >= KD) continue;
            for (dim_t oh = oh_l; oh < oh_r; ++oh) {
                const dim_t kh = ih - oh * SH + padT;
                if (kh < 0 || kh >= KH) continue;
                for (dim_t ow = ow_l; ow < ow_r; ++ow) {
                    const dim_t kw = iw - ow * SW + padL;
                    if (kw < 0 || kw >= KW) continue;
                    const size_t out_off = nspc
                            ? ((((size_t)mb * OD + od) * OH + oh) * OW + ow) * C
                            : (size_t)mb * C * out_spatial
                                    + ((size_t)od * OH + oh) * OW + ow;
                    auto &e = contribs[cnt++];
                    e.diff_dst = diff_dst + out_off;
                    if (is_max) {
                        e.ws = ws + out_off * ind_sz;
                        e.index = (int32_t)(kd * KHKW + kh * KW + kw);
                        e.scale = 0.f;
                    } else {
                        e.ws = nullptr;
                        e.index = 0;
                        dim_t num;
                        if (alg == alg_kind::pooling_avg_include_padding)
                            num = KD * KH * KW;
                        else {
                            const dim_t d0 = std::max(od * SD - padF, (dim_t)0);
                            const dim_t d1 = std::min(od * SD - padF + KD, ID);
                            const dim_t h0 = std::max(oh * SH - padT, (dim_t)0);
                            const dim_t h1 = std::min(oh * SH - padT + KH, IH);
                            const dim_t w0 = std::max(ow * SW - padL, (dim_t)0);
                            const dim_t w1 = std::min(ow * SW - padL + KW, IW);
                            num = (d1 - d0) * (h1 - h0) * (w1 - w0);
                        }
                        e.scale = num > 0 ? 1.0f / (float)num : 0.f;
                    }
                }
            }
        }

        jit_uni_pool_bwd_args_t p;
        p.diff_src = diff_src + src_off;
        p.contribs = contribs;
        p.count = cnt;
        p.channels = C;
        p.src_vec_byte_stride = src_vec;
        p.dst_vec_byte_stride = dst_vec;
        p.ws_vec_byte_stride = ws_vec;
        kernel(&p);
    });
}

} // namespace

// ============================ forward =====================================

template <cpu_isa_t isa>
jit_uni_pooling_fwd_t<isa>::jit_uni_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_pooling_fwd_t<isa>::~jit_uni_pooling_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_pooling_fwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);
    if (pd()->jpp_.use_native) {
        // Native forward-inference kernel (nspc / ncsp). nspc f32 additionally
        // uses the shape-baked interior kernel (f32-only; the d_type==f32 guard
        // lets the f16 path drop it). create_kernel() is CHECK'd here so a
        // codegen failure surfaces as a status instead of a null jit_ker_ that
        // segfaults when called (the interior kernel's fully-unrolled width
        // sweep can overrun a branch's reach for a very large window; the pd
        // gate declines such shapes, this is the backstop).
        CHECK(safe_ptr_assign(ncsp_kernel_,
                new jit_uni_pool_ncsp_kernel_t<isa, d_type>(pd()->jpp_)));
        CHECK(ncsp_kernel_->create_kernel());
        if (d_type == data_type::f32
                && pd()->jpp_.tag_kind == jit_pool_tag_kind_t::nspc) {
            CHECK(safe_ptr_assign(interior_kernel_,
                    new jit_uni_pool_interior_kernel_t<isa, d_type>(
                            pd()->jpp_)));
            CHECK(interior_kernel_->create_kernel());
        }
        return status::success;
    }
    CHECK(safe_ptr_assign(kernel_,
            new jit_uni_pool_kernel_t<isa>(pd()->jpp_, pd()->dst_md())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
void jit_uni_pooling_fwd_t<isa>::execute_forward_blk(const data_t *src,
        data_t *dst, char *indices, const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d = pd()->src_md();
    const memory_desc_wrapper dst_d = pd()->dst_md();
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const int ind_dt_size
            = indices ? (int)types::data_type_size(indices_d.data_type()) : 0;
    const auto &jpp = pd()->jpp_;

    std::vector<const void *> po_rhs
            = binary_injector::prepare_binary_args(jpp.post_ops, ctx);
    const void *const *po_rhs_arr = po_rhs.empty() ? nullptr : po_rhs.data();

    const int c_mult
            = (jpp.tag_kind == jit_pool_tag_kind_t::nspc) ? jpp.c_block : 1;

    auto ker = [&](dim_t n, dim_t b_c, dim_t oh) {
        auto arg = jit_uni_pooling_args_t();
        const int ij = oh * jpp.stride_h;
        const int i_t_over = nstl::max(0, jpp.t_pad - ij);
        const int i_b_over
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        const dim_t c_off = (dim_t)c_mult * b_c;
        arg.src = &src[src_d.blk_off(n, c_off, ih)];
        arg.dst = &dst[dst_d.blk_off(n, c_off, oh)];
        // full-dst binary offset = (arg.dst - dst_orig); dst_orig must be the
        // dst LOGICAL origin (raw base + off_l(0)) so the offset excludes the
        // md's offset0 (blk_off() already includes it). RHS pointers are at
        // their own logical origins, so the offset applies to both tensors.
        arg.dst_orig = dst + dst_d.off_l(0);
        if (indices)
            arg.indices
                    = &indices[indices_d.blk_off(n, c_off, oh) * ind_dt_size];
        arg.kh_padding = jpp.kh - i_t_over - i_b_over;
        arg.kh_padding_shift = i_t_over * jpp.kw;
        arg.kd_padding = jpp.kd;
        arg.kd_padding_shift = 0;
        arg.ker_area_h = (float)(jpp.kh - i_t_over - i_b_over);
        arg.b_c = b_c;
        arg.post_ops_binary_rhs_arg_vec = po_rhs_arr;
        (*kernel_)(&arg);
    };

    // Parallelise over (mb, oh) and loop the channel blocks inside each task.
    // The fine-grained (mb, nb_c, oh) split made one tiny task per channel block,
    // which scaled poorly on multi-core (per-call preamble amortised over little
    // work, plus output false-sharing between adjacent, contiguous channel blocks
    // handed to different threads). One task per (mb, oh) row matches the native
    // kernel's spatial parallelism; a single thread then writes the row's
    // contiguous nspc channels without contention.
    parallel_nd(jpp.mb, (dim_t)jpp.oh, [&](dim_t n, dim_t oh) {
        for (dim_t b_c = 0; b_c < (dim_t)jpp.nb_c; ++b_c)
            ker(n, b_c, oh);
    });
}

template <cpu_isa_t isa>
void jit_uni_pooling_fwd_t<isa>::execute_forward_blk_3d(const data_t *src,
        data_t *dst, char *indices, const exec_ctx_t &ctx) const {
    const memory_desc_wrapper src_d = pd()->src_md();
    const memory_desc_wrapper dst_d = pd()->dst_md();
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const int ind_dt_size
            = indices ? (int)types::data_type_size(indices_d.data_type()) : 0;
    const auto &jpp = pd()->jpp_;

    std::vector<const void *> po_rhs
            = binary_injector::prepare_binary_args(jpp.post_ops, ctx);
    const void *const *po_rhs_arr = po_rhs.empty() ? nullptr : po_rhs.data();

    const int c_mult
            = (jpp.tag_kind == jit_pool_tag_kind_t::nspc) ? jpp.c_block : 1;

    auto ker = [&](dim_t n, dim_t b_c, dim_t od, dim_t oh, dim_t id,
                       int d_t_over, int d_b_over) {
        auto arg = jit_uni_pooling_args_t();
        const int ij = oh * jpp.stride_h;
        const int i_t_over = nstl::max(0, jpp.t_pad - ij);
        const int i_b_over
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        const dim_t c_off = (dim_t)c_mult * b_c;
        arg.src = &src[src_d.blk_off(n, c_off, id, ih)];
        arg.dst = &dst[dst_d.blk_off(n, c_off, od, oh)];
        // dst_orig = dst LOGICAL origin (see execute_forward_blk); excludes
        // offset0 so the full-dst binary offset is not double-counted.
        arg.dst_orig = dst + dst_d.off_l(0);
        if (indices)
            arg.indices = &indices[indices_d.blk_off(n, c_off, od, oh)
                    * ind_dt_size];
        arg.kd_padding = jpp.kd - d_t_over - d_b_over;
        arg.kh_padding = jpp.kh - i_t_over - i_b_over;
        arg.kh_padding_shift = i_t_over * jpp.kw + d_t_over * jpp.kw * jpp.kh;
        arg.kd_padding_shift = (i_t_over + i_b_over) * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh - i_t_over - i_b_over)
                * (float)(jpp.kd - d_t_over - d_b_over);
        arg.b_c = b_c;
        arg.post_ops_binary_rhs_arg_vec = po_rhs_arr;
        (*kernel_)(&arg);
    };

    parallel_nd(jpp.mb, (dim_t)jpp.od, [&](dim_t n, dim_t od) {
        const int ik = od * jpp.stride_d;
        const int d_t_over = nstl::max(0, jpp.f_pad - ik);
        const int d_b_over
                = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad) - jpp.id;
        const dim_t id = nstl::max(ik - jpp.f_pad, 0);
        for (dim_t b_c = 0; b_c < (dim_t)jpp.nb_c; ++b_c)
            for (dim_t oh = 0; oh < jpp.oh; ++oh)
                ker(n, b_c, od, oh, id, d_t_over, d_b_over);
    });
}

template <cpu_isa_t isa>
status_t jit_uni_pooling_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto &jpp = pd()->jpp_;

    if (jpp.use_native) {
        // Native forward inference: nspc (vectorize along C) or ncsp (along
        // OW/C). Both take logical-origin pointers and the single-binary rhs.
        auto src_raw = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
        auto dst_raw = CTX_OUT_MEM(void *, DNNL_ARG_DST);
        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper dst_d(pd()->dst_md());
        const size_t dt_size = types::data_type_size(src_d.data_type());
        src_raw = (const uint8_t *)src_raw + src_d.off_l(0) * dt_size;
        dst_raw = (uint8_t *)dst_raw + dst_d.off_l(0) * dt_size;
        const data_t *src = static_cast<const data_t *>(src_raw);
        data_t *dst = static_cast<data_t *>(dst_raw);

        // Per-binary rhs (src1) origin pointer array for the indirect injector
        // mode (one f32 pointer per binary, in attribute order). The kernel adds
        // the shared per-call offset (post_op_off0 + channel offset).
        std::vector<const void *> po_rhs_vec;
        if (jpp.fuse_binary)
            po_rhs_vec
                    = binary_injector::prepare_binary_args(jpp.post_ops, ctx);
        const void *po_rhs = po_rhs_vec.empty()
                ? nullptr
                : (const void *)po_rhs_vec.data();

        const bool max_train
                = jpp.is_training && jpp.alg == alg_kind::pooling_max;
        if (max_train) {
            // Max training: per-output argmax into the workspace (contiguous over
            // channels for nspc, strided for ncsp).
            auto ws_raw = CTX_OUT_MEM(char *, DNNL_ARG_WORKSPACE);
            const memory_desc_wrapper ws_d(pd()->workspace_md());
            char *ws = ws_raw
                    + ws_d.off_l(0) * types::data_type_size(ws_d.data_type());
            if (jpp.tag_kind == jit_pool_tag_kind_t::nspc)
                pooling_train_max_nspc<isa, d_type>(
                        *ncsp_kernel_, jpp, src, dst, ws, po_rhs);
            else
                pooling_train_max_ncsp<isa, d_type>(
                        *ncsp_kernel_, jpp, src, dst, ws, po_rhs);
        } else if (jpp.tag_kind == jit_pool_tag_kind_t::nspc) {
            nspc_exec<isa, d_type>::run(*ncsp_kernel_, interior_kernel_.get(),
                    jpp, src, dst, po_rhs);
        } else if (jpp.ow == 1) {
            pooling_ncsp_ow1<isa, d_type>(*ncsp_kernel_, jpp, src, dst, po_rhs);
        } else {
            pooling_ncsp<isa, d_type>(*ncsp_kernel_, jpp, src, dst, po_rhs);
        }
        return status::success;
    }

    // Baked kernel: blocked (all fwd) and nspc/blocked training.
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(char *, DNNL_ARG_WORKSPACE);
    if (pd()->ndims() == 5)
        execute_forward_blk_3d(src, dst, ws, ctx);
    else
        execute_forward_blk(src, dst, ws, ctx);
    return status::success;
}

// ============================ backward ====================================

template <cpu_isa_t isa>
jit_uni_pooling_bwd_t<isa>::jit_uni_pooling_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_pooling_bwd_t<isa>::~jit_uni_pooling_bwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_pooling_bwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);
    if (pd()->jpp_.use_native) {
        // Native gather backward kernel (nspc/ncsp). create_kernel() is CHECK'd
        // here so a codegen failure surfaces as a status instead of a null
        // jit_ker_ that segfaults when called.
        CHECK(safe_ptr_assign(bwd_kernel_,
                new jit_uni_pool_bwd_kernel_t<isa, d_type>(pd()->jpp_)));
        CHECK(bwd_kernel_->create_kernel());
        return status::success;
    }
    CHECK(safe_ptr_assign(kernel_,
            new jit_uni_pool_kernel_t<isa>(pd()->jpp_, pd()->diff_dst_md())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
void jit_uni_pooling_bwd_t<isa>::execute_backward_blk(const data_t *diff_dst,
        const char *indices, data_t *diff_src, const exec_ctx_t &ctx) const {
    UNUSED(ctx);
    const memory_desc_wrapper diff_src_d = pd()->diff_src_md();
    const memory_desc_wrapper diff_dst_d = pd()->diff_dst_md();
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const int ind_dt_size
            = indices ? (int)types::data_type_size(indices_d.data_type()) : 0;
    const auto &jpp = pd()->jpp_;
    const int c_mult
            = (jpp.tag_kind == jit_pool_tag_kind_t::nspc) ? jpp.c_block : 1;

    auto get_first_ih = [&](int oh) {
        return nstl::min(nstl::max(oh * jpp.stride_h - jpp.t_pad, 0), jpp.ih);
    };
    auto get_last_ih = [&](int oh) {
        return nstl::min(
                nstl::max(oh * jpp.stride_h - jpp.t_pad + jpp.kh, 0), jpp.ih);
    };

    auto ker = [&](dim_t n, dim_t b_c, dim_t oh) {
        auto arg = jit_uni_pooling_args_t();
        const int ih = get_first_ih(oh);
        const dim_t c_off = (dim_t)c_mult * b_c;
        arg.src = &diff_src[diff_src_d.blk_off(n, c_off, ih)];
        arg.dst = &diff_dst[diff_dst_d.blk_off(n, c_off, oh)];
        if (indices)
            arg.indices
                    = &indices[indices_d.blk_off(n, c_off, oh) * ind_dt_size];

        const int zero_ih_start = (oh == 0) ? 0 : get_last_ih(oh - 1);
        const int zero_ih_end = (oh == jpp.oh - 1) ? jpp.ih : get_last_ih(oh);
        arg.zero_id = 1;
        arg.zero_ih = zero_ih_end - zero_ih_start;
        arg.zero_ptr = &diff_src[diff_src_d.blk_off(n, c_off, zero_ih_start)];

        const int i_t_over = nstl::max(0, jpp.t_pad - (int)oh * jpp.stride_h);
        const int i_b_over
                = nstl::max(jpp.ih, (int)oh * jpp.stride_h + jpp.kh - jpp.t_pad)
                - jpp.ih;
        arg.kh_padding = jpp.kh - i_t_over - i_b_over;
        arg.kh_padding_shift = i_t_over * jpp.kw;
        arg.kd_padding = jpp.kd;
        arg.ker_area_h = (float)(jpp.kh - i_t_over - i_b_over);
        arg.b_c = b_c;
        (*kernel_)(&arg);
    };

    parallel_nd(jpp.mb, (dim_t)jpp.nb_c, [&](dim_t n, dim_t b_c) {
        for (dim_t oh = 0; oh < jpp.oh; ++oh)
            ker(n, b_c, oh);
    });
}

template <cpu_isa_t isa>
void jit_uni_pooling_bwd_t<isa>::execute_backward_blk_3d(const data_t *diff_dst,
        const char *indices, data_t *diff_src, const exec_ctx_t &ctx) const {
    UNUSED(ctx);
    const memory_desc_wrapper diff_src_d = pd()->diff_src_md();
    const memory_desc_wrapper diff_dst_d = pd()->diff_dst_md();
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const int ind_dt_size
            = indices ? (int)types::data_type_size(indices_d.data_type()) : 0;
    const auto &jpp = pd()->jpp_;
    const int c_mult
            = (jpp.tag_kind == jit_pool_tag_kind_t::nspc) ? jpp.c_block : 1;

    auto get_last_ih = [&](int oh) {
        return nstl::min(
                nstl::max(oh * jpp.stride_h - jpp.t_pad + jpp.kh, 0), jpp.ih);
    };
    auto get_last_id = [&](int od) {
        return nstl::min(
                nstl::max(od * jpp.stride_d - jpp.f_pad + jpp.kd, 0), jpp.id);
    };

    auto ker = [&](dim_t n, dim_t b_c, dim_t od, dim_t oh, dim_t id,
                       int d_t_over, int d_b_over, bool zero_inp, int kd) {
        auto arg = jit_uni_pooling_args_t();
        const int ij = oh * jpp.stride_h;
        const int i_t_over = nstl::max(0, jpp.t_pad - ij);
        const int i_b_over
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        const dim_t c_off = (dim_t)c_mult * b_c;
        arg.src = &diff_src[diff_src_d.blk_off(n, c_off, id + kd, ih)];
        arg.dst = &diff_dst[diff_dst_d.blk_off(n, c_off, od, oh)];
        if (indices)
            arg.indices = &indices[indices_d.blk_off(n, c_off, od, oh)
                    * ind_dt_size];

        if (zero_inp) {
            const int zero_id_start = (od == 0) ? 0 : get_last_id(od - 1);
            const int zero_id_end
                    = (od == jpp.od - 1) ? jpp.id : get_last_id(od);
            arg.zero_id = zero_id_end - zero_id_start;
            const int zero_ih_start = (oh == 0) ? 0 : get_last_ih(oh - 1);
            const int zero_ih_end
                    = (oh == jpp.oh - 1) ? jpp.ih : get_last_ih(oh);
            arg.zero_ih = zero_ih_end - zero_ih_start;
            arg.zero_ptr = &diff_src[diff_src_d.blk_off(
                    n, c_off, zero_id_start, zero_ih_start)];
        } else {
            arg.zero_id = 0;
            arg.zero_ih = 0;
        }

        arg.kd_padding = jpp.kd - d_t_over - d_b_over;
        arg.kh_padding = jpp.kh - i_t_over - i_b_over;
        arg.kh_padding_shift = i_t_over * jpp.kw + d_t_over * jpp.kw * jpp.kh
                + kd * jpp.kw * jpp.kh;
        arg.kd_padding_shift = (i_t_over + i_b_over) * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh - i_t_over - i_b_over)
                * (float)(jpp.kd - d_t_over - d_b_over);
        arg.b_c = b_c;
        (*kernel_)(&arg);
    };

    if (jpp.simple_alg) {
        parallel_nd(jpp.mb, (dim_t)jpp.nb_c, [&](dim_t n, dim_t b_c) {
            for (dim_t od = 0; od < jpp.od; ++od) {
                const int ik = od * jpp.stride_d;
                const int d_t_over = nstl::max(0, jpp.f_pad - ik);
                const int d_b_over
                        = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad) - jpp.id;
                const dim_t id = nstl::max(ik - jpp.f_pad, 0);
                for (dim_t oh = 0; oh < jpp.oh; ++oh)
                    ker(n, b_c, od, oh, id, d_t_over, d_b_over, true, 0);
            }
        });
    } else {
        // Non-simple: zero diff_src, then accumulate over kd. For nspc the block
        // channels are strided (C apart), so zero the whole tensor per (mb,id)
        // plane; for blocked each block is contiguous.
        if (jpp.tag_kind == jit_pool_tag_kind_t::nspc) {
            const size_t chunk = (size_t)jpp.ih * jpp.iw * jpp.c;
            parallel_nd(jpp.mb, (dim_t)jpp.id, [&](dim_t n, dim_t id) {
                data_t *ds = &diff_src[diff_src_d.blk_off(n, 0, id, 0)];
                for (size_t i = 0; i < chunk; ++i)
                    ds[i] = (data_t)0;
            });
        } else {
            const size_t chunk = (size_t)jpp.id * jpp.ih * jpp.iw * jpp.c_block;
            parallel_nd(jpp.mb, (dim_t)jpp.nb_c, [&](dim_t n, dim_t b_c) {
                data_t *ds = &diff_src[diff_src_d.blk_off(n, b_c)];
                for (size_t i = 0; i < chunk; ++i)
                    ds[i] = (data_t)0;
            });
        }
        for (dim_t kd = 0; kd < jpp.kd; ++kd) {
            parallel_nd(jpp.mb, (dim_t)jpp.nb_c, [&](dim_t n, dim_t b_c) {
                for (dim_t od = 0; od < jpp.od; ++od) {
                    const int ik = od * jpp.stride_d;
                    const int d_t_over = nstl::max(0, jpp.f_pad - ik);
                    const int d_b_over
                            = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad)
                            - jpp.id;
                    if (kd >= jpp.kd - d_t_over - d_b_over) continue;
                    const dim_t id = nstl::max(ik - jpp.f_pad, 0);
                    for (dim_t oh = 0; oh < jpp.oh; ++oh)
                        ker(n, b_c, od, oh, id, d_t_over, d_b_over, false,
                                (int)kd);
                }
            });
        }
    }
}

template <cpu_isa_t isa>
status_t jit_uni_pooling_bwd_t<isa>::execute_backward(
        const exec_ctx_t &ctx) const {
    const auto &jpp = pd()->jpp_;

    if (jpp.use_native) {
        // Native gather backward (nspc/ncsp). Logical-origin pointers; the driver
        // indexes channel 0 of each input/output position.
        auto dd_raw = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
        auto ds_raw = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
        const memory_desc_wrapper ds_d(pd()->diff_src_md());
        const memory_desc_wrapper dd_d(pd()->diff_dst_md());
        const size_t dt_size = types::data_type_size(ds_d.data_type());
        dd_raw = (const uint8_t *)dd_raw + dd_d.off_l(0) * dt_size;
        ds_raw = (uint8_t *)ds_raw + ds_d.off_l(0) * dt_size;

        const char *ws = nullptr;
        if (jpp.alg == alg_kind::pooling_max) {
            auto ws_raw = CTX_IN_MEM(const char *, DNNL_ARG_WORKSPACE);
            const memory_desc_wrapper ws_d(pd()->workspace_md());
            ws = ws_raw
                    + ws_d.off_l(0) * types::data_type_size(ws_d.data_type());
        }
        pooling_bwd<isa, d_type>(*bwd_kernel_, jpp,
                static_cast<data_t *>(ds_raw),
                static_cast<const data_t *>(dd_raw), ws);
        return status::success;
    }

    // Baked kernel: blocked (and any plain window native declined).
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const char *, DNNL_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    if (pd()->ndims() == 5)
        execute_backward_blk_3d(diff_dst, ws, diff_src, ctx);
    else
        execute_backward_blk(diff_dst, ws, diff_src, ctx);
    return status::success;
}

template struct jit_uni_pooling_fwd_t<v>;
template struct jit_uni_pooling_fwd_t<zvfh>;
template struct jit_uni_pooling_fwd_t<zvfbfwma>;
template struct jit_uni_pooling_bwd_t<v>;
template struct jit_uni_pooling_bwd_t<zvfh>;
template struct jit_uni_pooling_bwd_t<zvfbfwma>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

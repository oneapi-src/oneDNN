/*******************************************************************************
* Copyright 2026 openKylin community
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
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/resampling_utils.hpp"
#include "cpu/rv64/jit_uni_resampling.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

using namespace resampling_utils;

template <cpu_isa_t isa>
jit_uni_resampling_fwd_t<isa>::jit_uni_resampling_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_resampling_fwd_t<isa>::~jit_uni_resampling_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);
    CHECK(safe_ptr_assign(kernel_,
            new jit_uni_resampling_kernel_t<isa, d_type>(pd()->conf_)));
    const auto &conf = pd()->conf_;
    if (conf.w_vec) {
        // Column tables of the OW-vectorized path; the same index and weight
        // math (resampling_utils) as the per-point path below.
        const bool linear = conf.alg == alg_kind::resampling_linear;
        for (int k = 0; k < (linear ? 2 : 1); k++) {
            w_idx_[k].resize(conf.ow);
            if (linear) w_wei_[k].resize(conf.ow);
        }
        for (dim_t ow = 0; ow < conf.ow; ow++) {
            if (!linear) {
                w_idx_[0][ow] = (int32_t)(nearest_idx(ow, conf.ow, conf.iw)
                        * conf.dt_size);
                continue;
            }
            const linear_coeffs_t wc(ow, conf.ow, conf.iw);
            for (int k = 0; k < 2; k++) {
                w_idx_[k][ow] = (int32_t)(wc.idx[k] * conf.dt_size);
                w_wei_[k][ow] = wc.wei[k];
            }
        }
    }
    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_resampling_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const data_t *src0 = src + src_d.off_l(0);
    data_t *dst0 = dst + dst_d.off_l(0);

    const auto &conf = pd()->conf_;
    const dim_t MB = conf.mb, C = conf.c;
    const dim_t ID = conf.id, IH = conf.ih, IW = conf.iw;
    const dim_t OD = conf.od, OH = conf.oh, OW = conf.ow;
    const dim_t ndims = conf.ndims;
    const alg_kind_t alg = conf.alg;

    // A blocked layout is walked one block at a time; a plain one is a single
    // group over all of C. Strides come from conf, so no tag is named here.
    const dim_t B = conf.block > 1 ? conf.block : C;
    const dim_t Cb = utils::div_up(C, B);
    const dim_t src_vec_byte_stride = conf.src_c_stride * conf.dt_size;
    const dim_t dst_vec_byte_stride = conf.dst_c_stride * conf.dt_size;

    // Fused binary (f32 only); the injector reads the rhs origin array and adds
    // the per-point byte offset set below (see jit_resampling_args_t).
    enum { BC_NONE, BC_SCALAR, BC_PER_OC, BC_FULL } bcast = BC_NONE;
    // The rhs origin array comes from the scratchpad booked by the pd, so this
    // path never allocates. post_ops_ok() caps binaries at one.
    const void **po_rhs = nullptr;
    if (conf.fuse_binary) {
        po_rhs = ctx.get_scratchpad_grantor().template get<const void *>(
                memory_tracking::names::key_binary_post_ops_rhs_ptrs);
        const int bin_idx = conf.post_ops.find(primitive_kind::binary);
        const memory_desc_wrapper s1(
                conf.post_ops.entry_[bin_idx].binary.src1_desc);
        const auto *base = static_cast<const char *>(ctx.host_ptr(
                DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_idx) | DNNL_ARG_SRC_1));
        po_rhs[0] = base + s1.off_l(0) * sizeof(float);
        if (s1.nelems(true) == 1)
            bcast = BC_SCALAR;
        else if (s1.nelems() == C)
            bcast = BC_PER_OC;
        else
            bcast = BC_FULL;
    }

    // Element offset of a spatial point.
    auto src_sp_off = [=](dim_t id, dim_t ih, dim_t iw) {
        return id * conf.src_d_stride + ih * conf.src_h_stride
                + iw * conf.src_w_stride;
    };

    if (conf.w_vec) {
        // Plain ncsp: one call per output row, vectorized along OW. A row
        // reads w_rows source rows (D then H corners, in the per-point
        // corner order) with weight d_w * h_w; columns come from the tables.
        parallel_nd(MB, C, OD, OH, [&](dim_t mb, dim_t c, dim_t od, dim_t oh) {
            jit_resampling_args_t args = {};
            args.channels = OW;
            args.dst = dst0 + mb * conf.dst_mb_stride + c * conf.dst_c_stride
                    + od * conf.dst_d_stride + oh * conf.dst_h_stride;
            for (int k = 0; k < 2; k++) {
                args.w_idx[k] = w_idx_[k].data();
                args.w_wei[k] = w_wei_[k].data();
            }
            const data_t *src_base
                    = src0 + mb * conf.src_mb_stride + c * conf.src_c_stride;
            if (alg == alg_kind::resampling_nearest) {
                const dim_t id = ndims >= 5 ? nearest_idx(od, OD, ID) : 0;
                const dim_t ih = ndims >= 4 ? nearest_idx(oh, OH, IH) : 0;
                args.src[0] = src_base + src_sp_off(id, ih, 0);
            } else {
                dim_t d_idx[2] = {0, 0}, h_idx[2] = {0, 0};
                float d_w[2] = {1.f, 0.f}, h_w[2] = {1.f, 0.f};
                int dn = 1, hn = 1;
                if (ndims >= 4) {
                    const linear_coeffs_t hc(oh, OH, IH);
                    h_idx[0] = hc.idx[0];
                    h_idx[1] = hc.idx[1];
                    h_w[0] = hc.wei[0];
                    h_w[1] = hc.wei[1];
                    hn = 2;
                }
                if (ndims >= 5) {
                    const linear_coeffs_t dc(od, OD, ID);
                    d_idx[0] = dc.idx[0];
                    d_idx[1] = dc.idx[1];
                    d_w[0] = dc.wei[0];
                    d_w[1] = dc.wei[1];
                    dn = 2;
                }
                int r = 0;
                for (int i = 0; i < dn; i++)
                    for (int j = 0; j < hn; j++) {
                        args.src[r]
                                = src_base + src_sp_off(d_idx[i], h_idx[j], 0);
                        args.weights[r] = d_w[i] * h_w[j];
                        r++;
                    }
            }
            (*kernel_)(&args);
        });
        return status::success;
    }

    parallel_nd(MB, Cb, OD, OH, OW,
            [&](dim_t mb, dim_t cb, dim_t od, dim_t oh, dim_t ow) {
        jit_resampling_args_t args = {};
        // Valid channels in this group (last blocked group may be partial).
        args.channels = nstl::min(B, C - cb * B);
        args.src_vec_byte_stride = src_vec_byte_stride;
        args.dst_vec_byte_stride = dst_vec_byte_stride;

        data_t *p_dst = dst0 + mb * conf.dst_mb_stride + cb * conf.dst_cb_stride
                + od * conf.dst_d_stride + oh * conf.dst_h_stride
                + ow * conf.dst_w_stride;
        args.dst = p_dst;

        const data_t *src_base
                = src0 + mb * conf.src_mb_stride + cb * conf.src_cb_stride;

        if (alg == alg_kind::resampling_nearest) {
            const dim_t id = ndims >= 5 ? nearest_idx(od, OD, ID) : 0;
            const dim_t ih = ndims >= 4 ? nearest_idx(oh, OH, IH) : 0;
            const dim_t iw = nearest_idx(ow, OW, IW);
            args.src[0] = src_base + src_sp_off(id, ih, iw);
        } else {
            dim_t d_idx[2] = {0, 0}, h_idx[2] = {0, 0}, w_idx[2] = {0, 0};
            float d_w[2] = {1.f, 0.f}, h_w[2] = {1.f, 0.f}, w_w[2] = {1.f, 0.f};
            int dn = 1, hn = 1, wn = 2;

            const linear_coeffs_t wc(ow, OW, IW);
            w_idx[0] = wc.idx[0];
            w_idx[1] = wc.idx[1];
            w_w[0] = wc.wei[0];
            w_w[1] = wc.wei[1];
            if (ndims >= 4) {
                const linear_coeffs_t hc(oh, OH, IH);
                h_idx[0] = hc.idx[0];
                h_idx[1] = hc.idx[1];
                h_w[0] = hc.wei[0];
                h_w[1] = hc.wei[1];
                hn = 2;
            }
            if (ndims >= 5) {
                const linear_coeffs_t dc(od, OD, ID);
                d_idx[0] = dc.idx[0];
                d_idx[1] = dc.idx[1];
                d_w[0] = dc.wei[0];
                d_w[1] = dc.wei[1];
                dn = 2;
            }

            int c = 0;
            for (int i = 0; i < dn; i++)
                for (int j = 0; j < hn; j++)
                    for (int k = 0; k < wn; k++) {
                        args.src[c] = src_base
                                + src_sp_off(d_idx[i], h_idx[j], w_idx[k]);
                        args.weights[c] = d_w[i] * h_w[j] * w_w[k];
                        c++;
                    }
        }

        if (conf.fuse_binary) {
            args.post_op_rhs = po_rhs;
            switch (bcast) {
                case BC_SCALAR: args.post_op_off0 = 0; break;
                case BC_PER_OC:
                    args.post_op_off0 = cb * B * (dim_t)sizeof(float);
                    break;
                case BC_FULL:
                    args.post_op_off0 = (p_dst - dst0) * (dim_t)sizeof(float);
                    break;
                default: break;
            }
        }

        (*kernel_)(&args);

        // A blocked layout zero-pads C: the last block's [valid, block) lanes
        // must read 0, and stay 0 regardless of post-ops (which apply to
        // logical elements only). The kernel wrote only the valid channels,
        // contiguous from p_dst, so zero the tail here.
        if (conf.block > 1 && cb == Cb - 1 && args.channels < B) {
            data_t *pd = static_cast<data_t *>(args.dst);
            for (dim_t c = args.channels; c < B; c++)
                pd[c] = data_t(0.f);
        }
    });

    return status::success;
}

template struct jit_uni_resampling_fwd_t<v>;
template struct jit_uni_resampling_fwd_t<zvfh>;
template struct jit_uni_resampling_fwd_t<zvfbfwma>;

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/ref_io_helper.hpp"
#include "cpu/ref_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

static bool is_padding(const memory_desc_wrapper &md) {
    for (int i = 0; i < md.ndims(); i++)
        if (md.dims()[i] != md.padded_dims()[i]) return true;
    return false;
}

status_t ref_softmax_fwd_t::execute_forward_dense(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    float *scratchpad_int8 = ctx.get_scratchpad_grantor().template get<float>(
            key_softmax_interim_store);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto interim_dt
            = pd()->need_int8_scratchpad() ? data_type::f32 : dst_d.data_type();

    const dim_t ou_stride = pd()->outer_stride();
    const auto is_inplace = (src == dst);
    const auto has_padding = is_padding(dst_d);
    const auto zero_padding = has_padding && !is_inplace;
    const auto axis = pd()->axis();
    const auto axis_size = pd()->axis_size(true);
    const auto axis_blk_size = src_d.padded_dims()[axis] - src_d.dims()[axis];
    const auto src_dt_size = types::data_type_size(pd()->src_md()->data_type);
    const auto dst_dt_size = types::data_type_size(pd()->dst_md()->data_type);

    const int nthr = pd()->nthr_;

    parallel_nd_ext(nthr, outer_size_, [&](int ithr, int, dim_t ou) {
        const void *src_data = reinterpret_cast<const char *>(src)
                + ou * ou_stride * src_dt_size;
        void *dst_data
                = reinterpret_cast<char *>(dst) + ou * ou_stride * dst_dt_size;
        void *interim_ptr = pd()->need_int8_scratchpad()
                ? (scratchpad_int8 + ithr * axis_size)
                : dst_data;

        float space_max = -FLT_MAX;
        float space_denom = 0;
        constexpr int unroll_factor = 32;

// Intel(R) C++ Compiler generates the maxps + shuffle pattern
// for the max search which works faster
#if !defined(__INTEL_COMPILER)
        // The code below makes the compiler generate maxps instruction.
        // rather than maxss, which is generated for the 'else' code path
        auto max_wrapper = [](float a, float b) { return nstl::max(a, b); };
        auto min_wrapper = [](int a, int b) { return nstl::min(a, b); };

        if (channels_ < unroll_factor) {
            float max_val = -FLT_MAX;
            for (int i = 0; i < channels_; i++) {
                max_val = max_wrapper(max_val,
                        io::load_float_value(src_d.data_type(), src_data, i));
            }
            space_max = max_val;
        } else {
            float max_values[unroll_factor];

            for (int i = 0; i < unroll_factor; i++) {
                max_values[i]
                        = io::load_float_value(src_d.data_type(), src_data, i);
            }
            for (int i = unroll_factor; i < channels_; i += unroll_factor) {
                int offset = min_wrapper(i, channels_ - unroll_factor);
                for (int j = 0; j < unroll_factor; j++) {
                    max_values[j] = max_wrapper(max_values[j],
                            io::load_float_value(
                                    src_d.data_type(), src_data, offset + j));
                }
            }
            float max_val = -FLT_MAX;
            for (int i = 0; i < unroll_factor; i++) {
                max_val = max_wrapper(max_val, max_values[i]);
            }
            space_max = max_val;
        }
#else
        for (int c = 0; c < channels_; ++c)
            space_max = nstl::max(space_max,
                    io::load_float_value(src_d.data_type(), src_data, c));
#endif

        // sub + exp + sum
        int tail = channels_ % unroll_factor;
        for (int i = 0; i < channels_ - tail; i += unroll_factor) {
            PRAGMA_OMP_SIMD(reduction(+ : space_denom))
            for (int j = 0; j < unroll_factor; j++) {
                float s = io::load_float_value(
                        src_d.data_type(), src_data, i + j);
                float d = s - space_max;
                if (pd()->is_softmax()) {
                    d = expf(d);
                    space_denom += d;
                } else if (pd()->is_logsoftmax()) {
                    space_denom += expf(d);
                }

                io::store_float_value(interim_dt, d, interim_ptr, i + j);
            }
        }
        for (int i = channels_ - tail; i < channels_; i++) {
            float s = io::load_float_value(src_d.data_type(), src_data, i);
            float d = s - space_max;
            if (pd()->is_softmax()) {
                d = expf(d);
                space_denom += d;
            } else if (pd()->is_logsoftmax()) {
                space_denom += expf(d);
            }
            io::store_float_value(interim_dt, d, interim_ptr, i);
        }

        // scal
        if (pd()->is_softmax()) {
            space_denom = space_denom ? (1.f / space_denom) : 1.f;
        } else if (pd()->is_logsoftmax()) {
            space_denom = logf(space_denom);
        }
        for (int c = 0; c < channels_; ++c) {
            float d = io::load_float_value(interim_dt, interim_ptr, c);
            float val = 0;
            if (pd()->is_softmax()) {
                val = d * space_denom;
            } else if (pd()->is_logsoftmax()) {
                val = d - space_denom;
            }
            val *= src_scales[0];

            // post-ops
            ref_post_ops_t::args_t args;
            args.ctx = &ctx;
            args.l_offset = ou * ou_stride + c;
            args.dst_md = pd()->dst_md();
            ref_post_ops->execute(val, args);

            val *= dst_scales[0];
            io::store_float_value(dst_d.data_type(), val, dst_data, c);
        }
        if (zero_padding) {
            PRAGMA_OMP_SIMD()
            for (int i = 0; i < axis_blk_size; i++)
                io::store_float_value(
                        dst_d.data_type(), 0, dst_data, channels_ + i);
        }
    });
    return status::success;
}

status_t ref_softmax_fwd_t::execute_forward_generic(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    float *scratchpad_int8 = ctx.get_scratchpad_grantor().template get<float>(
            key_softmax_interim_store);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    void *interim_ptr = pd()->need_int8_scratchpad() ? scratchpad_int8 : dst;
    const auto interim_dt
            = pd()->need_int8_scratchpad() ? data_type::f32 : dst_d.data_type();

    const auto is_inplace = (src == dst);
    const auto has_padding = is_padding(dst_d);
    if (has_padding && !is_inplace) {
        if (dst_d.is_dense(true)) {
            const auto res = std::div(static_cast<int>(dst_d.size()), PAGE_4K);
            if (!res.quot)
                std::memset(dst, 0, res.rem);
            else
                parallel_nd(res.quot, [&](dim_t i) {
                    const auto tail = (i + 1 == res.quot) ? res.rem : 0;
                    const auto ptr_dst = reinterpret_cast<unsigned char *>(dst)
                            + i * PAGE_4K;
                    std::memset(ptr_dst, 0, PAGE_4K + tail);
                });
        } else
            // needed for submemory correctness
            ctx.zero_pad_output(DNNL_ARG_DST);
    }

    const auto axis_size = pd()->axis_size(true);
    const int nthr = pd()->nthr_;

    parallel_nd_ext(nthr, outer_size_, [&](int ithr, int, dim_t ou) {
        const dim_t thr_shift = ithr * axis_size;

        float space_max_val = 0, space_denom_val = 0;
        float *space_max = &space_max_val, *space_denom = &space_denom_val;
        if (inner_size_ > 1) {
            space_max = ctx.get_scratchpad_grantor().template get<float>(
                                key_softmax_reduction)
                    + ou * 2 * inner_size_;
            space_denom = space_max + inner_size_;
        }

        utils::array_set(space_max, -FLT_MAX, inner_size_);
        utils::array_set(space_denom, 0, inner_size_);

        for (int in = 0; in < inner_size_; in++) {
            dim_t ou_in_offset = ou * channels_ * inner_size_ + in;

            for (int c = 0; c < channels_; c++) {
                size_t off = src_d.off_l(ou_in_offset + c * inner_size_);
                float s = io::load_float_value(src_d.data_type(), src, off);
                space_max[in] = nstl::max(space_max[in], s);
            }

            for (int c = 0; c < channels_; c++) {
                size_t src_off = src_d.off_l(ou_in_offset + c * inner_size_);
                float s = io::load_float_value(src_d.data_type(), src, src_off);
                float d = s - space_max[in];
                if (pd()->is_softmax()) {
                    d = expf(d);
                    space_denom[in] += d;
                } else if (pd()->is_logsoftmax()) {
                    space_denom[in] += expf(d);
                }
                size_t dst_off = dst_d.off_l(ou_in_offset + c * inner_size_);
                size_t interim_off = pd()->need_int8_scratchpad()
                        ? thr_shift + c
                        : dst_off;
                io::store_float_value(interim_dt, d, interim_ptr, interim_off);
            }

            if (pd()->is_logsoftmax()) {
                space_denom[in] = logf(space_denom[in]);
            }

            for (int c = 0; c < channels_; c++) {
                size_t dst_off = dst_d.off_l(ou_in_offset + c * inner_size_);
                size_t interim_off = pd()->need_int8_scratchpad()
                        ? thr_shift + c
                        : dst_off;
                float d = io::load_float_value(
                        interim_dt, interim_ptr, interim_off);
                float sd = space_denom[in];
                if (pd()->is_softmax()) {
                    d /= sd;
                } else if (pd()->is_logsoftmax()) {
                    d -= sd;
                }
                d *= src_scales[0];

                // post-ops
                ref_post_ops_t::args_t args;
                args.ctx = &ctx;
                args.l_offset = ou_in_offset + c * inner_size_;
                args.dst_md = pd()->dst_md();
                ref_post_ops->execute(d, args);
                d *= dst_scales[0];

                io::store_float_value(dst_d.data_type(), d, dst, dst_off);
            }
        }
    });
    return status::success;
}

// softmax along last physical dimension
status_t ref_softmax_bwd_t::execute_backward_dense(
        const exec_ctx_t &ctx) const {
    auto dst = CTX_IN_MEM(const void *, DNNL_ARG_DST);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const auto ou_stride = pd()->outer_stride();

    parallel_nd(outer_size_, [&](dim_t ou) {
        float sbr = 0;
        size_t off = ou * ou_stride;
        if (pd()->is_softmax()) {
            for (size_t loff = off; loff < off + channels_; ++loff) {
                float d = io::load_float_value(dst_d.data_type(), dst, loff);
                float dd = io::load_float_value(
                        diff_dst_d.data_type(), diff_dst, loff);
                sbr += dd * d;
            }
            for (size_t loff = off; loff < off + channels_; ++loff) {
                float d = io::load_float_value(dst_d.data_type(), dst, loff);
                float dd = io::load_float_value(
                        diff_dst_d.data_type(), diff_dst, loff);
                float val = d * (dd - sbr);
                io::store_float_value(
                        diff_src_d.data_type(), val, diff_src, loff);
            }
        } else if (pd()->is_logsoftmax()) {
            for (size_t loff = off; loff < off + channels_; ++loff) {
                float dd = io::load_float_value(
                        diff_dst_d.data_type(), diff_dst, loff);
                sbr += dd;
            }
            for (size_t loff = off; loff < off + channels_; ++loff) {
                float d = io::load_float_value(dst_d.data_type(), dst, loff);
                float dd = io::load_float_value(
                        diff_dst_d.data_type(), diff_dst, loff);
                float val = dd - expf(d) * sbr;
                io::store_float_value(
                        diff_src_d.data_type(), val, diff_src, loff);
            }
        }
    });
    return status::success;
}

status_t ref_softmax_bwd_t::execute_backward_generic(
        const exec_ctx_t &ctx) const {
    auto dst = CTX_IN_MEM(const void *, DNNL_ARG_DST);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const auto is_inplace = (diff_dst == diff_src);
    const auto has_padding = is_padding(diff_dst_d);
    if (has_padding && !is_inplace) {
        if (diff_dst_d.is_dense(true)) {
            const auto res
                    = std::div(static_cast<int>(diff_dst_d.size()), PAGE_4K);
            if (!res.quot)
                std::memset(diff_src, 0, res.rem);
            else
                parallel_nd(res.quot, [&](dim_t i) {
                    const auto tail = (i + 1 == res.quot) ? res.rem : 0;
                    const auto ptr_dst
                            = reinterpret_cast<unsigned char *>(diff_src)
                            + i * PAGE_4K;
                    std::memset(ptr_dst, 0, PAGE_4K + tail);
                });
        } else
            // needed for submemory correctness
            ctx.zero_pad_output(DNNL_ARG_DIFF_SRC);
    }

    parallel_nd(outer_size_, inner_size_, [&](dim_t ou, dim_t in) {
        dim_t ou_in_offset = ou * channels_ * inner_size_ + in;
        float sbr = 0;
        for (int c = 0; c < channels_; ++c) {
            auto diff_dst_off
                    = diff_dst_d.off_l(ou_in_offset + c * inner_size_);
            float dd = io::load_float_value(
                    diff_dst_d.data_type(), diff_dst, diff_dst_off);
            if (pd()->is_softmax()) {
                auto dst_off = dst_d.off_l(ou_in_offset + c * inner_size_);
                float d = io::load_float_value(dst_d.data_type(), dst, dst_off);
                sbr += dd * d;
            } else if (pd()->is_logsoftmax()) {
                sbr += dd;
            }
        }

        for (int c = 0; c < channels_; ++c) {
            auto diff_dst_off
                    = diff_dst_d.off_l(ou_in_offset + c * inner_size_);
            auto dst_off = dst_d.off_l(ou_in_offset + c * inner_size_);
            float d = io::load_float_value(dst_d.data_type(), dst, dst_off);
            float dd = io::load_float_value(
                    diff_dst_d.data_type(), diff_dst, diff_dst_off);
            float val = 0;
            if (pd()->is_softmax()) {
                val = d * (dd - sbr);
            } else if (pd()->is_logsoftmax()) {
                val = dd - expf(d) * sbr;
            }
            auto diff_src_off
                    = diff_src_d.off_l(ou_in_offset + c * inner_size_);
            io::store_float_value(
                    diff_src_d.data_type(), val, diff_src, diff_src_off);
        }
    });
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

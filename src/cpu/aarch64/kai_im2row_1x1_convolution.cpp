/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
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

#include "cpu/aarch64/kai_im2row_1x1_convolution.hpp"

#include <algorithm>
#include <cstring>

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

bool is_bf16_1x1(const kai_im2row_1x1_convolution_fwd_t::pd_t &pd) {
    return utils::everyone_is(data_type::bf16, pd.src_md()->data_type,
            pd.weights_md()->data_type, pd.dst_md()->data_type);
}

bool has_non_unit_stride(const kai_im2row_1x1_convolution_fwd_t::pd_t &pd) {
    return pd.KSH() > 1 || pd.KSW() > 1;
}

bool im2row_1x1_preferred(const kai_im2row_1x1_convolution_fwd_t::pd_t &pd) {
    if (is_bf16_1x1(pd)) return false;

    return has_non_unit_stride(pd);
}

size_t im2row_size_bytes(
        const kai_im2row_1x1_convolution_fwd_t::pd_t &pd, size_t src_dt_size) {
    return static_cast<size_t>(pd.MB()) * pd.OH() * pd.OW() * pd.IC()
            * src_dt_size;
}

void copy_strided_width(char *&dst, const char *&src, dim_t width_run,
        size_t row_bytes, size_t src_step) {
    for (dim_t i = 0; i < width_run; ++i) {
        std::memcpy(dst, src, row_bytes);
        src += src_step;
        dst += row_bytes;
    }
}

} // namespace

status_t kai_im2row_1x1_convolution_fwd_t::pd_t::init_datapath(
        const engine_t *engine) {
    UNUSED(engine);
    VDISPATCH_CONV(direct_1x1_kernel_ok(), "only supports 1x1 kernels");
    VDISPATCH_CONV(
            direct_1x1_padding_ok(), "only supports zero top and left padding");
    VDISPATCH_CONV(OH() > 0 && OW() > 0, VERBOSE_EMPTY_TENSOR, "dst");
    VDISPATCH_CONV(direct_1x1_output_samples_in_bounds(),
            "only supports output samples inside src bounds");
    VDISPATCH_CONV(im2row_1x1_preferred(*this),
            "1x1 heuristic prefers direct or indirect");

    return status::success;
}

void kai_im2row_1x1_convolution_fwd_t::pd_t::book_datapath_scratchpad(
        memory_tracking::registrar_t &scratchpad, size_t src_dt_size) const {
    const size_t bytes = im2row_size_bytes(*this, src_dt_size);
    scratchpad.book(
            memory_tracking::names::key_conv_gemm_col, bytes, 1, 64, 64);
}

void kai_im2row_1x1_convolution_fwd_t::copy_im2row(
        const kernel_call_args_t &args, char *im2row) const {
    const auto &pd
            = static_cast<const kai_im2row_1x1_convolution_fwd_t::pd_t &>(
                    args.pd);
    const dim_t MB = pd.MB();
    const dim_t IC = pd.IC();
    const dim_t OH = pd.OH();
    const dim_t OW = pd.OW();
    const dim_t KSH = pd.KSH();
    const dim_t KSW = pd.KSW();
    const size_t src_dt_size = args.src_dt_size;
    const size_t row_bytes = static_cast<size_t>(IC) * src_dt_size;
    const dim_t rows = MB * OH * OW;
    const dim_t batch_rows = OH * OW;
    const size_t src_batch_stride_bytes = args.src_batch_stride_bytes;
    const size_t src_h_stride_bytes = args.src_h_stride_bytes;
    const size_t src_col_stride_bytes = args.src_col_stride_bytes;
    const size_t src_h_step_bytes
            = static_cast<size_t>(KSH) * src_h_stride_bytes;
    const size_t src_ow_step_bytes
            = static_cast<size_t>(KSW) * src_col_stride_bytes;
    const char *src_base = args.src_base;

    // Minimum unit of parallelism is the row
    const int active_nthr
            = static_cast<int>(std::min<dim_t>(args.max_threads, pd.OH()));

    // Keep the OpenMP team size stable across im2row and GEMM while limiting
    // the number of threads that do copy work.
    const int team_nthr = active_nthr > 1 ? args.max_threads : 1;

    parallel(team_nthr, [=](int ithr, int) {
        if (ithr >= active_nthr) return;

        dim_t start = 0;
        dim_t end = 0;
        balance211(rows, active_nthr, ithr, start, end);

        dim_t mb = start / batch_rows;
        dim_t rem = start - mb * batch_rows;
        dim_t oh = rem / OW;
        dim_t ow = rem - oh * OW;
        dim_t row = start;
        char *dst = im2row + static_cast<size_t>(row) * row_bytes;
        const char *src = src_base
                + static_cast<size_t>(mb) * src_batch_stride_bytes
                + static_cast<size_t>(oh) * src_h_step_bytes
                + static_cast<size_t>(ow) * src_ow_step_bytes;
        // Unit horizontal stride makes adjacent OW samples adjacent in NHWC.
        // Copy those runs at once while still materializing compact GEMM rows.
        const bool contiguous_width = KSW == 1;

        if (contiguous_width) {
            while (row < end) {
                const dim_t width_run = std::min(end - row, OW - ow);
                const size_t bytes = static_cast<size_t>(width_run) * row_bytes;
                std::memcpy(dst, src, bytes);
                src += static_cast<size_t>(width_run) * src_col_stride_bytes;
                dst += bytes;

                row += width_run;
                ow += width_run;
                if (ow == OW && row < end) {
                    ow = 0;
                    if (++oh == OH) {
                        oh = 0;
                        ++mb;
                    }
                    src = src_base
                            + static_cast<size_t>(mb) * src_batch_stride_bytes
                            + static_cast<size_t>(oh) * src_h_step_bytes;
                }
            }
        } else {
            while (row < end) {
                const dim_t width_run = std::min(end - row, OW - ow);
                copy_strided_width(
                        dst, src, width_run, row_bytes, src_ow_step_bytes);

                row += width_run;
                ow += width_run;
                if (ow == OW && row < end) {
                    ow = 0;
                    if (++oh == OH) {
                        oh = 0;
                        ++mb;
                    }
                    src = src_base
                            + static_cast<size_t>(mb) * src_batch_stride_bytes
                            + static_cast<size_t>(oh) * src_h_step_bytes;
                }
            }
        }
    });
}

status_t kai_im2row_1x1_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd
            = static_cast<const kai_im2row_1x1_convolution_fwd_t::pd_t &>(
                    args.pd);
    auto *im2row = args.scratchpad.get<char>(
            memory_tracking::names::key_conv_gemm_col);
    copy_im2row(args, im2row);

    args.kernel.set_arrays_generic(im2row, static_cast<int>(pd.IC()), 0, 0,
            args.wei_base, args.ld_wei, 0, args.kernel_dst_base, args.ld_dst, 0,
            0, args.bias_base, 0);
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

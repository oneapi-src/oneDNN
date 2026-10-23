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

#include "cpu/aarch64/kai_direct_1x1_convolution.hpp"

#include "common/utils.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

bool uses_flattened_src(const kai_direct_1x1_convolution_fwd_t::pd_t &pd) {
    return pd.KSH() == 1 && pd.KSW() == 1 && pd.OH() == pd.IH()
            && pd.OW() == pd.IW();
}

bool is_bf16_1x1(const kai_direct_1x1_convolution_fwd_t::pd_t &pd) {
    return utils::everyone_is(data_type::bf16, pd.src_md()->data_type,
            pd.weights_md()->data_type, pd.dst_md()->data_type);
}

bool has_non_unit_stride(const kai_direct_1x1_convolution_fwd_t::pd_t &pd) {
    return pd.KSH() > 1 || pd.KSW() > 1;
}

bool direct_1x1_preferred(const kai_direct_1x1_convolution_fwd_t::pd_t &pd) {
    if (is_bf16_1x1(pd)) return false;

    // Match ACL's NHWC 1x1 fast path: unit-stride 1x1 skips im2col, while
    // strided 1x1 materializes im2col/row before GEMM.
    return !has_non_unit_stride(pd);
}

} // namespace

status_t kai_direct_1x1_convolution_fwd_t::pd_t::init_datapath(
        const engine_t *engine) {
    UNUSED(engine);
    VDISPATCH_CONV(direct_1x1_kernel_ok(), "only supports 1x1 kernels");
    VDISPATCH_CONV(
            direct_1x1_padding_ok(), "only supports zero top and left padding");
    VDISPATCH_CONV(OH() > 0 && OW() > 0, VERBOSE_EMPTY_TENSOR, "dst");
    VDISPATCH_CONV(direct_1x1_output_samples_in_bounds(),
            "only supports output samples inside src bounds");
    VDISPATCH_CONV(direct_1x1_preferred(*this),
            "1x1 heuristic prefers im2row or indirect");

    return status::success;
}

unsigned int kai_direct_1x1_convolution_fwd_t::pd_t::gemm_m() const {
    if (uses_flattened_src(*this))
        return static_cast<unsigned int>(OH() * OW());
    return static_cast<unsigned int>(OW());
}

unsigned int kai_direct_1x1_convolution_fwd_t::pd_t::gemm_n_batches() const {
    if (uses_flattened_src(*this)) return static_cast<unsigned int>(MB());
    return static_cast<unsigned int>(OH());
}

unsigned int kai_direct_1x1_convolution_fwd_t::pd_t::gemm_n_multi() const {
    if (uses_flattened_src(*this))
        return kai_convolution_fwd_base_t::pd_t::gemm_n_multi();
    return static_cast<unsigned int>(MB());
}

status_t kai_direct_1x1_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd
            = static_cast<const kai_direct_1x1_convolution_fwd_t::pd_t &>(
                    args.pd);

    if (uses_flattened_src(pd)) {
        args.kernel.set_arrays_generic(args.src_base, args.ld_src,
                args.src_batch_stride, 0, args.wei_base, args.ld_wei, 0,
                args.kernel_dst_base, args.ld_dst, args.dst_batch_stride, 0,
                args.bias_base, 0);
        return status::success;
    }

    args.kernel.set_arrays_generic(args.src_base,
            static_cast<int>(pd.KSW()) * args.ld_src,
            static_cast<int>(pd.KSH()) * args.src_h_stride,
            args.src_batch_stride, args.wei_base, args.ld_wei, 0,
            args.kernel_dst_base, args.ld_dst, args.dst_h_stride,
            args.dst_batch_stride, args.bias_base, 0);
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

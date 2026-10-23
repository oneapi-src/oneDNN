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

#include "cpu/aarch64/kai_indirect_convolution.hpp"

#include "kai/ops/gemm/gemm_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

unsigned int kai_indirect_convolution_fwd_t::pd_t::gemm_m() const {
    return static_cast<unsigned int>(OH() * OW());
}

unsigned int kai_indirect_convolution_fwd_t::pd_t::gemm_k() const {
    return static_cast<unsigned int>(IC());
}

unsigned int kai_indirect_convolution_fwd_t::pd_t::gemm_k_sections() const {
    return static_cast<unsigned int>(KH() * KW());
}

unsigned int kai_indirect_convolution_fwd_t::pd_t::gemm_n_batches() const {
    return static_cast<unsigned int>(MB());
}

status_t kai_indirect_convolution_fwd_t::setup_kernel_arrays(
        const kernel_call_args_t &args) const {
    const auto &pd = static_cast<const kai_indirect_convolution_fwd_t::pd_t &>(
            args.pd);

    args.kernel.set_convolution_parameters(
            kai::ops::ConvolutionParameters {pd.IW(), pd.IH(), pd.IC(), pd.KW(),
                    pd.KH(), pd.OW(), pd.OH(), pd.KSW(), pd.KSH(), pd.KDW() + 1,
                    pd.KDH() + 1, pd.padT(), pd.padL(), 0.f});
    args.kernel.set_arrays_generic(args.src_base, args.ld_src,
            args.src_batch_stride, 0, args.wei_base, args.ld_wei, 0,
            args.kernel_dst_base, args.ld_dst, args.dst_batch_stride, 0,
            args.bias_base, 0);
    return status::success;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

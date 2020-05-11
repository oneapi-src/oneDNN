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

#ifndef CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_NHWC_HPP
#define CPU_X64_LRN_JIT_AVX512_COMMON_LRN_BWD_NHWC_HPP

#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_base.hpp"
#include "cpu/x64/lrn/jit_avx512_common_lrn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;
using namespace Xbyak;
using namespace Xbyak::util;

template <data_type_t d_type>
class jit_avx512_common_lrn_kernel_bwd_nhwc_t
    : public jit_avx512_common_lrn_kernel_bwd_t<d_type> {
public:
    jit_avx512_common_lrn_kernel_bwd_nhwc_t(unsigned C, float alpha, float beta,
            int local_size, void *code_ptr = nullptr,
            size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_bwd_nhwc_f)

private:
    void set_up_ker_params();
    void execute_compute_loop(unsigned C);
    void compute_loop(across_version version, int loop_size_param = 1);
    void compute(int loop_size_param);
    void increment_loop_params(std::size_t offset);
    void load_compute_data(across_version version, int loop_size_param);
    void store_compute_data(int loop_size_param);

    const std::vector<int> tmp_mask_prev_;
    const std::vector<int> tmp_mask_next_;
    const Reg64 mask_ = r11;
    const Reg64 blockC_ = r12;

    const int half_ls_;
};

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

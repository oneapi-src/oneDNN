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

#ifndef CPU_JIT_SIMPLE_LAYER_NORMALIZATION_KERNELS_HPP
#define CPU_JIT_SIMPLE_LAYER_NORMALIZATION_KERNELS_HPP

#include "simple_layer_normalization_kernels.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace lnorm_utils {

statistics_kernel_t *jit_statistics_kernel_create(
        const layer_normalization_pd_t *pd);

data_kernel_t *jit_data_kernel_create(const layer_normalization_pd_t *pd);

diff_ss_kernel_t *jit_diff_ss_kernel_create(const layer_normalization_pd_t *pd);

diff_data_kernel_t *jit_diff_data_kernel_create(
        const layer_normalization_pd_t *pd);

} // namespace lnorm_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

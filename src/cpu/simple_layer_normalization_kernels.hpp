/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef CPU_SIMPLE_LAYER_NORMALIZATION_KERNELS_HPP
#define CPU_SIMPLE_LAYER_NORMALIZATION_KERNELS_HPP

#include "cpu/cpu_layer_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace lnorm_utils {

struct stat_and_data_kernel_t {
    static stat_and_data_kernel_t *create(const layer_normalization_pd_t *pd);
    virtual ~stat_and_data_kernel_t() = default;

    virtual void operator()(const void *src, void *dst, const float *scale,
            const float *shift, float *mean, float *var,
            const size_t block_size) const;

    virtual status_t create_kernel() { return status::success; }

protected:
    stat_and_data_kernel_t(const layer_normalization_pd_t *pd) : pd_(pd) {}

    const layer_normalization_pd_t *pd_;
};

struct diff_ss_kernel_t {
    static diff_ss_kernel_t *create(const layer_normalization_pd_t *pd);
    virtual ~diff_ss_kernel_t() = default;

    virtual void operator()(const void *src, const void *diff_dst,
            float *diff_gamma, float *diff_beta, const float *mean,
            const float *var, float *const inv_sqrtvar,
            const size_t block_size) const;

    virtual status_t create_kernel() { return status::success; }

protected:
    diff_ss_kernel_t(const layer_normalization_pd_t *pd) : pd_(pd) {}

    const layer_normalization_pd_t *pd_;
};

struct diff_data_kernel_t {
    static diff_data_kernel_t *create(const layer_normalization_pd_t *pd);
    virtual ~diff_data_kernel_t() = default;

    virtual void operator()(const void *src, const void *diff_dst,
            void *diff_src, const float *ss, const float *mean,
            float *const inv_sqrtvar, const size_t block_size) const;

    virtual status_t create_kernel() { return status::success; }

protected:
    diff_data_kernel_t(const layer_normalization_pd_t *pd) : pd_(pd) {}

    const layer_normalization_pd_t *pd_;
};

} // namespace lnorm_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_BNORM_BNORM_UTILS_HPP
#define GPU_INTEL_OCL_BNORM_BNORM_UTILS_HPP

#include "common/batch_normalization_pd.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/utils.hpp"

#include <string.h>

#ifndef __SHORT_FILE_NAME__
#define __SHORT_FILE_NAME__ \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#ifndef PRINTHEAD
#define PRINTHEAD __SHORT_FILE_NAME__, __FUNCTION__, __LINE__
#endif

#ifndef DPRINT
#define DPRINT(fmt, ...) \
    do { \
        if (get_verbose_dev_mode(verbose_t::debuginfo) >= 3) { \
            printf(fmt, __VA_ARGS__); \
            fflush(0); \
        } \
    } while (0)
#endif
#ifndef DPRINT_PARAMS
#define DPRINT_PARAMS(fmt, ...) \
    do { \
        if (get_verbose_dev_mode(verbose_t::debuginfo) >= 2) { \
            printf(fmt, __VA_ARGS__); \
            fflush(0); \
        } \
    } while (0)
#endif
#ifndef DPRINT_MODEL
#define DPRINT_MODEL(fmt, ...) \
    do { \
        if (get_verbose_dev_mode(verbose_t::debuginfo) >= 4) { \
            printf(fmt, __VA_ARGS__); \
            fflush(0); \
        } \
    } while (0)
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {
namespace bn_utils {

constexpr int aux_init_stage = 1;
constexpr int aux_finalize_stage = 0;
constexpr int aux_use_one_pass = 1;
constexpr int aux_use_regular = 0;
constexpr int aux_fwd = 1;
constexpr int aux_bwd = 0;

namespace kernel_id {
constexpr size_t norm_fwd = 0;
constexpr size_t calc_mean = 1;
constexpr size_t calc_var = 2;
constexpr size_t reduce_fwd_reg = 3;
constexpr size_t calc_mean_var = 4;
constexpr size_t reduce_fwd_1pass = 5;
constexpr size_t reduce_aux = 6;
constexpr size_t norm_bwd = 7;
constexpr size_t calc_stat = 8;
constexpr size_t reduce_stat = 9;
constexpr size_t norm_fwd_buff = 10;
constexpr size_t norm_bwd_buff = 11;
constexpr size_t calc_mean_buff = 12;
constexpr size_t calc_var_buff = 13;
constexpr size_t calc_mean_var_buff = 14;
constexpr size_t calc_stat_buff = 15;
} // namespace kernel_id

float get_ss_utilization(
        int max_ss, const compute::range_t &gws, const compute::range_t &lws);
float get_thr_utilization(int eu_count, int threads_per_eu, int sg_size,
        const compute::range_t &gws);
void init_flags_lookup_table(
        std::string &flags, const batch_normalization_pd_t *pd);
void init_conf_basic(bnorm_conf_t &conf, const batch_normalization_pd_t *pd);
std::string get_prb_desc_str(const batch_normalization_pd_t *pd);

} // namespace bn_utils
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

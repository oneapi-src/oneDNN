/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_BATCH_NORMALIZATION_UTILS_HPP
#define GPU_BATCH_NORMALIZATION_UTILS_HPP

#include "common/batch_normalization_pd.hpp"
#include "gpu/primitive_conf.hpp"
#include "gpu/utils.hpp"

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
namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace bn_utils {

float get_ss_utilization(int max_ss, const size_t *gws, const size_t *lws);
float get_thr_utilization(
        int eu_count, int threads_per_eu, int sg_size, const size_t *gws);
void init_flags_lookup_table(
        std::string &flags, const batch_normalization_pd_t *pd);
void init_conf_basic(bnorm_conf_t &conf, const batch_normalization_pd_t *pd);
std::string get_prb_desc_str(const batch_normalization_pd_t *pd);

} // namespace bn_utils
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

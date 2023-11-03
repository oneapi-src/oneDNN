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
#define __SHORT_FILE_NAME__ ((strrchr(__FILE__, '/') ?: __FILE__ - 1) + 1)
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

struct hw_params_t {
    compute::gpu_arch_t gpu_arch;
    int eu_count;
    int threads_per_eu;
    int max_lws;
    int eus_per_ss;
    int max_ss;
    int max_slm_size;
    compute::compute_engine_t *compute_engine;
    float HBM_bw;
    float L3_bw;
    float L3_size;
    float host_overheads_per_kernel;
};

float get_ss_utilization(int max_ss, const size_t *gws, const size_t *lws);
float get_thr_utilization(
        int eu_count, int threads_per_eu, int sg_size, const size_t *gws);
float get_used_ss_thr_utilization(hw_params_t &hw_params, int sg_size,
        const size_t *gws, const size_t *lws);
void init_flags_lookup_table(
        std::string &flags, const batch_normalization_pd_t *pd);
void init_conf_basic(bnorm_conf_t &conf, const batch_normalization_pd_t *pd);
std::string get_prb_desc_str(const batch_normalization_pd_t *pd);
void init_hw_params(hw_params_t &hw_params, engine_t *engine);

enum data_location_t { HBM, L3, SLM };
enum kernel_kind_t {
    default_fwd_ker,
    calc_mean_ker,
    calc_var_ker,
    calc_mean_var_ker,
    reduce_stats_fwd_ker,
    reduce_mean_var_ker,
    reduce_aux_init_ker,
    reduce_aux_finalize_ker,
    default_bwd_ker,
    calc_stats_ker,
    reduce_stats_bwd_ker
};

struct kernel_desc_t {
    kernel_kind_t kernel;
    int ncalls = 0;
    size_t input_nbytes = 0;
    size_t output_nbytes = 0;
    data_location_t input_location = data_location_t::HBM;
    data_location_t output_location = data_location_t::HBM;
    // estimations
    int num_wgs;
    float used_ss_thr_util = 0.0f;
    float ss_util = 0.0f;
    float time_ns = 0.0f;
};
struct model_params_t {
    int use_fused_atomics_reduction;
    int ic_block;
    int stat_sp_block;
    int vect_size;
    std::vector<kernel_desc_t> kernel_descs;
};

std::string get_str_kernel_name(const kernel_kind_t &kernel);
std::string get_str_data_location(const data_location_t &loc);
void dump_kernel_descriptor(kernel_desc_t &desc);
} // namespace bn_utils
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

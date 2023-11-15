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
#ifndef GPU_BATCH_NORMALIZATION_MODEL_HPP
#define GPU_BATCH_NORMALIZATION_MODEL_HPP

#include "gpu/ocl/bnorm/nhwc_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace bn_model {

struct hw_params_t {
    engine_t *engine;
    compute::gpu_arch_t gpu_arch;
    int eu_count;
    int threads_per_eu;
    size_t max_lws;
    int eus_per_ss;
    int max_ss;
    int max_slm_size;
    float HBM_bw;
    float L3_bw;
    float L3_size;
    float host_overheads_per_kernel;
};

enum data_location_t { HBM, L3, SLM };
struct kernel_desc_t {
    kernel_kind_t kernel;
    int ncalls = 0;
    size_t input_nbytes = 0;
    size_t output_nbytes = 0;
    data_location_t input_location = data_location_t::HBM;
    data_location_t output_location = data_location_t::HBM;
    // estimations
    size_t num_wgs;
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
void init_hw_params(hw_params_t &hw_params, engine_t *engine);
float get_used_ss_thr_utilization(hw_params_t &hw_params, int sg_size,
        const size_t *gws, const size_t *lws);
std::string to_string(const kernel_kind_t &kernel);
std::string to_string(const data_location_t &loc);
void dump_kernel_descriptor(kernel_desc_t &desc);

std::string to_string(const nhwc_bnorm_params_t &conf);
float get_vectorization_factor(const int vect_size, const data_type_t dt);
int get_ncalls(model_params_t &p, const nhwc_bnorm_params_t &conf,
        kernel_kind_t kernel);
size_t get_kernel_input_size(const model_params_t &p,
        const nhwc_bnorm_params_t &conf, const kernel_desc_t &desc);
size_t get_kernel_output_size(const model_params_t &p,
        const nhwc_bnorm_params_t &conf, const kernel_desc_t &desc);
void get_expected_data_location(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params, kernel_desc_t &desc);
float solve_2p_line(const float x, const float xa, const float xb,
        const float ya, const float yb);
float solve_2pieces_linear_function(const float x, const float x0,
        const float x1, const float x2, const float y0, const float y1,
        const float y2);
float get_ss_utilization_factor(const float util);
float get_thr_utilization_factor(const float ss_util, const float thr_util,
        const data_location_t location, const compute::gpu_arch_t gpu_arch);
void get_estimated_kernel_time(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params, kernel_desc_t &desc);
void init_ker_desc(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params, kernel_desc_t &desc,
        const kernel_kind_t kernel);
void init_kernel_descriptors(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params);
void dump_params(std::vector<model_params_t> &params);

status_t get_estimated_hw_utilization(model_params_t &p,
        nhwc_bnorm_params_t &conf, hw_params_t &hw_params, kernel_desc_t &desc);
status_t make_kernel_perf_estimation(model_params_t &p,
        nhwc_bnorm_params_t &conf, kernel_desc_t &desc, hw_params_t &hw_params);
status_t make_perf_estimations(
        model_params_t &p, nhwc_bnorm_params_t &conf, hw_params_t &hw_params);

} // namespace bn_model
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

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
#ifndef GPU_INTEL_OCL_BNORM_BNORM_MODEL_HPP
#define GPU_INTEL_OCL_BNORM_BNORM_MODEL_HPP

#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/ocl/bnorm/nhwc_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {
namespace bn_model {

enum data_location_t { HBM, L3, SLM };
enum mem_operation_t { read, write, atomic };
enum appr_alg_t { linear, ln };

constexpr int def_reduction_vect = 4;
constexpr float max_appr_ss_util = 8;
constexpr float max_appr_thr_util = 1;

struct hw_params_t {
    impl::engine_t *engine;
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

struct kernel_desc_t {
    kernel_kind_t kernel;
    int ncalls = 0;
    size_t input_nbytes = 0;
    size_t output_nbytes = 0;
    data_location_t input_location = data_location_t::HBM;
    data_location_t output_location = data_location_t::HBM;
    bool reusable_version = false;
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
void init_hw_params(hw_params_t &hw_params, impl::engine_t *engine);
float get_used_ss_thr_utilization(hw_params_t &hw_params, int sg_size,
        const compute::range_t &gws, const compute::range_t &lws);
std::string to_string(const kernel_kind_t &kernel);
std::string to_string(const data_location_t &loc);
void dump_kernel_descriptor(kernel_desc_t &desc);

std::string to_string(const nhwc_bnorm_params_t &conf);
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
void get_estimated_kernel_time(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params, kernel_desc_t &desc);
void init_kernel_descriptors(model_params_t &p, nhwc_bnorm_params_t &conf,
        const hw_params_t &hw_params, bool reusable = false);
void dump_params(std::vector<model_params_t> &params);
int get_nhwc_vect_size(int ic, int max_vect_size, int simd = 16);
int get_nhwc_sp_block_size(
        int sp, int ic_dim, int eu_count, int threads_per_eu, int simd = 16);
dim_t get_nhwc_calc_stat_ic(dim_t ic, int ic_block, int sg_size);

status_t get_estimated_hw_utilization(model_params_t &p,
        nhwc_bnorm_params_t &conf, hw_params_t &hw_params, kernel_desc_t &desc);
status_t make_kernel_perf_estimation(model_params_t &p,
        nhwc_bnorm_params_t &conf, kernel_desc_t &desc, hw_params_t &hw_params);
status_t make_perf_estimations(
        model_params_t &p, nhwc_bnorm_params_t &conf, hw_params_t &hw_params);

status_t get_params_by_model(nhwc_bnorm_params_t &conf,
        const batch_normalization_pd_t *pd, hw_params_t &hw_params,
        bool reusable_version);

struct appr_formula_t {
    float a;
    float b;
    appr_alg_t alg;
};

} // namespace bn_model
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

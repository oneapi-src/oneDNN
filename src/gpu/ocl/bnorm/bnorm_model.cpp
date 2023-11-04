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
#include "gpu/ocl/bnorm/bnorm_model.hpp"
#include "gpu/ocl/bnorm/bnorm_utils.hpp"
#include "gpu/ocl/bnorm/nhwc_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace bn_model {
using namespace dnnl::impl::utils;
using namespace dnnl::impl::gpu::ocl::bn_utils;

void init_hw_params(hw_params_t &hw_params, engine_t *engine) {
    const bool large_grf_mode = false;
    auto *compute_engine = downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();
    hw_params.gpu_arch = gpu_arch;
    hw_params.eu_count = compute_engine->device_info()->eu_count();
    hw_params.threads_per_eu
            = compute::device_info_t::threads_per_eu(gpu_arch, false);
    hw_params.max_lws
            = compute_engine->device_info()->max_wg_size(large_grf_mode);
    hw_params.eus_per_ss = compute_engine->device_info()->max_eus_per_wg();
    hw_params.max_ss = div_up(hw_params.eu_count, hw_params.eus_per_ss);
    hw_params.max_slm_size = compute::device_info_t::max_slm_size(gpu_arch);
    hw_params.compute_engine = compute_engine;

    if (hw_params.gpu_arch == compute::gpu_arch_t::xe_hpg) {
        hw_params.HBM_bw = getenv_int("GBW", 400); //GBs
        hw_params.L3_size = 16 * (2 << 19); //Bytes
        hw_params.L3_bw = 2000; //GBs
        hw_params.host_overheads_per_kernel = 8000; // ns
    } else if (hw_params.gpu_arch >= compute::gpu_arch_t::xe_hpc) {
        hw_params.HBM_bw = 1000; //GBs
        hw_params.L3_size = 192 * (2 << 19); //Bytes
        hw_params.L3_bw = 3000; //GBs
        hw_params.host_overheads_per_kernel = 6000; // ns
    } else {
        assert(!"not supported");
    }
}

float get_used_ss_thr_utilization(hw_params_t &hw_params, int sg_size,
        const size_t *gws, const size_t *lws) {
    const size_t gws_size = gws[0] * gws[1] * gws[2];
    const size_t lws_size = lws[0] * lws[1] * lws[2];
    const int num_thrs_generated = gws_size / sg_size;
    const int num_wgs = gws_size / lws_size; // == ss used
    // TODO: considering case when several wg are running on the same ss
    return (float)num_thrs_generated
            / std::min(
                    num_wgs * hw_params.eus_per_ss * hw_params.threads_per_eu,
                    hw_params.eu_count * hw_params.threads_per_eu);
}

std::string get_str_kernel_name(const kernel_kind_t &kernel) {
    std::string kernel_name;
    if (kernel == calc_mean_ker) {
        kernel_name = "calc_mean";
    } else if (kernel == calc_var_ker) {
        kernel_name = "calc_var";
    } else if (kernel == calc_mean_var_ker) {
        kernel_name = "calc_mean_var";
    } else if (kernel == calc_stats_ker) {
        kernel_name = "calc_stat";
    } else if (kernel == reduce_stats_fwd_ker) {
        kernel_name = "reduce_stats_fwd";
    } else if (kernel == reduce_mean_var_ker) {
        kernel_name = "reduce_mean_var";
    } else if (kernel == reduce_stats_bwd_ker) {
        kernel_name = "reduce_stats_bwd";
    } else if (kernel == reduce_aux_init_ker) {
        kernel_name = "reduce_aux_init";
    } else if (kernel == reduce_aux_finalize_ker) {
        kernel_name = "reduce_aux_finalize";
    } else if (kernel == default_fwd_ker) {
        kernel_name = "default_fwd";
    } else if (kernel == default_bwd_ker) {
        kernel_name = "default_bwd";
    } else {
        assert(!"Not expected");
    }
    return kernel_name;
}

std::string get_str_data_location(const data_location_t &loc) {
    std::string str_loc;
    if (loc == L3) {
        str_loc = "L3";
    } else if (loc == HBM) {
        str_loc = "HBM";
    } else if (loc == SLM) {
        str_loc = "SLM";
    } else {
        assert(!"Not expected");
    }
    return str_loc;
}

void dump_kernel_descriptor(kernel_desc_t &desc) {
    DPRINT("%s:%s:%d kernel desc:  %s : ncalls = %d : nbytes = %ld %ld : "
           "location = "
           "%s %s\n",
            PRINTHEAD, get_str_kernel_name(desc.kernel).c_str(), desc.ncalls,
            desc.input_nbytes, desc.output_nbytes,
            get_str_data_location(desc.input_location).c_str(),
            get_str_data_location(desc.output_location).c_str());
}

std::string get_params_str(const nhwc_bnorm_params_t &conf) {
    std::string s;
#define STR_PARAM(p) \
    s += std::to_string(conf.p##_param().is_overridden()) + ","; \
    s += std::to_string((int)conf.p()) + ","

    STR_PARAM(use_fused_atomics_reduction);
    STR_PARAM(max_vect_size);
    s += std::to_string((int)conf.vect_size) + ",";
    STR_PARAM(ic_block);
    s += std::to_string((int)conf.sp) + ",";
    STR_PARAM(stat_sp_block);
    STR_PARAM(update_sp_block);
    STR_PARAM(update_sp_unroll);
    s += std::to_string((int)conf.sub_group_size) + ",";
    s += std::to_string(conf.expected_time_ms);
    return s;
#undef STR_PARAM
}

// how short vector can increase r/w expected time
float get_vectorization_factor(const int vect_size, const data_type_t dt) {
    if (dt == data_type::f16 || data_type::bf16) {
        switch (vect_size) {
            case 1: return 4;
            case 2: return 1.5;
            case 4: return 1.3;
            case 8:
            default: return 1;
        }
    } else {
        switch (vect_size) {
            case 1: return 4;
            case 2: return 1.3;
            case 4:
            case 8:
            default: return 1;
        }
    }
}

} // namespace bn_model
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

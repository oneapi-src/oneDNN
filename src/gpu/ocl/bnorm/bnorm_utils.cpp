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
#include "gpu/ocl/bnorm/bnorm_utils.hpp"
#include "gpu/ocl/bnorm/bnorm_lookup_table.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace bn_utils {
using namespace dnnl::impl::utils;

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
float get_ss_utilization(int max_ss, const size_t *gws, const size_t *lws) {
    const size_t gws_size = gws[0] * gws[1] * gws[2];
    const size_t lws_size = lws[0] * lws[1] * lws[2];
    const size_t used_ss = utils::div_up(gws_size, lws_size);
    return (float)used_ss / max_ss;
}
float get_thr_utilization(
        int eu_count, int threads_per_eu, int sg_size, const size_t *gws) {
    const size_t gws_size = gws[0] * gws[1] * gws[2];
    return ((float)gws_size / sg_size) / (eu_count * threads_per_eu);
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

// Init basic fields of conf structure
void init_conf_basic(bnorm_conf_t &conf, const batch_normalization_pd_t *pd) {
    using namespace dnnl::impl::format_tag;

    const batch_normalization_desc_t &bd = *pd->desc();
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    const int ndims = data_mdw.ndims();

    conf.data_type = data_mdw.data_type();
    conf.elsz = dnnl::impl::types::data_type_size(conf.data_type);

    conf.ndims = ndims;
    conf.mb = data_mdw.dims()[0];
    conf.ic = data_mdw.dims()[1];
    conf.id = (ndims == 5) ? data_mdw.dims()[2] : 1;
    conf.ih = (ndims == 3) ? 1 : data_mdw.dims()[ndims - 2];
    conf.iw = data_mdw.dims()[ndims - 1];

    conf.is_forward = pd->is_fwd();
    conf.is_backward = !pd->is_fwd();

    conf.use_scale = pd->use_scale();
    conf.use_shift = pd->use_shift();
    conf.save_stats = pd->is_training();
    conf.is_training = pd->is_training();
    conf.fuse_norm_add_relu = pd->fuse_norm_add_relu();
    conf.fuse_norm_relu = pd->fuse_norm_relu() || pd->fuse_norm_add_relu();
    conf.calculate_stats = !pd->stats_is_src();
    conf.with_relu = pd->with_relu_post_op(pd->is_training());
    conf.relu_negative_slope = conf.with_relu ? pd->alpha() : 0.f;
    conf.eps = bd.batch_norm_epsilon;
    conf.calculate_diff_stats = !pd->use_global_stats();
    conf.diff_scale = (pd->use_scale() && bd.prop_kind == prop_kind::backward);
    conf.diff_shift = (pd->use_shift() && bd.prop_kind == prop_kind::backward);
}
std::string get_flags_str(const batch_normalization_pd_t *pd) {
    std::string s;
    if (pd->stats_is_src() || pd->use_global_stats()) s += 'G';
    if (pd->use_scale()) s += 'C';
    if (pd->use_shift()) s += 'H';
    if (pd->fuse_norm_relu()) s += 'R';
    if (pd->fuse_norm_add_relu()) s += 'A';
    return s;
}

std::string get_dt_str(const batch_normalization_pd_t *pd) {
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    auto dt = data_mdw.data_type();
    std::string s;
    if (dt == data_type::f32)
        s += "f32";
    else if (dt == data_type::s8)
        s += "s8";
    else if (dt == data_type::f16)
        s += "f16";
    else if (dt == data_type::bf16)
        s += "bf16";
    else
        assert(!"Not expected");
    return s;
}
std::string get_dir_str(const batch_normalization_pd_t *pd) {
    std::string s;
    if (pd->is_fwd() && !pd->is_training())
        s += "FWD_I";
    else if (pd->is_fwd() && pd->is_training())
        s += "FWD_D";
    else if (!pd->is_fwd())
        s += "BWD_DW";
    else
        assert(!"Not expected");
    return s;
}
std::string get_desc_str(const batch_normalization_pd_t *pd) {
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    const int ndims = data_mdw.ndims();
    const int mb = data_mdw.dims()[0];
    const int ic = data_mdw.dims()[1];
    const int id = (ndims == 5) ? data_mdw.dims()[2] : 1;
    const int ih = (ndims == 3) ? 1 : data_mdw.dims()[ndims - 2];
    const int iw = data_mdw.dims()[ndims - 1];
    std::string s;
    s += std::to_string(mb) + ",";
    s += std::to_string(ic) + ",";
    s += std::to_string(id) + ",";
    s += std::to_string(ih) + ",";
    s += std::to_string(iw);
    return s;
}
std::string get_prb_desc_str(const batch_normalization_pd_t *pd) {
    std::string s;
    s += "axb,";
    s += get_dt_str(pd) + ",";
    s += get_dir_str(pd) + ",";
    s += get_flags_str(pd) + ",";
    s += get_desc_str(pd);
    return s;
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
} // namespace bn_utils
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

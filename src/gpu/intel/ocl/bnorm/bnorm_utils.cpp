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
#include "gpu/intel/ocl/bnorm/bnorm_utils.hpp"
#include "common/utils.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/ocl/bnorm/bnorm_lookup_table.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {
namespace bn_utils {
using namespace dnnl::impl::utils;

float get_ss_utilization(
        int max_ss, const compute::range_t &gws, const compute::range_t &lws) {
    const size_t gws_size = gws.nelems();
    const size_t lws_size = lws.nelems();
    const size_t used_ss = utils::div_up(gws_size, lws_size);
    return (float)used_ss / max_ss;
}
float get_thr_utilization(int eu_count, int threads_per_eu, int sg_size,
        const compute::range_t &gws) {
    const size_t gws_size = gws.nelems();
    return ((float)gws_size / sg_size) / (eu_count * threads_per_eu);
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
        gpu_error_not_expected();
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
        gpu_error_not_expected();
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

} // namespace bn_utils
} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

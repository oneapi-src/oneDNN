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
#include "gpu/ocl/batch_normalization_utils.hpp"
#include "gpu/ocl/bnorm_lookup_table.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
using namespace bn_lookup_table;
namespace bn_utils {

// Gets bnorm parameters from BN_PARAMS env value
// Only used during tuning procedure, BN_TUNING env var must be set
void maybe_override_bn_conf_params_env(bnorm_conf_t &conf) {
    auto s_params = getenv_str("BN_PARAMS", "");
    assert(!s_params.empty());
    assert(conf.bn_tuning);
    bnorm_params_t params(s_params);
    params.override_params(conf);
}
// Gets bnorm parameters from a lookup table
// BN_TUNING env var must be unset or zero;
void maybe_override_bn_conf_params_table(bnorm_conf_t &conf, engine_t *engine) {
    assert(!conf.bn_tuning);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    auto gpu_arch = compute_engine->device_info()->gpu_arch();
    static bnorm_lookup_table_t table;
    auto *s_params = table.find(conf, gpu_arch);
    if (s_params) {
        bnorm_params_t params(s_params);
        params.override_params(conf);
    }
}

void maybe_override_bn_conf_params(bnorm_conf_t &conf, engine_t *engine) {
    conf.is_overrided_use_fused_atomics_reduction = false;
    conf.is_overrided_ic_block = false;
    conf.is_overrided_max_vect_size = false;
    conf.is_overrided_stat_sp_block = false;
    conf.is_overrided_update_sp_block = false;
    conf.is_overrided_update_sp_unroll = false;

    // Environment var BN_TUNING turns ON/OFF tuning mode
    conf.bn_tuning = getenv_int("BN_TUNING", 0);
    if (conf.bn_tuning) {
        maybe_override_bn_conf_params_env(conf);
    } else {
        // TODO: extend to 1pass
        if (!conf.use_stats_one_pass) {
            maybe_override_bn_conf_params_table(conf, engine);
        }
    }
}

float get_ss_utilization(int max_ss, const size_t *gws, size_t *lws) {
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

// Init conf flags for lookup table
void init_flags_lookup_table(
        std::string &flags, const batch_normalization_pd_t *pd) {
    if (pd->use_scale()) flags += 'C';
    if (pd->use_shift()) flags += 'H';
    if (pd->stats_is_src() || pd->use_global_stats()) flags += 'G';
    if (pd->fuse_norm_relu()) flags += 'R';
    if (pd->fuse_norm_add_relu()) flags += 'A';
}

// Init basic fields of conf structure
void init_conf_basic(bnorm_conf_t &conf, const batch_normalization_pd_t *pd) {
    using namespace dnnl::impl::format_tag;

    const batch_normalization_desc_t &bd = *pd->desc();
    const memory_desc_wrapper data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    const int ndims = data_mdw.ndims();

    conf.data_type = data_mdw.data_type();

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

    conf.use_workaround = false;
}

} // namespace bn_utils
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

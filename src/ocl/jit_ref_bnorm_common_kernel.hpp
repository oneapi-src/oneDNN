/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef JIT_REF_BNORM_COMMON_KERNEL_HPP
#define JIT_REF_BNORM_COMMON_KERNEL_HPP

#include "common/batch_normalization_pd.hpp"
#include "common/c_types_map.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace mkldnn::impl::format_tag;

struct jit_ref_bnorm_common_kernel {

    jit_ref_bnorm_common_kernel(jit_bnorm_conf_t ajbn) : jbn(ajbn){};

    ~jit_ref_bnorm_common_kernel(){};

    static status_t init_conf(jit_bnorm_conf_t &jbn,
            const batch_normalization_desc_t &bd,
            const memory_desc_wrapper &data_mdw,
            const batch_normalization_pd_t *bdesc, jit_offsets &jit_off) {

        const int ndims = data_mdw.ndims();

        jbn.data_type = data_mdw.data_type();

        jbn.ndims = ndims;
        jbn.mb = data_mdw.dims()[0];

        jbn.ic = data_mdw.dims()[1];
        jbn.id = (ndims == 5) ? data_mdw.dims()[2] : 1;
        jbn.ih = (ndims == 3) ? 1 : data_mdw.dims()[ndims - 2];
        jbn.iw = data_mdw.dims()[ndims - 1];

        jbn.is_forward = utils::one_of(bd.prop_kind,
                prop_kind::forward_training, prop_kind::forward_inference);
        jbn.is_backward = utils::one_of(
                bd.prop_kind, prop_kind::backward, prop_kind::backward_data);

        jbn.use_scaleshift = bdesc->use_scaleshift();
        jbn.save_stats = bdesc->is_training();
        jbn.is_training = bdesc->is_training();
        jbn.fuse_norm_relu = bdesc->fuse_norm_relu();
        jbn.calculate_stats = !bdesc->stats_is_src();
        jbn.with_relu = bdesc->with_relu_post_op();
        jbn.eps = bd.batch_norm_epsilon;
        jbn.calculate_diff_stats = !bdesc->use_global_stats();
        jbn.diff_scaleshift = (bdesc->use_scaleshift()
                && bd.prop_kind == prop_kind::backward);

        set_offsets(data_mdw, jit_off.src_off);

        jbn.lws_d[0] = 1;
        jbn.lws_d[1] = 1;
        jbn.lws_d[2] = 1;
        jbn.gws_d[0] = jbn.ic;
        jbn.gws_d[1] = 1;
        jbn.gws_d[2] = 1;
        jbn.use_16mb_unroll = 0;
        jbn.mb_chunk = 1;
        jbn.sp_chunk = 1;
        jbn.mb_block = 1;

        const bool has_padding = !data_mdw.is_dense();
        if (!has_padding
                && data_mdw.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c,
                        NCw16n16c, NChw16n16c, NCdhw16n16c)) {
            jbn.mb_block = data_mdw.matches_one_of_tag(
                                   NCw16n16c, NChw16n16c, NCdhw16n16c)
                    ? 16
                    : 1;
            jbn.mb_chunk = nstl::min((jbn.mb / jbn.mb_block), 256);
            jbn.sp_chunk = nstl::min(jbn.ih * jbn.iw * jbn.id,
                    nstl::max(1, utils::div_up(256, jbn.mb_chunk)));
            jbn.use_16mb_unroll = 1;
            jbn.lws_d[0] = 1;
            jbn.lws_d[1] = 16;
            jbn.lws_d[2] = 1;
            jbn.gws_d[2] = jbn.ih * jbn.iw * jbn.id;
            jbn.gws_d[1] = jbn.ic;
            jbn.gws_d[0] = jbn.mb / jbn.mb_block;
        }

        return status::success;
    };

    static status_t init_const_def(ocl_jit_t &jit, const jit_bnorm_conf_t &jbn,
            const jit_offsets &jit_off) {
        jit.set_data_type(jbn.data_type);

        jit.define_int("NDIMS", jbn.ndims);
        jit.define_int("MB", jbn.mb);
        jit.define_int("IC", jbn.ic);
        jit.define_int("ID", jbn.id);
        jit.define_int("IH", jbn.ih);
        jit.define_int("IW", jbn.iw);
        jit.define_int("LWS_0", jbn.lws_d[0]);
        jit.define_int("LWS_1", jbn.lws_d[1]);
        jit.define_int("LWS_2", jbn.lws_d[2]);
        jit.define_int("USE_16MB_UNROLL", jbn.use_16mb_unroll);
        jit.define_int("MB_CHUNK", jbn.mb_chunk);
        jit.define_int("SP_CHUNK", jbn.sp_chunk);
        jit.define_int("MB_BLOCK", jbn.mb_block);

        if (jbn.is_forward)
            jit.define_int("BNORM_FWD", 1);
        else if (jbn.is_backward)
            jit.define_int("BNORM_BWD", 1);

        jit.define_int("WITH_RELU", jbn.with_relu);
        jit.define_int("SAVE_STATS", jbn.save_stats);
        jit.define_int("IS_TRAINING", jbn.is_training);
        jit.define_int("FUSE_BN_RELU", jbn.fuse_norm_relu);
        jit.define_int("CALCULATE_STATS", jbn.calculate_stats);
        jit.define_int("USE_SCALESHIFT", jbn.use_scaleshift);
        jit.define_int("CALCULATE_DIFF_STATS", jbn.calculate_diff_stats);
        jit.define_int("DIFF_SCALESHIFT", jbn.diff_scaleshift);

        def_offsets(jit_off.src_off, jit, "SRC", jbn.ndims);

        return status::success;
    }

    jit_bnorm_conf_t jbn;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

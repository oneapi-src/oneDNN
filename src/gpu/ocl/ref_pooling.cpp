/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "gpu/ocl/ref_pooling.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(
        pool_conf_t &conf, offsets_t &off, const pooling_pd_t *pd) {
    using namespace dnnl::impl::format_tag;

    const memory_desc_wrapper src_d(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    const memory_desc_wrapper dst_d(
            pd->is_fwd() ? pd->dst_md() : pd->diff_dst_md());

    const pooling_desc_t &desc = *pd->desc();
    const int ndims = src_d.ndims();
    const auto &src_dims = src_d.padded_dims();
    const auto &dst_dims = dst_d.padded_dims();

    conf.ndims = ndims;
    conf.mb = src_dims[0];

    conf.c = src_dims[1];
    conf.id = (ndims == 5) ? src_dims[2] : 1;
    conf.ih = (ndims == 3) ? 1 : src_dims[ndims - 2];
    conf.iw = src_dims[ndims - 1];
    conf.od = (ndims == 5) ? dst_dims[2] : 1;
    conf.oh = (ndims == 3) ? 1 : dst_dims[ndims - 2];
    conf.ow = dst_dims[ndims - 1];

    conf.stride_d = (ndims == 5) ? desc.strides[0] : 1;
    conf.stride_h = (ndims == 3) ? 1 : desc.strides[ndims - 4];
    conf.stride_w = desc.strides[ndims - 3];
    conf.kd = (ndims == 5) ? desc.kernel[0] : 1;
    conf.kh = (ndims == 3) ? 1 : desc.kernel[ndims - 4];
    conf.kw = desc.kernel[ndims - 3];

    conf.f_pad = (ndims == 5) ? desc.padding[0][0] : 0;
    conf.t_pad = (ndims == 3) ? 0 : desc.padding[0][ndims - 4];
    conf.l_pad = desc.padding[0][ndims - 3];

    conf.alg = desc.alg_kind;

    conf.src_dt = src_d.data_type();

    conf.is_training = desc.prop_kind == prop_kind::forward_training;
    conf.is_backward = desc.prop_kind == prop_kind::backward_data;

    set_offsets(src_d, off.src_off);
    set_offsets(dst_d, off.dst_off);

    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(pd->engine());
    conf.dispatch = compute_engine->create_dispatch(
            conf.is_backward ? src_d.md_ : dst_d.md_);

    conf.sub_group_size = 1;
    conf.use_16mb_unroll = 0;
    conf.use_16c_unroll = 0;
    // disable subgroup optimization for s8
    if (utils::one_of(src_d.data_type(), data_type::f32, data_type::f16)
            && ((src_d.matches_tag(nCw16c) && dst_d.matches_tag(nCw16c))
                    || (src_d.matches_tag(nChw16c)
                            && dst_d.matches_tag(nChw16c))
                    || (src_d.matches_tag(nCdhw16c)
                            && dst_d.matches_tag(nCdhw16c))
                    || (src_d.matches_tag(NCw16n16c)
                            && dst_d.matches_tag(NCw16n16c))
                    || (src_d.matches_tag(NChw16n16c)
                            && dst_d.matches_tag(NChw16n16c))
                    || (src_d.matches_tag(NCdhw16n16c)
                            && dst_d.matches_tag(NCdhw16n16c)))) {
        conf.use_16mb_unroll
                = src_d.matches_one_of_tag(NCw16n16c, NChw16n16c, NCdhw16n16c);
        conf.use_16c_unroll = 1;
        conf.sub_group_size = 16;

        conf.dispatch.define_dim(
                "MB", 0, conf.mb, conf.use_16mb_unroll ? 16 : 1);
        conf.dispatch.define_dim("OC", 1, conf.c);

        if (!conf.is_backward) {
            conf.dispatch.define_dim("OD", nstl::max(2, ndims - 3), conf.od);
            conf.dispatch.define_dim("OH", nstl::max(2, ndims - 2), conf.oh);
        } else {
            conf.dispatch.define_dim("ID", nstl::max(2, ndims - 3), conf.id);
            conf.dispatch.define_dim("IH", nstl::max(2, ndims - 2), conf.ih);
            conf.dispatch.define_dim("IW", nstl::max(2, ndims - 1), conf.iw);
        }

        conf.dispatch.vectorize_dim("OC", 16);
    } else {
        conf.dispatch.define_dim("MB", 0, conf.mb);
        conf.dispatch.define_dim("OC", 1, conf.c);
    }

    conf.dispatch.generate();

    return status::success;
};

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const pool_conf_t &conf, const offsets_t &off) {
    using namespace dnnl::impl::alg_kind;
    status_t status = status::success;

    kernel_ctx.set_data_type(conf.src_dt);

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("C", conf.c);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);
    kernel_ctx.define_int("PD", conf.f_pad);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("USE_16MB_UNROLL", conf.use_16mb_unroll);
    kernel_ctx.define_int("USE_16C_UNROLL", conf.use_16c_unroll);
    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    if (conf.is_backward)
        kernel_ctx.define_int("IS_BWD", 1);
    else
        kernel_ctx.define_int("IS_FWD", 1);
    switch (conf.alg) {
        case pooling_max: kernel_ctx.define_int("POOLING_MAX", 1); break;
        case pooling_avg_exclude_padding:
            kernel_ctx.define_int("POOLING_AVG_EXCLUDE_PADDING", 1);
            break;
        case pooling_avg_include_padding:
            kernel_ctx.define_int("POOLING_AVG_INCLUDE_PADDING", 1);
            break;
        default: status = status::unimplemented;
    }
    if (status != status::success) return status;

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);
    def_offsets(off.dst_off, kernel_ctx, "DST", conf.ndims);

    def_dispatch(kernel_ctx, conf.dispatch);

    return status::success;
}

status_t ref_pooling_fwd_t::pd_t::init_conf() {
    return init_conf_common(conf, off, this);
}

status_t ref_pooling_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_pooling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &ws = CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, ws);
    arg_list.set(2, dst);

    auto nd_range = pd()->conf.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

status_t ref_pooling_bwd_t::pd_t::init_conf() {
    return init_conf_common(conf, off, this);
}

status_t ref_pooling_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_pooling_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &ws = CTX_IN_STORAGE(DNNL_ARG_WORKSPACE);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, ws);
    arg_list.set(2, diff_dst);

    auto nd_range = pd()->conf.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

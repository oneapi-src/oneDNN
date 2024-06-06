/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/intel/ocl/ref_convolution.hpp"

#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

static status_t init_conf_common(
        conv_conf_t &conf, const convolution_pd_t *pd, impl::engine_t *engine) {
    const convolution_desc_t &cd = *pd->desc();
    const memory_desc_t &src_md = *pd->invariant_src_md();
    const memory_desc_t &weights_md = *pd->invariant_wei_md();
    const memory_desc_t &dst_md = *pd->invariant_dst_md();
    const memory_desc_t &bias_md = *pd->invariant_bia_md();
    const primitive_attr_t &attr = *pd->attr();

    set_default_conf(conf, cd, src_md, weights_md, dst_md, bias_md, attr);

    int oc_idx = (int)conf.with_groups;
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    switch (cd.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference: {
            conf.dispatch = compute_engine->create_dispatch(&dst_md);
            conf.dispatch.define_dim("MB", 0, conf.mb);
            conf.dispatch.define_dim("G", 1, conf.ngroups);
            conf.dispatch.define_dim("OC", 1, conf.oc);
            conf.dispatch.define_dim(
                    "OD", nstl::max(2, conf.ndims - 3), conf.od);
            conf.dispatch.define_dim(
                    "OH", nstl::max(2, conf.ndims - 2), conf.oh);
            conf.dispatch.define_dim(
                    "OW", nstl::max(2, conf.ndims - 1), conf.ow);
            conf.dispatch.generate();
            break;
        }
        case prop_kind::backward_data: {
            conf.dispatch = compute_engine->create_dispatch(&src_md);
            conf.dispatch.define_dim_with_nesting_level(
                    "IC", conf.ndims, conf.ic);
            conf.dispatch.define_dim("MB", conf.mb);
            conf.dispatch.define_dim("G", conf.ngroups);
            conf.dispatch.define_dim(
                    "ID", nstl::max(2, conf.ndims - 3), conf.id);
            conf.dispatch.define_dim(
                    "IH", nstl::max(2, conf.ndims - 2), conf.ih);
            conf.dispatch.define_dim(
                    "IW", nstl::max(2, conf.ndims - 1), conf.iw);
            conf.dispatch.generate();
            break;
        }
        case prop_kind::backward_weights: {
            conf.dispatch = compute_engine->create_dispatch(&weights_md);
            conf.dispatch.define_dim("G", 0, conf.ngroups);
            conf.dispatch.define_dim("OC", oc_idx, conf.oc);
            conf.dispatch.define_dim("IC", oc_idx + 1, conf.ic);
            conf.dispatch.define_dim(
                    "KD", oc_idx + nstl::max(2, conf.ndims - 3), conf.kd);
            conf.dispatch.define_dim(
                    "KH", oc_idx + nstl::max(2, conf.ndims - 2), conf.kh);
            conf.dispatch.define_dim(
                    "KW", oc_idx + nstl::max(2, conf.ndims - 1), conf.kw);
            conf.dispatch.generate();
            break;
        }
        default: break;
    }

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const conv_conf_t &conf, const post_ops_t &post_ops,
        const memory_desc_t *dst_md) {
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("WITH_GROUPS", conf.with_groups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc);
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
    kernel_ctx.define_int("PD_R", conf.back_pad);
    kernel_ctx.define_int("PH_R", conf.b_pad);
    kernel_ctx.define_int("PW_R", conf.r_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);

    kernel_ctx.define_int("IS_FWD",
            utils::one_of(conf.prop_kind, prop_kind::forward_inference,
                    prop_kind::forward_training));
    kernel_ctx.define_int(
            "IS_BWD_D", conf.prop_kind == prop_kind::backward_data);
    kernel_ctx.define_int(
            "IS_BWD_W", conf.prop_kind == prop_kind::backward_weights);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.wei_md_info, "WEI");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    def_dispatch(kernel_ctx, conf.dispatch);

    switch (conf.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            kernel_ctx.set_data_type(conf.dst_data_type);
            break;
        case prop_kind::backward_data:
            kernel_ctx.set_data_type(conf.src_data_type);
            break;
        case prop_kind::backward_weights:
            kernel_ctx.set_data_type(conf.weights_data_type);
            break;
        default: break;
    }

    def_data_type(kernel_ctx, conf.src_data_type, "SRC");
    def_data_type(kernel_ctx, conf.weights_data_type, "WEI");
    def_data_type(kernel_ctx, conf.bias_data_type, "BIA");
    def_data_type(kernel_ctx, conf.dst_data_type, "DST");
    def_data_type(kernel_ctx, conf.acc_data_type, "ACC");
    def_data_type(kernel_ctx,
            conf.attr_info.sum_data_type == dnnl_data_type_undef
                    ? conf.dst_data_type
                    : conf.attr_info.sum_data_type,
            "SUM");

    CHECK(def_attr_info(kernel_ctx, conf.attr_info, post_ops, *dst_md));
    return status::success;
}

status_t ref_convolution_fwd_t::pd_t::init_conf(impl::engine_t *engine) {
    CHECK(init_conf_common(conf, this, engine));
    return status::success;
}

status_t ref_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(
            kernel_ctx, conf, attr()->post_ops_, invariant_dst_md());
}

status_t ref_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    status_t status = status::success;
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DST, status);
    CHECK(status);
    auto &src_scales = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &wei_scales = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    auto &dst_scales = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    auto &src_zpoints
            = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    auto &dst_zpoints
            = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);

    unsigned arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, 4, pd()->attr()->post_ops_);

    arg_list.set(arg_idx++, src_scales);
    arg_list.set(arg_idx++, wei_scales);
    arg_list.set(arg_idx++, dst_scales);

    if (conf.attr_info.with_src_zpoints)
        arg_list.set(arg_idx++, src_zpoints);
    else
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());

    if (conf.attr_info.with_dst_zpoints)
        arg_list.set(arg_idx++, dst_zpoints);
    else
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());

    auto nd_range = pd()->conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);
    return status;
}

status_t ref_convolution_bwd_data_t::pd_t::init_conf(impl::engine_t *engine) {
    CHECK(init_conf_common(conf, this, engine));
    return status::success;
}

status_t ref_convolution_bwd_data_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(
            kernel_ctx, conf, attr()->post_ops_, invariant_src_md());
}

status_t ref_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SRC, status);
    CHECK(status);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &src_scales = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &wei_scales = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    auto &dst_scales = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    auto &src_zpoints
            = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    auto &dst_zpoints
            = CTX_IN_STORAGE(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, diff_dst);
    arg_list.set(3, bias);

    unsigned arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, 4, pd()->attr()->post_ops_);

    arg_list.set(arg_idx++, src_scales);
    arg_list.set(arg_idx++, wei_scales);
    arg_list.set(arg_idx++, dst_scales);

    if (conf.attr_info.with_src_zpoints)
        arg_list.set(arg_idx++, src_zpoints);
    else
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());

    if (conf.attr_info.with_dst_zpoints)
        arg_list.set(arg_idx++, dst_zpoints);
    else
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());

    auto nd_range = pd()->conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

status_t ref_convolution_bwd_weights_t::pd_t::init_conf(
        impl::engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t ref_convolution_bwd_weights_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(
            kernel_ctx, conf, attr()->post_ops_, invariant_wei_md());
}

status_t ref_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {

    status_t status = status::success;
    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_WEIGHTS, status);
    CHECK(status);
    auto &diff_bias = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_BIAS, status);
    CHECK(status);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_weights);
    arg_list.set(2, diff_bias);
    arg_list.set(3, diff_dst);

    auto nd_range = pd()->conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

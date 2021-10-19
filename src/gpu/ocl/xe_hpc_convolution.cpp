/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "gpu/ocl/xe_hpc_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace data_type;
using namespace format_tag;

status_t xe_hpc_convolution_fwd_t::pd_t::init_conf() {
    const convolution_desc_t &cd = *desc();
    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    status_t status = status::success;

    if (conf.is_depthwise) return status::unimplemented;

    if (conf.mb < 8) return status::unimplemented;
    if (conf.ic <= 4) return status::unimplemented;

    conf.mb_block = 32;
    conf.oc_block = (utils::one_of(conf.src_data_type, u8, s8)) ? 32 : 16;
    conf.ic_block = (utils::one_of(conf.src_data_type, u8, s8)) ? 32 : 16;
    conf.calc_block = 32;

    if (!conf.is_depthwise && conf.with_groups && conf.ngroups > 1
            && (conf.oc % conf.calc_block != 0 || conf.ic % conf.ic_block != 0))
        return status::unimplemented;

    conf.sub_group_size = 16;

    conf.nchunk = utils::div_up(conf.oc * conf.ngroups, conf.calc_block);
    conf.wei_block = 32 * 32 / types::data_type_size(conf.weights_data_type);
    int oc_group = 1;

    conf.lws_d[0] = 16 * oc_group;
    conf.lws_d[1] = 1;
    conf.lws_d[2] = 1;

    conf.gws_d[0] = utils::rnd_up(conf.nchunk * 16, conf.lws_d[0]);
    conf.gws_d[1] = conf.od * conf.oh * utils::rnd_up(conf.ow, conf.lws_d[1]);
    conf.gws_d[2] = utils::div_up(conf.mb, conf.mb_block);

    format_tag_t src_tag, dst_tag, wei_tag;
    if (utils::one_of(conf.src_data_type, u8, s8)) {
        src_tag = utils::pick(
                conf.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        dst_tag = utils::pick(
                conf.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);

        wei_tag = utils::pick(conf.ndims - 3, format_tag::OIw2o8i16o4i,
                format_tag::OIhw2o8i16o4i, format_tag::OIdhw2o8i16o4i);
    } else {
        src_tag = utils::pick(conf.ndims - 3, format_tag::NCw32n16c,
                format_tag::NChw32n16c, format_tag::NCdhw32n16c);
        dst_tag = utils::pick(conf.ndims - 3, format_tag::NCw32n16c,
                format_tag::NChw32n16c, format_tag::NCdhw32n16c);
        wei_tag = conf.with_groups
                ? utils::pick(conf.ndims - 3, format_tag::gOIw2o8i16o2i,
                        format_tag::gOIhw2o8i16o2i, format_tag::gOIdhw2o8i16o2i)
                : utils::pick(conf.ndims - 3, format_tag::OIw2o8i16o2i,
                        format_tag::OIhw2o8i16o2i, format_tag::OIdhw2o8i16o2i);
    }

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    conf.src_tag = src_mdw.format_kind() == format_kind::any
            ? src_tag
            : src_mdw.matches_one_of_tag(src_tag);
    conf.wei_tag = weights_mdw.format_kind() == format_kind::any
            ? wei_tag
            : weights_mdw.matches_one_of_tag(wei_tag);
    conf.dst_tag = dst_mdw.format_kind() == format_kind::any
            ? dst_tag
            : dst_mdw.matches_one_of_tag(dst_tag);

    if (conf.src_tag != src_tag || conf.wei_tag != wei_tag
            || conf.dst_tag != dst_tag)
        return status::unimplemented;

    conf.wei_slm_size = conf.kw * 32 * 32 * utils::div_up(conf.lws_d[0], 8);

    return status;
}

status_t xe_hpc_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("G", conf.ngroups);
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
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);
    kernel_ctx.define_int("OW_PADDED", utils::rnd_up(conf.ow, conf.lws_d[1]));
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("OC_PADDED", utils::rnd_up(conf.oc, conf.oc_block));
    kernel_ctx.define_int("OC_CALC_BLOCK", conf.calc_block);
    kernel_ctx.define_int("WEI_BLOCK", conf.wei_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("OC_GROUP", utils::div_up(conf.lws_d[0], 8));
    kernel_ctx.define_int("MB_GROUP", 1);
    kernel_ctx.define_int("SP_GROUP", conf.lws_d[1]);
    kernel_ctx.define_int("OC_NCHUNK", utils::div_up(conf.oc, conf.oc_block));
    kernel_ctx.define_int("IC_NCHUNK", utils::div_up(conf.ic, conf.ic_block));
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    def_attr_info(kernel_ctx, conf.attr_info, attr()->post_ops_);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.set_data_type(conf.src_data_type);
    def_data_type(kernel_ctx, conf.src_data_type, "SRC");
    def_data_type(kernel_ctx, conf.dst_data_type, "DST");
    def_data_type(kernel_ctx, conf.weights_data_type, "WEI");
    def_data_type(kernel_ctx, conf.bias_data_type, "BIA");
    def_data_type(kernel_ctx,
            conf.attr_info.sum_data_type == dnnl_data_type_undef
                    ? conf.dst_data_type
                    : conf.attr_info.sum_data_type,
            "SUM");

    kernel_ctx.add_option("-Dcl_intel_subgroups_char");
    kernel_ctx.add_option("-Dcl_intel_subgroups_long");
    kernel_ctx.add_option(
            "-Dcl_intel_subgroup_split_matrix_multiply_accumulate");

    if (is_xe_hpc) kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");
    return status::success;
}

status_t xe_hpc_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &oscales = CTX_IN_STORAGE(DNNL_ARG_ATTR_OUTPUT_SCALES);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);

    unsigned arg_idx = append_post_ops_to_arg_list(
            ctx, arg_list, 4, pd()->attr()->post_ops_);

    if (conf.attr_info.common_oscales) {
        float scales = pd()->attr()->output_scales_.scales_[0];
        arg_list.set(arg_idx++, scales);
    } else {
        arg_list.set(arg_idx++, 1.0f);
    }

    if (conf.attr_info.with_per_oc_oscales) {
        if (conf.attr_info.with_runtime_oscales)
            arg_list.set(arg_idx++, oscales);
        else
            arg_list.set(arg_idx++, CTX_GPU_RES_STORAGE(SCALES_));
    } else {
        arg_list.set(arg_idx++, memory_storage_t::empty_storage());
    }

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

status_t xe_hpc_convolution_bwd_data_t::pd_t::init_conf() {
    const convolution_desc_t &cd = *desc();
    set_default_conf(conf, cd, *diff_src_md(), *weights_md(), *diff_dst_md(),
            *weights_md(1), *attr());

    status_t status = status::success;

    if (conf.mb < 8) return status::unimplemented;

    conf.mb_block = 32;
    conf.oc_block = (utils::one_of(conf.src_data_type, u8, s8)) ? 32 : 16;
    conf.ic_block = (utils::one_of(conf.src_data_type, u8, s8)) ? 32 : 16;
    conf.calc_block = 32;

    if (conf.with_groups && conf.ngroups > 1
            && (conf.oc % conf.oc_block != 0 || conf.ic % conf.calc_block != 0))
        return status::unimplemented;

    conf.sub_group_size = 16;
    conf.nchunk = utils::div_up(conf.ic * conf.ngroups, conf.calc_block);
    conf.wei_block = 32 * 32 / types::data_type_size(conf.weights_data_type);
    int ic_group = 1;

    conf.lws_d[0] = 16 * ic_group;
    conf.lws_d[1] = 1;
    conf.lws_d[2] = 1;

    conf.gws_d[0] = utils::rnd_up(conf.nchunk * 16, conf.lws_d[0]);
    conf.gws_d[1] = conf.id * conf.ih * utils::rnd_up(conf.iw, conf.lws_d[1]);
    conf.gws_d[2] = utils::div_up(conf.mb, conf.mb_block);

    format_tag_t src_tag, dst_tag, wei_tag;
    if (utils::one_of(conf.src_data_type, u8, s8)) {
        src_tag = utils::pick(
                conf.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        dst_tag = utils::pick(
                conf.ndims - 3, NCw32n32c, NChw32n32c, NCdhw32n32c);
        wei_tag = utils::pick(conf.ndims - 3, format_tag::IOw2i8o16i4o,
                format_tag::IOhw2i8o16i4o, format_tag::IOdhw2i8o16i4o);
    } else {
        src_tag = utils::pick(conf.ndims - 3, format_tag::NCw32n16c,
                format_tag::NChw32n16c, format_tag::NCdhw32n16c);

        dst_tag = utils::pick(conf.ndims - 3, format_tag::NCw32n16c,
                format_tag::NChw32n16c, format_tag::NCdhw32n16c);

        wei_tag = utils::pick(conf.ndims - 3, format_tag::IOw2i8o16i2o,
                format_tag::IOhw2i8o16i2o, format_tag::IOdhw2i8o16i2o);
    }

    const memory_desc_wrapper src_mdw(diff_src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());

    conf.src_tag = src_mdw.format_kind() == format_kind::any
            ? src_tag
            : src_mdw.matches_one_of_tag(src_tag);
    conf.wei_tag = weights_mdw.format_kind() == format_kind::any
            ? wei_tag
            : weights_mdw.matches_one_of_tag(wei_tag);
    conf.dst_tag = dst_mdw.format_kind() == format_kind::any
            ? dst_tag
            : dst_mdw.matches_one_of_tag(dst_tag);

    if (conf.src_tag != src_tag || conf.wei_tag != wei_tag
            || conf.dst_tag != dst_tag)
        return status::unimplemented;

    conf.wei_slm_size = conf.kw * 32 * 32 * utils::div_up(conf.lws_d[0], 16);

    return status;
}

status_t xe_hpc_convolution_bwd_data_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("G", conf.ngroups);
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
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);

    kernel_ctx.define_int("IW_PADDED", utils::rnd_up(conf.iw, conf.lws_d[1]));

    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_CALC_BLOCK", conf.calc_block);
    kernel_ctx.define_int("WEI_BLOCK", conf.wei_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);

    kernel_ctx.define_int("IC_GROUP", utils::div_up(conf.lws_d[0], 16));
    kernel_ctx.define_int("MB_GROUP", 1);
    kernel_ctx.define_int("SP_GROUP", conf.lws_d[1]);

    kernel_ctx.define_int("OC_NCHUNK", utils::div_up(conf.oc, conf.oc_block));
    kernel_ctx.define_int("IC_NCHUNK", utils::div_up(conf.ic, conf.ic_block));

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);

    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.set_data_type(conf.dst_data_type);
    def_data_type(kernel_ctx, conf.src_data_type, "SRC");
    def_data_type(kernel_ctx, conf.dst_data_type, "DST");
    def_data_type(kernel_ctx, conf.weights_data_type, "WEI");
    def_data_type(kernel_ctx, conf.bias_data_type, "BIA");

    kernel_ctx.add_option("-Dcl_intel_subgroups_char");

    if (is_xe_hpc) kernel_ctx.add_option("-cl-intel-256-GRF-per-thread");

    return status::success;
}

status_t xe_hpc_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {

    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, diff_dst);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

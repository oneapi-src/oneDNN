/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gpu/ocl/gen9_wino_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;

static void fwd_compute_block_sizes(
        conv_conf_t &conf, const convolution_pd_t *pd) {

    if (conf.ver == ver_16mb16c) {
        conf.mb_block = (conf.src_data_type == data_type::f16)
                ? (conf.mb % 32 == 0 ? 32 : 16)
                : 16;
    } else {
        conf.mb_block = 1;
    }

    conf.oc_block = 16;
    conf.ic_block = nstl::min(conf.ic, 16);
    conf.ocb = utils::div_up(conf.oc, conf.oc_block);
}

status_t gen9_wino_convolution_fwd_t::pd_t::init_conf() {

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
    conf.oc = utils::rnd_up(conf.oc_without_padding, 16);

    const bool is_wino_shape = conf.kh == 3 && conf.kw == 3 && conf.ngroups == 1
            && conf.stride_h == 1 && conf.stride_w == 1 && conf.dilate_h == 0
            && conf.dilate_w == 0 && conf.l_pad <= 1 && conf.r_pad <= 1
            && conf.t_pad <= 1 && conf.b_pad <= 1;
    if (!is_wino_shape) return status::unimplemented;

    // TODO: Implement bias
    if (conf.with_bias) return status::unimplemented;

    //Using F(m, r) for r = 3 and tile_size = m + r - 1
    const int m = 2;
    const int r = 3;

    conf.wino_m = m;
    conf.wino_r = r;
    conf.tile_size = m + r - 1;

    const bool is_16oc = conf.oc % 16 == 0;
    const bool is_16ic = conf.ic % 16 == 0;

    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = conf.wino_m;
    conf.ow_block = conf.wino_m;
    conf.ocb = 1;

    if ((is_16oc && is_16ic)) {
        conf.ver = (conf.mb % 16 == 0) ? ver_16mb16c : ver_8ow16c;
    } else {
        return status::unimplemented;
    }

    switch (conf.ver) {
        case ver_8ow16c:
        case ver_16mb16c: {
            fwd_compute_block_sizes(conf, this);
            conf.mb_block = 1;
            conf.lws_d[0] = 1;
            conf.lws_d[1] = 1;
            conf.lws_d[2] = 1;
            conf.gws_d[0]
                    = (conf.mb / conf.mb_block) * (conf.oc / conf.oc_block);
            conf.gws_d[1] = utils::div_up(conf.oh, conf.oh_block)
                    * utils::div_up(conf.ow, conf.ow_block);
            conf.gws_d[2] = 1;

            conf.U_lws_d[0] = 1;
            conf.U_lws_d[1] = 1;
            conf.U_lws_d[2] = 1;
            conf.U_gws_d[0]
                    = (conf.ic / conf.ic_block) * (conf.oc / conf.oc_block);
            conf.U_gws_d[1] = 1;
            conf.U_gws_d[2] = 1;
            break;
        }
        default: return status::unimplemented;
    }

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_16mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
                                        : utils::pick(conf.ndims - 3, OIw16i16o,
                                                OIhw16i16o, OIdhw16i16o));
            break;
        default: return status::unimplemented;
    }

    if (src_mdw.format_kind() == format_kind::any) {
        conf.src_tag = src_tag;
    } else {
        conf.src_tag = src_mdw.matches_one_of_tag(src_tag);
    }
    if (conf.src_tag != src_tag) return status::unimplemented;

    if (weights_mdw.format_kind() == format_kind::any) {
        conf.wei_tag = wei_tag;
    } else {
        conf.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
    }
    if (conf.wei_tag != wei_tag) return status::unimplemented;

    if (dst_mdw.format_kind() == format_kind::any) {
        conf.dst_tag = dst_tag;
    } else {
        conf.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
    }
    if (conf.dst_tag != dst_tag) return status::unimplemented;

    return status::success;
}

void gen9_wino_convolution_fwd_t::pd_t::init_scratchpad() {
    size_t U_sz = conf.tile_size * conf.tile_size * conf.ic * conf.oc;
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book<float>(key_wino_U, U_sz);
}

status_t gen9_wino_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("OCB", conf.ocb);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OH_BLOCK", conf.oh_block);
    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);
    kernel_ctx.define_int("OW_LAST", utils::rnd_dn(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OWB", utils::div_up(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OHB", utils::div_up(conf.oh, conf.oh_block));
    kernel_ctx.define_int("WINO_M", conf.wino_m);
    kernel_ctx.define_int("WINO_R", conf.wino_r);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);

    kernel_ctx.set_data_type(conf.src_data_type);

    kernel_ctx.define_int("VER_8OW16C", conf.ver == ver_8ow16c);
    kernel_ctx.define_int("VER_16MB16C", conf.ver == ver_16mb16c);

    kernel_ctx.define_int("SRC_16N16C",
            utils::one_of(conf.src_tag, NCw16n16c, NChw16n16c, NCdhw16n16c));
    kernel_ctx.define_int(
            "SRC_W16C", utils::one_of(conf.src_tag, nCw16c, nChw16c, nCdhw16c));

    kernel_ctx.define_int("WEI_16I16O",
            utils::one_of(conf.wei_tag, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o,
                    OIw16i16o, OIhw16i16o, OIdhw16i16o));

    kernel_ctx.define_int("DST_16N16C",
            utils::one_of(conf.dst_tag, NCw16n16c, NChw16n16c, NCdhw16n16c));
    kernel_ctx.define_int(
            "DST_W16C", utils::one_of(conf.dst_tag, nCw16c, nChw16c, nCdhw16c));

    def_attr_info(kernel_ctx, conf.attr_info);

    kernel_ctx.print_options();
    return status::success;
}

status_t gen9_wino_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;
    const auto &attr_info = conf.attr_info;

    std::unique_ptr<memory_storage_t> wei_trans
            = ctx.get_scratchpad_grantor().get_memory_storage(key_wino_U);

    compute::kernel_arg_list_t wei_transform_args;
    wei_transform_args.set(0, weights);
    wei_transform_args.set(1, *wei_trans);
    auto wei_trans_nd_range = compute::nd_range_t(conf.U_gws_d, conf.U_lws_d);
    status_t status = parallel_for(
            ctx, wei_trans_nd_range, wei_trans_kernel_, wei_transform_args);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, *wei_trans);
    arg_list.set(2, bias);
    arg_list.set(3, dst);
    append_post_ops_to_arg_list(ctx, arg_list, 4, attr_info.all_post_ops);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    if (attr_info.with_eltwise
            && !gpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                    attr_info.eltwise_alg, attr_info.eltwise_alpha,
                    attr_info.eltwise_beta)) {
        ctx.memory(DNNL_ARG_DST)->zero_pad(ctx.stream());
    }
    return status;
}
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

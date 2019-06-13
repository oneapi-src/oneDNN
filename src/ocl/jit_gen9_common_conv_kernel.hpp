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

#ifndef JIT_GEN9_COMMON_CONV_KERNEL_HPP
#define JIT_GEN9_COMMON_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace mkldnn::impl::format_tag;

struct jit_gen9_common_conv_fwd_kernel {

    jit_gen9_common_conv_fwd_kernel(jit_conv_conf_t ajcp) : jcp(ajcp){};

    ~jit_gen9_common_conv_fwd_kernel(){};

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t &src_md,
            const memory_desc_t &weights_md, const memory_desc_t &dst_md,
            const memory_desc_t &bias_md, const primitive_attr_t &attr) {

        const memory_desc_wrapper src_mdw(&src_md);
        const memory_desc_wrapper weights_mdw(&weights_md);
        const memory_desc_wrapper dst_mdw(&dst_md);
        const memory_desc_wrapper bias_mdw(&bias_md);

        set_default_conf(jcp, cd, src_md, weights_md, dst_md, attr);

        jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
        jcp.src_data_type = cd.src_desc.data_type;
        jcp.weights_data_type = cd.weights_desc.data_type;
        jcp.dst_data_type = cd.dst_desc.data_type;
        jcp.acc_data_type = cd.accum_data_type;
        jcp.bias_data_type = jcp.with_bias
            ? cd.bias_desc.data_type : data_type::f32;

        const bool is_1stconv = jcp.ic_without_padding == 3;
        const bool is_depthwise =
            jcp.with_groups && (jcp.ic_without_padding == 1)
            && (jcp.oc_without_padding == 1);
        jcp.is_depthwise = is_depthwise;

        if (is_1stconv || jcp.with_groups) {
            jcp.ic = jcp.ic_without_padding;
            jcp.oc = jcp.oc_without_padding;
        } else {
            jcp.ic = utils::rnd_up(jcp.ic_without_padding, 16);
            jcp.oc = utils::rnd_up(jcp.oc_without_padding, 16);
        }

        if (is_depthwise)
            jcp.ngroups = utils::rnd_up(jcp.ngroups, 16);

        const bool is_dw_16g = (jcp.is_depthwise && jcp.ngroups % 16 == 0);

        const bool is_16ic = jcp.ic % 16 == 0;
        const bool is_16oc = jcp.oc % 16 == 0;
        const bool use_16mb_unroll = !(jcp.mb == 1 || jcp.mb % 16 != 0)
                && !is_1stconv
                && ((is_16ic && is_16oc) || is_dw_16g)
                && IMPLICATION(src_mdw.data_type() == data_type::f16,
                           jcp.mb % 32 == 0)
                && IMPLICATION(src_mdw.data_type() == data_type::f16
                                   && jcp.is_depthwise,
                           jcp.ngroups % 32 == 0);

        const bool is_32oc = true
                && IMPLICATION(src_mdw.data_type() == data_type::f16,
                           jcp.oc % 32 == 0);

        jcp.mb_block = 1;
        jcp.oc_block = 1;
        jcp.ic_block = 1;
        jcp.od_block = 1;
        jcp.oh_block = 1;
        jcp.ow_block = 1;
        if (use_16mb_unroll)
            jcp.ver = ver_16mb16c;
        else if ((is_16oc && is_16ic) || is_dw_16g)
            jcp.ver = ver_8ow16c;
        else if (is_1stconv && is_16oc && is_32oc)
            jcp.ver = ver_1stconv;
        else
            return status::unimplemented;

        status_t status = status::success;
        jcp.ocb = 1;
        jcp.src_data_type = src_mdw.data_type();

        switch (jcp.ver) {
        case ver_16mb16c:
            jcp.mb_block = 16;
            if (src_mdw.data_type() == mkldnn_f32
                    || src_mdw.data_type() == mkldnn_f16) {
                if (src_mdw.data_type() == mkldnn_f16 && jcp.mb % 32 != 0) {
                    jcp.mb_block = (jcp.ver == ver_1stconv && jcp.mb % 16 == 0)
                            ? 16
                            : 1;
                    jcp.oc_block = 16;
                    jcp.ic_block = (jcp.ver == ver_1stconv) ? 1 : 16;
                    jcp.ow_block = 8;
                    jcp.oh_block = 1;
                    jcp.sub_group_size = 16;
                    jcp.lws_d[0] = 16;
                    jcp.lws_d[1] = 1;
                    jcp.lws_d[2] = 1;
                    jcp.gws_d[0] = jcp.ngroups * jcp.oc;
                    jcp.gws_d[1] = utils::div_up(jcp.oh, jcp.oh_block)
                            * utils::div_up(jcp.ow, jcp.ow_block) * jcp.od;
                    jcp.gws_d[2] = jcp.mb;
                } else {
                    jcp.oc_block = 16;
                    jcp.ic_block = 16;
                    jcp.sub_group_size = 16;
                    jcp.lws_d[0] = 16;
                    jcp.lws_d[1] = 1;
                    jcp.lws_d[2] = 1;
                    jcp.gws_d[0] = jcp.oc * jcp.ngroups;
                    jcp.gws_d[1] = jcp.oh * jcp.ow * jcp.od;
                    jcp.gws_d[2] = (src_mdw.data_type() == mkldnn_f16
                                           && !jcp.is_depthwise)
                            ? jcp.mb / (jcp.mb_block * 2)
                            : jcp.mb / (jcp.mb_block * 1);
                }
            } else {
                jcp.oc_block = utils::one_of(src_mdw.data_type(), mkldnn_s8,
                                       mkldnn_u8)
                        ? 32
                        : 16;
                jcp.ic_block = utils::one_of(src_mdw.data_type(), mkldnn_s8,
                                       mkldnn_u8)
                        ? 32
                        : 16;
                jcp.mb_block = 32;
                jcp.sub_group_size = 8;

                if (jcp.kw == 1) {
                    jcp.slm_ic = 2;
                } else {
                    jcp.slm_ic = 1;
                }

                if (jcp.oc == 64) {
                    jcp.lws_d[0] = 2 * 8;
                    jcp.lws_d[1] = 8;
                } else if (jcp.oc % 128 == 0) {
                    jcp.lws_d[0] = 4 * 8;
                    jcp.lws_d[1] = 4;
                } else {
                    jcp.lws_d[0] = utils::max_div(jcp.oc, 4) * 8;
                    jcp.lws_d[1] = utils::max_div(jcp.ow, 4);
                }

                jcp.lws_d[2] = 1;
                jcp.gws_d[0] = (jcp.oc / jcp.oc_block) * 8;
                jcp.gws_d[1] = jcp.oh * utils::rnd_up(jcp.ow, jcp.lws_d[1]);
                jcp.gws_d[2] = jcp.mb / (jcp.mb_block * 1);
            }

            jcp.wht_slm_size = jcp.slm_ic * jcp.ic_block * (jcp.lws_d[0] / 8)
                    * jcp.oc_block * jcp.kw;
            if (jcp.kw == 1)
                jcp.src_slm_size = jcp.slm_ic * jcp.ic_block
                        * (jcp.lws_d[1] + jcp.kw - 1) * jcp.mb_block;
            else
                jcp.src_slm_size = jcp.slm_ic * jcp.ic_block
                        * (jcp.stride_w * (jcp.lws_d[1] - 1) + jcp.kw)
                        * jcp.mb_block;

#ifdef DEBUG_PRINT
            printf("LWS = %ld\n", jcp.lws_d[0] * jcp.lws_d[2] * jcp.lws_d[1]);
            fflush(0);
            printf("USE SLM %ld KB\n",
                    utils::div_up(jcp.wht_slm_size + jcp.src_slm_size, 1024));
            fflush(0);
            printf("LWS GWS: (%ld %ld %ld) (%ld %ld %ld)\n", jcp.lws_d[0],
                    jcp.lws_d[1], jcp.lws_d[2], jcp.gws_d[0], jcp.gws_d[1],
                    jcp.gws_d[2]);
#endif

            break;
        case ver_1stconv:
            if (src_mdw.data_type() == mkldnn_f16) {
                jcp.mb_block = jcp.mb % 16 == 0 ? 16 : 1;
                jcp.oc_block = 16;
                jcp.ic_block = 16;
                jcp.ow_block = 8;
                while (jcp.ow_block > 1) {
                    if (jcp.stride_w * jcp.ow_block
                            + jcp.kw * (1 + jcp.dilate_w) > 32)
                        jcp.ow_block--;
                    else
                        break;
                };
                jcp.oh_block = 1;
                jcp.sub_group_size = 16;
                jcp.lws_d[0] = 16;
                jcp.lws_d[1] = 1;
                jcp.lws_d[2] = 1;
                jcp.gws_d[0] = (jcp.oc / 2) * jcp.ngroups;
                jcp.gws_d[1] = utils::div_up(jcp.oh, jcp.oh_block)
                        * utils::div_up(jcp.ow, jcp.ow_block) * jcp.od;
                jcp.gws_d[2] = jcp.mb % 2 == 0 ? jcp.mb / 2
                                               : jcp.mb; // unroll mb by 2
                break;
            } else if (utils::one_of(
                               src_mdw.data_type(), mkldnn_s8, mkldnn_u8)) {
                jcp.mb_block = 32;
                jcp.oc_block = 32;
                jcp.ic_block = 32;
                jcp.slm_ic = 1;
                jcp.sub_group_size = 8;
                jcp.lws_d[0] = 2 * 8;
                jcp.lws_d[1] = utils::max_div(jcp.ow, 16);
                jcp.lws_d[2] = 1;
                jcp.gws_d[0] = (jcp.oc / jcp.oc_block) * 8;
                jcp.gws_d[1] = jcp.oh * jcp.ow;
                jcp.gws_d[2] = jcp.mb / jcp.mb_block;
                jcp.wht_slm_size = jcp.ic_block * (jcp.lws_d[0] / 8)
                        * jcp.oc_block * jcp.kw;
                jcp.src_slm_size = jcp.ic_block
                        * (jcp.stride_w * (jcp.lws_d[1] - 1) + jcp.kw)
                        * jcp.mb_block;
#ifdef DEBUG_PRINT
                printf("1st: LWS = %ld\n",
                        jcp.lws_d[0] * jcp.lws_d[2] * jcp.lws_d[1]);
                fflush(0);
                printf("1st: USE SLM %ld KB\n",
                        utils::div_up(
                                jcp.wht_slm_size + jcp.src_slm_size, 1024));
                fflush(0);
                printf("1st: LWS GWS: (%ld %ld %ld) (%ld %ld %ld)\n",
                        jcp.lws_d[0], jcp.lws_d[1], jcp.lws_d[2], jcp.gws_d[0],
                        jcp.gws_d[1], jcp.gws_d[2]);
                printf("SRC_SP_GROUP=%ld\n",
                        jcp.stride_w * (jcp.lws_d[1] - 1) + jcp.kw);
#endif
                break;
            } else if (!jcp.is_depthwise){
                jcp.mb_block = (jcp.mb % 16 == 0) ? 16 : 1;
                jcp.oc_block = 16;
                jcp.ic_block = 1;
                jcp.ow_block = 8;
                while (jcp.ow_block > 1) {
                    if (jcp.stride_w * jcp.ow_block
                            + jcp.kw * (1 + jcp.dilate_w) > 32)
                        jcp.ow_block--;
                    else
                        break;
                };
                jcp.oh_block = 1;
                jcp.sub_group_size = 16;
                jcp.lws_d[0] = 16;
                jcp.lws_d[1] = 1;
                jcp.lws_d[2] = 1;
                jcp.ocb = (jcp.oc % 32 == 0) ? 32 : 16;
                jcp.gws_d[0] = 16;
                jcp.gws_d[1] = utils::div_up(jcp.oh, jcp.oh_block)
                        * utils::div_up(jcp.ow, jcp.ow_block) * jcp.od;
                jcp.gws_d[2] = jcp.mb * (jcp.oc / jcp.ocb) * jcp.ngroups;

                break;
            }
        case ver_8ow16c:
            switch (src_mdw.data_type()) {
            case data_type::f32:
                jcp.mb_block
                        = (jcp.ver == ver_1stconv && jcp.mb % 16 == 0) ? 16 : 1;
                jcp.oc_block = 16;
                jcp.ic_block = (jcp.ver == ver_1stconv) ? 1 : 16;
                if (jcp.is_depthwise) {
                    jcp.ow_block = utils::max_div(jcp.ow, 8);
                } else {
                    jcp.ow_block = (jcp.ver == ver_1stconv)
                            ? 4
                            : nstl::max(8, utils::max_div(jcp.ow, 16));
                }
                jcp.oh_block = 1;
                jcp.sub_group_size = 16;
                jcp.lws_d[0] = 16;
                jcp.lws_d[1] = 1;
                jcp.lws_d[2] = 1;
                if (jcp.is_depthwise) {
                    jcp.ocb = jcp.ngroups;
                } else {
                    jcp.ocb = 128;
                    while (jcp.ocb > 16) {
                        if (jcp.oc % jcp.ocb == 0)
                            break;
                        else
                            jcp.ocb /= 2;
                    }
                }
                jcp.gws_d[0] = jcp.ocb;
                jcp.gws_d[1] = utils::div_up(jcp.oh, jcp.oh_block)
                        * utils::div_up(jcp.ow, jcp.ow_block) * jcp.od;
                if (jcp.is_depthwise) {
                    jcp.gws_d[2] = jcp.mb * (jcp.ngroups / jcp.ocb);
                } else {
                    jcp.gws_d[2] = jcp.mb * (jcp.oc / jcp.ocb) * jcp.ngroups;
                }
                break;
            case data_type::f16:
                jcp.mb_block
                        = (jcp.ver == ver_1stconv && jcp.mb % 16 == 0) ? 16 : 1;
                jcp.oc_block = 16;
                jcp.ic_block = (jcp.ver == ver_1stconv) ? 1 : 16;
                if (jcp.is_depthwise) {
                    jcp.ow_block = utils::max_div(jcp.ow, 8);
                } else {
                    jcp.ow_block = (jcp.ver == ver_1stconv)
                            ? 8
                            : nstl::max(8, utils::max_div(jcp.ow, 16));
                }
                jcp.oh_block = 1;
                jcp.sub_group_size = 16;
                jcp.lws_d[0] = 16;
                jcp.lws_d[1] = 1;
                jcp.lws_d[2] = 1;
                jcp.ocb = 128;
                if (jcp.is_depthwise) {
                    jcp.ocb = jcp.ngroups;
                } else {
                    while (jcp.ocb > 16) {
                        if (jcp.oc % jcp.ocb == 0)
                            break;
                        else
                            jcp.ocb /= 2;
                    }
                }
                jcp.gws_d[0] = jcp.ocb;
                jcp.gws_d[1] = utils::div_up(jcp.oh, jcp.oh_block)
                        * utils::div_up(jcp.ow, jcp.ow_block) * jcp.od;
                if (jcp.is_depthwise) {
                    jcp.gws_d[2] = jcp.mb * (jcp.ngroups / jcp.ocb);
                } else {
                    jcp.gws_d[2] = jcp.mb * (jcp.oc / jcp.ocb) * jcp.ngroups;
                }
                break;
            default: return status::unimplemented;
            }
            break;
        default: status = status::unimplemented;
        }

        format_tag_t src_tag, dst_tag, wei_tag;

        switch (jcp.ver) {
        case ver_1stconv:
            if (utils::one_of(src_mdw.data_type(), mkldnn_u8, mkldnn_s8)) {
                src_tag = utils::pick(jcp.ndims - 3, ncw, chwn, ncdhw);
                dst_tag = jcp.mb % 16 == 0
                        ? utils::pick(jcp.ndims - 3, ncw, NChw32n32c, ncdhw)
                        : utils::pick(jcp.ndims - 3, ncw, nChw16c, ncdhw);
                wei_tag = jcp.with_groups
                        ? utils::pick(jcp.ndims - 3, goiw, gOhwi32o, goidhw)
                        : utils::pick(jcp.ndims - 3, oiw, Ohwi32o, oidhw);
            } else {
                src_tag = utils::pick(jcp.ndims - 3, ncw, nchw, ncdhw);
                dst_tag = jcp.mb % 16 == 0
                        ? utils::pick(jcp.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c)
                        : utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
                wei_tag = jcp.with_groups
                        ? utils::pick(jcp.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                        : utils::pick(jcp.ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
            }
            break;
        case ver_16mb16c:
            if (utils::one_of(src_mdw.data_type(), mkldnn_f16)) {
                if (jcp.mb % 32 == 0) {
                    src_tag = utils::pick(jcp.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
                    dst_tag = utils::pick(jcp.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
                    wei_tag = jcp.is_depthwise
                            ? utils::pick(jcp.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                            : (jcp.with_groups
                                ? utils::pick(jcp.ndims - 3, gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i)
                                : utils::pick(jcp.ndims - 3, OIw8i16o2i, OIhw8i16o2i, OIdhw8i16o2i));
                } else {
                    src_tag = utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
                    dst_tag = utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
                    wei_tag = jcp.with_groups
                            ? utils::pick(jcp.ndims - 3, gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                            : utils::pick(jcp.ndims - 3, IOw16i16o, IOhw16i16o, IOdhw16i16o);
                }
            } else if (utils::one_of(
                               src_mdw.data_type(), mkldnn_u8, mkldnn_s8)) {
                src_tag = utils::pick(jcp.ndims - 3, nCw16c, NChw32n32c, ncdhw);
                dst_tag = utils::pick(jcp.ndims - 3, nCw16c, NChw32n32c, ncdhw);
                wei_tag = jcp.with_groups
                        ? utils::pick(jcp.ndims - 3, gOiw16o, gOIhw4o8i8o4i, gOidhw16o)
                        : utils::pick(jcp.ndims - 3, Oiw16o, OIhw4o8i8o4i, Oidhw16o);
            } else {
                src_tag = utils::pick(jcp.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
                dst_tag = utils::pick(jcp.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
                wei_tag = jcp.is_depthwise
                        ? utils::pick(jcp.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                        : (jcp.with_groups
                            ? utils::pick(jcp.ndims - 3, gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                            : utils::pick(jcp.ndims - 3, IOw16i16o, IOhw16i16o, IOdhw16i16o));
            }
            break;
        case ver_8ow16c:
            src_tag = utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = jcp.is_depthwise
                    ? utils::pick(jcp.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (jcp.with_groups
                        ? utils::pick(jcp.ndims - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
                        : utils::pick(jcp.ndims - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o));
            break;
        default: status = status::unimplemented;
        }
        if (status != status::success)
            return status;

        if (src_mdw.format_kind() == format_kind::any) {
            jcp.src_tag = src_tag;
        } else {
            jcp.src_tag = src_mdw.matches_one_of_tag(src_tag);
        }
        if (jcp.src_tag != src_tag)
            return status::unimplemented;

        if (weights_mdw.format_kind() == format_kind::any) {
            jcp.wei_tag = wei_tag;
        } else {
            jcp.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
        }
        if (jcp.wei_tag != wei_tag)
            return status::unimplemented;

        if (dst_mdw.format_kind() == format_kind::any) {
            jcp.dst_tag = dst_tag;
        } else {
            jcp.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
        }
        if (jcp.dst_tag != dst_tag)
            return status::unimplemented;


        jcp.is_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
        jcp.is_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

        return status;
    };

    static status_t init_const_def(ocl_jit_t &jit, const jit_conv_conf_t &jcp) {
        jit.define_int("IS_DW", jcp.is_depthwise);
        jit.define_int("FWD_DATA", 1);
        jit.define_int("G", jcp.ngroups);
        jit.define_int("MB", jcp.mb);
        jit.define_int("IC", jcp.ic);
        jit.define_int("ID", jcp.id);
        jit.define_int("IH", jcp.ih);
        jit.define_int("IW", jcp.iw);
        jit.define_int("OC", jcp.oc);
        jit.define_int("OD", jcp.od);
        jit.define_int("OH", jcp.oh);
        jit.define_int("OW", jcp.ow);
        jit.define_int("KD", jcp.kd);
        jit.define_int("KH", jcp.kh);
        jit.define_int("KW", jcp.kw);
        jit.define_int("SD", jcp.stride_d);
        jit.define_int("SH", jcp.stride_h);
        jit.define_int("SW", jcp.stride_w);
        jit.define_int("PD", jcp.f_pad);
        jit.define_int("PH", jcp.t_pad);
        jit.define_int("PW", jcp.l_pad);
        jit.define_int("PD_R", jcp.back_pad);
        jit.define_int("PH_R", jcp.b_pad);
        jit.define_int("PW_R", jcp.r_pad);
        jit.define_int("DD", jcp.dilate_d);
        jit.define_int("DH", jcp.dilate_h);
        jit.define_int("DW", jcp.dilate_w);
        jit.define_int("OW_PADDED", utils::rnd_up(jcp.ow, 4));
        jit.define_int("OC_PADDED", jcp.oc);
        jit.define_int("OCB", jcp.ocb);
        jit.define_int("MB_BLOCK", jcp.mb_block);
        jit.define_int("OH_BLOCK", jcp.oh_block);
        jit.define_int("OW_BLOCK", jcp.ow_block);
        jit.define_int("OW_LAST", utils::rnd_dn(jcp.ow, jcp.ow_block));
        jit.define_int("OWB", utils::div_up(jcp.ow, jcp.ow_block));
        jit.define_int("OHB", utils::div_up(jcp.oh, jcp.oh_block));
        jit.define_int("WITH_BIAS", jcp.with_bias);
        jit.define_int("WITH_RELU", jcp.with_relu);
        jit.define_int("WITH_SUM", jcp.with_sum);
        jit.define_int("WITH_SUM_RELU", jcp.with_sum_relu);
        jit.define_int("SUM_SCALE", jcp.sum_scale == 1.0);
        jit.define_int("SUB_GROUP_SIZE", jcp.sub_group_size);
        jit.define_int("OC_BLOCK", jcp.oc_block);
        jit.define_int("IC_BLOCK", jcp.ic_block);
        jit.define_int("OC_GROUP", jcp.lws_d[0] / 8);
        jit.define_int("MB_GROUP", 1);
        jit.define_int("SP_GROUP", jcp.lws_d[1]);
        if (jcp.kw == 1)
            jit.define_int("SRC_SP_GROUP", jcp.lws_d[1] + jcp.kw - 1);
        else
            jit.define_int(
                    "SRC_SP_GROUP", jcp.stride_w * (jcp.lws_d[1] - 1) + jcp.kw);
        jit.define_int("SLM_IC", jcp.slm_ic);

        const int use_fast_path = 1 && jcp.scale_idx_mult == 0
                && jcp.ngroups == 1 && !jcp.with_bias;
        jit.define_int("USE_FAST_PATH", use_fast_path);
        jit.define_int("SCALE_IDX_MULT", jcp.scale_idx_mult);
        jit.define_int("RMODE", jcp.rmode);

        jit.set_data_type(jcp.src_data_type);

        switch (jcp.ver) {
        case ver_16mb16c: jit.define_int("VER_16MB16C", 1); break;
        case ver_1stconv:
        case ver_8ow16c: jit.define_int("VER_8OW16C", 1); break;
        default: break;
        }

        jit.define_int("LWS_0", jcp.lws_d[0]);
        jit.define_int("LWS_1", jcp.lws_d[1]);
        jit.define_int("LWS_2", jcp.lws_d[2]);

        if (jcp.is_nchw)
            jit.define_int("NCHW", 1);
        else if (jcp.is_nhwc)
            jit.define_int("NHWC", 1);

#ifdef DEBUG_PRINT
        printf("OPT:\n%s\n", jit.get_options());
#endif
        return status::success;
    }

    jit_conv_conf_t jcp;
};

struct jit_gen9_common_conv_bwd_data_kernel {

    jit_gen9_common_conv_bwd_data_kernel(jit_conv_conf_t ajcp) : jcp(ajcp){};

    ~jit_gen9_common_conv_bwd_data_kernel(){};

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t &diff_src_md,
            const memory_desc_t &weights_md, const memory_desc_t &diff_dst_md,
            const primitive_attr_t &attr) {

        const memory_desc_wrapper src_mdw(&diff_src_md);
        const memory_desc_wrapper weights_mdw(&weights_md);
        const memory_desc_wrapper dst_mdw(&diff_dst_md);

        set_default_conf(jcp, cd, diff_src_md, weights_md, diff_dst_md, attr);

        jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
        jcp.src_data_type = cd.diff_src_desc.data_type;
        jcp.weights_data_type = cd.weights_desc.data_type;
        jcp.dst_data_type = cd.diff_dst_desc.data_type;
        jcp.acc_data_type = cd.accum_data_type;
        jcp.bias_data_type = jcp.with_bias
            ? cd.bias_desc.data_type : data_type::f32;

        const bool is_1stconv = jcp.ic_without_padding == 3;
        const bool is_depthwise = jcp.with_groups && (jcp.ic_without_padding == 1)
            && (jcp.oc_without_padding == 1);
        jcp.is_depthwise = is_depthwise;

        if (is_1stconv || jcp.with_groups) {
            jcp.ic = jcp.ic_without_padding;
            jcp.oc = jcp.oc_without_padding;
        } else {
            jcp.ic = utils::rnd_up(jcp.ic_without_padding, 16);
            jcp.oc = utils::rnd_up(jcp.oc_without_padding, 16);
        }

        if (is_depthwise)
            jcp.ngroups = utils::rnd_up(jcp.ngroups, 16);
        const bool is_dw_16g = (jcp.is_depthwise && jcp.ngroups % 16 == 0);

        const bool is_16ic = jcp.ic % 16 == 0;
        const bool is_16oc = jcp.oc % 16 == 0;
        const bool use_16mb_unroll = true
                && !(jcp.mb == 1 || jcp.mb % 16 != 0)
                && !is_1stconv && ((is_16ic && is_16oc) || is_dw_16g);

        jcp.mb_block = 1;
        jcp.oc_block = 1;
        jcp.ic_block = 1;
        jcp.od_block = 1;
        jcp.oh_block = 1;
        jcp.ow_block = 1;
        if (use_16mb_unroll)
            jcp.ver = ver_16mb16c;
        else if (jcp.mb % 16 != 0 && ((is_16oc && is_16ic) || is_dw_16g))
            jcp.ver = ver_8ow16c;
        else
            return status::unimplemented;

        status_t status = status::success;

        switch (jcp.ver) {
        case ver_16mb16c:
            jcp.mb_block = 16;
            jcp.oc_block = 16;
            jcp.ic_block = 16;
            jcp.od_block = 1;
            jcp.ih_block = 1;
            jcp.iw_block = 1;
            jcp.sub_group_size = 16;
            jcp.lws_d[0] = 1;
            jcp.lws_d[1] = 16;
            jcp.lws_d[2] = 1;
            jcp.gws_d[0] = jcp.ih * jcp.iw * jcp.id;
            jcp.gws_d[1] = jcp.ic * jcp.ngroups;
            jcp.gws_d[2] = jcp.mb / 16;
            break;
        case ver_8ow16c:
            jcp.mb_block = 1;
            jcp.oc_block = 16;
            jcp.ic_block = 16;
            jcp.od_block = 1;
            jcp.ih_block = 1;
            jcp.iw_block = nstl::max(8, utils::max_div(jcp.iw, 16));
            jcp.sub_group_size = 16;
            jcp.lws_d[0] = 1;
            jcp.lws_d[1] = 16;
            jcp.lws_d[2] = 1;
            jcp.gws_d[0] = jcp.ih * utils::div_up(jcp.iw, jcp.iw_block)
                * jcp.id;
            jcp.gws_d[1] = jcp.ic * jcp.ngroups;
            jcp.gws_d[2] = jcp.mb;
            break;
        default: status = status::unimplemented;
        }

        format_tag_t src_tag, dst_tag, wei_tag;

        switch (jcp.ver) {
        case ver_16mb16c:
            src_tag = utils::pick(jcp.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(jcp.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = jcp.is_depthwise
                    ? utils::pick(jcp.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (jcp.with_groups
                        ? utils::pick(jcp.ndims - 3, gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
                        : utils::pick(jcp.ndims - 3, OIw16o16i, OIhw16o16i, OIdhw16o16i));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
                wei_tag = jcp.is_depthwise
                        ? utils::pick(jcp.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                        : (jcp.with_groups
                            ? utils::pick(jcp.ndims - 3, gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
                            : utils::pick(jcp.ndims - 3, OIw16o16i, OIhw16o16i, OIdhw16o16i));
            break;
        default: status = status::unimplemented;
        }
        if (status != status::success)
            return status;

        if (src_mdw.format_kind() == format_kind::any) {
            jcp.src_tag = src_tag;
        } else {
            jcp.src_tag = src_mdw.matches_one_of_tag(src_tag);
        }
        if (jcp.src_tag != src_tag)
            return status::unimplemented;

        if (weights_mdw.format_kind() == format_kind::any) {
            jcp.wei_tag = wei_tag;
        } else {
            jcp.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
        }
        if (jcp.wei_tag != wei_tag)
            return status::unimplemented;

        if (dst_mdw.format_kind() == format_kind::any) {
            jcp.dst_tag = dst_tag;
        } else {
            jcp.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
        }
        if (jcp.dst_tag != dst_tag)
            return status::unimplemented;

        jcp.is_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
        jcp.is_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

        return status::success;
    };

    static status_t init_const_def(ocl_jit_t &jit, const jit_conv_conf_t &jcp) {
        jit.define_int("IS_DW", jcp.is_depthwise);
        jit.define_int("BWD_DATA", 1);
        jit.define_int("G", jcp.ngroups);
        jit.define_int("MB", jcp.mb);
        jit.define_int("IC", jcp.ic);
        jit.define_int("ID", jcp.id);
        jit.define_int("IH", jcp.ih);
        jit.define_int("IW", jcp.iw);
        jit.define_int("OC", jcp.oc);
        jit.define_int("OD", jcp.od);
        jit.define_int("OH", jcp.oh);
        jit.define_int("OW", jcp.ow);
        jit.define_int("KD", jcp.kd);
        jit.define_int("KH", jcp.kh);
        jit.define_int("KW", jcp.kw);
        jit.define_int("SD", jcp.stride_d);
        jit.define_int("SH", jcp.stride_h);
        jit.define_int("SW", jcp.stride_w);
        jit.define_int("PD", jcp.f_pad);
        jit.define_int("PH", jcp.t_pad);
        jit.define_int("PW", jcp.l_pad);
        jit.define_int("PD_R", jcp.back_pad);
        jit.define_int("PH_R", jcp.b_pad);
        jit.define_int("PW_R", jcp.r_pad);
        jit.define_int("DD", jcp.dilate_d);
        jit.define_int("DH", jcp.dilate_h);
        jit.define_int("DW", jcp.dilate_w);
        jit.define_int("OC_PADDED", jcp.oc);
        jit.define_int("MB_BLOCK", jcp.mb_block);
        jit.define_int("IH_BLOCK", jcp.ih_block);
        jit.define_int("IW_BLOCK", jcp.iw_block);
        jit.define_int("IWB", utils::div_up(jcp.iw, jcp.iw_block));
        jit.define_int("WITH_BIAS", jcp.with_bias);
        jit.define_int("WITH_RELU", jcp.with_relu);
        jit.define_int("WITH_SUM", jcp.with_sum);
        jit.define_int("WITH_SUM_RELU", jcp.with_sum_relu);
        jit.define_int("SUM_SCALE", jcp.sum_scale == 1.0);
        jit.define_int("SUB_GROUP_SIZE", jcp.sub_group_size);
        jit.define_int("OC_BLOCK", jcp.oc_block);
        jit.define_int("IC_BLOCK", jcp.ic_block);

        jit.define_int("LWS_0", jcp.lws_d[0]);
        jit.define_int("LWS_1", jcp.lws_d[1]);
        jit.define_int("LWS_2", jcp.lws_d[2]);

        jit.set_data_type(jcp.src_data_type);

        switch (jcp.ver) {
        case ver_16mb16c: jit.define_int("VER_16MB16C", 1); break;
        case ver_8ow16c: jit.define_int("VER_8OW16C", 1); break;
        default: break;
        }

        return status::success;
    }

    jit_conv_conf_t jcp;
};

struct jit_gen9_common_conv_bwd_weights_kernel {

    jit_gen9_common_conv_bwd_weights_kernel(jit_conv_conf_t ajcp) : jcp(ajcp){};

    ~jit_gen9_common_conv_bwd_weights_kernel(){};

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t &src_md,
            const memory_desc_t &diff_weights_md,
            const memory_desc_t &diff_bias_md, const memory_desc_t &diff_dst_md,
            const primitive_attr_t &attr) {

        const memory_desc_wrapper src_mdw(&src_md);
        const memory_desc_wrapper weights_mdw(&diff_weights_md);
        const memory_desc_wrapper dst_mdw(&diff_dst_md);
        const memory_desc_wrapper bias_mdw(&diff_bias_md);

        set_default_conf(jcp, cd, src_md, diff_weights_md, diff_dst_md, attr);

        jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;
        jcp.src_data_type = cd.src_desc.data_type;
        jcp.weights_data_type = cd.diff_weights_desc.data_type;
        jcp.dst_data_type = cd.diff_dst_desc.data_type;
        jcp.acc_data_type = cd.accum_data_type;
        jcp.bias_data_type = jcp.with_bias
            ? cd.diff_bias_desc.data_type : data_type::f32;

        const bool is_1stconv = jcp.ic_without_padding == 3;
        const bool is_depthwise =
            jcp.with_groups && (jcp.ic_without_padding == 1)
            && (jcp.oc_without_padding == 1);
        jcp.is_depthwise = is_depthwise;

        if (is_1stconv || jcp.with_groups) {
            jcp.ic = jcp.ic_without_padding;
            jcp.oc = jcp.oc_without_padding;
        } else {
            jcp.ic = utils::rnd_up(jcp.ic_without_padding, 16);
            jcp.oc = utils::rnd_up(jcp.oc_without_padding, 16);
        }

        if (is_depthwise)
            jcp.ngroups = utils::rnd_up(jcp.ngroups, 16);
        const bool is_dw_16g = (jcp.is_depthwise && jcp.ngroups % 16 == 0);

        const bool is_16ic = jcp.ic % 16 == 0;
        const bool is_16oc = jcp.oc % 16 == 0;
        const bool use_16mb_unroll = true && !(jcp.mb == 1 || jcp.mb % 16 != 0)
                && !is_1stconv && ((is_16ic && is_16oc) || is_dw_16g);

        jcp.mb_block = 1;
        jcp.oc_block = 1;
        jcp.ic_block = 1;
        jcp.od_block = 1;
        jcp.oh_block = 1;
        jcp.ow_block = 1;
        jcp.oh_chunk = 1;
        jcp.mb_chunk = 1;
        if (use_16mb_unroll)
            jcp.ver = ver_16mb16c;
        else if (jcp.mb % 16 != 0 && ((is_16oc && is_16ic) || is_dw_16g)
                && (jcp.iw >= jcp.stride_w * 8))
            jcp.ver = ver_8ow16c;
        else if (is_1stconv && is_16oc)
            jcp.ver = ver_1stconv;
        else
            return status::unimplemented;

        status_t status = status::success;

        const int nthr = jcp.is_depthwise
            ? jcp.kh * jcp.kw * jcp.kd * jcp.ngroups
            : jcp.kh * jcp.kw * jcp.kd * (jcp.oc / 16) * (jcp.ic / 16) * jcp.ngroups;
        int opt_chunk = 1;

        switch (jcp.ver) {
        case ver_1stconv:
        case ver_8ow16c:
            jcp.mb_block = 1;
            jcp.oc_block = 16;
            jcp.ic_block = jcp.ver == ver_8ow16c ? 16 : 1;
            jcp.ow_block = 8;

            /* 2KB per thread (72 EU and 7 thr/EU)*/
            opt_chunk =
            utils::div_up((jcp.ic_block * jcp.ih * jcp.iw * jcp.id
                + jcp.oc_block * jcp.oh * jcp.ow * jcp.od) * jcp.mb * 4,
                1024 * 2 * 72 * 7);
            jcp.oh_chunk = 1;
            jcp.mb_chunk = utils::div_up(jcp.mb, opt_chunk);
            jcp.nchunk = jcp.oh_chunk * jcp.mb_chunk;
            jcp.oh_block
                    = utils::div_up(jcp.od * jcp.oh * jcp.ow, jcp.oh_chunk);
            jcp.sub_group_size = 16;
            jcp.lws_d[0] = 16;
            jcp.lws_d[1] = 1;
            jcp.lws_d[2] = 1;
            if (jcp.is_depthwise) {
                jcp.gws_d[0] = jcp.ngroups;
            } else {
                jcp.gws_d[0] = jcp.ver == ver_8ow16c
                        ? jcp.oc * (jcp.ic / 16) * jcp.ngroups
                        : jcp.oc * jcp.ngroups;
            }
            jcp.gws_d[1] = jcp.kh * jcp.kw * jcp.kd;
            jcp.gws_d[2] = jcp.nchunk;
            break;
        case ver_16mb16c:
            jcp.mb_block = 16;
            jcp.oc_block = 16;
            jcp.ic_block = 16;
            jcp.ow_block = 1;

            /* 2KB per thread (72 EU and 7 thr/EU)*/
            opt_chunk = nstl::max(utils::div_up(4096, nthr),
                utils::div_up((jcp.ic_block * jcp.ih * jcp.iw * jcp.id
                + jcp.oc_block * jcp.oh * jcp.ow * jcp.od) * jcp.mb * 4,
                1024 * 2 * 72 * 7));
            jcp.oh_chunk = nstl::min(jcp.oh * jcp.ow * jcp.od, opt_chunk);
            jcp.mb_chunk = nstl::min(jcp.mb / jcp.mb_block,
                utils::div_up(opt_chunk, jcp.oh_chunk));
            jcp.nchunk = jcp.oh_chunk * jcp.mb_chunk;
            jcp.oh_block
                    = utils::div_up(jcp.od * jcp.oh * jcp.ow, jcp.oh_chunk);
            jcp.sub_group_size = 16;
            jcp.lws_d[0] = 16;
            jcp.lws_d[1] = 1;
            jcp.lws_d[2] = 1;
            if (jcp.is_depthwise) {
                jcp.gws_d[0] = jcp.ngroups;
            } else {
                jcp.gws_d[0] = jcp.oc * (jcp.ic / 16) * jcp.ngroups;
            }
            jcp.gws_d[1] = jcp.kh * jcp.kw * jcp.kd;
            jcp.gws_d[2] = jcp.nchunk;
            break;
        default: status = status::unimplemented;
        }

        format_tag_t src_tag, dst_tag, wei_tag;

        switch (jcp.ver) {
        case ver_1stconv:
            assert(!jcp.is_depthwise);
            src_tag = utils::pick(jcp.ndims - 3, ncw, nchw, ncdhw);
            dst_tag = utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = jcp.with_groups
                    ? utils::pick(jcp.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                    : utils::pick(jcp.ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
            break;
        case ver_16mb16c:
            src_tag = utils::pick(jcp.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(jcp.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = jcp.is_depthwise
                    ? utils::pick(jcp.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (jcp.with_groups
                        ? utils::pick(jcp.ndims - 3, gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                        : utils::pick(jcp.ndims - 3, IOw16i16o, IOhw16i16o, IOdhw16i16o));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(jcp.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = jcp.is_depthwise
                    ? utils::pick(jcp.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (jcp.with_groups
                        ? utils::pick(jcp.ndims - 3, gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                        : utils::pick(jcp.ndims - 3, IOw16i16o, IOhw16i16o, IOdhw16i16o));
            break;
        default: status = status::unimplemented;
        }
        if (status != status::success)
            return status;

        if (src_mdw.format_kind() == format_kind::any) {
            jcp.src_tag = src_tag;
        } else {
            jcp.src_tag = src_mdw.matches_one_of_tag(src_tag);
        }
        if (jcp.src_tag != src_tag)
            return status::unimplemented;

        if (weights_mdw.format_kind() == format_kind::any) {
            jcp.wei_tag = wei_tag;
        } else {
            jcp.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
        }
        if (jcp.wei_tag != wei_tag)
            return status::unimplemented;

        if (dst_mdw.format_kind() == format_kind::any) {
            jcp.dst_tag = dst_tag;
        } else {
            jcp.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
        }
        if (jcp.dst_tag != dst_tag)
            return status::unimplemented;

        return status::success;
    };

    static status_t init_const_def(ocl_jit_t &jit, const jit_conv_conf_t &jcp) {
        jit.define_int("IS_DW", jcp.is_depthwise);
        jit.define_int("BWD_WEIGHTS", 1);
        jit.define_int("G", jcp.ngroups);
        jit.define_int("MB", jcp.mb);
        jit.define_int("IC", jcp.ic);
        jit.define_int("ID", jcp.id);
        jit.define_int("IH", jcp.ih);
        jit.define_int("IW", jcp.iw);
        jit.define_int("OC", jcp.oc);
        jit.define_int("OD", jcp.od);
        jit.define_int("OH", jcp.oh);
        jit.define_int("OW", jcp.ow);
        jit.define_int("KD", jcp.kd);
        jit.define_int("KH", jcp.kh);
        jit.define_int("KW", jcp.kw);
        jit.define_int("SD", jcp.stride_d);
        jit.define_int("SH", jcp.stride_h);
        jit.define_int("SW", jcp.stride_w);
        jit.define_int("PD", jcp.f_pad);
        jit.define_int("PH", jcp.t_pad);
        jit.define_int("PW", jcp.l_pad);
        jit.define_int("PD_R", jcp.back_pad);
        jit.define_int("PH_R", jcp.b_pad);
        jit.define_int("PW_R", jcp.r_pad);
        jit.define_int("DD", jcp.dilate_d);
        jit.define_int("DH", jcp.dilate_h);
        jit.define_int("DW", jcp.dilate_w);
        jit.define_int("OC_PADDED", jcp.oc);
        jit.define_int("OH_BLOCK", jcp.oh_block);
        jit.define_int("WITH_BIAS", jcp.with_bias);
        jit.define_int("WITH_RELU", jcp.with_relu);
        jit.define_int("WITH_SUM", jcp.with_sum);
        jit.define_int("WITH_SUM_RELU", jcp.with_sum_relu);
        jit.define_int("SUM_SCALE", jcp.sum_scale == 1.0);
        jit.define_int("SUB_GROUP_SIZE", jcp.sub_group_size);
        jit.define_int("MB_BLOCK", jcp.mb_block);
        jit.define_int("OC_BLOCK", jcp.oc_block);
        jit.define_int("IC_BLOCK", jcp.ic_block);
        jit.define_int("NCHUNK", jcp.nchunk);
        jit.define_int("OH_CHUNK", jcp.oh_chunk);
        jit.define_int("MB_CHUNK", jcp.mb_chunk);
        jit.define_int("MB_CHUNK_SIZE", utils::div_up(jcp.mb, jcp.mb_chunk));
        jit.define_int("OW_BLOCK", jcp.ow_block);
        jit.define_int("OW_LAST", utils::rnd_dn(jcp.ow, jcp.ow_block));

        jit.define_int("LWS_0", jcp.lws_d[0]);
        jit.define_int("LWS_1", jcp.lws_d[1]);
        jit.define_int("LWS_2", jcp.lws_d[2]);

        switch (jcp.ver) {
        case ver_16mb16c: jit.define_int("VER_16MB16C", 1); break;
        case ver_1stconv:
        case ver_8ow16c: jit.define_int("VER_8OW16C", 1); break;
        default: break;
        }

        return status::success;
    }

    jit_conv_conf_t jcp;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

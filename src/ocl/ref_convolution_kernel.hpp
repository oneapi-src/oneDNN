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

#ifndef OCL_REF_CONVOLUTION_KERNEL_HPP
#define OCL_REF_CONVOLUTION_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {
struct ref_convolution_kernel_t {
    ref_convolution_kernel_t() = default;

    status_t init(const convolution_pd_t *pd) {

        const convolution_desc_t &cd = *pd->desc();
        const memory_desc_t &src_md = *pd->invariant_src_md();
        const memory_desc_t &weights_md = *pd->invariant_wei_md();
        const memory_desc_t &dst_md = *pd->invariant_dst_md();
        const primitive_attr_t &attr = *pd->attr();

        set_default_conf(conf_, cd, src_md, weights_md, dst_md, attr);

        set_offsets(src_md, off_.src_off);
        set_offsets(weights_md, off_.wht_off);
        set_offsets(dst_md, off_.dst_off);

        int sp_dims = 1;
        switch (cd.prop_kind) {
            case prop_kind::forward_training:
            case prop_kind::forward_inference:
                conf_.with_bias
                        = cd.bias_desc.format_kind != format_kind::undef;
                conf_.src_data_type = cd.src_desc.data_type;
                conf_.weights_data_type = cd.weights_desc.data_type;
                conf_.dst_data_type = cd.dst_desc.data_type;
                conf_.acc_data_type = cd.accum_data_type;
                conf_.bias_data_type = conf_.with_bias ? cd.bias_desc.data_type
                                                       : data_type::f32;
                conf_.gws_d[0] = dst_md.dims[0];
                conf_.gws_d[1] = dst_md.dims[1];
                for (int i = 2; i < dst_md.ndims; ++i)
                    sp_dims *= dst_md.dims[i];
                conf_.gws_d[2] = sp_dims;
                break;
            case prop_kind::backward_data:
                conf_.with_bias
                        = cd.bias_desc.format_kind != format_kind::undef;
                conf_.src_data_type = cd.diff_src_desc.data_type;
                conf_.weights_data_type = cd.weights_desc.data_type;
                conf_.dst_data_type = cd.diff_dst_desc.data_type;
                conf_.acc_data_type = cd.accum_data_type;
                conf_.bias_data_type = conf_.with_bias ? cd.bias_desc.data_type
                                                       : data_type::f32;
                conf_.gws_d[0] = src_md.dims[0];
                conf_.gws_d[1] = src_md.dims[1];
                for (int i = 2; i < src_md.ndims; ++i)
                    sp_dims *= src_md.dims[i];
                conf_.gws_d[2] = sp_dims;
                break;
            case prop_kind::backward_weights:
                conf_.with_bias
                        = cd.diff_bias_desc.format_kind != format_kind::undef;
                conf_.src_data_type = cd.src_desc.data_type;
                conf_.weights_data_type = cd.diff_weights_desc.data_type;
                conf_.dst_data_type = cd.diff_dst_desc.data_type;
                conf_.acc_data_type = cd.accum_data_type;
                conf_.bias_data_type = conf_.with_bias
                        ? cd.diff_bias_desc.data_type
                        : data_type::f32;
                conf_.gws_d[0] = conf_.with_groups ? weights_md.dims[0] : 1;
                conf_.gws_d[1] = conf_.with_groups
                        ? weights_md.dims[1] * weights_md.dims[2]
                        : weights_md.dims[0] * weights_md.dims[1];
                for (int i = 2 + conf_.with_groups; i < weights_md.ndims; ++i)
                    sp_dims *= weights_md.dims[i];
                conf_.gws_d[2] = sp_dims;
                break;
            default: break;
        }

        conf_.lws_d[0] = 0;
        conf_.lws_d[1] = 0;
        conf_.lws_d[2] = 0;

        return status::success;
    }

    status_t apply_const(compute::kernel_ctx_t &kernel_ctx) const {
        kernel_ctx.define_int("NDIMS", conf_.ndims);
        kernel_ctx.define_int("G", conf_.ngroups);
        kernel_ctx.define_int("WITH_GROUPS", conf_.with_groups);
        kernel_ctx.define_int("MB", conf_.mb);
        kernel_ctx.define_int("IC", conf_.ic);
        kernel_ctx.define_int("ID", conf_.id);
        kernel_ctx.define_int("IH", conf_.ih);
        kernel_ctx.define_int("IW", conf_.iw);
        kernel_ctx.define_int("OC", conf_.oc);
        kernel_ctx.define_int("OD", conf_.od);
        kernel_ctx.define_int("OH", conf_.oh);
        kernel_ctx.define_int("OW", conf_.ow);
        kernel_ctx.define_int("KD", conf_.kd);
        kernel_ctx.define_int("KH", conf_.kh);
        kernel_ctx.define_int("KW", conf_.kw);
        kernel_ctx.define_int("SD", conf_.stride_d);
        kernel_ctx.define_int("SH", conf_.stride_h);
        kernel_ctx.define_int("SW", conf_.stride_w);
        kernel_ctx.define_int("PD", conf_.f_pad);
        kernel_ctx.define_int("PH", conf_.t_pad);
        kernel_ctx.define_int("PW", conf_.l_pad);
        kernel_ctx.define_int("PD_R", conf_.back_pad);
        kernel_ctx.define_int("PH_R", conf_.b_pad);
        kernel_ctx.define_int("PW_R", conf_.r_pad);
        kernel_ctx.define_int("DD", conf_.dilate_d);
        kernel_ctx.define_int("DH", conf_.dilate_h);
        kernel_ctx.define_int("DW", conf_.dilate_w);
        kernel_ctx.define_int("WITH_BIAS", conf_.with_bias);
        kernel_ctx.define_int("SUB_GROUP_SIZE", conf_.sub_group_size);

        def_offsets(off_.src_off, kernel_ctx, "SRC", conf_.ndims);
        def_offsets(off_.wht_off, kernel_ctx, "WHT",
                conf_.ndims + conf_.with_groups);
        def_offsets(off_.bias_off, kernel_ctx, "BIA", 1);
        def_offsets(off_.dst_off, kernel_ctx, "DST", conf_.ndims);

        switch (conf_.prop_kind) {
            case prop_kind::forward_training:
            case prop_kind::forward_inference:
                kernel_ctx.set_data_type(conf_.dst_data_type);
                break;
            case prop_kind::backward_data:
                kernel_ctx.set_data_type(conf_.src_data_type);
                break;
            case prop_kind::backward_weights:
                kernel_ctx.set_data_type(conf_.weights_data_type);
                break;
            default: break;
        }

        def_data_type(kernel_ctx, conf_.src_data_type, "SRC");
        def_data_type(kernel_ctx, conf_.weights_data_type, "WEI");
        def_data_type(kernel_ctx, conf_.bias_data_type, "BIA");
        def_data_type(kernel_ctx, conf_.dst_data_type, "DST");
        def_data_type(kernel_ctx, conf_.acc_data_type, "ACC");

        if (conf_.with_eltwise || conf_.with_post_sum_eltwise) {
            def_postops(kernel_ctx, conf_.eltwise.alg);
        }
        kernel_ctx.define_int("WITH_ELTWISE", conf_.with_eltwise);
        kernel_ctx.define_int("WITH_SUM", conf_.with_sum);
        kernel_ctx.define_int(
                "WITH_POST_SUM_ELTWISE", conf_.with_post_sum_eltwise);

        return status::success;
    }

    const size_t *gws() const { return conf_.gws_d; }
    const size_t *lws() const { return conf_.lws_d; }

private:
    jit_conv_conf_t conf_;
    jit_offsets off_;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

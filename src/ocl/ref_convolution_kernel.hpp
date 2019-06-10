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

#ifndef REF_CONV_KERNEL_HPP
#define REF_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {
struct ref_convolution_kernel_t {
    ref_convolution_kernel_t() = default;

    status_t init(const convolution_desc_t &cd, const memory_desc_t &src_md,
            const memory_desc_t &weights_md, const memory_desc_t &bias_md,
            const memory_desc_t &dst_md, const primitive_attr_t &attr) {

        set_default_conf(conf_, cd, src_md, weights_md, dst_md, attr);

        set_offsets(src_md, off_.src_off);
        set_offsets(weights_md, off_.wht_off);
        set_offsets(dst_md, off_.dst_off);

        int sp_dims = 1;
        switch(cd.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            conf_.with_bias = cd.bias_desc.format_kind != format_kind::undef;
            conf_.src_data_type = cd.src_desc.data_type;
            conf_.weights_data_type = cd.weights_desc.data_type;
            conf_.dst_data_type = cd.dst_desc.data_type;
            conf_.acc_data_type = cd.accum_data_type;
            conf_.bias_data_type = conf_.with_bias
                ? cd.bias_desc.data_type : data_type::f32;
            conf_.gws_d[0] = dst_md.dims[0];
            conf_.gws_d[1] = dst_md.dims[1];
            for (int i = 2; i < dst_md.ndims; ++i)
                sp_dims *= dst_md.dims[i];
            conf_.gws_d[2] = sp_dims;
        break;
        case prop_kind::backward_data:
            conf_.with_bias = cd.bias_desc.format_kind != format_kind::undef;
            conf_.src_data_type = cd.diff_src_desc.data_type;
            conf_.weights_data_type = cd.weights_desc.data_type;
            conf_.dst_data_type = cd.diff_dst_desc.data_type;
            conf_.acc_data_type = cd.accum_data_type;
            conf_.bias_data_type = conf_.with_bias
                ? cd.bias_desc.data_type : data_type::f32;
            conf_.gws_d[0] = src_md.dims[0];
            conf_.gws_d[1] = src_md.dims[1];
            for (int i = 2; i < src_md.ndims; ++i)
                sp_dims *= src_md.dims[i];
            conf_.gws_d[2] = sp_dims;
        break;
        case prop_kind::backward_weights:
            conf_.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;
            conf_.src_data_type = cd.src_desc.data_type;
            conf_.weights_data_type = cd.diff_weights_desc.data_type;
            conf_.dst_data_type = cd.diff_dst_desc.data_type;
            conf_.acc_data_type = cd.accum_data_type;
            conf_.bias_data_type = conf_.with_bias
                ? cd.diff_bias_desc.data_type : data_type::f32;
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

    status_t apply_const(ocl_jit_t &jit, bool with_eltwise,
        bool with_sum, alg_kind_t alg) const {
        jit.define_int("NDIMS", conf_.ndims);
        jit.define_int("G", conf_.ngroups);
        jit.define_int("WITH_GROUPS", conf_.with_groups);
        jit.define_int("MB", conf_.mb);
        jit.define_int("IC", conf_.ic);
        jit.define_int("ID", conf_.id);
        jit.define_int("IH", conf_.ih);
        jit.define_int("IW", conf_.iw);
        jit.define_int("OC", conf_.oc);
        jit.define_int("OD", conf_.od);
        jit.define_int("OH", conf_.oh);
        jit.define_int("OW", conf_.ow);
        jit.define_int("KD", conf_.kd);
        jit.define_int("KH", conf_.kh);
        jit.define_int("KW", conf_.kw);
        jit.define_int("SD", conf_.stride_d);
        jit.define_int("SH", conf_.stride_h);
        jit.define_int("SW", conf_.stride_w);
        jit.define_int("PD", conf_.f_pad);
        jit.define_int("PH", conf_.t_pad);
        jit.define_int("PW", conf_.l_pad);
        jit.define_int("PD_R", conf_.back_pad);
        jit.define_int("PH_R", conf_.b_pad);
        jit.define_int("PW_R", conf_.r_pad);
        jit.define_int("DD", conf_.dilate_d);
        jit.define_int("DH", conf_.dilate_h);
        jit.define_int("DW", conf_.dilate_w);
        jit.define_int("WITH_BIAS", conf_.with_bias);
        jit.define_int("SUB_GROUP_SIZE", conf_.sub_group_size);

        def_offsets(off_.src_off, jit, "SRC", conf_.ndims);
        def_offsets(off_.wht_off, jit, "WHT", conf_.ndims + conf_.with_groups);
        def_offsets(off_.bias_off, jit, "BIAS", 1);
        def_offsets(off_.dst_off, jit, "DST", conf_.ndims);

        if (conf_.src_data_type == data_type::f16)
            jit.set_data_type(data_type::f16);
        else
            jit.set_data_type(data_type::f32);

        def_data_type(jit, conf_.src_data_type, "SRC");
        def_data_type(jit, conf_.weights_data_type, "WEI");
        def_data_type(jit, conf_.bias_data_type, "BIAS");
        def_data_type(jit, conf_.dst_data_type, "DST");
        def_data_type(jit, conf_.acc_data_type, "ACC");

        if (with_eltwise) {
            def_postops(jit, alg);
        }
        jit.define_int("WITH_ELTWISE",with_eltwise);
        jit.define_int("WITH_SUM",with_sum);
        jit.define_int("WITH_SUM_ELTWISE",with_sum && with_eltwise);

        return status::success;
    }

    const size_t *gws() const { return conf_.gws_d; }
    const size_t *lws() const { return conf_.lws_d; }

private:
    jit_conv_conf_t conf_;
    jit_offsets off_;
};

}
}
}

#endif

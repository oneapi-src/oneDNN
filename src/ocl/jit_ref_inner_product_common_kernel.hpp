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

#ifndef JIT_REF_INNER_PRODUCT_COMMON_KERNEL_HPP
#define JIT_REF_INNER_PRODUCT_COMMON_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_ref_inner_product_fwd_kernel {

    jit_ref_inner_product_fwd_kernel(const jit_inner_product_conf_t &ajip)
        : jip(ajip) {}

    ~jit_ref_inner_product_fwd_kernel() {}

    static status_t init_conf(jit_inner_product_conf_t &jip,
            const inner_product_desc_t &ipd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr,
            jit_offsets &jit_off, data_type_t acc_data_type) {

        const int ndims = src_d.ndims();

        jip.ndims = ndims;
        jip.has_spatial = utils::one_of(jip.ndims, 3, 4, 5);

        const auto &src_dims = src_d.padded_dims();

        jip.mb = src_dims[0];
        jip.ic = src_dims[1];
        if (jip.has_spatial) {
            jip.id = (ndims == 5) ? src_dims[2] : 1;
            jip.ih = (ndims == 3) ? 1 : src_dims[ndims - 2];
            jip.iw = src_dims[ndims - 1];
        } else {
            jip.id = 1;
            jip.ih = 1;
            jip.iw = 1;
        }
        jip.ic_total = utils::array_product(&src_dims[1], jip.ndims - 1);

        const auto &dst_dims = dst_d.padded_dims();

        jip.oc = dst_dims[1];
        jip.od = (ndims == 5) ? dst_dims[2] : 1;
        jip.oh = (ndims == 3) ? 1 : dst_dims[ndims - 2];
        jip.ow = dst_dims[ndims - 1];

        jip.kd = (ndims == 5) ? weights_d.dims()[2] : 1;
        jip.kh = (ndims == 3) ? 1 : weights_d.dims()[ndims - 2];
        jip.kw = weights_d.dims()[ndims - 1];

        jip.src_dt = src_d.data_type();
        jip.wei_dt = weights_d.data_type();
        jip.dst_dt = dst_d.data_type();
        jip.acc_dt = acc_data_type;

        jip.is_forward = utils::one_of(ipd.prop_kind, prop_kind::forward,
                prop_kind::forward_inference);
        jip.is_backward_data = ipd.prop_kind == prop_kind::backward_data;
        jip.is_backward_weights = ipd.prop_kind == prop_kind::backward_weights;

        if (jip.is_forward) {
            jip.with_bias = ipd.bias_desc.format_kind != format_kind::undef;
            jip.bia_dt
                    = jip.with_bias ? ipd.bias_desc.data_type : data_type::f32;
        } else if (jip.is_backward_weights) {
            jip.with_bias
                    = ipd.diff_bias_desc.format_kind != format_kind::undef;
            jip.bia_dt = jip.with_bias ? ipd.diff_bias_desc.data_type
                                       : data_type::f32;
        } else {
            jip.with_bias = 0;
            jip.bia_dt = data_type::f32;
        }

        set_offsets(src_d, jit_off.src_off);
        set_offsets(weights_d, jit_off.wht_off);
        set_offsets(dst_d, jit_off.dst_off);

        return status::success;
    };

    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            const jit_inner_product_conf_t &jip, const jit_offsets &jit_off,
            bool with_eltwise, bool with_sum, alg_kind_t alg) {

        kernel_ctx.define_int("NDIMS", jip.ndims);
        kernel_ctx.define_int("MB", jip.mb);
        kernel_ctx.define_int("OC", jip.oc);
        kernel_ctx.define_int("IC", jip.ic);
        kernel_ctx.define_int("IC_TOTAL", jip.ic_total);
        kernel_ctx.define_int("ID", jip.id);
        kernel_ctx.define_int("IH", jip.ih);
        kernel_ctx.define_int("IW", jip.iw);
        kernel_ctx.define_int("OD", jip.od);
        kernel_ctx.define_int("OH", jip.oh);
        kernel_ctx.define_int("OW", jip.ow);
        kernel_ctx.define_int("KD", jip.kd);
        kernel_ctx.define_int("KH", jip.kh);
        kernel_ctx.define_int("KW", jip.kw);
        if (jip.with_bias) kernel_ctx.define_int("WITH_BIAS", 1);
        if (jip.has_spatial) kernel_ctx.define_int("HAS_SPATIAL", 1);

        if (jip.is_forward)
            kernel_ctx.define_int("INNER_PRODUCT_FWD", 1);
        else if (jip.is_backward_data)
            kernel_ctx.define_int("INNER_PRODUCT_BWD_DATA", 1);
        else if (jip.is_backward_weights)
            kernel_ctx.define_int("INNER_PRODUCT_BWD_WEIGHTS", 1);

        if (with_eltwise) { def_postops(kernel_ctx, alg); }
        kernel_ctx.define_int("WITH_ELTWISE", with_eltwise);
        kernel_ctx.define_int("WITH_SUM", with_sum);
        kernel_ctx.define_int("WITH_SUM_ELTWISE", with_sum && with_eltwise);

        def_offsets(jit_off.src_off, kernel_ctx, "SRC", jip.ndims);
        def_offsets(jit_off.wht_off, kernel_ctx, "WHT", jip.ndims);
        def_offsets(jit_off.dst_off, kernel_ctx, "DST", jip.ndims);

        if (jip.src_dt == data_type::f16)
            kernel_ctx.set_data_type(data_type::f16);
        else
            kernel_ctx.set_data_type(data_type::f32);

        def_data_type(kernel_ctx, jip.src_dt, "SRC");
        def_data_type(kernel_ctx, jip.wei_dt, "WEI");
        def_data_type(kernel_ctx, jip.bia_dt, "BIA");
        def_data_type(kernel_ctx, jip.dst_dt, "DST");
        def_data_type(kernel_ctx, jip.acc_dt, "ACC");

        return status::success;
    }

    jit_inner_product_conf_t jip;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

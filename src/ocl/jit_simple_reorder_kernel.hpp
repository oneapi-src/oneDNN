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

#ifndef JIT_SIMPLE_REORDER_KERNEL_HPP
#define JIT_SIMPLE_REORDER_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/reorder_pd.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace mkldnn::impl::format_tag;

struct jit_simple_reorder_kernel {

    jit_simple_reorder_kernel(jit_reorder_conf_t ajrp) : jrp(ajrp){};

    ~jit_simple_reorder_kernel(){};

    static status_t init_conf(const reorder_pd_t *pd, jit_reorder_conf_t &jrp,
            const memory_desc_wrapper &input_md,
            const memory_desc_wrapper &output_md) {

        status_t status = status::success;

        const auto &dims = output_md.padded_dims();
        jrp.is_alpha_beta = (pd->alpha() != 1.0 || pd->beta() != 0.0);
        jrp.do_reorder = jrp.is_alpha_beta ? true : input_md != output_md;
        jrp.has_padding = !input_md.is_dense() || !output_md.is_dense();
        jrp.ndims = input_md.ndims();
        jrp.nelems = utils::array_product(dims, jrp.ndims);
        jrp.par_dims = 4;
        jrp.ker_dims = jrp.ndims - jrp.par_dims;
        jrp.last_dims = jrp.ndims - 2;
        jrp.lws_d[0] = 1;
        jrp.lws_d[1] = 1;
        jrp.lws_d[2] = 1;
        jrp.with_group = 0;
        jrp.sub_group_size = 1;
        switch (jrp.ndims) {
        case 1:
            jrp.gws_d[0] = dims[0];
            jrp.block[0] = 1;
            jrp.gws_d[1] = 1;
            jrp.block[1] = 1;
            jrp.gws_d[2] = 1;
            jrp.block[2] = 1;
            break;
        case 2:
            jrp.gws_d[0] = dims[0] * dims[1];
            jrp.block[0] = dims[1];
            jrp.gws_d[1] = 1;
            jrp.block[1] = 1;
            jrp.gws_d[2] = 1;
            jrp.block[2] = 1;
            break;
        case 3:
            jrp.gws_d[0] = dims[0] * dims[1];
            jrp.block[0] = dims[1];
            jrp.gws_d[1] = dims[2];
            jrp.block[1] = 1;
            jrp.gws_d[2] = 1;
            jrp.block[2] = 1;
            break;
        case 4:
            jrp.gws_d[0] = dims[0] * dims[1];
            jrp.block[0] = dims[1];
            jrp.gws_d[1] = dims[2] * dims[3];
            jrp.block[1] = dims[3];
            jrp.gws_d[2] = 1;
            jrp.block[2] = 1;
            break;
        case 5:
            jrp.gws_d[0] = dims[0] * dims[1];
            jrp.block[0] = dims[1];
            jrp.gws_d[1] = dims[2] * dims[3];
            jrp.block[1] = dims[3];
            jrp.gws_d[2] = dims[4];
            jrp.block[2] = 1;
            break;
        case 6:
            jrp.gws_d[0] = dims[0] * dims[1];
            jrp.block[0] = dims[1];
            jrp.gws_d[1] = dims[2] * dims[3];
            jrp.block[1] = dims[3];
            jrp.gws_d[2] = dims[4] * dims[5];
            jrp.block[2] = dims[5];
            break;
        default: status = status::unimplemented; break;
        }

        if (input_md.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o, gOIw8i16o2i,
                    gOIhw8i16o2i, gOIdhw8i16o2i, gOIhw4o8i8o4i, gOIhw2o8i8o2i)
                || output_md.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o,
                        gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i, gOIhw4o8i8o4i,
                        gOIhw2o8i8o2i))
            jrp.with_group = 1;

        if (jrp.has_padding || jrp.is_alpha_beta)
            return status;

        const bool type_f32 = input_md.data_type() == mkldnn_f32
                && output_md.data_type() == mkldnn_f32;

        const bool type_f32_f16
                = utils::one_of(input_md.data_type(), mkldnn_f32, mkldnn_f16)
                && utils::one_of(output_md.data_type(), mkldnn_f32, mkldnn_f16);

        if (type_f32_f16
                && (input_md.matches_tag(NChw16n16c)
                           || output_md.matches_tag(NChw16n16c))) {
            jrp.sub_group_size = 16;
            jrp.lws_d[1] = 16;
            jrp.gws_d[0] = dims[0] / 16;
            jrp.block[0] = 1;
            jrp.gws_d[1] = dims[1];
            jrp.block[1] = 1;
            jrp.gws_d[2] = dims[2] * dims[3];
            jrp.block[2] = dims[3];
        } else if (type_f32_f16
                && (input_md.matches_tag(nChw16c)
                           || output_md.matches_tag(nChw16c))) {
            jrp.sub_group_size = 16;
            jrp.lws_d[1] = 16;
            jrp.gws_d[0] = dims[0];
            jrp.block[0] = 1;
            jrp.gws_d[1] = dims[1];
            jrp.block[1] = 1;
            jrp.gws_d[2] = dims[2] * dims[3];
            jrp.block[2] = dims[3];
        } else if (type_f32
                && (input_md.matches_one_of_tag(
                            IOhw16i16o, OIhw16o16i, gIOhw16i16o, gOIhw16o16i)
                           || output_md.matches_one_of_tag(IOhw16i16o,
                                      OIhw16o16i, gIOhw16i16o, gOIhw16o16i))) {
            jrp.with_group
                    = input_md.matches_one_of_tag(gIOhw16i16o, gOIhw16o16i)
                    || output_md.matches_one_of_tag(gIOhw16i16o, gOIhw16o16i);
            jrp.sub_group_size = 16;
            jrp.lws_d[0] = 16;
            jrp.gws_d[0] = jrp.with_group ? dims[0] * dims[1] : dims[0];
            jrp.gws_d[1] = dims[jrp.with_group + 1] / 16;
            jrp.gws_d[2] = dims[jrp.with_group + 2] * dims[jrp.with_group + 3];
            jrp.block[0] = jrp.with_group ? dims[1] : dims[0];
            jrp.block[1] = 1;
            jrp.block[2] = dims[jrp.with_group + 3];
        }

        return status;
    };

    static status_t init_const_def(ocl_jit_t &jit,
            const jit_reorder_conf_t &jrp, const memory_desc_wrapper &input_md,
            const memory_desc_wrapper &output_md) {

        jit.define_int("NDIMS", jrp.ndims);
        jit.define_int("PAR_DIMS", jrp.par_dims);
        jit.define_int("KER_DIMS", jrp.ker_dims);
        jit.define_int("LAST_DIMS", jrp.last_dims);
        jit.define_int("ALPHA_BETA", jrp.is_alpha_beta);
        jit.define_int("WITH_GROUP", jrp.with_group);

        jit.define_int("LWS_0", jrp.lws_d[0]);
        jit.define_int("LWS_1", jrp.lws_d[1]);
        jit.define_int("LWS_2", jrp.lws_d[2]);

        for (int i = 0; i < 3; i++) {
            char tempstr[32];
            snprintf(tempstr, 32, "BLOCK_%d", i);
            jit.define_int(tempstr, jrp.block[i]);
        }

        auto input_type = input_md.data_type();
        auto output_type = output_md.data_type();

        switch (input_type) {
        case mkldnn_u8: jit.define_int("IN_TYPE_U8", 1); break;
        case mkldnn_s8: jit.define_int("IN_TYPE_S8", 1); break;
        case mkldnn_f16: jit.define_int("IN_TYPE_F16", 1); break;
        case mkldnn_s32: jit.define_int("IN_TYPE_S32", 1); break;
        case mkldnn_f32: jit.define_int("IN_TYPE_F32", 1); break;
        default: return status::invalid_arguments;
        }
        switch (output_type) {
        case mkldnn_u8: jit.define_int("OUT_TYPE_U8", 1); break;
        case mkldnn_s8: jit.define_int("OUT_TYPE_S8", 1); break;
        case mkldnn_f16: jit.define_int("OUT_TYPE_F16", 1); break;
        case mkldnn_s32: jit.define_int("OUT_TYPE_S32", 1); break;
        case mkldnn_f32: jit.define_int("OUT_TYPE_F32", 1); break;
        default: return status::invalid_arguments;
        }

        const bool opt_reorder = true
                && !jrp.has_padding
                && !jrp.is_alpha_beta
                && (((input_type == mkldnn_f32 && output_type == mkldnn_f32)
                            && (input_md.matches_tag(nChw16c)
                                       || output_md.matches_tag(nChw16c)
                                       || input_md.matches_tag(NChw16n16c)
                                       || output_md.matches_tag(NChw16n16c)
                                       || input_md.matches_one_of_tag(
                                                  IOhw16i16o, gIOhw16i16o)
                                       || input_md.matches_one_of_tag(
                                                  OIhw16o16i, gOIhw16o16i)
                                       || output_md.matches_one_of_tag(
                                                  IOhw16i16o, gIOhw16i16o)
                                       || output_md.matches_one_of_tag(
                                                  OIhw16o16i, gOIhw16o16i)))
                           || ((input_type == mkldnn_f16
                                       || output_type == mkldnn_f16)
                                      && (input_md.matches_tag(nChw16c)
                                                 || output_md.matches_tag(
                                                            nChw16c)
                                                 || input_md.matches_tag(
                                                            NChw16n16c)
                                                 || output_md.matches_tag(
                                                            NChw16n16c))));
        jit.define_int("REF_REORDER", !opt_reorder);
        jit.define_int("SUB_GROUP_SIZE", jrp.sub_group_size);

        if (input_md.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c)) {
            jit.define_int("IN_NCHW16C", 1);
        } else if (input_md.matches_one_of_tag(
                           NCw16n16c, NChw16n16c, NCdhw16n16c)) {
            jit.define_int("IN_NCHW16N16C", 1);
        } else if (input_md.matches_one_of_tag(IOw16i16o, IOhw16i16o,
                           IOdhw16i16o, gIOw16i16o, gIOhw16i16o,
                           gIOdhw16i16o)) {
            jit.define_int("IN_IOHW16I16O", 1);
        } else if (input_md.matches_one_of_tag(OIw16o16i, OIhw16o16i,
                           OIdhw16o16i, gOIw16o16i, gOIhw16o16i,
                           gOIdhw16o16i)) {
            jit.define_int("IN_OIHW16O16I", 1);
        } else if (input_md.matches_one_of_tag(OIhw4o8i8o4i, gOIhw4o8i8o4i)) {
            jit.define_int("IN_OIHW4O8I8O4I", 1);
        } else if (input_md.matches_one_of_tag(OIhw2o8i8o2i, gOIhw2o8i8o2i)) {
            jit.define_int("IN_OIHW2O8I8O2I", 1);
        } else if (input_md.matches_one_of_tag(OIw8o16i2o, OIhw8o16i2o,
                           gOIw8o16i2o, gOIhw8o16i2o)) {
            jit.define_int("IN_OIHW8O16I2O", 1);
        } else if (input_md.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i,
                           OIdhw8i16o2i, gOIw8i16o2i, gOIhw8i16o2i,
                           gOIdhw8i16o2i)) {
            jit.define_int("IN_OIHW8I16O2I", 1);
        } else {
            jit.define_int("IN_REF_FORMAT", 1);
        }

        if (output_md.matches_one_of_tag(nCw16c, nChw16c, nCdhw16c)) {
            jit.define_int("OUT_NCHW16C", 1);
        } else if (output_md.matches_one_of_tag(
                           NCw16n16c, NChw16n16c, NCdhw16n16c)) {
            jit.define_int("OUT_NCHW16N16C", 1);
        } else if (output_md.matches_one_of_tag(IOw16i16o, IOhw16i16o,
                           IOdhw16i16o, gIOw16i16o, gIOhw16i16o,
                           gIOdhw16i16o)) {
            jit.define_int("OUT_IOHW16I16O", 1);
        } else if (output_md.matches_one_of_tag(OIw16o16i, OIhw16o16i,
                           OIdhw16o16i, gOIw16o16i, gOIhw16o16i,
                           gOIdhw16o16i)) {
            jit.define_int("OUT_OIHW16O16I", 1);
        } else if (output_md.matches_one_of_tag(OIhw4o8i8o4i, gOIhw4o8i8o4i)) {
            jit.define_int("OUT_OIHW4O8I8O4I", 1);
        } else if (output_md.matches_one_of_tag(OIhw2o8i8o2i, gOIhw2o8i8o2i)) {
            jit.define_int("OUT_OIHW2O8I8O2I", 1);
        } else if (output_md.matches_one_of_tag(OIw8o16i2o, OIhw8o16i2o,
                           gOIw8o16i2o, gOIhw8o16i2o)) {
            jit.define_int("OUT_OIHW8O16I2O", 1);
        } else if (output_md.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i,
                           OIdhw8i16o2i, gOIw8i16o2i, gOIhw8i16o2i,
                           gOIdhw8i16o2i)) {
            jit.define_int("OUT_OIHW8I16O2I", 1);
        } else {
            jit.define_int("OUT_REF_FORMAT", 1);
        }

        set_offsets(jit, input_md, "SRC");
        set_offsets(jit, output_md, "DST");

        const auto &in_dims = input_md.dims();
        const auto &out_dims = output_md.padded_dims();

        jit.define_int("PAD_FILL_ZERO", jrp.has_padding);

        if (jrp.has_padding) {
            char tempstr[32];
            for (int d = 0; d < input_md.ndims(); ++d) {
                snprintf(tempstr, 32, " SRC_DIM%d", d);
                jit.define_int(tempstr, in_dims[d]);
            }
            for (int d = input_md.ndims(); d < 6; ++d) {
                snprintf(tempstr, 32, " SRC_DIM%d", d);
                jit.define_int(tempstr, 0);
            }
            for (int d = 0; d < output_md.ndims(); ++d) {
                snprintf(tempstr, 32, " DST_DIM%d", d);
                jit.define_int(tempstr, out_dims[d]);
            }
            for (int d = output_md.ndims(); d < 6; ++d) {
                snprintf(tempstr, 32, " DST_DIM%d", d);
                jit.define_int(tempstr, 0);
            }
        }

        return status::success;
    }

    jit_reorder_conf_t jrp;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

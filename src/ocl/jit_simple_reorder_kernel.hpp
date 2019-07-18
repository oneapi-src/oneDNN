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
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace mkldnn::impl::format_tag;

struct jit_simple_reorder_kernel {

    jit_simple_reorder_kernel(jit_reorder_conf_t ajrp) : jrp(ajrp) {}

    ~jit_simple_reorder_kernel() {}

    static status_t init_conf(const reorder_pd_t *pd, jit_reorder_conf_t &jrp,
            const memory_desc_wrapper &input_md,
            const memory_desc_wrapper &output_md) {

        status_t status = status::success;

        const auto &dims = output_md.padded_dims();
        jrp.scale_quant = pd->attr()->output_scales_.mask_ != 0;
        jrp.scale_mask = jrp.scale_quant ? pd->attr()->output_scales_.mask_ : 0;
        jrp.with_sum_ab = jrp.scale_quant
                ? false
                : (pd->alpha() != 1.f || pd->beta() != 0.f);
        jrp.with_sum_a = jrp.with_sum_ab && pd->beta() == 0.f;
        jrp.do_reorder = jrp.scale_quant || jrp.with_sum_ab
                ? true
                : input_md != output_md;
        jrp.has_padding = !input_md.is_dense() || !output_md.is_dense();
        jrp.ndims = input_md.ndims();
        jrp.nelems = utils::array_product(dims, jrp.ndims);
        jrp.lws_d[0] = 1;
        jrp.lws_d[1] = 1;
        jrp.lws_d[2] = 1;

        jrp.use_ref_impl = 1;
        jrp.with_group = 0;
        jrp.sub_group_size = 1;

        jrp.block[0] = 1;
        jrp.block[1] = 1;
        jrp.block[2] = 1;

        if (jrp.ndims <= 3) {
            jrp.gws_d[0] = dims[0];
            jrp.gws_d[1] = jrp.ndims > 1 ? dims[1] : 1;
            jrp.gws_d[2] = jrp.ndims > 2 ? dims[2] : 1;
        } else if (jrp.ndims <= 5) {
            jrp.gws_d[0] = dims[0];
            jrp.gws_d[1] = dims[1];
            jrp.gws_d[2] = 1;
        } else {
            jrp.gws_d[0] = dims[0];
            jrp.gws_d[1] = dims[1];
            jrp.gws_d[2] = dims[2];
        }

        if (input_md.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o, gOIw8i16o2i,
                    gOIhw8i16o2i, gOIdhw8i16o2i, gOIhw4o8i8o4i, gOIhw2o8i8o2i)
                || output_md.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o,
                        gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i, gOIhw4o8i8o4i,
                        gOIhw2o8i8o2i))
            jrp.with_group = 1;

        if (jrp.has_padding || jrp.scale_quant)
            return status;

        const bool type_s8_u8
                = utils::one_of(input_md.data_type(), mkldnn_s8, mkldnn_u8)
                || utils::one_of(output_md.data_type(), mkldnn_s8, mkldnn_u8);

        const bool use_unroll_16a16b = true && !type_s8_u8
                && (input_md.matches_one_of_tag(ABc16a16b, ABc16b16a,
                            ABcd16a16b, ABcd16b16a, ABcde16a16b, ABcde16b16a,
                            BAc16a16b, BAc16b16a, BAcd16a16b, BAcd16b16a,
                            BAcde16b16a)
                        || output_md.matches_one_of_tag(ABc16a16b, ABc16b16a,
                                ABcd16a16b, ABcd16b16a, ABcde16a16b,
                                ABcde16b16a, BAc16a16b, BAc16b16a, BAcd16a16b,
                                BAcd16b16a, BAcde16b16a));

        const bool use_unroll_16b = true && !type_s8_u8
                && (input_md.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)
                        || output_md.matches_one_of_tag(
                                aBc16b, aBcd16b, aBcde16b));

        const bool use_unroll_16b16c = true && !type_s8_u8
                && (input_md.matches_one_of_tag(aBCd16b16c, aBCd16c16b,
                            aBCde16b16c, aBCde16c16b, aBCdef16b16c,
                            aBCdef16c16b, aCBd16b16c, aCBd16c16b, aCBde16b16c,
                            aCBde16c16b, aCBdef16c16b)
                        || output_md.matches_one_of_tag(aBCd16b16c, aBCd16c16b,
                                aBCde16b16c, aBCde16c16b, aBCdef16b16c,
                                aBCdef16c16b, aCBd16b16c, aCBd16c16b,
                                aCBde16b16c, aCBde16c16b, aCBdef16c16b));

        if (use_unroll_16a16b) {
            jrp.use_ref_impl = 0;
            jrp.sub_group_size = 16;
            jrp.gws_d[0] = dims[0] / 16;
            jrp.gws_d[1] = dims[1];
            jrp.lws_d[1] = 16;
            jrp.gws_d[2] = utils::array_product(&dims[2], jrp.ndims - 2);
        } else if (use_unroll_16b) {
            jrp.use_ref_impl = 0;
            jrp.sub_group_size = 16;
            jrp.gws_d[0] = dims[0];
            jrp.gws_d[1] = dims[1];
            jrp.lws_d[1] = 16;
            jrp.gws_d[2] = utils::array_product(&dims[2], jrp.ndims - 2);
        } else if (use_unroll_16b16c) {
            jrp.use_ref_impl = 0;
            jrp.with_group = 1;
            jrp.sub_group_size = 16;
            jrp.lws_d[0] = 16;
            jrp.gws_d[0] = dims[0] * dims[1];
            jrp.block[0] = dims[1];
            jrp.gws_d[1] = dims[2] / 16;
            jrp.gws_d[2] = utils::array_product(&dims[3], jrp.ndims - 3);
        }

        return status;
    };

    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            const jit_reorder_conf_t &jrp, const memory_desc_wrapper &input_md,
            const memory_desc_wrapper &output_md) {

        kernel_ctx.define_int("NDIMS", jrp.ndims);
        if (jrp.scale_quant) {
            kernel_ctx.define_int("SCALE_QUANT", 1);
            kernel_ctx.define_int("SCALE_MASK", jrp.scale_mask);
        } else if (jrp.with_sum_a)
            kernel_ctx.define_int("WITH_SUM_A", 1);
        else if (jrp.with_sum_ab)
            kernel_ctx.define_int("WITH_SUM_AB", 1);
        kernel_ctx.define_int("WITH_GROUP", jrp.with_group);

        kernel_ctx.define_int("LWS_0", jrp.lws_d[0]);
        kernel_ctx.define_int("LWS_1", jrp.lws_d[1]);
        kernel_ctx.define_int("LWS_2", jrp.lws_d[2]);

        for (int i = 0; i < 3; i++) {
            char tempstr[32];
            snprintf(tempstr, 32, "BLOCK_%d", i);
            kernel_ctx.define_int(tempstr, jrp.block[i]);
        }

        auto input_type = input_md.data_type();
        auto output_type = output_md.data_type();

        switch (input_type) {
        case mkldnn_u8: kernel_ctx.define_int("IN_TYPE_U8", 1); break;
        case mkldnn_s8: kernel_ctx.define_int("IN_TYPE_S8", 1); break;
        case mkldnn_f16: kernel_ctx.define_int("IN_TYPE_F16", 1); break;
        case mkldnn_s32: kernel_ctx.define_int("IN_TYPE_S32", 1); break;
        case mkldnn_f32: kernel_ctx.define_int("IN_TYPE_F32", 1); break;
        case mkldnn_bf16: kernel_ctx.define_int("IN_TYPE_BF16", 1); break;
        default: return status::invalid_arguments;
        }
        switch (output_type) {
        case mkldnn_u8: kernel_ctx.define_int("OUT_TYPE_U8", 1); break;
        case mkldnn_s8: kernel_ctx.define_int("OUT_TYPE_S8", 1); break;
        case mkldnn_f16: kernel_ctx.define_int("OUT_TYPE_F16", 1); break;
        case mkldnn_s32: kernel_ctx.define_int("OUT_TYPE_S32", 1); break;
        case mkldnn_f32: kernel_ctx.define_int("OUT_TYPE_F32", 1); break;
        case mkldnn_bf16: kernel_ctx.define_int("OUT_TYPE_BF16", 1); break;
        default: return status::invalid_arguments;
        }

        kernel_ctx.define_int("REF_REORDER", jrp.use_ref_impl);
        kernel_ctx.define_int("SUB_GROUP_SIZE", jrp.sub_group_size);

        set_offsets(kernel_ctx, input_md, "SRC");
        set_offsets(kernel_ctx, output_md, "DST");

        const auto &in_dims = input_md.dims();
        const auto &out_dims = output_md.padded_dims();

        kernel_ctx.define_int("PAD_FILL_ZERO", jrp.has_padding);

        {
            char tempstr[32];
            for (int d = 0; d < input_md.ndims(); ++d) {
                snprintf(tempstr, 32, " SRC_D%d", d);
                kernel_ctx.define_int(tempstr, in_dims[d]);
            }
            for (int d = input_md.ndims(); d < 6; ++d) {
                snprintf(tempstr, 32, " SRC_D%d", d);
                kernel_ctx.define_int(tempstr, 1);
            }
            for (int d = 0; d < output_md.ndims(); ++d) {
                snprintf(tempstr, 32, " DST_D%d", d);
                kernel_ctx.define_int(tempstr, out_dims[d]);
            }
            for (int d = output_md.ndims(); d < 6; ++d) {
                snprintf(tempstr, 32, " DST_D%d", d);
                kernel_ctx.define_int(tempstr, 1);
            }
        }

        if (!jrp.use_ref_impl) {
            if (input_md.matches_one_of_tag(ABc16a16b, ABcd16a16b, ABcde16a16b,
                        BAc16a16b, BAcd16a16b)) {
                kernel_ctx.define_int("IN_16A16B", 1);
            } else if (input_md.matches_one_of_tag(ABc16b16a, ABcd16b16a,
                               ABcde16b16a, BAc16b16a, BAcd16b16a,
                               BAcde16b16a)) {
                kernel_ctx.define_int("IN_16B16A", 1);
            } else if (input_md.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)) {
                kernel_ctx.define_int("IN_16B", 1);
            } else if (input_md.matches_one_of_tag(aBCd16b16c, aBCde16b16c,
                               aBCdef16b16c, aCBd16b16c, aCBde16b16c)) {
                kernel_ctx.define_int("IN_16B16C", 1);
            } else if (input_md.matches_one_of_tag(aBCd16c16b, aBCde16c16b,
                               aBCdef16c16b, aCBd16c16b, aCBde16c16b,
                               aCBdef16c16b)) {
                kernel_ctx.define_int("IN_16C16B", 1);
            }
        }

        if (input_md.matches_one_of_tag(OIw8o16i2o, OIhw8o16i2o, OIdhw8o16i2o,
                    gOIw8o16i2o, gOIhw8o16i2o, gOIdhw8o16i2o)) {
            kernel_ctx.define_int("IN_OIHW8O16I2O", 1);
        } else if (input_md.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i,
                           OIdhw8i16o2i, gOIw8i16o2i, gOIhw8i16o2i,
                           gOIdhw8i16o2i)) {
            kernel_ctx.define_int("IN_OIHW8I16O2I", 1);
        } else if (input_md.matches_one_of_tag(OIhw4o8i8o4i, gOIhw4o8i8o4i)) {
            kernel_ctx.define_int("IN_OIHW4O8I8O4I", 1);
        } else if (input_md.matches_one_of_tag(OIhw2o8i8o2i, gOIhw2o8i8o2i)) {
            kernel_ctx.define_int("IN_OIHW2O8I8O2I", 1);
        }

        if (!jrp.use_ref_impl) {
            if (output_md.matches_one_of_tag(ABc16a16b, ABcd16a16b, ABcde16a16b,
                        BAc16a16b, BAcd16a16b)) {
                kernel_ctx.define_int("OUT_16A16B", 1);
            } else if (output_md.matches_one_of_tag(ABc16b16a, ABcd16b16a,
                               ABcde16b16a, BAc16b16a, BAcd16b16a,
                               BAcde16b16a)) {
                kernel_ctx.define_int("OUT_16B16A", 1);
            } else if (output_md.matches_one_of_tag(
                               aBc16b, aBcd16b, aBcde16b)) {
                kernel_ctx.define_int("OUT_16B", 1);
            } else if (output_md.matches_one_of_tag(aBCd16b16c, aBCde16b16c,
                               aBCdef16b16c, aCBd16b16c, aCBde16b16c)) {
                kernel_ctx.define_int("OUT_16B16C", 1);
            } else if (output_md.matches_one_of_tag(aBCd16c16b, aBCde16c16b,
                               aBCdef16c16b, aCBd16c16b, aCBde16c16b,
                               aCBdef16c16b)) {
                kernel_ctx.define_int("OUT_16C16B", 1);
            }
        }

        if (output_md.matches_one_of_tag(OIw8o16i2o, OIhw8o16i2o, OIdhw8o16i2o,
                    gOIw8o16i2o, gOIhw8o16i2o, gOIdhw8o16i2o)) {
            kernel_ctx.define_int("OUT_OIHW8O16I2O", 1);
        } else if (output_md.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i,
                           OIdhw8i16o2i, gOIw8i16o2i, gOIhw8i16o2i,
                           gOIdhw8i16o2i)) {
            kernel_ctx.define_int("OUT_OIHW8I16O2I", 1);
        } else if (output_md.matches_one_of_tag(OIhw4o8i8o4i, gOIhw4o8i8o4i)) {
            kernel_ctx.define_int("OUT_OIHW4O8I8O4I", 1);
        } else if (output_md.matches_one_of_tag(OIhw2o8i8o2i, gOIhw2o8i8o2i)) {
            kernel_ctx.define_int("OUT_OIHW2O8I8O2I", 1);
        }

        return status::success;
    }

    jit_reorder_conf_t jrp;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

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
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/reorder_pd.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_primitive_conf.hpp"
#include "ocl/ocl_gpu_device_info.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct jit_simple_reorder_kernel {

    jit_simple_reorder_kernel(const jit_reorder_conf_t &ajrp) : jrp(ajrp) {}

    ~jit_simple_reorder_kernel() {}

    static status_t init_conf(jit_reorder_conf_t &jrp, const reorder_pd_t *pd) {
        using namespace format_tag;

        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(pd->engine());
        auto *dev_info = utils::downcast<const ocl_gpu_device_info_t *>(
                compute_engine->device_info());
        const size_t eu_count = static_cast<size_t>(dev_info->eu_count());

        const memory_desc_wrapper src_md(pd->src_md());
        const memory_desc_wrapper dst_md(pd->dst_md());

        jrp.src_md_info = jit_memory_desc_info_t::create(src_md);
        jrp.dst_md_info = jit_memory_desc_info_t::create(dst_md);

        status_t status = status::success;

        const auto &padded_dims = dst_md.padded_dims();
        jrp.scale_quant = pd->attr()->output_scales_.mask_ != 0;
        jrp.scale_mask = jrp.scale_quant ? pd->attr()->output_scales_.mask_ : 0;
        jrp.scales_num
                = jrp.scale_quant ? pd->attr()->output_scales_.count_ : 0;
        jrp.with_sum_ab = jrp.scale_quant
                ? false
                : (pd->alpha() != 1.f || pd->beta() != 0.f);
        jrp.with_sum_a = jrp.with_sum_ab && pd->beta() == 0.f;
        jrp.do_reorder
                = jrp.scale_quant || jrp.with_sum_ab ? true : src_md != dst_md;
        jrp.has_padding = !src_md.is_dense() || !dst_md.is_dense();
        jrp.ndims = src_md.ndims();
        jrp.nelems = utils::array_product(padded_dims, jrp.ndims);

        jrp.lws_d[0] = 1;
        jrp.lws_d[1] = 1;
        jrp.lws_d[2] = 1;

        jrp.gws_d[0] = 1;
        jrp.gws_d[1] = 1;
        jrp.gws_d[2] = 1;

        jrp.use_ref_impl = 1;
        jrp.with_group = 0;
        jrp.sub_group_size = 1;

        jrp.block[0] = 1;
        jrp.block[1] = 1;
        jrp.block[2] = 1;

        jrp.dim_block[0] = 1;
        jrp.dim_block[1] = 1;
        jrp.dim_block[2] = 1;
        jrp.dim_block[3] = 1;
        jrp.dim_block[4] = 1;
        jrp.dim_block[5] = 1;

        if (jrp.nelems == 0) return status::success;

        jrp.gws_d[0] = padded_dims[0];
        jrp.gws_d[1] = jrp.ndims > 1 ? padded_dims[1] : 1;
        jrp.gws_d[2] = 1;
        // Reduce amount of work done by a single work item to increase amount
        // of work items.
        const auto dst_ndims = dst_md.ndims();
        for (int d = 2; d < dst_ndims; ++d) {
            size_t global_work = utils::array_product(jrp.gws_d, 3);
            jrp.dim_block[d] = global_work < eu_count ? 1 : padded_dims[d];
            jrp.gws_d[2] *= padded_dims[d] / jrp.dim_block[d];
        }

        if (src_md.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o, gOIw8i16o2i,
                    gOIhw8i16o2i, gOIdhw8i16o2i, gOIhw4o8i8o4i, gOIhw2o8i8o2i)
                || dst_md.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o,
                        gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i, gOIhw4o8i8o4i,
                        gOIhw2o8i8o2i))
            jrp.with_group = 1;

        if (jrp.has_padding || jrp.scale_quant) return status;

        const bool type_s8_u8
                = utils::one_of(src_md.data_type(), dnnl_s8, dnnl_u8)
                || utils::one_of(dst_md.data_type(), dnnl_s8, dnnl_u8);

        const bool use_unroll_16a16b = true && !type_s8_u8
                && (src_md.matches_one_of_tag(ABc16a16b, ABc16b16a, ABcd16a16b,
                            ABcd16b16a, ABcde16a16b, ABcde16b16a, BAc16a16b,
                            BAc16b16a, BAcd16a16b, BAcd16b16a, BAcde16b16a)
                        || dst_md.matches_one_of_tag(ABc16a16b, ABc16b16a,
                                ABcd16a16b, ABcd16b16a, ABcde16a16b,
                                ABcde16b16a, BAc16a16b, BAc16b16a, BAcd16a16b,
                                BAcd16b16a, BAcde16b16a));

        const bool use_unroll_16b = true && !type_s8_u8
                && (src_md.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)
                        || dst_md.matches_one_of_tag(
                                aBc16b, aBcd16b, aBcde16b));

        const bool use_unroll_16b16c = true && !type_s8_u8
                && (src_md.matches_one_of_tag(aBCd16b16c, aBCd16c16b,
                            aBCde16b16c, aBCde16c16b, aBCdef16b16c,
                            aBCdef16c16b, aCBd16b16c, aCBd16c16b, aCBde16b16c,
                            aCBde16c16b, aCBdef16c16b)
                        || dst_md.matches_one_of_tag(aBCd16b16c, aBCd16c16b,
                                aBCde16b16c, aBCde16c16b, aBCdef16b16c,
                                aBCdef16c16b, aCBd16b16c, aCBd16c16b,
                                aCBde16b16c, aCBde16c16b, aCBdef16c16b));

        if (use_unroll_16a16b) {
            jrp.use_ref_impl = 0;
            jrp.sub_group_size = 16;
            jrp.gws_d[0] = padded_dims[0] / 16;
            jrp.gws_d[1] = padded_dims[1];
            jrp.lws_d[1] = 16;
            jrp.gws_d[2] = utils::array_product(&padded_dims[2], jrp.ndims - 2);
        } else if (use_unroll_16b) {
            jrp.use_ref_impl = 0;
            jrp.sub_group_size = 16;
            jrp.gws_d[0] = padded_dims[0];
            jrp.gws_d[1] = padded_dims[1];
            jrp.lws_d[1] = 16;
            jrp.gws_d[2] = utils::array_product(&padded_dims[2], jrp.ndims - 2);
        } else if (use_unroll_16b16c) {
            jrp.use_ref_impl = 0;
            jrp.with_group = 1;
            jrp.sub_group_size = 16;
            jrp.lws_d[0] = 16;
            jrp.gws_d[0] = padded_dims[0] * padded_dims[1];
            jrp.block[0] = padded_dims[1];
            jrp.gws_d[1] = padded_dims[2] / 16;
            jrp.gws_d[2] = utils::array_product(&padded_dims[3], jrp.ndims - 3);
        }

        return status;
    };

    static status_t init_const_def(compute::kernel_ctx_t &kernel_ctx,
            const jit_reorder_conf_t &jrp, const memory_desc_wrapper &src_md,
            const memory_desc_wrapper &dst_md) {
        using namespace format_tag;

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
            kernel_ctx.define_int(utils::format("BLOCK_%d", i), jrp.block[i]);
        }

        for (int i = 0; i < MAX_NDIMS; i++) {
            kernel_ctx.define_int(
                    utils::format("D%d_BLOCK", i), jrp.dim_block[i]);
            kernel_ctx.define_int(utils::format("D%d_NBLOCKS", i),
                    nstl::max(1,
                            jrp.dst_md_info.padded_dims[i] / jrp.dim_block[i]));
        }

        kernel_ctx.define_int("REF_REORDER", jrp.use_ref_impl);
        kernel_ctx.define_int("SUB_GROUP_SIZE", jrp.sub_group_size);

        kernel_ctx.define_int("PAD_FILL_ZERO", jrp.has_padding);

        def_memory_desc_info(kernel_ctx, jrp.src_md_info, "SRC");
        def_memory_desc_info(kernel_ctx, jrp.dst_md_info, "DST");

        if (!jrp.use_ref_impl) {
            if (src_md.matches_one_of_tag(ABc16a16b, ABcd16a16b, ABcde16a16b,
                        BAc16a16b, BAcd16a16b)) {
                kernel_ctx.define_int("SRC_16A16B", 1);
            } else if (src_md.matches_one_of_tag(ABc16b16a, ABcd16b16a,
                               ABcde16b16a, BAc16b16a, BAcd16b16a,
                               BAcde16b16a)) {
                kernel_ctx.define_int("SRC_16B16A", 1);
            } else if (src_md.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)) {
                kernel_ctx.define_int("SRC_16B", 1);
            } else if (src_md.matches_one_of_tag(aBCd16b16c, aBCde16b16c,
                               aBCdef16b16c, aCBd16b16c, aCBde16b16c)) {
                kernel_ctx.define_int("SRC_16B16C", 1);
            } else if (src_md.matches_one_of_tag(aBCd16c16b, aBCde16c16b,
                               aBCdef16c16b, aCBd16c16b, aCBde16c16b,
                               aCBdef16c16b)) {
                kernel_ctx.define_int("SRC_16C16B", 1);
            }
        }

        if (src_md.matches_one_of_tag(OIw8o16i2o, OIhw8o16i2o, OIdhw8o16i2o,
                    gOIw8o16i2o, gOIhw8o16i2o, gOIdhw8o16i2o)) {
            kernel_ctx.define_int("SRC_OIHW8O16I2O", 1);
        } else if (src_md.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i,
                           OIdhw8i16o2i, gOIw8i16o2i, gOIhw8i16o2i,
                           gOIdhw8i16o2i)) {
            kernel_ctx.define_int("SRC_OIHW8I16O2I", 1);
        } else if (src_md.matches_one_of_tag(OIhw4o8i8o4i, gOIhw4o8i8o4i)) {
            kernel_ctx.define_int("SRC_OIHW4O8I8O4I", 1);
        } else if (src_md.matches_one_of_tag(OIhw2o8i8o2i, gOIhw2o8i8o2i)) {
            kernel_ctx.define_int("SRC_OIHW2O8I8O2I", 1);
        }

        if (!jrp.use_ref_impl) {
            if (dst_md.matches_one_of_tag(ABc16a16b, ABcd16a16b, ABcde16a16b,
                        BAc16a16b, BAcd16a16b)) {
                kernel_ctx.define_int("DST_16A16B", 1);
            } else if (dst_md.matches_one_of_tag(ABc16b16a, ABcd16b16a,
                               ABcde16b16a, BAc16b16a, BAcd16b16a,
                               BAcde16b16a)) {
                kernel_ctx.define_int("DST_16B16A", 1);
            } else if (dst_md.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)) {
                kernel_ctx.define_int("DST_16B", 1);
            } else if (dst_md.matches_one_of_tag(aBCd16b16c, aBCde16b16c,
                               aBCdef16b16c, aCBd16b16c, aCBde16b16c)) {
                kernel_ctx.define_int("DST_16B16C", 1);
            } else if (dst_md.matches_one_of_tag(aBCd16c16b, aBCde16c16b,
                               aBCdef16c16b, aCBd16c16b, aCBde16c16b,
                               aCBdef16c16b)) {
                kernel_ctx.define_int("DST_16C16B", 1);
            }
        }

        if (dst_md.matches_one_of_tag(OIw8o16i2o, OIhw8o16i2o, OIdhw8o16i2o,
                    gOIw8o16i2o, gOIhw8o16i2o, gOIdhw8o16i2o)) {
            kernel_ctx.define_int("DST_OIHW8O16I2O", 1);
        } else if (dst_md.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i,
                           OIdhw8i16o2i, gOIw8i16o2i, gOIhw8i16o2i,
                           gOIdhw8i16o2i)) {
            kernel_ctx.define_int("DST_OIHW8I16O2I", 1);
        } else if (dst_md.matches_one_of_tag(OIhw4o8i8o4i, gOIhw4o8i8o4i)) {
            kernel_ctx.define_int("DST_OIHW4O8I8O4I", 1);
        } else if (dst_md.matches_one_of_tag(OIhw2o8i8o2i, gOIhw2o8i8o2i)) {
            kernel_ctx.define_int("DST_OIHW2O8I8O2I", 1);
        }

        kernel_ctx.print_options();
        return status::success;
    }

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_reorder_conf_t &jrp) {
        if (jrp.scales_num > 0)
            scratchpad.book(memory_tracking::names::key_reorder_scales,
                    sizeof(float) * jrp.scales_num);
    }

    jit_reorder_conf_t jrp;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

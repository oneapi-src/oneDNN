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

#include "gpu/ocl/simple_reorder.hpp"

#include "common/utils.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::memory_tracking::names;

status_t simple_reorder_init_conf(
        jit_reorder_conf_t &jrp, const reorder_pd_t *pd) {
    using namespace format_tag;

    const memory_desc_wrapper src_md(pd->src_md());
    const memory_desc_wrapper dst_md(pd->dst_md());

    jrp.src_md_info = jit_memory_desc_info_t::create(src_md);
    jrp.dst_md_info = jit_memory_desc_info_t::create(dst_md);

    status_t status = status::success;

    const auto &padded_dims = dst_md.padded_dims();
    jrp.scale_quant = pd->attr()->output_scales_.mask_ != 0;
    jrp.scale_mask = jrp.scale_quant ? pd->attr()->output_scales_.mask_ : 0;
    jrp.scales_num = jrp.scale_quant ? pd->attr()->output_scales_.count_ : 0;
    jrp.with_sum_ab = jrp.scale_quant
            ? false
            : (pd->alpha() != 1.f || pd->beta() != 0.f);
    jrp.with_sum_a = jrp.with_sum_ab && pd->beta() == 0.f;
    jrp.do_reorder
            = jrp.scale_quant || jrp.with_sum_ab ? true : src_md != dst_md;
    jrp.has_padding = !src_md.is_dense() || !dst_md.is_dense();
    jrp.ndims = src_md.ndims();
    jrp.nelems = utils::array_product(padded_dims, jrp.ndims);

    jrp.use_ref_impl = true;
    jrp.with_group = 0;
    jrp.sub_group_size = 1;

    if (jrp.nelems == 0) return status::success;

    if (src_md.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o, gOIw8i16o2i,
                gOIhw8i16o2i, gOIdhw8i16o2i, gOIhw4o8i8o4i, gOIhw2o8i8o2i)
            || dst_md.matches_one_of_tag(gOIw8o16i2o, gOIhw8o16i2o, gOIw8i16o2i,
                    gOIhw8i16o2i, gOIdhw8i16o2i, gOIhw4o8i8o4i, gOIhw2o8i8o2i))
        jrp.with_group = 1;

    bool has_padding_or_scale_quant = jrp.has_padding || jrp.scale_quant;

    const bool type_s8_u8 = utils::one_of(src_md.data_type(), dnnl_s8, dnnl_u8)
            || utils::one_of(dst_md.data_type(), dnnl_s8, dnnl_u8);

    const bool use_unroll_16a16b = !has_padding_or_scale_quant && !type_s8_u8
            && (src_md.matches_one_of_tag(ABc16a16b, ABc16b16a, ABcd16a16b,
                        ABcd16b16a, ABcde16a16b, ABcde16b16a, BAc16a16b,
                        BAc16b16a, BAcd16a16b, BAcd16b16a, BAcde16b16a)
                    || dst_md.matches_one_of_tag(ABc16a16b, ABc16b16a,
                            ABcd16a16b, ABcd16b16a, ABcde16a16b, ABcde16b16a,
                            BAc16a16b, BAc16b16a, BAcd16a16b, BAcd16b16a,
                            BAcde16b16a));

    const bool use_unroll_16b = !has_padding_or_scale_quant && !type_s8_u8
            && (src_md.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)
                    || dst_md.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b));

    const bool use_unroll_16b16c = !has_padding_or_scale_quant && !type_s8_u8
            && (src_md.matches_one_of_tag(aBCd16b16c, aBCd16c16b, aBCde16b16c,
                        aBCde16c16b, aBCdef16b16c, aBCdef16c16b, aCBd16b16c,
                        aCBd16c16b, aCBde16b16c, aCBde16c16b, aCBdef16c16b)
                    || dst_md.matches_one_of_tag(aBCd16b16c, aBCd16c16b,
                            aBCde16b16c, aBCde16c16b, aBCdef16b16c,
                            aBCdef16c16b, aCBd16b16c, aCBd16c16b, aCBde16b16c,
                            aCBde16c16b, aCBdef16c16b));

    dim_t blocks[6] = {1, 1, 1, 1, 1, 1};
    if (use_unroll_16a16b) {
        blocks[0] = 16;
    } else if (use_unroll_16b) {
        // No blocking.
    } else if (use_unroll_16b16c) {
        jrp.with_group = 1;
        blocks[2] = 16;
    }

    if (use_unroll_16a16b || use_unroll_16b || use_unroll_16b16c) {
        jrp.use_ref_impl = false;
        jrp.sub_group_size = 16;
    }

    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(pd->engine());
    jrp.dispatch = compute_engine->create_dispatch(dst_md.md_);
    for (int i = 0; i < 6; ++i) {
        auto dim_str = utils::format("D%d", i);
        if (i < dst_md.ndims()) {
            dim_t block = jrp.use_ref_impl ? (i < 2) ? 1 : 0 : blocks[i];
            jrp.dispatch.define_dim(dim_str, i, padded_dims[i], block);
        } else {
            jrp.dispatch.define_dim(dim_str, 1);
        }
    }

    if (use_unroll_16a16b || use_unroll_16b || use_unroll_16b16c) {
        jrp.dispatch.vectorize_dim("D1", 16);
    }

    jrp.dispatch.generate();

    return status;
}

status_t simple_reorder_init_const_def(compute::kernel_ctx_t &kernel_ctx,
        const jit_reorder_conf_t &jrp, const memory_desc_wrapper &src_md,
        const memory_desc_wrapper &dst_md) {
    using namespace format_tag;

    if (jrp.nelems == 0) return status::success;

    kernel_ctx.define_int("NDIMS", jrp.ndims);
    if (jrp.scale_quant) {
        kernel_ctx.define_int("SCALE_QUANT", 1);
        kernel_ctx.define_int("SCALE_MASK", jrp.scale_mask);
    } else if (jrp.with_sum_a)
        kernel_ctx.define_int("WITH_SUM_A", 1);
    else if (jrp.with_sum_ab)
        kernel_ctx.define_int("WITH_SUM_AB", 1);
    kernel_ctx.define_int("WITH_GROUP", jrp.with_group);

    def_dispatch(kernel_ctx, jrp.dispatch);

    kernel_ctx.define_int("REF_REORDER", jrp.use_ref_impl);
    kernel_ctx.define_int("SUB_GROUP_SIZE", jrp.sub_group_size);

    kernel_ctx.define_int("PAD_FILL_ZERO", jrp.has_padding);

    def_memory_desc_info(kernel_ctx, jrp.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, jrp.dst_md_info, "DST");

    if (!jrp.use_ref_impl) {
        if (src_md.matches_one_of_tag(ABc16a16b, ABcd16a16b, ABcde16a16b,
                    BAc16a16b, BAcd16a16b)) {
            kernel_ctx.define_int("SRC_16A16B", 1);
        } else if (src_md.matches_one_of_tag(ABc16b16a, ABcd16b16a, ABcde16b16a,
                           BAc16b16a, BAcd16b16a, BAcde16b16a)) {
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
    } else if (src_md.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i, OIdhw8i16o2i,
                       gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i)) {
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
        } else if (dst_md.matches_one_of_tag(ABc16b16a, ABcd16b16a, ABcde16b16a,
                           BAc16b16a, BAcd16b16a, BAcde16b16a)) {
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
    } else if (dst_md.matches_one_of_tag(OIw8i16o2i, OIhw8i16o2i, OIdhw8i16o2i,
                       gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i)) {
        kernel_ctx.define_int("DST_OIHW8I16O2I", 1);
    } else if (dst_md.matches_one_of_tag(OIhw4o8i8o4i, gOIhw4o8i8o4i)) {
        kernel_ctx.define_int("DST_OIHW4O8I8O4I", 1);
    } else if (dst_md.matches_one_of_tag(OIhw2o8i8o2i, gOIhw2o8i8o2i)) {
        kernel_ctx.define_int("DST_OIHW2O8I8O2I", 1);
    }

    kernel_ctx.print_options();
    return status::success;
}

void simple_reorder_init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_reorder_conf_t &jrp) {
    if (jrp.scales_num > 0)
        scratchpad.book(memory_tracking::names::key_reorder_scales,
                sizeof(float) * jrp.scales_num);
}

status_t simple_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_TO);

    const auto &jrp = pd()->jrp_;
    if (jrp.nelems == 0) return status::success;

    float alpha = pd()->alpha();
    float beta = pd()->beta();

    status_t status = status::success;

    std::unique_ptr<memory_storage_t> scales;
    if (jrp.scale_quant) {
        scales = ctx.get_scratchpad_grantor().get_memory_storage(
                key_reorder_scales);

        void *tmp_ptr = nullptr;
        status = scales->map_data(&tmp_ptr);
        if (status != status::success) return status;
        utils::array_copy((float *)tmp_ptr,
                pd()->attr()->output_scales_.scales_,
                pd()->attr()->output_scales_.count_);
        status = scales->unmap_data(tmp_ptr);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.set(2, alpha);
    arg_list.set(3, beta);
    arg_list.set(4, scales ? *scales : memory_storage_t::empty_storage());

    auto nd_range = jrp.dispatch.nd_range();
    status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP
#define GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

// Two grouped matmul patterns are supported:
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
struct ref_grouped_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:ref_grouped:any", ref_grouped_t);

        // For 3D weights [G, K, N], override masks to include 0th expert dim
        int wei_qmask_K() const { return (1 << 0) | (1 << 1); }
        int wei_qmask_N() const { return (1 << 0) | (1 << 2); }

        bool is_2dby2d() const { return is_2dby2d_; }

        status_t init(const impl::engine_t *engine) {
            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            VDISPATCH_MATMUL(
                    !with_reduce(), VERBOSE_UNSUPPORTED_FEATURE, "reduce");

            // Detect pattern (2Dx3D vs 2Dx2D) and initialize
            VDISPATCH_MATMUL(
                    src_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            is_2dby2d_ = wei_d.is_grouped_desc();

            const auto &src_grouped = src_d.sparse_desc().grouped_desc;
            group_count_ = src_grouped.group_count;

            return is_2dby2d_ ? init_2dby2d(engine) : init_2dby3d(engine);
        }

        dim_t group_count_ = 0;
        // Re-interpretted binary src1 as 3D view, while attr_.post_ops_
        // keeps the original grouped md
        post_ops_t generic_po_;
        // Re-interpretted grouped dst md as 3D view
        memory_desc_t group_po_dst_md_ = types::zero_md();

    private:
        bool is_2dby2d_ = false;

        status_t init_2dby3d(const impl::engine_t *engine) {
            using namespace data_type;

            // Resolve format_any to plain dense
            if (memory_desc_wrapper(weights_md(0)).format_any())
                CHECK(memory_desc_init_by_strides(weights_md_, nullptr));

            memory_desc_wrapper wei_d(weights_md(0));
            memory_desc_wrapper dst_d(dst_md());

            // Supported configurations: grouped src/dst, abc/acb 3D weights
            VDISPATCH_MATMUL(
                    dst_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(
                    wei_d.matches_one_of_tag(format_tag::abc, format_tag::acb),
                    VERBOSE_UNSUPPORTED_TAG);

            // Supported data types: fp and int for src/wei
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;
            const bool is_fp_src = utils::one_of(
                    src_type, f32, bf16, f16, f8_e5m2, f8_e4m3, f4_e2m1);
            const bool is_fp_wei = utils::one_of(
                    wei_type, f32, bf16, f16, f8_e5m2, f8_e4m3, f4_e2m1);
            const bool is_int_src = utils::one_of(src_type, u8, s8);
            const bool is_int_wei = utils::one_of(wei_type, u8, s8, s4, u4, u3);

            // Supported: fp src + int wei (WOQ), int src + int wei, fp src + fp wei
            VDISPATCH_MATMUL(is_fp_src || is_int_src, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(is_fp_wei || is_int_wei, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(utils::one_of(dst_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(IMPLICATION(is_int_src, is_int_wei),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            // WOQ requires weight scales and fpmath with apply_to_int
            VDISPATCH_MATMUL(IMPLICATION(is_fp_src && is_int_wei,
                                     !attr()->scales_.has_default_values(
                                             DNNL_ARG_WEIGHTS)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(IMPLICATION(is_fp_src && is_int_wei,
                                     attr()->fpmath_.apply_to_int_),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            // Check for supported quantization schemes
            const auto &attr_scales = attr()->scales_;
            if (!attr_scales.has_default_values(DNNL_ARG_SRC)) {
                const int src_mask = attr_scales.get_mask(DNNL_ARG_SRC);
                // Allow row-wise or blocked (K-grouping) scales for src
                VDISPATCH_MATMUL(src_mask == src_qmask_M()
                                || src_mask == (src_qmask_M() | src_qmask_K()),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        utils::one_of(attr_scales.get_data_type(DNNL_ARG_SRC),
                                f32, bf16, f16, e8m0, f8_e4m3, f8_e5m2),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                if (!attr_scales.get(DNNL_ARG_SRC).has_default_groups()) {
                    const auto gM = attr_scales.get_group(DNNL_ARG_SRC, -2);
                    VDISPATCH_MATMUL(gM == 1, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    const auto gK = attr_scales.get_group(DNNL_ARG_SRC, -1);
                    VDISPATCH_MATMUL(gK > 1, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    VDISPATCH_MATMUL(
                            K() % gK == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
                }
            }
            if (!attr_scales.has_default_values(DNNL_ARG_WEIGHTS)) {
                const int wei_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
                // Allow column-wise or blocked (K or KxN grouping) wei scales
                VDISPATCH_MATMUL(wei_mask == wei_qmask_N()
                                || wei_mask == (wei_qmask_K() | wei_qmask_N()),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        utils::one_of(
                                attr_scales.get_data_type(DNNL_ARG_WEIGHTS),
                                f32, bf16, f16, e8m0, f8_e4m3, f8_e5m2),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                if (!attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                    const auto gK = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
                    VDISPATCH_MATMUL(gK > 1, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    VDISPATCH_MATMUL(
                            K() % gK == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    const auto gN = attr_scales.get_group(DNNL_ARG_WEIGHTS, -1);
                    VDISPATCH_MATMUL(
                            N() % gN == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
                }
            }
            VDISPATCH_MATMUL(attr_scales.has_default_values(DNNL_ARG_DST),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            // Zero-points: src (int src), wei (int wei); row/col-wise or K-grouped
            const auto &attr_zps = attr()->zero_points_;
            VDISPATCH_MATMUL(attr_zps.has_default_values(DNNL_ARG_DST),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            if (!attr_zps.has_default_values(DNNL_ARG_SRC)) {
                VDISPATCH_MATMUL(is_int_src, VERBOSE_UNSUPPORTED_ZP_CFG);
                VDISPATCH_MATMUL(
                        utils::one_of(attr_zps.get_data_type(DNNL_ARG_SRC), u8,
                                s8, s32),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                const int zp_mask = attr_zps.get_mask(DNNL_ARG_SRC);
                VDISPATCH_MATMUL(zp_mask == src_qmask_M()
                                || zp_mask == (src_qmask_M() | src_qmask_K()),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                if (!attr_zps.get(DNNL_ARG_SRC).has_default_groups()) {
                    const auto gM = attr_zps.get_group(DNNL_ARG_SRC, -2);
                    VDISPATCH_MATMUL(gM == 1, VERBOSE_UNSUPPORTED_ZP_CFG);
                    const auto gK = attr_zps.get_group(DNNL_ARG_SRC, -1);
                    VDISPATCH_MATMUL(gK > 1, VERBOSE_UNSUPPORTED_ZP_CFG);
                    VDISPATCH_MATMUL(K() % gK == 0, VERBOSE_UNSUPPORTED_ZP_CFG);
                }
            }
            if (!attr_zps.has_default_values(DNNL_ARG_WEIGHTS)) {
                VDISPATCH_MATMUL(is_int_wei, VERBOSE_UNSUPPORTED_ZP_CFG);
                VDISPATCH_MATMUL(
                        utils::one_of(attr_zps.get_data_type(DNNL_ARG_WEIGHTS),
                                u8, s8, u4, s4, s32),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                const int zp_mask = attr_zps.get_mask(DNNL_ARG_WEIGHTS);
                VDISPATCH_MATMUL(zp_mask == wei_qmask_N()
                                || zp_mask == (wei_qmask_K() | wei_qmask_N()),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                if (!attr_zps.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                    const auto gK = attr_zps.get_group(DNNL_ARG_WEIGHTS, -2);
                    VDISPATCH_MATMUL(gK > 1, VERBOSE_UNSUPPORTED_ZP_CFG);
                    VDISPATCH_MATMUL(K() % gK == 0, VERBOSE_UNSUPPORTED_ZP_CFG);
                    const auto gN = attr_zps.get_group(DNNL_ARG_WEIGHTS, -1);
                    VDISPATCH_MATMUL(N() % gN == 0, VERBOSE_UNSUPPORTED_ZP_CFG);
                }
            }
            // For K grouping, src/wei scale group sizes must be multiples
            if (!attr_scales.has_default_values(DNNL_ARG_SRC)
                    && !attr_scales.get(DNNL_ARG_SRC).has_default_groups()
                    && !attr_scales.has_default_values(DNNL_ARG_WEIGHTS)
                    && !attr_scales.get(DNNL_ARG_WEIGHTS)
                                .has_default_groups()) {
                const auto src_gK = attr_scales.get_group(DNNL_ARG_SRC, -1);
                const auto wei_gK = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
                VDISPATCH_MATMUL(src_gK % wei_gK == 0 || wei_gK % src_gK == 0,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }
            // Weight scales and ZPs K-groups must match
            if (!attr_scales.has_default_values(DNNL_ARG_WEIGHTS)
                    && !attr_zps.has_default_values(DNNL_ARG_WEIGHTS)) {
                const auto scale_gK
                        = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
                const auto zp_gK = attr_zps.get_group(DNNL_ARG_WEIGHTS, -2);
                VDISPATCH_MATMUL(scale_gK == zp_gK, VERBOSE_INCONSISTENT_DIM,
                        "wei_scale_group_k", (int)scale_gK, "wei_zp_group_k",
                        (int)zp_gK);
            }

            if (attr_.post_ops_.len() > 0) CHECK(setup_post_ops(engine));

            return status::success;
        }

        // Re-interpret src1 as a 3D view so the generic post-op code applies:
        //   per-group  [G, 1]       -> [G, 1, 1]
        //   per-token  [total_M, 1] -> [1, total_M, 1]
        //   per-token  [total_M, N] -> [1, total_M, N]
        status_t setup_post_ops(const impl::engine_t *engine) {
            auto &attr_po = attr_.post_ops_;
            generic_po_ = attr_po;

            const dim_t total_tokens = src_md()->dims[0];
            const dim_t N = dst_md()->dims[1];

            // 3D view of grouped dst [G, DNNL_RUNTIME_DIM_VAL, N]
            const dims_t po_dst_dims = {group_count_, DNNL_RUNTIME_DIM_VAL, N};
            CHECK(memory_desc_init_by_strides(group_po_dst_md_, 3, po_dst_dims,
                    dst_md()->data_type, nullptr));

            for (int i = 0; i < attr_po.len(); ++i) {
                auto &e = attr_po.entry_[i];
                if (!e.is_binary()) continue;

                auto &attr_src1 = e.binary.src1_desc;
                // resolve format_any (e.g. NVFP4 per-group scale)
                if (memory_desc_wrapper(attr_src1).format_any())
                    CHECK(memory_desc_init_by_strides(attr_src1, nullptr));

                const memory_desc_wrapper src1_mdw(attr_src1);
                const bool per_group = src1_mdw.ndims() == 2
                        && src1_mdw.dims()[0] == group_count_
                        && src1_mdw.dims()[1] == 1;

                const dims_t dims_3d = {per_group ? group_count_ : 1,
                        per_group ? 1 : total_tokens, src1_mdw.dims()[1]};
                CHECK(memory_desc_init_by_strides(
                        generic_po_.entry_[i].binary.src1_desc, 3, dims_3d,
                        src1_mdw.data_type(), nullptr));
            }
            return status::success;
        }

        status_t init_2dby2d(const impl::engine_t *engine) {
            using namespace data_type;

            memory_desc_wrapper dst_d(dst_md());

            // Resolve format_any to plain dense
            if (dst_d.format_any())
                CHECK(memory_desc_init_by_strides(dst_md_, nullptr));

            // Only plain 3D dst is supported
            VDISPATCH_MATMUL(dst_d.is_plain(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(dst_d.ndims() == 3, VERBOSE_BAD_NDIMS, "dst",
                    dst_d.ndims());

            VDISPATCH_MATMUL(src_md()->data_type == weights_md(0)->data_type
                            && src_md()->data_type == dst_md()->data_type
                            && utils::one_of(
                                    src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.set_data_type(pd()->dst_md()->data_type);
        def_data_type(kernel_ctx, pd()->src_md()->data_type, "SRC");
        def_data_type(kernel_ctx, pd()->weights_md(0)->data_type, "WEI");
        def_data_type(kernel_ctx, pd()->dst_md()->data_type, "DST");
        def_data_type(kernel_ctx, pd()->desc()->accum_data_type, "ACC");

        // Zero-point data types (def_attr_info covers scale data types only)
        def_data_type(kernel_ctx,
                pd()->attr()->zero_points_.get_data_type(DNNL_ARG_SRC),
                "SRC_ZP");
        def_data_type(kernel_ctx,
                pd()->attr()->zero_points_.get_data_type(DNNL_ARG_WEIGHTS),
                "WEI_ZP");

        auto attr_info = attr_info_t::create(pd()->attr());
        CHECK(def_attr_info(kernel_ctx, attr_info, pd()->generic_po_,
                pd()->group_po_dst_md_));

        const bool with_bias = pd()->with_bias();
        kernel_ctx.define_int("WITH_BIAS", with_bias ? 1 : 0);
        if (with_bias)
            def_data_type(kernel_ctx, pd()->weights_md(1)->data_type, "BIA");

        return create_kernel(
                engine, &kernel_, "ref_grouped_gemm_matmul", kernel_ctx);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
#endif // GPU_INTEL_MATMUL_REF_GROUPED_GEMM_HPP

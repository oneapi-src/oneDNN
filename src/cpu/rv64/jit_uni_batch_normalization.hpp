/******************************************************************************
* Copyright 2025 ZTE Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
* Copyright 2026 Intel Corporation
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
******************************************************************************/

#ifndef CPU_RV64_JIT_UNI_BATCH_NORMALIZATION_HPP
#define CPU_RV64_JIT_UNI_BATCH_NORMALIZATION_HPP

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_batch_normalization_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

template <cpu_isa_t isa>
struct jit_uni_batch_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        using cpu_batch_normalization_fwd_pd_t::
                cpu_batch_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_batch_normalization_fwd_t);

        status_t init(const engine_t *engine) {
            UNUSED(engine);

            using namespace data_type;

            // Vector kernels are JIT-emitted; the rv64gc baseline build means a
            // non-V CPU must defer to the next (reference) implementation.
            VDISPATCH_BNORM(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_BNORM(is_fwd(), VERBOSE_BAD_PROPKIND);

            const data_type_t dtsrc = src_md()->data_type;
            const data_type_t dtdst = dst_md()->data_type;
            const data_type_t expected_dt = isa == zvfh ? f16
                    : isa == zvfbfwma                   ? bf16
                                                        : f32;
            const bool types_ok = dtsrc == expected_dt && dtdst == dtsrc
                    && platform::has_data_type_support(dtsrc)
                    && IMPLICATION(is_training(),
                            platform::has_training_support(dtsrc));
            VDISPATCH_BNORM(types_ok, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(check_scale_shift_data_type(),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale or shift data type");

            VDISPATCH_BNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            VDISPATCH_BNORM(!fuse_norm_add_relu(), VERBOSE_UNSUPPORTED_FEATURE,
                    "fuse_norm_add_relu not supported");
            using smask_t = primitive_attr_t::skip_mask_t;
            VDISPATCH_BNORM(!(fuse_norm_relu()
                                    && desc()->prop_kind
                                            == prop_kind::forward_training),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "forward training with fused ReLU is not supported");
            // Only a single plain ReLU (negative-slope 0, scale 1) is supported
            // as a post-op; it is applied inside the bnorm kernel as max(0, x).
            VDISPATCH_BNORM(attr()->has_default_values(smask_t::post_ops),
                    VERBOSE_UNSUPPORTED_ATTR);
            {
                // is_relu(true, true) requires scale == 1 and negative-slope == 0,
                // matching x64/aarch64's with_relu_post_op(). A scaled or leaky
                // ReLU (whose scale/slope the kernel would silently ignore) falls
                // back to ref.
                const post_ops_t &po = attr()->post_ops_;
                const bool relu_ok = po.len() == 0
                        || (po.len() == 1 && po.entry_[0].is_relu(true, true));
                VDISPATCH_BNORM(relu_ok, VERBOSE_UNSUPPORTED_ATTR);
            }

            // Simplest memory layouts only: plain, dense, same layout src/dst, no blocking/padding.
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            VDISPATCH_BNORM(
                    check_layouts(src_d, dst_d), VERBOSE_UNSUPPORTED_TAG);
            const bool channels_dense = src_d.blocking_desc().strides[1] == 1;
            // Statistics are reduced over one contiguous D*H*W row per (n, c).
            VDISPATCH_BNORM(
                    IMPLICATION(!use_global_stats(),
                            utils::one_of(dtsrc, f32, f16) && !channels_dense
                                    && spatial_dense(src_d)),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "computed statistics require f32 or f16 ncx data");

            fused_relu_in_kernel_ = fuse_norm_relu();
            init_scratchpad();

            return status::success;
        }
        bool check_layouts(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &dst_d) const {
            // Require plain, dense, no blocking/padding, same plain layout.
            bool ndims_ok = utils::one_of(ndims(), 3, 4, 5);
            bool plain_dense = src_d.blocking_desc().inner_nblks == 0
                    && dst_d.blocking_desc().inner_nblks == 0
                    && src_d.is_dense(/*with_padding=*/false)
                    && dst_d.is_dense(/*with_padding=*/false);
            bool same_layouts = src_d.similar_to(dst_d, /*with_strides=*/true,
                    /*with_pads=*/false);
            const bool vector_dim_dense = src_d.blocking_desc().strides[1] == 1
                    || src_d.blocking_desc().strides[ndims() - 1] == 1;
            return ndims_ok && plain_dense && same_layouts && vector_dim_dense;
        }
        // True if the spatial dims of every (n, c) form one contiguous row.
        bool spatial_dense(const memory_desc_wrapper &src_d) const {
            const auto &strides = src_d.blocking_desc().strides;
            dim_t sp_stride = 1;
            for (int d = ndims() - 1; d >= 2; --d) {
                if (strides[d] != sp_stride) return false;
                sp_stride *= src_d.dims()[d];
            }
            return true;
        }

        bool fused_relu_in_kernel() const { return fused_relu_in_kernel_; }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            // Reserve per-channel temporary buffers for axb (channels-dense) path
            scratchpad.template book<float>(key_bnorm_tmp_mean, C());
            scratchpad.template book<float>(key_bnorm_tmp_var, C());
            // Computed statistics that are not outputs (inference)
            if (!use_global_stats() && !is_training())
                scratchpad.template book<float>(key_bnorm_tmp_stats, 2 * C());
        }
        bool fused_relu_in_kernel_ = false;
    };

    jit_uni_batch_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

template <cpu_isa_t isa>
struct jit_uni_batch_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_batch_normalization_bwd_pd_t {
        using cpu_batch_normalization_bwd_pd_t::
                cpu_batch_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_batch_normalization_bwd_t);

        status_t init(const engine_t *engine) {
            UNUSED(engine);
            using namespace data_type;

            VDISPATCH_BNORM(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_BNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_BNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

            const data_type_t dt = src_md()->data_type;
            const data_type_t expected_dt = isa == zvfh ? f16
                    : isa == zvfbfwma                   ? bf16
                                                        : f32;
            VDISPATCH_BNORM(dt == expected_dt
                            && utils::everyone_is(dt, diff_dst_md()->data_type,
                                    diff_src_md()->data_type)
                            && platform::has_data_type_support(dt)
                            && platform::has_training_support(dt),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(check_scale_shift_data_type(),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale or shift data type");
            VDISPATCH_BNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(!fuse_norm_add_relu(), VERBOSE_UNSUPPORTED_FEATURE,
                    "fuse_norm_add_relu not supported");
            VDISPATCH_BNORM(!fuse_norm_relu(), VERBOSE_UNSUPPORTED_FEATURE,
                    "fused ReLU backward not supported");

            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper diff_src_d(diff_src_md());
            VDISPATCH_BNORM(check_layouts(src_d, diff_dst_d, diff_src_d),
                    VERBOSE_UNSUPPORTED_TAG);

            nthr_ = dnnl_get_max_threads();
            init_scratchpad();
            return status::success;
        }

        int nthr() const { return nthr_; }

    private:
        bool check_layouts(const memory_desc_wrapper &src_d,
                const memory_desc_wrapper &diff_dst_d,
                const memory_desc_wrapper &diff_src_d) const {
            const bool ndims_ok = utils::one_of(ndims(), 3, 4, 5);
            const bool plain_dense = src_d.blocking_desc().inner_nblks == 0
                    && diff_dst_d.blocking_desc().inner_nblks == 0
                    && diff_src_d.blocking_desc().inner_nblks == 0
                    && src_d.is_dense(false) && diff_dst_d.is_dense(false)
                    && diff_src_d.is_dense(false);
            const bool vector_dim_dense = src_d.blocking_desc().strides[1] == 1
                    || src_d.blocking_desc().strides[ndims() - 1] == 1;
            return ndims_ok && plain_dense
                    && src_d.similar_to(
                            diff_dst_d, /*with_strides=*/true, false)
                    && src_d.similar_to(
                            diff_src_d, /*with_strides=*/true, false)
                    && vector_dim_dense;
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.template book<float>(
                    key_bnorm_reduction, 2 * C() * nthr_);
            scratchpad.template book<float>(key_bnorm_tmp_diff_ss, 5 * C());
        }

        int nthr_ = 1;
    };

    jit_uni_batch_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_JIT_UNI_BATCH_NORMALIZATION_HPP

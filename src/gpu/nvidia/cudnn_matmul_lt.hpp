/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_LT_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_LT_HPP

#include "gpu/nvidia/cudnn_matmul_base.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_lt_t : cudnn_matmul_base_t {
    using cudnn_matmul_base_t::cudnn_matmul_base_t;

    struct pd_t : public pd_base_t {
        using pd_base_t::pd_base_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_matmul_lt_t);

        status_t init(impl::engine_t *engine) override {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            data_type_t src_dt = src_md()->data_type;
            data_type_t dst_dt = dst_md()->data_type;
            data_type_t wei_dt = weights_md(0)->data_type;
            data_type_t bia_dt
                    = with_bias() ? weights_md(1)->data_type : data_type::f32;

            bool f32_case = utils::everyone_is(f32, src_dt, wei_dt, dst_dt);
            bool f16_case = utils::everyone_is(f16, src_dt, wei_dt, dst_dt);
            bool bf16_case = utils::everyone_is(bf16, src_dt, wei_dt, dst_dt);
#ifdef DNNL_NO_IMMA_INT8_DST
            bool s8_case = utils::everyone_is(s8, src_dt, wei_dt)
                    && utils::one_of(dst_dt, s8, s32);
#else
            bool s8_case = utils::everyone_is(s8, src_dt, wei_dt)
                    && utils::one_of(dst_dt, s32);
#endif
            auto *sycl_engine_impl
                    = utils::downcast<const xpu::sycl::engine_impl_t *>(
                            engine->impl());

            bool is_imma_blocks = imma_blocks();
            bool is_eltwise_ok = eltwise_ok();

            bool ok = is_dense_format_kind()
                    && (blocking_ok() || is_imma_blocks)
                    && attr()->has_default_values(smask_t::scales_runtime)
                    && attr_post_ops_ok(attr())
                    && IMPLICATION(bf16_case,
                            has_bf16_support(sycl_engine_impl->device()))
                    && set_default_formats()
                    && (f32_case || f16_case || bf16_case || s8_case)
                    && IMPLICATION(with_bias(),
                            (IMPLICATION(f32_case, utils::one_of(bia_dt, f32))
                                    && IMPLICATION(f16_case,
                                            utils::one_of(bia_dt, f16, f32))
                                    && IMPLICATION(bf16_case,
                                            utils::one_of(bia_dt, bf16, f32))
                                    && IMPLICATION(s8_case,
                                            utils::one_of(bia_dt, s8, s32, f32))
                                    && IMPLICATION(s8_case, scales_ok())
                                    && IMPLICATION(!s8_case, bia_dt == dst_dt)))
                    && IMPLICATION(with_bias(), !has_runtime_dims_or_strides());

            memory_desc_wrapper weight_wrap(weights_md());
            memory_desc_wrapper dst_wrap(dst_md());

            ok = ok
                    && IMPLICATION(
                            is_md_col32(weight_wrap) || is_md_col32(dst_wrap),
                            s8_case);
            ok = ok && (is_imma_blocks || dst_ok()) && bias_ok()
                    && is_eltwise_ok;
            if (!ok) return status::unimplemented;

            if (!with_bias() && !with_eltwise() && !s8_case) {
                return status::unimplemented;
            }
            if (s8_case && with_eltwise() && is_eltwise_ok) {
                return status::unimplemented;
            }

            if (separate_bias() && !s8_case) { return status::unimplemented; }

            if (src_md()->ndims > 3) return status::unimplemented;
            if (with_bias()) {
                primitive_attr_t binary_attr;

                auto binary_desc = binary_desc_t();
                binary_desc.primitive_kind = primitive_kind::binary;
                binary_desc.alg_kind = alg_kind::binary_add;
                binary_desc.src_desc[0] = *dst_md();
                binary_desc.src_desc[1] = *weights_md(1);
                binary_desc.dst_desc = *dst_md();

                primitive_desc_iterator_t it(engine, (op_desc_t *)&binary_desc,
                        &binary_attr, nullptr);
                while (++it != it.end()) {
                    binary_pd_ = *it;
                    if (binary_pd_) { break; }
                }
                if (!binary_pd_) return status::unimplemented;
            }
            status_t status;
            if (!single_scale(DNNL_ARG_SRC)) {
                auto scale_md = dnnl_memory_desc();
                scale_md.ndims = attr()->scales_.get(DNNL_ARG_SRC).ndims_;
                scale_md.data_type
                        = attr()->scales_.get(DNNL_ARG_SRC).data_type_;
                scale_md.format_kind = dnnl_blocked;
                auto format_desc
                        = create_scaling_format_desc(DNNL_ARG_SRC, scale_md);

                scale_md.format_desc = {format_desc};

                status = init_scale_binary_pd(engine, DNNL_ARG_SRC,
                        src_scale_binary_pd_, src_md(), scale_md,
                        alg_kind::binary_mul);
            }
            if (!single_scale(DNNL_ARG_WEIGHTS)
                    && status == status_t::dnnl_success) {
                auto scale_md = dnnl_memory_desc();
                scale_md.ndims = attr()->scales_.get(DNNL_ARG_WEIGHTS).ndims_;
                scale_md.data_type
                        = attr()->scales_.get(DNNL_ARG_WEIGHTS).data_type_;
                scale_md.format_kind = dnnl_blocked;
                auto format_desc = create_scaling_format_desc(
                        DNNL_ARG_WEIGHTS, scale_md);
                scale_md.format_desc = {format_desc};
                status = init_scale_binary_pd(engine, DNNL_ARG_WEIGHTS,
                        wei_scale_binary_pd_, weights_md(0), scale_md,
                        alg_kind::binary_mul);
            }
            if (status == status_t::dnnl_success) {
                auto scale_md = dnnl_memory_desc();
                scale_md.ndims = attr()->scales_.get(DNNL_ARG_DST).ndims_;
                scale_md.data_type
                        = attr()->scales_.get(DNNL_ARG_DST).data_type_;
                scale_md.format_kind = dnnl_blocked;
                auto format_desc
                        = create_scaling_format_desc(DNNL_ARG_DST, scale_md);
                scale_md.format_desc = {format_desc};
                status = init_scale_binary_pd(engine, DNNL_ARG_DST,
                        dst_scale_binary_pd_, dst_md(), scale_md,
                        alg_kind::binary_div);
            }
            return status;
        }

        std::shared_ptr<primitive_desc_t> src_scale_binary_pd_;
        std::shared_ptr<primitive_desc_t> wei_scale_binary_pd_;
        std::shared_ptr<primitive_desc_t> dst_scale_binary_pd_;
        std::shared_ptr<primitive_desc_t> binary_pd_;

    private:
        blocking_desc_t create_scaling_format_desc(
                int ARG, dnnl_memory_desc &scale_md) {
            blocking_desc_t format_desc;
            memory_desc_t md;
            if (ARG == DNNL_ARG_SRC) {
                md = *src_md();
            } else if (ARG == DNNL_ARG_WEIGHTS) {
                md = *weights_md(0);
            } else if (ARG == DNNL_ARG_DST) {
                md = *dst_md();
            }

            scale_md.ndims = md.ndims;
            for (int i = 0; i < md.ndims; i++) {
                if (attr()->scales_.get(1).mask_ & (1 << i)) {
                    scale_md.dims[i] = md.dims[i];
                } else {
                    scale_md.dims[i] = 1;
                }
            }
            for (int i = 0; i < scale_md.ndims; i++) {
                auto stride = 1;
                for (int j = i + 1; j < scale_md.ndims; j++) {
                    stride *= scale_md.dims[j];
                }
                format_desc.strides[i] = stride;
            }
            format_desc.inner_nblks = 0;

            return format_desc;
        }

        status_t init_scale_binary_pd(impl::engine_t *engine, int ARG,
                std::shared_ptr<primitive_desc_t> &scale_binary_pd_,
                const memory_desc_t *in_out, memory_desc_t &in2,
                alg_kind_t mul_or_div) {
            primitive_attr_t scale_binary_attr;

            auto scale_binary_desc = binary_desc_t();
            scale_binary_desc.primitive_kind = primitive_kind::binary;
            scale_binary_desc.alg_kind = mul_or_div;
            scale_binary_desc.src_desc[0] = *in_out;
            scale_binary_desc.src_desc[1] = in2;
            scale_binary_desc.dst_desc = *in_out;

            primitive_desc_iterator_t it(engine,
                    (op_desc_t *)&scale_binary_desc, &scale_binary_attr,
                    nullptr);
            while (++it != it.end()) {
                scale_binary_pd_ = *it;
                if (scale_binary_pd_) { break; }
            }
            if (!scale_binary_pd_) return status::unimplemented;
            return status::success;
        }

        bool single_scale(int ARG) const {
            const auto &scales = attr()->scales_;
            if (scales.get(ARG).mask_ == 0) return true;
            return false;
        }

        bool scales_ok() {
            data_type_t src_scale_dt
                    = attr()->scales_.get(DNNL_ARG_SRC).data_type_;
            data_type_t wei_scale_dt
                    = attr()->scales_.get(DNNL_ARG_WEIGHTS).data_type_;
            bool src_scales_ok = single_scale(DNNL_ARG_SRC)
                    || utils::one_of(
                            src_scale_dt, data_type::s8, data_type::s32);
            bool wei_scales_ok = single_scale(DNNL_ARG_WEIGHTS)
                    || utils::one_of(
                            wei_scale_dt, data_type::s8, data_type::s32);
            return src_scales_ok && wei_scales_ok;
        }

        bool dst_ok() {
            bool ok = false;

            memory_desc_wrapper dst_wrap(dst_md());
            //check if dst is col_major
            bool isbatched = batched() && dst_wrap.dims()[0];
            const auto &md_strides
                    = &dst_wrap.blocking_desc().strides[isbatched];
            ok = (md_strides[1] == 1 && dst_wrap.dims()[isbatched + 0] > 1);
            // dst not supported for ndims = 1
            ok = ok
                    && (dst_wrap.dims()[isbatched + 1] != 1
                            && dst_wrap.dims()[isbatched + 0] != 1);

            return ok;
        }

        bool bias_ok() {
            if (!with_bias()) { return true; }
            memory_desc_wrapper dst_wrap(dst_md());
            memory_desc_wrapper bia_wrap(weights_md(1));

            bool isbatched = batched() && dst_wrap.dims()[0];
            if (bia_wrap.dims()[0 + isbatched] != 1) { return false; }
            return true;
        }

        bool separate_bias() {
            if (!with_bias()) { return false; }
            memory_desc_wrapper dst_wrap(dst_md());
            memory_desc_wrapper bia_wrap(weights_md(1));

            bool bias_dt_mismatch
                    = (dst_md()->data_type != weights_md(1)->data_type);

            bool isbatched = batched() && dst_wrap.dims()[0];
            const auto &md_strides
                    = &dst_wrap.blocking_desc().strides[isbatched];
            bool col_maj_dst
                    = md_strides[1] == 1 && dst_wrap.dims()[isbatched] > 1;

            if (bias_dt_mismatch || col_maj_dst
                    || (bia_wrap.dims()[1 + isbatched]
                                    != static_cast<uint64_t>(
                                            dst_wrap.dims()[isbatched])
                            || bia_wrap.dims()[0 + isbatched] != 1)
                    || static_cast<uint64_t>(dst_wrap.dims()[isbatched]) == 1
                    || static_cast<uint64_t>(dst_wrap.dims()[isbatched]) == 1) {
                return true;
            }
            return false;
        }

        bool with_eltwise() {
            return attr()->post_ops_.contain(primitive_kind::eltwise, 0)
                    || attr()->post_ops_.contain(primitive_kind::eltwise, 1);
        }

        bool eltwise_ok() {
            if (!with_eltwise()) { return true; }

            int eltwise_idx_ = attr()->post_ops_.find(primitive_kind::eltwise);
            auto eltwise_algo
                    = attr()->post_ops_.entry_[eltwise_idx_].eltwise.alg;
            if (eltwise_algo == alg_kind::eltwise_relu) { return true; }
            return false;
        }

        bool imma_blocks() {
            // weights should be blocked in Ab32a, ab or ba
            bool weights_supported = false;
            memory_desc_wrapper weight_wrap(weights_md());
            if (is_md_col32(weight_wrap) || weight_wrap.is_plain()) {
                weights_supported = true;
            }
            // src not blocked
            bool src_supported = false;
            memory_desc_wrapper src_wrap(src_md());
            if (src_wrap.is_plain()) { src_supported = true; }
            // dst blocked in Ab32a, ab or ba
            bool dst_supported = false;
            memory_desc_wrapper dst_wrap(dst_md());
            if (is_md_col32(dst_wrap) || dst_wrap.is_plain()) {
                dst_supported = true;
            }
            return (weights_supported && src_supported && dst_supported);
        }
    };

    status_t init(impl::engine_t *engine) override {
        // LT matmul
        matmul_impl_.reset(new cudnn_matmul_lt_impl_t());
        auto status = matmul_impl_->init((matmul_pd_t *)pd(), engine);

        bool has_runtime_args = matmul_impl_->has_runtime_params();
        if (has_runtime_args) {
            executor_.reset(new cudnn_matmul_lt_runtime_args_exec_t);
        } else if (!has_runtime_args) {
            executor_.reset(new cudnn_matmul_lt_exec_t);
        }
        if (status == dnnl_success && matmul_impl_->with_bias()) {
            status = create_nested_primitive(binary_, pd()->binary_pd_, engine);
        }
        if (status == dnnl_success && matmul_impl_->multi_src_scale()) {
            status = create_nested_primitive(
                    src_scale_binary_, pd()->src_scale_binary_pd_, engine);
        }
        if (status == dnnl_success && matmul_impl_->multi_wei_scale()) {
            status = create_nested_primitive(
                    wei_scale_binary_, pd()->wei_scale_binary_pd_, engine);
        }
        if (status == dnnl_success && matmul_impl_->multi_dst_scale()) {
            status = create_nested_primitive(
                    dst_scale_binary_, pd()->dst_scale_binary_pd_, engine);
        }

        return status;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

    std::shared_ptr<impl::primitive_t> binary_;
    std::shared_ptr<impl::primitive_t> src_scale_binary_;
    std::shared_ptr<impl::primitive_t> wei_scale_binary_;
    std::shared_ptr<impl::primitive_t> dst_scale_binary_;
    std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_;
    std::shared_ptr<cudnn_matmul_lt_base_exec_t> executor_;

private:
    const pd_t *pd() const {
        return (const pd_t *)primitive_t::pd().get();
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

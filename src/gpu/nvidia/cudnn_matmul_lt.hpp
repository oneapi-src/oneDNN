/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2024 Codeplay Software Limited
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

#include <cublas_v2.h>

#include "gpu/gpu_matmul_pd.hpp"

#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/nvidia/cudnn_matmul_executor.hpp"
#include "gpu/nvidia/cudnn_matmul_lt_impl.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_lt_t : public gpu::primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("cuda:cublaslt:any", cudnn_matmul_lt_t);

        status_t init(impl::engine_t *engine) {
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

            bool s8_case = false;
            if (has_imma_dst_int8_support()) {
                s8_case = utils::everyone_is(s8, src_dt, wei_dt)
                        && utils::one_of(dst_dt, s8, s32);
            } else {
                s8_case = utils::everyone_is(s8, src_dt, wei_dt)
                        && utils::one_of(dst_dt, s32);
            }
            auto *sycl_engine_impl
                    = utils::downcast<const xpu::sycl::engine_impl_t *>(
                            engine->impl());

            bool is_eltwise_ok = eltwise_ok();

            bool ok = is_dense_format_kind()
                    && attr()->has_default_values(smask_t::scales_runtime)
                    // src & weights scaling is not supported as this implementation uses integer types
                    // for the compute type, but the scales are floating point numbers
                    && attr()->scales_.get(DNNL_ARG_SRC).has_default_values()
                    && attr()->scales_.get(DNNL_ARG_WEIGHTS)
                               .has_default_values()
                    && attr_post_ops_ok(attr())
                    && IMPLICATION(bf16_case,
                            has_bf16_support(sycl_engine_impl->device()))
                    && (s8_case ? set_default_formats_lt()
                                : (set_default_formats() && blocking_ok()))
                    && tags_ok()
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

            memory_desc_wrapper src_wrap(src_md());
            memory_desc_wrapper weight_wrap(weights_md());
            memory_desc_wrapper dst_wrap(dst_md());

            ok = ok && src_wrap.ndims() <= 3;
            ok = ok
                    && IMPLICATION(
                            is_md_col32(weight_wrap) || is_md_col32(dst_wrap),
                            s8_case);
            bool is_imma_blocks = imma_blocks();
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
                if (dst_dt == dnnl_s8) {
                    s32_dst_md_ = types::zero_md();
                    auto tag = dst_wrap.matches_one_of_tag(format_tag::ab,
                            format_tag::abc, format_tag::acb, format_tag::Ab32a,
                            format_tag::aBc32b);

                    if (tag == format_tag::undef) return status::unimplemented;

                    memory_desc_init_by_tag(s32_dst_md_, dst_md()->ndims,
                            dst_md()->dims, dnnl_s32, tag);
                    binary_desc.src_desc[0] = s32_dst_md_;
                } else {
                    binary_desc.src_desc[0] = *dst_md();
                }
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

            // When src is in cublaslt blocked format only default scaling is supported.
            if (!IMPLICATION(src_wrap.is_cublaslt_blocked_desc(),
                        default_scale(DNNL_ARG_SRC))) {
                return status::unimplemented;
            }

            const bool is_scale_s32
                    = (s8_case && dst_wrap.data_type() == dnnl_s32);
            auto is_scale_ok = [&](int ARG) {
                return !default_scale(ARG)
                        && (!single_scale(ARG) || is_scale_s32);
            };

            if (is_scale_ok(DNNL_ARG_DST)) {
                CHECK(create_scale_binary_pd(engine, DNNL_ARG_DST));
            }

            params_ = std::make_shared<cublas_lt_params>();
            CHECK(params_->init(engine, src_md(), weights_md(), dst_md(),
                    weights_md(1), attr(), batched(), with_bias()));

            if (!params_->has_runtime_params()) {
                auto scratchpad = scratchpad_registry().registrar();
                params_->init_scratchpad(scratchpad);
            }

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> dst_scale_binary_pd_;
        std::shared_ptr<primitive_desc_t> binary_pd_;
        std::shared_ptr<cublas_lt_params> params_;

        memory_desc_t s32_dst_md_;

        bool default_scale(int ARG) const {
            return attr()->scales_.get(ARG).has_default_values();
        }

    private:
        bool tags_ok() const {
            memory_desc_wrapper src_wrap(src_md());
            memory_desc_wrapper dst_wrap(dst_md());
            memory_desc_wrapper wei_wrap(weights_md());

            bool ok = (src_wrap.is_cublaslt_blocked_desc()
                    || src_wrap.matches_one_of_tag(format_tag::ab,
                            format_tag::ba, format_tag::abc, format_tag::acb));
            for (auto &wrap : {dst_wrap, wei_wrap}) {
                ok = ok
                        && wrap.matches_one_of_tag(format_tag::ab,
                                format_tag::ba, format_tag::abc,
                                format_tag::acb, format_tag::Ab32a,
                                format_tag::aBc32b);
            }

            return ok;
        }

        status_t create_scale_binary_pd(impl::engine_t *engine, int ARG) {
            if (ARG != DNNL_ARG_DST) return status::unimplemented;

            auto md = arg_md(ARG);
            dims_t dims;
            dims_t strides;
            for (int i = 0; i < md->ndims; i++) {
                if (attr()->scales_.get(1).mask_ & (1 << i)) {
                    dims[i] = md->dims[i];
                } else {
                    dims[i] = 1;
                }
            }
            for (int i = 0; i < md->ndims; i++) {
                auto stride = 1;
                for (int j = i + 1; j < md->ndims; j++) {
                    stride *= md->dims[j];
                }
                strides[i] = stride;
            }

            memory_desc_t scale_md;
            CHECK(memory_desc_init_by_strides(scale_md, md->ndims, dims,
                    attr()->scales_.get(ARG).data_type_, strides));

            return init_scale_binary_pd(engine, ARG, dst_scale_binary_pd_,
                    arg_md(ARG), scale_md, alg_kind::binary_div);
        }

        status_t init_scale_binary_pd(impl::engine_t *engine, int ARG,
                std::shared_ptr<primitive_desc_t> &scale_binary_pd,
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
                if (*it) {
                    scale_binary_pd = *it;
                    break;
                }
            }
            if (!scale_binary_pd) return status::unimplemented;
            return status::success;
        }

        bool single_scale(int ARG) const {
            const auto &scales = attr()->scales_;
            return scales.get(ARG).mask_ == 0;
        }

        bool scales_ok() {
            bool src_scales_ok = default_scale(DNNL_ARG_SRC);
            bool wei_scales_ok = default_scale(DNNL_ARG_WEIGHTS);
            return src_scales_ok && wei_scales_ok;
        }

        bool dst_ok() {
            bool ok = false;

            memory_desc_wrapper dst_wrap(dst_md());
            bool isbatched = batched() && dst_wrap.dims()[0];
            //check if dst is col_major
            if (dst_wrap.is_plain()) {
                const auto &md_strides
                        = &dst_wrap.blocking_desc().strides[isbatched];
                ok = (md_strides[1] == 1 && dst_wrap.dims()[isbatched + 0] > 1);
            } else {
                // Ensure blocked format is Ab32a or aBc32b
                ok = is_md_col32(*dst_md());
            }
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

            if (!separate_bias()) {
                bool isbatched = batched() && dst_wrap.dims()[0];
                if (bia_wrap.dims()[0 + isbatched] != 1) { return false; }
            }

            return true;
        }

        bool separate_bias() {
            if (!with_bias()) { return false; }
            memory_desc_wrapper dst_wrap(dst_md());
            memory_desc_wrapper bia_wrap(weights_md(1));

            bool bias_dt_mismatch
                    = (dst_md()->data_type != weights_md(1)->data_type);

            if (bia_wrap.data_type() == dnnl_s8) { return true; }

            bool isbatched = batched() && dst_wrap.dims()[0];
            const auto &md_strides
                    = &dst_wrap.blocking_desc().strides[isbatched];
            bool col_maj_dst
                    = md_strides[1] == 1 && dst_wrap.dims()[isbatched] > 1;

            if (bias_dt_mismatch || col_maj_dst
                    || (bia_wrap.dims()[1 + isbatched]
                                    != dst_wrap.dims()[isbatched]
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
            // src plain format or internal cublaslt format
            bool src_supported = false;
            memory_desc_wrapper src_wrap(src_md());
            if (src_wrap.is_cublaslt_blocked_desc() || src_wrap.is_plain()) {
                src_supported = true;
            }
            // dst blocked in Ab32a, ab or ba
            bool dst_supported = false;
            memory_desc_wrapper dst_wrap(dst_md());
            if (is_md_col32(dst_wrap) || dst_wrap.is_plain()) {
                dst_supported = true;
            }
            return (weights_supported && src_supported && dst_supported);
        }

        bool set_default_formats_lt() {
            memory_desc_wrapper w_wrap(this->weights_md_);
            if (w_wrap.format_any()) {
                auto tag = batched() ? format_tag::aBc32b : format_tag::Ab32a;
                CHECK(memory_desc_init_by_tag(this->weights_md_, w_wrap.ndims(),
                        w_wrap.dims(), w_wrap.data_type(), tag));
            }

            memory_desc_wrapper dst_wrap(dst_md());
            if (dst_wrap.format_any()) {
                auto tag = batched() ? format_tag::aBc32b : format_tag::Ab32a;
                CHECK(memory_desc_init_by_tag(this->dst_md_, dst_wrap.ndims(),
                        dst_wrap.dims(), dst_wrap.data_type(), tag));
            }

            memory_desc_wrapper src_wrap(this->src_md_);
            if (src_wrap.format_any()) {
                auto ceildiv = [](dim_t n, dim_t d) { return (n + d - 1) / d; };
                auto n_rows = 32 * ceildiv(src_wrap.dims()[batched()], 32);
                auto n_cols = 32 * ceildiv(src_wrap.dims()[batched() + 1], 32);
                auto n_batch = batched() ? src_wrap.dims()[0] : 1;
                size_t size = n_batch * n_rows * n_cols;

                this->src_md_.padded_dims[batched()] = n_rows;
                this->src_md_.padded_dims[batched() + 1] = n_cols;
                this->src_md_.format_kind = format_kind::cublaslt_blocked;
                this->src_md_.format_desc.cublaslt_blocked_desc
                        = cublaslt_blocked_desc_t {
                                cublaslt_memory_format_t::col32_2r_4r4, size};
            }

            memory_desc_wrapper b_wrap(this->bias_md_);
            if (b_wrap.format_any()) {
                auto tag = batched() ? format_tag::aBc32b : format_tag::Ab32a;
                CHECK(memory_desc_init_by_tag(this->bias_md_, b_wrap.ndims(),
                        b_wrap.dims(), b_wrap.data_type(), tag));
            }

            return true;
        }

        bool blocking_ok() const {
            std::vector<const memory_desc_t *> mds
                    = {src_md(), dst_md(), weights_md(0)};
            if (with_bias()) mds.push_back(weights_md(1));
            for (const memory_desc_t *md : mds) {
                memory_desc_wrapper mdw(md);
                if (mdw.is_blocking_desc()) {
                    if (mdw.blocking_desc().inner_nblks != 0) { return false; }
                }
            }
            return true;
        }
    };

    status_t init(impl::engine_t *engine) override {
        // LT matmul
        matmul_impl_.reset(new cudnn_matmul_lt_impl_t());

        bool has_runtime_args = pd()->params_->has_runtime_params();
        if (has_runtime_args) {
            executor_.reset(new cudnn_matmul_lt_runtime_args_exec_t);
        } else if (!has_runtime_args) {
            executor_.reset(new cudnn_matmul_lt_exec_t);
            matmul_impl_->set_non_runtime_params(pd()->params_);
        }

        if (pd()->params_->with_bias_) {
            CHECK(create_nested_primitive(binary_, pd()->binary_pd_, engine));
        }

        if (!pd()->default_scale(DNNL_ARG_DST)
                && (pd()->params_->multi_dst_scale_
                        || pd()->params_->acc_type_ == CUDA_R_32I)) {
            CHECK(create_nested_primitive(
                    dst_scale_binary_, pd()->dst_scale_binary_pd_, engine));
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

    std::shared_ptr<impl::primitive_t> binary_;
    std::shared_ptr<impl::primitive_t> dst_scale_binary_;
    std::shared_ptr<cudnn_matmul_lt_impl_t> matmul_impl_;
    std::shared_ptr<cudnn_matmul_lt_base_exec_t> executor_;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

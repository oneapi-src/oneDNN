/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
* Copyright 2024-2025 Codeplay Software Limited
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

#ifndef GPU_GENERIC_SYCL_REF_INNER_PRODUCT_HPP
#define GPU_GENERIC_SYCL_REF_INNER_PRODUCT_HPP

#include "gpu/generic/sycl/ref_matmul.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl::impl::gpu::generic::sycl {
struct ref_inner_product_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_inner_product_fwd_pd_t {
        using gpu_inner_product_fwd_pd_t::gpu_inner_product_fwd_pd_t;
        using sm = primitive_attr_t::skip_mask_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_inner_product_fwd_t);

        status_t init(impl::engine_t *engine) {
            auto src_dt = arg_md(DNNL_ARG_SRC)->data_type;
            auto weights_dt = arg_md(DNNL_ARG_WEIGHTS)->data_type;
            auto dst_dt = arg_md(DNNL_ARG_DST)->data_type;
            auto bias_dt = with_bias() ? arg_md(DNNL_ARG_BIAS)->data_type
                                       : data_type::undef;

            const bool ok = (set_default_params() == status::success)
                    && is_fwd()
                    && check_if_dtypes_valid(
                            src_dt, dst_dt, bias_dt, weights_dt)
                    && sycl_post_ops_t::post_ops_ok(attr())
                    && (attr_.set_default_formats(dst_md()) == status::success)
                    // Blocked memory formats are not supported
                    && memory_desc_wrapper(src_md()).is_plain()
                    && memory_desc_wrapper(dst_md()).is_plain()
                    && memory_desc_wrapper(weights_md()).is_plain();

            if (!ok) { return status::unimplemented; }
            CHECK(create_ip_mds());
            CHECK(init_matmul(engine));

            // book scratchpad for the matmul
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    matmul_pd->scratchpad_registry());
            return status::success;
        }

        std::shared_ptr<primitive_desc_t> matmul_pd;

    private:
        bool check_if_dtypes_valid(const data_type_t &src_dt,
                const data_type_t &dst_dt, const data_type_t &bias_dt,
                const data_type_t &weight_dt) const {
            using namespace data_type;
            return (utils::one_of(src_dt, f32) && utils::one_of(weight_dt, f32)
                           && utils::one_of(dst_dt, f32)
                           && utils::one_of(bias_dt, f32, undef))
                    || (utils::one_of(src_dt, f16)
                            && utils::one_of(weight_dt, f16)
                            && utils::one_of(dst_dt, f16, f32, s8, u8)
                            && utils::one_of(bias_dt, f16, f32, undef))
                    || (utils::one_of(src_dt, u8, s8)
                            && utils::one_of(weight_dt, s8)
                            && utils::one_of(dst_dt, u8, s8, s32, bf16, f32)
                            && utils::one_of(
                                    bias_dt, u8, s8, s32, bf16, f32, undef))
                    || (utils::one_of(src_dt, bf16)
                            && utils::one_of(weight_dt, bf16)
                            && utils::one_of(dst_dt, f32, bf16)
                            && utils::one_of(bias_dt, f32, bf16, undef));
        }

        std::vector<int> get_dim_order(int ndims, const dims_t strides) {
            std::vector<int> order(ndims);
            for (int i = 0; i < ndims; ++i) {
                order[i] = i;
            }

            std::sort(
                    order.begin(), order.end(), [&strides](size_t i, size_t j) {
                        return strides[i] < strides[j];
                    });

            return order;
        }

        status_t create_ip_mds() {
            auto accumulate_dimensions = [](const dims_t dimensions, int start,
                                                 int end) -> int64_t {
                int64_t accum = 1;
                for (int i = start; i < end; i++) {
                    accum *= dimensions[i];
                }
                return accum;
            };

            const auto src_md_ = arg_md(DNNL_ARG_SRC);
            const auto weights_md_ = arg_md(DNNL_ARG_WEIGHTS);
            const auto bias_md_ = arg_md(DNNL_ARG_BIAS);
            auto src_wrap = memory_desc_wrapper(src_md_);
            auto w_wrap = memory_desc_wrapper(weights_md_);

            // src and weights dims need to be in the same order
            if (get_dim_order(src_wrap.ndims(), src_wrap.strides())
                    != get_dim_order(w_wrap.ndims(), w_wrap.strides())) {
                return status::unimplemented;
            }

            // Reshape input into the form of Batch x (\prod_{dim_{n-1}}^dim_0)
            if (src_md_->ndims == 2) {
                src_md_reshaped = *src_md_;
            } else {
                int64_t src_flattened_dimension = accumulate_dimensions(
                        src_md_->dims, 1, src_md_->ndims);
                dims_t src_reshaped_dims {
                        src_md_->dims[0], src_flattened_dimension};
                CHECK(memory_desc_init_by_tag(src_md_reshaped, 2,
                        src_reshaped_dims, src_md_->data_type, format_tag::ab));
            }

            // Reshape weights as (OC x (\prod_{dim_{n-1}}^dim_0))^T
            int weights_flattened_dimensions = accumulate_dimensions(
                    weights_md_->dims, 1, weights_md_->ndims);
            dims_t weights_reshaped_dims {
                    weights_flattened_dimensions, weights_md_->dims[0]};
            CHECK(memory_desc_init_by_tag(weights_md_reshaped, 2,
                    weights_reshaped_dims, weights_md_->data_type,
                    format_tag::ba));
            if (with_bias()) {
                dims_t bias_reshaped_dims {1, bias_md_->dims[0]};
                CHECK(memory_desc_init_by_tag(bias_md_reshaped, 2,
                        bias_reshaped_dims, bias_md_->data_type,
                        format_tag::ab));
            }
            return status::success;
        }

        status_t init_matmul(impl::engine_t *engine);
        // Memory descriptors to contain reshaped tensors from nD to 2D for IP
        memory_desc_t src_md_reshaped;
        memory_desc_t weights_md_reshaped;
        memory_desc_t bias_md_reshaped;
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
    std::shared_ptr<impl::primitive_t> matmul_primitive;
};
} // namespace dnnl::impl::gpu::generic::sycl

#endif

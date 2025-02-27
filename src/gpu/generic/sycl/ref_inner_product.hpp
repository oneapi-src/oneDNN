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

#include "common/opdesc.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/reduction_pd.hpp"
#include "common/reorder.hpp"
#include "common/tag_traits.hpp"
#include "gpu/generic/sycl/ref_matmul.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/gpu_primitive.hpp"

namespace dnnl::impl::gpu::generic::sycl {

namespace detail {
status_t init_matmul_pd(impl::engine_t *engine,
        const primitive_attr_t *attributes, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_desc,
        std::shared_ptr<primitive_desc_t> &matmul_pd);

status_t init_reorder_pd(impl::engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *dst_md,
        std::shared_ptr<primitive_desc_t> &reorder_pd);

status_t get_primitive_descriptor(op_desc_t *op_desc,
        const primitive_attr_t *attributes, impl::engine_t *engine,
        std::shared_ptr<primitive_desc_t> &pd);

std::vector<int> get_dim_order(int ndims, const dims_t strides);

void get_flattened_dimension(const dims_t &dims, dims_t &squished_dims,
        dim_t ndims, bool swap_dimensions = false);

bool strides_in_desc_order(const dims_t &strides, dim_t ndims);
} // namespace detail

struct ref_inner_product_fwd_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_inner_product_fwd_pd_t {
        using gpu_inner_product_fwd_pd_t::gpu_inner_product_fwd_pd_t;
        using sm = primitive_attr_t::skip_mask_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_inner_product_fwd_t);

        status_t init(impl::engine_t *engine);

        std::shared_ptr<primitive_desc_t> matmul_pd;
        std::shared_ptr<primitive_desc_t> src_reorder_pd;
        std::shared_ptr<primitive_desc_t> weights_reorder_pd;

        bool has_zero_dim = false;
        bool src_needs_reorder = false;
        bool wei_needs_reorder = false;

    private:
        bool check_if_dtypes_valid(const data_type_t &src_dt,
                const data_type_t &dst_dt, const data_type_t &bias_dt,
                const data_type_t &weight_dt) const;
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_primitive;
    std::shared_ptr<impl::primitive_t> src_reorder_primitive;
    std::shared_ptr<impl::primitive_t> weights_reorder_primitive;
};

struct ref_inner_product_bwd_data_t : public gpu::generic::sycl::primitive_t {

    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_inner_product_bwd_data_pd_t {
        using gpu_inner_product_bwd_data_pd_t::gpu_inner_product_bwd_data_pd_t;
        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_inner_product_bwd_data_t);

        status_t init(impl::engine_t *engine);

        std::shared_ptr<primitive_desc_t> matmul_pd;
        std::shared_ptr<primitive_desc_t> dst_reorder_pd;
        std::shared_ptr<primitive_desc_t> wei_reorder_pd;

        bool has_zero_dim = false;
        bool dst_needs_reorder = false;
        bool wei_needs_reorder = false;

    private:
        bool check_bwd_data_dtypes(const data_type_t &src_dt,
                const data_type_t &dst_dt, const data_type_t &weight_dt) const;
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_primitive;
    std::shared_ptr<impl::primitive_t> dst_reorder_primitive;
    std::shared_ptr<impl::primitive_t> wei_reorder_primitive;
};

struct ref_inner_product_bwd_weights_t
    : public gpu::generic::sycl::primitive_t {

    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_inner_product_bwd_weights_pd_t {
        using gpu_inner_product_bwd_weights_pd_t::
                gpu_inner_product_bwd_weights_pd_t;
        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_inner_product_bwd_weights_t);

        status_t init(impl::engine_t *engine);

        std::shared_ptr<primitive_desc_t> matmul_pd;
        std::shared_ptr<primitive_desc_t> reduction_pd;
        std::shared_ptr<primitive_desc_t> dst_reorder_pd;
        std::shared_ptr<primitive_desc_t> wei_reorder_pd;

        bool has_zero_dim = false;
        bool wei_requires_reorder = false;
        bool dst_requires_reorder = false;

    private:
        bool check_bwd_weights_dtypes(const data_type_t &src_dt,
                const data_type_t &dst_dt, const data_type_t &weight_dt,
                const data_type_t &bias_dt) const;

        status_t init_reduction_pd(impl::engine_t *engine,
                const memory_desc_t *src_desc, const memory_desc_t *dest_desc);
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_primitive;
    std::shared_ptr<impl::primitive_t> reduction_primitive;

    std::shared_ptr<impl::primitive_t> dst_reorder_primitve;
    std::shared_ptr<impl::primitive_t> wei_reorder_primitive;
};

} // namespace dnnl::impl::gpu::generic::sycl

#endif

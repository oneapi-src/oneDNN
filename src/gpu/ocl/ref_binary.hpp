/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_OCL_REF_BINARY_HPP
#define GPU_OCL_REF_BINARY_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_binary_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_binary_t : public primitive_impl_t {
    struct pd_t : public gpu_binary_pd_t {
        using gpu_binary_pd_t::gpu_binary_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_binary_t);

        status_t init() {
            using namespace data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;
            bool ok = set_default_params() == status::success
                    && (utils::everyone_is(f32, src_md(0)->data_type,
                                src_md(1)->data_type, dst_md()->data_type)
                            || utils::everyone_is(bf16, src_md(0)->data_type,
                                    src_md(1)->data_type, dst_md()->data_type)
                            || utils::everyone_is(f16, src_md(0)->data_type,
                                    src_md(1)->data_type, dst_md()->data_type))
                    && attr()->has_default_values(attr_skip_mask)
                    && attr_post_ops_ok();

            if (!ok) return status::unimplemented;

            return init_conf();
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        bool with_eltwise(int position) const {
            return attr()->post_ops_.contain(primitive_kind::eltwise, position);
        }

        bool with_sum() const {
            return attr()->post_ops_.find(primitive_kind::sum) != -1;
        }

        float eltwise_alpha() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise(0) || with_eltwise(1)
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                    : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise(0) || with_eltwise(1)
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                    : 0.0f;
        }

        float sum_scale() const {
            const int sum_idx = attr()->post_ops_.find(primitive_kind::sum);
            return with_sum() ? attr()->post_ops_.entry_[sum_idx].sum.scale
                              : 0.0f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return with_eltwise(0) || with_eltwise(1)
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                    : dnnl_alg_kind_undef;
        }

        binary_conf_t conf;
    };

    ref_binary_t(const pd_t *apd) : primitive_impl_t(apd) {}

    ~ref_binary_t() {}

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        compute_engine->create_kernel(&kernel_, "ref_binary", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    status_t execute_ref(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

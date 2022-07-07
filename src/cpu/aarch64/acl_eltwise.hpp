/*******************************************************************************
* Copyright 2021-2022 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_ELTWISE_HPP
#define CPU_AARCH64_ACL_ELTWISE_HPP

#include "cpu/cpu_eltwise_pd.hpp"

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_eltwise_obj_t {
    arm_compute::NEActivationLayer act;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_eltwise_conf_t {
    arm_compute::ActivationLayerInfo act_info;
    // src and dst have the same info
    arm_compute::TensorInfo data_info;
};

struct acl_eltwise_resource_t : public resource_t {
    acl_eltwise_resource_t()
        : acl_eltwise_obj_(utils::make_unique<acl_eltwise_obj_t>()) {}

    status_t configure(const acl_eltwise_conf_t &aep) {
        if (!acl_eltwise_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_eltwise_obj_->src_tensor.allocator()->init(aep.data_info);
        acl_eltwise_obj_->dst_tensor.allocator()->init(aep.data_info);

        acl_eltwise_obj_->act.configure(&acl_eltwise_obj_->src_tensor,
                &acl_eltwise_obj_->dst_tensor, aep.act_info);

        return status::success;
    }

    acl_eltwise_obj_t &get_acl_obj() const { return *acl_eltwise_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_eltwise_resource_t);

private:
    std::unique_ptr<acl_eltwise_obj_t> acl_eltwise_obj_;
}; // acl_eltwise_resource_t

struct acl_eltwise_fwd_t : public primitive_t {
    struct pd_t : public cpu_eltwise_fwd_pd_t {
        using cpu_eltwise_fwd_pd_t::cpu_eltwise_fwd_pd_t;
        pd_t(const eltwise_desc_t *adesc, const primitive_attr_t *attr,
                const eltwise_fwd_pd_t *hint_fwd_pd)
            : cpu_eltwise_fwd_pd_t(adesc, attr, hint_fwd_pd), aep() {}

        DECLARE_COMMON_PD_T("acl", acl_eltwise_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            using namespace data_type;
            const memory_desc_wrapper data_d(data_md());

            bool ok = is_fwd() && one_of(data_d.data_type(), f32, s32, s8)
                    && !has_zero_dim_memory() && attr()->has_default_values()
                    && data_d.is_dense();
            if (!ok) return status::unimplemented;

            auto acl_data_t = acl_utils::get_acl_data_t(data_d.data_type());

            // Operator acts elementwise, so we only require that the product of
            // all the dimensions equals the total number of elements. We are
            // free to swap/combine dimensions. ACL performs SIMD parallelism
            // over the first dimension and thread parallelism over the second.
            // We pick a single dimension to thread over (taking the max of 2 to
            // reduce the chance of it being 1), with the remaining dimensions
            // to SIMD over.
            dim_t thread_dim = std::max(W(), ndims() >= 2 ? C() : 1);
            auto shape = arm_compute::TensorShape(
                    data_d.nelems() / thread_dim, thread_dim);
            aep.data_info = arm_compute::TensorInfo(shape, 1, acl_data_t);

            if (!acl_utils::acl_act_ok(desc()->alg_kind))
                return status::unimplemented;

            aep.act_info = acl_utils::get_acl_act(*desc());

            ACL_CHECK_VALID(arm_compute::NEActivationLayer::validate(
                    &aep.data_info, &aep.data_info, aep.act_info));

            return status::success;
        }

        acl_eltwise_conf_t aep;
    };

    acl_eltwise_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_eltwise_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        CHECK(r->configure(pd()->aep));
        mapper.add(this, std::move(r));

        return status::success;
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_eltwise_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_ELTWISE_HPP

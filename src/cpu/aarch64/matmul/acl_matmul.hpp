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

#ifndef ACL_MATMUL_HPP
#define ACL_MATMUL_HPP

#include "cpu/aarch64/matmul/acl_matmul_utils.hpp"

#include "cpu/aarch64/acl_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

struct acl_resource_t : public resource_t {
    acl_resource_t() : acl_obj_(utils::make_unique<acl_matmul_obj_t>()) {}

    status_t configure(const acl_matmul_conf_t &amp) {
        if (!acl_obj_) return status::out_of_memory;
        acl_obj_->src_tensor.allocator()->init(amp.src_info);
        acl_obj_->wei_tensor.allocator()->init(amp.wei_info);
        acl_obj_->dst_tensor.allocator()->init(amp.dst_info);
        // Configure transpose kernel for src, wei or both
        if (amp.is_transA) {
            acl_obj_->src_acc_tensor.allocator()->init(amp.src_acc_info);
            acl_obj_->transA.configure(
                    &acl_obj_->src_acc_tensor, &acl_obj_->src_tensor);
        }
        if (amp.is_transB) {
            acl_obj_->wei_acc_tensor.allocator()->init(amp.wei_acc_info);
            acl_obj_->transB.configure(
                    &acl_obj_->wei_acc_tensor, &acl_obj_->wei_tensor);
        }
        // Configure GEMM
        acl_obj_->gemm.configure(&acl_obj_->src_tensor, &acl_obj_->wei_tensor,
                nullptr, &acl_obj_->dst_tensor, amp.alpha, 0.0f, amp.gemm_info);
        return status::success;
    }
    acl_matmul_obj_t &get_acl_obj() const { return *acl_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_resource_t);

private:
    std::unique_ptr<acl_matmul_obj_t> acl_obj_;
};

struct acl_matmul_t : public primitive_t {
    struct pd_t : public dnnl::impl::cpu::matmul::cpu_matmul_pd_t {

        pd_t(const matmul_desc_t *adesc, const primitive_attr_t *attr,
                const cpu_matmul_pd_t *hint_fwd_pd)
            : cpu_matmul_pd_t(adesc, attr, hint_fwd_pd), amp_(), post_ops() {}

        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("gemm:acl", acl_matmul_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            using smask_t = primitive_attr_t::skip_mask_t;
            const bool is_fp32_ok
                    = utils::everyone_is(data_type::f32, src_md()->data_type,
                              weights_md()->data_type, dst_md()->data_type,
                              desc()->accum_data_type)
                    && platform::has_data_type_support(data_type::f32);
            const bool is_fp16_ok
                    = utils::everyone_is(data_type::f16, src_md()->data_type,
                              weights_md()->data_type, dst_md()->data_type)
                    && platform::has_data_type_support(data_type::f16);
            bool ok = utils::one_of(true, is_fp32_ok, is_fp16_ok)
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            smask_t::oscale | smask_t::post_ops)
                    && attr_oscale_ok() && !has_runtime_dims_or_strides();
            if (!ok) return status::unimplemented;

            CHECK(acl_matmul_utils::init_conf_matmul(
                    amp_, src_md_, weights_md_, dst_md_, *desc(), *attr()));

            arm_compute::ActivationLayerInfo act_info;
            CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_, act_info));
            amp_.gemm_info.set_activation_info(act_info);
            amp_.use_dst_acc = post_ops.has_sum();

            // Validate ACL GEMM
            ACL_CHECK_VALID(arm_compute::NEGEMM::validate(&amp_.src_info,
                    &amp_.wei_info, nullptr, &amp_.dst_info, amp_.alpha, 0.0f,
                    amp_.gemm_info));

            return status::success;
        }

        acl_matmul_conf_t amp_;

        acl_post_ops_t post_ops;

    protected:
        bool attr_oscale_ok() const {
            const auto &oscale = attr()->output_scales_;
            return oscale.mask_ == 0;
        }
    };

    acl_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;
        auto r = utils::make_unique<acl_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        CHECK(r->configure(pd()->amp_));
        mapper.add(this, std::move(r));

        CHECK(pd()->post_ops.create_resource(engine, mapper));

        return status::success;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    // To guard the const execute_forward(), the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_matmul_t

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

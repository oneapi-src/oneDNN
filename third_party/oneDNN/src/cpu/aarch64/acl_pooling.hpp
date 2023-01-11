/*******************************************************************************
* Copyright 2022 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_POOLING_HPP
#define CPU_AARCH64_ACL_POOLING_HPP

#include "cpu/aarch64/acl_utils.hpp"
#include "cpu/cpu_pooling_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_pooling_obj_t {
    arm_compute::NEPoolingLayer pool;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_pooling_conf_t {
    arm_compute::PoolingLayerInfo pool_info;
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo dst_info;
};

struct acl_pooling_resource_t : public resource_t {
    acl_pooling_resource_t()
        : acl_pooling_obj_(utils::make_unique<acl_pooling_obj_t>()) {}

    status_t configure(const acl_pooling_conf_t &app) {
        if (!acl_pooling_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_pooling_obj_->src_tensor.allocator()->init(app.src_info);
        acl_pooling_obj_->dst_tensor.allocator()->init(app.dst_info);

        acl_pooling_obj_->pool.configure(&acl_pooling_obj_->src_tensor,
                &acl_pooling_obj_->dst_tensor, app.pool_info);

        return status::success;
    }

    acl_pooling_obj_t &get_acl_obj() const { return *acl_pooling_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_pooling_resource_t);

private:
    std::unique_ptr<acl_pooling_obj_t> acl_pooling_obj_;
}; // acl_pooling_resource_t

struct acl_pooling_fwd_t : public primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;
        pd_t(const pooling_v2_desc_t *adesc, const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : cpu_pooling_fwd_pd_t(adesc, attr, hint_fwd_pd), app() {}

        DECLARE_COMMON_PD_T("acl", acl_pooling_fwd_t);

        status_t init(engine_t *engine) {
            bool ok = set_default_params() == status::success
                    && is_fwd() // ACL supports forward propagation only
                    && utils::everyone_is(data_type::f32, src_md()->data_type,
                            dst_md()->data_type)
                    && attr()->has_default_values()
                    && attr_.set_default_formats(dst_md(0)) == status::success
                    && !is_dilated() && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            const pooling_v2_desc_t *pod = desc();

            // Choose the pooling type
            const alg_kind_t alg = pod->alg_kind;
            const bool is_max_pool = (alg == alg_kind::pooling_max);
            app.pool_info.pool_type = is_max_pool
                    ? arm_compute::PoolingType::MAX
                    : arm_compute::PoolingType::AVG;

            // Max forward training requires a worksace tensor.
            // For this workspace tensor, oneDNN uses pool window coordinates,
            // Whereas ACL uses absolute image coordinates.
            // Due to this mismatch, reject max forward training cases
            ACL_CHECK_SUPPORT(
                    (is_max_pool
                            && pod->prop_kind == prop_kind::forward_training),
                    "ACL does not support training for max pooling in oneDNN");

            // When padding is larger than the kernel, infinite values are
            // produced.
            // ACL and oneDNN use different values to represent infinity
            // which is difficult to account for, so return unimplemented.
            // See https://github.com/oneapi-src/oneDNN/issues/1205
            ACL_CHECK_SUPPORT(KH() <= padT() || KH() <= padB() || KW() <= padL()
                            || KW() <= padR(),
                    "ACL does not support pooling cases where padding >= kernel"
                    " in oneDNN");

            auto src_tag = memory_desc_matches_one_of_tag(
                    *src_md(), format_tag::nhwc, format_tag::nchw);
            auto dst_tag = memory_desc_matches_one_of_tag(
                    *dst_md(), format_tag::nhwc, format_tag::nchw);

            ACL_CHECK_SUPPORT(
                    utils::one_of(format_tag::undef, src_tag, dst_tag),
                    "src or dst is not format nhwc or nchw");
            ACL_CHECK_SUPPORT(src_tag != dst_tag,
                    "src and dst have different memory formats");

            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            const int ndims = src_d.ndims();
            ACL_CHECK_SUPPORT(ndims != 4, "Tensor is not 4d");

            // Pooling window
            app.pool_info.pool_size = arm_compute::Size2D(KW(), KH());
            // Choose the data layout
            bool is_nspc = utils::one_of(src_tag, format_tag::nhwc);
            const auto acl_layout = is_nspc ? arm_compute::DataLayout::NHWC
                                            : arm_compute::DataLayout::NCHW;
            app.pool_info.data_layout = acl_layout;
            const auto acl_data_t
                    = acl_utils::get_acl_data_t(src_d.data_type());

            ACL_CHECK_SUPPORT(
                    !use_acl_heuristic(MB() * IC() * OH() * OW() * KH() * KW(),
                            dnnl_get_max_threads(), is_max_pool, is_nspc),
                    "ACL is unoptimal in this case");

            app.pool_info.exclude_padding
                    = (alg == alg_kind::pooling_avg_exclude_padding);

            app.pool_info.pad_stride_info = arm_compute::PadStrideInfo(KSW(),
                    KSH(), padL(), padR(), padT(), padB(),
                    arm_compute::DimensionRoundingType::FLOOR);

            app.src_info = arm_compute::TensorInfo(is_nspc
                            ? arm_compute::TensorShape(IC(), IW(), IH(), MB())
                            : arm_compute::TensorShape(IW(), IH(), IC(), MB()),
                    1, acl_data_t, acl_layout);
            app.dst_info = arm_compute::TensorInfo(is_nspc
                            ? arm_compute::TensorShape(OC(), OW(), OH(), MB())
                            : arm_compute::TensorShape(OW(), OH(), OC(), MB()),
                    1, acl_data_t, acl_layout);

            ACL_CHECK_VALID(arm_compute::NEPoolingLayer::validate(
                    &app.src_info, &app.dst_info, app.pool_info));

            return status::success;
        }

        bool use_acl_heuristic(
                int problem_size, int thread_count, bool is_max, bool is_nhwc) {
            // For nhwc, ACL is faster above a certain problem size 'cutoff'
            // This cutoff scales linearly with thread count (except 1 thread)
            // So return true iff problem size is larger than this cutoff.
            // Note: This rule is approximate, Not all problems follow this rule
            if (is_nhwc) {
                if (is_max) {
                    if (thread_count == 1)
                        return problem_size > 512;
                    else
                        return problem_size > 4096 * thread_count;
                } else { // pooling_alg == avg_p || pooling_alg == avg_np
                    if (thread_count == 1)
                        return problem_size > 1024;
                    else
                        return problem_size > 8192 * thread_count;
                }
            } else { // memory_format == nchw
                return false;
            }
        }

        acl_pooling_conf_t app;
    };

    acl_pooling_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_pooling_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        auto st = r->configure(pd()->app);
        if (st == status::success) { mapper.add(this, std::move(r)); }

        return st;
    }

private:
    // execute_forward has to be const thus mutability of mtx
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_pooling_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_POOLING_HPP

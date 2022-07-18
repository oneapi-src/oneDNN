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

#ifndef CPU_AARCH64_ACL_INNER_PRODUCT_HPP
#define CPU_AARCH64_ACL_INNER_PRODUCT_HPP

#include "cpu/aarch64/acl_utils.hpp"
#include "cpu/cpu_inner_product_pd.hpp"

#include "cpu/aarch64/acl_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_ip_obj_t {
    arm_compute::NEFullyConnectedLayer fc;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
};

struct acl_ip_conf_t {
    bool with_bias;
    // If this is true, the result of the inner product goes into a temporarily
    // allocated ACL tensor to be accumulated into the oneDNN dst during postops
    bool use_dst_acc;
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo wei_info;
    arm_compute::TensorInfo bia_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::FullyConnectedLayerInfo fc_info;
};
struct acl_ip_resource_t : public resource_t {
    acl_ip_resource_t() : acl_ip_obj_(utils::make_unique<acl_ip_obj_t>()) {}

    status_t configure(const acl_ip_conf_t &aip) {
        if (!acl_ip_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_ip_obj_->src_tensor.allocator()->init(aip.src_info);
        acl_ip_obj_->wei_tensor.allocator()->init(aip.wei_info);
        acl_ip_obj_->dst_tensor.allocator()->init(aip.dst_info);
        acl_ip_obj_->bia_tensor.allocator()->init(aip.bia_info);

        // clang-format off
        acl_ip_obj_->fc.configure(
            &acl_ip_obj_->src_tensor,
            &acl_ip_obj_->wei_tensor,
            aip.with_bias ? &acl_ip_obj_->bia_tensor : nullptr,
            &acl_ip_obj_->dst_tensor,
            aip.fc_info);
        // clang-format on

        return status::success;
    }

    acl_ip_obj_t &get_acl_obj() const { return *acl_ip_obj_; }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_ip_resource_t);

private:
    std::unique_ptr<acl_ip_obj_t> acl_ip_obj_;
}; // acl_ip_resource_t

struct acl_inner_product_fwd_t : public primitive_t {
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_inner_product_fwd_pd_t(adesc, attr, hint_fwd_pd), aip() {}

        DECLARE_COMMON_PD_T("acl", acl_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            const bool ok = is_fwd() && !has_zero_dim_memory()
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops,
                            data_type::f32)
                    && set_default_params() == status::success;

            if (!ok) return status::unimplemented;

            CHECK(init_conf_ip(engine));

            return status::success;
        }

        acl_ip_conf_t aip;

        acl_post_ops_t post_ops;

        status_t init_conf_ip(engine_t *engine) {

            ACL_CHECK_SUPPORT(src_md()->ndims != weights_md()->ndims,
                    "source and weights dimensions must match");

            const int ndims = src_md()->ndims;

            const bool is_2d = (ndims == 2);
            const bool is_4d = (ndims == 4);

            ACL_CHECK_SUPPORT(
                    !(is_2d || is_4d), "ACL supports only 2d or 4d cases");

            // batch size
            const int n = src_md()->dims[0];

            // input and output channels
            const int ic = src_md()->dims[1];
            const int oc = dst_md()->dims[1];

            // source spatial dimensions
            const int ih = is_4d ? src_md()->dims[ndims - 2] : 0;
            const int iw = is_4d ? src_md()->dims[ndims - 1] : 0;

            // weights spatial dimensions
            const int kh = is_4d ? weights_md()->dims[ndims - 2] : 0;
            const int kw = is_4d ? weights_md()->dims[ndims - 1] : 0;

            // Only NCHW or NHWC derivatives supported by ACL kernels
            using namespace format_tag;
            auto src_tag = memory_desc_matches_one_of_tag(
                    src_md_, nhwc, nchw, nc, cn);
            auto wei_tag = memory_desc_matches_one_of_tag(
                    weights_md_, ohwi, oihw, oi, io);
            auto dst_tag = memory_desc_matches_one_of_tag(dst_md_, nc, cn);

            ACL_CHECK_SUPPORT(
                    utils::one_of(format_tag::undef, src_tag, wei_tag, dst_tag),
                    "unsupported memory layout");

            ACL_CHECK_SUPPORT(is_2d && src_tag != dst_tag,
                    "for src and dst layouts must match");

            arm_compute::TensorShape src_shape, wei_shape;
            if (is_2d) {
                src_shape = (src_tag == nc) ? arm_compute::TensorShape(ic, n)
                                            : arm_compute::TensorShape(n, ic);

                wei_shape = (wei_tag == io) ? arm_compute::TensorShape(oc, ic)
                                            : arm_compute::TensorShape(ic, oc);
            }
            if (is_4d) {
                src_shape = (src_tag == nhwc)
                        ? arm_compute::TensorShape(ic, iw, ih, n)
                        : arm_compute::TensorShape(iw, ih, ic, n);

                // ACL requires the weights to be in 2D flattened shape
                const int flattened_ic = is_4d ? ic * kh * kw : ic;
                wei_shape = arm_compute::TensorShape(flattened_ic, oc);
            }

            arm_compute::DataLayout src_layout = (src_tag == nhwc)
                    ? arm_compute::DataLayout::NHWC
                    : arm_compute::DataLayout::NCHW;

            arm_compute::DataLayout wei_layout = (wei_tag == ohwi)
                    ? arm_compute::DataLayout::NHWC
                    : arm_compute::DataLayout::NCHW;

            aip.src_info = arm_compute::TensorInfo(
                    src_shape, 1, arm_compute::DataType::F32, src_layout);

            aip.wei_info = arm_compute::TensorInfo(
                    wei_shape, 1, arm_compute::DataType::F32, wei_layout);

            aip.dst_info
                    = arm_compute::TensorInfo(arm_compute::TensorShape(oc, n),
                            1, arm_compute::DataType::F32);

            aip.with_bias = desc()->bias_desc.format_kind != format_kind::undef;
            aip.bia_info = arm_compute::TensorInfo(aip.with_bias
                            ? arm_compute::TensorShape(oc)
                            : arm_compute::TensorShape(),
                    1, arm_compute::DataType::F32);

            aip.fc_info.weights_trained_layout = wei_layout;
            if (is_2d && wei_tag != src_tag) {
                // weights are already transposed
                aip.fc_info.transpose_weights = false;

                if (desc()->prop_kind == dnnl_forward_training) {
                    aip.wei_info.set_are_values_constant(false);
                    aip.fc_info.are_weights_reshaped = true;
                }
            }

            // Fast math mode
            auto math_mode = get_fpmath_mode();
            bool is_fastmath_enabled = utils::one_of(
                    math_mode, fpmath_mode::bf16, fpmath_mode::any);
            aip.fc_info.enable_fast_math = is_fastmath_enabled;

            CHECK(post_ops.init(engine, attr_.post_ops_, dst_md_,
                    aip.fc_info.activation_info));
            aip.use_dst_acc = post_ops.has_sum();

            // clang-format off
            // Validate fully connected layer manually to check for return status
            ACL_CHECK_VALID(arm_compute::NEFullyConnectedLayer::validate(
                &aip.src_info,
                &aip.wei_info,
                aip.with_bias ? &aip.bia_info : nullptr,
                &aip.dst_info,
                aip.fc_info));
            // clang-format on
            return status::success;
        }
    }; // pd_t

    acl_inner_product_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_ip_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        CHECK(r->configure(pd()->aip));
        mapper.add(this, std::move(r));

        CHECK(pd()->post_ops.create_resource(engine, mapper));

        return status::success;
    }

    using data_t = typename prec_traits<data_type::f32>::type;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    //To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
}; // acl_inner_product_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_INNER_PRODUCT_HPP

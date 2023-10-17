/*******************************************************************************
* Copyright 2023 Arm Ltd. and affiliates
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
#ifndef CPU_AARCH64_ACL_REORDER_HPP
#define CPU_AARCH64_ACL_REORDER_HPP

#include "arm_compute/core/Types.h"
#include "common/utils.hpp"
#include "cpu/aarch64/acl_utils.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_reorder_obj_t {
    arm_compute::NEReorderLayer reorder;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor dst_tensor;
    arm_compute::WeightFormat src_wf;
    arm_compute::WeightFormat dst_wf;
};

struct acl_reorder_conf_t {
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::WeightFormat src_wf;
    arm_compute::WeightFormat dst_wf;
};

struct acl_reorder_resource_t : public resource_t {
    acl_reorder_resource_t()
        : acl_obj_(utils::make_unique<acl_reorder_obj_t>()) {}

    status_t configure(const acl_reorder_conf_t &app) {
        if (!acl_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_obj_->src_tensor.allocator()->init(app.src_info);
        acl_obj_->dst_tensor.allocator()->init(app.dst_info);

        // clang-format off
        acl_obj_->reorder.configure(
            &acl_obj_->src_tensor,
            &acl_obj_->dst_tensor,
            app.src_wf,
            app.dst_wf
            );
        // clang-format on

        return status::success;
    }

    acl_reorder_obj_t &get_acl_obj() const { return *acl_obj_; }
    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_reorder_resource_t);

private:
    std::unique_ptr<acl_reorder_obj_t> acl_obj_;
}; // acl_reorder_resource_t

struct acl_reorder_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_reorder_pd_t {

        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("acl", acl_reorder_fwd_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {

            using namespace acl_utils;

            bool ok = src_md->data_type
                            == dst_md->data_type // ACL reorder only supports matching src/dst data types
                    && utils::one_of(src_md->data_type,
                            data_type::f32) // Only supports f32 for now
                    && attr->has_default_values();
            if (!ok) return status::unimplemented;

            int mask = -1;
            bool is_set = false;
            CHECK(attr->scales_.get(DNNL_ARG_DST, &mask, &is_set));
            const memory_desc_wrapper input_d(src_md);
            if (input_d.has_runtime_dims_or_strides() && is_set && mask > 0)
                return status::unimplemented;

            // Create and check primitive descriptor
            auto _pd = make_unique_pd<pd_t>(attr, src_engine->kind(), src_md,
                    dst_engine->kind(), dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init(engine, src_engine, dst_engine) != status::success) {
                return status::unimplemented;
            }

            auto src_tag = memory_desc_matches_one_of_tag(
                    *src_md, format_tag::ba, format_tag::cdba);
            ACL_CHECK_SUPPORT(utils::one_of(format_tag::undef, src_tag),
                    "Only ba and cdba source formats supported");

            arm_compute::TensorShape acl_tensor_shape_in;
            arm_compute::TensorShape acl_tensor_shape_out;
            const memory_desc_wrapper src_d(*src_md);
            const int ndims = src_d.ndims();
            // Need even amount of dims in dim 0 for ACL kernel (eg mulitple of 8 rows when blocking by 8)
            int dim_0_rounded_up;

            // Switch for 2 or 4 dim tensors
            switch (ndims) {
                // Currently for Ab4a and Ab8a
                // No format_tag for these, have to deduce from stride
                case 2: {
                    if (dst_md->dims[0] == 1 || dst_md->dims[1] == 1) {
                        return status::unimplemented;
                    }
                    int dst_dim_1 = dst_md->dims[1];
                    int dst_dim_0_stride
                            = dst_md->format_desc.blocking.strides[0];
                    int dst_dim_1_stride
                            = dst_md->format_desc.blocking.strides[1];
                    // Interleave of 4 or 8 that stride for dim 1
                    if (dst_dim_1_stride != 4 && dst_dim_1_stride != 8) {
                        return status::unimplemented;
                    }
                    // Check to ensure it's a blocking transpose
                    if (dst_dim_1 * dst_dim_1_stride != dst_dim_0_stride) {
                        return status::unimplemented;
                    }
                    if (dst_dim_1_stride == 4) {
                        // Set Dest WeightFormat
                        _pd->app_.dst_wf = arm_compute::WeightFormat::OHWIo4;
                        dim_0_rounded_up = utils::rnd_up(src_md->dims[0], 4);
                        // Blocking for Ab8a only supported with SVE length 256
                    } else if (dst_dim_1_stride == 8 && mayiuse(sve_256)) {
                        // Set Dest WeightFormat
                        _pd->app_.dst_wf = arm_compute::WeightFormat::OHWIo8;
                        dim_0_rounded_up = utils::rnd_up(src_md->dims[0], 8);
                    } else {
                        return status::unimplemented;
                    }
                    acl_tensor_shape_in = arm_compute::TensorShape(
                            src_md->dims[1], src_md->dims[0]);
                    acl_tensor_shape_out = arm_compute::TensorShape(
                            src_md->dims[1], dim_0_rounded_up);

                    break;
                }
                // Currently supports Acdb4a and Acdb8a
                case 4: {

                    auto dst_tag = memory_desc_matches_one_of_tag(
                            *dst_md, format_tag::Acdb4a, format_tag::Acdb8a);
                    ACL_CHECK_SUPPORT(utils::one_of(format_tag::undef, dst_tag),
                            "Only Acdb4a and Acdb8a dst format supported for "
                            "4d tensors");

                    if (dst_tag == format_tag::Acdb4a) {
                        // Set Dest WeightFormat
                        _pd->app_.dst_wf = arm_compute::WeightFormat::OHWIo4;
                        dim_0_rounded_up = utils::rnd_up(src_md->dims[0], 4);
                        // Blocking for Acdb8a only supported with SVE length 256
                    } else if (dst_tag == format_tag::Acdb8a
                            && mayiuse(sve_256)) {
                        // Set Dest WeightFormat
                        _pd->app_.dst_wf = arm_compute::WeightFormat::OHWIo8;
                        dim_0_rounded_up = utils::rnd_up(src_md->dims[0], 8);
                    } else {
                        return status::unimplemented;
                    }
                    // Currently only supporting AxBx1x1 cases
                    if (dst_md->dims[2] != 1 || dst_md->dims[3] != 1) {
                        return status::unimplemented;
                    }

                    if (dst_md->dims[0] == 1 || dst_md->dims[1] == 1) {
                        return status::unimplemented;
                    }

                    acl_tensor_shape_in = arm_compute::TensorShape(
                            src_md->dims[3], src_md->dims[2], src_md->dims[1],
                            src_md->dims[0]);
                    acl_tensor_shape_out = arm_compute::TensorShape(
                            src_md->dims[3], src_md->dims[2], src_md->dims[1],
                            dim_0_rounded_up);
                    break;
                }
                default: return status::unimplemented;
            }

            // Choose the data layout
            const auto acl_layout = arm_compute::DataLayout::NCHW;

            // Set Source WeightFormat
            _pd->app_.src_wf = arm_compute::WeightFormat::OHWI;

            // Create ACL tensor infos
            const data_type_t data_type = src_d.data_type();
            const arm_compute::DataType acl_data_t
                    = acl_utils::get_acl_data_t(data_type);
            _pd->app_.src_info = arm_compute::TensorInfo(
                    acl_tensor_shape_in, 1, acl_data_t, acl_layout);
            _pd->app_.dst_info = arm_compute::TensorInfo(
                    acl_tensor_shape_out, 1, acl_data_t, acl_layout);

            ACL_CHECK_VALID(arm_compute::NEReorderLayer::validate(
                    &_pd->app_.src_info, &_pd->app_.dst_info, _pd->app_.src_wf,
                    _pd->app_.dst_wf));

            // Init scratch memory, not used so 0 in this implementation
            _pd->init_scratchpad_md();

            return safe_ptr_assign(*reorder_pd, _pd.release());
        } // create

        friend dnnl::impl::impl_list_item_t;
        acl_reorder_conf_t app_;

    }; // pd_t

    acl_reorder_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_reorder_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        CHECK(r->configure(pd()->app_));

        mapper.add(this, std::move(r));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    // To guard the const execute_forward, the mutex must be 'mutable'
    mutable std::mutex mtx;
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

}; // acl_reorder_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_REORDER_HPP

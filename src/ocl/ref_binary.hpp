/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef OCL_REF_BINARY_HPP
#define OCL_REF_BINARY_HPP

#include "common/c_types_map.hpp"
#include "compute/compute.hpp"
#include "ocl/jit_ref_binary_common_kernel.hpp"
#include "ocl/ocl_binary_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct ref_binary_t : public primitive_impl_t {
    struct pd_t : public ocl_binary_pd_t {
        using ocl_binary_pd_t::ocl_binary_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_binary_t);

        status_t init() {
            using namespace data_type;
            bool ok = true && set_default_params() == status::success
                    && (utils::everyone_is(f32, src_md(0)->data_type,
                                src_md(1)->data_type, dst_md()->data_type)
                            || utils::everyone_is(bf16, src_md(0)->data_type,
                                    src_md(1)->data_type, dst_md()->data_type)
                            || utils::everyone_is(f16, src_md(0)->data_type,
                                    src_md(1)->data_type, dst_md()->data_type))
                    && attr()->has_default_values();

            if (!ok) return status::unimplemented;

            return jit_ref_binary_common_kernel::init_conf(jib_, this);
        }

        jit_binary_conf_t jib_;
    };

    ref_binary_t(const pd_t *apd) : primitive_impl_t(apd) {}

    ~ref_binary_t() {}

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        auto status = jit_ref_binary_common_kernel::init_const_def(
                kernel_ctx, pd()->jib_);
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
} // namespace impl
} // namespace dnnl

#endif

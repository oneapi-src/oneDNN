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

#ifndef OCL_CROSS_ENGINE_REORDER_PD_HPP
#define OCL_CROSS_ENGINE_REORDER_PD_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/utils.hpp"
#include "ocl/cl_engine.hpp"
#include "ocl/jit_simple_reorder_kernel.hpp"
#include "ocl/ocl_reorder_pd.hpp"
#include "ocl/ocl_utils.hpp"

/* cross reorder manages all reorders between ocl and other engines.
 * It manages:
 * 1. reorder on alien side
 * 2. transition between engines
 * 3. reorder on ocl side if needed
 */

extern const char *simple_reorder_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

struct ocl_cross_engine_reorder_t : public primitive_t {
    struct pd_t : public reorder_pd_t {
        using reorder_pd_t::reorder_pd_t;

        DECLARE_COMMON_PD_T("cross_engine::any", ocl_cross_engine_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {

            auto _pd = new pd_t(
                    engine, attr, src_engine, src_md, dst_engine, dst_md);
            if (_pd == nullptr)
                return status::out_of_memory;
            if (_pd->init() != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        status_t init() {
            const auto &post_ops = attr()->post_ops_;
            bool args_ok = true
                    && utils::one_of(engine_kind::gpu, src_engine_->kind(),
                               dst_engine_->kind())
                    && utils::one_of(src_engine_->kind(), engine_kind::gpu,
                               engine_kind::cpu)
                    && utils::one_of(dst_engine_->kind(), engine_kind::gpu,
                               engine_kind::cpu)
                    && (attr()->has_default_values()
                        || IMPLICATION(post_ops.len_ != 0,
                            post_ops.len_ == 1
                            && post_ops.entry_[0].kind == primitive_kind::sum));
            if (!args_ok)
                return status::unimplemented;

            auto *cl_engine = utils::downcast<cl_engine_t *>(
                    dst_engine_->kind() == engine_kind::gpu ? dst_engine_
                                                            : src_engine_);

            args_ok = args_ok
                    && cl_engine->mayiuse(cl_device_ext_t::intel_subgroups)
                    && IMPLICATION(utils::one_of(data_type::f16,
                                           src_md()->data_type,
                                           dst_md()->data_type),
                               true
                                       && cl_engine->mayiuse(
                                                  cl_device_ext_t::khr_fp16)
                                       && cl_engine->mayiuse(cl_device_ext_t::
                                                          intel_subgroups_short));

            jit_simple_reorder_kernel::init_conf(
                    this, jrp_, src_md(), dst_md());
            return status::success;
        }

        jit_reorder_conf_t jrp_;
    };

    virtual status_t init() override {
        auto jit = ocl_jit_t(simple_reorder_kernel);

        auto status = jit_simple_reorder_kernel::init_const_def(
                jit, pd()->jrp_, pd()->src_md(), pd()->dst_md());
        if (status != status::success)
            return status;

        status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("any2any_kernel");
        if (!kernel_)
            return status::runtime_error;

        if (pd()->jrp_.do_reorder) {
            size_t size = pd()->jrp_.nelems * sizeof(float);
            memory_storage_t *temp_buf_ptr;
            engine()->create_memory_storage(&temp_buf_ptr, size);
            temp_buf.reset(temp_buf_ptr);
            if (!temp_buf)
                return status::runtime_error;
        }

        return status::success;
    }

    ocl_cross_engine_reorder_t(const pd_t *apd) : primitive_t(apd) {
        ker_ = new jit_simple_reorder_kernel(pd()->jrp_);
    }
    ~ocl_cross_engine_reorder_t() { delete ker_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    ocl_kernel_t kernel_;
    jit_simple_reorder_kernel *ker_;
    std::unique_ptr<memory_storage_t> temp_buf;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

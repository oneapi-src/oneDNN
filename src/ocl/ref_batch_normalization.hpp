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

#ifndef OCL_REF_BATCH_NORMALIZATION_FWD_HPP
#define OCL_REF_BATCH_NORMALIZATION_FWD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "ocl/cl_engine.hpp"
#include "ocl/jit_ref_bnorm_common_kernel.hpp"
#include "ocl/ocl_batch_normalization_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

extern const char *ref_bnorm_common_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t data_type>
struct ref_batch_normalization_fwd_t : public primitive_t {
    struct pd_t : public ocl_batch_normalization_fwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : ocl_batch_normalization_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jbn_()
            , jit_off_() {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_batch_normalization_fwd_t);

        status_t init() {
            auto *cl_engine = utils::downcast<cl_engine_t *>(engine());

            bool ok = true
                    && is_fwd()
                    && utils::everyone_is(data_type, src_md()->data_type,
                               dst_md()->data_type)
                    && IMPLICATION(data_type == data_type::f16,
                               !is_training() && stats_is_src())
                    && (attr()->has_default_values() || with_relu_post_op())
                    && cl_engine->mayiuse(cl_device_ext_t::intel_subgroups);
            if (!ok)
                return status::unimplemented;

            if (src_md()->data_type == data_type::s8 && !stats_is_src())
                return status::unimplemented;

            if (is_training() && fuse_norm_relu())
                init_default_ws(8);

            return jit_ref_bnorm_common_kernel::init_conf(
                    jbn_, desc_, src_md(), this, jit_off_);
        }

        jit_bnorm_conf_t jbn_;
        jit_offsets jit_off_;
    };

    status_t init() override {
        auto jit = ocl_jit_t(ref_bnorm_common_kernel);
        jit_ref_bnorm_common_kernel::init_const_def(
                jit, pd()->jbn_, pd()->jit_off_);

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_bnorm_fwd_kernel");
        if (!kernel_)
            return status::runtime_error;

        if (pd()->jbn_.use_16mb_unroll && pd()->jbn_.calculate_stats) {
            size_t size = 2 * pd()->jbn_.mb_chunk * pd()->jbn_.sp_chunk
                    * pd()->jbn_.ic * sizeof(data_t);
            memory_storage_t *temp_reduce_ptr;
            engine()->create_memory_storage(&temp_reduce_ptr, size);
            temp_reduce.reset(temp_reduce_ptr);
            if (!temp_reduce)
                return status::runtime_error;

            calculate_mean_kernel_ = jit.get_kernel("calculate_mean");
            if (!calculate_mean_kernel_)
                return status::runtime_error;

            calculate_variance_kernel_ = jit.get_kernel("calculate_variance");
            if (!calculate_variance_kernel_)
                return status::runtime_error;

            reduce_mean_kernel_ = jit.get_kernel("reduce_mean");
            if (!reduce_mean_kernel_)
                return status::runtime_error;

            reduce_variance_kernel_ = jit.get_kernel("reduce_variance");
            if (!reduce_variance_kernel_)
                return status::runtime_error;
        }
        return status::success;
    }

    ref_batch_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {
        ker_ = new jit_ref_bnorm_common_kernel(pd()->jbn_);
    }
    ~ref_batch_normalization_fwd_t() { delete ker_; }

    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_ref_bnorm_common_kernel *ker_;
    ocl_kernel_t kernel_;
    ocl_kernel_t calculate_mean_kernel_;
    ocl_kernel_t reduce_mean_kernel_;
    ocl_kernel_t calculate_variance_kernel_;
    ocl_kernel_t reduce_variance_kernel_;
    std::unique_ptr<memory_storage_t> temp_reduce;
};

template <impl::data_type_t data_type>
struct ref_batch_normalization_bwd_t : public primitive_t {
    struct pd_t : public ocl_batch_normalization_bwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : ocl_batch_normalization_bwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jbn_()
            , jit_off_() {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_batch_normalization_bwd_t);

        status_t init() {
            bool ok = true
                    && is_bwd()
                    && utils::everyone_is(data_type, src_md()->data_type,
                               diff_src_md()->data_type)
                    && IMPLICATION(use_scaleshift(),
                               utils::everyone_is(data_type,
                                       weights_md()->data_type,
                                       diff_weights_md()->data_type))
                    && attr()->has_default_values();
            if (!ok)
                return status::unimplemented;

            if (fuse_norm_relu()) {
                init_default_ws(8);
                if (!compare_ws(hint_fwd_pd_))
                    return status::unimplemented;
            }

            return jit_ref_bnorm_common_kernel::init_conf(
                    jbn_, desc_, diff_src_md(), this, jit_off_);
        }

        jit_bnorm_conf_t jbn_;
        jit_offsets jit_off_;
    };

    status_t init() override {
        auto jit = ocl_jit_t(ref_bnorm_common_kernel);
        jit_ref_bnorm_common_kernel::init_const_def(
                jit, pd()->jbn_, pd()->jit_off_);

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_bnorm_bwd_kernel");
        if (!kernel_)
            return status::runtime_error;

        if (pd()->jbn_.use_16mb_unroll) {
            size_t size = 2 * pd()->jbn_.mb_chunk * pd()->jbn_.sp_chunk
                    * pd()->jbn_.ic * sizeof(data_t);

            memory_storage_t *temp_reduce_ptr;
            engine()->create_memory_storage(&temp_reduce_ptr, size);
            temp_reduce.reset(temp_reduce_ptr);
            if (!temp_reduce)
                return status::runtime_error;

            calculate_stats_kernel_ = jit.get_kernel("calculate_stats");
            if (!calculate_stats_kernel_)
                return status::runtime_error;

            reduce_stats_kernel_ = jit.get_kernel("reduce_stats");
            if (!reduce_stats_kernel_)
                return status::runtime_error;
        }

        return status::success;
    }

    ref_batch_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {
        ker_ = new jit_ref_bnorm_common_kernel(pd()->jbn_);
    }
    ~ref_batch_normalization_bwd_t() { delete ker_; }

    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_ref_bnorm_common_kernel *ker_;
    ocl_kernel_t kernel_;
    ocl_kernel_t calculate_stats_kernel_;
    ocl_kernel_t reduce_stats_kernel_;
    std::unique_ptr<memory_storage_t> temp_reduce;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

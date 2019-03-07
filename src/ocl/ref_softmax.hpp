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

#ifndef OCL_REF_SOFTMAX_HPP
#define OCL_REF_SOFTMAX_HPP

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "ocl/jit_primitive_conf.hpp"
#include "ocl/ocl_softmax_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

extern const char *ref_softmax_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

template <impl::data_type_t data_type>
struct ref_softmax_fwd_t : public primitive_t {
    struct pd_t : public ocl_softmax_fwd_pd_t {
        pd_t(engine_t *engine, const softmax_desc_t *adesc,
                const primitive_attr_t *attr,
                const softmax_fwd_pd_t *hint_fwd_pd)
            : ocl_softmax_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_fwd_t);

        status_t init() {
            auto *ocl_engine = utils::downcast<const ocl_engine_t *>(engine());

            bool ok = true
                    && utils::one_of(desc()->prop_kind,
                               prop_kind::forward_inference,
                               prop_kind::forward_training)
                    && data_type == desc()->data_desc.data_type
                    && IMPLICATION(data_type == data_type::f16,
                            ocl_engine->mayiuse(cl_device_ext_t::khr_fp16))
                    && attr()->has_default_values();
            if (!ok)
                return status::unimplemented;

            for (int i = 0; i < src_md()->ndims; ++i) {
                if (i != desc()->softmax_axis)
                    gws.push_back(src_md()->dims[i]);
            }

            return status::success;
        }

        std::vector<size_t> gws;
    };

    ref_softmax_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    ~ref_softmax_fwd_t() = default;

    virtual status_t init() override {
        auto jit = ocl_jit_t(ref_softmax_kernel);

        const auto *desc = pd()->desc();
        jit.define_int("SOFTMAX_AXIS_IDX", desc->softmax_axis);
        jit.define_int(
                "SOFTMAX_AXIS", desc->data_desc.dims[desc->softmax_axis]);
        jit.set_data_type(desc->data_desc.data_type);

        set_offsets(jit, desc->data_desc, "DATA");

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_softmax_fwd_generic");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_generic(ctx);
    }

protected:
    status_t execute_generic(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    ocl_kernel_t kernel_;
};

template <impl::data_type_t data_type>
struct ref_softmax_bwd_t : public primitive_t {
    struct pd_t : public ocl_softmax_bwd_pd_t {
        pd_t(engine_t *engine, const softmax_desc_t *adesc,
            const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
            : ocl_softmax_bwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_bwd_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_data
                && desc()->data_desc.data_type == data_type::f32
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            for (int i = 0; i < desc()->data_desc.ndims; ++i) {
                if (i != desc()->softmax_axis)
                    gws.push_back(desc()->data_desc.dims[i]);
            }

            return status::success;
        }

        std::vector<size_t> gws;
    };

    ref_softmax_bwd_t(const pd_t *apd) : primitive_t(apd) {}

    ~ref_softmax_bwd_t() = default;

    virtual status_t init() override {
        auto jit = ocl_jit_t(ref_softmax_kernel);

        const auto *desc = pd()->desc();
        jit.define_int("SOFTMAX_AXIS_IDX", desc->softmax_axis);
        jit.define_int(
                "SOFTMAX_AXIS", desc->data_desc.dims[desc->softmax_axis]);
        jit.set_data_type(desc->data_desc.data_type);

        set_offsets(jit, desc->data_desc, "DATA");

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_softmax_bwd_generic");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_generic(ctx);
    }

protected:
    status_t execute_generic(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    ocl_kernel_t kernel_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

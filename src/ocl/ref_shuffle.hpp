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

#ifndef OCL_REF_SHUFFLE_HPP
#define OCL_REF_SHUFFLE_HPP

#include "common/c_types_map.hpp"
#include "ocl/jit_ref_shuffle_kernel.hpp"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_shuffle_pd.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_utils.hpp"

extern const char *ref_shuffle_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

template<int data_type_size>
struct ref_shuffle_t : public primitive_t {
    using shuffle_class = ref_shuffle_t<data_type_size>;
    struct pd_t : public ocl_shuffle_pd_t {
        using ocl_shuffle_pd_t::ocl_shuffle_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", shuffle_class);

        status_t init() {
            using namespace format_tag;
            auto *cl_engine = utils::downcast<cl_engine_t *>(engine());

            bool ok = true
                && data_type_size
                    == types::data_type_size(data_md()->data_type)
                && IMPLICATION(
                           desc()->data_desc.data_type == data_type::f16,
                           cl_engine->mayiuse(cl_device_ext_t::khr_fp16))
                && desc()->data_desc.data_type != data_type::bf16;
            if (!ok) return status::unimplemented;

            dat_tag_ = any;
            return jit_ref_shuffle_kernel::init_conf(this, jshfl_, jit_off_,
                src_md(), dst_md(), diff_src_md(), diff_dst_md());
        }

        jit_shuffle_conf_t jshfl_;
        jit_offsets jit_off_;
        format_tag_t dat_tag_;
    };

    ref_shuffle_t(const pd_t *apd) : primitive_t(apd) {}

    virtual status_t init() override {
        auto jit = ocl_jit_t(ref_shuffle_kernel);

        status_t status = jit_ref_shuffle_kernel::init_const_def(jit,
            pd()->jshfl_, pd()->jit_off_);
        if (status != status::success)
            return status;

        status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("ref_shuffle");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    ~ref_shuffle_t() {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_<format_tag::any>(ctx);
    }

private:
    template<format_tag_t tag>
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    ocl_kernel_t kernel_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

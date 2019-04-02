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

#ifndef SIMPLE_SUM_HPP
#define SIMPLE_SUM_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "ocl/jit_simple_sum_kernel.hpp"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_stream.hpp"
#include "ocl/ocl_sum_pd.hpp"
#include "ocl/ocl_utils.hpp"

extern const char *simple_sum_kernel;

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace mkldnn::impl::status;

template <data_type_t data_type>
struct simple_sum_t : public primitive_t {
    struct pd_t : public ocl_sum_pd_t {
        using ocl_sum_pd_t::ocl_sum_pd_t;

        DECLARE_SUM_PD_T("ocl:simple:any", simple_sum_t);

        status_t init() {
            const int n = n_inputs();

            bool ok = true
                    && ocl_sum_pd_t::init() == status::success
                    && n <= max_num_arrs;
            if (!ok)
                return unimplemented;

            const memory_desc_wrapper o_d(dst_md());
            ok = ok
                    && o_d.data_type() == data_type
                    && o_d.is_dense();
            if (!ok)
                return status::unimplemented;

            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(src_md(i));
                if (i_d != o_d)
                    return status::unimplemented;
            }

            return jit_simple_sum_kernel::init_conf(jss_, src_md(0));
        }
        jit_simple_sum_conf_t jss_;
    };

    simple_sum_t(const pd_t *apd) : primitive_t(apd) {
        ker_ = new jit_simple_sum_kernel(pd()->jss_);
    }

    virtual status_t init() override {
        auto jit = ocl_jit_t(simple_sum_kernel);
        jit_simple_sum_kernel::init_const_def(jit, pd()->jss_);

        status_t status = jit.build(engine());
        if (status != status::success)
            return status;

        kernel_ = jit.get_kernel("simple_sum_kernel");
        if (!kernel_)
            return status::runtime_error;

        return status::success;
    }

    ~simple_sum_t() { delete ker_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override;

    enum { max_num_arrs = 16 };
    typedef typename prec_traits<data_type>::type data_t;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_simple_sum_kernel *ker_;
    ocl_kernel_t kernel_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

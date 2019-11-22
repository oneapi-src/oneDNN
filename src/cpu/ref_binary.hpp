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

#ifndef REF_BINARY_HPP
#define REF_BINARY_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_isa_traits.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_binary_t : public primitive_impl_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_binary_t);

        status_t init() {
            using namespace data_type;
            bool ok = utils::everyone_is(data_type, src_md(0)->data_type,
                              src_md(1)->data_type, dst_md()->data_type)
                    && IMPLICATION(data_type == bf16, mayiuse(avx512_core))
                    && attr()->has_default_values()
                    && set_default_params() == status::success;
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_binary_t(const pd_t *apd) : primitive_impl_t(apd) {}

    ~ref_binary_t() {}

    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_ref(ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    void execute_ref(const exec_ctx_t &ctx) const;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

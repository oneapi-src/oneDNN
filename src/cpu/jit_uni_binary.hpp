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

#ifndef JIT_UNI_BINARY_HPP
#define JIT_UNI_BINARY_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_binary_pd.hpp"
#include "cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace binary_impl {
template <cpu_isa_t isa>
struct driver_t;
}

template <cpu_isa_t isa>
struct jit_uni_binary_t : public primitive_impl_t {
    struct pd_t : public cpu_binary_pd_t {
        pd_t(engine_t *engine, const binary_desc_t *adesc,
                const primitive_attr_t *attr, const binary_pd_t *hint_pd)
            : cpu_binary_pd_t(engine, adesc, attr, hint_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_binary_t);

        status_t init() {
            using namespace data_type;
            bool ok = mayiuse(isa)
                    && utils::everyone_is(
                            f32, src_md(0)->data_type, src_md(1)->data_type)
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && memory_desc_wrapper(src_md(0))
                            == memory_desc_wrapper(src_md(1))
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        };
    };

    jit_uni_binary_t(const pd_t *apd);
    ~jit_uni_binary_t();

    typedef float data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    binary_impl::driver_t<isa> *binary_driver_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

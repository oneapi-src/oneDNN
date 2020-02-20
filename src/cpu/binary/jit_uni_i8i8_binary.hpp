/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_JIT_UNI_I8I8_BINARY_HPP
#define CPU_JIT_UNI_I8I8_BINARY_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_binary_pd.hpp"
#include "cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct i8i8_binary_kernel_t;

template <data_type_t src0_type, data_type_t src1_type>
struct jit_uni_i8i8_binary_t : public primitive_impl_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_i8i8_binary_t);

        status_t init() {
            using namespace data_type;
            bool ok = mayiuse(avx2) && src_md(0)->data_type == src0_type
                    && src_md(1)->data_type == src1_type
                    && dst_md(0)->data_type == src0_type
                    && set_default_params()
                            == status::success /* should precede comparison */
                    && !has_zero_dim_memory()
                    && memory_desc_wrapper(src_md(0)).similar_to(
                            memory_desc_wrapper(src_md(1)), true, false, 0)
                    && memory_desc_wrapper(src_md(0))
                            == memory_desc_wrapper(dst_md(0))
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::scales)
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask());
            if (!ok) return status::unimplemented;

            return status::success;
        };

    private:
        bool check_scales_mask() const {
            for (const auto &s : attr()->scales_.scales_) {
                if (s.second.mask_ != 0) return false;
            }
            return true;
        }
    };

    jit_uni_i8i8_binary_t(const pd_t *apd);
    ~jit_uni_i8i8_binary_t();

    virtual status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    std::unique_ptr<i8i8_binary_kernel_t> kernel_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

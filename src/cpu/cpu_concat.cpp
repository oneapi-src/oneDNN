/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "cpu/cpu_engine.hpp"

#include "cpu/ref_concat.hpp"
#include "cpu/simple_concat.hpp"
#include "common/dnnl_sel_build.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using cpd_create_f = dnnl::impl::engine_t::concat_primitive_desc_create_f;
using namespace dnnl::impl::data_type;

namespace {
// clang-format off
#if defined(SELECTIVE_BUILD_ANALYZER)

DNNL_DEF_PD_BUILDER(cpd_builder,
            cpd_create_f,
            dnnl::impl::concat_pd_t **,
            dnnl::impl::engine_t *,
            const dnnl::impl::primitive_attr_t *,
            const dnnl::impl::memory_desc_t *,
            int,
            int,
            const dnnl::impl::memory_desc_t *);

# define CPD_INSTANCE(...) REG_DNNL_FN(cpd_builder, __VA_ARGS__)

#else   // !SELECTIVE_BUILD_ANALYZER

# define CPD_INSTANCE REG_DNNL_FN

#endif

#define INSTANCE CPD_INSTANCE
const cpd_create_f cpu_concat_impl_list[] = {
        INSTANCE(simple_concat_t, f32)
        INSTANCE(simple_concat_t, u8)
        INSTANCE(simple_concat_t, s8)
        INSTANCE(simple_concat_t, s32)
        INSTANCE(simple_concat_t, bf16)
        INSTANCE(ref_concat_t)
        nullptr,
};
#undef CPD_INSTANCE
#undef INSTANCE
// clang-format on
} // namespace

const cpd_create_f *cpu_engine_t::get_concat_implementation_list() const {
    return cpu_concat_impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

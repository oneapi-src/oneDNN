/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "cpu/ref_resampling.hpp"
#include "cpu/simple_resampling.hpp"

#if DNNL_X64
#include "cpu/x64/jit_avx512_common_resampling.hpp"
#include "cpu/x64/jit_uni_resampling.hpp"
using namespace dnnl::impl::cpu::x64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::prop_kind;

const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> &impl_list_map() {
    // clang-format off
    static std::map<pk_impl_key_t, std::vector<impl_list_item_t>> the_map = REG_RESAMPLING_P({
        {{forward}, {
            CPU_INSTANCE_X64(jit_uni_resampling_fwd_t)
            CPU_INSTANCE(simple_resampling_fwd_t)
            CPU_INSTANCE(ref_resampling_fwd_t)
            nullptr,
        }},
        {{backward}, REG_BWD_PK({
            CPU_INSTANCE_X64(jit_avx512_common_resampling_bwd_t)
            CPU_INSTANCE(simple_resampling_bwd_t)
            CPU_INSTANCE(ref_resampling_bwd_t)
            nullptr,
        })},
    });
    // clang-format on
    return the_map;
}
} // namespace

const impl_list_item_t *get_resampling_impl_list(
        const resampling_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : backward;

    pk_impl_key_t key {prop_kind};

    const auto impl_list_it = impl_list_map().find(key);
    return impl_list_it != impl_list_map().cend() ? impl_list_it->second.data()
                                                  : empty_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "gpu/gpu_impl_list.hpp"

#include "gpu/jit/binary_format.hpp"

#include "gpu/jit/conv/gen_convolution.hpp"
#include "gpu/ocl/gen9_wino_convolution.hpp"
#include "gpu/ocl/ref_convolution.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

namespace {
using namespace dnnl::impl::prop_kind;

// clang-format off
const std::map<pk_impl_key_t, std::vector<impl_list_item_t>>
        impl_list_map REG_CONV_P({
    {{forward}, {
        INSTANCE(jit::gen_convolution_fwd_t)
        INSTANCE(ocl::gen9_wino_convolution_fwd_t)
        INSTANCE(ocl::ref_convolution_fwd_t)
        nullptr,
    }},
    {{backward_data}, REG_BWD_D_PK({
        INSTANCE(jit::gen_convolution_bwd_data_t)
        INSTANCE(ocl::ref_convolution_bwd_data_t)
        nullptr,
    })},
    {{backward_weights}, REG_BWD_PK({
        INSTANCE(jit::gen_convolution_bwd_weights_t)
        INSTANCE(ocl::ref_convolution_bwd_weights_t)
        nullptr,
    })},
});
// clang-format on
} // namespace

const impl_list_item_t *get_convolution_impl_list(
        const convolution_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : desc->prop_kind;

    const auto impl_list_it = impl_list_map.find({prop_kind});
    return impl_list_it != impl_list_map.cend() ? impl_list_it->second.data()
                                                : empty_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl

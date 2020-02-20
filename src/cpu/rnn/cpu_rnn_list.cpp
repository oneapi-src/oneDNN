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

#include "cpu_engine.hpp"

#include "cpu/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using pd_create_f = engine_t::primitive_desc_create_f;

namespace {
using namespace dnnl::impl::data_type;

#define INSTANCE(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
static const pd_create_f impl_list[] = {
        INSTANCE(ref_rnn_fwd_f32_t),
        INSTANCE(ref_rnn_fwd_bf16_t),
        INSTANCE(ref_rnn_fwd_u8s8_t),
        INSTANCE(ref_rnn_bwd_f32_t),
        INSTANCE(ref_rnn_bwd_bf16_t),
        /* eol */
        nullptr,
};
#undef INSTANCE
} // namespace

const pd_create_f *get_rnn_impl_list(const rnn_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "jit_avx512_mic_gemm_f32.hpp"
#include "jit_avx512_mic_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;

jit_avx512_mic_inner_product_fwd_t::jit_avx512_mic_inner_product_fwd_t(
        const pd_t *pd, const input_vector &inputs,
        const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
{
    sgemm_ = new jit_avx512_mic_gemm_f32('T', 'N', 0.0, conf_.with_bias());
}

jit_avx512_mic_inner_product_fwd_t::~jit_avx512_mic_inner_product_fwd_t()
{
    delete sgemm_;
}

void jit_avx512_mic_inner_product_fwd_t::execute_forward()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper dst_d(conf_.dst_pd());

    // TODO: consistency checks
    int M = conf_.MB();
    int N = conf_.OC();
    int K = conf_.IC_total();

    float alpha = 1.0, beta = 0.0;
    sgemm_->sgemm("T", "N", &N, &M, &K, &alpha, weights, &K, src, &K, &beta,
            dst, &N, bias);
}
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

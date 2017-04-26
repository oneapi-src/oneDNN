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

#include "jit_avx2_gemm_f32.hpp"
#include "jit_avx512_common_gemm_f32.hpp"
#include "jit_uni_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;

template <cpu_isa_t isa>
jit_uni_inner_product_fwd_t<isa>::jit_uni_inner_product_fwd_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
{
    sgemm_ = new jit_uni_gemm_f32('T', 'N', 0.0, conf_.with_bias());
}

template <cpu_isa_t isa>
jit_uni_inner_product_fwd_t<isa>::~jit_uni_inner_product_fwd_t()
{
    delete sgemm_;
}

template <cpu_isa_t isa>
void jit_uni_inner_product_fwd_t<isa>::execute_forward()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    // TODO: consistency checks
    int MB = conf_.MB();
    int OC = conf_.OC();
    int IC = conf_.IC_total();

    float alpha = 1.0, beta = 0.0;
    sgemm_->sgemm("T", "N", &OC, &MB, &IC, &alpha, weights, &IC, src, &IC, &beta,
            dst, &OC, bias);
}

template <cpu_isa_t isa>
jit_uni_inner_product_bwd_weights_t<isa>::jit_uni_inner_product_bwd_weights_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
{
    sgemm_ = new jit_uni_gemm_f32('N', 'T', 0.0, false);
}

template <cpu_isa_t isa>
jit_uni_inner_product_bwd_weights_t<isa>::~jit_uni_inner_product_bwd_weights_t()
{
    delete sgemm_;
}

template <cpu_isa_t isa>
void jit_uni_inner_product_bwd_weights_t<isa>::execute_backward_weights()
{
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    // TODO: consistency checks
    int MB = conf_.MB();
    int OC = conf_.OC();
    int IC = conf_.IC_total();

    float alpha = 1.0, beta = 0.0;
    sgemm_->sgemm("N", "T", &IC, &OC, &MB, &alpha, src, &IC, diff_dst, &OC, &beta,
            diff_weights, &IC, nullptr);

    if (diff_bias) {
#       pragma omp parallel for schedule(static)
        for (int oc = 0; oc < OC; ++oc) {
            data_t *db = &diff_bias[diff_bias_d.off(oc)];
            *db = data_t(0);
            for (int mb = 0; mb < MB; ++mb) {
                *db += diff_dst[diff_dst_d.off(mb, oc)];
            }
        }
    }

}

template <cpu_isa_t isa>
jit_uni_inner_product_bwd_data_t<isa>::jit_uni_inner_product_bwd_data_t(const pd_t *pd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
{
    sgemm_ = new jit_uni_gemm_f32('N', 'N', 0.0, false);
}

template <cpu_isa_t isa>
jit_uni_inner_product_bwd_data_t<isa>::~jit_uni_inner_product_bwd_data_t()
{
    delete sgemm_;
}

template <cpu_isa_t isa>
void jit_uni_inner_product_bwd_data_t<isa>::execute_backward_data()
{
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory());

    // TODO: consistency checks
    int MB = conf_.MB();
    int OC = conf_.OC();
    int IC = conf_.IC_total();

    float alpha = 1.0, beta = 0.0;

    sgemm_->sgemm("N", "N", &IC, &MB, &OC, &alpha, weights, &IC, diff_dst, &OC, &beta,
            diff_src, &IC, nullptr);
}

template struct jit_uni_inner_product_bwd_data_t<avx512_common>;
template struct jit_uni_inner_product_bwd_weights_t<avx512_common>;
template struct jit_uni_inner_product_fwd_t<avx512_common>;
template struct jit_uni_inner_product_bwd_data_t<avx2>;
template struct jit_uni_inner_product_bwd_weights_t<avx2>;
template struct jit_uni_inner_product_fwd_t<avx2>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

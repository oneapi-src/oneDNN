/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
#include "mkldnn_thread.hpp"

#include "gemm_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::primitive_kind;

template <impl::data_type_t data_type>
void gemm_inner_product_fwd_t<data_type>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t*>(this->memory());

    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int IC = conf_.IC_total_padded();

    float alpha = 1.0, beta = 0.0;
    extended_sgemm("T", "N", &OC, &MB, &IC, &alpha, weights, &IC, src, &IC,
            &beta, dst, &OC, bias);
}

template <impl::data_type_t data_type>
void gemm_inner_product_bwd_data_t<data_type>::execute_backward_data() {
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_src = reinterpret_cast<data_t*>(this->memory());

    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int IC = conf_.IC_total_padded();

    float alpha = 1.0, beta = 0.0;
    extended_sgemm("N", "N", &IC, &MB, &OC, &alpha, weights, &IC, diff_dst,
            &OC, &beta, diff_src, &IC);
}

template <impl::data_type_t data_type>
void gemm_inner_product_bwd_weights_t<data_type>::execute_backward_weights() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto diff_dst = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto diff_weights = reinterpret_cast<data_t *>(this->memory(0));
    auto diff_bias = reinterpret_cast<data_t *>(this->memory(1));

    const memory_desc_wrapper diff_dst_d(conf_.diff_dst_pd());
    const memory_desc_wrapper diff_bias_d(conf_.diff_weights_pd(1));

    diff_dst += diff_dst_d.blocking_desc().offset_padding;

    const int MB = conf_.MB();
    const int OC = conf_.OC();
    const int IC = conf_.IC_total_padded();

    float alpha = 1.0, beta = 0.0;
    extended_sgemm("N", "T", &IC, &OC, &MB, &alpha, src, &IC, diff_dst, &OC,
            &beta, diff_weights, &IC);

    if (diff_bias) {
        diff_bias += diff_bias_d.blocking_desc().offset_padding;
        constexpr int blksize = 8;
        int OC_blocks = OC / blksize;
        int rem_OC = OC % blksize;
#       pragma omp parallel
        {
            const int ithr = omp_get_thread_num();
            const int nthr = omp_get_num_threads();
            int oc_st{0}, oc_e{0};
            balance211(OC_blocks, nthr, ithr, oc_st, oc_e);
            oc_st = oc_st * blksize;
            oc_e = oc_e * blksize;

            PRAGMA_OMP_SIMD()
            for (int oc = oc_st; oc < oc_e; ++oc) {
                diff_bias[oc] = diff_dst[oc];
            }

            for (int mb = 1; mb < MB; ++mb) {
                PRAGMA_OMP_SIMD()
                for (int oc = oc_st; oc < oc_e; ++oc) {
                    diff_bias[oc] += diff_dst[mb * OC + oc];
                }
            }

            if (rem_OC != 0 && ithr == nthr-1) {
                for (int oc = OC_blocks * blksize; oc < OC; oc++)
                    diff_bias[oc] = diff_dst[oc];
                for (int mb = 1; mb < MB; ++mb) {
                    for (int oc = OC_blocks * blksize; oc < OC; oc++) {
                        diff_bias[oc] += diff_dst[mb * OC + oc];
                    }
                }
            }
        }
    }
}

template struct gemm_inner_product_fwd_t<data_type::f32>;
template struct gemm_inner_product_bwd_data_t<data_type::f32>;
template struct gemm_inner_product_bwd_weights_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

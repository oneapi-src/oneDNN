/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "reference_pooling.hpp"
#include "type_helpers.hpp"

#include <limits>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;

template <impl::precision_t prec>
status_t reference_pooling<prec>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(
            this->input()[0].primitive->output()[
            this->input()[0].output_index]->memory_const());
    index_t *indices = (this->_is_training
        && (this->_ppd.pooling_desc.alg_kind != alg_kind::pooling_avg))
        ? reinterpret_cast<index_t*>(this->input()[1].primitive->output()[
                this->input()[1].output_index]->memory())
        : nullptr;
    auto dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_ppd.src_primitive_desc.memory_desc),
        indices_d(this->_ppd.indices_primitive_desc.memory_desc),
        dst_d(this->_ppd.dst_primitive_desc.memory_desc);

    const int IH = src_d.dims()[2];
    const int IW = src_d.dims()[3];
    const int KH = this->_ppd.pooling_desc.kernel[0];
    const int KW = this->_ppd.pooling_desc.kernel[1];
    const int SH = this->_ppd.pooling_desc.strides[0];
    const int SW = this->_ppd.pooling_desc.strides[1];
    const int PH = this->_ppd.pooling_desc.padding[0];
    const int PW = this->_ppd.pooling_desc.padding[1];

    auto ker_max = [=](data_t *d, int mb, int oc, int oh,
            int ow)
    {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                if (oh*SH + kh < nstl::max(0, PH)) continue;
                if (ow*SW + kw < nstl::max(0, PW)) continue;

                if (oh*SH + kh >= IH + PH) continue;
                if (ow*SW + kw >= IW + PW) continue;

                const int ih = oh * SH - PH + kh;
                const int iw = ow * SW - PW + kw;

                if (src[src_d.off(mb, oc, ih, iw)] > d[0]) {
                    d[0] = src[src_d.off(mb, oc, ih, iw)];
                    if (this->_is_training)
                        indices[indices_d.off(mb, oc, oh, ow)] = kh*KW + kw;
                }
            }
        }
    };

    auto ker_avg = [=](data_t *d, int mb, int oc, int oh,
        int ow)
    {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                if (oh*SH + kh < nstl::max(0, PH)) continue;
                if (ow*SW + kw < nstl::max(0, PW)) continue;

                if (oh*SH + kh >= IH + PH) continue;
                if (ow*SW + kw >= IW + PW) continue;

                const int ih = oh * SH - PH + kh;
                const int iw = ow * SW - PW + kw;

                d[0] += src[src_d.off(mb, oc, ih, iw)];
            }
        }
    };

    const int MB = src_d.dims()[0];
    const int OC = dst_d.dims()[1];
    const int OH = dst_d.dims()[2];
    const int OW = dst_d.dims()[3];

    if (this->_ppd.pooling_desc.alg_kind == mkldnn_pooling_max)
    {
#   pragma omp parallel for collapse(4) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        data_t *d = &dst[dst_d.off(mb, oc, oh, ow)];
                        d[0] = -std::numeric_limits<data_t>::infinity();
                        ker_max(d, mb, oc, oh, ow);
                    }
                }
            }
        }
    }
    else if (this->_ppd.pooling_desc.alg_kind == mkldnn_pooling_avg)
    {
#   pragma omp parallel for collapse(4) schedule(static)
        for (int mb = 0; mb < MB; ++mb) {
            for (int oc = 0; oc < OC; ++oc) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        data_t *d = &dst[dst_d.off(mb, oc, oh, ow)];
                        d[0] = 0;
                        ker_avg(d, mb, oc, oh, ow);
                        d[0] /= this->_ppd.pooling_desc.kernel[0] * this->_ppd.pooling_desc.kernel[1];
                    }
                }
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t reference_pooling<prec>::constraint(const pooling_desc_t &pool_d) {
    bool args_ok = true
        && one_of(pool_d.prop_kind, prop_kind::forward_training,
                prop_kind::forward_scoring)
        && one_of(pool_d.alg_kind, alg_kind::pooling_max,
                alg_kind::pooling_avg);
    return args_ok ? success : unimplemented;
}

template <impl::precision_t prec>
const primitive_impl reference_pooling<prec>::implementation = {
    reference_pooling<prec>::create
};

template class reference_pooling<precision::f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

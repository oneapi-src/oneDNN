/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "ref_softmax.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::execute_forward_dense() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

    for (int ou = 0; ou < outer_size_; ou++) {
        max_[0] = denom_[0] = 0;

        for (int c = 0; c < channels_; c++)
            max_[0] = nstl::max(max_[0], src[c]);

        for (int c = 0; c < channels_; c++)
            denom_[0] += dst[c] = exp(src[c] - max_[0]);

        for (int c = 0; c < channels_; c++)
            dst[c] /= denom_[0];

        src += channels_;
        dst += channels_;
    }
}

template <impl::data_type_t data_type>
void ref_softmax_fwd_t<data_type>::execute_forward_generic() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const size_t dim = channels_ * inner_size_;

    for (int ou = 0; ou < outer_size_; ou++) {
        utils::array_set(max_, 0, inner_size_);
        utils::array_set(denom_, 0, inner_size_);

        for (int c = 0; c < channels_; c++) {
            for(int in = 0; in < inner_size_; in++) {
                size_t off = data_d.off_l(ou * dim + c * inner_size_ + in);
                max_[in] = nstl::max(max_[in], src[off]);
            }
        }

        for (int c = 0; c < channels_; c++) {
            for(int in = 0; in < inner_size_; in++) {
                size_t off = data_d.off_l(ou * dim + c * inner_size_ + in);
                denom_[in] += dst[off] = exp(src[off] - max_[in]);
            }
        }

        for (int c = 0; c < channels_; c++) {
            for (int in = 0; in < inner_size_; in++) {
                size_t off = data_d.off_l(ou * dim + c * inner_size_ + in);
                dst[off] /= denom_[in];
            }
        }
    }
}

template struct ref_softmax_fwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

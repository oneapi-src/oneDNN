/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <string.h>

#include "nhwc_concat.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
void nhwc_concat_t<data_type>::execute() {
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const int num_srcs = conf_.n_inputs();

    for (int i = 0; i < num_srcs; ++i) {
        const memory_desc_wrapper src_d(conf_.src_pd(i));
        const memory_desc_wrapper img_d(conf_.src_image_pd(i));
        ic[i] = src_d.dims()[1];
        src[i] = reinterpret_cast<const data_t *>(this->input_memory(i));
        img[i] = dst + img_d.blk_off(0);
    }

    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const int n = dst_d.dims()[0];
    const int oc = dst_d.dims()[1];
    const int h = dst_d.dims()[2];
    const int w = dst_d.dims()[3];

#   pragma omp parallel for schedule(static) collapse(2)
    for (int iter_n = 0; iter_n < n; ++iter_n) {
        for (int iter_h = 0; iter_h < h; ++iter_h) {
            for (int iter_w = 0; iter_w < w; ++iter_w) {
                for (int iter_srcs = 0; iter_srcs < num_srcs; ++iter_srcs) {
                    const size_t e = iter_n * h * w + iter_h * w + iter_w;
                    const data_t *i = &src[iter_srcs][e*ic[iter_srcs]];
                    data_t *o = &img[iter_srcs][e*oc];
                    memcpy(o, i, ic[iter_srcs] * sizeof(data_t));
                }
            }
        }
    }
}

template struct nhwc_concat_t<data_type::f32>;
template struct nhwc_concat_t<data_type::u8>;
template struct nhwc_concat_t<data_type::s8>;
template struct nhwc_concat_t<data_type::s32>;

}
}
}

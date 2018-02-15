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

#include "simple_concat.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
void simple_concat_t<data_type>::execute() {
    const int num_arrs = conf_.n_inputs();
    const data_t *input_ptrs[max_num_arrs];
    data_t *output_ptrs[max_num_arrs];
    size_t nelems_no_d0[max_num_arrs];
    size_t is[max_num_arrs];

    auto o_base_ptr = reinterpret_cast<data_t *>(this->memory());

    for (int a = 0; a < num_arrs; ++a) {
        const memory_desc_wrapper i_d(conf_.src_pd(a));
        const memory_desc_wrapper o_d(conf_.src_image_pd(a));

        input_ptrs[a] = reinterpret_cast<const data_t *>(
                this->input_memory(a)) + i_d.blk_off(0);
        output_ptrs[a] = o_base_ptr + o_d.blk_off(0);

        nelems_no_d0[a] = nelems_no_dim_0(i_d);
        is[a] = size_t(i_d.blocking_desc().strides[0][0]);
    }

    const memory_desc_wrapper o_d(conf_.src_image_pd());
    const int N = o_d.dims()[0];
    const size_t os = size_t(o_d.blocking_desc().strides[0][0]);

#   pragma omp parallel for collapse(2) schedule(static)
    for (int n = 0; n < N; ++n) {
        for (int a = 0; a < num_arrs; ++a) {
            /* do coping */
            const data_t *i = &input_ptrs[a][is[a] * size_t(n)];
            data_t *o = &output_ptrs[a][os * size_t(n)];
            for (size_t e = 0; e < nelems_no_d0[a]; ++e) o[e] = i[e];
        }
    }
}

template struct simple_concat_t<data_type::f32>;
template struct simple_concat_t<data_type::u8>;
template struct simple_concat_t<data_type::s8>;
template struct simple_concat_t<data_type::s32>;

}
}
}

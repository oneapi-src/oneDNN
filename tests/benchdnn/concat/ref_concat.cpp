/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#include "utils/parallel.hpp"

#include "concat/concat.hpp"

namespace concat {

void get_sizes(const prb_t *prb, int64_t &outer_size, int64_t &inner_size,
        int64_t &axis_size) {
    outer_size = inner_size = 1;
    for (int i = 0; i < prb->axis; i++)
        outer_size *= prb->vdims[0][i];
    for (int i = prb->axis + 1; i < prb->ndims; i++)
        inner_size *= prb->vdims[0][i];
    axis_size = prb->axis_size();
}

void compute_ref(
        const prb_t *prb, const args_t &args, dnnl_primitive_t prim_ref) {
    const dnn_mem_t &dst = args.find(DNNL_ARG_DST);

    float *dst_ptr = (float *)dst;

    int64_t outer_size {0}, inner_size {0}, axis_size {0};
    get_sizes(prb, outer_size, inner_size, axis_size);

    benchdnn_parallel_nd(outer_size, inner_size, [&](int64_t ou, int64_t in) {
        int64_t off_dst = ou * axis_size * inner_size;
        for (int i_input = 0; i_input < prb->n_inputs(); ++i_input) {
            const dnn_mem_t &src_i = args.find(DNNL_ARG_MULTIPLE_SRC + i_input);
            int64_t i_axis_size = prb->vdims[i_input][prb->axis];
            int64_t off_src = ou * i_axis_size * inner_size;

            float scale_i
                    = prb->attr.scales.get(DNNL_ARG_MULTIPLE_SRC + i_input)
                              .scale;

            for (int64_t as = 0; as < i_axis_size; ++as) {
                int64_t idx = as * inner_size + in;
                dst_ptr[off_dst + idx]
                        = src_i.get_elem(off_src + idx) * scale_i;
            }
            // the next input start point
            off_dst += i_axis_size * inner_size;
        }
    });
}

} // namespace concat

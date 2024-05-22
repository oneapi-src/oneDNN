/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

namespace dnnl {
namespace impl {
namespace gpu {

const impl_list_item_t *gpu_impl_list_t::get_implementation_list(
        const op_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    // clang-format off
#define CASE(kind) \
    case primitive_kind::kind: \
        return get_##kind##_impl_list((const kind##_desc_t *)desc);
        switch ((int)desc->kind) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(gemm);
            CASE(group_normalization);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(matmul);
            CASE(pooling);
            CASE(prelu);
            CASE(reduction);
            CASE(resampling);
            CASE(rnn);
            CASE(sdpa);
            CASE(shuffle);
            CASE(softmax);
            CASE(zero_pad);
            default: assert(!"unknown primitive kind"); return empty_list;
        }
#undef CASE
    // clang-format on
}

const impl_list_item_t *gpu_impl_list_t::get_concat_implementation_list() {
    return get_concat_impl_list();
}

const impl_list_item_t *gpu_impl_list_t::get_sum_implementation_list() {
    return get_sum_impl_list();
}

const impl_list_item_t *gpu_impl_list_t::get_reorder_implementation_list(
        const memory_desc_t *src_md, const memory_desc_t *dst_md) {
    return get_reorder_impl_list(src_md, dst_md);
}

} // namespace gpu
} // namespace impl
} // namespace dnnl

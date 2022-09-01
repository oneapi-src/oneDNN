/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef GPU_GPU_IMPL_LIST_HPP
#define GPU_GPU_IMPL_LIST_HPP

#include <map>
#include <vector>

#include "common/engine.hpp"
#include "common/impl_list_item.hpp"
#include "common/impl_registration.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

#define INSTANCE(...) \
    impl_list_item_t( \
            impl_list_item_t::type_deduction_helper_t<__VA_ARGS__::pd_t>()),

#define DECLARE_IMPL_LIST(kind) \
    const impl_list_item_t *get_##kind##_impl_list(const kind##_desc_t *desc);

DECLARE_IMPL_LIST(batch_normalization);
DECLARE_IMPL_LIST(binary);
DECLARE_IMPL_LIST(convolution);
DECLARE_IMPL_LIST(deconvolution);
DECLARE_IMPL_LIST(eltwise);
DECLARE_IMPL_LIST(gemm);
DECLARE_IMPL_LIST(inner_product);
DECLARE_IMPL_LIST(layer_normalization);
DECLARE_IMPL_LIST(lrn);
DECLARE_IMPL_LIST(matmul);
DECLARE_IMPL_LIST(pooling);
DECLARE_IMPL_LIST(prelu);
DECLARE_IMPL_LIST(reduction);
DECLARE_IMPL_LIST(resampling);
DECLARE_IMPL_LIST(rnn);
DECLARE_IMPL_LIST(shuffle);
DECLARE_IMPL_LIST(softmax);
DECLARE_IMPL_LIST(zero_pad);

#undef DECLARE_IMPL_LIST

class gpu_impl_list_t {
public:
    static const impl_list_item_t *get_concat_implementation_list();
    static const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md, const memory_desc_t *dst_md);
    static const impl_list_item_t *get_sum_implementation_list();
    static const impl_list_item_t *get_implementation_list(
            const op_desc_t *desc);
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_GPU_IMPL_LIST_HPP

/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_CONV_TRANSPOSE_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_CONV_TRANSPOSE_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/backend/dnnl/kernels/conv_base.hpp"

#include "graph/backend/dnnl/dnnl_constant_tensor_cache.hpp"
#include "graph/backend/dnnl/dnnl_partition_impl.hpp"
#include "graph/backend/dnnl/scratchpad.hpp"
#include "graph/backend/dnnl/subgraph.hpp"
#include "graph/backend/dnnl/thread_local_cache.hpp"

#include "graph/backend/dnnl/passes/memory_planning.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

template <bool quantized>
struct conv_transpose_fwd_t : public conv_base_t {
public:
    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    status_t prepare_inplace_pairs_impl() override;

    DEF_KERNEL_METHOD_STR(conv_transpose_fwd_t)
};

using float_convtranspose_fwd = conv_transpose_fwd_t</* quantized */ false>;
using quantized_convtranspose = conv_transpose_fwd_t</* quantized */ true>;

#if BUILD_TRAINING
struct conv_transpose_bwd_data_t : public conv_base_t {
public:
    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    DEF_KERNEL_METHOD_STR(conv_transpose_bwd_data_t)
};

struct conv_transpose_bwd_weights_t : public conv_base_t {
public:
    status_t compile_impl(const dnnl_partition_impl_t *part,
            const engine_t *g_engine,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    DEF_KERNEL_METHOD_STR(conv_transpose_bwd_weights_t)
};
#endif

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#ifndef BACKEND_DNNL_PATTERNS_FUSIONS_HPP
#define BACKEND_DNNL_PATTERNS_FUSIONS_HPP

#include "backend/dnnl/patterns/transformation_pattern.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {
namespace pass {

DNNL_BACKEND_REGISTER_PASSES_DECLARE(conv_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(matmul_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(binary_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(bn_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(convtranspose_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(eltwise_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(gelu_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(interpolate_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(pool_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(quantize_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(reduction_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(reorder_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(shuffle_fusion)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(single_op_pass)
DNNL_BACKEND_REGISTER_PASSES_DECLARE(sum_fusion)

} // namespace pass
} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
#endif

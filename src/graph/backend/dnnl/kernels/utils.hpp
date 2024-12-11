/*******************************************************************************
* Copyright 2024 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_KERNELS_UTILS_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_UTILS_HPP

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// An internal env var is provided to force using primitive based SDPA
// implementation and skipping ukernel based optimization on GPU or
// decomposition based optimization on CPU. Currently it's for oneDNN debug
// and testing only.
bool force_primitive();

// It is used to check if enable the decomposition kernel based on user's
// env and params. Decomposition kernel is enabled when:
// - CPU runtime is OMP or THREADPOOl.
// - Primitive based implementation is not forced by the internal env var.
bool enable_decomp_kernel();

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/compute_stream.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "graph/interface/c_types_map.hpp"

using namespace dnnl::impl::gpu::intel;
#define MAX_NDIMS 6

#ifndef GPU_INTEL_OCL_GENINDEX_HPP
#define GPU_INTEL_OCL_GENINDEX_HPP

void create_genindex_kernel(const compute::compute_engine_t *compute_engine,
        compute::kernel_t &kernel, const int ndims,
        const dnnl::impl::graph::dims_t output_dims,
        const dnnl::impl::graph::dims_t output_strides);
void execute_genindex_kernel(compute::compute_stream_t *compute_stream,
        const compute::kernel_t &kernel,
        const dnnl::impl::memory_storage_t &dst, const int nelems,
        const int axis);
#endif
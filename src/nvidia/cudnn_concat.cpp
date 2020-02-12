/***************************************************************************
 *  Copyright 2020 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 **************************************************************************/

#include "gpu/ocl/ref_concat.hpp"
#include "nvidia/sycl_cuda_engine.hpp"

namespace dnnl {
namespace impl {
namespace cuda {

namespace {

static const engine_t::concat_primitive_desc_create_f cuda_concat_impl_list[]
        = {gpu::ocl::ref_concat_t::pd_t::create, nullptr};
} // namespace

const engine_t::concat_primitive_desc_create_f *
cuda_gpu_engine_impl_list_t::get_concat_implementation_list() {
    return cuda_concat_impl_list;
}

} // namespace cuda
} // namespace impl
} // namespace dnnl

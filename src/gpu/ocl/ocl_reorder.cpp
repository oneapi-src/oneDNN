/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "common/engine.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/ocl/cross_engine_reorder.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/rnn/rnn_reorders.hpp"
#include "gpu/ocl/simple_reorder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using rpd_create_f = engine_t::reorder_primitive_desc_create_f;

namespace {

using namespace dnnl::impl::data_type;

static const rpd_create_f ocl_ce_reorder_impl_list[]
        = {rnn_weights_reorder_t::pd_t::create,
                cross_engine_reorder_t::pd_t::create,
                simple_reorder_t::pd_t::create, nullptr};
} // namespace

const rpd_create_f *ocl_gpu_engine_impl_list_t::get_reorder_implementation_list(
        const memory_desc_t *, const memory_desc_t *) {
    return ocl_ce_reorder_impl_list;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

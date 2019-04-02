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
#include "ocl/ocl_cross_engine_reorder_pd.hpp"
#include "ocl/ocl_reorder_pd.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using rpd_create_f = engine_t::reorder_primitive_desc_create_f;

namespace {

using namespace mkldnn::impl::data_type;

static const rpd_create_f ocl_ce_reorder_impl_list[] = {
    ocl_cross_engine_reorder_t::pd_t::create,
    nullptr
};
} // namespace

const rpd_create_f *ocl_engine_t::get_reorder_implementation_list() const {
    return ocl_ce_reorder_impl_list;
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn

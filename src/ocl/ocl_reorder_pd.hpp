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

#ifndef OCL_REORDER_PD_HPP
#define OCL_REORDER_PD_HPP

#include "common/c_types_map.hpp"
#include "common/reorder_pd.hpp"
#include "common/utils.hpp"
#include "ocl/ocl_engine.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

struct ocl_reorder_pd_t : public reorder_pd_t {
    using reorder_pd_t::reorder_pd_t;

    status_t init() {
        bool args_ok = true
                && attr()->has_default_values();
        if (!args_ok)
            return status::unimplemented;
        return status::success;
    }
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn
#endif

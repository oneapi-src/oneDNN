/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef REORDER_HPP
#define REORDER_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"

namespace mkldnn {
namespace impl {

typedef status_t (*reorder_primitive_desc_init_f)(
        primitive_desc_t *primitive_desc,
        const memory_primitive_desc_t *input,
        const memory_primitive_desc_t *output);

}
}

#endif

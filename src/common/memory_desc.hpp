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

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {

status_t memory_desc_init_by_tag(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, format_tag_t tag);

status_t memory_desc_init_by_strides(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, const dims_t strides);

status_t memory_desc_init_submemory(memory_desc_t &memory_desc,
        const memory_desc_t &parent_memory_desc, const dims_t dims,
        const dims_t offsets);

status_t memory_desc_reshape(memory_desc_t &out_memory_desc,
        const memory_desc_t &in_memory_desc, int ndims, const dims_t dims);

status_t memory_desc_permute_axes(memory_desc_t &out_memory_desc,
        const memory_desc_t &in_memory_desc, const int *perm);

} // namespace impl
} // namespace dnnl

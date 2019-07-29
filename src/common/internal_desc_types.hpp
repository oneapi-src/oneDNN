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

#ifndef INTERNAL_DESC_TYPES_HPP
#define INTERNAL_DESC_TYPES_HPP

#include <vector>
#include "mkldnn_types.h"

namespace mkldnn {
namespace impl {

// The types are not exposed
typedef struct {
    mkldnn_primitive_kind_t primitive_kind;
    mkldnn_memory_desc_t src_md;
    mkldnn_memory_desc_t dst_md;
    mkldnn_engine_kind_t src_engine_kind;
    mkldnn_engine_kind_t dst_engine_kind;
} mkldnn_reorder_desc_t;

typedef struct {
    mkldnn_primitive_kind_t primitive_kind;
    mkldnn_memory_desc_t dst_md;
    mkldnn_dim_t n;
    mkldnn_dim_t concat_dimension;
    std::vector<mkldnn_memory_desc_t> src_mds;
} mkldnn_concat_desc_t;

typedef struct {
    mkldnn_primitive_kind_t primitive_kind;
    mkldnn_memory_desc_t dst_md;
    mkldnn_dim_t n;
    std::vector<float> scales;
    std::vector<mkldnn_memory_desc_t> src_mds;
} mkldnn_sum_desc_t;

} // namespace impl
} // namespace mkldnn

#endif // INTERNAL_DESC_TYPES

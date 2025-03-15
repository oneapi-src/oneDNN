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

#ifndef CPU_UKERNEL_C_TYPES_MAP_HPP
#define CPU_UKERNEL_C_TYPES_MAP_HPP

#include "oneapi/dnnl/dnnl_ukernel_types.h"

#include "common/c_types_map.hpp"

#ifdef DNNL_EXPERIMENTAL_UKERNEL

// A section identical to c_map_types.hpp but just for brgemm ukernel so far.
namespace dnnl {
namespace impl {
namespace cpu {
namespace ukernel {

using pack_type_t = dnnl_pack_type_t;
namespace pack_type {
const pack_type_t undef = dnnl_pack_type_undef;
const pack_type_t no_trans = dnnl_pack_type_no_trans;
const pack_type_t trans = dnnl_pack_type_trans;
const pack_type_t pack32 = dnnl_pack_type_pack32;
} // namespace pack_type

using attr_params_t = dnnl_ukernel_attr_params;
using brgemm_t = dnnl_brgemm;
using transform_t = dnnl_transform;

} // namespace ukernel
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

// Default macro settings (if any) can be defined here.
#ifndef NGEN_CONFIG_HPP
#define NGEN_CONFIG_HPP

#include "common/bfloat16.hpp"
#include "common/float16.hpp"

namespace ngen {
using bfloat16 = dnnl::impl::bfloat16_t;
using half = dnnl::impl::float16_t;
} // namespace ngen

#define NGEN_BFLOAT16_TYPE
#define NGEN_HALF_TYPE

#endif /* header guard */

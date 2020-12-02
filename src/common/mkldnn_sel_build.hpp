/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#pragma once
#include "mkldnn_macros.hpp"

#if defined(SELECTIVE_BUILD_ANALYZER)

#include <openvino/cc/selective_build.h>

namespace dnnl {

OV_CC_DOMAINS(MKLDNN)

} // namespace dnnl

#define MKLDNN_CSCOPE(region, ...) OV_SCOPE(MKLDNN, region, __VA_ARGS__)

#elif defined(SELECTIVE_BUILD)

#include <openvino/cc/selective_build.h>

#define MKLDNN_CSCOPE(region, ...) OV_SCOPE(MKLDNN, region, __VA_ARGS__)

#else

#define MKLDNN_CSCOPE(region, ...) __VA_ARGS__

#endif

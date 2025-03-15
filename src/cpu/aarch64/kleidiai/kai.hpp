/*******************************************************************************
* Copyright 2025 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed tos in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#ifndef CPU_AARCH64_KLEIDIAI_KAI_HPP
#define CPU_AARCH64_KLEIDIAI_KAI_HPP

#if defined(DNNL_EXPERIMENTAL_UKERNEL) && defined(DNNL_AARCH64_USE_KAI)

#include "cpu/aarch64/kleidiai/kai_f32_f32_f32p_kernel.hpp"
#include "cpu/aarch64/kleidiai/kai_f32_qai8dxp_qsi4c32p_kernel.hpp"
#include "cpu/aarch64/kleidiai/kai_f32_qai8dxp_qsi4cxp_kernel.hpp"
#include "cpu/aarch64/kleidiai/kai_types.hpp"

#endif

#endif

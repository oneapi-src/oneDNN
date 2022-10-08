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

#ifndef GPU_OCL_GEMM_OCL_GEMM_ATTRS_H
#define GPU_OCL_GEMM_OCL_GEMM_ATTRS_H

#if WITH_SCALES
#define ATTR_ALPHA alpha[0]
#else
#define ATTR_ALPHA 1.0f
#endif

#if WITH_SRC_ZPOINTS
#define ATTR_A0 ao[0]
#else
#define ATTR_A0 0
#endif

#if WITH_WEI_ZPOINTS
#define ATTR_B0 bo[0]
#else
#define ATTR_B0 0
#endif

#endif // GPU_OCL_GEMM_OCL_GEMM_ATTRS_H

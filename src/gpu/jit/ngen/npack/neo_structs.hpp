/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef NGEN_NPACK_NEO_STRUCTS_H
#define NGEN_NPACK_NEO_STRUCTS_H

#include <cstdint>

/*********************************************************************/
/* NEO binary format definitions, adapted from IGC:                  */
/*      inc/common/igfxfmid.h                                        */
/*      IGC/AdaptorOCL/ocl_igc_shared/executable_format/patch_list.h */
/*********************************************************************/


namespace NGEN_NAMESPACE {
namespace npack {

static constexpr uint32_t MAGIC_CL = 0x494E5443;

enum class GfxCoreFamily : uint32_t {
    Unknown = 0,
    Gen9 = 12,
    Gen10 = 13,
    Gen10LP = 14,
    Gen11 = 15,
    Gen11LP = 16,
    Gen12 = 17,
    Gen12LP = 18,
    XeHP = 0xC05,
    XeHPG = 0xC07,
    XeHPC = 0xC08,
    Xe2 = 0xC09,
};

enum class ProductFamily : uint32_t {
    Unknown = 0,
    SKL = 0x12,
    CNL = 0x1A,
    ICL = 0x1C,
    TGLLP = 0x21,
    DG1 = 1210,
    XE_HP_SDV = 1250,
    DG2 = 1270,
    PVC = 1271,
    MTL = 1272,
    ARL = 1273,
    LNL = 1275,
    LNL_M = 1276,
};

typedef struct
{
    uint32_t Magic; // = MAGIC_CL ("INTC")
    uint32_t Version;
    GfxCoreFamily Device;
    uint32_t GPUPointerSizeInBytes;
    uint32_t NumberOfKernels;
    uint32_t SteppingId;
    uint32_t PatchListSize;
} SProgramBinaryHeader;

typedef struct
{
    uint32_t CheckSum;
    uint32_t ShaderHashCode[2];
    uint32_t KernelNameSize;
    uint32_t PatchListSize;
    uint32_t KernelHeapSize;
    uint32_t GeneralStateHeapSize;
    uint32_t DynamicStateHeapSize;
    uint32_t SurfaceStateHeapSize;
    uint32_t KernelUnpaddedSize;
} SKernelBinaryHeader;

} /* namespace npack */
} /* namespace NGEN_NAMESPACE */

#endif /* header guard */

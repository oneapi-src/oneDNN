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

#ifndef NEO_STRUCTS_H
#define NEO_STRUCTS_H

#include <stdint.h>

/*********************************************************************/
/* NEO binary format definitions, adapted from                       */
/*      IGC/AdaptorOCL/ocl_igc_shared/executable_format/patch_list.h */
/*********************************************************************/

#define MAGIC_CL 0x494E5443

typedef struct
{
    uint32_t Magic; // = MAGIC_CL ("INTC")
    uint32_t Version;
    uint32_t Device;
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

typedef struct
{
    uint32_t Token;
    uint32_t Size;
} SPatchItemHeader;

enum {
    PatchTokenThreadPayload = 22
};

#endif /* header guard */

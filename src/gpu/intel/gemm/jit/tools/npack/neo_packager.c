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

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "elf_structs.h"
#include "hash.h"
#include "neo_packager.h"
#include "neo_structs.h"

packager_status replace_kernel(const char *elf_binary, size_t elf_size,
                               const void *kernel, size_t kernel_size,
                               const void *patches, size_t patches_size,
                               bool no_inline_data,
                               char **new_elf_binary, size_t *new_elf_size)
{
    if (patches == NULL)
        patches_size = 0;
    *new_elf_binary = NULL;
    *new_elf_size = 0;

    // Pad kernel with 0s.
    size_t kernel_padded_size;

    kernel_padded_size = kernel_size + (8 * 8);
    kernel_padded_size = (kernel_padded_size + 0xFF) & ~0xFF;

    // Read ELF
    const SElf64Header *eheader = (const SElf64Header *)elf_binary;

    // Check ELF header
    if (eheader->Magic != ELF_MAGIC)
        return PACKAGER_BAD_ELF;

    // Look for device binary in section table.
    const SElf64SectionHeader *sheader = (const SElf64SectionHeader *)(elf_binary + eheader->SectionHeadersOffset);
    bool found_dev_binary = false;
    int sections_after_binary;

    for (int entry = 0; entry < eheader->NumSectionHeaderEntries; entry++, sheader++) {
        if (sheader->Type == OPENCL_DEV_BINARY_SECTION) {
            found_dev_binary = true;
            sections_after_binary = eheader->NumSectionHeaderEntries - 1 - entry;
            break;
        }
    }

    if (!found_dev_binary || sheader->DataSize < sizeof(SProgramBinaryHeader))
        return PACKAGER_NO_BINARY_SECTION;

    const SProgramBinaryHeader *pheader = (const SProgramBinaryHeader *)(elf_binary + sheader->DataOffset);

    // Check for proper device binary header, with one kernel and no program patches.
    if (pheader->Magic != MAGIC_CL || pheader->NumberOfKernels != 1 || pheader->PatchListSize != 0)
        return PACKAGER_BAD_BINARY_SECTION;

    // Kernel binary header immediately follows.
    const SKernelBinaryHeader *kheader = (const SKernelBinaryHeader *)(pheader + 1);

    // Verify checksum.
    size_t heap_plus_patches = kheader->GeneralStateHeapSize + kheader->DynamicStateHeapSize
        + kheader->SurfaceStateHeapSize + kheader->PatchListSize;
    size_t start_xsum = (const char *)(kheader + 1) - elf_binary;
    size_t end_xsum = start_xsum + kheader->KernelNameSize + kheader->KernelHeapSize + heap_plus_patches;

    if (neo_hash(elf_binary + start_xsum, end_xsum - start_xsum) != kheader->CheckSum)
        return PACKAGER_INVALID_CHECKSUM;

    // Find existing kernel size and allocate memory for new binary.
    ptrdiff_t size_adjust = kernel_padded_size - kheader->KernelHeapSize + patches_size;
    *new_elf_size = elf_size + size_adjust;
    char *new_elf = (char *)malloc(*new_elf_size);

    if (!new_elf)
        return PACKAGER_OUT_OF_MEMORY;

    // Copy ELF up to kernel heap to new ELF.
    size_t before_kernel = start_xsum + kheader->KernelNameSize;
    size_t after_kernel = before_kernel + kheader->KernelHeapSize;
    memmove(new_elf, elf_binary, before_kernel);

    // Copy kernel heap and pad with zeros.
    memmove(new_elf + before_kernel, kernel, kernel_size);
    memset(new_elf + before_kernel + kernel_size, 0, kernel_padded_size - kernel_size);

    // Copy other heaps and patch list.
    size_t after_patches = after_kernel + heap_plus_patches;
    memmove(new_elf + before_kernel + kernel_padded_size, elf_binary + after_kernel, after_patches - after_kernel);

    // Copy extra patches.
    if (patches != NULL && patches_size > 0)
        memmove(new_elf + before_kernel + kernel_padded_size + heap_plus_patches, patches, patches_size);

    // Update kernel header.
    SKernelBinaryHeader *new_kheader = (SKernelBinaryHeader *)(((const char *)kheader - elf_binary) + new_elf);
    size_t new_end_xsum = before_kernel + kernel_padded_size + heap_plus_patches + patches_size;

    new_kheader->KernelHeapSize = (uint32_t) kernel_padded_size;
    new_kheader->KernelUnpaddedSize = (uint32_t) kernel_size;
    new_kheader->PatchListSize += (uint32_t) patches_size;

    // Disable inline data if requested.
    if (no_inline_data) {
        SPatchItemHeader *patch    = (SPatchItemHeader *)(new_elf + new_end_xsum - new_kheader->PatchListSize);
        SPatchItemHeader *patchEnd = (SPatchItemHeader *)(new_elf + new_end_xsum);
        for (; patch < patchEnd && patch->Size > 0; patch = (SPatchItemHeader *)((char *)patch + patch->Size))
            if (patch->Token == PatchTokenThreadPayload && patch->Size == 88)
                *((uint32_t *) patch + 16) = 0;
    }

    // Update checksum.
    new_kheader->CheckSum = neo_hash(new_elf + start_xsum, new_end_xsum - start_xsum);

    // Copy remainder of ELF.
    memmove(new_elf + new_end_xsum, elf_binary + end_xsum, elf_size - end_xsum);

    // Update ELF section header, and all following headers.
    SElf64SectionHeader *new_sheader = (SElf64SectionHeader *)(((const char *)sheader - elf_binary) + new_elf);
    new_sheader->DataSize += size_adjust;

    for (new_sheader++; sections_after_binary > 0; sections_after_binary--, new_sheader++)
        new_sheader->DataOffset += size_adjust;

    *new_elf_binary = new_elf;

    // Look for a debug section, and remove it if found. Fix up string table index as well.
    SElf64Header *new_eheader = (SElf64Header *) new_elf;
    new_sheader = (SElf64SectionHeader *)(new_elf + eheader->SectionHeadersOffset);

    for (int entry = 0; entry < new_eheader->NumSectionHeaderEntries; entry++, new_sheader++)
        if (new_sheader->Type == OPENCL_DEV_DEBUG_SECTION)
            new_sheader->Type = STRING_TABLE_SECTION;

    return PACKAGER_SUCCESS;
}

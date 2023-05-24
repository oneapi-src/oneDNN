/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef NGEN_NPACK_NEO_PACKAGER_HPP
#define NGEN_NPACK_NEO_PACKAGER_HPP

#include <cstring>
#include <vector>

#include "elf_structs.hpp"
#include "neo_structs.hpp"
#include "hash.hpp"
#include "../ngen_utils.hpp"

namespace ngen {
namespace npack {

class bad_elf : public std::runtime_error {
public:
    bad_elf() : std::runtime_error("Incompatible OpenCL runtime: program is not in expected ELF format.") {}
};
class no_binary_section : public std::runtime_error {
public:
    no_binary_section() : std::runtime_error("Incompatible OpenCL runtime: no binary section found.") {}
};
class bad_binary_section : public std::runtime_error {
public:
    bad_binary_section() : std::runtime_error("Incompatible OpenCL runtime: invalid binary section.") {}
};
class invalid_checksum : public std::runtime_error {
public:
    invalid_checksum() : std::runtime_error("Incompatible OpenCL runtime: invalid checksum.") {}
};

inline void findDeviceBinary(const std::vector<uint8_t> &binary, const SElf64SectionHeader **sheaderOut,
                             const SProgramBinaryHeader **pheaderOut, int *sectionsAfterBinaryOut)
{
    auto elf_binary = binary.data();
    auto elf_size = binary.size();

    // Read ELF
    auto *eheader = (const SElf64Header *)elf_binary;

    // Check ELF header
    if (eheader->Magic != ELF_MAGIC)
        throw bad_elf();

    // Look for device binary in section table.
    auto sheader = (const SElf64SectionHeader *)(elf_binary + eheader->SectionHeadersOffset);
    bool found_dev_binary = false;
    int sections_after_binary;

    if ((const uint8_t *) (sheader + eheader->NumSectionHeaderEntries) > (elf_binary + elf_size))
        throw bad_elf();

    for (int entry = 0; entry < eheader->NumSectionHeaderEntries; entry++, sheader++) {
        if (sheader->Type == OPENCL_DEV_BINARY_TYPE) {
            found_dev_binary = true;
            sections_after_binary = eheader->NumSectionHeaderEntries - 1 - entry;
            break;
        }
    }

    if (!found_dev_binary || sheader->DataSize < sizeof(SProgramBinaryHeader))
        throw no_binary_section();

    auto pheader = (const SProgramBinaryHeader *)(elf_binary + sheader->DataOffset);

    // Check for proper device binary header, with one kernel and no program patches.
    if (pheader->Magic != MAGIC_CL || pheader->NumberOfKernels != 1 || pheader->PatchListSize != 0)
        throw bad_binary_section();

    if (sheaderOut != nullptr) *sheaderOut = sheader;
    if (pheaderOut != nullptr) *pheaderOut = pheader;
    if (sectionsAfterBinaryOut != nullptr) *sectionsAfterBinaryOut = sections_after_binary;
}

inline void replaceKernel(std::vector<uint8_t> &binary, const std::vector<uint8_t> &kernel, const std::vector<uint8_t> &patches = std::vector<uint8_t>())
{
    auto elf_binary = binary.data();
    auto elf_size = binary.size();
    auto kernel_size = kernel.size();
    auto patches_size = patches.size();

    // Pad kernel with 0s.
    size_t kernel_padded_size;

    kernel_padded_size = kernel.size() + (8 * 8);
    kernel_padded_size = (kernel_padded_size + 0xFF) & ~0xFF;

    // Read and validate ELF; find device binary section.
    int sections_after_binary;
    const SElf64SectionHeader *sheader;
    const SProgramBinaryHeader *pheader;

    findDeviceBinary(binary, &sheader, &pheader, &sections_after_binary);

    // Kernel binary header immediately follows.
    auto kheader = (const SKernelBinaryHeader *)(pheader + 1);

    // Verify checksum.
    size_t heap_plus_patches = kheader->GeneralStateHeapSize + kheader->DynamicStateHeapSize
        + kheader->SurfaceStateHeapSize + kheader->PatchListSize;
    size_t start_xsum = (const unsigned char *)(kheader + 1) - elf_binary;
    size_t end_xsum = start_xsum + kheader->KernelNameSize + kheader->KernelHeapSize + heap_plus_patches;

    if (end_xsum > elf_size)
        throw bad_binary_section();
    if (neo_hash(elf_binary + start_xsum, end_xsum - start_xsum) != kheader->CheckSum)
        throw invalid_checksum();

    // Find existing kernel size and allocate memory for new binary.
    ptrdiff_t size_adjust = kernel_padded_size - kheader->KernelHeapSize + patches_size;
    auto new_elf_size = elf_size + size_adjust;
    std::vector<uint8_t> new_binary(new_elf_size);
    auto new_elf = new_binary.data();

    // Copy ELF up to kernel heap to new ELF.
    size_t before_kernel = start_xsum + kheader->KernelNameSize;
    size_t after_kernel = before_kernel + kheader->KernelHeapSize;
    utils::copy_into(new_binary, 0, binary, 0, before_kernel);

    // Copy kernel heap and pad with zeros.
    utils::copy_into(new_binary, before_kernel, kernel, 0, kernel_size);
    memset(new_elf + before_kernel + kernel_size, 0, kernel_padded_size - kernel_size);

    // Copy other heaps and patch list.
    size_t after_patches = after_kernel + heap_plus_patches;
    utils::copy_into(new_binary, before_kernel + kernel_padded_size, binary, after_kernel, after_patches - after_kernel);

    // Copy extra patches.
    utils::copy_into(new_binary, before_kernel + kernel_padded_size + heap_plus_patches, patches, 0, patches_size);

    // Update kernel header.
    auto new_kheader = (SKernelBinaryHeader *)(((const unsigned char *)kheader - elf_binary) + new_elf);
    size_t new_end_xsum = before_kernel + kernel_padded_size + heap_plus_patches + patches_size;

    new_kheader->CheckSum = neo_hash(new_elf + start_xsum, new_end_xsum - start_xsum);
    new_kheader->KernelHeapSize = uint32_t(kernel_padded_size);
    new_kheader->KernelUnpaddedSize = uint32_t(kernel_size);
    new_kheader->PatchListSize += uint32_t(patches_size);

    // Copy remainder of ELF.
    utils::copy_into(new_binary, new_end_xsum, binary, end_xsum, elf_size - end_xsum);

    // Update ELF section header, and all following headers.
    auto new_sheader = (SElf64SectionHeader *)(((const unsigned char *)sheader - elf_binary) + new_elf);
    new_sheader->DataSize += size_adjust;

    for (new_sheader++; sections_after_binary > 0; sections_after_binary--, new_sheader++)
        new_sheader->DataOffset += size_adjust;

    // Update binary.
    std::swap(new_binary, binary);
}

inline HW decodeGfxCoreFamily(GfxCoreFamily family)
{
    switch (family) {
        case GfxCoreFamily::Gen9:     return HW::Gen9;
        case GfxCoreFamily::Gen10:    return HW::Gen10;
        case GfxCoreFamily::Gen10LP:  return HW::Gen10;
        case GfxCoreFamily::Gen11:    return HW::Gen11;
        case GfxCoreFamily::Gen11LP:  return HW::Gen11;
        case GfxCoreFamily::Gen12LP:  return HW::Gen12LP;
        case GfxCoreFamily::Gen12:
        case GfxCoreFamily::XeHP:     return HW::XeHP;
        case GfxCoreFamily::XeHPG:    return HW::XeHPG;
        case GfxCoreFamily::XeHPC:    return HW::XeHPC;
        default:                      return HW::Unknown;
    }
}

inline GfxCoreFamily encodeGfxCoreFamily(HW hw)
{
    switch (hw) {
        case HW::Gen9:    return GfxCoreFamily::Gen9;
        case HW::Gen10:   return GfxCoreFamily::Gen10;
        case HW::Gen11:   return GfxCoreFamily::Gen11LP;
        case HW::Gen12LP: return GfxCoreFamily::Gen12LP;
        case HW::XeHP:    return GfxCoreFamily::XeHP;
        case HW::XeHPG:   return GfxCoreFamily::XeHPG;
        case HW::XeHPC:   return GfxCoreFamily::XeHPC;
        default:          return GfxCoreFamily::Unknown;
    }
}

inline ngen::ProductFamily decodeProductFamily(ProductFamily family)
{
    if (family >= ProductFamily::SKL && family < ProductFamily::CNL) return ngen::ProductFamily::GenericGen9;
    if (family >= ProductFamily::CNL && family < ProductFamily::ICL) return ngen::ProductFamily::GenericGen10;
    if (family >= ProductFamily::ICL && family < ProductFamily::TGLLP) return ngen::ProductFamily::GenericGen11;
    if (family >= ProductFamily::TGLLP && family <= ProductFamily::DG1) return ngen::ProductFamily::GenericGen12LP;
    if (family == ProductFamily::XE_HP_SDV) return ngen::ProductFamily::GenericXeHP;
    if (family == ProductFamily::DG2) return ngen::ProductFamily::DG2;
    if (family == ProductFamily::MTL) return ngen::ProductFamily::MTL;
    if (family == ProductFamily::PVC) return ngen::ProductFamily::PVC;
    return ngen::ProductFamily::Unknown;
}

inline void getBinaryHWInfo(const std::vector<uint8_t> &binary, HW &outHW, Product &outProduct)
{
    const SProgramBinaryHeader *pheader = nullptr;

    findDeviceBinary(binary, nullptr, &pheader, nullptr);
    outHW = decodeGfxCoreFamily(pheader->Device);
    outProduct.family = ngen::ProductFamily::Unknown;
    outProduct.stepping = pheader->SteppingId;

    // XeHPG identifies with older runtimes as XeHP. Check whether EOT goes to TS (XeHP) or gateway (XeHPG).
    using b14 = std::array<uint8_t, 14>;
    b14 gtwyEOT{{3, 0x80, 4, 0, 0, 0, 0xC, 0x7F, 0x20, 0x30, 0, 0, 0, 0}};
    if (outHW == HW::XeHP) for (size_t i = 0; i < binary.size() - 0x10; i++) {
        if (binary[i] == 0x31 && *(b14 *)(binary.data() + i + 2) == gtwyEOT) {
            outHW = HW::XeHPG;
            break;
        }
    }
}

inline ngen::Product decodeHWIPVersion(uint32_t rawVersion)
{
    struct HWIPVersion {
        union {
            uint32_t raw;
            struct{
                uint32_t revision : 6;
                uint32_t reserved : 8;
                uint32_t release : 8;
                uint32_t architecture : 10;
            };
        };
    } version;

    ngen::Product outProduct = {ngen::ProductFamily::Unknown, 0};

    version.raw = rawVersion;
    switch (version.architecture) {
        case 9:  outProduct.family = ngen::ProductFamily::GenericGen9; break;
        case 11: outProduct.family = ngen::ProductFamily::GenericGen11; break;
        case 12:
            if (version.release <= 10)
                outProduct.family = ngen::ProductFamily::GenericGen12LP;
            else if (version.release == 50)
                outProduct.family = ngen::ProductFamily::GenericXeHP;
            else if (version.release > 50 && version.release <= 59)
                outProduct.family = ngen::ProductFamily::DG2;
            else if (version.release == 60)
                outProduct.family = ngen::ProductFamily::PVC;
            else if (version.release >= 70 && version.release <= 71)
                outProduct.family = ngen::ProductFamily::MTL;
            break;
        default: outProduct.family = ngen::ProductFamily::Unknown; break;
    }

    if (outProduct.family != ngen::ProductFamily::Unknown)
        outProduct.stepping = version.revision;

    return outProduct;
}

} /* namespace npack */
} /* namespace ngen */

#endif /* header guard */

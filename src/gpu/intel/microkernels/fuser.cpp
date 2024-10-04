/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "elf.hpp"
#include "fuser.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace micro {

static void fixupJumpTargets(uint8_t *start, size_t len, ptrdiff_t adjust);

void fuseMicrokernel(std::vector<uint8_t> &binary,
        const std::vector<uint8_t> &microkernel, long id) {
    auto base = binary.data();
    auto bytes = binary.size();

    auto fheaderPtr = reinterpret_cast<FileHeader *>(base);

    bool ok = bytes >= sizeof(fheaderPtr) && fheaderPtr->magic == ELFMagic
            && fheaderPtr->elfClass == ELFClass64
            && fheaderPtr->endian == ELFLittleEndian
            && fheaderPtr->sectionHeaderSize == sizeof(SectionHeader)
            && (fheaderPtr->version == 0 || fheaderPtr->version == ELFVersion1)
            && (fheaderPtr->type == ZebinExec
                    || fheaderPtr->type == ELFRelocatable)
            && bytes >= sizeof(fheaderPtr)
                            + sizeof(SectionHeader) * fheaderPtr->sectionCount;

    if (!ok)
        throw std::runtime_error(
                "IGC did not generate a valid zebin program binary");

    bool foundZeInfo = false;
    SectionHeader *text = nullptr;
    const char *snames = nullptr;
    int textSectionID = -1, relSectionID = -1;

    auto *sheaders = reinterpret_cast<SectionHeader *>(
            base + fheaderPtr->sectionTableOff);

    snames = reinterpret_cast<char *>(
            base + sheaders[fheaderPtr->strTableIndex].offset);

    for (int s = 0; s < fheaderPtr->sectionCount; s++) {
        switch (sheaders[s].type) {
            case SectionHeader::Type::ZeInfo: foundZeInfo = true; break;
            case SectionHeader::Type::Program: {
                if (snames) {
                    std::string sname(snames + sheaders[s].name);
                    if (sname == ".text.Intel_Symbol_Table_Void_Program")
                        continue;
                    if (sname.substr(0, 6) != ".text.") continue;
                }
                if (text)
                    throw std::runtime_error("Multiple kernels in program");
                text = sheaders + s;
                textSectionID = s;
                break;
            }
            default: break;
        }
    }

    if (!foundZeInfo || !text || text->offset + text->size > bytes)
        throw std::runtime_error(
                "IGC did not generate a valid zebin program binary");

    std::string rname = ".rel";
    rname += (snames + text->name);
    for (int s = 0; s < fheaderPtr->sectionCount; s++) {
        if (sheaders[s].type != SectionHeader::Type::Relocation) continue;
        if (rname != (snames + sheaders[s].name)) continue;
        if (relSectionID >= 0)
            throw std::runtime_error("Multiple relocation sections for kernel");
        relSectionID = s;
    }

    auto *insn = reinterpret_cast<const uint32_t *>(base + text->offset);
    auto *iend = reinterpret_cast<const uint32_t *>(
            base + text->offset + text->size);

    const uint8_t *spliceStart = nullptr;
    const uint8_t *spliceEnd = nullptr;

    for (; insn < iend; insn += 4) {
        if (insn[0] & (1u << 29))
            insn -= 2;
        else if (insn[3] == (sigilStart ^ id))
            spliceStart = reinterpret_cast<const uint8_t *>(insn);
        else if (insn[3] == (sigilEnd ^ id)) {
            spliceEnd = reinterpret_cast<const uint8_t *>(insn);
            break;
        }
    }

    if (!spliceStart || !spliceEnd) return;

    auto removeBytes = spliceEnd - spliceStart + 16;

    size_t before = spliceStart - base;
    auto after = bytes - before - removeBytes;
    ptrdiff_t sizeAdjust = microkernel.size() - removeBytes;

    auto kbefore = before - text->offset;
    auto kafter = text->size - kbefore - removeBytes;

    std::vector<uint8_t> newBinary(bytes + sizeAdjust);
    auto newBase = newBinary.data();

    memmove(newBase, base, before);
    memmove(newBase + before, microkernel.data(), microkernel.size());
    memmove(newBase + before + microkernel.size(), spliceStart + removeBytes,
            after);

    fixupJumpTargets(newBase + text->offset, kbefore, +sizeAdjust);
    fixupJumpTargets(
            newBase + before + microkernel.size(), kafter, -sizeAdjust);

    fheaderPtr = reinterpret_cast<FileHeader *>(newBase);

    if (fheaderPtr->sectionTableOff > before)
        fheaderPtr->sectionTableOff += sizeAdjust;

    sheaders = reinterpret_cast<SectionHeader *>(
            newBase + fheaderPtr->sectionTableOff);
    sheaders[textSectionID].size += sizeAdjust;
    for (int s = 0; s < fheaderPtr->sectionCount; s++)
        if (sheaders[s].offset > before) sheaders[s].offset += sizeAdjust;

    if (relSectionID >= 0) {
        auto relSection = sheaders + relSectionID;
        auto rel = reinterpret_cast<Relocation *>(newBase + relSection->offset);
        auto relEnd = reinterpret_cast<Relocation *>(
                newBase + relSection->offset + relSection->size);
        for (; rel < relEnd; rel++) {
            if (rel->offset >= kbefore) rel->offset += sizeAdjust;
        }
    }

#ifdef SPLICE_DEBUG
    std::ofstream dump0("original.bin");
    dump0.write((const char *)binary.data(), binary.size());

    std::ofstream dump("patched.bin");
    dump.write((const char *)newBinary.data(), newBinary.size());
#endif

    std::swap(binary, newBinary);

    // Tail-recurse to handle any further instances of this microkernel
    fuseMicrokernel(binary, microkernel, id);
}

void fuseMicrokernels(std::vector<uint8_t> &binary, const char *source) {
    std::vector<uint8_t> microkernel;
    const auto sigilLen = strlen(sigilBinary);

    auto toNybble = [](char c) {
        return ((c >= 'A') ? (c - 'A' + 10) : (c - '0')) & 0xF;
    };

    for (const char *s = std::strstr(source, sigilBinary); s;
            s = std::strstr(s, sigilBinary)) {
        s += sigilLen;
        char *after;
        long id = strtol(s, &after, 10);
        microkernel.clear();
        for (s = after + 1; *s != '\n'; s += 2) {
            if (!s[0] || !s[1]) break;
            microkernel.push_back(static_cast<uint8_t>(
                    (toNybble(s[0]) << 4) | toNybble(s[1])));
        }
        fuseMicrokernel(binary, microkernel, id);
    }
}

static void fixupJumpTargets(uint8_t *start, size_t len, ptrdiff_t adjust) {
    auto istart = reinterpret_cast<int32_t *>(start);
    auto iend = reinterpret_cast<int32_t *>(start + len);

    for (auto insn = istart; insn < iend; insn += 4) {
        if (insn[0] & (1u << 29)) {
            insn -= 2; /* skip compacted instructions */
            continue;
        }
        uint8_t op = insn[0] & 0xFF;
        if ((op & 0xF0) != 0x20) continue; /* skip non-jumps */
        if (op == 0x2B || op == 0x2D) continue; /* skip ret/calla */
        bool hasUIP = (op == 0x22 || op == 0x23 || op == 0x24 || op == 0x28
                || op == 0x2A || op == 0x2E);

        auto jumpFixup = [=](int32_t &ip) {
            auto target = ((insn - istart) << 2) + ip;
            if (target < 0 || target >= ptrdiff_t(len)) ip += adjust;
        };

        if (hasUIP) jumpFixup(insn[2]);
        jumpFixup(insn[3]);
    }
}

bool hasMicrokernels(const char *source) {
    return std::strstr(source, sigilBinary);
}

} /* namespace micro */
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

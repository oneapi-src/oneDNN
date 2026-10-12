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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "generator/microkernel/elf.hpp"
#include "generator/microkernel/payload.hpp"
#include "gemmstone/microkernel/fuser.hpp"
#include "ngen_elf.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

static void fixupJumpTargets(uint8_t *start, size_t len, ptrdiff_t adjust);

/* Host payloads, checked against the microkernel's register placement. */
struct PayloadCheck {
    const std::vector<KernelInfo> *kernels;
    uint32_t argumentBase;
    int grfBytes;
    bool *validated;
};

static void checkPayload(const PayloadCheck &check, const char *kernelName);

/* Compacted no-op, used to keep microkernel instructions 16-byte aligned.
   Encodings are per-arch compaction table entries; verify additions with
   iga64 (sync.nop {Compacted} / mov (1|M0) null:ud r0.0:ud {Compacted}). */
static const uint8_t *alignmentFiller(ngen::HW hw) {
    static const uint8_t syncNop[8] = {0x01, 0, 0, 0xE8, 0x01, 0, 0x11, 0};
    static const uint8_t movNull[8] = {0x61, 0, 0x84, 0xBC, 0, 0, 0, 0};
    static const uint8_t movNullXe2[8] = {0x61, 0, 0x84, 0xFC, 0, 0, 0x10, 0};
    static const uint8_t movNullXe3p[8] = {0x61, 0, 0x84, 0xA4, 0, 0, 0, 0};
    switch (hw) {
        case ngen::HW::Unknown:
        case ngen::HW::Gen9:
        case ngen::HW::Gen10:
        case ngen::HW::Gen11:
        case ngen::HW::XeLP:
        case ngen::HW::XeHP: return nullptr;
        case ngen::HW::XeHPG: return syncNop;
        case ngen::HW::XeHPC: return movNull;
        case ngen::HW::Xe2:
        case ngen::HW::Xe3: return movNullXe2;
        case ngen::HW::Xe3p: return movNullXe3p;
    }
    return nullptr;
}

static void fuse(std::vector<uint8_t> &binary,
        const std::vector<uint8_t> &microkernel, long id,
        const PayloadCheck *check, const uint8_t *filler) {
    auto base = binary.data();
    auto bytes = binary.size();

    auto fheader = fileHeader(binary);

    bool foundZeInfo = false;
    std::vector<std::pair<SectionHeader, int>> textSections;

    auto sheaders = sectionHeaders(binary);

    for (int s = 0; s < fheader.sectionCount; s++) {
        switch (sheaders[s].type) {
            case SectionHeader::Type::ZeInfo: foundZeInfo = true; break;
            case SectionHeader::Type::Program: {
                std::string sname = sectionName(binary, s);
                if (sname == ".text.Intel_Symbol_Table_Void_Program")
                    continue;
                if (sname.substr(0, 6) != ".text.") continue;
                textSections.emplace_back(sheaders[s], s);
                break;
            }
            default: break;
        }
    }

    if (!foundZeInfo || textSections.empty())
        throw std::runtime_error(
                "IGC did not generate a valid zebin program binary");

    for (auto &entry : textSections) {
        auto text = entry.first;
        int textSectionID = entry.second;
        if (!sectionInBounds(binary, text)) continue;

        auto *insn = reinterpret_cast<const uint32_t *>(base + text.offset);
        auto *iend = reinterpret_cast<const uint32_t *>(
                base + text.offset + text.size);

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

        if (!spliceStart || !spliceEnd) continue;

        // Validate the argument placement against the selected base.
        if (check)
            checkPayload(*check,
                    sectionName(binary, textSectionID) + 6);

        int relSectionID = -1;
        std::string rname = ".rel";
        rname += sectionName(binary, textSectionID);
        for (int s = 0; s < fheader.sectionCount; s++) {
            if (sheaders[s].type != SectionHeader::Type::Relocation) continue;
            if (rname != sectionName(binary, s)) continue;
            if (relSectionID >= 0)
                throw std::runtime_error(
                        "Multiple relocation sections for kernel");
            relSectionID = s;
        }

        auto removeBytes = spliceEnd - spliceStart + 16;

        size_t before = spliceStart - base;
        auto after = bytes - before - removeBytes;

        auto kbefore = before - text.offset;

        /* Keep the microkernel's (uncompacted) instructions 16-byte aligned
           so they do not straddle instruction cache lines. */
        std::vector<uint8_t> blob;
        if (filler && (kbefore & 8)) blob.assign(filler, filler + 8);
        blob.insert(blob.end(), microkernel.begin(), microkernel.end());

        ptrdiff_t sizeAdjust = blob.size() - removeBytes;

        auto kafter = text.size - kbefore - removeBytes;

        std::vector<uint8_t> newBinary(bytes + sizeAdjust);
        auto newBase = newBinary.data();

        memmove(newBase, base, before);
        memmove(newBase + before, blob.data(), blob.size());
        memmove(newBase + before + blob.size(),
                spliceStart + removeBytes, after);

        fixupJumpTargets(newBase + text.offset, kbefore, +sizeAdjust);
        fixupJumpTargets(
                newBase + before + blob.size(), kafter, -sizeAdjust);

        if (fheader.sectionTableOff > before) {
            fheader.sectionTableOff += sizeAdjust;
            storeFileHeader(newBinary, fheader);
        }

        sheaders = sectionHeaders(newBinary);
        sheaders[textSectionID].size += sizeAdjust;
        for (int s = 0; s < fheader.sectionCount; s++)
            if (sheaders[s].offset > before) sheaders[s].offset += sizeAdjust;
        storeSectionHeaders(newBinary, sheaders);

        if (relSectionID >= 0) {
            auto relocations
                    = sectionEntries<Relocation>(newBinary, relSectionID);
            for (auto &relocation : relocations)
                if (relocation.offset >= kbefore) relocation.offset += sizeAdjust;
            storeSectionEntries(newBinary, relSectionID, relocations);
        }

#ifdef SPLICE_DEBUG
        std::ofstream dump0("original." + std::to_string(id) + ".bin");
        dump0.write((const char *)binary.data(), binary.size());

        std::ofstream dump("patched." + std::to_string(id) + ".bin");
        dump.write((const char *)newBinary.data(), newBinary.size());
#endif

        std::swap(binary, newBinary);

        // Tail-recurse to handle any further instances of this microkernel
        fuse(binary, microkernel, id, check, filler);
        return;
    }
}

// Drop the zebin SPIR-V section (ZebinSpirv -> Null) so the runtime can't
// rebuild the program from stale IR and lose the spliced-in microkernels.
static void stripIntermediateRepresentation(std::vector<uint8_t> &binary) {
    if (!isZebin(binary)) return;
    auto fheader = fileHeader(binary);
    auto sheaders = sectionHeaders(binary);
    for (int s = 0; s < fheader.sectionCount; s++)
        if (sheaders[s].type == SectionHeader::ZebinSpirv)
            sheaders[s].type = SectionHeader::Null;
    storeSectionHeaders(binary, sheaders);
}

static bool findZeInfo(
        const std::vector<uint8_t> &binary, const char *&text, size_t &length) {
    if (!isZebin(binary)) return false;
    auto fheader = fileHeader(binary);

    auto sheaders = sectionHeaders(binary);
    for (int s = 0; s < fheader.sectionCount; s++) {
        if (sheaders[s].type != SectionHeader::Type::ZeInfo) continue;
        if (!sectionInBounds(binary, sheaders[s])) return false;
        text = reinterpret_cast<const char *>(
                binary.data() + sheaders[s].offset);
        length = size_t(sheaders[s].size);
        return true;
    }
    return false;
}

// vISA does not reserve registers against IGC's thread payload, so a microkernel
// placed too low silently corrupts the host kernel's arguments.
static void checkPayload(const PayloadCheck &check, const char *kernelName) {
    for (auto &kernel : *check.kernels) {
        if (kernel.name != kernelName) continue;
        *check.validated = true;

        auto payloadEnd = payloadEndBytes(kernel, check.grfBytes);
        if (payloadEnd <= check.argumentBase) return;

        auto argumentBytes = payloadEnd - crossthreadBase(kernel, check.grfBytes);
        throw std::runtime_error("Microkernel registers (r"
                + std::to_string(check.argumentBase / check.grfBytes)
                + " and up) overlap the thread payload of kernel "
                + kernel.name + ", which ends at r"
                + std::to_string((payloadEnd - 1) / check.grfBytes)
                + ". Raise HostPayload::argumentBytes to "
                + std::to_string(argumentBytes) + " or more");
    }
}

bool fuse(std::vector<uint8_t> &binary, const char *source, int grfBytes) {
    std::vector<uint8_t> microkernel;
    const auto sigilLen = strlen(sigilBinary);

    auto filler = alignmentFiller(
            ngen::ELFCodeGenerator<ngen::HW::Unknown>::getBinaryArch(binary));

    auto toNybble = [](char c) {
        return ((c >= 'A') ? (c - 'A' + 10) : (c - '0')) & 0xF;
    };

    /* Read the payload layout before fusing moves the .ze_info section. */
    std::vector<KernelInfo> kernels;
    const char *zeInfo = nullptr;
    size_t zeInfoLength = 0;
    bool haveLayout = grfBytes > 0 && findZeInfo(binary, zeInfo, zeInfoLength)
            && parseZeInfo(zeInfo, zeInfoLength, kernels);
    bool validated = false;

    for (const char *s = std::strstr(source, sigilBinary); s;
            s = std::strstr(s, sigilBinary)) {
        s += sigilLen;
        char *after;
        long id = strtol(s, &after, 10);
        uint32_t argumentBase = 0;
        if (*after == ':')
            argumentBase = uint32_t(strtoul(after + 1, &after, 10));
        microkernel.clear();
        for (s = after + 1; *s != '\n'; s += 2) {
            if (!s[0] || !s[1]) break;
            microkernel.push_back(static_cast<uint8_t>(
                    (toNybble(s[0]) << 4) | toNybble(s[1])));
        }
        PayloadCheck check {&kernels, argumentBase, grfBytes, &validated};
        fuse(binary, microkernel, id,
                (haveLayout && argumentBase > 0) ? &check : nullptr, filler);
    }
    stripIntermediateRepresentation(binary);
    return validated;
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
            if (target < 0 || target >= ptrdiff_t(len))
                ip += static_cast<int32_t>(adjust);
        };

        if (hasUIP) jumpFixup(insn[2]);
        jumpFixup(insn[3]);
    }
}

bool hasMicrokernels(const char *source) {
    return std::strstr(source, sigilBinary);
}

}
GEMMSTONE_NAMESPACE_END

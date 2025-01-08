/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef NGEN_ELF_HPP
#define NGEN_ELF_HPP

#include "ngen.hpp"
#include "ngen_interface.hpp"

#include "npack/neo_packager.hpp"

namespace NGEN_NAMESPACE {

// ELF binary format generator class.
template <HW hw>
class ELFCodeGenerator : public BinaryCodeGenerator<hw>
{
public:
    inline std::vector<uint8_t> getBinary();
    static inline HW getBinaryArch(const std::vector<uint8_t> &binary);
    static inline void getBinaryHWInfo(const std::vector<uint8_t> &binary, HW &outHW, Product &outProduct);

    explicit ELFCodeGenerator(Product product_, DebugConfig debugConfig = {})  : BinaryCodeGenerator<hw>(product_, debugConfig) {}
    explicit ELFCodeGenerator(int stepping_ = 0, DebugConfig debugConfig = {}) : BinaryCodeGenerator<hw>(stepping_, debugConfig) {}

protected:
    NEOInterfaceHandler interface_{hw};

    void externalName(const std::string &name)                           { interface_.externalName(name); }

    const std::string &getExternalName() const                           { return interface_.getExternalName(); }
    int getSIMD() const                                                  { return interface_.getSIMD(); }
    int getGRFCount() const                                              { return interface_.getGRFCount(); }
    size_t getSLMSize() const                                            { return interface_.getSLMSize(); }

    void require32BitBuffers()                                           { interface_.require32BitBuffers(); }
    void requireArbitrationMode(ThreadArbitrationMode mode)              { interface_.requireArbitrationMode(mode); }
    void requireBarrier()                                                { interface_.requireBarrier(); }
    void requireBarriers(int nbarriers)                                  { interface_.requireBarriers(nbarriers); }
    void requireDPAS()                                                   { interface_.requireDPAS(); }
    void requireGlobalAtomics()                                          { interface_.requireGlobalAtomics(); }
    void requireGRF(int grfs)                                            { BinaryCodeGenerator<hw>::requireGRF(grfs); interface_.requireGRF(grfs); }
    void requireLocalID(int dimensions)                                  { interface_.requireLocalID(dimensions); }
    void requireLocalSize()                                              { interface_.requireLocalSize(); }
    void requireNonuniformWGs()                                          { interface_.requireNonuniformWGs(); }
    void requireNoPreemption()                                           { interface_.requireNoPreemption(); }
    void requireScratch(size_t bytes = 1)                                { interface_.requireScratch(bytes); }
    void requireSIMD(int simd_)                                          { interface_.requireSIMD(simd_); }
    void requireSLM(size_t bytes)                                        { interface_.requireSLM(bytes); }
    void requireStatelessWrites(bool req = true)                         { interface_.requireStatelessWrites(req); }
    inline void requireType(DataType type)                               { interface_.requireType(type); }
    template <typename T> void requireType()                             { interface_.requireType<T>(); }
    void requireWalkOrder(int o1, int o2)                                { interface_.requireWalkOrder(o1, o2); }
    void requireWalkOrder(int o1, int o2, int o3)                        { interface_.requireWalkOrder(o1, o2, o3); }
    void requireWorkgroup(size_t x, size_t y = 1, size_t z = 1)          { interface_.requireWorkgroup(x, y, z); }

    void finalizeInterface()                                             { interface_.finalize(); }

    template <typename DT>
    void newArgument(const std::string &name)                            { interface_.newArgument<DT>(name); }
    void newArgument(const std::string &name, DataType type,
                     ExternalArgumentType exttype = ExternalArgumentType::Scalar,
                     GlobalAccessType access = GlobalAccessType::Default)
    {
        interface_.newArgument(name, type, exttype, access);
    }
    void newArgument(const std::string &name, Subregister reg,
                     ExternalArgumentType exttype = ExternalArgumentType::Scalar,
                     GlobalAccessType access = GlobalAccessType::Default)
    {
        interface_.newArgument(name, reg, exttype, access);
    }
    void newArgument(const std::string &name, ExternalArgumentType exttype,
                     GlobalAccessType access = GlobalAccessType::Default)
    {
        interface_.newArgument(name, exttype, access);
    }

    void allowArgumentRearrangement(bool allow)                          { return interface_.allowArgumentRearrangement(allow); }

    Subregister getArgument(const std::string &name) const               { return interface_.getArgument(name); }
    Subregister getArgumentIfExists(const std::string &name) const       { return interface_.getArgumentIfExists(name); }
    int getArgumentSurface(const std::string &name) const                { return interface_.getArgumentSurface(name); }
    int getArgumentSurfaceIfExists(const std::string &name) const        { return interface_.getArgumentSurfaceIfExists(name); }
    GRF getLocalID(int dim) const                                        { return interface_.getLocalID(dim); }
    Subregister getSIMD1LocalID(int dim) const                           { return interface_.getSIMD1LocalID(dim); }
    Subregister getLocalSize(int dim) const                              { return interface_.getLocalSize(dim); }

    void prologue()                                                      { interface_.generatePrologue(*this); }
    void epilogue(RegData r0_info = RegData())
    {
        if (r0_info.isInvalid()) r0_info = this->r0;
        int GRFCount = interface_.getGRFCount();
        bool hasSLM = (interface_.getSLMSize() > 0);
        BinaryCodeGenerator<hw>::epilogue(GRFCount, hasSLM, r0_info);
    }

    inline std::vector<uint8_t> getBinary(const std::vector<uint8_t> &code);

private:
    using BinaryCodeGenerator<hw>::labelManager;
    using BinaryCodeGenerator<hw>::rootStream;

    struct ZebinELF {
        enum {
            ELFMagic = 0x464C457F,             // '\x7FELF'
            ELFClass64 = 2,
            ELFLittleEndian = 1,
            ELFVersion1 = 1,
            ELFRelocatable = 1,
        };
        enum {
            MachineIntelGT = 205,
            ZebinExec = 0xFF12
        };

        enum DWARF_UT : uint8_t {
            COMPILE = 0x01,
        };
        enum DWARF_TAG : uint8_t {
            COMPILATION_UNIT = 0x11,
            SUBPROGRAM = 0x2e,
        };
        enum DWARF_AT : uint8_t {
            NAME = 0x03,
            STMT_LIST = 0x10,
            LOW_PC = 0x11,
            HIGH_PC = 0x12,
            DECL_COLUMN = 0x39,
            DECL_FILE = 0x3a,
            DECL_LINE = 0x3b,
        };
        enum DWARF_FORM : uint8_t {
            ADDR = 0x1,
            DATA2 = 0x05,
            DATA4 = 0x06,
            DATA8 = 0x07,
            STRING = 0x08,
            DATA1 = 0x0b,
            STRP = 0x0e,
            LINEPTR = 0x17,
            FLAG_PRESENT = 0x19,
            LINE_STRP = 0x1f,
        };
        enum DWARF_LNCT : uint8_t {
            PATH = 0x1,
            DIRECTORY_INDEX = 0x2,
        };

        union TargetMetadata {
            uint32_t all;
            struct {
                unsigned genFlags : 8;
                unsigned minHWRevision : 5;
                unsigned validateRevision : 1;
                unsigned disableExtValidation : 1;
                unsigned useGfxCoreFamily : 1;
                unsigned maxHWRevision : 5;
                unsigned generator : 3;
                unsigned reserved : 8;
            } parts;
        };
        struct FileHeader {
            uint32_t magic = ELFMagic;
            uint8_t elfClass = ELFClass64;
            uint8_t endian = ELFLittleEndian;
            uint8_t version = ELFVersion1;
            uint8_t osABI = 0;
            uint64_t pad = 0;
            uint16_t type = ELFRelocatable;
            uint16_t machine = MachineIntelGT;
            uint32_t version2 = 1;
            uint64_t entrypoint = 0;
            uint64_t programHeaderOff = 0;
            uint64_t sectionTableOff;
            TargetMetadata flags;
            uint16_t size;
            uint16_t programHeaderSize = 0;
            uint16_t programTableEntries = 0;
            uint16_t sectionHeaderSize;
            uint16_t sectionCount;
            uint16_t strTableIndex = 1;
        } fileHeader;
        struct SectionHeader {
            uint32_t name;
            enum Type : uint32_t {
                Null = 0, Program = 1, SymbolTable = 2, StringTable = 3, RelocationWithAddend=4, Note = 7, ZeInfo = 0xFF000011
            } type;
            uint64_t flags = 0;
            uint64_t addr = 0;
            uint64_t offset;
            uint64_t size;
            uint32_t link = 0;
            uint32_t info = 0;
            uint64_t align = 0x10;
            uint64_t entrySize = 0;
        } sectionHeaders[13];
        struct SymbolEntry {
            uint32_t name = 0;
            enum Info : uint8_t {
                NoType = 0, Object = 1, Func = 2, Section = 3, File = 4, Common = 5, TLS = 6, LOOS = 10, HIOS = 12, LOPROC = 13, HIPROC = 15
            } info = Info::NoType;
            uint8_t other = 0;
            uint16_t shndx = 0;
            uint64_t value = 0;
            uint64_t size = 0;
        } symTable[3];
        struct Note {
            uint32_t nameSize = 8;
            uint32_t descSize = 4;
            enum Type : uint32_t {
                ProductFamily = 1, GfxCoreFamily = 2, TargetMetadata = 3
            } type = Type::GfxCoreFamily;
            const char name[8] = "IntelGT";
            uint32_t payload;
        } noteGfxCore;

#pragma pack(push, 1)
        struct DebugInfo {
            struct {
                uint32_t unitLength;
                uint16_t version = 5;
                uint8_t unitType = DWARF_UT::COMPILE;
                uint8_t addressSize = sizeof(void*);
                uint32_t debugAbbrevOffset = 0;
            } CUHeader;
            struct {
                uint8_t abbrevCode = 1;
                uint32_t name = 0;
                uint32_t lineTable = 0;
                uint64_t low_pc = 0;
                uint64_t high_pc = 0;

                struct {
                    uint8_t abbrevCode = 2;
                    uint32_t name;
                    uint8_t file = 1;
                    uint32_t line = 0;
                    uint8_t column = 1;
                    uint64_t low_pc = 0;
                    uint64_t high_pc = 0;
                } subProgram;

                struct {
                    uint8_t abbrevCode = 0;
                } End;
            } CU;
        } debugInfo;
#pragma pack(pop)

#pragma pack(push, 1)
        struct DebugAbbrev {
            struct {
                uint8_t abbrevCode = 1;
                uint8_t tag = DWARF_TAG::COMPILATION_UNIT;
                uint8_t hasChildren = 1;

                struct {
                    uint8_t attrName;
                    uint8_t attrForm;
                } attributes[5] = {
                    {DWARF_AT::NAME, DWARF_FORM::LINE_STRP},
                    {DWARF_AT::STMT_LIST, DWARF_FORM::LINEPTR},
                    {DWARF_AT::LOW_PC, DWARF_FORM::ADDR},
                    {DWARF_AT::HIGH_PC, DWARF_FORM::ADDR},
                    {0, 0},
                };
            } CU;
            struct {
                uint8_t abbrevCode = 2;
                uint8_t tag = DWARF_TAG::SUBPROGRAM;
                uint8_t hasChildren = 0;

                struct {
                    uint8_t attrName;
                    uint8_t attrForm;
                } attributes[7] = {
                    {DWARF_AT::NAME, DWARF_FORM::STRP},
                    {DWARF_AT::DECL_FILE, DWARF_FORM::DATA1},
                    {DWARF_AT::DECL_LINE, DWARF_FORM::DATA4},
                    {DWARF_AT::DECL_COLUMN, DWARF_FORM::DATA1},
                    {DWARF_AT::LOW_PC, DWARF_FORM::ADDR},
                    {DWARF_AT::HIGH_PC, DWARF_FORM::ADDR},
                    {0, 0},
                };
            } subProgram;
            struct {
                uint8_t code = 0;
            } end;
        } debugAbbrev;
#pragma pack(pop)

        struct Rela {
            uint64_t offset;
            uint64_t info;
            uint64_t addend;
        };

        struct StringTable {
            const char zero = '\0';
            const char snStrTable[10] = ".shstrtab";
            const char snMetadata[9] = ".ze_info";
            const char snNote[21] = ".note.intelgt.compat";
            const char snSym[8] = ".symtab";
            const char kernelEntry[7] = "_entry";
            const char snDebugInfo[17] = ".rela.debug_info";
            const char snDebugAbbrev[14] = ".debug_abbrev";
            const char snDebugLine[17] = ".rela.debug_line";
            const char snDebugLineStr[16] = ".debug_line_str";
            const char snDebugStr[11] = ".debug_str";
            const char snText[6] = {'.', 't', 'e', 'x', 't', '.'};
        } stringTable;

        static size_t align(size_t sz) {
            return (sz + 0xF) & ~0xF;
        }

        ZebinELF(size_t szKernelName, size_t szMetadata, size_t szKernel, size_t offKernelEntry,
                 size_t szDebugLine, size_t szDebugLineStr, uint32_t file1, uint32_t subProgramLine) {
            fileHeader.size = sizeof(fileHeader);
            fileHeader.sectionHeaderSize = sizeof(SectionHeader);
            fileHeader.sectionTableOff = offsetof(ZebinELF, sectionHeaders);
            fileHeader.sectionCount = sizeof(sectionHeaders) / sizeof(SectionHeader);

            fileHeader.flags.all = 0;

            debugInfo.CUHeader.unitLength = sizeof(debugInfo) - sizeof(debugInfo.CUHeader.unitLength);
            debugInfo.CU.name = file1;
            debugInfo.CU.lineTable = 0; // Offset into .debug_line, currently always 0
            debugInfo.CU.high_pc = szKernel;

            debugInfo.CU.subProgram.name = static_cast<uint32_t>(offsetof(StringTable, snText) + strlen(".text."));
            debugInfo.CU.subProgram.line = subProgramLine;
            debugInfo.CU.subProgram.high_pc = szKernel;

            sectionHeaders[0].name = 0;
            sectionHeaders[0].type = SectionHeader::Type::Null;
            sectionHeaders[0].offset = 0;
            sectionHeaders[0].size = 0;

            sectionHeaders[1].name = offsetof(StringTable, snStrTable);
            sectionHeaders[1].type = SectionHeader::Type::StringTable;
            sectionHeaders[1].offset = offsetof(ZebinELF, stringTable);
            sectionHeaders[1].size = sizeof(stringTable) + szKernelName;

            sectionHeaders[2].name = offsetof(StringTable, snMetadata);
            sectionHeaders[2].type = SectionHeader::Type::ZeInfo;
            sectionHeaders[2].offset = align(sizeof(ZebinELF) + szKernelName);
            sectionHeaders[2].size = szMetadata;

            sectionHeaders[3].name = offsetof(StringTable, snText);
            sectionHeaders[3].type = SectionHeader::Type::Program;
            sectionHeaders[3].offset = sectionHeaders[2].offset + align(szMetadata);
            sectionHeaders[3].size = szKernel;
            sectionHeaders[3].flags = 6;    /* SHF_ALLOC | SHF_EXECINSTR */

            sectionHeaders[4].name = offsetof(StringTable, snNote);
            sectionHeaders[4].type = SectionHeader::Type::Note;
            sectionHeaders[4].offset = offsetof(ZebinELF, noteGfxCore);
            sectionHeaders[4].size = sizeof(noteGfxCore);

            sectionHeaders[5].name = offsetof(StringTable, snSym);
            sectionHeaders[5].type = SectionHeader::Type::SymbolTable;
            sectionHeaders[5].offset = offsetof(ZebinELF, symTable);
            sectionHeaders[5].size = sizeof(symTable);
            sectionHeaders[5].link = 1; // String Table Header Index
            sectionHeaders[5].info = sizeof(symTable)/sizeof(symTable[0]);
            sectionHeaders[5].entrySize = sizeof(symTable[0]);

            // The string for the kernel name is appended immediately following
            // the StringTable structure.
            symTable[1].name = sizeof(StringTable);
            symTable[1].info = SymbolEntry::Info::Func;
            symTable[1].shndx = 3; // Program Header Index
            symTable[1].value = 0;
            symTable[1].size = szKernel;

            symTable[2].name = offsetof(StringTable, kernelEntry);
            symTable[2].info = SymbolEntry::Info::NoType;
            symTable[2].shndx = 3; // Program Header Index
            symTable[2].value = offKernelEntry;
            symTable[2].size = 0;

            noteGfxCore.payload = static_cast<uint32_t>(npack::encodeGfxCoreFamily(hw));


            sectionHeaders[6].name = offsetof(StringTable, snDebugInfo) + 5;
            sectionHeaders[6].type = SectionHeader::Type::Program;
            sectionHeaders[6].offset = offsetof(ZebinELF, debugInfo);
            sectionHeaders[6].size = sizeof(debugInfo);

            sectionHeaders[7].name = offsetof(StringTable, snDebugAbbrev);
            sectionHeaders[7].type = SectionHeader::Type::Program;
            sectionHeaders[7].offset = offsetof(ZebinELF, debugAbbrev);
            sectionHeaders[7].size = sizeof(debugAbbrev);

            sectionHeaders[8] = sectionHeaders[1]; /* Dup of strtab */
            sectionHeaders[8].name = offsetof(StringTable, snDebugStr);
            sectionHeaders[8].type = SectionHeader::Type::Program;

            sectionHeaders[9].name = offsetof(StringTable, snDebugLine) + 5;
            sectionHeaders[9].type = SectionHeader::Type::Program;
            sectionHeaders[9].offset = sectionHeaders[3].offset + align(szKernel);
            sectionHeaders[9].size = szDebugLine;

            sectionHeaders[10].name = offsetof(StringTable, snDebugLineStr);
            sectionHeaders[10].type = SectionHeader::Type::Program;
            sectionHeaders[10].offset = sectionHeaders[9].offset + align(szDebugLine);
            sectionHeaders[10].size = szDebugLineStr;

            sectionHeaders[11].name = offsetof(StringTable, snDebugLine);
            sectionHeaders[11].type = SectionHeader::Type::RelocationWithAddend;
            sectionHeaders[11].offset = sectionHeaders[10].offset + align(szDebugLineStr);
            sectionHeaders[11].size = sizeof(Rela);
            sectionHeaders[11].link = 5; // Symbol table header index
            sectionHeaders[11].info = 9; // Debug Line header index
            sectionHeaders[11].entrySize = sizeof(Rela);

            sectionHeaders[12].name = offsetof(StringTable, snDebugInfo);
            sectionHeaders[12].type = SectionHeader::Type::RelocationWithAddend;
            sectionHeaders[12].offset = sectionHeaders[11].offset + align(sectionHeaders[11].size);
            sectionHeaders[12].size = 4*sizeof(Rela);
            sectionHeaders[12].link = 5; // Symbol table header index
            sectionHeaders[12].info = 6; // Debug Line header index
            sectionHeaders[12].entrySize = sizeof(Rela);

        }

        static size_t kernelNameOffset() {
            return offsetof(ZebinELF, stringTable.snText) + sizeof(stringTable.snText);
        }

        bool valid() const {
            if (fileHeader.magic != ELFMagic || fileHeader.elfClass != ELFClass64
                    || fileHeader.endian != ELFLittleEndian || fileHeader.sectionHeaderSize != sizeof(SectionHeader)
                    || (fileHeader.version != 0 && fileHeader.version != ELFVersion1)
                    || (fileHeader.type != ZebinExec && fileHeader.type != ELFRelocatable))
                return false;
            auto *base = reinterpret_cast<const uint8_t *>(&fileHeader);
            auto *sheader = reinterpret_cast<const SectionHeader *>(base + fileHeader.sectionTableOff);
            for (int s = 0; s < fileHeader.sectionCount; s++, sheader++)
                if (sheader->type == SectionHeader::Type::ZeInfo)
                    return true;
            return false;
        }

        void findNotes(const Note *&start, const Note *&end) const {
            auto *base = reinterpret_cast<const uint8_t *>(&fileHeader);
            auto *sheader0 = reinterpret_cast<const SectionHeader *>(base + fileHeader.sectionTableOff);
            const char *strtab = nullptr;
            uint64_t strtabSize = 0;

            auto sheader = sheader0;
            for (int s = 0; s < fileHeader.sectionCount; s++, sheader++) {
                if (sheader->type == SectionHeader::Type::StringTable) {
                    strtab = reinterpret_cast<const char *>(base + sheader->offset);
                    strtabSize = sheader->size;
                }
            }

            bool found = false;
            sheader = sheader0;
            for (int s = 0; s < fileHeader.sectionCount; s++, sheader++)
                if (sheader->type == SectionHeader::Type::Note)
                    if (sheader->name < strtabSize)
                        if (!strcmp(strtab + sheader->name, ".note.intelgt.compat"))
                            { found = true; break; }

            if (found) {
                start = reinterpret_cast<const Note *>(base + sheader->offset);
                end = reinterpret_cast<const Note *>(base + sheader->offset + sheader->size);
            } else
                start = end = nullptr;
        }
    };
};

#define NGEN_FORWARD_ELF(hw) \
NGEN_FORWARD_NO_ELF_OVERRIDES(hw) \
NGEN_FORWARD_ELF_EXTRA(hw) \
template <typename... Targs> void externalName(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::externalName(std::forward<Targs>(args)...); } \
const std::string &getExternalName() const { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getExternalName(); } \
int getSIMD() const { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getSIMD(); } \
int getGRFCount() const { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getGRFCount(); } \
size_t getSLMSize() const { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getSLMSize(); } \
template <typename... Targs> void require32BitBuffers(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::require32BitBuffers(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireArbitrationMode(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireArbitrationMode(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireBarrier(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireBarrier(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireBarriers(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireBarriers(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireDPAS(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireDPAS(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireGlobalAtomics(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireGlobalAtomics(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireGRF(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireGRF(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireLocalID(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireLocalID(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireLocalSize(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireLocalSize(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireNonuniformWGs(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireNonuniformWGs(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireNoPreemption(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireNoPreemption(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireScratch(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireScratch(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireSIMD(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireSIMD(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireSLM(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireSLM(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireStatelessWrites(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireStatelessWrites(std::forward<Targs>(args)...); } \
void requireType(NGEN_NAMESPACE::DataType type) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireType(type); } \
template <typename DT = void> void requireType() { NGEN_NAMESPACE::BinaryCodeGenerator<hw>::template requireType<DT>(); } \
template <typename... Targs> void requireWalkOrder(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireWalkOrder(std::forward<Targs>(args)...); } \
template <typename... Targs> void requireWorkgroup(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::requireWorkgroup(std::forward<Targs>(args)...); } \
template <typename... Targs> void finalizeInterface(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::finalizeInterface(std::forward<Targs>(args)...); } \
template <typename... Targs> void newArgument(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::newArgument(std::forward<Targs>(args)...); } \
template <typename... Targs> void allowArgumentRearrangement(Targs&&... args) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::allowArgumentRearrangement(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::Subregister getArgument(Targs&&... args) { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getArgument(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::Subregister getArgumentIfExists(Targs&&... args) { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getArgumentIfExists(std::forward<Targs>(args)...); } \
template <typename... Targs> int getArgumentSurface(Targs&&... args) { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getArgumentSurface(std::forward<Targs>(args)...); } \
template <typename... Targs> int getArgumentSurfaceIfExists(Targs&&... args) { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getArgumentSurfaceIfExists(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::GRF getLocalID(Targs&&... args) { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getLocalID(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::Subregister getSIMD1LocalID(Targs&&... args) { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getSIMD1LocalID(std::forward<Targs>(args)...); } \
template <typename... Targs> NGEN_NAMESPACE::Subregister getLocalSize(Targs&&... args) { return NGEN_NAMESPACE::ELFCodeGenerator<hw>::getLocalSize(std::forward<Targs>(args)...); } \
void prologue() { NGEN_NAMESPACE::ELFCodeGenerator<hw>::prologue(); } \
void epilogue(const NGEN_NAMESPACE::RegData &r0_info = NGEN_NAMESPACE::RegData()) { NGEN_NAMESPACE::ELFCodeGenerator<hw>::epilogue(r0_info); }

#define NGEN_FORWARD_ELF_EXTRA(hw)

template <HW hw>
std::vector<uint8_t> ELFCodeGenerator<hw>::getBinary()
{
    return getBinary(this->getCode());
}

template <HW hw>
std::vector<uint8_t> ELFCodeGenerator<hw>::getBinary(const std::vector<uint8_t> &kernel)
{
    using super = BinaryCodeGenerator<hw>;
    std::vector<uint8_t> binary;
    std::string metadata;

    // Locate entrypoints for XeHP+.
    if (hw >= HW::XeHP) {
        auto idPerThread = super::_labelLocalIDsLoaded.getID(labelManager);
        auto idCrossThread = super::_labelArgsLoaded.getID(labelManager);

        if (labelManager.hasTarget(idPerThread))
            interface_.setSkipPerThreadOffset(labelManager.getTarget(idPerThread));
        if (labelManager.hasTarget(idCrossThread))
            interface_.setSkipCrossThreadOffset(labelManager.getTarget(idCrossThread));
    }

    // Generate metadata.
    metadata = interface_.generateZeInfo();

    auto debugLine_ = super::debugLine.createDebugLine();
    std::vector<char> debugLine = debugLine_.first;
    uint64_t kernelRela = 1 | (1ull << 32);
    typename ZebinELF::Rela debugLineRelocation = {
        debugLine_.second, kernelRela, 0};
    const std::vector<char> &debugLineStr = super::debugLine.getDebugLineStr();
    uint32_t file1 = super::debugLine.fileEntries[1].strTableOffset;


    std::array<typename ZebinELF::Rela, 4> debugInfoRelocation;
    debugInfoRelocation[0] = {offsetof(typename ZebinELF::DebugInfo, CU.low_pc) , kernelRela, 0};
    debugInfoRelocation[1] = {offsetof(typename ZebinELF::DebugInfo, CU.high_pc), kernelRela, kernel.size()};
    debugInfoRelocation[2] = {offsetof(typename ZebinELF::DebugInfo, CU.subProgram.low_pc), kernelRela, 0};
    debugInfoRelocation[3] = {offsetof(typename ZebinELF::DebugInfo, CU.subProgram.high_pc), kernelRela, kernel.size()};

    // Construct ELF.
    size_t paddedSzKernelName = interface_.getExternalName().length() + 1;
    size_t paddedSzELF = ZebinELF::align(sizeof(ZebinELF) + paddedSzKernelName);
    size_t paddedSzMetadata = ZebinELF::align(metadata.size());
    size_t paddedSzKernel = ZebinELF::align(kernel.size());
    size_t paddedSzDebugLine = ZebinELF::align(debugLine.size());
    size_t paddedSzDebugLineStr = ZebinELF::align(debugLineStr.size());
    size_t paddedSzRelDebugLine = ZebinELF::align(sizeof(debugLineRelocation));
    size_t paddedSzRelDebugInfo = ZebinELF::align(sizeof(debugInfoRelocation));

    binary.resize(paddedSzELF + paddedSzMetadata + paddedSzKernel + paddedSzDebugLine + paddedSzDebugLineStr + paddedSzRelDebugLine + paddedSzRelDebugInfo);

    (void) new(binary.data()) ZebinELF(paddedSzKernelName, metadata.size(), kernel.size(), interface_.getSkipCrossThreadOffset(),
                                       debugLine.size(), debugLineStr.size(), file1, super::debugLine.programLine);
    utils::copy_into(binary, ZebinELF::kernelNameOffset(), interface_.getExternalName());
    size_t offset = paddedSzELF;
    utils::copy_into(binary, offset, metadata);
    offset += paddedSzMetadata;
    utils::copy_into(binary, offset, kernel);
    offset += paddedSzKernel;
    utils::copy_into(binary, offset, debugLine);
    offset += paddedSzDebugLine;
    utils::copy_into(binary, offset, debugLineStr);
    offset += paddedSzDebugLineStr;
    utils::copy_into(binary, offset, debugLineRelocation);
    offset += paddedSzRelDebugLine;
    utils::copy_into(binary, offset, debugInfoRelocation);
    return binary;
}

template <HW hw>
inline HW ELFCodeGenerator<hw>::getBinaryArch(const std::vector<uint8_t> &binary)
{
    HW outHW;
    Product outProduct;

    getBinaryHWInfo(binary, outHW, outProduct);

    return outHW;
}

template <HW hw>
inline void ELFCodeGenerator<hw>::getBinaryHWInfo(const std::vector<uint8_t> &binary, HW &outHW, Product &outProduct)
{
    using Note = typename ZebinELF::Note;

    outHW = HW::Unknown;
    outProduct.family = ProductFamily::Unknown;
    outProduct.stepping = 0;

    auto zebinELF = reinterpret_cast<const ZebinELF *>(binary.data());
    if (zebinELF->valid()) {
        // Check for .note.intelgt.compat section first. If not present, fall back to flags.
        const Note *start, *end;
        zebinELF->findNotes(start, end);
        if (start && end) {
            while (start < end) {
                auto rstart = reinterpret_cast<const uint8_t *>(start);
                if (start->descSize == sizeof(start->payload)) {
                    auto *actualPayload = reinterpret_cast<const uint32_t *>(
                        rstart + offsetof(Note, payload) - sizeof(Note::name) + utils::alignup_pow2(start->nameSize, 4)
                    );
                    switch (start->type) {
                        case Note::Type::ProductFamily: {
                            auto decodedFamily = npack::decodeProductFamily(static_cast<npack::ProductFamily>(*actualPayload));
                            if (decodedFamily != ProductFamily::Unknown)
                                outProduct.family = decodedFamily;
                            break;
                        }
                        case Note::Type::GfxCoreFamily:
                            if (outHW == HW::Unknown)
                                outHW = npack::decodeGfxCoreFamily(static_cast<npack::GfxCoreFamily>(*actualPayload));
                            break;
                        case Note::Type::TargetMetadata: {
                            typename ZebinELF::TargetMetadata metadata;
                            metadata.all = *actualPayload;
                            outProduct.stepping = metadata.parts.minHWRevision;
                        }
                        default: break;
                    }
                }
                start = reinterpret_cast<const Note *>(
                    rstart + offsetof(Note, name)
                           + utils::alignup_pow2(start->nameSize, 4)
                           + utils::alignup_pow2(start->descSize, 4)
                );
            }
        } else {
            if (zebinELF->fileHeader.flags.parts.useGfxCoreFamily)
                outHW = npack::decodeGfxCoreFamily(static_cast<npack::GfxCoreFamily>(zebinELF->fileHeader.machine));
            else
                outProduct.family = npack::decodeProductFamily(static_cast<npack::ProductFamily>(zebinELF->fileHeader.machine));
            outProduct.stepping = zebinELF->fileHeader.flags.parts.minHWRevision;
        }
    } else
        npack::getBinaryHWInfo(binary, outHW, outProduct);

    if (outHW != HW::Unknown && outProduct.family == ProductFamily::Unknown)
        outProduct.family = genericProductFamily(outHW);
    else if (outHW == HW::Unknown && outProduct.family != ProductFamily::Unknown)
        outHW = getCore(outProduct.family);
}

} /* namespace NGEN_NAMESPACE */

#endif /* NGEN_ELF_HPP */

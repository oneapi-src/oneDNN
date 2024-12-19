/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef NGEN_DEBUGINFO_HPP
#define NGEN_DEBUGINFO_HPP

#include <sstream>
#include <unordered_map>
#include <vector>

#include "ngen_utils.hpp"

#ifdef NGEN_ENABLE_SOURCE_LOCATION
#include <source_location>
#endif


namespace NGEN_NAMESPACE {

struct DebugConfig {
    DebugConfig() = default;
    DebugConfig(const char * name, uint32_t line, bool enableLineMapping): nameCU(name), programLine(line), enableLineMapping(enableLineMapping) {};

    const char *nameCU = "(unknown)";
    uint32_t programLine = 0;
    bool enableLineMapping = false;
};

struct SourceLocation {
#ifdef NGEN_ENABLE_SOURCE_LOCATION
    SourceLocation(std::source_location where = std::source_location::current())
        : value(where) {}
    SourceLocation(const SourceLocation &) = default;
    SourceLocation(SourceLocation &&) = default;
    const char *filePath() { return value.file_name(); }
    uint32_t line() { return value.line(); }
    std::source_location value;
    std::string str() {
        std::ostringstream oss;
        oss << value.file_name() << ":" << value.line();
        return oss.str();
    }
#endif
};

struct DebugLine {
    DebugLine(const DebugConfig & conf): enableSrcLines(conf.enableLineMapping), programLine(conf.programLine) {
#ifndef NGEN_ENABLE_SOURCE_LOCATION
        // Unsupported
        enableSrcLines = false;
#endif
        // Reuse .debug_line_str for simplicity
        add(conf.nameCU);
    }

    struct Line {
        Line(uint64_t address, uint32_t fileEntry, uint32_t line)
            : address(address), fileEntry(fileEntry), line(line) {}
        uint64_t address;
        uint32_t fileEntry;
        uint32_t line;
        std::string str() {
            std::ostringstream oss;
            oss << "0x" << std::hex << address << " -> " << fileEntry;
            return oss.str();
        }
    };
    uint32_t add(const char * filePath) {
        uint32_t entry = 0;
        auto f = fileMap.find(filePath);
        if(f == fileMap.end()) {
            const char * filenamePtr = [&]() {
                for(int64_t i = strlen(filePath) - 1; i >= 0; i--) {
                    if(filePath[i] == '/' || filePath[i] == '\\')
                        return filePath + i + 1;
                }
                return filePath;
            }();
            std::string filename = filenamePtr;
            std::string dirName = filenamePtr == filePath ? "" : std::string(filePath, filenamePtr - filePath - 1);
            uint32_t dirEntryIndex = [&]() {
                auto e = dirMap.find(dirName);
                if (e != dirMap.end())
                    return e->second;

                uint32_t ret = static_cast<uint32_t>(dirEntries.size());

                // Add new directory entry
                DirEntry dirEntry = {static_cast<uint32_t>(strTable.size())};
                for (char c : dirName) {
                    strTable.emplace_back(c);
                }
                strTable.emplace_back(0);
                dirEntries.emplace_back(dirEntry);
                dirMap[dirName] = ret;

                return ret;
            }();

            FileEntry fileEntry = {static_cast<uint32_t>(strTable.size()),
                                   dirEntryIndex};
            for (char c : filename) {
                strTable.emplace_back(c);
            }
            strTable.emplace_back(0);
            fileEntries.emplace_back(fileEntry);
            entry = static_cast<uint32_t>(fileEntries.size() - 1);
            fileMap[filePath] = entry;
        } else {
            entry = f->second;
        }
        return entry;
    }

    void add(uint64_t address, SourceLocation loc) {
#ifdef NGEN_ENABLE_SOURCE_LOCATION
        if(enableSrcLines) srcLines.emplace_back(address, add(loc.filePath()), loc.line());
#endif
    }

    std::string str() {
        std::ostringstream oss;
        oss << "Directory Table: \n";
        for (auto &e : dirEntries)
            if (strTable[e.strTableOffset] == 0)
                oss << "\t.\n";
            else
                oss << "\t" << &strTable[e.strTableOffset] << "\n";

        oss << "File Table: \n";
        for (auto &e : fileEntries) {
            const char *dir =
                &strTable[dirEntries[e.dirEntriesIndex].strTableOffset];
            if (*dir == 0)
                oss << "\t" << &strTable[e.strTableOffset] << "\n";
            else
                oss << "\t" << dir << "/" << &strTable[e.strTableOffset] << "\n";
        }

        for (auto &line : srcLines) {
            oss << line.str() << "\n";
        }
        return oss.str();
    }

#pragma pack(push, 1)
    struct DebugLineHeaderBase {
        uint32_t unitLength; // Length of all debug line information excluding itself
        uint16_t version = 5;
        uint8_t addressSize = sizeof(void*);
        uint8_t segmentSelectorSize = 0;
        uint32_t headerLength;
        uint8_t minimumInstructionLength = 16; // sizeof(Instruction12), hard-coded to avoid dependency
        uint8_t maximumOperationsPerInstruction = 1;
        uint8_t defaultIsStmt = 1;
        int8_t lineBase = -32;
        uint8_t lineRange = 192;
        uint8_t opcodeBase = 5;
        uint8_t opCodeLengths[4] = {0, 1, 1, 1};

        enum DWARF_FORM : uint8_t {
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

        struct DirEntriesHeader {
            uint8_t directoryEntryFormatCount = 1;
            struct {
                uint8_t type;
                uint8_t form;
            } directoryEntryFormat[1] = {
                {DWARF_LNCT::PATH, DWARF_FORM::LINE_STRP},
            };
        };

        struct FileEntriesHeader {
            uint8_t filenameEntryFormatCount = 2;
            struct {
                uint8_t type;
                uint8_t form;
            } filenameEntryFormat[2] = {
                {DWARF_LNCT::PATH, DWARF_FORM::LINE_STRP},
                {DWARF_LNCT::DIRECTORY_INDEX, DWARF_FORM::DATA4},
            };
        };
    };
#pragma pack(pop)

    static void encodeLEB128(std::vector<char> &out, int64_t a) {
        do {
            out.emplace_back(static_cast<uint8_t>((a & 0x7f) | 0x80));
            a >>= 7;
        } while (a != 0 && a != -1);
        if (a == -1 && (out.back() & 0x40) == 0) {
            out.emplace_back(0x7f);
        } else if (a == 0 && (out.back() & 0x40)) {
            out.emplace_back(0x00);
        } else  {
            out.back() &= 0x7f;
        }
    }

    static void encodeULEB128(std::vector<char> &out, size_t a) {
        do {
            out.emplace_back(static_cast<uint8_t>((a & 0x7f) | 0x80));
            a >>= 7;
        } while (a > 0);
        out.back() &= 0x7f;
    };

    static std::vector<char> toULEB128(size_t a) {
        std::vector<char> ret;
        encodeULEB128(ret, a);
        return ret;
    };

    std::pair<std::vector<char>, uint64_t> encodeLineStatements(const DebugLineHeaderBase & base) const {
        struct {
            uint64_t address = 0;
            uint64_t line = 1;
            uint32_t file = 1;
            bool isStmt = 1;
            uint32_t isa = 0;
            uint32_t discriminator = 0;

            int8_t lineBase;
            uint8_t lineRange;
            uint8_t opcodeBase;
            uint8_t addressScale;

            enum StandardOps {
                extendedOp = 0,
                copy = 1,
                advancePC = 2,
                advanceLine = 3,
                setFile = 4,
            };

            enum ExtendedOps { endSequence = 1, setAddress = 2 };

            uint64_t initPC(std::vector<char> &out) {
                out.emplace_back(extendedOp);
                encodeULEB128(out, 9);
                out.emplace_back(setAddress);
                uint64_t relocationOffset = out.size();
                for (int i = 0; i < 8; i++) {
                    out.emplace_back(0);
                }
                return relocationOffset;
            }

            void finalize(std::vector<char> &out) {
                out.emplace_back(advancePC);
                encodeULEB128(out, 1);
                out.emplace_back(extendedOp);
                encodeULEB128(out, 1);
                out.emplace_back(endSequence);
            }

            void encode(std::vector<char> &out, const Line &l) {
                size_t lineAddress = l.address / addressScale;

                if(l.fileEntry != file) {
                    out.emplace_back(setFile);
                    encodeULEB128(out, l.fileEntry);
                    file = l.fileEntry;
                }

                if (l.line >= line + lineBase &&
                    l.line < line + lineBase + lineRange) {
                    int64_t specialOp = (l.line - line) - lineBase +
                        (lineAddress - address) * lineRange +
                        opcodeBase;
                    if (specialOp > 0 && specialOp <= 255) {
                        out.emplace_back(static_cast<uint8_t>(specialOp));
                        line = l.line;
                        address = lineAddress;
                        return;
                    }
                }

                if(lineAddress != address) {
                    out.emplace_back(advancePC);
                    encodeULEB128(out, lineAddress - address);
                    address = lineAddress;
                }

                if(l.line != line) {
                    out.emplace_back(advanceLine);
                    encodeLEB128(out, static_cast<int64_t>(l.line) - line);
                    line = l.line;
                }

                out.emplace_back(copy);
            }
        } encodeState;

        encodeState.isStmt = base.defaultIsStmt;
        encodeState.lineBase = base.lineBase;
        encodeState.lineRange = base.lineRange;
        encodeState.opcodeBase = base.opcodeBase;
        encodeState.addressScale = base.minimumInstructionLength;

        std::vector<char> ret;
        uint64_t relocation_offset = encodeState.initPC(ret);
        for(const auto & l: srcLines) {
            encodeState.encode(ret, l);
        }
        encodeState.finalize(ret);

        return {ret, relocation_offset};
    }

    std::pair<std::vector<char>, uint64_t> createDebugLine() const {

        DebugLineHeaderBase base;
        DebugLineHeaderBase::DirEntriesHeader dirHeader;
        std::vector<char> directoriesCount = toULEB128(dirEntries.size());
        uint32_t dirEntriesBytes = static_cast<uint32_t>(dirEntries.size() * sizeof(dirEntries[0]));
        DebugLineHeaderBase::FileEntriesHeader fileHeader;
        std::vector<char> filenamesCount = toULEB128(fileEntries.size());
        uint32_t fileEntriesBytes = static_cast<uint32_t>(fileEntries.size() * sizeof(fileEntries[0]));
        auto encode = encodeLineStatements(base);
        std::vector<char> lineStatements = encode.first;
        uint64_t relocation_offset = encode.second;

        uint32_t headerEntriesSize = static_cast<uint32_t> (sizeof(dirHeader) + directoriesCount.size()
            + dirEntriesBytes + sizeof(fileHeader) + filenamesCount.size() + fileEntriesBytes);
        uint32_t lineTableSize = static_cast<uint32_t>(lineStatements.size());

        base.unitLength = sizeof(base) - sizeof(base.unitLength) + headerEntriesSize + lineTableSize;
        base.headerLength = sizeof(base) -
            ((offsetof(DebugLineHeaderBase, headerLength)) +
            sizeof(base.headerLength)) + headerEntriesSize;

        std::vector<char> ret(base.unitLength + sizeof(base.unitLength));

        char *data = ret.data();

        // Write header base data
        memcpy(data, &base, sizeof(base));
        data += sizeof(base);

        // Write directory entries
        memcpy(data, &dirHeader, sizeof(dirHeader));
        data += sizeof(dirHeader);
        memcpy(data, directoriesCount.data(), directoriesCount.size());
        data += directoriesCount.size();
        memcpy(data, dirEntries.data(), dirEntriesBytes);
        data += dirEntriesBytes;

        // Write file entries
        memcpy(data, &fileHeader, sizeof(fileHeader));
        data += sizeof(fileHeader);
        memcpy(data, filenamesCount.data(), filenamesCount.size());
        data += filenamesCount.size();
        memcpy(data, fileEntries.data(), fileEntriesBytes);
        data += fileEntriesBytes;

        // Write line statements
        relocation_offset += data - ret.data();
        memcpy(data, lineStatements.data(), lineStatements.size());
        data += lineStatements.size();

        return {ret, relocation_offset};
    };
    const std::vector<char> & getDebugLineStr() {
        return strTable;
    }

    std::vector<char> strTable = {0};
    struct DirEntry {
        uint32_t strTableOffset;
    };
    std::vector<DirEntry> dirEntries = {{0}};
    std::unordered_map<std::string, uint32_t> dirMap = {{"", 0}};

    struct FileEntry {
        uint32_t strTableOffset;
        uint32_t dirEntriesIndex;
    };

    std::vector<FileEntry> fileEntries = {{0, 0}};
    std::unordered_map<std::string, uint32_t> fileMap {{"", 0}};

    std::vector<Line> srcLines;

    bool enableSrcLines = false;
    uint32_t programLine = 0;
};

} // namespace NGEN_NAMESPACE

#endif

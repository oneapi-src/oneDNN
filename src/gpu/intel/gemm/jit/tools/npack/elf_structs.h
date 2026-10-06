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

/*********************************************************************/
/* ELF definitions, adapted from NEO's                               */
/*      elf/types.h                                                  */
/*********************************************************************/

#ifndef ELF_STRUCTS_H
#define ELF_STRUCTS_H

#include <stddef.h>
#include <stdint.h>

#define STRING_TABLE_SECTION 0x3
#define OPENCL_DEV_BINARY_SECTION 0xFF000005
#define OPENCL_DEV_DEBUG_SECTION 0xFF000008
#define ELF_MAGIC 0x464C457F

typedef struct {
    uint32_t Magic;
    uint8_t Class;
    uint8_t Endian;
    uint8_t ElfVersion;
    uint8_t ABI;
    uint8_t Pad[8];
    uint16_t Type;
    uint16_t Machine;
    uint32_t Version;
    uint64_t EntryAddress;
    uint64_t ProgramHeadersOffset;
    uint64_t SectionHeadersOffset;
    uint32_t Flags;
    uint16_t ElfHeaderSize;
    uint16_t ProgramHeaderEntrySize;
    uint16_t NumProgramHeaderEntries;
    uint16_t SectionHeaderEntrySize;
    uint16_t NumSectionHeaderEntries;
    uint16_t SectionNameTableIndex;
} SElf64Header;

typedef struct {
    uint32_t Name;
    uint32_t Type;
    uint64_t Flags;
    uint64_t Address;
    uint64_t DataOffset;
    uint64_t DataSize;
    uint32_t Link;
    uint32_t Info;
    uint64_t Alignment;
    uint64_t EntrySize;
} SElf64SectionHeader;

#endif /* header guard */

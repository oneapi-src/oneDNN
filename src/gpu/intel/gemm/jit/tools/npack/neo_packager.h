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

#ifndef NEO_PACKAGER_H
#define NEO_PACKAGER_H 1

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    PACKAGER_SUCCESS = 0,
    PACKAGER_BAD_ELF = -1,
    PACKAGER_NO_BINARY_SECTION = -2,
    PACKAGER_BAD_BINARY_SECTION = -3,
    PACKAGER_OUT_OF_MEMORY = -4,
    PACKAGER_INVALID_CHECKSUM = -5
} packager_status;

// Read OpenCL ELF binary and replace its kernel with the given kernel.
// Optionally, extra patch tokens (if patches != NULL) can be appended to the patch list.
packager_status replace_kernel(const char *elf_binary, size_t elf_size,
                               const void *kernel, size_t kernel_size,
                               const void *patches, size_t patches_size,
                               bool no_inline_data,
                               char **new_elf_binary, size_t *new_elf_size);

#endif /* header guard */

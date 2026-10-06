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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neo_packager.h"

void usage(const char *pname)
{
    fprintf(stderr, "Usage: %s arch orig_binary kernel [extra_patches]\n\n", pname);
    fprintf(stderr, "  arch:          architecture abbreviation (e.g. 12p7)\n");
    fprintf(stderr, "  orig_binary:   file containing compiled OpenCL program (from ocloc or clGetProgramInfo(..., CL_PROGRAM_BINARIES, ...))\n");
    fprintf(stderr, "  kernel:        file containing Gen kernel binary\n");
    fprintf(stderr, "  extra_patches: (optional) file containing additional patch tokens to append to the OpenCL device binary.\n\n");
    fprintf(stderr, "The new OpenCL binary is output to stdout.\n");
    exit(2);
}

void handle_packager_error(packager_status status)
{
    if (status != PACKAGER_SUCCESS) {
        const char *msg;
        switch (status)
        {
            case PACKAGER_SUCCESS: break;
            case PACKAGER_BAD_ELF:
                msg = "Error parsing ELF.\n";
                break;
            case PACKAGER_BAD_BINARY_SECTION:
                msg = "Error parsing device binary.\n";
                break;
            case PACKAGER_NO_BINARY_SECTION:
                msg = "No device binary section found in ELF.\n";
                break;
            case PACKAGER_OUT_OF_MEMORY:
                msg = "Out of memory.\n";
                break;
            case PACKAGER_INVALID_CHECKSUM:
                msg = "Checksum mismatch in device binary.\n";
                break;
            }
            fputs(msg, stderr);
            exit(1);
    }
}

void read_file(const char *fname, char **data, size_t *size)
{
    FILE *file = fopen(fname, "rb");

    if (file == NULL) {
        fprintf(stderr, "Could not open file %s\n", fname);
        exit(3);
    }

    fseek(file, 0L, SEEK_END);
    *size = ftell(file);
    rewind(file);

    *data = (char *)malloc(*size);
    if (!*data) {
        fprintf(stderr, "Out of memory reading file %s\n", fname);
        fclose(file);
        exit(3);
    }

    if (*size != fread(*data, 1, *size, file)) {
        fprintf(stderr, "Error reading file %s\n", fname);
        fclose(file);
        exit(3);
    }

    fclose(file);
}

int main(int argc, char **argv)
{
    char *elf_binary, *kernel_binary, *new_elf_binary, *patches = NULL;
    size_t elf_size, kernel_size, new_elf_size, patches_size;

    if (argc < 4 || argc > 5)
        usage(argv[0]);

    const char *arch = argv[1];
    read_file(argv[2], &elf_binary, &elf_size);
    read_file(argv[3], &kernel_binary, &kernel_size);
    if (argc > 4)
        read_file(argv[4], &patches, &patches_size);

    bool no_inline_data = !strcmp(arch, "Xe2") && !strcmp(arch, "Xe3");

    handle_packager_error(replace_kernel(elf_binary, elf_size, kernel_binary, kernel_size, patches, patches_size, no_inline_data, &new_elf_binary, &new_elf_size));

    if (new_elf_size != fwrite(new_elf_binary, 1, new_elf_size, stdout))
        return 255;

    free(patches);
    free(elf_binary);
    free(kernel_binary);
    free(new_elf_binary);
}

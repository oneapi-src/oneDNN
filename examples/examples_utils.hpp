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

#ifndef EXAMPLE_UTILS_HPP
#define EXAMPLE_UTILS_HPP

#include <string>
#include <iostream>
#include <stdlib.h>
#include <algorithm>

#include "mkldnn.hpp"

static mkldnn::engine::kind get_engine_kind(int argc, char **argv) {
    if (argc == 2) {
        std::string engine_kind_str = argv[1];
        if (engine_kind_str == "cpu") {
            return mkldnn::engine::kind::cpu;
        } else if (engine_kind_str == "gpu") {
            // Checking if a gpu exists on the machine
            if (mkldnn::engine::get_count(mkldnn::engine::kind::gpu) == 0) {
                std::cerr << "Application couldn't find GPU, please run with "
                             "CPU instead. Thanks!"
                          << std::endl;
                exit(1);
            }
            return mkldnn::engine::kind::gpu;
        } else {
            std::cerr << "Please run example like this: ";
            std::cerr << argv[0] << " cpu or  ";
            std::cerr << argv[0] << "gpu, thanks!";
            exit(1);
        }
    }
    // If argc == 1, return default engine i.e. CPU
    return mkldnn::engine::kind::cpu;
}

// Read from memory, write to handle
static void read(void *handle, const mkldnn::memory &mem) {
    size_t bytes = mem.get_desc().get_size();
    uint8_t *src = mem.map_data<uint8_t>();
    std::copy(src, src + bytes, (uint8_t *)handle);
    mem.unmap_data(src);
}

// Read from handle, write to memory
static void write(const void *handle, mkldnn::memory &mem) {
    size_t bytes = mem.get_desc().get_size();
    uint8_t *dst = mem.map_data<uint8_t>();
    std::copy((uint8_t *)handle, (uint8_t *)handle + bytes, dst);
    mem.unmap_data(dst);
}

#endif

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

#ifndef GPU_MICROKERNELS_PACKAGE_HPP
#define GPU_MICROKERNELS_PACKAGE_HPP

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "protocol.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace micro {

struct Argument;
struct RegisterRange;
struct Setting;

// Microkernel package.
//
// Fields marked [*] are automatically filled in by the entrance agent.
struct Package {
    /* Identifiers */
    Protocol protocol; // Protocol implemented by microkernel
    uint64_t luid; // Unique package ID for use in catalog [*]
    std::vector<uint8_t>
            providerID; // Optional free-form identifier for use by microkernel provider

    /* Code */
    std::vector<uint8_t> binary; // Raw binary blob

    /* Register usage */
    std::vector<Argument>
            arguments; // Input and output arguments for microkernel
    std::vector<RegisterRange>
            clobbers; // Registers clobbered by microkernel (includes arguments) [*]

    /* Requirements */
    uint32_t gmdidCompat; // Compatible GMDID
    int grfMin = 0; // Minimum GRF size [*]
    int barrierCount = 0; // Number of barriers used by microkernel
    bool systolic = false; // Does microkernel use systolic array? [*]

    /* Configuration */
    std::vector<Setting>
            settings; // Description of this microkernel's configuration (WG size, tile size, etc.) for host kernel to interpret

    inline int getSetting(const char *name) const;
};

// Contiguous span of register space.
struct RegisterRange {
    uint32_t boffset = 0; // Byte offset into GRF
    uint32_t blen = 0; // Length of range in bytes

    RegisterRange() = default;
    RegisterRange(uint32_t boffset_, uint32_t blen_)
        : boffset(boffset_), blen(blen_) {}
};

// Encapsulation of tensor size information.
struct TensorConfig {
    static constexpr int maxDims = 4;
    std::array<int, maxDims> dims
            = {1, 1, 1, 1}; // Tensor tile size (elements per dimension)
    std::array<int, maxDims> block = {1, 1, 1,
            1}; // Block sizes within tile (equal to dims if only one block)

    int elements() const {
        int result = 1;
        for (auto d : dims)
            result *= d;
        return result;
    }

    int blockElements() const {
        int result = 1;
        for (auto d : block)
            result *= d;
        return result;
    }

    bool blocked() const {
        for (int i = 0; i < maxDims; i++)
            if (block[i] < dims[i]) return true;
        return false;
    }

    int blocks() const {
        int result = 1;
        for (int i = 0; i < maxDims; i++)
            result *= dims[i] / block[i];
        return result;
    }
};

// Information on a single argument (input/output).
struct Argument {
    std::string name; // Argument name
    std::vector<RegisterRange> location; // Register location(s)
    StructuredType::Type actualType
            = StructuredType::any; // Type, if not specified by protocol
    TensorConfig sizes; // Tensor size, for tensor arguments
};

// Information on a single configuration setting.
struct Setting {
    std::string name; // Setting name
    int value; // Setting numeric value
};

int Package::getSetting(const char *name) const {
    for (auto &setting : settings)
        if (setting.name == name) return setting.value;
    throw std::runtime_error(
            std::string(
                    "Microkernel package does not provide requested setting: ")
            + name);
}

} /* namespace micro */
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

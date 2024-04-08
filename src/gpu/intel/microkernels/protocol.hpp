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

#ifndef GPU_MICROKERNELS_PROTOCOL_HPP
#define GPU_MICROKERNELS_PROTOCOL_HPP

#include <cstdint>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace micro {

struct ProtocolArgument;
struct ProtocolSetting;

// A protocol describes a class of microkernels that provide the same functionality
//  and share a high-level interface.
// A Protocol object should not be created directly; instead protocols are created via
//  a specific subclass (e.g. GEMMProtocol).
class Protocol {
public:
    const char *kernelBaseName() const;
    std::vector<ProtocolArgument> arguments() const;
    std::vector<ProtocolSetting> settings() const;

protected:
    enum Family : uint32_t { Invalid = 0, GEMM = 0x39bfca02 };

    uint32_t family = Family::Invalid;
    uint32_t ioptions = 0;
};

class GEMMProtocol : public Protocol {
public:
    struct Options {
        bool localA = false;
        bool localB = false;
        bool addToC = false;
        bool slmPtr = false;
    };

    GEMMProtocol() : GEMMProtocol(Options {}) {}
    GEMMProtocol(const Options &options);

    Options options() const;

protected:
    friend class Protocol;
    const char *kernelBaseName() const;
    std::vector<ProtocolArgument> arguments() const;
    std::vector<ProtocolSetting> settings() const;
};

// Describes the type of a microkernel argument (scalar/pointer/tensor).
struct StructuredType {
    enum Type { // Element data type
        u64,
        s64,
        u32,
        s32,
        u16,
        s16,
        u8,
        s8, //    integral
        f64,
        f32,
        f16,
        bf16, //    floating-point
        any, //    unspecified
    } type
            = Type::any;
    enum Format { Scalar, GlobalPointer, LocalPointer, Tensor } format = Scalar;
    int ndims = 1;

    StructuredType() {}
    StructuredType(Type type_) : type(type_) {}
    StructuredType(Format format_) : format(format_) {}
    StructuredType(int ndims_) : format(Tensor), ndims(ndims_) {}
};

// Description of a single argument from a protocol's prototype.
struct ProtocolArgument {
    const char *name;
    enum { In = 0b01, Out = 0b10, InOut = In | Out } direction;
    StructuredType stype;

    bool in() const { return direction & In; }
    bool out() const { return direction & Out; }
};

// Description of a single protocol setting.
struct ProtocolSetting {
    const char *name;
};

} /* namespace micro */
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

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

#ifndef NGEN_DECODER_HPP
#define NGEN_DECODER_HPP

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-int-conversion"
#endif

#include "ngen_core.hpp"
#include "ngen_auto_swsb.hpp"

namespace NGEN_NAMESPACE {

#include "ngen_gen8.hpp"
#include "ngen_gen12.hpp"

using DependencyRegion = autoswsb::DependencyRegion;

#ifdef NGEN_SAFE
class unsupported_compaction : public std::runtime_error {
public:
    unsupported_compaction() : std::runtime_error("Compacted instructions are not supported") {}
};
class unimplemented : public std::runtime_error {
public:
    unimplemented() : std::runtime_error("Operation is not implemented") {}
};
#endif

class Decoder
{
public:
    Decoder(HW hw_, const std::vector<uint8_t> &program) : Decoder(hw_, program.data(), program.size()) {}
    Decoder(HW hw_, const uint8_t *program, size_t bytes)
        : hw(hw_), current(program), end(program + bytes) {}

    void advance()    { checkCompaction(); current += 0x10; }
    bool done() const { return current >= end; }

    Opcode opcode() const { return static_cast<Opcode>(*current & 0x7F); }
    inline bool getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const;

protected:
    HW hw;
    const uint8_t *current, *end;

    void checkCompaction() const {
#ifdef NGEN_SAFE
        if (get<Instruction12>().common.cmptCtrl)   /* same bit pre-Gen12 */
            throw unsupported_compaction();
#endif
    }

    template <typename Instruction>
    Instruction get() const {
        Instruction i;
        std::memcpy(&i, current, sizeof(i));
        return i;
    }
};

bool Decoder::getOperandRegion(autoswsb::DependencyRegion &region, int opNum) const
{
    checkCompaction();
    region.hw = hw;
    if (hw >= HW::XeHPC)
        return get<InstructionXeHPC>().getOperandRegion(region, opNum);
    if (hw >= HW::Gen12LP)
        return get<Instruction12>().getOperandRegion(region, opNum);
#ifdef NGEN_SAFE
    throw unimplemented();
#else
    return false;
#endif
}

} /* namespace NGEN_NAMESPACE */

#ifdef ENABLE_LLVM_WCONVERSION
#pragma clang diagnostic pop
#endif

#endif /* header guard */

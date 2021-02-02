/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef NGEN_INTERFACE_HPP
#define NGEN_INTERFACE_HPP


#include "ngen_core.hpp"
#include <sstream>


namespace ngen {

template <HW hw> class OpenCLCodeGenerator;
template <HW hw> class L0CodeGenerator;

// Exceptions.
#ifdef NGEN_SAFE
class unknown_argument_exception : public std::runtime_error {
public:
    unknown_argument_exception() : std::runtime_error("Argument not found") {}
};

class bad_argument_type_exception : public std::runtime_error {
public:
    bad_argument_type_exception() : std::runtime_error("Bad argument type") {}
};

class interface_not_finalized : public std::runtime_error {
public:
    interface_not_finalized() : std::runtime_error("Interface has not been finalized") {}
};
#endif

enum class ExternalArgumentType { Scalar, GlobalPtr, LocalPtr, Hidden };

class InterfaceHandler
{
    template <HW hw> friend class OpenCLCodeGenerator;
    template <HW hw> friend class L0CodeGenerator;

public:
    InterfaceHandler(HW hw_) : hw(hw_), simd(GRF::bytes(hw_) >> 2) {}

    inline void externalName(const std::string &name)   { kernelName = name; }

    template <typename DT>
    inline void newArgument(std::string name)           { newArgument(name, getDataType<DT>()); }
    inline void newArgument(std::string name, DataType type, ExternalArgumentType exttype = ExternalArgumentType::Scalar);
    inline void newArgument(std::string name, ExternalArgumentType exttype);

    inline Subregister getArgument(const std::string &name) const;
    inline Subregister getArgumentIfExists(const std::string &name) const;
    inline int getArgumentSurface(const std::string &name) const;
    inline GRF getLocalID(int dim) const;
    inline Subregister getLocalSize(int dim) const;

    const std::string &getExternalName() const           { return kernelName; }
    int getSIMD() const                                  { return simd; }

    void require32BitBuffers()                           { allow64BitBuffers = false; }
    void requireBarrier()                                { barrierCount = 1; }
    void requireGRF(int grfs)                            { needGRF = grfs; }
    void requireNonuniformWGs()                          { needNonuniformWGs = true; }
    void requireNoPreemption()                           { needNoPreemption = true; }
    void requireLocalID(int dimensions)                  { needLocalID = dimensions; }
    void requireLocalSize()                              { needLocalSize = true; }
    void requireScratch(size_t bytes = 1)                { scratchSize = bytes; }
    void requireSIMD(int simd_)                          { simd = simd_; }
    void requireSLM(size_t bytes)                        { slmSize = bytes; }
    void requireStatelessWrites(bool req = true)         { needStatelessWrites = req; }
    inline void requireType(DataType type);
    template <typename T> void requireType()             { requireType(getDataType<T>()); }
    void requireWorkgroup(size_t x, size_t y, size_t z)  { wg[0] = x; wg[1] = y; wg[2] = z; }

    void setSkipPerThreadOffset(int32_t offset)          { offsetSkipPerThread = offset; }
    void setSkipCrossThreadOffset(int32_t offset)        { offsetSkipCrossThread = offset; }

    inline void finalize();

    inline void generateDummyCL(std::ostream &stream) const;
    inline std::string generateZeInfo() const;

#ifdef NGEN_ASM
    inline void dumpAssignments(std::ostream &stream) const;
#endif

protected:
    struct Assignment {
        std::string name;
        DataType type;
        ExternalArgumentType exttype;
        Subregister reg;
        int surface;
        int index;
    };

    HW hw;

    std::vector<Assignment> assignments;
    std::string kernelName = "default_kernel";

    int nextArgIndex = 0;
    bool finalized = false;

    bool allow64BitBuffers = 0;
    int barrierCount = 0;
    int32_t needGRF = 128;
    int needLocalID = 0;
    bool needLocalSize = false;
    bool needNonuniformWGs = false;
    bool needNoPreemption = false;
    bool needHalf = false;
    bool needDouble = false;
    bool needStatelessWrites = true;
    int32_t offsetSkipPerThread = 0;
    int32_t offsetSkipCrossThread = 0;
    size_t scratchSize = 0;
    int simd = 8;
    size_t slmSize = 0;
    size_t wg[3] = {0, 0, 0};

    int crossthreadGRFs = 0;
    inline int getCrossthreadGRFs() const;
    inline GRF getCrossthreadBase(bool effective = true) const;
    int grfsPerLID() const { return (simd > 16) ? 2 : 1; }
};

using NEOInterfaceHandler = InterfaceHandler;

void InterfaceHandler::newArgument(std::string name, DataType type, ExternalArgumentType exttype)
{
    assignments.push_back({name, type, exttype, GRF(0).ud(0), -1, nextArgIndex++});
}

void InterfaceHandler::newArgument(std::string name, ExternalArgumentType exttype)
{
    DataType type = DataType::invalid;

    switch (exttype) {
        case ExternalArgumentType::GlobalPtr: type = DataType::uq; break;
        case ExternalArgumentType::LocalPtr:  type = DataType::ud; break;
        default:
#ifdef NGEN_SAFE
            throw bad_argument_type_exception();
#else
        break;
#endif
    }

    newArgument(name, type, exttype);
}

Subregister InterfaceHandler::getArgumentIfExists(const std::string &name) const
{
    for (auto &assignment : assignments) {
        if (assignment.name == name)
            return assignment.reg;
    }

    return Subregister{};
}

Subregister InterfaceHandler::getArgument(const std::string &name) const
{
    Subregister arg = getArgumentIfExists(name);

#ifdef NGEN_SAFE
    if (arg.isInvalid())
        throw unknown_argument_exception();
#endif

    return arg;
}

int InterfaceHandler::getArgumentSurface(const std::string &name) const
{
    for (auto &assignment : assignments) {
        if (assignment.name == name) {
#ifdef NGEN_SAFE
            if (assignment.exttype != ExternalArgumentType::GlobalPtr)
                throw unknown_argument_exception();
#endif

            return assignment.surface;
        }
    }

#ifdef NGEN_SAFE
    throw unknown_argument_exception();
#else
    return 0x80;
#endif
}

GRF InterfaceHandler::getLocalID(int dim) const
{
#ifdef NGEN_SAFE
    if (dim > needLocalID) throw unknown_argument_exception();
#endif

    return GRF(1 + dim * grfsPerLID()).uw();
}

void InterfaceHandler::requireType(DataType type)
{
    switch (type) {
        case DataType::hf: needHalf = true;   break;
        case DataType::df: needDouble = true; break;
        default: break;
    }
}

static inline const char *getCLDataType(DataType type)
{
    static const char *names[16] = {"uint", "int", "ushort", "short", "uchar", "char", "double", "float", "ulong", "long", "half", "ushort", "INVALID", "INVALID", "INVALID", "INVALID"};
    return names[static_cast<uint8_t>(type) & 0xF];
}

void InterfaceHandler::generateDummyCL(std::ostream &stream) const
{
#ifdef NGEN_SAFE
    if (!finalized) throw interface_not_finalized();
#endif

    if (needHalf)   stream << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
    if (needDouble) stream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

    if (wg[0] > 0 && wg[1] > 0 && wg[2] > 0)
        stream << "__attribute__((reqd_work_group_size(" << wg[0] << ',' << wg[1] << ',' << wg[2] << ")))\n";
    stream << "__attribute__((intel_reqd_sub_group_size(" << simd << ")))\n";
    stream << "kernel void " << kernelName << '(';

    bool firstArg = true;
    for (const auto &assignment : assignments) {
        if (assignment.exttype == ExternalArgumentType::Hidden) continue;

        if (!firstArg) stream << ", ";

        switch (assignment.exttype) {
            case ExternalArgumentType::GlobalPtr: stream << "global void *"; break;
            case ExternalArgumentType::LocalPtr: stream << "local void *"; break;
            case ExternalArgumentType::Scalar: stream << getCLDataType(assignment.type) << ' '; break;
            default: break;
        }

        stream << assignment.name;
        firstArg = false;
    }
    stream << ") {\n";
    stream << "    global volatile int *____;\n";

    if (needLocalID)        stream << "    (void) ____[get_local_id(0)];\n";
    if (needLocalSize)      stream << "    (void) ____[get_enqueued_local_size(0)];\n";
    if (barrierCount > 0)   stream << "    barrier(CLK_GLOBAL_MEM_FENCE);\n";
    if (scratchSize > 0)    stream << "    volatile char scratch[" << scratchSize << "] = {0};\n";
    if (slmSize > 0)        stream << "    volatile local char slm[" << slmSize << "]; slm[0]++;\n";
    if (needNoPreemption) {
        if (hw == HW::Gen9)
            stream << "    volatile double *__df; *__df = 1.1 / *__df;\n"; // IEEE macro causes IGC to disable MTP.
        /* To do: Gen11 */
    }

    stream << "}\n";
}

inline Subregister InterfaceHandler::getLocalSize(int dim) const
{
    static const std::string localSizeArgs[3] = {"__local_size0", "__local_size1", "__local_size2"};
    return getArgument(localSizeArgs[dim]);
}

void InterfaceHandler::finalize()
{
    // Make assignments, following NEO rules:
    //  - all inputs are naturally aligned
    //  - all sub-DWord inputs are DWord-aligned
    //  - first register is
    //      r3 (no local IDs)
    //      r5 (SIMD8/16, local IDs)
    //      r8 (SIMD32, local IDs)
    // [- assign local ptr arguments left-to-right? not checked]
    //  - assign global pointer arguments left-to-right
    //  - assign scalar arguments left-to-right
    //  - assign surface indices left-to-right for global pointers
    //  - no arguments can cross a GRF boundary. Arrays like work size count
    //     as 1 argument for this rule.

    static const std::string localSizeArgs[3] = {"__local_size0", "__local_size1", "__local_size2"};
    static const std::string scratchSizeArg = "__scratch_size";

    GRF base = getCrossthreadBase();
    int offset = 32;
    int nextSurface = 0;
    const int grfSize = GRF::bytes(hw);

    auto assignArgsOfType = [&](ExternalArgumentType exttype) {
        for (auto &assignment : assignments) {
            if (assignment.exttype != exttype) continue;

            auto bytes = getBytes(assignment.type);
            auto size = getDwords(assignment.type) << 2;

            if (assignment.name == localSizeArgs[0]) {
                // Move to next GRF if local size arguments won't fit in this one.
                if (offset > grfSize - (3 * 4)) {
                    offset = 0;
                    base++;
                }
            }

            offset = (offset + size - 1) & -size;
            if (offset >= grfSize) {
                offset = 0;
                base++;
            }

            assignment.reg = base.sub(offset / bytes, assignment.type);

            if (assignment.exttype == ExternalArgumentType::GlobalPtr)
                assignment.surface = nextSurface++;
            else if (assignment.exttype == ExternalArgumentType::Scalar)
                requireType(assignment.type);

            offset += size;
        }
    };

    assignArgsOfType(ExternalArgumentType::LocalPtr);
    assignArgsOfType(ExternalArgumentType::GlobalPtr);
    assignArgsOfType(ExternalArgumentType::Scalar);

    // Add private memory size arguments.
    if (scratchSize > 0)
        newArgument(scratchSizeArg, DataType::uq, ExternalArgumentType::Hidden);

    // Add enqueued local size arguments.
    if (needLocalSize && needNonuniformWGs)
        for (int dim = 0; dim < 3; dim++)
            newArgument(localSizeArgs[dim], DataType::ud, ExternalArgumentType::Hidden);

    assignArgsOfType(ExternalArgumentType::Hidden);

    crossthreadGRFs = base.getBase() - getCrossthreadBase().getBase() + 1;

    // Manually add regular local size arguments.
    if (needLocalSize && !needNonuniformWGs)
        for (int dim = 0; dim < 3; dim++)
            assignments.push_back({localSizeArgs[dim], DataType::ud, ExternalArgumentType::Hidden,
                                   GRF(getCrossthreadBase()).ud(dim + 3), -1, -1});

    finalized = true;
}

GRF InterfaceHandler::getCrossthreadBase(bool effective) const
{
    if (!needLocalID)
        return GRF(effective ? 2 : 1);
    else
        return GRF(1 + 3 * grfsPerLID());
}

int InterfaceHandler::getCrossthreadGRFs() const
{
#ifdef NGEN_SAFE
    if (!finalized) throw interface_not_finalized();
#endif
    return crossthreadGRFs;
}

std::string InterfaceHandler::generateZeInfo() const
{
#ifdef NGEN_SAFE
    if (!finalized) throw interface_not_finalized();
#endif

    std::stringstream md;

    md << "kernels : \n"
          "  - name : \"" << kernelName << "\"\n"
          "    execution_env : \n"
          "      grf_count : " << needGRF << "\n"
          "      simd_size : " << simd << "\n";
    if (simd > 1)
        md << "      required_sub_group_size : " << simd << "\n";
    md << "      actual_kernel_start_offset : " << offsetSkipCrossThread << '\n';
    if (barrierCount > 0)
        md << "      barrier_count : " << barrierCount << '\n';
    if (allow64BitBuffers)
        md << "      has_4gb_buffers : true\n";
    if (slmSize > 0)
        md << "      slm_size : " << slmSize << '\n';
    if (!needStatelessWrites)
        md << "      has_no_stateless_write : true\n";
    if (needNoPreemption)
        md << "      disable_mid_thread_preemption : true\n";
    md << "\n";
    md << "    payload_arguments : \n";
    for (auto &assignment : assignments) {
        uint32_t size = 0;
        bool skipArg = false;
        bool explicitArg = true;
        switch (assignment.exttype) {
            case ExternalArgumentType::Scalar:
                md << "      - arg_type : arg_byvalue\n";
                size = (assignment.reg.getDwords() << 2);
                break;
            case ExternalArgumentType::LocalPtr:
            case ExternalArgumentType::GlobalPtr:
                md << "      - arg_type : arg_bypointer\n";
                size = (assignment.reg.getDwords() << 2);
                break;
            case ExternalArgumentType::Hidden: {
                explicitArg = false;
                if (assignment.name == "__local_size0") {
                    // from Zebin spec : local_size Argument size : int32x3
                    // may need refining to allow
                    // either int32x1, int32x2, int32x3 (x, xy, xyz)
                    // or fine grain : local_size_x, local_size_y, local_size_z
                    md << "      - arg_type : "
                       << (needNonuniformWGs ? "enqueued_local_size\n" : "local_size\n");
                    size = (assignment.reg.getDwords() << 2) * 3;
                } else
                    skipArg = true;
                break;
            }
        }
        if (skipArg)
            continue;

        auto offset = (assignment.reg.getBase() - getCrossthreadBase().getBase()) * 32 + assignment.reg.getByteOffset();
        if (explicitArg)
            md << "        arg_index : " << assignment.index << "\n";
        md << "        offset : " << offset << "\n"
              "        size : " << size << '\n';

        if (assignment.exttype == ExternalArgumentType::GlobalPtr) {
            md << "        addrmode : stateless\n"
                  "        addrspace : global\n"
                  "        access_type : readwrite\n";
        } else if (assignment.exttype == ExternalArgumentType::LocalPtr) {
            md << "        addrmode : slm\n"
                  "        addrspace : local\n"
                  "        access_type : readwrite\n";
        }
        md << "\n";

        if (assignment.exttype == ExternalArgumentType::GlobalPtr) {
            md << "      - arg_type : arg_bypointer\n"
                  "        arg_index : " << assignment.index << "\n"
                  "        offset : 0\n"
                  "        size : 0\n"
                  "        addrmode : stateful\n"
                  "        addrspace : global\n"
                  "        access_type : readwrite\n"
                  "\n";
        }
    }

    md << "\n";
    md << "    binding_table_indices : \n";

    for (auto &assignment : assignments) {
        if (assignment.exttype == ExternalArgumentType::GlobalPtr
                || assignment.exttype == ExternalArgumentType::LocalPtr) {
            md << "      - bti_value : " << assignment.surface << "\n"
                  "        arg_index : " << assignment.index << "\n"
                  " \n";
        }
    }

    md << "\n";
    md << "    per_thread_payload_arguments : \n";

    if (needLocalID) {
        auto localIDBytes = grfsPerLID() * GRF::bytes(hw);
        localIDBytes *= sizeof(short);
        localIDBytes *= 3; // runtime currently supports 0 or 3 localId channels in per thread data
        md << "      - arg_type : local_id\n"
              "        offset : 0\n"
              "        size : " << localIDBytes << "\n"
              "  \n";
    }

    md << "\n"; // ensure file ends with newline

    return md.str();
}

#ifdef NGEN_ASM
void InterfaceHandler::dumpAssignments(std::ostream &stream) const
{
    LabelManager manager;

    for (auto &assignment : assignments) {
        stream << "//  ";
        assignment.reg.outputText(stream, PrintDetail::sub, manager);
        stream << '\t' << assignment.name << std::endl;
    }
}
#endif

} /* namespace ngen */

#endif /* header guard */

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

#include "protocol.hpp"

#include <stdexcept>
namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace micro {

[[noreturn]] void unknownProtocol() {
    throw std::runtime_error("Unknown protocol");
}

GEMMProtocol::GEMMProtocol(const Options &options) {
    family = Family::GEMM;
    ioptions = options.toOptionsMask();
}

GEMMProtocol::Options GEMMProtocol::options() const {
    return Options(ioptions);
}

#define PDISPATCH(routine, cand) \
    if (family == Family::cand) \
    return reinterpret_cast<const cand##Protocol *>(this)->routine()

const char *Protocol::kernelBaseName() const {
    PDISPATCH(kernelBaseName, GEMM);
    unknownProtocol();
}

std::vector<ProtocolArgument> Protocol::arguments() const {
    PDISPATCH(arguments, GEMM);
    unknownProtocol();
}

std::vector<ProtocolSetting> Protocol::settings() const {
    PDISPATCH(settings, GEMM);
    unknownProtocol();
}

#undef PDISPATCH

const char *GEMMProtocol::kernelBaseName() const {
    return "ugemm";
}

std::vector<ProtocolArgument> GEMMProtocol::arguments() const {
    auto In = ProtocolArgument::In;
    auto Out = ProtocolArgument::Out;

    auto LocalPointer = StructuredType::LocalPointer;
    auto GlobalPointer = StructuredType::GlobalPointer;
    auto s32 = StructuredType::s32;

    static ProtocolArgument args[] = {
            {"a", In, GlobalPointer},
            {"lda", In, s32},
            {"b", In, GlobalPointer},
            {"ldb", In, s32},
            {"c", Out, 2},
            {"m", In, s32},
            {"n", In, s32},
            {"k", In, s32},
            {"i0", In, s32},
            {"j0", In, s32},
            {"h0", In, s32},
            {"local_id_m", In, s32},
            {"local_id_n", In, s32},
    };
    std::vector<ProtocolArgument> argsV
            = {args, args + sizeof(args) / sizeof(args[0])};

    auto o = options();

    if (o.localA) argsV[0].stype.format = LocalPointer;
    if (o.localB) argsV[2].stype.format = LocalPointer;
    if (o.addToC) argsV[4].direction = ProtocolArgument::InOut;
    if (o.slmPtr) argsV.push_back({"slm", In, LocalPointer});
    if (o.scaleA) argsV.push_back({"a_scale", In, GlobalPointer});
    if (o.offsetA) argsV.push_back({"a_offset", In, GlobalPointer});
    if (o.offsetA || o.scaleA) argsV.push_back({"ldaq", In, s32});
    if (o.scaleB) argsV.push_back({"b_scale", In, GlobalPointer});
    if (o.offsetB) argsV.push_back({"b_offset", In, GlobalPointer});
    if (o.offsetB || o.scaleB) { argsV.push_back({"ldbq", In, s32}); }

    return argsV;
}

std::vector<ProtocolSetting> GEMMProtocol::settings() const {
    static ProtocolSetting settings[] = {
            {"sg_tile_m"},
            {"sg_tile_n"},
            {"wg_tile_m"},
            {"wg_tile_n"},
            {"sg_per_wg_m"},
            {"sg_per_wg_n"},
            {"sg_per_wg_k"},
            {"slm_size"},
    };
    static std::vector<ProtocolSetting> settingsV
            = {settings, settings + sizeof(settings) / sizeof(settings[0])};
    return settingsV;
}

} /* namespace micro */
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

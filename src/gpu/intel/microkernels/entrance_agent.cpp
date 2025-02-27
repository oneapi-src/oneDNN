/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "entrance_agent.hpp"

#include <array>

#include "ngen_config.hpp"
#include "ngen_decoder.hpp"
#include "npack/neo_packager.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace micro {

EntranceAgent::Status EntranceAgent::scan(Package &package) {
    using namespace ngen;

    auto status = Status::Success;

    auto product = npack::decodeHWIPVersion(package.gmdidCompat);
    auto hw = getCore(product.family);

    if (hw == HW::Unknown || hw < HW::Gen12LP) return Status::UnsupportedHW;

    Decoder decoder(hw, package.binary);
    DependencyRegion dstRegion;

    /* Track clobbered registers at full register granularity for simplicity. */
    std::array<bool, GRF::maxRegs() + 1> clobbered = {false};

    for (; !decoder.done(); decoder.advance()) {
        // Check for systolic usage.
        auto op = decoder.opcode();
        package.systolic |= (op == Opcode::dpas || op == Opcode::dpasw);

        // Get destination region and add to clobbers.
        if (decoder.getOperandRegion(dstRegion, -1)) {
            if (dstRegion.unspecified) {
                // Indirect destination -- cannot reliably detect clobbers.
                status = Status::UncertainClobbers;
            } else
                for (int j = 0; j < dstRegion.size; j++)
                    clobbered[dstRegion.base + j] = true;
        }
    }

    // Group clobber array into consecutive ranges.
    package.clobbers.clear();

    int regBytes = GRF::bytes(hw);
    int base = 0, len = 0;
    for (int j = 0; j < int(clobbered.size()); j++) {
        if (clobbered[j]) {
            if (len > 0)
                len++;
            else
                base = j, len = 1;
        } else if (len > 0) {
            package.clobbers.emplace_back(
                    RegisterRange(base * regBytes, len * regBytes));
            len = 0;
        }
    }

    // Capture GRF usage from clobbers and arguments.
    uint32_t last = 0;
    if (!package.clobbers.empty()) {
        auto &final = package.clobbers.back();
        last = final.boffset + final.blen;
    }
    for (const auto &argument : package.arguments)
        for (auto &range : argument.location)
            last = std::max(last, range.boffset + range.blen);

    package.grfMin = (last + regBytes - 1) / regBytes;

    // Generate LUID from hash of kernel. Later, the cataloguer can update it in case of collisions.
    uint32_t luid = 0;
    uint32_t multiplier = 1357;

    auto *u32ptr = (const uint32_t *)package.binary.data();
    for (size_t i = 0; i < (package.binary.size() >> 2); i++) {
        luid ^= u32ptr[i] * multiplier;
        multiplier += 2;
        luid = (luid << 3) | (luid >> 29);
    }

    package.luid = luid;

    return status;
}

} /* namespace micro */
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

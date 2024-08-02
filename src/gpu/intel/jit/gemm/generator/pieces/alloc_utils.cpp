/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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


#include "alloc_utils.hpp"

using namespace ngen;
using std::vector;

#include "hw_utils.hpp"
#include "layout_utils.hpp"

#include "internal/namespace_start.hxx"


GRFMultirange tryChunkAlloc(int nreg, int chunk, Bundle hint, BundleGroup mask, CommonState &state)
{
    GRFMultirange r;
    bool ok = true;
    for (; nreg > 0; nreg -= chunk) {
        auto nr = std::min(nreg, chunk);
        auto rr = state.ra.tryAllocRange(nr, hint, mask);
        if (rr.isInvalid()) {
            ok = false; break;
        }
        r.ranges.push_back(rr);
    }
    if (!ok)
        safeReleaseRanges(r, state);
    return r;
}

GRFMultirange chunkAlloc(int nreg, int chunk, Bundle hint, BundleGroup mask, CommonState &state)
{
    auto r = tryChunkAlloc(nreg, chunk, hint, mask, state);
    if (r.empty() && nreg > 0)
        throw out_of_registers_exception();
    return r;
}

GRFMultirange trySplitAlloc(HW hw, Type T, const vector<RegisterBlock> &layout, std::array<Bundle, 2> hints,
                            BundleGroup mask, CommonState &state, int copies)
{
    auto oddHint = Bundle(0, 0).group_size(hw) * elementsPerGRF(hw, T);

    GRFMultirange r;
    struct Request {
        int length, offset, index, hint;
    };
    vector<Request> requests;
    requests.reserve(layout.size());

    for (auto &block: layout) {
        if (block.isLoadBlock()) {
            int hint = ((block.colMajor ? block.offsetR : block.offsetC) & oddHint) != 0;
            requests.push_back({block.msgRegs, block.offsetReg(), 0, hint});
        }
    }

    if (requests.empty() && !layout.empty()) for (auto &block: layout) {
        // No memory backing for layout. Split by rows/columns if possible.
        int hint = ((block.colMajor ? block.offsetR : block.offsetC) & oddHint) != 0;
        auto &ny = block.colMajor ? block.nc : block.nr;
        int xElems = block.ld * block.crosspack;
        int xGRFs = xElems / elementsPerGRF(hw, T);
        if (xElems % elementsPerGRF(hw, T))
            requests.push_back({block.nregs(), block.offsetReg(), 0, hint});    /* can't split */
        else for (int y = 0, off = block.offsetReg(); y < ny; y += block.crosspack, off += xGRFs)
            requests.push_back({xGRFs, off, 0, hint});
    }

    // Figure out which order the ranges belong in.
    std::sort(requests.begin(), requests.end(), [](const Request &r1, const Request &r2) {
        return (r1.offset < r2.offset);
    });
    for (size_t i = 0; i < requests.size(); i++)
        requests[i].index = int(i);

    // Sort again and allocate largest to smallest.
    std::sort(requests.begin(), requests.end(), [](const Request &r1, const Request &r2) {
        return (r1.length > r2.length) ||
               (r1.length == r2.length && r1.offset < r2.offset);
    });
    r.ranges.resize(requests.size() * copies);

    bool ok = true;
    for (size_t i = 0; i < requests.size(); i++) {
        for (int c = 0; c < copies; c++) {
            auto newRange = state.ra.try_alloc_range(requests[i].length, hints[requests[i].hint], mask);
            r.ranges[requests[i].index + c * requests.size()] = newRange;
            ok &= newRange.isValid();
        }
    }

    if (!ok) {
        for (auto &rr: r.ranges)
            state.ra.release(rr);
        r.ranges.clear();
    }

    return r;
}

GRFMultirange splitOrChunkAlloc(HW hw, Type T, const vector<RegisterBlock> &layout, int chunk, std::array<Bundle, 2> hints,
                                BundleGroup mask, CommonState &state, bool forceChunk)
{
    if (!forceChunk) {
        auto r = trySplitAlloc(hw, T, layout, hints, mask, state);
        if (!r.empty())
            return r;
    }
    return chunkAlloc(getRegCount(layout), chunk, hints[0], mask, state);
}

#include "internal/namespace_end.hxx"

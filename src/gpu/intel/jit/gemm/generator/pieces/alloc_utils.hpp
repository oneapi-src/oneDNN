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


#ifndef GEMMSTONE_GUARD_ALLOC_UTILS_HPP
#define GEMMSTONE_GUARD_ALLOC_UTILS_HPP

#include "internal/ngen_includes.hpp"
#include "types.hpp"
#include "problem.hpp"
#include "strategy.hpp"
#include "state.hpp"

#include "internal/namespace_start.hxx"



static inline void safeRelease(SubregisterPair &pair, CommonState &state) {
    state.ra.release(pair.getReg(0));
    state.ra.release(pair.getReg(1));
    pair.invalidate();
}

static inline void safeReleaseRanges(std::vector<ngen::GRFRange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        state.ra.safeRelease(a);
    ranges.clear();
}

static inline void safeReleaseRanges(GRFMultirange &ranges, CommonState &state) {
    safeReleaseRanges(ranges.ranges, state);
    ranges.ranges.clear();
}

static inline void safeReleaseRanges(std::vector<GRFMultirange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        safeReleaseRanges(a, state);
    ranges.clear();
}

static inline void releaseRanges(const std::vector<ngen::GRFRange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        state.ra.release(a);
}

static inline void releaseRanges(const GRFMultirange &ranges, CommonState &state) {
    releaseRanges(ranges.ranges, state);
}

static inline void releaseRanges(const std::vector<GRFMultirange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        releaseRanges(a, state);
}

static inline void reclaimRanges(const std::vector<ngen::GRFRange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        state.ra.claim(a);
}

static inline void reclaimRanges(const GRFMultirange &ranges, CommonState &state) {
    reclaimRanges(ranges.ranges, state);
}

// Reclaim a list of GRF multiranges.
static inline void reclaimRanges(const std::vector<GRFMultirange> &ranges, CommonState &state) {
    for (auto &a : ranges)
        reclaimRanges(a, state);
}


// Allocate nreg registers in chunks of a fixed size `chunk`.
GRFMultirange chunkAlloc(int nreg, int chunk, ngen::Bundle hint, ngen::BundleGroup mask, CommonState &state);

static inline GRFMultirange chunkAlloc(int nreg, int chunk, ngen::Bundle hint, CommonState &state) {
    return chunkAlloc(nreg, chunk, hint, ngen::BundleGroup::AllBundles(), state);
}

static inline GRFMultirange chunkAlloc(int nreg, int chunk, CommonState &state) {
    return chunkAlloc(nreg, chunk, ngen::Bundle(), state);
}

// Like chunkAlloc, but returns an empty GRFMultirange on failure instead of throwing.
GRFMultirange tryChunkAlloc(int nreg, int chunk, ngen::Bundle hint, ngen::BundleGroup mask, CommonState &state);

// Attempt to allocate data registers for a layout, using one contiguous allocation per block.
// Returns an empty GRFMultirange on failure.
GRFMultirange trySplitAlloc(ngen::HW hw, Type T, const std::vector<RegisterBlock> &layout, std::array<ngen::Bundle, 2> hints,
                            ngen::BundleGroup mask, CommonState &state, int copies = 1);

// Split allocate if possible, otherwise chunk allocate.
GRFMultirange splitOrChunkAlloc(ngen::HW hw, Type T, const std::vector<RegisterBlock> &layout, int chunk, std::array<ngen::Bundle, 2> hints,
                                ngen::BundleGroup mask, CommonState &state, bool forceChunk = false);

#include "internal/namespace_end.hxx"

#endif /* header guard */

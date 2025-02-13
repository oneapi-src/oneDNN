/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GEMMSTONE_GUARD_DRIVER_INFO_HPP
#define GEMMSTONE_GUARD_DRIVER_INFO_HPP

#include <cstdint>
#include <string>

#include "config.hpp"

#include "internal/namespace_start.hxx"

// Loop identifiers.
enum LoopType : uint8_t {
    LoopM = 0,
    LoopN = 1,
    LoopK = 2,
    LoopPersistent = 0x40,          // Flag OR'ed with other loop types, indicating persistent threads.
    LoopMNBoustrophedonMNK = 0x80,  // Fused m/n indices (boustrophedon ordering), with MNK nested inside
    LoopMNBoustrophedonNMK = 0x81,  // Fused n/m indices (boustrophedon ordering), with NMK nested inside
    LoopMNHilbertMNK = 0x90,        // Fused m/n indices (Hilbert ordering), with MNK nested inside
    LoopMNHilbertNMK = 0x91,        // Fused n/m indices (Hilbert ordering), with NMK nested inside
    LoopMNLinearMNK = 0xA0,         // Fused m/n indices (simple linear ordering), with MNK nested inside
    LoopMNLinearNMK = 0xA1,         // Fused n/m indices (simple linear ordering), with NMK nested inside
    LoopMNNestedLinearMNK = 0xB0,   // Fused m/n indices (nested linear ordering), with MNK nested inside
    LoopMNNestedLinearNMK = 0xB1,   // Fused n/m indices (nested linear ordering), with NMK nested inside
    LoopAny = 0xFF,
    LoopNone = 0xFF
};

// Workgroup sizing options.
enum WGType : uint8_t {
    WGDynamic = 0,                  // Dynamic m/n work group size (can shrink or expand)
    WGFixed = 1,                    // Fixed m/n work group size
    WGShrinkable = 2,               // m/n work group size can shrink but not expand
};

// Flags.
enum DriverInfoFlags : uint32_t {
    FlagKRemainderHandling = 1,     // GEMM kernel performs k remainder handling
    FlagKParallel = 2,              // GEMM kernel is parallelized in the k dimension.
    FlagZParallel = 2,              // Copy kernel is parallelized in the z dimension.
    FlagKParallelLocal = 4,         // GEMM kernel is parallelized in the k dimension inside a workgroup.
    FlagKParallelVariable = 8,      // GEMM kernel uses variable k-parallelization (see GEMMStrategy::kParallelVariable).
    FlagFusedBeta = 0x10,           // GEMM kernel does fused beta scaling + atomics.
    FlagFusedPostOps = 0x20,        // GEMM kernel does fused atomics + post-ops.
    FlagTempC = 0x40,               // GEMM kernel needs temporary C buffer.
    FlagAltFusedBeta = 0x80,        // GEMM kernel uses alternate fused beta scaling + atomics logic.
    FlagAutoAtomic = 0x100,         // GEMM kernel may use atomic C accesses automatically for beta = 1.
    FlagShrinkWGK = 0x200,          // With local k-parallelization, automatically shrink wgK to fit dispatch to GPU.
    FlagAlphaPtr = 0x400,           // Pass alpha by pointer.
    FlagBetaPtr = 0x800,            // Pass beta by pointer.
    FlagFixedWGK = 0x1000,          // With local k-parallelization, wgK is fixed
    FlagZeroTempC = 0x2000,         // GEMM kernel expects temporary C buffer to be pre-zeroed and will leave zeros there.
    FlagNondeterministic = 0x4000,  // Kernel produces nondeterministic results.
    FlagMaskFillGoal = 0xF0000,     // Fraction of available thread slots to fill, in sixteenths
    FlagShiftFillGoal = 16,         //   (starting bit)
};

// Driver information, shared by all kernel types.
struct CommonDriverInfo {
    int subgroupSize;               // Declared subgroup size (unrelated to actual SIMD lengths in kernel)
    LoopType fusedLoop;             // Loop dimension in which EUs are fused (if any).
    int grfCount;                   // # of GRFs used by kernel.
    LoopType loopOrder[3];          // Loops corresponding to x/y/z dimensions of kernel dispatch.
    int blocking[3];                // Standard blocking sizes in m/n/k dimensions.
    int blockingAlt[3];             // Alternative blocking sizes in m/n/k dimensions.
    int unroll[3];                  // m/n/k unrolls.
    int wg[3];                      // HW threads per workgroup in m/n/k dimensions.
    int wgExpand;                   // If > 1, workgroup size needs to be scaled by this factor.
    WGType wgUpdate;                // Work group type showing how/if work group sizes can be updated.
    uint32_t flags;                 // Bitfield with additional boolean kernel attributes (see DriverInfoFlags enum).
    int slm;                        // Minimum SLM allocation.
    int perKSLM;                    // If > 0, dynamically allocate at least perKSLM * wg[LoopK] bytes of SLM.
    int alignment[3];               // Address alignment requirements for A,B,C (gemm) or S,D (copy).
    bool support4GB[3];             // True if >4GB buffers allowed for A,B,C (gemm) or S,D (copy).

    bool fusedEUs()           const { return (fusedLoop != LoopNone); }
    bool isMNK()              const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopM || l == LoopMNHilbertMNK || l == LoopMNBoustrophedonMNK || l == LoopMNLinearMNK || l == LoopMNNestedLinearMNK; }
    bool isNMK()              const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopN || l == LoopMNHilbertNMK || l == LoopMNBoustrophedonNMK || l == LoopMNLinearNMK || l == LoopMNNestedLinearNMK; }
    bool isHilbert()          const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopMNHilbertMNK || l == LoopMNHilbertNMK; }
    bool isBoustrophedon()    const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopMNBoustrophedonMNK || l == LoopMNBoustrophedonNMK; }
    bool isSimpleLinear()     const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopMNLinearMNK || l == LoopMNLinearNMK; }
    bool isNestedLinear()     const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopMNNestedLinearMNK || l == LoopMNNestedLinearNMK; }
    bool isLinearOrder()      const { return (loopOrder[0] != LoopNone) && (loopOrder[0] & 0x80); }
    bool isPersistent()       const { return (loopOrder[0] != LoopNone) && (loopOrder[0] & LoopPersistent); }
    bool fixedWG()            const { return wgUpdate == WGFixed; }
    int  threadsPerWG()       const { return wg[LoopM] * wg[LoopN] * wg[LoopK] * wgExpand; }
    bool kRemainderHandling() const { return flags & FlagKRemainderHandling; }
    bool kParallel()          const { return flags & FlagKParallel; }
    bool zParallel()          const { return flags & FlagZParallel; }
    bool kParallelLocal()     const { return flags & FlagKParallelLocal; }
    bool kParallelVariable()  const { return flags & FlagKParallelVariable; }
    bool fusedBeta()          const { return flags & FlagFusedBeta; }
    bool fusedPostOps()       const { return flags & FlagFusedPostOps; }
    bool needsTempC()         const { return flags & FlagTempC; }
    bool zeroTempC()          const { return flags & FlagZeroTempC; }
    bool altFusedBeta()       const { return flags & FlagAltFusedBeta; }
    bool mayUseAutoAtomic()   const { return flags & FlagAutoAtomic; }
    bool shrinkWGK()          const { return flags & FlagShrinkWGK; }
    bool alphaPtr()           const { return flags & FlagAlphaPtr; }
    bool betaPtr()            const { return flags & FlagBetaPtr; }
    bool fixedWGK()           const { return flags & FlagFixedWGK; }
    bool nondeterministic()   const { return flags & FlagNondeterministic; }

    int wgTile(LoopType l)    const { return unroll[l] * wg[l]; }
    int kPadding()            const { return (kParallel() || kParallelVariable()) ? blockingAlt[LoopK] : 0; }

    float fillGoal() const {
        auto sixteenths = (flags & FlagMaskFillGoal) >> FlagShiftFillGoal;
        return (sixteenths > 0) ? (sixteenths / 16.0f) : 1.0f;
    }

};

// Definitions for flag arguments to kernels.
enum {
    FlagCOColumn = 0x4,
    FlagCORow = 0x8,
    FlagNonfinalKBlock = 0x10,
    FlagNoninitialKBlock = 0x80,
    FlagLateFusedGEMMDone = 0x100,
    FlagEarlyFusedGEMMDone = 0x200,
    FlagStoreSums = 0x400,
    FlagKSlicing = 0x1000,
    FlagLeader = 0x2000,
    FlagKPartitioned = 0x4000,
    FlagDidBeta = 0x100,
    FlagSkipBetaCheck = 0x200,
    FlagKSlice2 = 0x10000,
};

#include "internal/namespace_end.hxx"

#endif /* header guard */

/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef GPU_JIT_GEMM_GEN_GEMM_KERNEL_COMMON_HPP
#define GPU_JIT_GEMM_GEN_GEMM_KERNEL_COMMON_HPP

#define STANDALONE 0

#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Loop identifiers.
enum LoopType : uint8_t {
    LoopM = 0,
    LoopN = 1,
    LoopK = 2,
    LoopPersistent
    = 0x40, // Flag OR'ed with other loop types, indicating persistent threads.
    LoopMNBoustrophedonMNK
    = 0x80, // Fused m/n indices (boustrophedon ordering), with MNK nested inside
    LoopMNBoustrophedonNMK
    = 0x81, // Fused n/m indices (boustrophedon ordering), with NMK nested inside
    LoopMNHilbertMNK
    = 0x90, // Fused m/n indices (Hilbert ordering), with MNK nested inside
    LoopMNHilbertNMK
    = 0x91, // Fused n/m indices (Hilbert ordering), with NMK nested inside
    LoopAny = 0xFF,
    LoopNone = 0xFF
};

// WG identifiers.
enum WGType : uint8_t {
    WGDynamic = 0, // Dynamic work group size (can shrink or expand)
    WGFixed = 1, // Fixed work group size
    WGShrinkable = 2 // Work group size can shrink but not expand
};

// Driver information, shared by all kernel types.
struct CommonDriverInfo {
    int subgroupSize; // Declared subgroup size (unrelated to actual SIMD lengths in kernel)
    LoopType fusedLoop; // Loop dimension in which EUs are fused (if any).
    int grfCount; // # of GRFs used by kernel.
    LoopType loopOrder
            [3]; // Loops corresponding to x/y/z dimensions of kernel dispatch.
    int blocking[3]; // Standard blocking sizes in m/n/k dimensions.
    int blockingAlt[3]; // Alternative blocking sizes in m/n/k dimensions.
    int unroll[3]; // m/n/k unrolls.
    int wg[3]; // HW threads per workgroup in m/n/k dimensions.
    int wgExpand; // If > 1, workgroup size needs to be scaled by this factor.
    WGType wgUpdate; // Work group type showing how/if work group sizes can be updated.
    bool kRemainderHandling; // True if kernel performs k remainder handling (gemm).
    bool kParallel; // True if gemm kernel can be parallelized in the k dimension.
    bool kParallelLocal; // True if gemm kernel can be parallelized in the k dimension inside a workgroup.
    int slm; // Minimum SLM allocation.
    int perKSLM; // If > 0, dynamically allocate at least perKSLM * wg[LoopK] bytes of SLM.
    int alignment
            [3]; // Address alignment requirements for A,B,C (gemm) or S,D (copy).
    bool support4GB
            [3]; // True if >4GB buffers allowed for A,B,C (gemm) or S,D (copy).

    bool fusedEUs() const { return (fusedLoop != LoopNone); }
    bool isMNK() const {
        auto l = loopOrder[0] & ~LoopPersistent;
        return l == LoopM || l == LoopMNHilbertMNK
                || l == LoopMNBoustrophedonMNK;
    }
    bool isNMK() const {
        auto l = loopOrder[0] & ~LoopPersistent;
        return l == LoopN || l == LoopMNHilbertNMK
                || l == LoopMNBoustrophedonNMK;
    }
    bool isHilbert() const {
        auto l = loopOrder[0] & ~LoopPersistent;
        return l == LoopMNHilbertMNK || l == LoopMNHilbertNMK;
    }
    bool isBoustrophedon() const {
        auto l = loopOrder[0] & ~LoopPersistent;
        return l == LoopMNBoustrophedonMNK || l == LoopMNBoustrophedonNMK;
    }
    bool isLinearOrder() const { return isHilbert() || isBoustrophedon(); }
    bool isPersistent() const {
        return (loopOrder[0] != LoopNone) && (loopOrder[0] & LoopPersistent);
    }
    bool fixedWG() const { return wgUpdate == WGFixed; }

    int wgTile(LoopType l) const { return unroll[l] * wg[l]; }
};

// Definitions for flag arguments to kernels.
enum {
    FlagCOColumn = 4,
    FlagCORow = 8,
    FlagNonfinalKBlock = 16,
    FlagNoninitialKBlock = 128,
    FlagLateFusedGEMMDone = 256,
    FlagEarlyFusedGEMMDone = 512,
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* header guard */

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

#ifndef GPU_JIT_GEMM_GEN_GEMM_KERNEL_COMMON_HPP
#define GPU_JIT_GEMM_GEN_GEMM_KERNEL_COMMON_HPP

#include <string>

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

// Driver information, shared by all kernel types.
struct CommonDriverInfo {
    int subgroupSize
            = 0; // Declared subgroup size (unrelated to actual SIMD lengths in kernel)
    LoopType fusedLoop
            = LoopNone; // Loop dimension in which EUs are fused (if any).
    int grfCount = 128; // # of GRFs used by kernel.
    LoopType loopOrder[3] = {LoopNone, LoopNone,
            LoopNone}; // Loops corresponding to x/y/z dimensions of kernel dispatch.
    int blocking[3] = {0}; // Standard blocking sizes in m/n/k dimensions.
    int blockingAlt[3] = {0}; // Alternative blocking sizes in m/n/k dimensions.
    int unroll[3] = {0}; // m/n/k unrolls.
    int wg[3] = {1, 1, 1}; // HW threads per workgroup in m/n/k dimensions.
    int wgExpand
            = 1; // If > 1, workgroup size needs to be scaled by this factor.
    bool fixedWG
            = false; // True if m/n workgroup size is fixed; false if size may be reduced.
    bool kRemainderHandling
            = false; // True if kernel performs k remainder handling (gemm).
    bool kParallel
            = false; // True if gemm kernel can be parallelized in the k dimension.
    bool kParallelLocal
            = false; // True if gemm kernel can be parallelized in the k dimension inside a workgroup.
    int slm = 0; // Minimum SLM allocation.
    int perKSLM
            = 0; // If > 0, dynamically allocate at least perKSLM * wg[LoopK] bytes of SLM.
    int alignment[3] = {0, 0,
            0}; // Address alignment requirements for A,B,C (gemm) or S,D (copy).
    bool support4GB[3] = {
            false}; // True if >4GB buffers allowed for A,B,C (gemm) or S,D (copy).

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

#endif /* header guard */

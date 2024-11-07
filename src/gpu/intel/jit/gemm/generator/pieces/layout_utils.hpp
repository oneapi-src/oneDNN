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


#ifndef GEMMSTONE_GUARD_LAYOUT_UTILS_HPP
#define GEMMSTONE_GUARD_LAYOUT_UTILS_HPP

#include "internal/ngen_includes.hpp"
#include "type.hpp"
#include "problem.hpp"
#include "strategy.hpp"
#include "state.hpp"

#include "internal/namespace_start.hxx"


// Get an element's linear offset in a tiled layout (in registers or in memory).
int untile(Type T, const MatrixAddressing &atype, int component, int i, int j, int r, int c, int tileR, int tileC, bool reverse = false);

static inline int untile(Type T, const MatrixAddressing &atype, const RegisterBlock &block, int r, int c, int tileR, int tileC, bool reverse = false) {
    return untile(T, atype, block.component, block.offsetR, block.offsetC, r, c, tileR, tileC, reverse);
}

static inline int untile(Type T, const MatrixAddressing &atype, const RegisterBlock &block, int r = 0, int c = 0, bool reverse = false) {
    return untile(T, atype, block, r, c, atype.tileR, atype.tileC, reverse);
}

static inline int untile(Type T, const MatrixAddressing &atype, int component, int i, int j, int r = 0, int c = 0, bool reverse = false) {
    return untile(T, atype, component, i, j, r, c, atype.tileR, atype.tileC, reverse);
}

// Return the number of matrix elements in a tile of size (r,c) that are
//   guaranteed to be consecutive in memory.
int consecutiveElements(int r, int c, const MatrixAddressing &atype);

// Get minimum row/column granularity for a matrix in memory.
//   (i.e. the number of rows/columns must be a multiple of rgran/cgran, respectively.)
void getGranularities(const MatrixAddressing &atype, int &rgran, int &cgran);

// Check if a matrix will arrive column-major in registers, without creating a RegisterBlock.
static inline bool isRegisterColMajor(Type T, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) {
    return isColMajor(atype.layout) ^ isTransposing(astrategy.accessType) ^ isLargeCrosspack(T, atype.crosspack);
}

// Check if pseudo-block (rather than true block) access is required for this block.
bool needsPseudoblock(ngen::HW hw, Type T, int r, int c,
                      const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                      bool writable, bool masked);

// Check if pseudo-block access should use channel scattered access internally.
bool pseudoblockUseChannelScattered(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const RegisterBlock &block);

// Get effective access type to use when setting up addresses.
AccessType effectiveAccessType(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const RegisterBlock &block);

// Get effective access type to use when performing loads/stores.
AccessType implAccessType(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const RegisterBlock &block);

// Get width/height/array size parameters for underlying 2D block load message.
void getBlock2DWH(int &w, int &h, int &count, const MatrixAddressing &atype, const RegisterBlock &block, int *outMultiX = nullptr);

// Count the number of address/header GRFs required by a RegisterBlock.
int addrGRFCount(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, const RegisterBlock &block);

// Allocate address registers for a layout.
void allocAddrRegs(std::vector<ngen::GRFRange> &addrRegs, const std::vector<RegisterBlock> &layout,
                   const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                   CommonState &state, ngen::Bundle hint = ngen::Bundle());

// Attempt to allocate address registers for a layout. Returns true if successful.
bool tryAllocAddrRegs(std::vector<ngen::GRFRange> &addrRegs, const std::vector<RegisterBlock> &layout,
                      const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                      CommonState &state, ngen::Bundle hint = ngen::Bundle());

// Check if a layout is completely column-major.
bool isLayoutColMajor(const std::vector<RegisterBlock> &layout);

// Get the matrix size represented by a layout.
void getLayoutDims(const std::vector<RegisterBlock> &layout, int &m, int &n);

// Check if every block in a layout has the given crosspack, with no padding.
bool hasFullCrosspack(const std::vector<RegisterBlock> &layout, int crosspack);

// Check if the layout is tiled with the given tiling.
bool hasTiling(const std::vector<RegisterBlock> &layout, int tileR, int tileC);

// Check if a layout has remainders enabled.
bool hasRemainders(const std::vector<RegisterBlock> &layout, bool remainderR = true, bool remainderC = true);

// Check if a layout has any kind of fragmenting.
bool hasFragmenting(const std::vector<RegisterBlock> &layout, bool ignoreWholeFragR = false, bool ignoreWholeFragC = false);

// Check if a layout has any masking.
bool hasMasking(const std::vector<RegisterBlock> &layout);

// Check if a layout has any flag registers assigned.
bool hasFlags(const std::vector<RegisterBlock> &layout);

// Find the maximum block size in a layout, in registers.
int getMaxLoadBlock(const std::vector<RegisterBlock> &layout);

// Count the number of registers needed by a register layout.
int getRegCount(const std::vector<RegisterBlock> &layout);

// Find the subregister in a RegisterBlock corresponding to the element at offset (rr,cc),
//  as well as the contiguous elements following it (nelems).
ngen::Subregister findBlockReg(Type T, const RegisterBlock &block, int rr, int cc,
                               const GRFMultirange &regs, int &nelems,
                               int cxComponent = -1, int component = 0);

// Find the subregister in a layout corresponding to element (r,c), as well as the
//  associated block, and the number of contiguous elements following it (nelems).
ngen::Subregister findBlockReg(Type T, const std::vector<RegisterBlock> &layout, int r, int c,
                               const GRFMultirange &regs, int &nelems, const RegisterBlock *&block,
                               int cxComponent = -1, int component = 0);

// Similar to findBlockReg, but returns the region associated with consecutive elements in the block.
// If allow2D is true, the return value is allowed to be a true 2D region.
//   Otherwise, the return region will always be a constant stride (1D) region.
ngen::RegisterRegion findBlockRegion(Type T, const RegisterBlock &block, int rr, int cc,
                                     const GRFMultirange &regs, int &nelems,
                                     int cxComponent = -1, int component = 0, bool allow2D = false);
ngen::RegisterRegion findBlockRegion(Type T, const std::vector<RegisterBlock> &layout, int r, int c,
                                     const GRFMultirange &regs, int &nelems, const RegisterBlock *&block,
                                     int cxComponent = -1, int component = 0, bool allow2D = false);

// Find the subregister offset containing the first address of a header.
int getAddr0Offset(const RegisterBlock &block, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy);

// Get a subregister containing the (shifted) address of the (0,0) entry of a layout.
ngen::Subregister getOriginAddr(const std::vector<RegisterBlock> &layout, const std::vector<ngen::GRFRange> &addrRegs,
                                const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, int *shiftOut = nullptr);


// Check if a block occupies a contiguous portion of registers in the given GRFMultirange.
// If so, return index of the block's first register in the range.
int contiguityCheck(ngen::HW hw, const RegisterBlock &block, const GRFMultirange &range);

// Retrieve the subrange of a given GRFMultirange holding the matrix data from a given block.
GRFMultirange subrange(GRFMultirange r, ngen::HW hw, Type T, const RegisterBlock &block);

// Unlink a layout from its in-memory representation.
void unlinkFromMemory(std::vector<RegisterBlock> &layout);

// Re-order a layout so that registers appear in appropriate order (row or column major).
void sortRegLayout(Type T, std::vector<RegisterBlock> &layout, int r, int c,
                   const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool reverse = false);

// Match the register offsets in one register layout to another, reference layout.
// Returns true if successful. If not successful, the layout is unchanged.
bool matchLayouts(Type T, std::vector<RegisterBlock> &layout, const std::vector<RegisterBlock> &layoutRef);

// Like matchLayouts but allows either layout to change to match the other.
static inline bool matchLayoutsBidirectional(Type T, std::vector<RegisterBlock> &layout1, std::vector<RegisterBlock> &layout2) {
    return matchLayouts(T, layout1, layout2) || matchLayouts(T, layout2, layout1);
}

// Assign a single mask to all blocks in a layout.
void assignUniformMask(std::vector<RegisterBlock> &layout, ngen::FlagRegister flag, int idx = 0);

// Assign runtime-computed descriptor information to all blocks in this layout.
// Returns true if successful; false if not all blocks in layout are compatible.
bool assignAllDescs(std::vector<RegisterBlock> &layout);

void postprocessLayout(Type T, std::vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy);

void finalizeLayout(ngen::HW hw, Type T, std::vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy);

void coalesceAddrs(ngen::HW hw, Type T, std::vector<RegisterBlock> &layout, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy);

bool needsRemask(Type T, bool column, const std::vector<RegisterBlock> &layout,
                 const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool ignoreMasks);
#include "internal/namespace_end.hxx"

#endif /* header guard */

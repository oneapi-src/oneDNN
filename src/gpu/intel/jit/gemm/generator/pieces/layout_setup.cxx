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
#include "generator.hpp"
#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "map.hpp"

using namespace ngen;
using namespace ngen::utils;
using std::vector;

#include "internal/namespace_start.hxx"


// Set up a RegisterBlock structure.
template <HW hw>
bool BLASKernelGenerator<hw>::getBlockInfo(Type T, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                           int r, int c, bool remainderR, bool remainderC, bool writable, RemainderOptions remOpts,
                                           int maxRBlock, int maxCBlock, int &rblock, int &cblock, RegisterBlock &block)
{
    bool avoidFragment = (remOpts & AllowFragment) == 0;
    bool allowDesc = remOpts & AllowDescriptors;
    bool allowFixedMasks = (remOpts & NoFixedMasks) == 0;
    bool prefetch = astrategy.prefetch;
    bool atomic = astrategy.atomic;
    if (hw >= HW::Xe2) allowDesc = false;

    int R = rounddown_pow2(r);
    int C = rounddown_pow2(c);

    if (maxRBlock == 0) maxRBlock = r;
    if (maxCBlock == 0) maxCBlock = c;

    if (isPacked(atype.layout)) {
        // Don't cross nonconsecutive tiles in a packed layout.
        bool cm = isColMajor(atype.layout) ^ isTransposing(astrategy.accessType);
        if (cm) {
            if (maxRBlock < atype.packSize && atype.tileC > 0)
                maxCBlock = std::min<int>(maxCBlock, atype.tileC);
        } else {
            if (maxCBlock < atype.packSize && atype.tileR > 0)
                maxRBlock = std::min<int>(maxRBlock, atype.tileR);
        }
    }

    // Set default parameters.
    block.colMajor = isColMajor(atype.layout);
    block.splitComplex = false;
    block.byteGlue = false;
    block.cxComponent = RegisterBlock::Interleaved;
    block.crosspack = 1;
    block.rowMask = MaskInfo::None();
    block.colMask = MaskInfo::None();
    block.rowFragment = 0;
    block.colFragment = 0;
    block.remainderR = remainderR;
    block.remainderC = remainderC;
    block.noRowsOK = false;
    block.noColsOK = false;
    block.descRemR = false;
    block.descRemC = false;
    block.descAssigned = false;
    block.addrShift = 0;
    block.writable = writable;
    block.clearFlag();
    block.log2GRFBytes = GRF::log2Bytes(hw);
    block.msgRegs = 0;
    block.bytes = 0;
    block.hasNoLoad = false;
    block.offsetAddr = 0;

    auto &vrmask = block.rowMask.variable;
    auto &vcmask = block.colMask.variable;

    vrmask.rsize = 0;
    vcmask.rsize = 0;

    auto accessType = astrategy.accessType;

    switch (accessType) {
        case AccessType::ChannelScattered:
        case AccessType::Scattered:
        {
            bool channelScattered = (accessType == AccessType::ChannelScattered);

            // Detect large crosspack case.
            bool largeCP = isLargeCrosspack(T, atype.crosspack);
            int effCP = largeCP ? 1 : atype.crosspack;

            // Scattered read/write messages effectively transpose DW/QW matrices.
            block.colMajor = !block.colMajor ^ largeCP;

            // Let X be the contiguous dimension, Y the scattered dimension (in memory).
            int *xblock, *yblock;
            int maxXBlock, maxYBlock;
            int X, Y;
            bool remainderX, remainderY;
            int tileX, tileY;
            int scxExpand = 1;
            auto &vxmask = block.colMajor ? vcmask : vrmask;
            auto &vymask = block.colMajor ? vrmask : vcmask;
            auto &fragment = block.colMajor ? block.colFragment : block.rowFragment;
            auto smode = astrategy.smode;

            if (block.colMajor) {
                Y = allowFixedMasks ? r : R; X = C;
                yblock = &rblock;
                xblock = &cblock;
                maxYBlock = maxRBlock;
                maxXBlock = maxCBlock;
                remainderY = remainderR;
                remainderX = remainderC;
                tileY = atype.tileR;
                tileX = atype.tileC;
            } else {
                X = R; Y = allowFixedMasks ? c : C;
                xblock = &rblock;
                yblock = &cblock;
                maxXBlock = maxRBlock;
                maxYBlock = maxCBlock;
                remainderX = remainderR;
                remainderY = remainderC;
                tileX = atype.tileR;
                tileY = atype.tileC;
            }

            if (tileX > 0) maxXBlock = std::min(maxXBlock, tileX);
            if (tileY > 0) maxYBlock = std::min(maxYBlock, tileY);

            // Allowed accesses:
            //   A64             Essentially max 256 bytes.
            //                    8 slots x (1,2,4,8) dwords [Gen12/surface: 1,2,4]
            //                    8 slots x (1,2,4) qwords
            //                   16 slots x (1,2,4) dwords
            //                   16 slots x (1,2) qwords
            //   Others           8 slots x 1 dword
            //                   16 slots x 1 dword
            // Slot counts doubled for 64-byte GRFs.

            // Native (col major in memory) matrix block sizes, as a result:
            //   SIMD8:          1x8  2x4 4x2 8x1      (count 1)  2x8  4x8  8x8  [others]
            //   SIMD16:         1x16 2x8 4x4 8x2 16x1 (count 1)  2x16 4x16
            // Other layouts are possible too but require dummy (non-load) blocks.
            // Only kx8 and kx16 are supported for now for {4,8}-byte types.
            // For 16-byte types, only 1x4 and 1x8 are supported.

            auto maxSIMD = maxScatteredSIMD(hw, astrategy);
            auto minSIMD = minScatteredSIMD(hw, astrategy);

            auto Xc = ((avoidFragment && !allowDesc && remainderX) || atomic) ? 1 : X;
            bool byte = (atype.alignment < 4) || (Xc * T * effCP < 4);
            bool a64 = (astrategy.base.getModel() == ModelA64);

            channelScattered |= byte;

            bool qword = (T.paddedSize() >= 8 && !channelScattered && !prefetch && (a64 || astrategy.newDP));
            if (atomic && hasNativeAtomicAdd(hw, T.real(), atype, astrategy))
                qword &= (T.real().paddedSize() >= 8);
            int width = qword ? 8 : 4;
            block.ebytes = byte ? 1 : width;
            block.crosspack = std::max<int>(1, width / T);
            int consecutive = std::max<int>(1, T.paddedSize() / width);

            if (prefetch) consecutive = 1;

            if (block.ebytes == 4 && astrategy.base.getModel() == ModelSLM && !astrategy.newDP)
                channelScattered = true;

            bool simd1 = !a64 && !channelScattered && !astrategy.newDP;

            // Handle source crosspack.
            int uncrosspack = 1;
            if (effCP > 1) {
                if (effCP == block.crosspack) {
                    block.crosspack = 1;
                    uncrosspack = effCP;
                } else
                    stub();
            }

            // Try to fit a native matrix block size to X and Y.
            auto logicalSlots = std::min(Y, maxYBlock) * consecutive / uncrosspack;
            auto slots = roundup_pow2(logicalSlots);
            if (prefetch) {
                // Prefetch only: maximize Y usage.
                block.simdSize = maxSIMD;
            } else if (smode == ScatterSIMD::Narrow || (smode == ScatterSIMD::Default && block.ebytes * minSIMD > GRF::bytes(hw))) {
                // Maximize X usage because we always have at least 2 consecutive GRFs.
                block.simdSize = (slots >= maxSIMD && X <= 2) ? maxSIMD : minSIMD;
            } else {
                // Otherwise, try to maximize Y usage (larger SIMD, worse memory access).
                block.simdSize = maxSIMD;
            }
            block.simdSize = slots = std::min<int>(block.simdSize, slots);
            logicalSlots = std::min<int>(block.simdSize, logicalSlots);

            bool no8x8DW = isGen12;
            bool no16x4QW = false;

            no8x8DW &= !astrategy.newDP;
            no16x4QW |= (!astrategy.newDP && GRF::bytes(hw) == 64);

            int hwMaxXBlock;

            if (prefetch)
                hwMaxXBlock = 64 / T;
            else if (consecutive > 1)
                hwMaxXBlock = 1;
            else if (byte)
                hwMaxXBlock = (remainderX || atomic) ? 1 : block.crosspack;
            else if (simd1)
                hwMaxXBlock = block.crosspack;
            else if (remainderX && avoidFragment && !allowDesc)
                hwMaxXBlock = block.crosspack * scxExpand;
            else if (atomic)
                hwMaxXBlock = block.crosspack * scxExpand;
            else if (channelScattered || (block.ebytes == 4 && no8x8DW) || (block.ebytes == 8 && no16x4QW) || (block.simdSize == maxSIMD))
                hwMaxXBlock = 16 / T / uncrosspack;
            else
                hwMaxXBlock = 32 / T / uncrosspack;

            maxXBlock = std::min(maxXBlock, hwMaxXBlock);

            *xblock = std::min<int>(X, maxXBlock);
            block.count = *xblock;

            *yblock = logicalSlots * uncrosspack / consecutive;

            if (prefetch)
                block.count = 1;
            else if (byte)
                block.count *= T;
            else
                block.count = std::max<int>(1, block.count / block.crosspack);

            // LD is determined by actual # of SIMD slots in HW. But for X = 1 we may
            //  shrink the LD to avoid allocating unnecessary registers.
            auto ldSIMD = block.simdSize;
            if (*xblock > 1 || (minSIMD * block.ebytes <= GRF::bytes(hw)))
                ldSIMD = std::max<int>(ldSIMD, minSIMD);
            block.ld = ldSIMD * uncrosspack / consecutive;

            // Handle remainder. Masking handles Y remainders.
            if (remainderY) {
                vymask.isFixed = false;
                vymask.bitRep = consecutive;
                vymask.maskRep = 1;
                vymask.rsize = *yblock;
                vymask.rshift = 0;
            } else if (logicalSlots < slots) {
                auto &fymask = block.colMajor ? block.rowMask.fixed : block.colMask.fixed;
                fymask.isFixed = true;
                fymask.rsize = *yblock;
                fymask.value = (uint32_t(1) << logicalSlots) - 1;
            }

            // X remainders require fragmenting. Channel scattered float doesn't need complete fragmenting.
            //   (ditto for regular scattered float with new dataport messages.)
            //  Otherwise, fragment 2 is possible for DWord+ types but not implemented.
            if (remainderX) {
                if (avoidFragment && (*xblock == 1 || block.count == 1)) {
                    vxmask.isFixed = false;
                    vxmask.bitRep = (block.simdSize > 16) ? 32 : 16;
                    vxmask.maskRep = 1;
                    vxmask.rsize = 1;
                    vxmask.rshift = 0;
                } else if (allowDesc && (channelScattered || astrategy.newDP) && *xblock > 1 && !byte) {
                    fragment = std::min(*xblock, 4 * width / T);
                    if (block.colMajor)             // Clang can't handle the ternary operator equivalent of this.
                        block.descRemC = true;
                    else
                        block.descRemR = true;
                } else
                    fragment = 1;
            }

            block.extra = consecutive;

            // BTS scattered accesses are addressed by elements.
            if (!astrategy.newDP && !channelScattered && !astrategy.base.isStateless())
                block.addrShift = ilog2(block.ebytes);

            break;
        }
        case AccessType::Block:
        case AccessType::PseudoBlock:
        {
            // Three types of block messages:
            //    block_oword: 16 byte align, BLK masking (= dw except ow channel on R Gen9 only -- silently ignore, can't fault)
            //  aligned_oword:  4 byte align, no masking, read only
            //    block_hword: [Gen9-12LP] A64; 4 byte align R, BLKCM masking (= dw but can do ow channel on Gen9 only)
            //                             A64; 16 byte align W
            //                 [XeHP]   A64/BTS; 32 byte align R/W
            // New dataport messages support {DW, QW}x{1...64} with DW/QW alignment, no masking.
            //
            // Prefer block_hword in all cases. When block_hword can't be used:
            //   Use oword if alignment can be assured (i.e. packed row/column layout, or oword-sized scalar)
            //   Otherwise, use aligned oword. load/storeMatrixBlock will emit an error if masking/stores attempted.
            //
            // Pseudoblock messages have similar layouts, but are limited to
            //  {8,16}x{dw,qw} sizes, so lengths 8,16 allowed for float, 4,8,16 for double.

            bool colMajor = block.colMajor;
            bool effCM = colMajor ^ isLargeCrosspack(T, atype.crosspack);
            auto consecutive = consecutiveElements(r, c, atype);
            bool masking = (effCM ? remainderR : remainderC);
            bool bytePartialCP = (T.paddedSize() & 3) && ((colMajor ? C : R) % atype.crosspack);
            bool byte = (atype.alignment & 3) || (consecutive * T & 3) || bytePartialCP || ((T.paddedSize() & 3) && writable && masking);
            bool byte1PerSlot = byte && (bytePartialCP || masking || atomic);
            bool pseudo = (accessType == AccessType::PseudoBlock)
                        | needsPseudoblock(hw, T, R, C, atype, astrategy, writable, masking);
            int maxElements = 0;
            int maskGranularity = 1;
            int maxSIMD = maxScatteredSIMD(hw, astrategy);
            bool oword = false, aoword = false;
            int npack = 0;
            bool canQW = false, mustQW = false;

            bool a32 = (astrategy.base.getModel() == ModelA32);
            bool a64 = (astrategy.base.getModel() == ModelA64);
            bool sc = (astrategy.base.getModel() == ModelSC);
            bool slm = (astrategy.base.getModel() == ModelSLM);

            if (!pseudo && byte) return false;

            if (astrategy.newDP && !pseudo) {
                bool qword = ((atype.alignment | (consecutive * T)) % 8 == 0);
                block.ebytes = qword ? 8 : 4;
                maxElements = (64 * block.ebytes) / T;
                maskGranularity = T.paddedSize();         // Convenience value; LSC cannot mask individual elements
            } else if (!pseudo) {
                int maxCount = 8;
                oword = !a64;
                aoword = ((atype.alignment & 0xF) != 0) || sc;
                if (hw > HW::Gen12LP) {
                    oword |= (atype.alignment & 0x1F) != 0;
                    if (slm) maxCount = 16;
                }
                block.ebytes = oword ? 16 : 32;
                maxElements = maxCount * block.ebytes / T;
                maskGranularity = 4;                // Block accesses mask by dwords
            } else {
                bool nativeAtomic = atomic && hasNativeAtomicAdd(hw, T.real(), atype, astrategy);
                canQW = ((atype.alignment | (consecutive * T)) % 8 == 0);
                if (astrategy.newDP)
                    canQW |= byte;
                else
                    canQW &= !byte && a64;
                if (slm && atomic)        // QW SLM atomics are implemented in XeHPC, but seeing functionality issues.
                    canQW = false;
                if (remainderR || remainderC)
                    canQW &= (T.paddedSize() % 8 == 0);
                if (nativeAtomic)
                    canQW = mustQW = (T.real().paddedSize() >= 8);
                auto stride = canQW ? 8 : 4;
                auto maxNPack = byte1PerSlot ? 1 : std::max<int>(1, stride / T.paddedSize());
                int simdCap = maxSIMD;
                if (atomic && !nativeAtomic)
                    simdCap = 16;
                maxElements = simdCap * maxNPack;
                if (T.paddedSize() > stride)
                    maxElements = maxElements * stride / T;
                if (allowFixedMasks)
                    R = r, C = c;
            }

            auto maxABlock = maxElements / (byte1PerSlot ? 1 : atype.crosspack);

            auto choosePackedRCBlock = [=, &block](int &xblock, int &yblock, int tileX, int tileY, int X, int Y) {
                xblock = std::min<int>(maxABlock, X);

                if (tileX) {
                    int ntileY = tileY ? (maxElements / (xblock * tileY)) : 0;
                    if (xblock < atype.packSize || Y < tileY || ntileY == 0)
                        xblock = std::min<int>(xblock, tileX);
                }
                if ((tileX ? tileX : atype.packSize) <= xblock) {
                    yblock = std::min<int>(maxElements / xblock, Y);
                    if (yblock < atype.crosspack && isLargeCrosspack(T, atype.crosspack)) {
                        yblock = atype.crosspack;
                        xblock = std::min<int>(xblock, maxElements / yblock);
                    }
                    if (tileY > 0 && yblock > tileY)
                        yblock = align_down(yblock, tileY);
                } else
                    yblock = atype.crosspack;     // Remainder loop: no longer packed in memory

                block.crosspack = atype.crosspack;
            };

            switch (atype.layout) {
                case MatrixLayout::Pc:
                    choosePackedRCBlock(rblock, cblock, atype.tileR, atype.tileC, R, C);
                    break;
                case MatrixLayout::N:
                    if (atype.crosspack > 1) stub();
                    if (atype.tileR == R && R <= maxElements) {
                        cblock = std::min<int>(maxElements / R, C);
                        rblock = R;
                    } else {
                        cblock = 1;
                        rblock = std::min<int>(maxElements, R);
                    }
                    break;
                case MatrixLayout::Pr:
                    choosePackedRCBlock(cblock, rblock, atype.tileC, atype.tileR, C, R);
                    break;
                case MatrixLayout::T:
                    if (atype.crosspack > 1) stub();
                    if (atype.tileC == C && C <= maxElements) {
                        rblock = std::min<int>(maxElements / C, R);
                        cblock = C;
                    } else {
                        rblock = 1;
                        cblock = std::min<int>(maxElements, C);
                    }
                    break;
            }

            rblock = std::min(rblock, maxRBlock);
            cblock = std::min(cblock, maxCBlock);

            if (pseudo) {
                bool qword = mustQW || (canQW && (rblock * cblock * T >= 4 * maxSIMD));
                npack = std::max<int>(1, (qword ? 8 : 4) / T);
                if (byte1PerSlot) {
                    if (isLargeCrosspack(T, block.crosspack)) {
                        if (block.crosspack == (colMajor ? cblock : rblock))
                            block.colMajor = colMajor = effCM;
                        else
                            stub();
                    }
                    block.crosspack = npack / T.perByte();
                    block.byteGlue = (T.bits() < 8);
                    npack = T.perByte();
                    (effCM ? cblock : rblock) = 1;
                }
                maskGranularity = qword ? 8 :
                           byte1PerSlot ? T.paddedSize() :
                                          4;
            }

            if (remainderR) {
                if (effCM) {
                    // rblock cannot be more than 16 dwords = 64 bytes for masking
                    //  except for pseudo-block
                    int rblockLimit = pseudo ? rblock : 64 / T;

                    if (avoidFragment) rblock = std::min<int>(rblock, rblockLimit);
                    if (rblock > rblockLimit)
                        block.rowFragment = rblockLimit;
                    else {
                        // For sizeof(T) < maskGranularity, this is a bit of a cheat.
                        //
                        // As long as we do not need to write to this matrix, we can read
                        // in maskGranularity-sized chunks knowing we will never cross a page boundary.

                        if (writable && (T.paddedSize() & (maskGranularity - 1)))
                            return false;
                        if (!pseudo && oword && aoword)
                            hw_unsupported();

                        if (!pseudo && !(isPacked(atype.layout) && (atype.packSize == rblock))) cblock = 1;

                        vrmask.isFixed = false;
                        vrmask.rsize = rblock;
                        vrmask.bitRep = std::max<int>(T.paddedSize() / maskGranularity, 1);
                        vrmask.maskRep = cblock;
                        vrmask.rshift = ilog2(std::max<int>(maskGranularity / T, 1));
                    }
                } else {
                    if (avoidFragment) {
                        // No native masking in this dimension. One mask/row.
                        rblock = 1;
                        vrmask.isFixed = false;
                        vrmask.bitRep = 0;  /* will be filled in later */
                        vrmask.maskRep = 1;
                        vrmask.rsize = 1;
                        vrmask.rshift = 0;
                    } else {
                        // Fragment it. Could actually handle rowFragment = 2 by changing descriptor.
                        block.rowFragment = 1;
                    }
                }
            }

            if (remainderC) {
                if (!effCM) {
                    // cblock cannot be more than 16 dwords = 64 bytes except for pseudo-block
                    int cblockLimit = pseudo ? cblock : 64 / T;

                    if (avoidFragment) cblock = std::min<int>(cblock, cblockLimit);
                    if (cblock > cblockLimit)
                        block.colFragment = cblockLimit;
                    else {
                        if (writable && (T.paddedSize() & (maskGranularity - 1)))
                            return false;
                        if (!pseudo && oword && aoword)
                            hw_unsupported();

                        if (!pseudo && !(isPacked(atype.layout) && (atype.packSize == cblock))) rblock = 1;

                        vcmask.isFixed = false;
                        vcmask.rsize = cblock;
                        vcmask.bitRep = std::max<int>(T.paddedSize() / maskGranularity, 1);
                        vcmask.maskRep = rblock;
                        vcmask.rshift = ilog2(std::max<int>(maskGranularity / T, 1));
                    }
                } else {
                    if (avoidFragment) {
                        // No native masking in this dimension. One mask/column.
                        cblock = 1;
                        vcmask.isFixed = false;
                        vcmask.bitRep = 0;
                        vcmask.maskRep = 1;
                        vcmask.rsize = 1;
                        vcmask.rshift = 0;
                    } else {
                        // Fragment it. Could actually handle colFragment = 2 by changing descriptor.
                        block.colFragment = 1;
                    }
                }
            }

            bool needFRMask = pseudo && (!remainderR && !is_zero_or_pow2(rblock));
            bool needFCMask = pseudo && (!remainderC && !is_zero_or_pow2(cblock));

            if (needFRMask || needFCMask) {
                // Create fixed mask for this block.
                auto &fmask = needFRMask ? block.rowMask.fixed : block.colMask.fixed;
                int logicalSlots = (rblock * cblock * T) / maskGranularity;
                fmask.isFixed = true;
                fmask.rsize = rblock * cblock;
                fmask.value = (uint32_t(1) << logicalSlots) - 1;
            }

            int nbytes = roundup_pow2(rblock * cblock) * T;
            block.simdSize = clamp(roundup_pow2(nbytes) / maskGranularity, 1, maxSIMD);
            block.ld = colMajor ? rblock : cblock;
            if (!pseudo) {
                if (astrategy.newDP) block.simdSize = 1;
                block.count = div_up(nbytes, block.ebytes);
                block.extra = aoword;
                if (block.ebytes == 16 && !(a32 || a64) && !aoword)         // BTS/SLM oword loads are oword-addressed.
                    block.addrShift = 4;
            } else {
                block.count = byte ? std::min(nbytes, npack * T) : 1;
                block.ebytes = byte ? 1 : maskGranularity;
                block.extra = 1;
                if (!(a32 || a64 || pseudoblockUseChannelScattered(atype, astrategy, block) || atomic))
                    block.addrShift = ilog2(block.ebytes);
            }
            if (astrategy.newDP) block.addrShift = 0;

            int maskAllBitRep = (pseudo && block.simdSize > 16) ? 32 : 16;
            if (!vrmask.isFixed && vrmask.bitRep == 0) vrmask.bitRep = maskAllBitRep;
            if (!vcmask.isFixed && vcmask.bitRep == 0) vcmask.bitRep = maskAllBitRep;
            break;
        }
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI: {
            // bytes * array length <= 8
            // width * array length <= 64 bytes
            //  => width <= 1 GRF
            // height <= 32 (load) 8 (store)
            // array length = 1 for store, transpose
            //
            // normal: width >= 4 bytes
            // transpose: d32/d64 only
            // vnni: d8/d16 only, height >= 4 bytes
            bool transpose = (accessType == AccessType::Block2DTranspose);
            bool vnni      = (accessType == AccessType::Block2DVNNI);

            bool memCM = block.colMajor;
            block.colMajor ^= transpose;
            auto X = memCM ? r : c;
            auto Y = memCM ? c : r;
            auto &xblock = memCM ? rblock : cblock;
            auto &yblock = memCM ? cblock : rblock;
            auto maxXBlock = memCM ? maxRBlock : maxCBlock;
            auto maxYBlock = memCM ? maxCBlock : maxRBlock;

            if (hw < HW::XeHPC || !astrategy.newDP) hw_unsupported();

            // Choose underlying type.
            auto Tblock = T;
            if (transpose) {
                int maxW;
                if (Tblock.paddedSize() > 8) hw_unsupported();
                if (Tblock.paddedSize() > 4) {
                    Tblock = Type::u64;
                    maxW = 4;
                    maxYBlock = 8;
                } else {
                    Tblock = Type::u32;
                    maxW = 8;
                }
                maxXBlock = std::min(maxXBlock, (maxW * Tblock) / T);
            } else if (vnni) {
                if (Tblock.paddedSize() >= 4) hw_unsupported();
                if ((Y * Tblock) % 4) hw_unsupported();
                maxXBlock = std::min(maxXBlock, 16);
            } else {
                if (Tblock.paddedSize() > 8) Tblock = Type::u64;
                block.crosspack = atype.crosspack;
            }
            if ((X * T) % 4) hw_unsupported();

            int minAlign = block2DMinAlignment(hw, atype, astrategy);
            if (atype.alignment % minAlign) hw_unsupported();

            // Reinterpret X/maxXBlock to underlying type.
            maxXBlock = (maxXBlock * T) / Tblock;
            auto X_logical = X;
            X = (X * T) / Tblock;

            // Carve out a maximal allowed block size.
            xblock = std::min(X, 64 / Tblock);
            xblock = std::max(xblock, 4 / Tblock);
            int yblockLimit = writable ? 8 : 32;

            if (isPacked(atype.layout) && 2 * xblock <= X && X_logical == atype.packSize) {
                // Split logical x dimension into multiple spans to accomodate width restriction.
                if (astrategy.address2D) stub();
                int multiX = X / xblock;
                xblock *= multiX;
                yblockLimit /= multiX;
            }

            yblock = std::min({maxYBlock, Y, yblockLimit});

            if (transpose && Tblock.paddedSize() == 8 && yblock != 8) hw_unsupported();

            // Choose # of blocks. In postprocessLayout, this RegisterBlock will be
            //  split into one RegisterBlock for each block in the array.
            int count = 1;
            if (!(writable || transpose)) {
                count = rounddown_pow2(xblock / maxXBlock);
                count = std::min({count, 8 / Tblock, 4});
                count = std::max(count, 1);
            }
            xblock = std::min(xblock, maxXBlock * count);

            // Crosspack calculation.
            int crosspack = (transpose || vnni) ? std::max(1, 4 / T) : 1;
            if (atype.crosspack == 1)
                block.crosspack = crosspack;
            else if (atype.crosspack == crosspack)
                block.crosspack = 1;
            else return false;

            // Convert size from underlying type to our actual type.
            xblock = (xblock * Tblock) / T;

            block.simdSize = 1;
            block.ld = roundup_pow2(transpose ? yblock : xblock);
            block.ebytes = Tblock.paddedSize();
            block.count = count;
            block.extra = T.bits();
            auto bytes = align_up((block.colMajor ? cblock : rblock) / count, block.crosspack) * block.ld * count * T;
            block.msgRegs = GRF::bytesToGRFs(hw, bytes);
            if (vnni && (T.bits() < 8)) {
                block.byteGlue = true;
                block.crosspack /= T.perByte();
            }

            // Xe2: manually mask in the height dimension to work around slow LSC
            //      out-of-bounds checks.
            bool remainderH = memCM ? remainderC : remainderR;
            if (hw >= HW::Xe2 && remainderH) {
                auto &vymask = memCM ? block.colMask.variable : block.rowMask.variable;
                vymask.isFixed = false;
                vymask.bitRep = vymask.maskRep = vymask.rsize = 1;
                vymask.rshift = 0;
            }
            break;
        }
        case AccessType::CacheLine: {
            // Let X be the contiguous dimension in memory, Y the scattered dimension.
            int x = block.colMajor ? r : c;
            int y = block.colMajor ? c : r;
            auto &xblock = block.colMajor ? rblock : cblock;
            auto &yblock = block.colMajor ? cblock : rblock;
            auto maxXBlock = block.colMajor ? maxRBlock : maxCBlock;
            auto maxYBlock = block.colMajor ? maxCBlock : maxRBlock;
            bool remainderX = block.colMajor ? remainderR : remainderC;
            bool remainderY = block.colMajor ? remainderC : remainderR;

            auto maxSIMD = maxScatteredSIMD(hw, astrategy);
            int trailing = (atype.alignment % 64) ? 1 : 0;      // Do we need a trailing pointer?
            int elemsPerCL = 64 / T;

            xblock = std::min({maxXBlock, x, (maxSIMD - trailing) * elemsPerCL});
            int xCacheLines = roundup_pow2(div_up(x * T, 64) + trailing);

            yblock = rounddown_pow2(std::min({maxYBlock, y, maxSIMD / xCacheLines}));

            block.simdSize = xCacheLines * yblock;
            block.ld = xCacheLines;
            if (atype.alignment >= 4) {
                block.ebytes = 4;
                block.count = 1;
            } else {
                block.ebytes = 1;
                block.count = atype.alignment;
            }
            block.extra = 1;

            if (remainderX) {
                // All on/off mask for x remainders. Finer grained remainders
                //  are handled by adjusting addresses to be in bounds.
                auto &vxmask = block.colMajor ? block.rowMask.variable : block.colMask.variable;
                vxmask.isFixed = false;
                vxmask.bitRep = block.simdSize;
                vxmask.maskRep = vxmask.rsize = 1;
                vxmask.rshift = 0;
            }

            if (remainderY) {
                auto &vymask = block.colMajor ? block.colMask.variable : block.rowMask.variable;
                vymask.isFixed = false;
                vymask.bitRep = xCacheLines;
                vymask.maskRep = 1;
                vymask.rsize = yblock;
                vymask.rshift = 0;
            }
            break;
        }
    }

    // The mask moduli are almost always rblock/cblock.
    // Also, clamp mask reps to ensure mask length does not exceed SIMD size.
    if (block.rowMask && !block.rowMask.fixed.isFixed) {
        if (vrmask.rsize == 0)
            vrmask.rsize = rblock;
        vrmask.maskRep = std::min<int>(vrmask.maskRep, std::max<int>(1, (block.simdSize << vrmask.rshift) / (vrmask.bitRep * vrmask.rsize)));
        block.noRowsOK = true;          // All-zero masks are always OK.
    }
    if (block.colMask && !block.colMask.fixed.isFixed) {
        if (vcmask.rsize == 0)
            vcmask.rsize = cblock;
        vcmask.maskRep = std::min<int>(vcmask.maskRep, std::max<int>(1, (block.simdSize << vcmask.rshift) / (vcmask.bitRep * vcmask.rsize)));
        block.noColsOK = true;
    }

    return true;
}

// Attempt to add remainder handling to an existing block. Returns true if successful.
template <HW hw>
bool BLASKernelGenerator<hw>::tryAddRemainder(Type T, RegisterBlock &block, bool remainderR, bool remainderC, RemainderOptions remOpts,
                                              const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    auto blockNew = block;
    blockNew.remainderR |= remainderR;
    blockNew.remainderC |= remainderC;

    auto curAccessType = implAccessType(atype, astrategy, block);

    if (curAccessType == AccessType::Block) {
        if (astrategy.newDP) return false;
        if (hw >= HW::XeHPC) return false;
    }

    bool remChanged = (remainderR && !block.remainderR)
                   || (remainderC && !block.remainderC);

    if (remChanged && !isBlock2D(curAccessType)) {
        int rblock = 0, cblock = 0;
        if (!getBlockInfo(T, atype, astrategy, block.nr, block.nc, blockNew.remainderR, blockNew.remainderC,
                          block.writable, remOpts, 0, 0, rblock, cblock, blockNew)) return false;
        if (rblock != block.nr || cblock != block.nc)
            return false;
        if (implAccessType(atype, astrategy, blockNew) != curAccessType)
            return false;
        if (curAccessType != AccessType::Block) {
            if (blockNew.ebytes != block.ebytes)
                return false;
            if (blockNew.ebytes == 1 && blockNew.count != block.count)
                return false;
        }
        blockNew.bytes = block.bytes;
        blockNew.offsetAddr = block.offsetAddr;
    }

    block = blockNew;
    return true;
}

// Add a submatrix to a register layout.
template <HW hw>
bool BLASKernelGenerator<hw>::addToRegLayout(Type T, std::vector<RegisterBlock> &layout,
                                             int nr, int nc, int roff, int coff,
                                             bool remainderR, bool remainderC, bool writable, RemainderOptions remOpts,
                                             int maxRBlock, int maxCBlock,
                                             const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    int rblock, cblock;
    RegisterBlock blockTemplate;
    if (!getBlockInfo(T, atype, astrategy, nr, nc, remainderR, remainderC, writable, remOpts, maxRBlock, maxCBlock, rblock, cblock, blockTemplate))
        return false;       /* Cannot handle requested block and remainder. */

    if (rblock == 0 || cblock == 0)
        return false;

    blockTemplate.nr = rblock;
    blockTemplate.nc = cblock;

    for (int q = 0; q < T.components(); q++) {
        blockTemplate.component = q;
        if (isColMajor(atype.layout)) {
            // Order blocks in column-major fashion.
            for (int c = 0; c + cblock <= nc; c += cblock) {
                for (int r = 0; r + rblock <= nr; r += rblock) {
                    auto thisBlock = blockTemplate;

                    thisBlock.offsetR = r + roff;
                    thisBlock.offsetC = c + coff;

                    layout.push_back(thisBlock);
                }
            }
        } else {
            // Order blocks in row-major fashion.
            for (int r = 0; r + rblock <= nr; r += rblock) {
                for (int c = 0; c + cblock <= nc; c += cblock) {
                    auto thisBlock = blockTemplate;

                    thisBlock.offsetR = r + roff;
                    thisBlock.offsetC = c + coff;

                    layout.push_back(thisBlock);
                }
            }
        }
    }

    // Handle remainder recursively, checking for infinite recursion.
    int rrem = nr % rblock;
    int crem = nc % cblock;

    status << "Register layout: " << nr << 'x' << nc << " -> blocks " << rblock << 'x' << cblock << " remainder " << rrem << 'x' << crem << status_stream::endl;

    bool success = true;
    if (rrem || crem) {
        if ((nr == rrem || rrem == 0) && (nc == crem || crem == 0)) {
            status << "Cannot load/store requested matrix block size." << status_stream::endl;
            success = false;
        } else {
            if (rrem) success &= addToRegLayout(T, layout, rrem, nc - crem, nr - rrem, 0, remainderR, remainderC, writable, remOpts, maxRBlock, maxCBlock, atype, astrategy);
            if (crem) success &= addToRegLayout(T, layout, nr, crem, 0, nc - crem, remainderR, remainderC, writable, remOpts, maxRBlock, maxCBlock, atype, astrategy);
        }
    }
    return success;
}

// Add a submatrix (contiguous in memory) to a block-accessed register layout.
template <HW hw>
bool BLASKernelGenerator<hw>::add1DBlockToRegLayout(Type T, vector<RegisterBlock> &layout, int r, int c, bool writable, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    // Skip pseudoblock cases (possible to support though)
    if (needsPseudoblock(hw, T, r, c, atype, astrategy, writable, false))
        return false;

    // Get total number of bytes to load. No masking supported, so stub if
    //  number of bytes not divisible by 16 (1 oword).
    int nbytes = r * c * T * T.components();
    int align = 16;
    if (astrategy.newDP) align = 4;

    if (nbytes & (align - 1))
        return false;

    // Get block info.
    int maxBBytes = 0;
    int ebytes = 0;
    int extra = 0;
    int addrShift = 0;
    int maxSIMD = 1;

    if (astrategy.newDP) {
        bool qword = (nbytes | atype.alignment) % 8 == 0;
        ebytes = qword ? 8 : 4;
        maxBBytes = ebytes * 64;
    } else {
        bool a64 = (astrategy.base.getModel() == ModelA64);
        bool oword = !a64;
        bool aoword = (astrategy.base.getModel() == ModelSC); // SC only does aligned oword
        if (hw >= HW::XeHP) oword |= ((atype.alignment & 0x1F) != 0);

        extra = aoword;
        ebytes = oword ? 16 : 32;
        maxBBytes = oword ? 128 : 256;
        if (astrategy.base.getModel() == ModelSLM && hw >= HW::XeHP) maxBBytes = 256;
        addrShift = (!a64 && oword && !aoword) ? 4 : 0;
        maxSIMD = 16;
    }

    // Get normalized dimensions.
    bool colMajor = isColMajor(atype.layout);
    int x = colMajor ? r : c;
    auto crosspack = atype.crosspack;

    // Counters for current x and y positions.
    int cx = 0, cy = 0;

    while (nbytes > 0) {
        // Carve out the largest chunk possible.
        int bbytes = std::min<int>(maxBBytes, rounddown_pow2(nbytes));
        int belems = bbytes / T;

        // Create a true load block for first (possibly partial) row/column.
        // Then, create additional no-load blocks for any further (possible partial)
        //   rows/columns until block is exhausted.
        bool first = true;
        while (belems > 0) {
            int nxRem = belems / crosspack;
            int nx = std::min<int>(nxRem, x - cx);
            if (nx <= 0) stub();
            if (cy % crosspack) return false;

            RegisterBlock block;

            block.ld = nx;
            (colMajor ? block.nr : block.nc) = nx;
            (colMajor ? block.nc : block.nr) = crosspack;
            (colMajor ? block.offsetR : block.offsetC) = cx;
            (colMajor ? block.offsetC : block.offsetR) = cy;
            block.component = 0;
            block.colMajor = colMajor;
            block.splitComplex = false;
            block.byteGlue = false;
            block.cxComponent = RegisterBlock::Interleaved;

            if (first) {
                block.ebytes = ebytes;
                block.count = div_up(bbytes, ebytes);
                block.simdSize = std::min(maxSIMD, roundup_pow2(bbytes) >> 2);
            } else
                block.ebytes = block.count = block.simdSize = 0;

            block.extra = extra;
            block.clearFlag();
            block.colMask = MaskInfo::None();
            block.rowMask = MaskInfo::None();
            block.colFragment = 0;
            block.rowFragment = 0;
            block.log2GRFBytes = GRF::log2Bytes(hw);

            block.crosspack = crosspack;
            block.remainderR = false;
            block.remainderC = false;
            block.noRowsOK = false;
            block.noColsOK = false;
            block.descRemR = false;
            block.descRemC = false;
            block.descAssigned = false;
            block.addrShift = addrShift;
            block.offsetAddr = 0;
            block.hasNoLoad = false;
            block.msgRegs = std::max(1, bbytes >> GRF::log2Bytes(hw));

            if (first && cx == 0 && (nxRem % x) == 0) {
                // Shortcut: one register block can represent this block access.
                int ny = belems / x;
                (colMajor ? block.nc : block.nr) = ny;
                cy += ny;
                belems = 0;
            } else {
                cx += nx;
                belems -= nx * crosspack;
                if (cx == x) {
                    cy += crosspack; cx = 0;
                }
                block.hasNoLoad = first && (belems > 0);
                first = false;
            }

            layout.push_back(block);
        }

        nbytes -= bbytes;
    }

    return true;
}

// Create a register layout for a matrix.
template <HW hw>
bool BLASKernelGenerator<hw>::getRegLayout(Type T, vector<RegisterBlock> &layout, int r, int c,
    bool remainderR, bool remainderC, bool writable, RemainderOptions remOpts,
    int maxRBlock, int maxCBlock,
    const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool reverseOrder)
{
    bool success = false;

    layout.clear();

    // If no associated address space, create an empty layout.
    if (astrategy.base.getModel() == ModelInvalid)
        return true;

    // Tiling handling.
    auto forceTiling = [](int &maxBlock, int tile) {
        maxBlock = (maxBlock == 0) ? tile : gcd(tile, maxBlock);
    };

    if (astrategy.tileR > 0) forceTiling(maxRBlock, astrategy.tileR);
    if (astrategy.tileC > 0) forceTiling(maxCBlock, astrategy.tileC);

    if (atype.layout == MatrixLayout::Pc) forceTiling(maxRBlock, atype.packSize);
    if (atype.layout == MatrixLayout::Pr) forceTiling(maxCBlock, atype.packSize);

    // Two separate strategies for creating register layout:
    //    - standard 2D partitioning
    //    - special 1D partitioning for block access to packed inputs.
    if (((atype.layout == MatrixLayout::Pc && atype.packSize == r)
      || (atype.layout == MatrixLayout::Pr && atype.packSize == c))
            && (astrategy.accessType == AccessType::Block)
            && !remainderR && !remainderC
            && !atype.tileR && !atype.tileC
            && (maxRBlock >= r || maxRBlock == 0)
            && (maxCBlock >= c || maxCBlock == 0)) {
        success = add1DBlockToRegLayout(T, layout, r, c, writable, atype, astrategy);
    }
    if (!success) {
        success = addToRegLayout(T, layout, r, c, 0, 0, remainderR, remainderC, writable,
                                 remOpts, maxRBlock, maxCBlock, atype, astrategy);
        sortRegLayout(T, layout, r, c, atype, astrategy, reverseOrder);
        postprocessLayout(T, layout, atype, astrategy);
    }
    if (!success)
        return false;

    finalizeLayout(hw, T, layout, atype, astrategy);

    coalesceAddrs(hw, T, layout, atype, astrategy);

    return true;
}

// Create a register layout for a uniform matrix not backed by memory.
template <HW hw>
void BLASKernelGenerator<hw>::makeUnbackedRegLayout(Type T, vector<RegisterBlock> &layout, int r, int c, bool colMajor, int crosspack, int tileR, int tileC, bool allowPartialRegs, bool fullySplitCx)
{
    auto block = RegisterBlock();

    auto y = (colMajor ? c : r);
    if (y > crosspack && y % crosspack) stub();

    layout.clear();

    if (tileR <= 0) tileR = r;
    if (tileC <= 0) tileC = c;

    int offsetBytes = 0;
    int qCXMin = -1, qCXMax = -1;

    for (int qCX = qCXMin; qCX <= qCXMax; qCX++) {
        for (int q = 0; q < T.components(); q++) {
            for (int i = 0; i < r; i += tileR) {
                for (int j = 0; j < c; j += tileC) {
                    block.log2GRFBytes = GRF::log2Bytes(hw);
                    block.nr = std::min(r - i, tileR);
                    block.nc = std::min(c - j, tileC);
                    block.ld = colMajor ? tileR : tileC;
                    if (!allowPartialRegs)
                        block.ld = align_up(block.ld, elementsPerGRF(hw, T));
                    block.offsetR = i;
                    block.offsetC = j;
                    block.colMajor = colMajor;
                    block.crosspack = crosspack;
                    block.offsetBytes = offsetBytes;
                    block.splitComplex = false;
                    block.byteGlue = false;
                    block.cxComponent = qCX;
                    block.component = q;
                    block.remainderR = false;
                    block.remainderC = false;
                    block.simdSize = 0;         // Not backed by memory.

                    block.calcBytes(T);
                    offsetBytes += block.bytes;

                    layout.push_back(block);
                }
            }
        }
    }

}

// Retrieve a slice [x1,x2) of a block's rows or columns.
template <HW hw>
bool BLASKernelGenerator<hw>::getSubblock(Type T, RegisterBlock &blockDst, const RegisterBlock &blockSrc,
                                          bool column, int x1, int x2, int x1Unclamped, int x2Unclamped, bool overrunOK,
                                          const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    auto Telem = T;
    auto effAccessType = effectiveAccessType(atype, astrategy, blockSrc);
    blockDst = blockSrc;

    auto &ns = (column ? blockDst.nc : blockDst.nr);
    auto &nt = (column ? blockDst.nr : blockDst.nc);
    int oldNS = ns;

    (column ? blockDst.offsetC : blockDst.offsetR) += x1;
    ns = x2 - x1;

    if ((ns == oldNS) && (overrunOK || !blockSrc.hasNoLoad))
        return true;

    if (blockSrc.colMajor == column) {
        if (x1 % blockSrc.crosspack) return false;

        blockDst.offsetBytes += (x1 * blockSrc.bytes) / oldNS;

        if (blockSrc.isLoadBlock()) switch (effAccessType) {
            case AccessType::Scattered:
            case AccessType::ChannelScattered:
                blockDst.count = x2 - x1;
                if (blockDst.ebytes == 1)
                    blockDst.count *= T;
                else if (blockDst.splitComplex)
                    blockDst.count *= 2;
                else if (T.paddedSize() < blockDst.ebytes) {
                    // Extra alignment path with small types.
                    // Check to see if we can still use this element size,
                    //  if not downgrade to scattered byte.
                    // Note for surface accesses this requires shifting the addresses back.
                    auto bcount = blockDst.count * T;
                    if (bcount % 4) {
                        blockDst.ebytes = 1;
                        blockDst.addrShift = 0;
                        blockDst.count = bcount;
                        if (blockDst.count > 4) stub();
                    } else
                        blockDst.count = bcount >> 2;
                }
                break;
            case AccessType::Block:
            case AccessType::PseudoBlock: {
                auto offBytes = x1 * nt * T;
                if (offBytes % blockDst.ebytes)
                    return false;
                auto reqBytes = (x2 - x1) * nt * T;
                auto align = std::min<int>(blockDst.ebytes, blockDst.simdSize * 4);
                if (!overrunOK && (reqBytes & (align - 1)))
                    return false;
                auto ncount = div_up(reqBytes, blockDst.ebytes);
                auto count = roundup_pow2(ncount);
                if (!overrunOK && (count != ncount))
                    return false;
                if (effAccessType == AccessType::Block)
                    blockDst.count = count;
                else
                    blockDst.simdSize = std::max(1, count / blockDst.count);
                break;
            }
            case AccessType::Block2D:
                break;
            case AccessType::Block2DTranspose:
            case AccessType::Block2DVNNI: {
                int crosspack = std::max(1, 4 / blockDst.ebytes);
                if (x1 % crosspack || x2 % crosspack)
                    return false;
                break;
            }
            case AccessType::CacheLine:
                blockDst.simdSize = blockDst.simdSize * (x2 - x1) / ns;
                break;
        }
    } else {
        blockDst.offsetBytes += x1 * Telem * blockSrc.crosspack;

        if (blockSrc.isLoadBlock()) switch (effAccessType) {
            case AccessType::Block:
            case AccessType::PseudoBlock: {
                // Update count and mask information.
                // Beware, cheat: with DW-aligned sub-DW types, true block may be downgraded to byte PseudoBlock,
                //                which requires 2 address registers, though only 1 is used, and only 1 may be allocated.
                int rblock, cblock;
                auto opts = (blockDst.rowFragment || blockDst.colFragment) ? AllowFragment
                                                                           : AvoidFragment;
                (void) getBlockInfo(T, atype, astrategy, blockDst.nr, blockDst.nc,
                                    blockDst.remainderR, blockDst.remainderC, blockDst.writable,
                                    opts, 0, 0, rblock, cblock, blockDst);
                blockDst.flag = blockSrc.flag;
                if (blockDst.flag[column] && x1 > 0) stub();
                blockDst.simplify(T);
                break;
            }
            case AccessType::Scattered:
            case AccessType::ChannelScattered: {
                if (T.paddedSize() > blockDst.ebytes)   return false;
                if (x1 != 0)                            return false;
                if (!is_zero_or_pow2(x2))               return false;

                blockDst.simdSize = div_up(ns * T, blockDst.ebytes);

                auto minSIMD = minScatteredSIMD(hw, astrategy);
                if (blockDst.simdSize <= minSIMD && blockSrc.simdSize > minSIMD) {
                    if (blockDst.count > 1 && blockDst.ebytes > 1)
                        return false;
                    blockDst.ld >>= 1;
                }
                break;
            }
            case AccessType::Block2D:
            case AccessType::Block2DTranspose:
            case AccessType::Block2DVNNI:
                if (ns != oldNS) stub();        // Can do this, but not implemented.
                if (blockDst.simdSize != 0)     // Recompute block array length.
                    blockDst.count = div_up(x2Unclamped, isColMajor(atype.layout) ? blockDst.nr : blockDst.nc);
                // TODO: need to recompute ld
                break;
            case AccessType::CacheLine:
                if (ns != oldNS) stub();
                break;
        }
    }

    if (!blockDst.isLoadBlock()) {
        // Shrink LD
        auto nx = blockDst.colMajor ? blockDst.nr : blockDst.nc;
        auto ny = blockDst.colMajor ? blockDst.nc : blockDst.nr;
        if (ny == 1)
            blockDst.ld = std::min(blockDst.ld, nx);
    }

    blockDst.calcBytes(T, astrategy);

    return true;
}

// Get list of subblocks intersecting rows/columns [x1, x2).
template <HW hw>
bool BLASKernelGenerator<hw>::getSubblocks(Type T, vector<RegisterBlock> &sublayout, const vector<RegisterBlock> &layout,
                                           bool column, int x1, int x2, bool overrunOK,
                                           const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, bool decoalesce)
{
    auto RegisterBlock::*nq      = column ? &RegisterBlock::nc      : &RegisterBlock::nr;
    auto RegisterBlock::*offsetQ = column ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    sublayout.clear();

    for (auto &block : layout) {
        int qq1Unclamped = x1 - block.*offsetQ;
        int qq2Unclamped = x2 - block.*offsetQ;
        int qq1 = clamp<int>(qq1Unclamped, 0, block.*nq);
        int qq2 = clamp<int>(qq2Unclamped, 0, block.*nq);
        if (qq2 > qq1) {
            RegisterBlock subblock;
            if (!getSubblock(T, subblock, block, column, qq1, qq2, qq1Unclamped, qq2Unclamped, overrunOK, atype, astrategy)) {
                status << "Could not make subblock." << status_stream::endl;
                return false;
            }
            if (decoalesce)
                subblock.offsetAddr = 0;
            sublayout.push_back(subblock);
        }
    }
    return true;
}

// Get list of subblocks intersecting rows/columns [x1, x2), and associated address registers and/or indices.
// Returns false if fragmenting failed, or an address register doesn't match a previous one.
template <HW hw>
bool BLASKernelGenerator<hw>::getSubblocks(Type T, vector<RegisterBlock> &sublayout, vector<GRFRange> *subaddrs, vector<int> *indices,
                                           const vector<RegisterBlock> &layout, const vector<GRFRange> *addrs,
                                           bool column, int x1, int x2, bool overrunOK,
                                           const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    auto RegisterBlock::*nq      = column ? &RegisterBlock::nc      : &RegisterBlock::nr;
    auto RegisterBlock::*offsetQ = column ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    if (subaddrs) subaddrs->clear();
    if (indices) indices->clear();
    sublayout.clear();

    bool sharedOK = true;

    for (int b = 0; b < int(layout.size()); b++) {
        auto &block = layout[b];
        if (block.offsetAddr == 0) sharedOK = true;
        int qq1Unclamped = x1 - block.*offsetQ;
        int qq2Unclamped = x2 - block.*offsetQ;
        int qq1 = clamp<int>(qq1Unclamped, 0, block.*nq);
        int qq2 = clamp<int>(qq2Unclamped, 0, block.*nq);
        if (qq2 > qq1) {
            RegisterBlock subblock;
            if (!getSubblock(T, subblock, block, column, qq1, qq2, qq1Unclamped, qq2Unclamped, overrunOK, atype, astrategy)) {
                status << "Could not make subblock." << status_stream::endl;
                return false;
            }
            if (subblock.offsetR != block.offsetR || subblock.offsetC != block.offsetC) {
                status << "Subblock is not aligned to parent block." << status_stream::endl;
                return false;
            }
            if (subblock.offsetAddr != 0 && !sharedOK) {
                status << "Subblock has a shared address register." << status_stream::endl;
                return false;
            }
            if (subaddrs) subaddrs->push_back((*addrs)[b]);
            if (indices) indices->push_back(int(b));
            sublayout.push_back(subblock);
        }
        else if (block.offsetAddr == 0) sharedOK = false;
    }
    return true;
}

// Get list of subblocks intersecting rows/columns [x1, x2), and associated address registers.
// Returns false if fragmenting failed, or an address register doesn't match a previous one.
template <HW hw>
bool BLASKernelGenerator<hw>::getSubblocks(Type T, vector<RegisterBlock> &sublayout, vector<GRFRange> &subaddrs,
                                           const vector<RegisterBlock> &layout, const vector<GRFRange> &addrs,
                                           bool column, int x1, int x2, bool overrunOK,
                                           const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    return getSubblocks(T, sublayout, &subaddrs, nullptr, layout, &addrs, column, x1, x2, overrunOK, atype, astrategy);
}

// Get list of subblocks intersecting rows/columns [x1, x2), and indices of associated address registers.
// Returns false if fragmenting failed, or an address register doesn't match a previous one.
template <HW hw>
bool BLASKernelGenerator<hw>::getSubblocks(Type T, vector<RegisterBlock> &sublayout, vector<int> &indices, const vector<RegisterBlock> &layout,
                                           bool column, int x1, int x2, bool overrunOK,
                                           const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    return getSubblocks(T, sublayout, nullptr, &indices, layout, nullptr, column, x1, x2, overrunOK, atype, astrategy);
}

// Adjust address registers as needed for a newly-created subblock.
template <HW hw>
void BLASKernelGenerator<hw>::adjustSubblockAddrs(Type T, const vector<RegisterBlock> &sublayout, const vector<GRFRange> &subaddrs,
                                                  const vector<RegisterBlock> &layout, const vector<GRFRange> &addrs,
                                                  const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                                  const CommonStrategy &strategy, const CommonState &state)
{
    bool a64 = (astrategy.base.getModel() == ModelA64);

    auto nsubs = int(sublayout.size());
    auto nblocks = int(layout.size());

    for (int isub = 0; isub < nsubs; isub++) {
        // Find parent block by comparing address registers.
        auto &subaddr = subaddrs[isub];
        const RegisterBlock *pptr = nullptr;
        for (int i = 0; i < nblocks; i++) {
            if (addrs[i].getBase() == subaddr.getBase()) {
                pptr = &layout[i];
                break;
            }
        }
        if (!pptr) stub();

        auto &block = *pptr;
        auto &subblock = sublayout[isub];

        auto off = getAddr0Offset(block, atype, astrategy);
        auto suboff = getAddr0Offset(subblock, atype, astrategy);

        // Perform any necessary shifts/moves. Moves are only for non-A64 block->pseudoblock settings.
        if (suboff != off) {
            if (subblock.simdSize != 1) stub(); // Need to prepare more pseudoblock addresses.
            mov<uint32_t>(1, subaddr[0][suboff], subaddr[0][off]);
        }
        if (subblock.addrShift != block.addrShift) {
            map(hw, a64 ? Type::u64 : Type::u32, subaddr, subaddr, strategy, [&](int simd, GRF r, GRF _) {
                auto shift = block.addrShift - subblock.addrShift;
                (shift > 0) ? eshl(simd, r, r, +shift, strategy, state)
                            : eshr(simd, r, r, -shift, strategy, state);
            });
        }

        if (isBlock2D(astrategy.accessType)) {
            // Adjust 2D block header as needed.
            int bw, bh, bcount;
            bool memCM = isColMajor(atype.layout);
            auto RegisterBlock::* nw = memCM ? &RegisterBlock::nr : &RegisterBlock::nc;
            auto RegisterBlock::* nh = memCM ? &RegisterBlock::nc : &RegisterBlock::nr;
            bool remW = memCM ? subblock.remainderR : subblock.remainderC;
            bool remH = memCM ? subblock.remainderC : subblock.remainderR;
            getBlock2DWH(bw, bh, bcount, atype, subblock);

            if (!astrategy.address2D) {
                if (subblock.*nw != block.*nw || subblock.count != block.count) {
                    int newW = bw * bcount * subblock.ebytes - 1;
                    remW ? min_(1, subaddr[0].ud(2), subaddr[0].ud(2), newW)
                         : mov(1, subaddr[0].ud(2), newW);
                }
                if (subblock.*nh != block.*nh) {
                    int newH = bh * subblock.ebytes - 1;
                    remH ? min_(1, subaddr[0].ud(3), subaddr[0].ud(3), newH)
                         : mov(1, subaddr[0].ud(3), newH);
                }
            }
	    if (subaddr.isValid())
		updateBlock2DSizes(subaddr[0], subblock, block, atype);
        }
    }
}

// Update block 2D width/height/count parameters as needed after cloning an address register.
template <HW hw>
void BLASKernelGenerator<hw>::updateBlock2DSizes(GRF addr, const RegisterBlock &dst, const RegisterBlock &src, const MatrixAddressing &atype)
{
    int bw, bh, bcount;
    getBlock2DWH(bw, bh, bcount, atype, dst);

    if (dst.nr != src.nr || dst.nc != src.nc || dst.count != src.count)
        mov(1, addr.ud(7), (bw - 1) | ((bh - 1) << 8) | ((bcount - 1) << 16));
}

// Attempt to add remainder handling to a layout without changing its blocks. Returns true if successful.
template <HW hw>
bool BLASKernelGenerator<hw>::tryAddRemainder(Type T, vector<RegisterBlock> &layout, bool remainderR, bool remainderC, RemainderOptions remOpts,
                                              const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    auto layoutNew = layout;
    for (auto &block: layoutNew) {
        if (!tryAddRemainder(T, block, remainderR, remainderC, remOpts, atype, astrategy))
            return false;
    }
    std::swap(layout, layoutNew);
    return true;
}

// Add remainder handling to a layout without changing its blocks. Throws if unsuccessful.
template <HW hw>
void BLASKernelGenerator<hw>::addRemainder(Type T, vector<RegisterBlock> &layout, bool remainderR, bool remainderC, RemainderOptions remOpts,
                                           const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    for (auto &block: layout)
        if (!tryAddRemainder(T, block, remainderR, remainderC, remOpts, atype, astrategy)) stub();
}

// Add remainder handling to a layout, setting it up again from scratch if required.
template <HW hw>
void BLASKernelGenerator<hw>::addRemainder(Type T, vector<RegisterBlock> &layout, vector<GRFRange> &addrs, const Subregister &ld,
                                           bool remainderR, bool remainderC, RemainderOptions remOpts,
                                           const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                           const CommonStrategy &strategy, CommonState &state, int dataRegs)
{
    // Check if masking can be trivially enabled without changing the layout.
    if (tryAddRemainder(T, layout, remainderR, remainderC, remOpts, atype, astrategy))
        return;

    // If not, tear down the old layout and create a new one in its place, recalculating address registers.
    vector<RegisterBlock> layoutNew;
    int r, c;
    bool remR = remainderR || hasRemainders(layout, true, false);
    bool remC = remainderC || hasRemainders(layout, false, true);
    getLayoutDims(layout, r, c);
    if (!getRegLayout(T, layoutNew, r, c, remR, remC, false, remOpts, 0, 0, atype, astrategy)) stub();
    if (dataRegs < 0) dataRegs = getRegCount(layout);
    if (getRegCount(layoutNew) > dataRegs) stub();
    if (isLayoutColMajor(layoutNew) != isLayoutColMajor(layout)) stub();

    int shift = 0;
    auto addr0 = getOriginAddr(layout, addrs, atype, astrategy, &shift);
    std::swap(layout, layoutNew);
    if (shift > 0)
        shl(1, addr0, addr0, shift);
    safeReleaseRanges(addrs, state);
    state.ra.claim(addr0);

    Address2DParams params2D{};
    if (astrategy.address2D) stub();
    allocAddrRegs(addrs, layout, atype, astrategy, state);
    setupAddr(T, addrs, addr0, layout, ld, atype, astrategy, strategy, state, params2D);

    state.ra.safeRelease(addr0);
}

// Check if descriptor-based remainders are available for the given set of parameters.
//   Returns zero if not, otherwise the maximum fragment size.
template <HW hw>
int BLASKernelGenerator<hw>::checkDescriptorRemainder(Type T, int r, int c, bool column, bool writable,
                                                      const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    RegisterBlock block;
    int rblock = 0, cblock = 0;
    bool success = getBlockInfo(T, atype, astrategy, r, c, !column, column, writable, AllowFragDesc, 0, 0,
                                rblock, cblock, block);

    if (!success) return false;
    if (r % rblock || c % cblock) return false;
    if (!(column ? block.descRemC : block.descRemR)) return false;

    return column ? block.colFragment : block.rowFragment;
}

// Attempt to create a 2D block layout that matches an existing layout.
// Currently only generates regular/transpose 2D block (no VNNI support).
template <HW hw>
bool BLASKernelGenerator<hw>::upgradeLayoutToBlock2D(Type T, const vector<RegisterBlock> &layoutSrc, vector<RegisterBlock> &layout2D,
                                                     bool remainderR, bool remainderC, bool writable,
                                                     const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    layout2D.clear();
    layout2D.reserve(layoutSrc.size());

    if (layoutSrc.empty()) return true;
    if (isPacked(atype.layout)) return false;

    bool transpose = isTransposing(astrategy.accessType);
    bool regCM = isLayoutColMajor(layoutSrc);

    if (transpose && !one_of(sizeof(T), 4u, 8u))
        return false;

    int r0 = -1, c0 = -1, b0 = -1;
    int nr = 0, nc = 0;
    bool ok = true;

    auto make2DBlock = [&] {
        if (r0 < 0 || c0 < 0) return;
        ok = ok && addToRegLayout(T, layout2D, nr, nc, r0, c0, remainderR, remainderC, writable, AllowFragment, 0, 0, atype, astrategy);
    };

    for (size_t i = 0; i < layoutSrc.size(); i++) {
        auto &block = layoutSrc[i];
        unsigned omask = GRF::bytes(hw) - 1;

        if ((block.offsetBytes & omask) || (block.bytes & omask)) return false;
        if (!transpose && (block.colMajor ? block.nr : block.nc) * T > 64) return false;    /* avoid lots of small blocks */

        bool consecutive = (block.offsetBytes == (b0 + GRF::bytes(hw)));
        if (regCM && block.offsetC == c0 + nc && consecutive && nr == block.nr)
            nc++;
        else if (!regCM && block.offsetR == r0 + nr && consecutive && nc == block.nc)
            nr++;
        else {
            make2DBlock();
            r0 = block.offsetR; c0 = block.offsetC;
            nr = block.nr; nc = block.nc;
        }
        b0 = block.offsetBytes;
    }

    make2DBlock();

    int r = 0, c = 0;
    getLayoutDims(layoutSrc, r, c);
    sortRegLayout(T, layout2D, r, c, atype, astrategy);
    postprocessLayout(T, layout2D, atype, astrategy);
    finalizeLayout(hw, T, layout2D, atype, astrategy);

    // Update offsets to match source layout.
    for (auto &block: layout2D) {
        int ne;
        const RegisterBlock *blockRef;
        auto sr = findBlockReg(T, layoutSrc, block.offsetR, block.offsetC, GRFRange(0, 254), ne, blockRef);

        block.offsetBytes = (sr.getBase() << block.log2GRFBytes) + sr.getByteOffset();
    }

    return ok;
}

// Copy layoutSrc to layoutDst, breaking up blocks as needed so all destination blocks are also
//   subblocks of layoutRef.
template <HW hw>
bool BLASKernelGenerator<hw>::reblockLayout(Type Tdst, vector<int32_t> &blockMap, vector<RegisterBlock> &layoutDst,
                                            const vector<RegisterBlock> &layoutRef, const vector<RegisterBlock> &layoutSrc,
                                            const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    auto nblockRef = layoutRef.size();
    layoutDst.clear();
    layoutDst.reserve(nblockRef);
    blockMap.clear();
    blockMap.reserve(nblockRef + 1);
    blockMap.push_back(0);
    for (auto &blockRef : layoutRef) {
        RegisterBlock blockDst, blockMid;
        for (auto &blockSrc : layoutSrc) {
            int rr1 = blockRef.offsetR - blockSrc.offsetR, rr2 = rr1 + blockRef.nr;
            int cc1 = blockRef.offsetC - blockSrc.offsetC, cc2 = cc1 + blockRef.nc;
            if (rr1 >= blockSrc.nr || rr2 <= 0) continue;
            if (cc1 >= blockSrc.nc || cc2 <= 0) continue;
            rr1 = std::max(rr1, 0);
            cc1 = std::max(cc1, 0);
            rr2 = std::min(rr2, int(blockSrc.nr));
            cc2 = std::min(cc2, int(blockSrc.nc));
            if (!getSubblock(Tdst, blockMid, blockSrc, false, rr1, rr2, rr1, rr2, true, atype, astrategy)) return false;
            if (!getSubblock(Tdst, blockDst, blockMid, true,  cc1, cc2, cc1, cc2, true, atype, astrategy)) return false;
            layoutDst.push_back(blockDst);
        }
        blockMap.push_back(int32_t(layoutDst.size()));
    }
    return true;
}

#include "internal/namespace_end.hxx"

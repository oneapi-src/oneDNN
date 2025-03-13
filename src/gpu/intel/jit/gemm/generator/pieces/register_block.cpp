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


#include "register_block.hpp"
#include "strategy.hpp"
#include "internal/utils.hpp"

using namespace ngen;
using namespace ngen::utils;

#include "internal/namespace_start.hxx"

constexpr int8_t RegisterBlock::Interleaved;

void RegisterBlock::calcBytes(Type T, const MatrixAddressingStrategy &astrategy)
{
    if (astrategy.newDP && astrategy.prefetch)
        bytes = 0;
    else
        calcBytes(T);
}

void RegisterBlock::calcBytes(Type T)
{
    if (cxComponent != Interleaved)
        T = T.real();
    bytes = align_up(colMajor ? nc : nr, crosspack) * ld * T;
    if (isLoadBlock() && msgRegs == 0)
        msgRegs = (bytes + (1 << log2GRFBytes) - 1) >> log2GRFBytes;
}

int RegisterBlock::nregs() const
{
    if (!grfAligned()) stub();
    auto grfBytes = (1 << log2GRFBytes);
    return (bytes + grfBytes - 1) >> log2GRFBytes;
}

int RegisterBlock::offsetReg() const
{
    if (!grfAligned()) stub();
    return offsetBytes >> log2GRFBytes;
}

void RegisterBlock::simplify(Type T)
{
    // If block is completely crosspacked, convert to equivalent layout without crosspack.
    if (crosspack == (colMajor ? nc : nr) && isLargeCrosspack(T, crosspack)) {
        auto od = colMajor ? nr : nc;
        if (ld == od) {
            colMajor = !colMajor;
            ld = crosspack;
            crosspack = 1;
        }
    }
}

// Make a RegisterBlock smaller by contracting the leading dimension, if possible.
void RegisterBlock::compact(Type T)
{
    auto newLD = std::max<int>(roundup_pow2(colMajor ? nr : nc), (1 << log2GRFBytes) / T);
    if (newLD < ld) {
        ld = newLD;
        calcBytes(T);
    }
}

#include "internal/namespace_end.hxx"

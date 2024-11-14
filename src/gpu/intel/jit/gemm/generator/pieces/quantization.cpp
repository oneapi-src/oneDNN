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


#include "layout_utils.hpp"

using namespace ngen;
using std::vector;

#include "internal/namespace_start.hxx"


bool canDequantizeInt4(Type Tsrc, Type Tdst,
                       const vector<RegisterBlock> &layoutSrc, const vector<RegisterBlock> &layoutDst,
                       const vector<RegisterBlock> layoutOffset, const vector<RegisterBlock> layoutScale)
{
    if (!Tsrc.isInt4() || !one_of(Tdst, Type::f16, Type::bf16, Type::f32))
        return false;

    if (layoutOffset.empty() || layoutScale.empty()) {
        int m, n, md, nd;
        getLayoutDims(layoutSrc, m, n);
        getLayoutDims(layoutDst, md, nd);

        if (m < md || n < nd) return false;
    }

    return true;
}

#include "internal/namespace_end.hxx"

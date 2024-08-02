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


#include "problem.hpp"

using namespace ngen;

#include "internal/namespace_start.hxx"


// Transpose a GEMM problem.
void GEMMProblem::transpose()
{
    std::swap(A, B);
    std::swap(AO, BO);
    std::swap(A_scale, B_scale);
    std::swap(Ta, Tb);
    std::swap(Ta_ext, Tb_ext);
    std::swap(Tao, Tbo);
    std::swap(Ta_scale, Tb_scale);
    std::swap(aOffset, bOffset);
    std::swap(aoPtrDims, boPtrDims);
    std::swap(aScale2D, bScale2D);
    std::swap(sumA, sumB);
    std::swap(binaryRow, binaryCol);
    binaryTrans.flip();
    for (auto &bsrc: binary)
        bsrc.transpose();
    A.transpose();
    B.transpose();
    C.transpose();
    AO.transpose();
    BO.transpose();
    CO.transpose();
}

#include "internal/namespace_end.hxx"

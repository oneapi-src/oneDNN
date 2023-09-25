/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef KERNEL_EVALUATOR_HPP
#define KERNEL_EVALUATOR_HPP

#include "kernel_catalog.hpp"

#include "gen_gemm_kernel_generator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct SizeParams {
    int64_t batch = 0;
    int64_t m = 0, n = 0, k = 0;
};

struct EvaluateParams {
    SizeParams sizes;
    double alpha, beta;
    int euCount;
    int tileCount = 1;
    bool effective = false;
    bool cConvert = false;
    bool postOps = false;
    bool batch = false;
};

struct DerivedEvaluateParams : public EvaluateParams {
    int64_t wgCountM, wgCountN, wgCountK;
    int64_t mPad, nPad;
    double threadCount;
    int threadsPerEU;
    int hwThreadCapacity;
    int hwThreadsPartialWave;
    int partialWaveCount;
    bool autoatomic;
};

struct EvaluateAuxOutput {
    int64_t k0 = 0;
    int wgK = 1;
    bool kParallel = false;
    bool kParallelVariable = false;
};

DerivedEvaluateParams getDerivedParams(
        const kcatalog::Entry &e, const EvaluateParams &p);
double evaluate(const kcatalog::Entry &e, const EvaluateParams &p,
        EvaluateAuxOutput &aux);
double evaluate(const kcatalog::Entry &e, const DerivedEvaluateParams &p,
        EvaluateAuxOutput &aux);

void modifyStrategy(GEMMStrategy &strategy, const EvaluateAuxOutput &aux);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* header guard */

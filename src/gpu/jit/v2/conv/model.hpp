/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_JIT_V2_CONV_MODEL_HPP
#define GPU_JIT_V2_CONV_MODEL_HPP

#include "gpu/jit/v2/conv/bench_data.hpp"
#include "gpu/jit/v2/conv/ml.hpp"
#include "gpu/jit/v2/conv/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

class model_t {
public:
    model_t() = default;
    model_t(const kernel_desc_t &kernel_desc, const ml_model_t &ml_model)
        : kernel_desc_(kernel_desc), ml_model_(ml_model) {}
    float predict(const problem_t &prb) const;
    float eff(const problem_t &prb) const;
    void score(const bench_data_t &bd);
    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);

private:
    kernel_desc_t kernel_desc_;
    ml_model_t ml_model_;
};

void to_model_xy(const bench_data_t &bd, vec2d &X, vec1d &y);

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

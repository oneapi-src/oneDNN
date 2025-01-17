/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_MODEL_HPP
#define GPU_INTEL_JIT_V2_CONV_MODEL_HPP

#include "gpu/intel/jit/v2/conv/bench_data.hpp"
#include "gpu/intel/jit/v2/conv/problem.hpp"
#include "gpu/intel/serialization.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

using vec1d = std::vector<float>;
using vec2d = std::vector<std::vector<float>>;

enum class model_kind_t : uint8_t {
    undef = 0,
    data_parallel = 1,
};

static auto model_kind_names = nstl::to_array({
        make_enum_name(model_kind_t::undef, "undef"),
        make_enum_name(model_kind_t::data_parallel, "data_parallel"),
});
GPU_DEFINE_PARSE_ENUM(model_kind_t, model_kind_names)

class model_t {
public:
    model_t() = default;
    model_t(model_kind_t kind, const vec1d &coef) : kind_(kind), coef_(coef) {}
    bool is_empty() const { return kind_ == model_kind_t::undef; }
    model_kind_t kind() const { return kind_; }
    const vec1d &coef() const { return coef_; }
    float predict(const vec1d &x) const;
    float predict(const problem_t &prb, const kernel_desc_t &desc) const;
    float eff(const problem_t &prb, const kernel_desc_t &desc) const;
    void score(const bench_data_t &bd);
    void stringify(std::ostream &out) const;
    void parse(std::istream &in);

    static float predict(model_kind_t kind, const vec1d &x, const vec1d &coef);
    static size_t coef_count(model_kind_t kind);

private:
    model_kind_t kind_;
    vec1d coef_;
};

class model_set_t {
public:
    model_set_t() = default;
    model_set_t(const model_t &model) { models_.push_back(model); }
    float eff(const problem_t &prb, const kernel_desc_t &desc) const;
    void stringify(std::ostream &out) const;
    void parse(std::istream &in);

private:
    std::vector<model_t> models_;
};

void to_model_data(
        model_kind_t kind, const bench_data_t &bd, vec2d &X, vec1d &y);
void dump_csv(const bench_data_t &bd, const model_t &model);
void dump_model_params(const kernel_desc_t &kernel_desc, const model_t &model);

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

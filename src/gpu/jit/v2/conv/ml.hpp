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

#ifndef GPU_JIT_V2_CONV_ML_HPP
#define GPU_JIT_V2_CONV_ML_HPP

#include <iostream>
#include <memory>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {

using vec1d = std::vector<float>;
using vec2d = std::vector<std::vector<float>>;

enum class ml_model_kind_t {
    undef,
    linear_regression,
};

class ml_model_impl_t {
public:
    virtual ml_model_kind_t kind() const = 0;
    virtual float predict(const vec1d &x) const = 0;
    virtual void serialize(std::ostream &out) const = 0;
    virtual void deserialize(std::istream &in) = 0;
};

class linear_model_t final : public ml_model_impl_t {
public:
    linear_model_t() = default;
    linear_model_t(const vec1d &coef) : coef_(coef) {}
    ml_model_kind_t kind() const override {
        return ml_model_kind_t::linear_regression;
    }
    float predict(const vec1d &x) const override;
    void serialize(std::ostream &out) const override;
    void deserialize(std::istream &in) override;

private:
    vec1d coef_;
};

class ml_model_t {
public:
    ml_model_t() = default;

    template <typename MLModelT>
    ml_model_t(const MLModelT &impl)
        : impl_(std::make_shared<MLModelT>(impl)) {}

    ml_model_kind_t kind() const {
        return impl_ ? impl_->kind() : ml_model_kind_t::undef;
    }
    float predict(const vec1d &x) const { return impl_->predict(x); }
    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);

private:
    std::shared_ptr<ml_model_impl_t> impl_;
};

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

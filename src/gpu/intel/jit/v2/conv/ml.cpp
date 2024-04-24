/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/intel/jit/v2/conv/ml.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

float linear_model_t::predict(const vec1d &x) const {
    ir_assert(x.size() == coef_.size() - 1);
    float ret = coef_[0];
    for (size_t i = 0; i < x.size(); i++)
        ret += coef_[i + 1] * x[i];
    return ret;
}

void linear_model_t::serialize(std::ostream &out) const {
    ir_utils::serialize(coef_, out);
}

void linear_model_t::deserialize(std::istream &in) {
    coef_ = ir_utils::deserialize<std::vector<float>>(in);
}

void ml_model_t::serialize(std::ostream &out) const {
    ir_utils::serialize(kind(), out);
    if (impl_) impl_->serialize(out);
}

void ml_model_t::deserialize(std::istream &in) {
    auto kind = ir_utils::deserialize<ml_model_kind_t>(in);
    switch (kind) {
        case ml_model_kind_t::undef: impl_.reset(); break;
        case ml_model_kind_t::linear_regression:
            impl_ = std::make_shared<linear_model_t>();
            break;
        default: ir_error_not_expected();
    }
    if (impl_) impl_->deserialize(in);
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

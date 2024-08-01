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

void linear_model_t::serialize(serialized_data_t &s) const {
    s.append(coef_);
}

linear_model_t linear_model_t::deserialize(deserializer_t &d) {
    linear_model_t m;
    d.pop(m.coef_);
    return m;
}

void ml_model_t::serialize(serialized_data_t &s) const {
    s.append(kind());
    if (!impl_) return;
    switch (impl_->kind()) {
        case ml_model_kind_t::linear_regression:
            static_cast<const linear_model_t *>(impl_.get())->serialize(s);
            break;
        default: ir_error_not_expected();
    }
}

ml_model_t ml_model_t::deserialize(deserializer_t &d) {
    ml_model_kind_t kind;
    d.pop(kind);
    switch (kind) {
        case ml_model_kind_t::undef: return ml_model_t();
        case ml_model_kind_t::linear_regression: {
            ml_model_t m;
            m.impl_ = std::make_shared<linear_model_t>(
                    linear_model_t::deserialize(d));
            return m;
        }
        default: ir_error_not_expected();
    }
    return ml_model_t();
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

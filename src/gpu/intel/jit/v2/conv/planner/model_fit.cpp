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

#include "gpu/intel/jit/v2/conv/planner/model_fit.hpp"

#include "gpu/intel/jit/v2/conv/planner/mkl_iface.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

linear_model_t linear_model_fit(const vec2d &X, const vec1d &y) {
    int nsamples = (int)y.size();
    int nfeatures = (int)X[0].size();
    std::vector<float> _X(nsamples * (nfeatures + 1));
    auto _y = y;
    int idx = 0;
    for (int i = 0; i < nsamples; i++)
        _X[idx++] = 1.0f;
    for (int j = 0; j < nfeatures; j++) {
        for (int i = 0; i < nsamples; i++) {
            _X[idx++] = X[i][j];
        }
    }
    int m = nsamples;
    int n = nfeatures + 1;
    int nrhs = 1;
    int lda = m;
    int ldb = std::max(m, n);
    _y.resize(std::max(m, n));
    float *a = _X.data();
    float *b = _y.data();
    std::vector<float> s_vec(std::min(m, n));
    int LAPACK_COL_MAJOR = 102;
    float rcond = -1;
    int rank = 0;
    int info = mkl_iface_t::instance().LAPACKE_sgelsd(LAPACK_COL_MAJOR, m, n,
            nrhs, a, lda, b, ldb, s_vec.data(), rcond, &rank);
    ir_assert(info == 0);
    _y.resize(n);
    return linear_model_t(_y);
}

ml_model_t ml_model_fit(ml_model_kind_t kind, const vec2d &X, const vec1d &y) {
    switch (kind) {
        case ml_model_kind_t::linear_regression: {
            auto model = linear_model_fit(X, y);
            return ml_model_t(model);
        }
        default: ir_error_not_expected();
    }
    return ml_model_t();
}

model_t model_fit(const bench_data_t &bd) {
    if (!bd) {
        std::cout << "Warning: empty bench_data." << std::endl;
        return model_t();
    }
    vec2d X;
    vec1d y;
    to_model_xy(bd, X, y);
    auto ml_model = ml_model_fit(ml_model_kind_t::linear_regression, X, y);
    return model_t(bd.kernel_desc, ml_model);
}

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

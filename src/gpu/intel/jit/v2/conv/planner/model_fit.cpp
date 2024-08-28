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

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

namespace {

float r2_score(const vec2d &X, const vec1d &y, const vec1d &coef) {
    std::vector<float> y_true, y_pred;
    for (size_t i = 0; i < X.size(); i++) {
        y_true.push_back(y[i]);
        y_pred.push_back(model_t::predict(X[i][0], X[i][1], coef));
    }
    float u = 0;
    float v = 0;
    float y_mean = 0;
    int n = (int)y_true.size();
    for (int i = 0; i < n; i++)
        y_mean += y_true[i];
    y_mean /= n;
    for (int i = 0; i < n; i++) {
        u += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        v += (y_true[i] - y_mean) * (y_true[i] - y_mean);
    }
    return 1 - u / v;
}

struct model_params_t {
    struct param_t {
        std::string name;
        float val = 0;
        float lo = 0;
        float hi = 0;
        float step = 0;

        param_t(const std::string &name, float val, float lo, float hi)
            : name(name), val(val), lo(lo), hi(hi) {
            step = (hi - lo) / 5;
        }

        void set(float v) { val = std::min(hi, std::max(lo, v)); }

        float operator()() const { return val; }
    };

    void add(const std::string &name, float val, float lo, float hi) {
        vec.emplace_back(param_t(name, val, lo, hi));
    }

    param_t &operator[](int idx) { return vec[idx]; }
    const param_t &operator[](int idx) const { return vec[idx]; }
    int size() const { return (int)vec.size(); }

    std::string str() const {
        std::ostringstream oss;
        bool is_first = true;
        oss << "(";
        for (auto &p : vec) {
            if (!is_first) oss << ", ";
            oss << p.val;
            is_first = false;
        }
        oss << ")";
        return oss.str();
    }

    std::vector<param_t> vec;
};

float r2_score(const vec2d &X, const vec1d &y, const model_params_t &params) {
    vec1d coef;
    for (int i = 0; i < params.size(); i++)
        coef.push_back(params[i].val);
    return r2_score(X, y, coef);
}

void find_optimal_param(
        model_params_t &params, int idx, const vec2d &X, const vec1d &y) {
    auto &p = params[idx];
    float step = p.step;
    for (int iter = 0; iter < 10; iter++) {
        float p_val = p.val;
        float p_val_best = p_val;
        float r2_best = r2_score(X, y, params);
        for (int sign : {-1, 1}) {
            p.set(p_val + sign * step);
            float r2 = r2_score(X, y, params);
            if (r2 > r2_best) {
                p_val_best = p.val;
                r2_best = r2;
            }
        }
        p.val = p_val_best;
        step /= 2;
    }
    p.step /= 2;
}

} // namespace

model_t model_fit(const vec2d &X, const vec1d &y, bool verbose = false) {
    model_params_t params;
    // Empirically-based parameter ranges.
    params.add("T0", 1000, 1, 100000);
    params.add("a_kl", 1, 0.0001, 100);
    params.add("b_kl", 1, 0.0001, 100);
    params.add("a_wp", 2, 1, 100);
    params.add("b_wp", 1, 0.0001, 100);
    int nparams = params.size();
    // Perform a coordinate descent search optimizing one parameter at a time.
    // The goal is to maximize R2. See conv/model.cpp file for more details on
    // modeling.
    int niters = 10 * nparams;
    for (int i = 0; i < niters; i++) {
        find_optimal_param(params, i % nparams, X, y);
    }
    if (verbose) {
        std::cout << "R2: " << r2_score(X, y, params) << " (cases: " << X.size()
                  << ") model params = " << params.str() << std::endl;
    }
    vec1d coef;
    for (int i = 0; i < params.size(); i++)
        coef.push_back(params[i].val);
    return model_t(coef);
}

model_t model_fit(const bench_data_t &bd) {
    if (!bd) {
        std::cout << "Warning: empty bench_data." << std::endl;
        return model_t();
    }
    // Step 1. Fit model.
    vec2d X;
    vec1d y;
    to_model_xy(bd, X, y);
    auto ml_model = model_fit(X, y);

    // Step 2. Remove outliers where the fitted model predicts significantly
    // higher times. For example this may happen due to better L1 cache reuse
    // for some shapes which is hard to control. It is preferable for the model
    // to overestimate time rather than underestimate it.
    vec2d X_adjusted;
    vec1d y_adjusted;
    for (size_t i = 0; i < X.size(); i++) {
        float pred = ml_model.predict(X[i]);
        if ((pred - y[i]) > 0.25 * y[i]) continue;
        X_adjusted.push_back(X[i]);
        y_adjusted.push_back(y[i]);
    }
    ml_model = model_fit(X_adjusted, y_adjusted, /*verbose=*/true);
    dump_csv(bd, ml_model);
    return model_t(ml_model);
}

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

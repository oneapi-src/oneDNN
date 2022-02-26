/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#ifndef UTILS_NORM_HPP
#define UTILS_NORM_HPP

#include <limits>

#include "common.hpp"

struct norm_t {
    /* strictly speaking L0 is not a norm... it stands for the biggest
     * absolute element-wise difference and is used in diff_norm_t only */
    enum { L0, L1, L2, LINF, L8 = LINF, L_LAST };

    norm_t() : num_(0) {
        for (int i = 0; i < L_LAST; ++i)
            norm_[i] = 0;
    }

    void update(float v) {
        norm_[L1] += ABS(v);
        norm_[L2] += v * v;
        norm_[L8] = MAX2(norm_[L8], ABS(v));
        num_++;
    }

    void done() { norm_[L2] = sqrt(norm_[L2]); }

    float operator[](int type) const { return norm_[type]; }

    double norm_[L_LAST];
    size_t num_;
};

struct diff_norm_t {
    void update(float a, float b) {
        float diff = a - b;
        a_.update(a);
        b_.update(b);
        diff_.update(diff);
        diff_.norm_[norm_t::L0] = MAX2(diff_.norm_[norm_t::L0],
                ABS(diff) / (ABS(a) > FLT_MIN ? ABS(a) : 1.));
    }
    void done() {
        a_.done();
        b_.done();
        diff_.done();
    }

    float rel_diff(int type) const {
        if (type == norm_t::L0) return diff_.norm_[type];
        if (a_.norm_[type] == 0)
            return diff_.norm_[type] == 0
                    ? 0
                    : std::numeric_limits<float>::infinity();
        assert(a_.norm_[type] != 0);
        return diff_.norm_[type] / a_.norm_[type];
    }

    norm_t a_, b_, diff_;
};

#endif

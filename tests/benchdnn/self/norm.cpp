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

#include "utils/norm.hpp"
#include "utils/compare.hpp"

#include "self/self.hpp"

namespace self {

static int check_norm() {
    norm_t n0, n1;
#define N 10
    for (int i = 1; i <= N; i++) {
        n0.update(i);
        n1.update(i + 1);
    }

    int exp_sum0 = N * (N + 1) / 2;
    int exp_sq_sum0 = N * (N + 1) * (2 * N + 1) / 6;
    int exp_max0 = N;

    int exp_sum1 = ((N + 1) * (N + 2) / 2) - 1;
    int exp_sq_sum1 = ((N + 1) * (N + 2) * (2 * N + 3) / 6) - 1;
    int exp_max1 = N + 1;

    SELF_CHECK_EQ(n0.norm_[norm_t::L1], exp_sum0);
    SELF_CHECK_EQ(n0.norm_[norm_t::L2], exp_sq_sum0);
    SELF_CHECK_EQ(n0.norm_[norm_t::L8], exp_max0);

    SELF_CHECK_EQ(n1.norm_[norm_t::L1], exp_sum1);
    SELF_CHECK_EQ(n1.norm_[norm_t::L2], exp_sq_sum1);
    SELF_CHECK_EQ(n1.norm_[norm_t::L8], exp_max1);

    n1.update(n0);
    n1.done();

    SELF_CHECK_EQ(n1.norm_[norm_t::L1], exp_sum0 + exp_sum1);
    SELF_CHECK(n1.norm_[norm_t::L2] - sqrt(exp_sq_sum0 + exp_sq_sum1)
            <= 10.f * epsilon_dt(dnnl_f32));
    SELF_CHECK_EQ(n1.norm_[norm_t::L8], exp_max1);
#undef N
    return OK;
}

static int check_diff_norm() {
    diff_norm_t n0, n1;

#define N 10
    for (int i = 1; i <= N; i++) {
        n0.update(i, i + 1);
        n1.update(i, i + 1);
    }

    int exp_sum0 = N * (N + 1) / 2;
    int exp_sq_sum0 = N * (N + 1) * (2 * N + 1) / 6;
    int exp_max0 = N;

    int exp_sum1 = ((N + 1) * (N + 2) / 2) - 1;
    int exp_sq_sum1 = ((N + 1) * (N + 2) * (2 * N + 3) / 6) - 1;
    int exp_max1 = N + 1;

    SELF_CHECK_EQ(n0.a_[norm_t::L1], exp_sum0);
    SELF_CHECK_EQ(n0.a_[norm_t::L2], exp_sq_sum0);
    SELF_CHECK_EQ(n0.a_[norm_t::L8], exp_max0);

    SELF_CHECK_EQ(n0.b_[norm_t::L1], exp_sum1);
    SELF_CHECK_EQ(n0.b_[norm_t::L2], exp_sq_sum1);
    SELF_CHECK_EQ(n0.b_[norm_t::L8], exp_max1);

    SELF_CHECK_EQ(n0.diff_[norm_t::L1], N);
    SELF_CHECK_EQ(n0.diff_[norm_t::L2], N);
    SELF_CHECK_EQ(n0.diff_[norm_t::L8], 1);

    n1.update(n0);
    n1.done();

    SELF_CHECK_EQ(n1.a_[norm_t::L1], 2 * exp_sum0);
    SELF_CHECK(n1.a_[norm_t::L2] - sqrt(2 * exp_sq_sum0)
            <= 10.f * epsilon_dt(dnnl_f32));
    SELF_CHECK_EQ(n1.a_[norm_t::L8], exp_max0);

    SELF_CHECK_EQ(n1.b_[norm_t::L1], 2 * exp_sum1);
    SELF_CHECK(n1.b_[norm_t::L2] - sqrt(2 * exp_sq_sum1)
            <= 10.f * epsilon_dt(dnnl_f32));
    SELF_CHECK_EQ(n1.b_[norm_t::L8], exp_max1);

    SELF_CHECK_EQ(n1.diff_[norm_t::L1], 2 * N);
    SELF_CHECK(
            n1.diff_[norm_t::L2] - sqrt(2 * N) <= 10.f * epsilon_dt(dnnl_f32));
    SELF_CHECK_EQ(n1.diff_[norm_t::L8], 1);

#undef N
    return OK;
}

static int check_compare_norm() {
    dnnl_dim_t dims {10};
    dnn_mem_t m0(1, &dims, dnnl_f32, tag::abx, get_test_engine());
    dnn_mem_t m1(1, &dims, dnnl_f32, tag::abx, get_test_engine());

#define N 10
    for (int i = 1; i <= N; i++) {
        m0.set_elem(i - 1, i);
        m1.set_elem(i - 1, i + 1);
    }
    int exp_sq_sum0 = N * (N + 1) * (2 * N + 1) / 6;

    res_t res_bad {};
    res_bad.state = EXECUTED;
    compare::compare_t cmp;
    cmp.set_norm_validation_mode(true);
    cmp.set_threshold(
            sqrt(N) / sqrt(exp_sq_sum0) - 10.f * epsilon_dt(dnnl_f32));
    cmp.compare(m0, m1, attr_t(), &res_bad);
    SELF_CHECK_EQ(res_bad.state, FAILED);

    res_t res_good {};
    res_good.state = EXECUTED;
    cmp.set_threshold(
            sqrt(N) / sqrt(exp_sq_sum0) + 10.f * epsilon_dt(dnnl_f32));
    SAFE(cmp.compare(m0, m1, attr_t(), &res_good), WARN);
    SELF_CHECK_EQ(res_good.state, PASSED);
#undef N
    return OK;
}

void norm() {
    RUN(check_norm());
    RUN(check_diff_norm());
    RUN(check_compare_norm());
}

} // namespace self

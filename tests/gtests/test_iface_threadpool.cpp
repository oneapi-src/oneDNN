/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_threadpool.h"
#include "tests/test_isa_common.hpp"

namespace dnnl {

class threadpool_test_t : public ::testing::Test {
protected:
    void SetUp() override {}
};

void test_threadpool_maxconcurrency_st(dnnl_status_t &res) {
    int tid = std::hash<std::thread::id> {}(std::this_thread::get_id()) % 23;
    tid++; // to avoid zeros.

    auto multipliers = {1, 5, 7, 12, 24, 56};
    for (auto m : multipliers) {
        dnnl_status_t st = dnnl_success;

        int expected = tid * m % 29;
        st = dnnl_threadpool_interop_set_max_concurrency(expected);
        if (st != dnnl_success) {
            res = st;
            return;
        }

        int obtained = 0;
        st = dnnl_threadpool_interop_get_max_concurrency(&obtained);
        if (st != dnnl_success) {
            res = st;
            return;
        }

        if (expected != obtained) {
            res = dnnl_runtime_error;
            return;
        }
    }
    res = dnnl_success;
}

TEST_F(threadpool_test_t, TestMaxConcurrencyConcurrent) {
    const int nthreads = 100;
    std::vector<std::thread> threads;
    std::vector<dnnl_status_t> results(nthreads);
    for (int i = 0; i <= nthreads; i++)
        threads.emplace_back(
                test_threadpool_maxconcurrency_st, std::ref(results[i]));
    for (auto &t : threads)
        t.join();
    for (auto &r : results)
        ASSERT_EQ(r, dnnl_success);
}

} // namespace dnnl

/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "dnnl.hpp"
#include "dnnl_test_common.hpp"

#include "gtest/gtest.h"

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "dnnl_threadpool_iface.hpp"
class fake_threadpool : public dnnl::threadpool_iface {
    virtual int get_num_threads() const override { return 1; }
    virtual bool get_in_parallel() const override { return 0; }
    virtual void parallel_for(
            int n, const std::function<void(int, int)> &fn) override {
        fn(0, 1);
    }
    virtual uint64_t get_flags() const override { return 0; }
};
#endif

class stream_attr_test : public ::testing::Test {
protected:
    dnnl::stream_attr sa_cpu {get_test_engine_kind()};
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    fake_threadpool tp;
    bool expect_threadpool_failure
            = get_test_engine_kind() != dnnl::engine::kind::cpu;
#endif
    virtual void SetUp() {}
};

TEST_F(stream_attr_test, TestConstructor) {}

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
TEST_F(stream_attr_test, TestGetThreadPool) {
    catch_expected_failures([&] { sa_cpu.get_threadpool(); },
            expect_threadpool_failure, dnnl_invalid_arguments);
}

TEST_F(stream_attr_test, TestSetThreadPoolNULL) {
    catch_expected_failures([&] { sa_cpu.set_threadpool(nullptr); },
            expect_threadpool_failure, dnnl_invalid_arguments);
};

TEST_F(stream_attr_test, TestSetThreadPool) {
    catch_expected_failures([&] { sa_cpu.set_threadpool(&tp); },
            expect_threadpool_failure, dnnl_invalid_arguments);
};
#endif

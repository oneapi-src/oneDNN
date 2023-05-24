/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "gtest/internal/gtest-internal.h"

#include "unit_test_common.hpp"

#include "interface/c_types_map.hpp"

using namespace testing;
namespace graph = dnnl::impl::graph;

static std::string find_cmd_option(
        char **argv_beg, char **argv_end, const std::string &option) {
    for (auto arg = argv_beg; arg != argv_end; arg++) {
        std::string s(*arg);
        auto pos = s.find(option);
        if (pos != std::string::npos) return s.substr(pos + option.length());
    }
    return {};
}

inline graph::engine_kind_t to_engine_kind(const std::string &str) {
    if (str.empty() || str == "cpu") return graph::engine_kind::cpu;

    if (str == "gpu") return graph::engine_kind::gpu;

    assert(!"not expected");
    return graph::engine_kind::cpu;
}

int main(int argc, char *argv[]) {
    int result;

    auto engine_str = find_cmd_option(argv, argv + argc, "--engine=");
    set_test_engine_kind(to_engine_kind(engine_str));

#if DNNL_GPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (get_test_engine_kind() == graph::engine_kind::gpu) {
        std::cout << "GPU runtime is not enabled" << std::endl;
        return 0;
    }
#endif
    (void)(::testing::GTEST_FLAG(death_test_style) = "threadsafe");
    ::testing::InitGoogleTest(&argc, argv);
    std::string filter_str = ::testing::GTEST_FLAG(filter);
    if (get_test_engine_kind() == graph::engine_kind::cpu) {
        // Exclude non-CPU tests
        ::testing::GTEST_FLAG(filter) = filter_str + ":-*_GPU*";
    } else if (get_test_engine_kind() == graph::engine_kind::gpu) {
        // Exclude non-GPU tests
        ::testing::GTEST_FLAG(filter) = filter_str + ":-*_CPU*";
    }
    result = RUN_ALL_TESTS();

    return result;
}

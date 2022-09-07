/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <gtest/gtest.h>

#include "unit_test_common.hpp"

using namespace testing;
namespace impl = dnnl::graph::impl;

static std::string find_cmd_option(
        char **argv_beg, char **argv_end, const std::string &option) {
    for (auto arg = argv_beg; arg != argv_end; arg++) {
        std::string s(*arg);
        auto pos = s.find(option);
        if (pos != std::string::npos) return s.substr(pos + option.length());
    }
    return {};
}

inline impl::engine_kind_t to_engine_kind(const std::string &str) {
    if (str.empty() || str == "cpu") return impl::engine_kind::cpu;

    if (str == "gpu") return impl::engine_kind::gpu;

    assert(!"not expected");
    return impl::engine_kind::cpu;
}

int main(int argc, char *argv[]) {
    int result;

    auto engine_str = find_cmd_option(argv, argv + argc, "--engine=");
    set_test_engine_kind(to_engine_kind(engine_str));

#ifndef DNNL_GRAPH_GPU_SYCL
    if (get_test_engine_kind() == impl::engine_kind::gpu) {
        std::cout << "GPU runtime is not enabled" << std::endl;
        return 0;
    }
#endif

    ::testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();

    return result;
}

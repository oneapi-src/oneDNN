/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

// clang-format off
#include <test_utils.hpp>
// clang-format on
#include "test_utils_arr_fill.hpp"
#include <compiler/ir/graph/lowering.hpp>
#if SC_CFAKE_JIT_ENABLED
#include <compiler/jit/cfake/cfake_jit.hpp>
#endif
#include <compiler/jit/llvm/llvm_jit.hpp>
#if SC_BUILTIN_JIT_ENABLED
#include <compiler/jit/xbyak/xbyak_jit.hpp>
#endif
#include <runtime/config.hpp>

#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace std;

struct runtime_config_saver_t {
    bool execution_verbose;

    runtime_config_saver_t() {
        execution_verbose = runtime_config_t::get().execution_verbose_;
    }

    ~runtime_config_saver_t() {
        runtime_config_t::get().execution_verbose_ = execution_verbose;
    }
};

static map<string, shared_ptr<jit_engine_t>> test_jit_engines {
#if SC_CFAKE_JIT_ENABLED
    {"cfake_jit", make_shared<cfake_jit>()},
#endif
#ifdef SC_LLVM_BACKEND
            {"llvm_jit", make_shared<llvm_jit>()},
#endif
#if SC_BUILTIN_JIT_ENABLED
            {"xbyak_jit", make_shared<xbyak_jit>()},
#endif
};

TEST(GCCore_CPU_test_execution_verbose, TestTimer) {
    REQUIRE_AVX2();
    sc_graph_t g;
    auto ins = g.make_input(
            {graph_tensor::make({2, 2}), graph_tensor::make({2, 2})});
    auto add = g.make(
            "add", {ins->get_outputs()[0], ins->get_outputs()[1]}, {}, {});
    auto out = g.make_output(add->get_outputs());

    vector<float> v1(4), v2(4), outf(4);
    test_utils::fill_data(v1.data(), 4);
    test_utils::fill_data(v2.data(), 4);

    vector<generic_val> generic_args;
    generic_args.emplace_back((void *)v1.data());
    generic_args.emplace_back((void *)v2.data());
    generic_args.emplace_back((void *)outf.data());

    context_ptr ctx = make_shared<context_t>(*get_default_context());
    runtime_config_saver_t conf;
    runtime_config_t::get().execution_verbose_ = true;
    auto mod = lower_graph(ctx, g, {ins, out});
    for (auto &kv : test_jit_engines) {
        if (kv.first == "xbyak_jit"
                && !get_default_context()->machine_.cpu_flags_.fAVX512F) {
            continue;
        }
        testing::internal::CaptureStdout();
        shared_ptr<jit_engine_t> je = kv.second;
        auto jitf = je->get_entry_func(mod, true);
        jitf->call_generic_default(generic_args.data());
        std::string output = testing::internal::GetCapturedStdout();

        ASSERT_TRUE(output.find("Entry point: main_entry") != string::npos);
    }
}

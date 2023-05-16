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

#include "gtest/gtest.h"
#include <compiler/jit/cfake/cfake_jit.hpp>
#include <runtime/target_machine.hpp>
#include <util/utils.hpp>

namespace gc = dnnl::impl::graph::gc;
TEST(GCCore_CPU_targetmachine_cpp, TestTargetMachine) {
    // mainly test if we can compile and run the instructions (cpuid)
    {
        gc::runtime::target_machine_t tm
                = gc::runtime::get_native_target_machine();
        std::cout << "Native: AVX2:" << tm.cpu_flags_.fAVX2
                  << " AXV512:" << tm.cpu_flags_.fAVX512F
                  << " AXV512BF16:" << tm.cpu_flags_.fAVX512BF16
                  << " max bits:" << tm.cpu_flags_.max_simd_bits << '\n';
    }
#if SC_CFAKE_JIT_ENABLED
    {
        gc::runtime::target_machine_t tm
                = gc::runtime::get_native_target_machine();
        gc::cfake_jit::set_target_machine(tm);
        std::cout << "Compiler: AVX2:" << tm.cpu_flags_.fAVX2
                  << " AXV512:" << tm.cpu_flags_.fAVX512F
                  << " AXV512BF16:" << tm.cpu_flags_.fAVX512BF16
                  << " max bits:" << tm.cpu_flags_.max_simd_bits << '\n';
        // test bf16 flag
        std::vector<std::string> option = {"g++", "-v"};
        int exit_status;
        std::string rstderr;
        bool success = gc::utils::create_process_and_await(
                "g++", option, exit_status, nullptr, nullptr, &rstderr);
        if (success && !exit_status) {
            std::size_t vstart = rstderr.find("gcc version", 0) + 12;
            if (std::stoi(rstderr.substr(vstart, 2)) < 10) {
                ASSERT_EQ(tm.cpu_flags_.fAVX512BF16, false);
            }
        }
    }
#endif
}

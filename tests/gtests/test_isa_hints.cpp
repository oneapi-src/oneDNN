/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <map>
#include <set>

#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "src/cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {

using namespace impl::cpu::x64;

const std::set<cpu_isa_t> cpu_isa_all = {sse41, avx, avx2, avx2_vnni,
        avx512_mic, avx512_mic_4ops, avx512_core, avx512_core_vnni,
        avx512_core_bf16, avx512_core_bf16_amx_int8, avx512_core_bf16_amx_bf16,
        avx512_core_amx, isa_all};

class isa_hints_test_t : public ::testing::TestWithParam<cpu_isa_hints> {
protected:
    void SetUp() override {
        auto hints = ::testing::TestWithParam<cpu_isa_hints>::GetParam();

        // soft version of mayiuse that allows resetting the cpu_isa_hints
        auto test_mayiuse = [](cpu_isa_t isa) { return mayiuse(isa, true); };

        std::map<cpu_isa_t, bool> compat_before_hint;

        for (auto isa : cpu_isa_all)
            compat_before_hint[isa] = test_mayiuse(isa);

        status st = set_cpu_isa_hints(hints);
        // status::unimplemented if the feature was disabled at compile time
        if (st == status::unimplemented) return;

        ASSERT_TRUE(st == status::success);

        for (auto isa : cpu_isa_all) {
            bool is_compat = test_mayiuse(isa);
            // ISA specific hint will never lower down the ISA
            ASSERT_TRUE(is_compat == compat_before_hint[isa]);
        }
    }
};

TEST_P(isa_hints_test_t, TestISAHints) {}
INSTANTIATE_TEST_SUITE_P(TestISAHints, isa_hints_test_t,
        ::testing::Values(cpu_isa_hints::no_hints, cpu_isa_hints::prefer_ymm));

} // namespace dnnl

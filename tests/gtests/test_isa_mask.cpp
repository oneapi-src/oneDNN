/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <algorithm>
#include <iterator>
#include <map>
#include <set>

#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#if DNNL_X64
#include "tests/cpu_x64_isa_common.hpp"
#endif

namespace dnnl {

class isa_test_t : public ::testing::TestWithParam<cpu_isa> {
protected:
    void SetUp() override {
        const cpu_isa cur_isa = ::testing::TestWithParam<cpu_isa>::GetParam();
        const auto cur_internal_isa = cvt_to_internal_cpu_isa(cur_isa);

        // Use soft version of mayiuse that allows resetting the max_cpu_isa
        const bool test_flag = true;

        status st = set_max_cpu_isa(cur_isa);
        // status::unimplemented if the feature was disabled at compile time
        if (st == status::unimplemented) return;

        ASSERT_TRUE(st == status::success);

        const std::set<cpu_isa> &compatible_isa = compatible_cpu_isa(cur_isa);
        const std::set<cpu_isa> &all_isa = cpu_isa_all();

        std::set<cpu_isa> incompatible_isa;
        std::set_difference(all_isa.cbegin(), all_isa.cend(),
                compatible_isa.cbegin(), compatible_isa.cend(),
                std::inserter(incompatible_isa, incompatible_isa.begin()));

        for (const cpu_isa cmpt_isa : compatible_isa) {
            const auto &internal_isa_set = masked_internal_cpu_isa(cmpt_isa);
            for (auto internal_isa : internal_isa_set) {
                ASSERT_TRUE(!mayiuse(cur_internal_isa, test_flag)
                        || mayiuse(internal_isa, test_flag));
            }
        }

        for (const cpu_isa incmpt_isa : incompatible_isa) {
            ASSERT_TRUE(!mayiuse(incmpt_isa, test_flag));
        }
    }
};

TEST_P(isa_test_t, TestISA) {}
INSTANTIATE_TEST_SUITE_P(TestISACompatibility, isa_test_t,
        ::testing::ValuesIn(cpu_isa_all().cbegin(), cpu_isa_all().cend()));

} // namespace dnnl

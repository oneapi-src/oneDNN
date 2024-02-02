/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include "tests/test_isa_common.hpp"

namespace dnnl {

TEST(isa_test_t, TestISA) {

    // Use soft version of mayiuse that allows resetting the max_cpu_isa
    const bool test_flag = true;
    const cpu_isa cur_isa = get_max_cpu_isa(test_flag);

    status st = set_max_cpu_isa(cur_isa);
    // status::unimplemented if the feature was disabled at compile time
    if (st == status::unimplemented) return;

    ASSERT_TRUE(st == status::success);

    const auto cur_internal_isa = get_max_cpu_isa_mask(test_flag);

    const std::set<cpu_isa> &compatible_isa = compatible_cpu_isa(cur_isa);
    const std::set<cpu_isa> &isa_list = cpu_isa_list();

    std::set<cpu_isa> incompatible_isa;
    std::set_difference(isa_list.cbegin(), isa_list.cend(),
            compatible_isa.cbegin(), compatible_isa.cend(),
            std::inserter(incompatible_isa, incompatible_isa.begin()));

    for (const cpu_isa cmpt_isa : compatible_isa) {
        const auto &internal_isa_set = masked_internal_cpu_isa(cmpt_isa);
        for (auto internal_isa : internal_isa_set) {
            ASSERT_TRUE((cur_internal_isa & internal_isa) == internal_isa);
        }
    }

    for (const cpu_isa incmpt_isa : incompatible_isa) {
        const auto &internal_isa = cvt_to_internal_cpu_isa(incmpt_isa);
        ASSERT_TRUE((cur_internal_isa & internal_isa) != internal_isa);
    }
}

class isa_enumeration_test_t : public ::testing::TestWithParam<cpu_isa> {
protected:
    virtual void Test() {
        const auto isa = ::testing::TestWithParam<cpu_isa>::GetParam();
        const auto isa1i = cvt_to_internal_cpu_isa(isa);
        const auto isa_all = impl::cpu::x64::cpu_isa_t::isa_all;
        SKIP_IF(isa1i == isa_all, "skip comparison with isa_all");
        const auto &subsets = compatible_cpu_isa(isa);
        for (const auto isa2 : cpu_isa_list()) {
            const auto isa2i = cvt_to_internal_cpu_isa(isa2);
            if (isa2i == isa_all) continue;
            const bool condition = subsets.find(isa2) != subsets.end();
            const bool superset_check
                    = impl::cpu::x64::is_superset(isa1i, isa2i);
            const bool subset_check = impl::cpu::x64::is_subset(isa2i, isa1i);
            ASSERT_TRUE(superset_check == condition);
            ASSERT_TRUE(subset_check == condition);
        }
    }
};

TEST_P(isa_enumeration_test_t, IsaEnumerationTests) {
    Test();
};

INSTANTIATE_TEST_SUITE_P(TestIsaEnums, isa_enumeration_test_t,
        ::testing::ValuesIn(cpu_isa_list()));

} // namespace dnnl

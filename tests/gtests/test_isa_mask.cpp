/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "dnnl.h"
//#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "dnnl.hpp"
#include "src/cpu/cpu_isa_traits.hpp"

namespace dnnl {

using namespace impl::cpu;

const std::set<cpu_isa_t> cpu_isa_set
        = {vanilla, sse41, avx, avx2, avx512_mic, avx512_mic_4ops, avx512_core,
                avx512_core_vnni, avx512_core_bf16, vednn, vejit};

struct isa_compat_info {
    cpu_isa_t this_isa;
    const std::set<cpu_isa_t> cpu_isa_compatible;
};

// This mostly duplicates isa_traits, but the idea is to *not* rely on that
// information...
static std::map<cpu_isa, isa_compat_info> isa_compatibility_table = {
        {cpu_isa::vanilla, {vanilla, {vanilla}}},
        {cpu_isa::sse41, {sse41, {vanilla, sse41}}},
        {cpu_isa::avx, {avx, {vanilla, sse41, avx}}},
        {cpu_isa::avx2, {avx2, {vanilla, sse41, avx, avx2}}},
        {cpu_isa::avx512_mic,
                {avx512_mic, {vanilla, sse41, avx, avx2, avx512_mic}}},
        {cpu_isa::avx512_mic_4ops,
                {avx512_mic_4ops,
                        {vanilla, sse41, avx, avx2, avx512_mic,
                                avx512_mic_4ops}}},
        {cpu_isa::avx512_core,
                {avx512_core, {vanilla, sse41, avx, avx2, avx512_core}}},
        {cpu_isa::avx512_core_vnni,
                {avx512_core_vnni,
                        {vanilla, sse41, avx, avx2, avx512_core,
                                avx512_core_vnni}}},
        {cpu_isa::avx512_core_bf16,
                {avx512_core_bf16,
                        {vanilla, sse41, avx, avx2, avx512_core,
                                avx512_core_vnni, avx512_core_bf16}}},
        {cpu_isa::vednn, {vednn, {vanilla, vednn}}},
        {cpu_isa::vejit, {vejit, {vanilla, vednn, vejit}}}};

class isa_test : public ::testing::TestWithParam<cpu_isa> {
protected:
    virtual void SetUp() {
        auto isa = ::testing::TestWithParam<cpu_isa>::GetParam();

        // soft version of mayiuse that allows resetting the max_cpu_isa
        auto test_mayiuse = [](cpu_isa_t isa) { return mayiuse(isa, true); };

        //status st = set_max_cpu_isa(isa);
        // equiv:
        dnnl_cpu_isa_t dnnlcpuisa = static_cast<dnnl_cpu_isa_t>(isa);
        status st = (status)dnnl_set_max_cpu_isa(dnnlcpuisa);

        // status::unimplemented if the feature was disabled at compile time
        if (st == status::unimplemented) return;

        ASSERT_TRUE(st == status::success);

        auto info = isa_compatibility_table[isa];
        for (auto cur_isa : cpu_isa_set) {
            if (info.cpu_isa_compatible.find(cur_isa)
                    != info.cpu_isa_compatible.end())
                ASSERT_TRUE(
                        !test_mayiuse(info.this_isa) || test_mayiuse(cur_isa))
                        << (test_mayiuse(info.this_isa) ? "can" : "cannot")
                        << " use this_isa=" << (void *)isa << ", and "
                        << (test_mayiuse(cur_isa) ? "can" : "cannot")
                        << " use [compatible] cur_isa=" << (void *)cur_isa;
            else
                ASSERT_TRUE(!test_mayiuse(cur_isa))
                        << " cur_isa=" << (void *)cur_isa
                        << " not in compat table"
                        << ", but "
                        << (test_mayiuse(cur_isa) ? "can" : "cannot")
                        << " use cur_isa=" << (void *)cur_isa << " (fix "
                        << __FILE__ << ")";
        }
    }
};

TEST_P(isa_test, TestISA) {}
#if TARGET_X86
INSTANTIATE_TEST_SUITE_P(TestISACompatibility, isa_test,
        ::testing::Values(cpu_isa::vanilla, cpu_isa::sse41, cpu_isa::avx,
                cpu_isa::avx2, cpu_isa::avx512_mic, cpu_isa::avx512_mic_4ops,
                cpu_isa::avx512_core, cpu_isa::avx512_core_vnni,
                cpu_isa::avx512_core_bf16));
#elif TARGET_VE
INSTANTIATE_TEST_SUITE_P(TestISACompatibility, isa_test,
        ::testing::Values(cpu_isa::vanilla,
                //cpu_isa::any, // probably aliased
                //cpu_isa::all, // probably aliased
                //cpu_isa::ve_common, // not there
                cpu_isa::vednn, cpu_isa::vejit));
#endif

// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s
} // namespace dnnl

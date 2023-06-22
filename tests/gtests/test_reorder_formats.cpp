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

#include <memory>
#include <numeric>
#include <utility>
#include <type_traits>

#include "dnnl_test_common.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "tests/test_isa_common.hpp"
#endif

#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_debug.h"

namespace dnnl {

using dt = memory::data_type;
using tag = memory::format_tag;
using md = memory::desc;

class reorder_formats_test : public ::testing::Test {
public:
    engine e;

protected:
    void SetUp() override {
        e = get_test_engine();
        SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
                "GPU takes a lot of time to complete this test.");

        bool has_bf16 = false;
        bool has_f16 = false;
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
        has_bf16 = dnnl::impl::cpu::platform::has_data_type_support(dnnl_bf16);
        has_f16 = dnnl::impl::cpu::platform::has_data_type_support(dnnl_f16);
#endif

#if DNNL_X64 && DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
        static auto isa = get_effective_cpu_isa();
        // to be removed once {sse41, avx2} are enabled
        bool has_int8_zp_support = is_superset(isa, cpu_isa::avx512_core);
#else
        bool has_int8_zp_support = false;
#endif

        memory::dims SP4D = {2, 2, 2, 2};

        unsigned start_dt = 1;
        unsigned end_dt = 7;

        unsigned start_tag = static_cast<unsigned>(dnnl_format_tag_any) + 1;
        unsigned end_tag = static_cast<unsigned>(dnnl_format_tag_last);

        dt in_dt, out_dt;
        tag in_tag, out_tag;
        md in_md, out_md;

        auto flag_comp = dnnl::impl::memory_extra_flags::compensation_conv_s8s8;
        dnnl::impl::memory_extra_desc_t none {}, conv_s8s8 {}, gconv_s8s8 {};
        gconv_s8s8.flags = conv_s8s8.flags = flag_comp;
        conv_s8s8.compensation_mask = (1 << 0);
        gconv_s8s8.compensation_mask = (1 << 0) + (1 << 1);

        auto flag_zp = dnnl::impl::memory_extra_flags::
                compensation_conv_asymmetric_src;
        dnnl::impl::memory_extra_desc_t conv_zp {}, gconv_zp {},
                conv_s8s8_zp {}, gconv_s8s8_zp {};

        // test zero_point compensation for {s8, u8}
        gconv_zp.flags = conv_zp.flags = conv_s8s8_zp.flags
                = gconv_s8s8_zp.flags = flag_zp;
        conv_s8s8_zp.flags |= flag_comp;
        gconv_s8s8_zp.flags |= flag_comp;
        conv_s8s8_zp.compensation_mask = (1 << 0);
        gconv_s8s8_zp.compensation_mask = (1 << 0) + (1 << 1);
        conv_s8s8_zp.asymm_compensation_mask = conv_zp.asymm_compensation_mask
                = (1 << 0);
        gconv_s8s8_zp.asymm_compensation_mask = gconv_zp.asymm_compensation_mask
                = (1 << 0) + (1 << 1);

        std::vector<dnnl::impl::memory_extra_desc_t> extra {none, conv_s8s8,
                gconv_s8s8, conv_zp, gconv_zp, conv_s8s8_zp, gconv_s8s8_zp};

        for (unsigned i_dt = start_dt; i_dt < end_dt; i_dt++) {
            in_dt = static_cast<dt>(i_dt);
            if (in_dt == dt::bf16 && !has_bf16) continue;
            if (in_dt == dt::f16 && !has_f16) continue;
            if ((in_dt == dt::s8 || in_dt == dt::u8) && !has_int8_zp_support)
                continue;

            for (unsigned i_tag = start_tag; i_tag < end_tag; i_tag++) {
                in_tag = static_cast<tag>(i_tag);
                in_md = md(SP4D, in_dt, in_tag, true);
                if (!in_md.get(true)) continue;
                ASSERT_TRUE(in_md);

                const dnnl::impl::memory_desc_wrapper in_d(in_md.get());
                bool abx2any = in_d.matches_one_of_tag(dnnl_abcd);

                for (unsigned o_dt = start_dt; o_dt < end_dt; o_dt++) {
                    out_dt = static_cast<dt>(o_dt);
                    if (out_dt == dt::bf16 && !has_bf16) continue;
                    if (out_dt == dt::f16 && !has_f16) continue;

                    for (unsigned o_tag = start_tag; o_tag < end_tag; o_tag++) {
                        out_tag = static_cast<tag>(o_tag);
                        out_md = md(SP4D, out_dt, out_tag, true);
                        if (!out_md.get(true)) continue;
                        ASSERT_TRUE(out_md);

                        const dnnl::impl::memory_desc_wrapper out_d(
                                out_md.get());
                        bool any2abx = out_d.matches_one_of_tag(dnnl_abcd);
                        // test only abx->any and any->abx reorders, otherwise
                        // it takes too long. These combinations cover most
                        // popular reorder use cases.
                        if (!abx2any && !any2abx) continue;

                        for (const auto &i_extra : extra) {
                            out_md.get()->extra = i_extra;

                            catch_expected_failures(
                                    [&]() { TestFormat(in_md, out_md); }, false,
                                    dnnl_success);
                        }
                    }
                }
            }
        }
    }

    void TestFormat(const md &in_md, const md &out_md) const {
        reorder::primitive_desc r_pd(
                e, in_md, e, out_md, primitive_attr(), true);
        if (r_pd) {
            auto r = reorder(r_pd);
            auto src = test::make_memory(in_md, e);
            auto dst = test::make_memory(out_md, e);
            auto strm = make_stream(r_pd.get_engine());
            EXPECT_NO_THROW(r.execute(strm, src, dst));
            strm.wait();
        }
    }
};

TEST_F(reorder_formats_test, TestChecksAllFormats) {}

} // namespace dnnl

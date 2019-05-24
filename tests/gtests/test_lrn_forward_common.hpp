/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <cmath>

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

enum {ACROSS=0,WITHIN=1};

struct test_lrn_desc_t {
    memory::dim mb, c;
    memory::dim h, w;
    memory::dim local_size;
    float alpha, beta, k;
    int kind; // 0 ac, 1 wc
};

template <typename data_t>
void check_lrn_fwd(const test_lrn_desc_t &ld,
        const memory::desc &src_d, const memory::desc &dst_d,
        const memory &src, const memory &dst)
{
    auto src_ptr = map_memory<data_t>(src);
    auto dst_ptr = map_memory<data_t>(dst);

    const memory::dim C = ld.c;
    const memory::dim H = ld.h;
    const memory::dim W = ld.w;
    const memory::dim size = ld.local_size;
    const memory::dim CSIZE = ld.kind == ACROSS ? size : 1;
    const memory::dim HWSIZE = size + 1 - CSIZE;
    const memory::dim summands = ld.kind == ACROSS ? size : size*size;
    const auto padded_c = src.get_desc().data.padded_dims[1];

    const mkldnn::impl::memory_desc_wrapper src_mdw(src_d.data);
    const mkldnn::impl::memory_desc_wrapper dst_mdw(dst_d.data);

    auto off = [=](memory::dim n, memory::dim c, memory::dim h, memory::dim w)
    { return ((n * padded_c + c) * ld.h + h) * ld.w + w; };

    auto ker = [&](data_t *d, memory::dim n, memory::dim oc, memory::dim oh,
            memory::dim ow)
    {
        data_t sum = 0.0;
        for (memory::dim c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2) continue;
            if (c >= C + (CSIZE - 1) / 2) continue;
            for (memory::dim h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2) continue;
                if (h >= H + (HWSIZE - 1) / 2) continue;
                for (memory::dim w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2) continue;
                    if (w >= W + (HWSIZE - 1) / 2) continue;
                    data_t s = src_ptr[src_mdw.off_l(off(n, c - (CSIZE - 1) / 2, h - (HWSIZE - 1) / 2, w - (HWSIZE - 1) / 2), true)];
                    sum += s * s;
                }
            }
        }
        data_t norm_coef = powf(static_cast<float>(ld.k + ld.alpha * sum / summands),
                                static_cast<float>(ld.beta));
        data_t ref_out = src_ptr[src_mdw.off_l(off(n, oc, oh, ow), true)] / norm_coef;
        data_t eps = static_cast<data_t>(1.e-7f * ( 2 * summands + 5));

        memory::data_type data_type = data_traits<data_t>::data_type;
        if (data_type == mkldnn::memory::data_type::f16)
            eps = static_cast<data_t>(1.e-4f * 2 * summands);

        data_t out = d[0];
        data_t norm_max = std::max(fabs(out), fabs(ref_out));
        if (norm_max < eps) norm_max = 1.;
        ASSERT_NEAR(out, ref_out, eps * norm_max);
    };

    const memory::dim N = ld.mb;
    mkldnn::impl::parallel_nd(N, padded_c, H, W,
        [&](memory::dim n, memory::dim c, memory::dim h, memory::dim w)
        { ker(&dst_ptr[dst_mdw.off_l(off(n, c, h, w), true)], n, c, h, w); }
    );
}

struct lrn_fwd_test_params {
    prop_kind aprop_kind;
    algorithm aalgorithm;
    memory::format_tag src_format;
    memory::format_tag dst_format;
    test_lrn_desc_t test_ld;
    bool expect_to_fail;
    mkldnn_status_t expected_status;
};

template <typename data_t>
class lrn_forward_test : public ::testing::TestWithParam<lrn_fwd_test_params> {
    lrn_fwd_test_params p;

protected:
    virtual void SetUp() {
        p = ::testing::TestWithParam<decltype(p)>::GetParam();
        catch_expected_failures([=](){Test();}, p.expect_to_fail,
                    p.expected_status);
    }

    void Test() {
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        auto eng = engine(get_test_engine_kind(), 0);
        auto strm = stream(eng);
        memory::data_type data_type = data_traits<data_t>::data_type;

        test_lrn_desc_t ld = p.test_ld;
        bool with_workspace = p.aprop_kind == prop_kind::forward_training;

        auto l_src_desc = create_md({ ld.mb, ld.c, ld.h, ld.w },
                data_type, p.src_format);
        auto l_dst_desc = create_md({ ld.mb, ld.c, ld.h, ld.w },
                data_type, p.dst_format);

        auto l_src = test_memory(l_src_desc, eng);
        auto l_dst = test_memory(l_dst_desc, eng);

        // Only true for dense format
        fill_data<data_t>(l_src.get_size() / sizeof(data_t),
                l_src.get());
        fill_data<data_t>(l_dst.get_size() / sizeof(data_t),
                l_dst.get());
        check_zero_tail<data_t>(1, l_src.get());
        check_zero_tail<data_t>(1, l_dst.get());

        auto lrn_desc = lrn_forward::desc(p.aprop_kind, p.aalgorithm,
                l_src_desc, ld.local_size, ld.alpha, ld.beta, ld.k);
        auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, eng);

        memory workspace;

        // Execute
        auto l = lrn_forward(lrn_prim_desc);
        std::unordered_map<int, memory> args = {
            {MKLDNN_ARG_SRC, l_src.get()},
            {MKLDNN_ARG_DST, l_dst.get()}
        };
        if (with_workspace) {
            auto workspace_md = lrn_prim_desc.workspace_desc();
            workspace = memory(workspace_md, eng);
            args.insert({MKLDNN_ARG_WORKSPACE, workspace});
        }
        l.execute(strm, args);
        strm.wait();

        check_zero_tail<data_t>(0, l_dst.get());

        check_lrn_fwd<data_t>(ld, l_src_desc, l_dst_desc, l_src.get(),
                l_dst.get());
    }
};

}

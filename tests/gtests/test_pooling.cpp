/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.hpp"

namespace mkldnn {

struct test_pool_desc_t {
    uint32_t mb, c;
    uint32_t ih, iw;
    uint32_t oh, ow;
    uint32_t kh, kw;
    int32_t padh, padw;
    uint32_t strh, strw;
};

template <typename data_t>
void check_pool_max_fwd(test_pool_desc_t pd, memory &src, memory &dst)
{
    data_t *src_data = (data_t *)src.get_data_handle();
    data_t *dst_data = (data_t *)dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc dst_d = dst.get_primitive_desc().desc();

#pragma omp parallel for collapse(4)
    for (uint32_t n = 0; n < pd.mb; n++) {
        for (uint32_t c = 0; c < pd.c; c++) {
            for (uint32_t oh = 0; oh < pd.oh; oh++) {
                for (uint32_t ow = 0; ow < pd.ow; ow++) {
                    uint32_t oidx = n * pd.c * pd.oh * pd.ow + c * pd.oh * pd.ow
                            + oh * pd.ow + ow;
                    data_t out = dst_data[map_index(dst_d, oidx)];
                    data_t out_ref = data_t(0);
                    bool is_initialized = false;
                    for (uint32_t kh = 0; kh < pd.kh; kh++) {
                        for (uint32_t kw = 0; kw < pd.kw; kw++) {
                            int32_t iw = ow * pd.strw - pd.padw + kw;
                            int32_t ih = oh * pd.strh - pd.padh + kh;
                            if (iw < 0 || iw >= (int32_t)pd.iw || ih < 0
                                    || ih >= (int32_t)pd.ih)
                                continue;
                            uint32_t iidx = n * pd.c * pd.ih * pd.iw
                                    + c * pd.ih * pd.iw + ih * pd.iw + iw;

                            data_t d = src_data[map_index(src_d, iidx)];
                            if (!is_initialized) {
                                out_ref = d;
                                is_initialized = true;
                            } else {
                                if (out_ref < d)
                                    out_ref = d;
                            }
                        }
                    }
                    EXPECT_NEAR(out, out_ref, 1e-6);
                }
            }
        }
    }
}

struct pool_test_params {
    prop_kind aprop_kind;
    const engine::kind engine_kind;
    pooling::algorithm aalgorithm;
    memory::format src_format;
    memory::format dst_format;
    test_pool_desc_t test_pd;
};

template <typename data_t>
class pooling_test : public ::testing::TestWithParam<pool_test_params> {
protected:
    virtual void SetUp()
    {
        pool_test_params p
                = ::testing::TestWithParam<pool_test_params>::GetParam();

        ASSERT_TRUE(p.engine_kind == engine::kind::cpu
                || p.engine_kind == engine::kind::cpu_lazy);
        ASSERT_TRUE(p.aprop_kind == prop_kind::forward_training
                || p.aprop_kind == prop_kind::forward_scoring);
        ASSERT_EQ(p.aalgorithm, pooling::max);
        auto eng = engine(p.engine_kind, 0);
        memory::precision prec = data_traits<data_t>::prec;
        ASSERT_EQ(prec, mkldnn::memory::precision::f32);

        test_pool_desc_t pd = p.test_pd;

        auto p_src_desc
                = create_md({ pd.mb, pd.c, pd.ih, pd.iw }, prec, p.src_format);
        auto p_dst_desc
                = create_md({ pd.mb, pd.c, pd.oh, pd.ow }, prec, p.dst_format);

        auto pool_desc = pooling::desc(p.aprop_kind, p.aalgorithm, p_src_desc,
                p_dst_desc, { pd.strh, pd.strw }, { pd.kh, pd.kw },
                { pd.padh, pd.padw }, padding_kind::zero);

        auto pool_prim_desc = pooling::primitive_desc(pool_desc, eng);

        auto p_idx_desc = pool_prim_desc.data.indices_primitive_desc;
        auto p_src = memory(memory::primitive_desc(p_src_desc, eng));
        auto p_idx = memory(memory::primitive_desc(p_idx_desc));
        auto p_dst = memory(memory::primitive_desc(p_dst_desc, eng));

        fill_data<data_t>(p_src.get_primitive_desc().get_number_of_elements(),
                (data_t *)p_src.get_data_handle());

        auto pool = pooling(pool_prim_desc, p_src, p_idx, p_dst);

        std::vector<primitive> pipeline;
        pipeline.push_back(pool);

        stream().submit(pipeline).wait();

        check_pool_max_fwd<data_t>(pd, p_src, p_dst);
    }
};

using pooling_test_float = pooling_test<float>;
using pool_test_params_float = pool_test_params;

TEST_P(pooling_test_float, TestsPooling)
{
}
INSTANTIATE_TEST_CASE_P(
        TestPoolingForward, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardNHWC, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nhwc,
            memory::format::nhwc, { 2, 4, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } }));

INSTANTIATE_TEST_CASE_P(
        TestPoolingForwardBlocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 4, 4, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 13, 13, 12, 12, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 4, 4, 4, 4, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 3, 3, 4, 4, 3, 3, 1, 1, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 32, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 122, 32, 32, 2, 32, 2, 3, 3, 1, 1, 1, 1 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardNCHW, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nchw,
            memory::format::nchw, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingAlexnetForwardBlocked, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 27, 27, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 13, 13, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 6, 6, 3, 3, 0, 0, 2, 2 } }
            ));

INSTANTIATE_TEST_CASE_P(
        TestPoolingBlockedStride1, pooling_test_float, ::testing::Values(
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 55, 55, 53, 53, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 27, 27, 25, 25, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_training,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } },
            pool_test_params_float{ prop_kind::forward_scoring,
            engine::kind::cpu, pooling::max, memory::format::nChw8c,
            memory::format::nChw8c, { 2, 16, 13, 13, 11, 11, 3, 3, 0, 0, 1, 1 } }
            ));
}

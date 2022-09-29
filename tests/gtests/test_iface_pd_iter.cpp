/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_types.h"

namespace dnnl {

const dnnl_status_t ok = dnnl_success;

class pd_iter_test_t : public ::testing::Test {
protected:
    dnnl_engine_t engine;
    void SetUp() override {
        auto engine_kind
                = static_cast<dnnl_engine_kind_t>(get_test_engine_kind());
        ASSERT_EQ(dnnl_engine_create(&engine, engine_kind, 0), ok);
    }
    void TearDown() override { dnnl_engine_destroy(engine); }
};

TEST_F(pd_iter_test_t, TestReLUImpls) {
    dnnl_memory_desc_t dense_md;
    dnnl_dims_t dims = {4, 16, 16, 16};
    ASSERT_EQ(dnnl_memory_desc_create_with_tag(
                      &dense_md, 4, dims, dnnl_f32, dnnl_nchw),
            ok);

    dnnl_primitive_desc_t pd;
    dnnl_status_t rc = dnnl_eltwise_forward_primitive_desc_create(&pd, engine,
            dnnl_forward_inference, dnnl_eltwise_relu, dense_md, dense_md, 0.f,
            0.f, nullptr);
    ASSERT_EQ(rc, ok); /* there should be at least one impl */

    while ((rc = dnnl_primitive_desc_next_impl(pd)) == ok)
        ;
    ASSERT_EQ(rc, dnnl_last_impl_reached);

    // Primitive descriptor has to be valid when the iterator
    // reaches the end.
    dnnl_primitive_t p;
    rc = dnnl_primitive_create(&p, pd);
    ASSERT_EQ(rc, ok);

    rc = dnnl_primitive_desc_destroy(pd);
    ASSERT_EQ(rc, ok);
    rc = dnnl_primitive_destroy(p);
    ASSERT_EQ(rc, ok);
    rc = dnnl_memory_desc_destroy(dense_md);
    ASSERT_EQ(rc, ok);
}

TEST_F(pd_iter_test_t, UnsupportedPrimitives) {
    const float scales[2] = {1.0f, 1.0f};
    dnnl_memory_desc_t mds[2];

    dnnl_dims_t dims = {1, 16, 16, 16};
    ASSERT_EQ(dnnl_memory_desc_create_with_tag(
                      &mds[0], 4, dims, dnnl_f32, dnnl_nchw),
            ok);
    ASSERT_EQ(dnnl_memory_desc_create_with_tag(
                      &mds[1], 4, dims, dnnl_f32, dnnl_nchw),
            ok);

    dnnl_primitive_desc_t reorder_pd;
    dnnl_primitive_desc_t concat_pd;
    dnnl_primitive_desc_t sum_pd;

    ASSERT_EQ(dnnl_reorder_primitive_desc_create(
                      &reorder_pd, mds[0], engine, mds[1], engine, nullptr),
            ok);
    ASSERT_EQ(dnnl_concat_primitive_desc_create(
                      &concat_pd, engine, nullptr, 2, 0, mds, nullptr),
            ok);
    ASSERT_EQ(dnnl_sum_primitive_desc_create(
                      &sum_pd, engine, mds[0], 2, scales, mds, nullptr),
            ok);

    ASSERT_EQ(
            dnnl_primitive_desc_next_impl(reorder_pd), dnnl_last_impl_reached);
    ASSERT_EQ(dnnl_primitive_desc_next_impl(concat_pd), dnnl_last_impl_reached);
    ASSERT_EQ(dnnl_primitive_desc_next_impl(sum_pd), dnnl_last_impl_reached);

    ASSERT_EQ(dnnl_primitive_desc_destroy(reorder_pd), ok);
    ASSERT_EQ(dnnl_primitive_desc_destroy(concat_pd), ok);
    ASSERT_EQ(dnnl_primitive_desc_destroy(sum_pd), ok);

    ASSERT_EQ(dnnl_memory_desc_destroy(mds[0]), ok);
    ASSERT_EQ(dnnl_memory_desc_destroy(mds[1]), ok);
}

TEST(pd_next_impl, TestEltwiseImpl) {
    SKIP_IF_CUDA(true, "Unsupported memory format for CUDA");
    SKIP_IF_HIP(true, "Unsupported memory format for HIP");
    auto eng = get_test_engine();
    memory::desc md(
            {8, 32, 4, 4}, memory::data_type::f32, memory::format_tag::nChw8c);

    eltwise_forward::primitive_desc epd(eng, prop_kind::forward_training,
            algorithm::eltwise_relu, md, md, 0.f);

    std::string impl0(epd.impl_info_str());
    eltwise_forward e0(epd);

    while (epd.next_impl()) {
        std::string impl1(epd.impl_info_str());
        eltwise_forward e1(epd);
        ASSERT_NE(impl0, impl1);
        impl0 = impl1;
    }

    // When the last implementation is reached all subsequent `next_impl()`
    // calls should return `false`.
    ASSERT_EQ(epd.next_impl(), false);
    ASSERT_EQ(epd.next_impl(), false);
}

} // namespace dnnl

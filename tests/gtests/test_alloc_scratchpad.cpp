/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <unordered_map>

#include "dnnl.hpp"

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

namespace dnnl {

using dt = memory::data_type;
using tag = memory::format_tag;

class scratchpad_alloc_test : public ::testing::Test {
protected:
    engine eng = get_test_engine();
    virtual void SetUp() {}
};

TEST_F(scratchpad_alloc_test, ScratchpadAllocTest) {

    auto dtype = dt::s8;

    auto strm = stream(eng);

    // For this test, inner product primitive is used to test scratchpad
    // allocation by the library since its size can be commensurable with
    // the dst tensor. Therefore we set OC to 2^60 (or ~10^18) such that the
    // dst tensor has size larger than available RAM. NOTE: it is important
    // for this test to use a primitive that requires a global scratchpad
    // memory comparable to the src/dst.
    const memory::dim N = 3, IC = 3, IH = 73, IW = 73,
                      OC1 = (memory::dim)1 << 2, OC2 = (memory::dim)1 << 60;

    auto src_md = memory::desc({N, IC, IH, IW}, dtype, tag::nchw);
    auto wei_md = memory::desc({OC1, IC, IH, IW}, dtype, tag::oihw);
    auto dst_md = memory::desc({N, OC1}, dtype, tag::nc);

    auto src_mem = memory(src_md, eng);
    auto wei_mem = memory(wei_md, eng);
    auto dst_mem = memory(dst_md, eng);

    auto op_desc = inner_product_forward::desc(
            prop_kind::forward_inference, src_md, wei_md, dst_md);
    auto pd = inner_product_forward::primitive_desc(op_desc, eng);
    auto p1 = inner_product_forward(pd);

    wei_md = memory::desc({OC2, IC, IH, IW}, dtype, tag::oihw);
    dst_md = memory::desc({N, OC2}, dtype, tag::nc);

    op_desc = inner_product_forward::desc(
            prop_kind::forward_inference, src_md, wei_md, dst_md);
    pd = inner_product_forward::primitive_desc(op_desc, eng);

    // Scratchpad allocation should fail here, thereby throwing an
    // out_of_memory error
    EXPECT_ANY_THROW(auto p2 = inner_product_forward(pd));

    std::unordered_map<int, memory> args;
    args.insert({DNNL_ARG_SRC, src_mem});
    args.insert({DNNL_ARG_WEIGHTS, wei_mem});
    args.insert({DNNL_ARG_DST, dst_mem});

    // Here, a successful execution p1 is expected, despite that the
    // scratchpad allocation failed for p2
    EXPECT_NO_THROW(p1.execute(strm, args));

    strm.wait();
}

} // namespace dnnl

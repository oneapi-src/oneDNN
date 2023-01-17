/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "backend/dnnl/internal_attrs.hpp"

namespace graph = dnnl::impl::graph;

TEST(InternalAttrs, InternalAttr2str) {
    using namespace graph::dnnl_impl::op_attr;
#define CASE(a) ASSERT_EQ(internal_attr2str(a), #a)
    CASE(canonicalized);
    CASE(change_layout);
    CASE(is_constant);
    CASE(is_convtranspose);
    CASE(is_training);
    CASE(fwd_alg_kind);
    CASE(fuse_relu);
    CASE(with_bias);
    CASE(with_runtime_scales);
    CASE(with_runtime_zps);
    CASE(with_runtime_src_zps);
    CASE(with_runtime_dst_zps);
    CASE(is_bias_add);
    CASE(with_sum);
    CASE(alg_kind);
    CASE(fusion_info_key);
    CASE(dw_type);
    CASE(kind);
    CASE(p);
    CASE(dst_zps);
    CASE(src_zps);
    CASE(permutation);
#undef CASE
    ASSERT_EQ(internal_attr2str(graph::op_attr::alpha), "undefined_attr");
}

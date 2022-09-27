/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "interface/c_types_map.hpp"
#include "interface/shape_infer.hpp"

namespace impl = dnnl::graph::impl;

TEST(ShapeInfer, OneWayBroadcast) {
    using dims = impl::dims;
    dims src_shape {2, 3};
    dims dst1_shape {2};
    dims dst2_shape {2, 3};
    dims dst3_shape {4, 3};
    dims dst4_shape {1, 2, 3};

    ASSERT_EQ(impl::one_way_broadcast(dst1_shape, src_shape),
            impl::status::invalid_shape);

    ASSERT_EQ(impl::one_way_broadcast(dst2_shape, src_shape),
            impl::status::success);

    ASSERT_EQ(impl::one_way_broadcast(dst3_shape, src_shape),
            impl::status::invalid_shape);

    ASSERT_EQ(impl::one_way_broadcast(dst4_shape, src_shape),
            impl::status::success);
}

/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <gtest/gtest.h>

#include <vector>

#include "api/test_api_common.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

using namespace dnnl::graph;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
TEST(OCLApi, Tensor) {
    dnnl::engine::kind ekind
            = static_cast<dnnl::engine::kind>(api_test_engine_kind);
    SKIP_IF(ekind != dnnl::engine::kind::gpu,
            "skip ocl api test for non-gpu engine");
    dnnl::engine eng(ekind, 0);

    logical_tensor lt0 {0, logical_tensor::data_type::f32,
            logical_tensor::layout_type::any};
    logical_tensor lt1 {1, logical_tensor::data_type::f32, {2, 2},
            logical_tensor::layout_type::strided};

    EXPECT_NO_THROW({ tensor(lt0, eng, nullptr); });
    EXPECT_NO_THROW({ tensor(lt1, eng); });
    tensor t(lt1, eng);
    ASSERT_EQ(t.get_engine().get_kind(), dnnl::engine::kind::gpu);
}
#endif

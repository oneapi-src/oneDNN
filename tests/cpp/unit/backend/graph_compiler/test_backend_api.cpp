/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include <gtest/gtest.h>

#include "backend/graph_compiler/compiler_backend.hpp"
#include "cpp/unit/utils.hpp"
#include "interface/partition.hpp"

namespace impl = dnnl::graph::impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(GCBackendApi, GetMemSize) {
    impl::logical_tensor_t a, b, c, d;
    const std::vector<impl::dim_t> a_dim {1, 4, 3};
    const std::vector<impl::dim_t> b_dim {32, 16, 64, 64};
    const std::vector<impl::dim_t> c_dim {32};
    const std::vector<impl::dim_t> d_dim {1, 1};

    a = utils::logical_tensor_init(
            0, a_dim, impl::data_type::u8, impl::layout_type::strided);
    b = utils::logical_tensor_init(
            1, b_dim, impl::data_type::s8, impl::layout_type::strided);
    c = utils::logical_tensor_init(
            2, c_dim, impl::data_type::f32, impl::layout_type::strided);
    d = utils::logical_tensor_init(
            3, d_dim, impl::data_type::s32, impl::layout_type::strided);

    auto &compiler_backend_ptr
            = impl::compiler_impl::compiler_backend_t::get_singleton();

    size_t a_mem_res = utils::product(a_dim) * sizeof(signed char);
    size_t b_mem_res = utils::product(b_dim) * sizeof(signed char);
    size_t c_mem_res = utils::product(c_dim) * sizeof(float);
    size_t d_mem_res = utils::product(d_dim) * sizeof(int32_t);
    ASSERT_EQ(compiler_backend_ptr.get_mem_size(a), a_mem_res);
    ASSERT_EQ(compiler_backend_ptr.get_mem_size(b), b_mem_res);
    ASSERT_EQ(compiler_backend_ptr.get_mem_size(c), c_mem_res);
    ASSERT_EQ(compiler_backend_ptr.get_mem_size(d), d_mem_res);
}

TEST(GCBackendApi, CompilerBackendRegistration) {
    std::vector<const impl::backend *> &backends
            = impl::backend_registry_t::get_singleton()
                      .get_registered_backends();
    auto compiler_backend = std::find_if(
            backends.begin(), backends.end(), [](const impl::backend *bkd) {
                return bkd->get_name() == "compiler_backend";
            });
    ASSERT_NE(compiler_backend, backends.end());
    EXPECT_FLOAT_EQ((*compiler_backend)->get_priority(), 2.0);
}

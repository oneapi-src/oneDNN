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

#include <iostream>
#include <memory>

#include "backend/dnnl/backend.hpp"
#include "backend/dnnl/tensor.hpp"
#include "interface/backend.hpp"
#include "gtest/gtest.h"

namespace impl = llga::impl;
namespace dnnl_impl = llga::impl::dnnl_impl;

TEST(layout_id_test, opaque_md_layout_id_mapping) {
    using tensor = dnnl_impl::tensor;
    using data_type = dnnl_impl::tensor::desc::data_type;
    using format_tag = dnnl_impl::tensor::desc::format_tag;

    dnnl_impl::dnnl_layout_id_manager &mgr
            = std::dynamic_pointer_cast<dnnl_impl::dnnl_backend>(
                    impl::backend_manager::get_backend("dnnl"))
                      ->get_layout_id_manager();

    // opaque md should be cached and generate a layout id, and the later
    // layout id should be greater than the former one
    tensor::desc md1({8, 3, 224, 224}, data_type::f32, format_tag::nChw16c);
    auto id1 = mgr.set_mem_desc(md1);
    ASSERT_TRUE(id1.has_value());

    tensor::desc md2({8, 16, 96, 96}, data_type::f32, format_tag::nChw8c);
    auto id2 = mgr.set_mem_desc(md2);
    ASSERT_TRUE(id2.has_value());

    ASSERT_GT(id2.value(), id1.value());

    // we should be able to get cached opaque md out according to the
    // layout id
    auto recovered_md1 = mgr.get_mem_desc(id1.value());
    ASSERT_TRUE(recovered_md1.has_value());
    ASSERT_EQ(llga::impl::utils::any_cast<tensor::desc>(recovered_md1.value()),
            md1);

    auto recovered_md2 = mgr.get_mem_desc(id2.value());
    ASSERT_TRUE(recovered_md2.has_value());
    ASSERT_EQ(llga::impl::utils::any_cast<tensor::desc>(recovered_md2.value()),
            md2);
}

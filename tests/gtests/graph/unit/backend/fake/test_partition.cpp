/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include "interface/op.hpp"
#include "interface/partition.hpp"

#include "backend/fake/fake_partition_impl.hpp"

using namespace dnnl::impl::graph;

TEST(Partition, Unsupported) {
    fake_impl::fake_partition_impl_t p(engine_kind::cpu);
    size_t id = 100;
    std::shared_ptr<op_t> n(new op_t(id, op_kind::Wildcard, "Wildcard"));
    p.init(n.get());
    ASSERT_TRUE(p.is_initialized());
    ASSERT_TRUE(p.get_assigned_backend()->get_name() == "fake_backend");
    ASSERT_EQ(p.get_fused_op()->get_kind(), op_kind::Wildcard);

    // clone
    std::shared_ptr<partition_impl_t> p_share = p.clone();
    fake_impl::fake_partition_impl_t *p_share_raw
            = dynamic_cast<fake_impl::fake_partition_impl_t *>(p_share.get());
    ASSERT_TRUE(p_share_raw->is_initialized());
    ASSERT_TRUE(
            p_share_raw->get_assigned_backend()->get_name() == "fake_backend");
    ASSERT_EQ(p_share_raw->get_fused_op()->get_kind(), op_kind::Wildcard);
}

/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "interface/graph.hpp"
#include "interface/partition.hpp"

#include "utils/pm/pass_base.hpp"
#include "utils/pm/pass_manager.hpp"

#include "backend/fake/fake_backend.hpp"
#include "backend/fake/fake_partition_impl.hpp"

#include "graph/unit/utils.hpp"

TEST(Pass, FakeSingleOpReplacement) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::impl::graph::op_kind;

    auto &fake_backend_ptr = fake_impl::fake_backend_t::get_singleton();
    auto fake_pm = pass::pass_manager_t(fake_backend_ptr.get_pass_registry());
    std::vector<op_kind_t> single_op_set_unsupported = {
            /* not enabling ops = */ Concat,
            Divide,
            EluBackward,
            LayerNormBackward,
            Round,
            Sigmoid,
            SigmoidBackward,
            SqrtBackward,
            TanhBackward,
            StaticReshape,
            StaticTranspose,
            /* no dnnl primitive support = */ BiasAdd,
            BiasAddBackward,
            Clamp,
            ClampBackward,
            ReduceSum,
            Select,
            SoftPlus,
            SoftPlusBackward,
            Wildcard,
            End,
            Reciprocal,
    };
    for (auto akind : single_op_set_unsupported) {
        graph_t agraph;
        op_t *op = agraph.create_op(akind);
        ASSERT_EQ(op->get_kind(), akind);
        fake_pm.run_passes(agraph, "no_config");

        auto orig_op = agraph.get_ops()[0];
        ASSERT_NE(orig_op->get_partition(), nullptr);
        ASSERT_EQ(orig_op->get_partition()->get_assigned_backend()->get_name(),
                std::string("fake_backend"));

        ASSERT_EQ(agraph.get_partitions().size(), 1U);
        auto replaced_op = static_cast<fake_impl::fake_partition_impl_t *>(
                agraph.get_partitions()[0].get())
                                   ->get_fused_op();
        ASSERT_EQ(replaced_op->get_kind(), akind);
    }
}

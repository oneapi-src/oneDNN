/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#include "context.hpp"
#include "reference/gemm_ref.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"

#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/managed_matmul_core.hpp>
#include <ops/matmul_core.hpp>
#include <ops/templates/managed_matmul_core.hpp>
#include <ops/templates/matmul_core.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
using namespace dnnl::impl::graph::gc;
class customized_managed_matmul_core_op_t
    : public ops::managed_matmul_core_op_t {
public:
    using inner_config = ops::managed_matmul_core_config_t;
    customized_managed_matmul_core_op_t(
            const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs)
        : ops::managed_matmul_core_op_t(producer_lt, consumer_lt, attrs) {
        op_name_ = "customized_managed_matmul_core";
    }
    std::vector<config_ptr> get_dynamic_config_candidates(
            const context_ptr &ctx) override {
        config_ptr_vec ret;
        int num_threads = runtime_config_t::get().get_num_threads();
        auto M_split_candidates = utils::get_factors(num_threads);
        auto N_split_candidates = utils::get_factors(num_threads);
        std::vector<int> MNK_sub_candidates = {1, 2, 8};
        for (auto &M_split_num : M_split_candidates) {
            for (auto &N_split_num : N_split_candidates) {
                if (num_threads % (M_split_num * N_split_num)) { continue; }
                for (auto &M_sub_block : MNK_sub_candidates) {
                    for (auto &N_sub_block : MNK_sub_candidates) {
                        for (auto &K_sub_block : MNK_sub_candidates) {
                            auto gcfg = reflection::general_object_t::make<
                                    inner_config>();
                            inner_config &cfg
                                    = *gcfg.unchecked_get_as<inner_config>();
                            cfg.M_split_num = M_split_num;
                            cfg.N_split_num = N_split_num;
                            cfg.M_sub_block = M_sub_block;
                            cfg.N_sub_block = N_sub_block;
                            cfg.K_sub_block = K_sub_block;
                            cfg.im_loop_order = 0;
                            ret.emplace_back(std::move(gcfg));
                        }
                    }
                }
            }
        }
        return ret;
    }
};
TEST(GCCore_CPU_dynamic_impl_kind_cpp, TestImplKindManagedMatmulCore) {
    sc_graph_t g;
    auto in_a = g.make_input(
            {graph_tensor::make({28, 64}, sc_data_format_t::MK())});
    auto in_b = g.make_input(
            {graph_tensor::make({64, 32}, sc_data_format_t::KN())});
    auto ctx = get_test_ctx();
    auto cfg_sz = sizeof(ops::managed_matmul_core_config_t);
    {
        // threads == 1
        SET_THREADS_OR_SKIP(1);
        auto mmm = g.make<customized_managed_matmul_core_op_t>(
                std::vector<graph_tensor_ptr> {
                        in_a->get_outputs()[0], in_b->get_outputs()[0]},
                std::vector<graph_tensor_ptr> {graph_tensor::make({28, 32})},
                any_map_t());
        auto tun_mmm = mmm->dyn_cast<tunable_op_t>();
        auto configs = tun_mmm->get_dynamic_config_candidates(ctx);
        EXPECT_EQ(configs.size(), UINT64_C(1) * 1 * 3 * 3 * 3);
        auto cfg0 = configs[0]
                            .unchecked_get_as<
                                    ops::managed_matmul_core_config_t>();
        auto cfg1 = configs[3]
                            .unchecked_get_as<
                                    ops::managed_matmul_core_config_t>();
        auto cfg2 = configs[5]
                            .unchecked_get_as<
                                    ops::managed_matmul_core_config_t>();
        ops::managed_matmul_core_config_t expect0 {1, 1, 1, 1, 1, 0};
        ops::managed_matmul_core_config_t expect1 {1, 1, 1, 2, 1, 0};
        ops::managed_matmul_core_config_t expect2 {1, 1, 1, 2, 8, 0};
        EXPECT_TRUE(!memcmp(cfg0, &expect0, cfg_sz));
        EXPECT_TRUE(!memcmp(cfg1, &expect1, cfg_sz));
        EXPECT_TRUE(!memcmp(cfg2, &expect2, cfg_sz));
    }
    {
        // threads == 4
        SET_THREADS_OR_SKIP(4);
        auto mmm = g.make<customized_managed_matmul_core_op_t>(
                std::vector<graph_tensor_ptr> {
                        in_a->get_outputs()[0], in_b->get_outputs()[0]},
                std::vector<graph_tensor_ptr> {graph_tensor::make({28, 32})},
                any_map_t());
        auto tun_mmm = mmm->dyn_cast<tunable_op_t>();
        auto configs = tun_mmm->get_dynamic_config_candidates(ctx);
        EXPECT_EQ(configs.size(), UINT64_C(6) * 3 * 3 * 3);
        auto cfg0 = configs[27]
                            .unchecked_get_as<
                                    ops::managed_matmul_core_config_t>();
        auto cfg1 = configs[44]
                            .unchecked_get_as<
                                    ops::managed_matmul_core_config_t>();
        auto cfg2 = configs[50]
                            .unchecked_get_as<
                                    ops::managed_matmul_core_config_t>();
        ops::managed_matmul_core_config_t expect0 {1, 2, 1, 1, 1, 0};
        ops::managed_matmul_core_config_t expect1 {1, 2, 2, 8, 8, 0};
        ops::managed_matmul_core_config_t expect2 {1, 2, 8, 2, 8, 0};
        EXPECT_TRUE(!memcmp(cfg0, &expect0, cfg_sz));
        EXPECT_TRUE(!memcmp(cfg1, &expect1, cfg_sz));
        EXPECT_TRUE(!memcmp(cfg2, &expect2, cfg_sz));
    }

    {
        // threads == 56
        SET_THREADS_OR_SKIP(56);
        auto mmm = g.make<customized_managed_matmul_core_op_t>(
                std::vector<graph_tensor_ptr> {
                        in_a->get_outputs()[0], in_b->get_outputs()[0]},
                std::vector<graph_tensor_ptr> {graph_tensor::make({28, 32})},
                any_map_t());
        auto tun_mmm = mmm->dyn_cast<tunable_op_t>();
        auto configs = tun_mmm->get_dynamic_config_candidates(ctx);
        EXPECT_EQ(configs.size(), UINT64_C(30) * 3 * 3 * 3);
        auto cfg0 = configs[808]
                            .unchecked_get_as<
                                    ops::managed_matmul_core_config_t>();
        ops::managed_matmul_core_config_t expect0 {56, 1, 8, 8, 2, 0};
        EXPECT_TRUE(!memcmp(cfg0, &expect0, cfg_sz));
    }
}

// use customized matmul core here as currently we only support matmul but
// matmul does not need dynamic impl kinds.
class customized_matmul_core_op_t : public ops::matmul_core_op_t {
public:
    customized_matmul_core_op_t(
            const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs)
        : matmul_core_op_t(producer_lt, consumer_lt, attrs) {
        op_name_ = "customized_matmul_core";
    }
    std::vector<int> get_impl_dispatch_candidates(
            const context_ptr &ctx) override {
        return get_dynamic_impl_dispatch_candidates(this, ctx);
    }
    std::vector<config_ptr> get_dynamic_config_candidates(
            const context_ptr &ctx) override {
        if (dyn_config_candidates_.empty()) {
            auto set = get_dispatch_key_set()->get_inner_set();
            auto &ret = dyn_config_candidates_;
            ret.reserve(set.size());
            for (auto &it : set) {
                auto gcfg = reflection::general_object_t::make<
                        ops::matmul_core_config_t>();
                auto cfg = gcfg.unchecked_get_as<ops::matmul_core_config_t>();
                if (it.var_block_[0][0] < 16) { continue; }
                cfg->M_block = it.var_block_[0][0];
                cfg->N_block = it.var_block_[1][1];
                cfg->K_block = it.var_block_[0][1];
                ret.emplace_back(std::move(gcfg));
            }
        }
        return dyn_config_candidates_;
    }
    impl_kind_map convert_config_candidates_to_impl_map(
            const std::vector<config_ptr> &configs) override {
        impl_kind_map ret;
        ret.reserve(configs.size());
        for (int i = 0; i < static_cast<int>(configs.size()); i++) {
            auto cfg = configs[i].unchecked_get_as<ops::matmul_core_config_t>();
            std::vector<uint64_t> keys = {static_cast<uint64_t>(cfg->M_block),
                    static_cast<uint64_t>(cfg->N_block),
                    static_cast<uint64_t>(cfg->K_block)};
            ret[keys] = i;
        }
        return ret;
    }
};

TEST(GCCore_CPU_dynamic_impl_kind_cpp, TestImplKindMatmulCoreExec) {
    REQUIRE_AVX2();
    sc_graph_t g;
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    auto in_a = g.make_input(
            {graph_tensor::make({-1, 64}, sc_data_format_t::MK())});
    auto in_b = g.make_input(
            {graph_tensor::make({64, 32}, sc_data_format_t::KN())});

    auto mmm = g.make<customized_matmul_core_op_t>(
            std::vector<graph_tensor_ptr> {
                    in_a->get_outputs()[0], in_b->get_outputs()[0]},
            std::vector<graph_tensor_ptr> {graph_tensor::make({-1, 32})},
            any_map_t());
    auto out = g.make_output(mmm->get_outputs());
    // disable copy during fusion as copy may remake op with op name.
    g.attrs_.set("temp.disable_graph_fusion", 1);
    graph_driver(g, ctx);
    std::vector<sc_op_ptr> gargs {out, in_a, in_b};
    auto modu = lower_graph(ctx, g, gargs);
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(modu);
    test_buffer<float> sc_a(28 * 64), sc_b(64 * 32), sc_out(28 * 32),
            ref_out(28 * 32);
    test_utils::fill_data<float>(sc_a.data(), 28 * 64);
    test_utils::fill_data<float>(sc_b.data(), 32 * 64);
    runtime::dynamic_tensor_t dyn_a, dyn_b, dyn_out;
    sc_dims shape_a = {28, 64}, shape_b = {64, 32}, shape_out = {28, 32};
    dyn_a.data_ = sc_a.data();
    dyn_a.dims_ = shape_a.data();
    dyn_a.ndims_ = 2;
    dyn_a.dtype_ = uint32_t(sc_data_etype::F32);
    dyn_a.dyn_mask_ = 1 << 0;
    dyn_b.data_ = sc_b.data();
    dyn_b.dims_ = shape_b.data();
    dyn_b.ndims_ = 2;
    dyn_b.dtype_ = uint32_t(sc_data_etype::F32);
    dyn_b.dyn_mask_ = 0;
    dyn_out.data_ = sc_out.data();
    dyn_out.dims_ = shape_out.data();
    dyn_out.ndims_ = 2;
    fptr->call_default(&dyn_out, &dyn_a, &dyn_b);
    gemm_params l_param = {false, false, 28, 32, 64, 1.0, 0, 64, 32, 32};
    ref_gemm(l_param, sc_a.data(), sc_b.data(), ref_out.data());
    test_utils::compare_data(sc_out, ref_out, 1e-4f, 1e-4f);
}

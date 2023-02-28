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

#include <iostream>
#include <utility>
#include "context.hpp"
#include "test_utils.hpp"
#include "util/bf16.hpp"
#include "gtest/gtest.h"
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/managed_matmul_core.hpp>
#include <ops/templates/managed_matmul_core.hpp>
#include <reference/act_ref.hpp>
#include <reference/gemm_ref.hpp>
#include <runtime/config.hpp>
#include <util/reflection.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::ops;
using namespace dnnl::impl::graph::gc::test_utils;
struct managed_gemm_params_t {
    managed_gemm_params_t(sc_dims input_dims, sc_dims weight_dims,
            sc_dims out_dims, sc_data_type_t input_dtype = datatypes::f32,
            sc_data_type_t weight_dtype = datatypes::f32,
            bool is_input_constant = true)
        : input_dims_(std::move(input_dims))
        , weight_dims_(std::move(weight_dims))
        , out_dims_(std::move(out_dims))
        , input_dtype_(input_dtype)
        , weight_dtype_(weight_dtype)
        , is_input_constant_(is_input_constant) {}
    sc_dims input_dims_;
    sc_dims weight_dims_;
    sc_dims out_dims_;
    sc_data_type_t input_dtype_;
    sc_data_type_t weight_dtype_;
    bool is_input_constant_;
};

static bool is_param_valid(const managed_gemm_params_t &param,
        const managed_matmul_core_config_t &cfg, const int imm_block,
        const int imn_block, const int imk_block) {
    if (runtime_config_t::get().get_num_threads()
                    % (cfg.M_split_num * cfg.N_split_num)
            != 0) {
        std::cout << "Skip as the given splits are invalid, (M_split_num, "
                     "N_split_num)=("
                  << cfg.M_split_num << ", " << cfg.N_split_num
                  << "), while the number of threads is: "
                  << runtime_config_t::get().get_num_threads() << ". "
                  << std::endl;
        return false;
    }
    if (utils::divide_and_ceil(param.input_dims_[0], imm_block)
                            / cfg.M_split_num
                    < (size_t)cfg.M_sub_block
            || utils::divide_and_ceil(param.weight_dims_[1], imn_block)
                            / cfg.N_split_num
                    < (size_t)cfg.N_sub_block
            || utils::divide_and_ceil(param.weight_dims_[0], imk_block)
                            / (runtime_config_t::get().get_num_threads()
                                    / (cfg.M_split_num * cfg.N_split_num))
                    < (size_t)cfg.K_sub_block) {
        std::cout << "Skip as the given sub_blocks are invalid when the number "
                     "of threads is: "
                  << runtime_config_t::get().get_num_threads() << ". "
                  << std::endl;
        return false;
    }
    return true;
}

template <typename Atype, typename Btype>
void alloc_sc_input_and_weight(test_buffer<Atype> &sc_input,
        test_buffer<Btype> &sc_weight, const sc_dims input_dims,
        const sc_dims weight_dims) {
    sc_input = alloc_array<Atype>(cal_size(input_dims));
    sc_weight = alloc_array<Btype>(cal_size(weight_dims));
}

template <typename Atype, typename Btype, typename Ctype>
void run_mmm_test(const std::shared_ptr<jit_function_t> &fptr, int M, int N,
        int K, const sc_dims input_dims, const sc_dims weight_dims,
        const sc_dims out_dims, bool fuse_sigmoid = false) {
    test_buffer<Atype> sc_input;
    test_buffer<Btype> sc_weight;
    alloc_sc_input_and_weight(sc_input, sc_weight, input_dims, weight_dims);

    auto sc_output = alloc_array<Ctype>(cal_size(out_dims));
    auto ref_input = std::vector<Ctype>(sc_input.begin(), sc_input.end());
    auto ref_weight = std::vector<Ctype>(sc_weight.begin(), sc_weight.end());
    auto ref_output = std::vector<Ctype>(cal_size(out_dims));
    auto ref_output_s = std::vector<Ctype>(cal_size(out_dims));

    fptr->call_default(&sc_output[0], &sc_input[0], &sc_weight[0]);
    int out_size = M * N;

    gemm_params gemm_param {false, false, M, N, K, 1.0, 0.0, K, N, N};
    ref_gemm(gemm_param, &ref_input[0], &ref_weight[0], &ref_output[0]);
    if (fuse_sigmoid) {
        ref_sigmoid(ref_output_s.data(), ref_output.data(), cal_size(out_dims));
        test_utils::compare_data(
                sc_output.data(), ref_output_s.data(), out_size, 1e-3, 1e-3);
    } else {
        test_utils::compare_data(
                sc_output.data(), ref_output.data(), out_size, 1e-3, 1e-3);
    }
}

static void check_managed_matmul(const managed_gemm_params_t &param,
        const managed_matmul_core_config_t &cfg, bool default_cfg = false,
        bool fuse_sigmoid = false) {
    const sc_dims input_dims = param.input_dims_;
    const sc_dims weight_dims = param.weight_dims_;
    const sc_dims out_dims = param.out_dims_;
    const sc_data_type_t input_dtype = param.input_dtype_;
    const sc_data_type_t weight_dtype = param.weight_dtype_;
    sc_dims batch_dims;
    sc_graph_t graph;
    int M, N, K;
    bool is_quantized
            = utils::is_one_of(input_dtype, datatypes::u8, datatypes::s8);
    bool is_s8s8
            = input_dtype == datatypes::s8 && weight_dtype == datatypes::s8;
    bool is_u8s8
            = input_dtype == datatypes::u8 && weight_dtype == datatypes::s8;
    bool is_bf16
            = input_dtype == datatypes::bf16 && weight_dtype == datatypes::bf16;

    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.mixed_fusion_ = 1;

    M = input_dims[input_dims.size() - 2];
    N = weight_dims[weight_dims.size() - 1];
    K = input_dims[input_dims.size() - 1];

    auto data = graph.make_input(
            {graph_tensor::make(input_dims, sc_data_format_t(), input_dtype)});
    auto weight = graph.make_input({graph_tensor::make(
            weight_dims, sc_data_format_t(), weight_dtype)});
    auto mmm = graph.make("managed_matmul_core",
            {data->get_outputs()[0], weight->get_outputs()[0]},
            {graph_tensor::make(out_dims, sc_data_format_t(),
                    is_quantized ? datatypes::s32 : datatypes::f32)},
            {});
    if (default_cfg) {
        auto mmm_gen = mmm->dyn_cast<ops::managed_matmul_core_op_t>()
                               ->create_generator();
        auto dcfg = *(managed_matmul_core_config_t *)mmm_gen
                             ->get_default_config(get_default_context())
                             .get();
        mmm->stc_cast<tunable_op_t>()->set_config(
                reflection::general_object_t::make(dcfg));
    } else {
        auto mmm_gen = mmm->dyn_cast<ops::managed_matmul_core_op_t>()
                               ->create_generator();
        auto gen = static_cast<gen_managed_matmul_core_t *>(mmm_gen.get());
        if (!is_param_valid(param, cfg, gen->iim_block_, gen->iin_block_,
                    gen->iik_block_)) {
            GTEST_SKIP();
        }
        mmm->stc_cast<tunable_op_t>()->set_config(
                reflection::general_object_t::make(cfg));
    }

    mmm->dyn_cast<op_traits::may_quantize_t>()->is_quantized_ = is_quantized;
    sc_op_ptr output;
    if (fuse_sigmoid) {
        auto sig = graph.make("sigmoid", {mmm->get_outputs()[0]}, {}, {});
        output = graph.make_output(sig->get_outputs());
    } else {
        output = graph.make_output(mmm->get_outputs());
    }
    graph.attrs_[sc_graph_t::attr_key_t::quantize] = is_quantized;
    if (param.is_input_constant_) {
        data->attrs_.set("constant", const_kind::local_const);
        weight->attrs_.set("constant", const_kind::local_const);
    }
    graph_driver(graph, ctx);

    auto f = lower_graph(
            ctx, graph, std::vector<sc_op_ptr> {output, data, weight});
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f);

    if (is_quantized && is_s8s8) { // s8s8
        run_mmm_test<int8_t, int8_t, int32_t>(
                fptr, M, N, K, input_dims, weight_dims, out_dims, fuse_sigmoid);
    } else if (is_quantized && is_u8s8) { // u8s8
        run_mmm_test<uint8_t, int8_t, int32_t>(
                fptr, M, N, K, input_dims, weight_dims, out_dims, fuse_sigmoid);
    } else if (is_bf16) { // bf16
        run_mmm_test<bf16_t, bf16_t, float>(
                fptr, M, N, K, input_dims, weight_dims, out_dims, fuse_sigmoid);
    } else { // f32
        run_mmm_test<float, float, float>(
                fptr, M, N, K, input_dims, weight_dims, out_dims, fuse_sigmoid);
    }
}

const managed_matmul_core_config_t cfg1 = {
        28, // M_split_num
        1, // N_split_num
        2, // M_sub_block
        2, // N_sub_block
        2, // K_sub_block
        1, // im_loop_order
};

const managed_matmul_core_config_t cfg2 = {
        14, // M_split_num
        2, // N_split_num
        2, // M_sub_block
        2, // N_sub_block
        2, // K_sub_block
        1, // im_loop_order
};

const managed_matmul_core_config_t cfg3 = {
        7, // M_split_num
        2, // N_split_num
        3, // M_sub_block
        3, // N_sub_block
        3, // K_sub_block
        1, // im_loop_order
};

// f32
TEST(GCCore_managed_matmul_test, TestMATMUL2D_1) {
    check_managed_matmul({{1792, 1792}, {1792, 1792}, {1792, 1792},
                                 datatypes::f32, datatypes::f32, false},
            cfg1);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_2) {
    check_managed_matmul({{1792, 1792}, {1792, 1792}, {1792, 1792},
                                 datatypes::f32, datatypes::f32, false},
            cfg2);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_3) {
    check_managed_matmul({{1792, 1792}, {1792, 1792}, {1792, 1792},
                                 datatypes::f32, datatypes::f32, false},
            cfg3);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_4) {
    check_managed_matmul({{1792, 1792}, {1792, 1792}, {1792, 1792},
                                 datatypes::f32, datatypes::f32, false},
            managed_matmul_core_config_t(), true);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_5) {
    check_managed_matmul({{1792, 1792}, {1792, 1792}, {1792, 1792},
                                 datatypes::f32, datatypes::f32, true},
            cfg1);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_6) {
    check_managed_matmul({{1792, 1792}, {1792, 1792}, {1792, 1792},
                                 datatypes::f32, datatypes::f32, true},
            cfg2);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_7) {
    check_managed_matmul({{1792, 1792}, {1792, 1792}, {1792, 1792},
                                 datatypes::f32, datatypes::f32, true},
            cfg3);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_8) {
    check_managed_matmul({{1792, 1792}, {1792, 1792}, {1792, 1792},
                                 datatypes::f32, datatypes::f32, true},
            managed_matmul_core_config_t(), true);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_9) {
    check_managed_matmul({{1125, 1115}, {1115, 1120}, {1125, 1120},
                                 datatypes::f32, datatypes::f32, false},
            cfg1);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_10) {
    check_managed_matmul({{1125, 1115}, {1115, 1120}, {1125, 1120},
                                 datatypes::f32, datatypes::f32, false},
            cfg2);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_11) {
    check_managed_matmul({{1125, 1115}, {1115, 1120}, {1125, 1120},
                                 datatypes::f32, datatypes::f32, false},
            cfg3);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_12) {
    check_managed_matmul({{1125, 1115}, {1115, 1120}, {1125, 1120},
                                 datatypes::f32, datatypes::f32, false},
            managed_matmul_core_config_t(), true);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_13) {
    REQUIRE_VNNI();
    check_managed_matmul({{2250, 2230}, {2230, 2240}, {2250, 2240},
                                 datatypes::s8, datatypes::s8, false},
            cfg1);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_14) {
    REQUIRE_VNNI();
    check_managed_matmul({{2250, 2230}, {2230, 2240}, {2250, 2240},
                                 datatypes::s8, datatypes::s8, false},
            cfg2);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_15) {
    REQUIRE_VNNI();
    check_managed_matmul({{2250, 2230}, {2230, 2240}, {2250, 2240},
                                 datatypes::s8, datatypes::s8, false},
            cfg3);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_16) {
    REQUIRE_VNNI();
    check_managed_matmul({{2250, 2230}, {2230, 2240}, {2250, 2240},
                                 datatypes::s8, datatypes::s8, false},
            managed_matmul_core_config_t(), true);
}
// test iter anchor, currently only mmm that 1) outputs plain format; 2) has
// post fusion; 3) has splits on K; 4) has imbalance, will use iter anchor
TEST(GCCore_managed_matmul_test, TestMATMUL2D_FUSED_SIGMOID1) {
    check_managed_matmul({{912, 1344}, {1344, 912}, {912, 912}, datatypes::f32,
                                 datatypes::f32, false},
            cfg3, false, true);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_FUSED_SIGMOID2) {
    check_managed_matmul({{912, 1344}, {1344, 1360}, {912, 1360},
                                 datatypes::f32, datatypes::f32, false},
            cfg3, false, true);
}
TEST(GCCore_managed_matmul_test, TestMATMUL2D_FUSED_SIGMOID3) {
    check_managed_matmul({{1136, 912}, {912, 1344}, {1136, 1344},
                                 datatypes::f32, datatypes::f32, false},
            cfg3, false, true);
}

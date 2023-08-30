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
#include <limits>
#include "compiler/ir/graph/fusible_op_utils.hpp"
#include "context.hpp"
#include "util/bf16.hpp"
#include "gtest/gtest.h"
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/viewer.hpp>
#include <compiler/jit/jit.hpp>
#include <reference/gemm_ref.hpp>
#include <runtime/config.hpp>
#include <test_utils.hpp>
#include <util/any_map.hpp>
#include <util/parallel.hpp>
#include <util/utils.hpp>
using namespace dnnl::impl::graph::gc;

static bool verbose = false;
static const float the_atol = 1e-5f;
static const float the_rtol = 1e-4f;

template <typename Dtype>
static void do_test_reduce_op(const sc_dims &in_shape,
        const std::vector<int> &rd_axis, const std::string &reduce_name,
        int out_size, sc_data_type_t in_dtype,
        const std::function<std::vector<float>(std::vector<float> &)> &ref_func,
        bool keep_dims = false, const sc_dims &out_shape = {}) {
    REQUIRE_AVX2();
    sc_graph_t graph;
    auto input = graph.make_input(
            {graph_tensor::make(in_shape, sc_data_format_t(), in_dtype)});
    auto output_tensors = std::vector<graph_tensor_ptr>();
    if (out_shape.size() > 0) {
        output_tensors.emplace_back(
                graph_tensor::make(out_shape, sc_data_format_t(), in_dtype));
    }
    auto reduce
            = graph.make(reduce_name, {input->get_outputs()[0]}, output_tensors,
                    {
                            {"rd_axis", rd_axis},
                            {"keep_dims", keep_dims},
                    });
    auto output = graph.make_output(reduce->get_outputs());
    graph_driver(graph, get_test_ctx());
    auto f = lower_graph(get_test_ctx(), graph, {output, input});
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);

    auto in = alloc_array<Dtype>(test_utils::product(in_shape));
    auto out = alloc_array<Dtype>(out_size, INIT_NOOP);

    std::vector<generic_val> generic_args;
    generic_args.emplace_back(&out[0]);
    generic_args.emplace_back(&in[0]);

    fptr->call_generic_default(generic_args.data());

    auto ref_in = std::vector<float>(in.begin(), in.end());
    auto ref_out = ref_func(ref_in);
    auto out_f32 = std::vector<float>(out.begin(), out.end());
    if (verbose) {
        std::cout << ref_out[0] << " " << ref_out[1] << " " << ref_out[2] << " "
                  << std::endl;
        std::cout << out_f32[0] << " " << out_f32[1] << " " << out_f32[2] << " "
                  << std::endl;
    }
    test_utils::compare_data(out_f32, ref_out, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp1) {
    const int out_size = 3;
    do_test_reduce_op<float>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_sum", out_size, datatypes::f32,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = 0;
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] += input[i * 3 * 3 + j * 3 + k];
                return ref_out;
            });
}

// test reduce on all axis with predefined output tensor
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp2) {
    const int out_size = 1;
    do_test_reduce_op<float>(sc_dims({3, 3, 3}), std::vector<int>({0, 1, 2}),
            "reduce_sum", out_size, datatypes::f32,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                ref_out[0] = 0;
                for (size_t i = 0; i < input.size(); ++i)
                    ref_out[0] += input[i];
                return ref_out;
            });
}

// test reduce on all axis with auto-infered output tensor
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp3) {
    const int out_size = 1;
    do_test_reduce_op<float>(sc_dims({3, 3, 3, 3}),
            std::vector<int>({0, 1, 2, 3}), "reduce_sum", out_size,
            datatypes::f32, [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                ref_out[0] = 0;
                for (size_t i = 0; i < input.size(); ++i)
                    ref_out[0] += input[i];
                return ref_out;
            });
}

// test reduce mul
static void test_single_mul(int lastdim) {
    const int out_size = 3;
    do_test_reduce_op<float>(sc_dims({3, 3, lastdim}), std::vector<int>({1, 2}),
            "reduce_prod", out_size, datatypes::f32,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = 1;
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < (size_t)lastdim; ++k)
                            ref_out[i]
                                    *= input[i * 3 * lastdim + j * lastdim + k];
                return ref_out;
            });
}
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp4) {
    test_single_mul(3);
    SET_THREADS_OR_SKIP(1);
    test_single_mul(256);
}

// test reduce max
static void test_single_max() {
    const int out_size = 3;
    do_test_reduce_op<float>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_max", out_size, datatypes::f32,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] = std::max(
                                    ref_out[i], input[i * 3 * 3 + j * 3 + k]);
                return ref_out;
            });
}

TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp5) {
    test_single_max();
    SET_THREADS_OR_SKIP(1);
    test_single_max();
}

// test reduce mean
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp6) {
    const int out_size = 3;
    do_test_reduce_op<float>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_mean", out_size, datatypes::f32,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = 0;
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] += input[i * 3 * 3 + j * 3 + k] / 9;
                return ref_out;
            });
}

// test reduce min
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp7) {
    const int out_size = 3;
    do_test_reduce_op<float>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_min", out_size, datatypes::f32,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] = std::min(
                                    ref_out[i], input[i * 3 * 3 + j * 3 + k]);
                return ref_out;
            });
}

// test bf16 reduce max
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp8) {
    const int out_size = 3;
    do_test_reduce_op<bf16_t>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_max", out_size, datatypes::bf16,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i) {
                    ref_out[i] = -std::numeric_limits<float>::infinity();
                }
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] = std::max(
                                    ref_out[i], input[i * 3 * 3 + j * 3 + k]);
                return ref_out;
            });
}

// test bf16 reduce min
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp9) {
    const int out_size = 3;
    do_test_reduce_op<bf16_t>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_min", out_size, datatypes::bf16,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] = std::min(
                                    ref_out[i], input[i * 3 * 3 + j * 3 + k]);
                return ref_out;
            });
}

// test s8 reduce max
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp10) {
    REQUIRE_PARALLEL();
    const int out_size = 3;
    do_test_reduce_op<int8_t>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_max", out_size, datatypes::s8,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i) {
                    ref_out[i] = -std::numeric_limits<float>::infinity();
                }
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] = std::max(
                                    ref_out[i], input[i * 3 * 3 + j * 3 + k]);
                return ref_out;
            });
}

// test s8 reduce min
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp11) {
    REQUIRE_PARALLEL();
    const int out_size = 3;
    do_test_reduce_op<int8_t>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_min", out_size, datatypes::s8,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] = std::min(
                                    ref_out[i], input[i * 3 * 3 + j * 3 + k]);
                return ref_out;
            });
}

// test u8 reduce max
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp12) {
    REQUIRE_PARALLEL();
    const int out_size = 3;
    do_test_reduce_op<uint8_t>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_max", out_size, datatypes::u8,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i) {
                    ref_out[i] = -std::numeric_limits<float>::infinity();
                }
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] = std::max(
                                    ref_out[i], input[i * 3 * 3 + j * 3 + k]);
                return ref_out;
            });
}

// test u8 reduce min
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp13) {
    REQUIRE_PARALLEL();
    const int out_size = 3;
    do_test_reduce_op<uint8_t>(sc_dims({3, 3, 3}), std::vector<int>({1, 2}),
            "reduce_min", out_size, datatypes::u8,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] = std::min(
                                    ref_out[i], input[i * 3 * 3 + j * 3 + k]);
                return ref_out;
            });
}

// test all reduce + partial reduce + last reduce
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOp14) {
    const int out_size = 1;
    // set num threads to trigger corner condition
    SET_THREADS_OR_SKIP(4);
    do_test_reduce_op<float>(sc_dims({3, 3, 3, 3}),
            std::vector<int>({0, 1, 2, 3}), "reduce_sum", out_size,
            datatypes::f32, [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                ref_out[0] = 0;
                for (size_t i = 0; i < input.size(); ++i)
                    ref_out[0] += input[i];
                return ref_out;
            });
}

// test reduce on all axis with fusing enabled
TEST(GCCore_CPU_reduce_op_cpp, TestReduceOpFuse) {
    REQUIRE_AVX2();
    // auto skip in avoid of reduce split
    if (runtime_config_t::get().get_num_threads() < 6) { GTEST_SKIP(); }
    sc_graph_t graph;
    sc_dims in_shape = {3, 3, 3, 3};
    auto input_a = graph.make_input(
            {graph_tensor::make(in_shape, sc_data_format_t())});
    auto input_b = graph.make_input(
            {graph_tensor::make(in_shape, sc_data_format_t())});
    auto input_c = graph.make_input(
            {graph_tensor::make(sc_dims({1}), sc_data_format_t())});
    auto add = graph.make("add",
            {input_a->get_outputs()[0], input_b->get_outputs()[0]}, {}, {});
    auto reduce = graph.make("reduce", {add->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int>({0, 1, 2, 3})}, {"keep_dims", false},
                    {"rd_op", 0}});
    auto mul = graph.make("mul",
            {reduce->get_outputs()[0], input_c->get_outputs()[0]}, {}, {});
    auto output = graph.make_output(mul->get_outputs());
    graph_driver(graph, get_test_ctx());

    std::stringstream ss;
    print_graph(graph, ss, true);
    std::string expected_str
            = R"(graph(v0: f32[3, 3, 3, 3], v1: f32[3, 3, 3, 3], v2: f32[1]) -> [v3: f32[1]] {
  [v3: f32[1]] = add_reduce_mul(v0, v1, v2)
}
)";
    EXPECT_EQ(ss.str(), expected_str);

    auto f = lower_graph(
            get_test_ctx(), graph, {output, input_a, input_b, input_c});

    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);

    auto in_a = alloc_array<float>(test_utils::product(in_shape));
    auto in_b = alloc_array<float>(test_utils::product(in_shape));
    auto in_c = alloc_array<float>(1);
    auto out = alloc_array<float>(1, INIT_NOOP);
    auto ref_out = std::vector<float>(1);

    std::vector<generic_val> generic_args;
    generic_args.emplace_back(&out[0]);
    generic_args.emplace_back(&in_a[0]);
    generic_args.emplace_back(&in_b[0]);
    generic_args.emplace_back(&in_c[0]);
    fptr->call_generic_default(generic_args.data());

    for (size_t i = 0; i < in_a.size(); i++)
        ref_out[0] += (in_a[i] + in_b[i]) * in_c[0];

    if (verbose) {
        for (size_t i = 0; i < in_a.size(); i++) {
            std::cout << in_a[i] << " ";
            if ((i + 1) % 3 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < in_b.size(); i++) {
            std::cout << in_b[i] << " ";
            if ((i + 1) % 3 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << in_c[0] << std::endl;
        std::cout << out[0] << std::endl;
        std::cout << ref_out[0] << std::endl;
    }
    test_utils::compare_data(out, ref_out, the_rtol, the_atol);
}

class reduce_checker : public ir_viewer_t {
public:
    int num_assign = 0;
    int num_reduce_add = 0;
    bool is_var_ = true;
    const char *buffer_name = "_reduce_compute_buf";
    void view(assign_c v) override {
        if (v->var_->dtype_
                == sc_data_type_t::f32(get_test_ctx()->get_max_vector_lanes(
                        sc_data_etype::F32))) {
            std::stringstream ss;
            v->var_->to_string(ss);
            if (ss.str().find(buffer_name) != std::string::npos) {
                num_assign++;
            }
        }
        ir_viewer_t::view(v);
    }

    void view(tensor_c v) override {
        if (utils::string_startswith(v->name_, buffer_name)) {
            is_var_ &= v->attr_
                    && v->attr_->get_or_else("must_tensor2var", false);
        }
    }

    void view(intrin_call_c v) override {
        if (v->type_ == intrin_type::reduce_add) {
            std::stringstream ss;
            v->to_string(ss);
            if (ss.str().find("_reduce_compute_buf") != std::string::npos) {
                num_reduce_add++;
            }
        }
    }
};
// test gemm+relu+reduce+add. Reduction on K axis for MK
static void do_test_last_axis(bool &done, bool reduce_output, bool input_plain,
        bool keep_dims, bool mean, test_buffer<float> &out,
        test_buffer<float> &refout, int rd_axis = 1, int num_threads = 16,
        bool use_mixed_fuse = false) {
    done = false;
    const int shape = 1024;
    sc_graph_t graph;
    SET_THREADS_OR_SKIP(num_threads);
    // make sure matmul has last level fusion anchor
    sc_data_format_t fmt1 = input_plain ? sc_data_format_t::MK()
                                        : sc_data_format_t::MKmk(32, 32);
    sc_data_format_t fmt2 = sc_data_format_t::MK();
    auto input_a = graph.make_input({graph_tensor::make({shape, shape}, fmt1),
            graph_tensor::make({shape, shape}, fmt2)});
    auto mm = graph.make("matmul_core", input_a->get_outputs(), {}, {});
    auto relu = graph.make("relu", {mm->get_outputs()[0]}, {}, {});
    std::vector<int> rdax {rd_axis};
    if (rd_axis == -1) {
        // all reduce
        rdax = {0, 1};
    }
    std::string reduce_op_name = mean ? "reduce_mean" : "reduce_sum";
    auto reduce = graph.make(reduce_op_name, {relu->get_outputs()[0]}, {},
            {
                    {"rd_axis", std::move(rdax)},
                    {"keep_dims", keep_dims},
            });
    if (!reduce_output) {
        reduce = graph.make("add",
                {reduce->get_outputs()[0], input_a->get_outputs()[0]}, {}, {});
    }

    auto output = graph.make_output(reduce->get_outputs());

    context_ptr ctx;
    if (use_mixed_fuse) {
        ctx = std::make_shared<context_t>(*get_test_ctx());
        ctx->flags_.mixed_fusion_ = true;
    } else {
        ctx = get_test_ctx();
    }
    graph_driver(graph, ctx);

    auto f = lower_graph(ctx, graph, {output, input_a});

    reduce_checker chk;
    func_t thefunc;

    for (auto &func : f->get_contents()) {
        if (func->name_.find("matmul_core") != std::string::npos) {
            thefunc = func;
            if (!use_mixed_fuse) break;
        }
    }
    ASSERT_TRUE(thefunc);
    if (use_mixed_fuse) {
        // check that local-reduce-compute is enabled
        chk.buffer_name = "reduce_compute_";
        chk.dispatch(thefunc);
        ASSERT_EQ(chk.num_assign, 1);
        ASSERT_EQ(chk.is_var_, true);
    }
    if (rd_axis == 1) {
        chk.dispatch(thefunc);
        ASSERT_EQ(chk.num_assign, 1);
        ASSERT_EQ(chk.num_reduce_add, 1);
    } else {
        ASSERT_TRUE(thefunc->name_.find("reduce_compute") != std::string::npos);
    }

    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);
    if (!runtime_config_t::get().set_num_threads(reseter__.old_)) {
        GTEST_SKIP();
    }

    auto in_a = alloc_array<float>(shape * shape);
    auto &fmt = input_a->get_outputs()[0]->details_.get_format();
    test_buffer<float> in_a_plain;
    auto in_b = alloc_array<float>(shape * shape);
    size_t out_size;
    if (reduce_output) {
        if (rd_axis == -1) {
            out_size = 1;
        } else {
            out_size = shape;
        }
    } else {
        out_size = shape * shape;
    }
    out = alloc_array<float>(out_size, INIT_NOOP);

    auto ref_out = alloc_array<float>(shape * shape, INIT_NOOP);
    auto ref_tmp = alloc_array<float>((rd_axis == -1) ? 1 : shape, INIT_ZERO);
    gemm_params gparams {
            false, false, shape, shape, shape, 1.0f, 0.0f, shape, shape, shape};
    if (input_plain) {
        ref_gemm(gparams, in_a.data(), in_b.data(), ref_out.data());
    } else {
        in_a_plain = MKmk2MK(in_a, shape / fmt.blocks_[0],
                shape / fmt.blocks_[1], fmt.blocks_[0], fmt.blocks_[1]);
        ref_gemm(gparams, in_a_plain.data(), in_b.data(), ref_out.data());
    }
    test_buffer<float> &plain_a = input_plain ? in_a : in_a_plain;
    for (auto &v : ref_out) {
        v = std::max(v, 0.0f);
    }
    if (rd_axis == 1) {
        utils::parallel_for(0, shape, 1, [&](int64_t i) {
            for (int j = 0; j < shape; j++) {
                ref_tmp[i] += ref_out[i * shape + j];
            }
        });
    } else if (rd_axis == 0) {
        utils::parallel_for(0, shape, 1, [&](int64_t j) {
            for (int i = 0; i < shape; i++) {
                ref_tmp[j] += ref_out[i * shape + j];
            }
        });
    } else {
        for (auto v : ref_out) {
            ref_tmp[0] += v;
        }
    }
    if (mean) {
        utils::parallel_for(
                0, ref_tmp.size(), 1, [&](int64_t b) { ref_tmp[b] /= shape; });
    }

    if (!reduce_output) {
        utils::parallel_for(0, shape, 1, [&](int64_t i) {
            for (int j = 0; j < shape; j++) {
                ref_out[i * shape + j] = ref_tmp[i] + plain_a[i * shape + j];
            }
        });
        refout = std::move(ref_out);
    } else {
        refout = std::move(ref_tmp);
    }

    if (!runtime_config_t::get().set_num_threads(num_threads)) { GTEST_SKIP(); }
    fptr->call_default(out.data(), in_a.data(), in_b.data());
    done = true;
}

TEST(GCCore_CPU_reduce_op_cpp, TestPartialReduceAsOutput) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    bool done = false;
    do_test_last_axis(done, true, true, true, false, out, refout, 0);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestPaddingReduceAsOutput) {
    REQUIRE_AVX2();
    sc_graph_t graph;
    auto ctx = get_test_ctx();
    const int cols = vectorize_step(ctx, sc_data_etype::F32) - 1;
    sc_dims input_shape = {32, cols};
    auto input0 = graph.make_input(
            {graph_tensor::make({32, cols}, sc_data_format_t::MK())});
    auto input1 = graph.make_input(
            {graph_tensor::make({32, cols}, sc_data_format_t::MK())});
    auto reo0 = graph.make("reorder", {input0->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(16, cols + 1)},
                    {"internal", true}});
    auto reo1 = graph.make("reorder", {input1->get_outputs()[0]}, {},
            {{"out_format", sc_data_format_t::MKmk(16, cols + 1)},
                    {"internal", true}});
    auto add0 = graph.make(
            "add", {reo0->get_outputs()[0], reo1->get_outputs()[0]}, {}, {});
    auto reduce0 = graph.make("reduce", add0->get_outputs(), {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 2}});
    auto output0 = graph.make_output(reduce0->get_outputs());

    mixed_partition(graph, ctx);

    auto f = lower_graph(get_test_ctx(), graph, {output0, input0, input1});
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(f, true);

    auto in_a = alloc_array<float>(test_utils::product(input_shape),
            init_action::INIT_RANGE, -100, -1);
    auto in_b = alloc_array<float>(test_utils::product(input_shape),
            init_action::INIT_RANGE, -100, -1);
    // auto in_c = alloc_array<float>(1);
    auto out = alloc_array<float>(32, INIT_NOOP);
    auto ref_out = std::vector<float>(32);

    std::vector<generic_val> generic_args;
    generic_args.emplace_back(&out[0]);
    generic_args.emplace_back(&in_a[0]);
    generic_args.emplace_back(&in_b[0]);
    fptr->call_generic_default(generic_args.data());

    for (auto i = 0; i < input_shape[0]; i++) {
        float max = -std::numeric_limits<float>::infinity();
        for (auto j = 0; j < cols; j++) {
            max = std::max(max, in_a[i * cols + j] + in_b[i * cols + j]);
        }
        ref_out[i] = max;
    }
    if (verbose) {
        for (size_t i = 0; i < in_a.size(); i++) {
            std::cout << in_a[i] << " ";
            if ((i + 1) % cols == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
        for (size_t i = 0; i < in_b.size(); i++) {
            std::cout << in_b[i] << " ";
            if ((i + 1) % cols == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << out[0] << std::endl;
        std::cout << ref_out[0] << std::endl;
    }
    test_utils::compare_data(out, ref_out, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestPartialReduceAsOutputMixedFuse) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    bool done = false;
    do_test_last_axis(done, true, true, true, false, out, refout, 0, 16, true);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestPartialReduceAsOutputNoKeepDims) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    bool done = false;
    do_test_last_axis(done, true, true, false, false, out, refout, 0);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestPartialReduceAsOutputAllReduce) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    bool done = false;
    do_test_last_axis(done, true, true, true, false, out, refout, -1);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceAsOutput) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    bool done = false;
    do_test_last_axis(done, true, true, true, false, out, refout);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceAsOutputNoKeepDims) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    bool done = false;
    do_test_last_axis(done, true, true, false, false, out, refout);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceNotOutput) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    bool done = false;
    do_test_last_axis(done, false, true, true, false, out, refout);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceBlockingNotOutput) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    bool done = false;
    do_test_last_axis(done, false, false, true, false, out, refout);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceAsOutputNoKeepDimsMean) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    bool done = false;
    do_test_last_axis(done, true, true, false, true, out, refout);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

// test bmm+relu+reduce+add. Reduction on M axis for B_MN
static void do_test_not_last_axis(bool reduce_output, bool input_plain,
        bool keep_dims, bool mean, test_buffer<float> &out,
        test_buffer<float> &refout) {
    sc_graph_t graph;
    sc_data_format_t fmt1 = input_plain
            ? sc_data_format_t(format_kinds::ABCD)
            : sc_data_format_t(format_kinds::ABCDcd, {32, 32});
    sc_data_format_t fmt2 = sc_data_format_t(format_kinds::ABCD);
    const int B = 128, M = 256, N = 256, K = 256;
    auto input_a
            = graph.make_input({graph_tensor::make({16, 8, 256, 256}, fmt1),
                    graph_tensor::make({16, 8, 256, 256}, fmt2)});
    auto mm = graph.make("matmul_core", input_a->get_outputs(), {}, {});
    auto relu = graph.make("relu", {mm->get_outputs()[0]}, {}, {});
    std::string reduce_op_name = mean ? "reduce_mean" : "reduce_sum";
    auto reduce = graph.make(reduce_op_name, {relu->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int>({2})}, {"keep_dims", keep_dims}});
    if (!reduce_output) {
        reduce = graph.make("add",
                {reduce->get_outputs()[0], input_a->get_outputs()[0]}, {}, {});
    }

    auto output = graph.make_output(reduce->get_outputs());

    graph_driver(graph, get_test_ctx());

    auto f = lower_graph(get_test_ctx(), graph, {output, input_a});

    reduce_checker chk;
    if (reduce_output) { chk.buffer_name = "__outs_0"; }
    func_t thefunc;

    for (auto &func : f->get_contents()) {
        if (func->name_.find("matmul_core") == 0) {
            thefunc = func;
            break;
        }
    }
    ASSERT_TRUE(thefunc);
    chk.dispatch(thefunc);
    ASSERT_EQ(chk.num_assign, 1);
    ASSERT_EQ(chk.num_reduce_add, 0);

    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);

    auto in_a = alloc_array<float>(B * M * K);
    auto &fmt = input_a->get_outputs()[0]->details_.get_format();
    test_buffer<float> in_a_plain;
    auto in_b = alloc_array<float>(B * K * N);
    out = alloc_array<float>(reduce_output ? B * K : B * M * N, INIT_NOOP);

    auto ref_out = alloc_array<float>(B * M * N, INIT_NOOP);
    auto ref_tmp = alloc_array<float>(B * N, INIT_ZERO);
    gemm_params gparams {false, false, M, N, K, 1.0f, 0.0f, M, N, K};
    test_buffer<float> &plain_a = input_plain ? in_a : in_a_plain;
    if (!input_plain) {
        in_a_plain = batch_MKmk2MK(in_a, B, M / fmt.blocks_[0],
                K / fmt.blocks_[1], fmt.blocks_[0], fmt.blocks_[1]);
    }
    for (int b = 0; b < B; b++) {
        ref_gemm(gparams, plain_a.data() + b * M * K, in_b.data() + b * K * N,
                ref_out.data() + b * M * N);
    }
    for (auto &v : ref_out) {
        v = std::max(v, 0.0f);
    }
    utils::parallel_for(0, B, 1, [&](int64_t b) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ref_tmp[b * N + j] += ref_out[b * M * N + i * N + j];
            }
        }
    });
    if (mean) {
        utils::parallel_for(
                0, ref_tmp.size(), 1, [&](int64_t b) { ref_tmp[b] /= 256; });
    }

    if (!reduce_output) {
        utils::parallel_for(0, B, 1, [&](int64_t b) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    ref_out[b * M * N + i * N + j] = ref_tmp[b * N + j]
                            + plain_a[b * M * N + i * N + j];
                }
            }
        });
        refout = std::move(ref_out);
    } else {
        refout = std::move(ref_tmp);
    }
    fptr->call_default(out.data(), in_a.data(), in_b.data());
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceNotLastAxisNotOutput) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    do_test_not_last_axis(false, true, true, false, out, refout);
    test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceNotLastAxisAsOutput) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    do_test_not_last_axis(true, true, true, false, out, refout);
    test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp,
        TestTwoStageReduceNotLastAxisAsOutputNoKeepDims) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    do_test_not_last_axis(true, true, false, false, out, refout);
    test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceNotLastAxisBlockingNotOutput) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    do_test_not_last_axis(false, false, true, false, out, refout);
    test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceNotLastAxisAsOutputMean) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    do_test_not_last_axis(true, true, true, true, out, refout);
    test_utils::compare_data(out, refout, the_rtol, the_atol);
}

static void do_no_main_op_single_core(
        test_buffer<float> &out, test_buffer<float> &refout) {
    thread_num_reset reseter;
    sc_graph_t graph;
    const int shape = 1024;
    if (!runtime_config_t::get().set_num_threads(1)) { GTEST_SKIP(); }
    // make sure matmul has last level fusion anchor
    sc_data_format_t fmt1 = sc_data_format_t::MK();
    std::vector<int> rdax {0};
    auto input_a = graph.make_input({graph_tensor::make({shape, shape}, fmt1)});
    auto relu = graph.make("relu", {input_a->get_outputs()[0]}, {}, {});
    auto reduce = graph.make("reduce", {relu->get_outputs()[0]}, {},
            {{"rd_axis", std::move(rdax)}, {"keep_dims", true}, {"rd_op", 0}});

    auto output = graph.make_output(reduce->get_outputs());

    graph_driver(graph, get_test_ctx());

    auto f = lower_graph(get_test_ctx(), graph, {output, input_a});
    func_t thefunc;

    for (auto &func : f->get_contents()) {
        if (func->name_.find("relu_reduce") == 0) {
            thefunc = func;
            break;
        }
    }
    ASSERT_TRUE(thefunc);

    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);
    if (!runtime_config_t::get().set_num_threads(reseter.old_)) {
        GTEST_SKIP();
    }
    auto in_a = alloc_array<float>(shape * shape);
    size_t out_size = shape;

    out = alloc_array<float>(out_size, INIT_NOOP);

    auto ref_out = in_a.copy();
    auto ref_tmp = alloc_array<float>(shape, INIT_ZERO);

    for (auto &v : ref_out) {
        v = std::max(v, 0.0f);
    }

    utils::parallel_for(0, shape, 1, [&](int64_t j) {
        for (int i = 0; i < shape; i++) {
            ref_tmp[j] += ref_out[i * shape + j];
        }
    });

    refout = std::move(ref_tmp);

    if (!runtime_config_t::get().set_num_threads(1)) { GTEST_SKIP(); }
    fptr->call_default(out.data(), in_a.data());
}

TEST(GCCore_CPU_reduce_op_cpp, TestTwoStageReduceNoMainOp) {
    REQUIRE_AVX2();
    test_buffer<float> out;
    test_buffer<float> refout;
    do_no_main_op_single_core(out, refout);
    test_utils::compare_data(out, refout, the_rtol, the_atol);
}

static void do_test_bf16(
        test_buffer<bf16_t> &out, test_buffer<bf16_t> &refout, bool &done) {
    thread_num_reset reseter;
    sc_graph_t graph;
    const int shape = 1024;
    done = false;
    if (!runtime_config_t::get().set_num_threads(16)) { GTEST_SKIP(); }
    // make sure matmul has last level fusion anchor
    sc_data_format_t fmt1 = sc_data_format_t::MK();
    auto input_a = graph.make_input(
            {graph_tensor::make({shape, shape}, fmt1, datatypes::bf16)});
    std::vector<int> rdax {0};

    auto reduce = graph.make("reduce", {input_a->get_outputs()[0]}, {},
            {{"rd_axis", std::move(rdax)}, {"keep_dims", true}, {"rd_op", 0}});

    auto output = graph.make_output(reduce->get_outputs());

    graph_driver(graph, get_test_ctx());

    auto f = lower_graph(get_test_ctx(), graph, {output, input_a});
    reduce_checker chk;
    func_t thefunc;

    for (auto &func : f->get_contents()) {
        if (func->name_.find("reduce_compute") != std::string::npos) {
            thefunc = func;
            break;
        }
    }
    ASSERT_TRUE(thefunc);

    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f, true);
    if (!runtime_config_t::get().set_num_threads(reseter.old_)) {
        GTEST_SKIP();
    }

    auto in_a = alloc_array<bf16_t>(shape * shape);
    out = alloc_array<bf16_t>(shape, INIT_NOOP);
    auto ref_outf32 = alloc_array<float>(shape, INIT_ZERO);
    refout = alloc_array<bf16_t>(shape, INIT_NOOP);

    utils::parallel_for(0, shape, 1, [&](int64_t j) {
        for (int i = 0; i < shape; i++) {
            ref_outf32[j] += in_a[i * shape + j];
        }
    });
    utils::parallel_for(
            0, shape, 1, [&](int64_t j) { refout[j] = bf16_t(ref_outf32[j]); });
    if (!runtime_config_t::get().set_num_threads(16)) { GTEST_SKIP(); }
    fptr->call_default(out.data(), in_a.data());
    done = true;
}

TEST(GCCore_CPU_reduce_op_cpp, TestPartialBf16) {
    REQUIRE_BF16();
    test_buffer<bf16_t> out;
    test_buffer<bf16_t> refout;
    bool done = false;
    do_test_bf16(out, refout, done);
    if (done) test_utils::compare_data(out, refout, the_rtol, the_atol);
}

TEST(GCCore_CPU_reduce_op_cpp, TestReduceOpkeepDims1) {
    const int out_size = 3;
    do_test_reduce_op<float>(
            sc_dims({3, 3, 3}), std::vector<int>({1, 2}), "reduce_mean",
            out_size, datatypes::f32,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = 0;
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] += input[i * 3 * 3 + j * 3 + k] / 9;
                return ref_out;
            },
            true);
}

TEST(GCCore_CPU_reduce_op_cpp, TestReduceOpkeepDims2) {
    const int out_size = 3;
    do_test_reduce_op<float>(
            sc_dims({3, 3, 3}), std::vector<int>({1, 2}), "reduce_mean",
            out_size, datatypes::f32,
            [&](std::vector<float> &input) {
                auto ref_out = std::vector<float>(out_size, 0);
                for (size_t i = 0; i < ref_out.size(); ++i)
                    ref_out[i] = 0;
                for (size_t i = 0; i < 3; ++i)
                    for (size_t j = 0; j < 3; ++j)
                        for (size_t k = 0; k < 3; ++k)
                            ref_out[i] += input[i * 3 * 3 + j * 3 + k] / 9;
                return ref_out;
            },
            false, sc_dims({3, 1, 1}));
}

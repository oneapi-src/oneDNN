/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include <chrono>
#include <iostream>
#include <utility>
#include <vector>
#include "context.hpp"
#include "util/bf16.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <compiler/ir/visitor.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <ops/matmul_core.hpp>
#include <ops/reduce_mean.hpp>
#include <ops/templates/matmul_core.hpp>
#include <reference/act_ref.hpp>
#include <test_utils.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;
using namespace std::chrono;

static auto ctx = get_test_ctx();

static void do_commit(fusion_manager &mgr, const std::vector<expr> &fuse_outs,
        const std::vector<expr> &additional_ins = {}) {
    fuse_state_t fstate;
    auto out_failed = mgr.prepare_and_check(ctx, fstate);
    ASSERT_TRUE(out_failed.empty());
    mgr.commit(std::make_shared<ir_module_t>(ctx), fstate, fuse_outs,
            additional_ins);
}

#define CMP_SIMPLIFIED_IR(AAA, BBB) \
    ir_simplifier_t simp {false}; \
    ir_comparer cmper(true); \
    EXPECT_TRUE(cmper.compare(simp(AAA), simp(BBB), false));

static logical_tensor_t make_tsr(
        const sc_dims &dims, sc_data_type_t dtype = datatypes::s32) {
    return logical_tensor_t(sc_data_format_t(), dims, dtype);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerElemBlock) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput = fusion.make<input_op>(make_tsr({100}, datatypes::s32));
    auto frelu = fusion.make<relu_op_t>(finput->get_outputs()[0]);
    auto fmul = fusion.make<mul_op_t>(
            frelu->get_outputs()[0], finput->get_outputs()[0]);
    auto foutput = fusion.make<output_op>(fmul->get_outputs()[0]);
    EXPECT_EQ(fmul->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa, _arg_("ii", datatypes::s32),
            _arg_("jj", datatypes::s32), _arg_("buf", datatypes::s32, {100}),
            _arg_("out", datatypes::s32, {100})) {
        _bind_(ii, jj, buf, out);
        _for_(i, 0, 10, 1, for_type::PARALLEL) { buf[i] = ii + jj; }
        fusion.create_output_fusion_anchor({tensor_slice(buf, {{0, 32}})},
                {tensor_slice(out, {{20, 32}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[3]});
    int32_t lanes = frelu->get_vx_info().lanes;
    ///// Expected func:
    _function_(datatypes::s32, bbb, _arg_("ii", datatypes::s32),
            _arg_("jj", datatypes::s32), _arg_("buf", datatypes::s32, {100}),
            _arg_("out", datatypes::s32, {100})) {
        _bind_(ii, jj, buf, out);
        _for_(i, 0, 10, 1, for_type::PARALLEL) { buf[i] = ii + jj; }
        _tensor_(tbuf, datatypes::s32, {100UL});
        auto tb = tensor_ptr(tbuf, {0}, {}, true);
        auto b = tensor_ptr(buf, {0}, {}, true);
        _for_(i, 0, 32, lanes) {
            tb[span_t({i}, lanes)] = builder::make_max(b[span_t({i}, lanes)],
                    builder::make_constant(
                            {0UL}, sc_data_type_t(sc_data_etype::S32, lanes)));
            auto o = tensor_ptr(out, {i + UINT64_C(20)}, {}, true);
            tb = tensor_ptr(tbuf, {i}, {}, true);
            b = tensor_ptr(buf, {i}, {}, true);
            o[span_t({0}, lanes)]
                    = tb[span_t({0}, lanes)] * b[span_t({0}, lanes)];
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

namespace fusiontest {

// generates a new function specialized with the input shape
func_t get_block_compute_func(
        sc_data_type_t dtype, const std::vector<expr> &shapes) {
    _function_(datatypes::void_t, bias_add, _arg_("in", dtype, {1}),
            _arg_("out", dtype, {1}), _arg_("bias", dtype, {1})) {
        _bind_(in, out, bias);
        // func body omitted
    }
    return bias_add;
}

} // namespace fusiontest

#define get_detail(v) fdmap.get(v)

// A fuser that generates a function call on the slices of in & out tensors
class block_func_call_fuser_t : public fusible_op_t {
public:
    func_t addfunc;

    void prepare_fusion_data(fdata_map &fdmap) override {
        get_detail(info_.outputs_[0]).need_alloc_ = false;
    }
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override {
        infer_binary_slice_ranges(this, fsmap, stat_map);
    }
    void pre_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override {};
    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs) override {
        addfunc = fusiontest::get_block_compute_func(
                info_.inputs_[0]->details_.dtype_, {});
        auto shape = dst[0]->get_shape();
        EXPECT_EQ(get_const_as_int(shape[0].checked_as<constant>()), 10);
        EXPECT_EQ(get_const_as_int(shape[1].checked_as<constant>()), 200);
        expr callnode = addfunc(inputs[0]->get_tensor_ptr(),
                dst[0]->get_tensor_ptr(), inputs[1]->get_tensor_ptr());
        builder::get_current_builder()->push_evaluate(callnode);
        // We should mark the function call as "must_inline", because we are
        // operating on a tensor slice and the slice is not continous in
        // memory.
        // Using tensor_ptr() with normal function call will not work
        callnode->attr()["inline_level"] = 2;
    }
    block_func_call_fuser_t(graph_tensor_ptr in, graph_tensor_ptr arg) {
        info_.inputs_ = {std::move(in), std::move(arg)};
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        info_.tensor_share_info_ = {{0, {0}}};
    }
};

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerBlock) {
    builder::ir_builder_t builder;
    fusion_manager fusion2;
    auto finput = fusion2.make<input_op>(make_tsr({100, 200}, datatypes::s32));
    auto finput_bias = fusion2.make<input_op>(sc_dims {1, 200}, datatypes::s32);
    auto toutput = fusion2.make<block_func_call_fuser_t>(
            finput->get_outputs()[0], finput_bias->get_outputs()[0]);
    auto output = fusion2.make<output_op>(toutput->get_outputs()[0]);
    _function_(datatypes::s32, aaa, _arg_("buf", datatypes::s32, {100, 200}),
            _arg_("out", datatypes::s32, {100, 200}),
            _arg_("bias", datatypes::s32, {1, 200})) {
        _bind_(buf, out, bias);
        fusion2.create_output_fusion_anchor(
                {tensor_slice(buf, {/*dim1*/ {0, 10}, /*dim2*/ {0, 200}})},
                {tensor_slice(out, {/*dim1*/ {20, 10}, /*dim2*/ {0, 200}})});
        _return_(123);
    }

    do_commit(fusion2, {aaa->params_[1]}, {aaa->params_[2]});

    _function_(datatypes::s32, bbb, _arg_("buf", datatypes::s32, {100, 200}),
            _arg_("out", datatypes::s32, {100, 200}),
            _arg_("bias", datatypes::s32, {1, 200})) {
        _bind_(buf, out, bias);
        _evaluate_call_(toutput->addfunc, tensor_ptr(buf, {0, 0}, {}, true),
                tensor_ptr(out, {20, 0}, {}, true),
                tensor_ptr(bias, {0, 0}, {}, true));
        _return_(123);
    }
    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerComplex) {
    builder::ir_builder_t builder;
    class testfusion_manager_t : public fusion_manager {
    public:
        using fusion_manager::do_allocate_tensor;
        using fusion_manager::do_infer_slice_ranges;
        using fusion_manager::do_prepare_fusion_data;
        using fusion_manager::init_sorted_ops;
        using fusion_manager::input_idx_map_;
        using fusion_manager::output_idx_map_;
    };
    testfusion_manager_t fusion2;
    auto finput = fusion2.make<input_op>(make_tsr({100, 200}, datatypes::bf16));
    auto relu = fusion2.make<relu_op_t>(finput->get_outputs()[0]);
    auto mul = fusion2.make<mul_op_t>(
            relu->get_outputs()[0], finput->get_outputs()[0]);
    auto relu_mul = fusion2.make<relu_op_t>(mul->get_outputs()[0]);
    auto mul2 = fusion2.make<mul_op_t>(
            mul->get_outputs()[0], relu_mul->get_outputs()[0]);
    auto out = fusion2.make<output_op>(mul2->get_outputs()[0]);
    auto tsr1 = make_tensor("A", {100, 200}, datatypes::bf16);
    auto tsr2 = make_tensor("B", {100, 200}, datatypes::bf16);

    builder.push_scope();
    fusion2.create_output_fusion_anchor({tsr1}, {tsr2});
    builder.pop_scope();

    fuse_state_t fstate;
    fusion2.prepare_and_check(ctx, fstate);

    fdata_map &fdmap = fstate.fdmap_;
    fusion2.do_prepare_fusion_data(fdmap);
    auto vec = std::unordered_map<int, std::vector<int>> {{0, {0}}};
    EXPECT_EQ(relu->get_info().tensor_share_info_, vec);
    EXPECT_EQ(relu->get_outputs()[0]->details_.dtype_, datatypes::bf16);
    EXPECT_EQ(relu->logical_op_id_, 1);
    EXPECT_EQ(get_detail(relu->get_outputs()[0]).use_count_, 1);

    vec = {{0, {0}}};
    EXPECT_EQ(mul->get_info().tensor_share_info_, vec);
    EXPECT_EQ(mul->get_outputs()[0]->details_.dtype_, datatypes::bf16);
    EXPECT_EQ(mul->logical_op_id_, 2);
    EXPECT_EQ(get_detail(mul->get_outputs()[0]).use_count_, 2);

    vec = {{0, {0}}};
    EXPECT_EQ(relu_mul->get_info().tensor_share_info_, vec);
    EXPECT_EQ(relu_mul->get_outputs()[0]->details_.dtype_, datatypes::bf16);
    EXPECT_EQ(relu_mul->logical_op_id_, 3);
    EXPECT_EQ(get_detail(relu_mul->get_outputs()[0]).use_count_, 1);

    vec = {{0, {0}}};
    EXPECT_EQ(mul2->get_info().tensor_share_info_, vec);
    EXPECT_EQ(mul2->get_outputs()[0]->details_.dtype_, datatypes::bf16);
    EXPECT_EQ(mul2->logical_op_id_, 4);
    EXPECT_EQ(get_detail(mul2->get_outputs()[0]).use_count_, 0);
    std::vector<sc_op_ptr> failed_ops;
    fslice_map &fsmap = fstate.fsmap_list_[0];
    infer_status_map_t stat_map;
    fusion2.do_infer_slice_ranges(fsmap, 0, stat_map);
    fusion2.do_allocate_tensor(fdmap, {tsr2});
    EXPECT_TRUE(
            tsr1.ptr_same(fdmap.get(finput->get_outputs()[0]).get_buffer()));
    auto relu_tsr = fdmap.get(relu->get_outputs()[0]).get_buffer();
    EXPECT_FALSE(tsr1.ptr_same(relu_tsr));

    auto mul_tsr = fdmap.get(mul->get_outputs()[0]).get_buffer();
    EXPECT_FALSE(tsr1.ptr_same(mul_tsr));
    EXPECT_TRUE(relu_tsr.ptr_same(mul_tsr));

    auto relu_mul_tsr = fdmap.get(relu_mul->get_outputs()[0]).get_buffer();
    EXPECT_FALSE(relu_mul_tsr.ptr_same(mul_tsr));

    EXPECT_TRUE(tsr2.ptr_same(fdmap.get(mul2->get_outputs()[0]).get_buffer()));
    EXPECT_TRUE(tsr2.ptr_same(fdmap.get(out->get_inputs()[0]).get_buffer()));
}

class multi_inout_op_t : public fusible_op_t {
public:
    void prepare_fusion_data(fdata_map &fdmap) override {}
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override {
        // search known_ranges from any input of cur fusbile op
        slice_range_map known_ranges_map
                = search_known_slice_ranges(this, fsmap, stat_map);
        if (known_ranges_map.empty()) return;
        // set the other unknown slice range by achieved known_ranges
        set_unknown_slice_ranges(this, known_ranges_map, fsmap, stat_map);
        // set all outputs slice, no additional process is needed.
        for (auto &output : get_outputs())
            fsmap.get(output) = known_ranges_map[0];
    }
    void pre_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override {};
    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs) override {
        auto shape = dst[0]->get_shape();
        EXPECT_EQ(get_const_as_int(shape[0].checked_as<constant>()), 10);
        EXPECT_EQ(get_const_as_int(shape[1].checked_as<constant>()), 200);
    }
    multi_inout_op_t(const std::vector<graph_tensor_ptr> &in) {
        info_.inputs_ = in;
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, info_.inputs_[0]->details_));
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, info_.inputs_[0]->details_));
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, info_.inputs_[0]->details_));
        op_name_ = "multi_inout";
        auto &output0 = info_.outputs_[0], &output1 = info_.outputs_[1],
             &output2 = info_.outputs_[2];
        info_.tensor_share_info_ = {{0, {0, 1}}, {1, {0, 1}}, {2, {0, 1}}};

        output0->details_.dtype_ = info_.inputs_[0]->details_.dtype_;
        output1->details_.dtype_ = info_.inputs_[0]->details_.dtype_;
        output2->details_.dtype_ = info_.inputs_[0]->details_.dtype_;
    }
};

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerMultiInput) {
    builder::ir_builder_t builder;
    fusion_manager fusion3;
    auto finput0 = fusion3.make<input_op>(make_tsr({100, 200}, datatypes::s32));
    auto finput1 = fusion3.make<input_op>(make_tsr({100, 200}, datatypes::s32));
    auto fmul_inout = fusion3.make<multi_inout_op_t>(
            {finput0->get_outputs()[0], finput1->get_outputs()[0]});
    auto frelu = fusion3.make<relu_op_t>(fmul_inout->get_outputs()[0]);
    auto fmul = fusion3.make<mul_op_t>(
            fmul_inout->get_outputs()[1], fmul_inout->get_outputs()[2]);
    auto foutput0 = fusion3.make<output_op>(frelu->get_outputs()[0]);
    auto foutput1 = fusion3.make<output_op>(fmul->get_outputs()[0]);
    _function_(datatypes::s32, aaa, _arg_("inp0", datatypes::s32, {100, 200}),
            _arg_("inp1", datatypes::s32, {100, 200}),
            _arg_("out0", datatypes::s32, {100, 200}),
            _arg_("out1", datatypes::s32, {100, 200})) {
        _bind_(inp0, inp1, out0, out1);
        fusion3.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 200}}),
                        tensor_slice(
                                inp1, {/*dim1*/ {20, 10}, /*dim2*/ {0, 200}})},
                {tensor_slice(out0, {/*dim1*/ {30, 10}, /*dim2*/ {0, 200}}),
                        tensor_slice(
                                out1, {/*dim1*/ {40, 10}, /*dim2*/ {0, 200}})});
        _return_(123);
    }
    do_commit(fusion3, {aaa->params_[2], aaa->params_[3]}, {});
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestConcatOP) {
    REQUIRE_AVX512(); // vec lane is 16 for s32 dtype
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput0 = fusion.make<input_op>(make_tsr({100, 200, 10}));
    auto finput1 = fusion.make<input_op>(make_tsr({100, 300, 10}));
    auto finput2 = fusion.make<input_op>(make_tsr({100, 400, 10}));
    auto fconcat = fusion.make<concat_op_t>(
            {finput0->get_outputs()[0], finput1->get_outputs()[0],
                    finput2->get_outputs()[0]},
            1);
    auto fout = fusion.make<output_op>(fconcat->get_outputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("inp0", datatypes::s32, {100, 200, 10}),
            _arg_("inp1", datatypes::s32, {100, 300, 10}),
            _arg_("inp2", datatypes::s32, {100, 400, 10}),
            _arg_("out0", datatypes::s32, {100, 900, 10})) {
        _bind_(inp0, inp1, inp2, out0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0,
                         {/*dim1*/ {0, 10}, /*dim2*/ {0, 200},
                                 /*dim3*/ {0, 6}}),
                        tensor_slice(inp1,
                                {/*dim1*/ {0, 10}, /*dim2*/ {0, 300},
                                        /*dim3*/ {0, 6}}),
                        tensor_slice(inp2,
                                {/*dim1*/ {0, 10}, /*dim2*/ {0, 400},
                                        /*dim3*/ {0, 6}})},
                {tensor_slice(out0,
                        {/*dim1*/ {0, 10}, /*dim2*/ {0, 900},
                                /*dim3*/ {0, 6}})});
        _return_(123);
    }

    do_commit(fusion, {aaa->params_[3]}, {});

    ///// Expected func:
    uint32_t lanes = 16; // for avx512 and s32 dtype
    _function_(datatypes::s32, bbb,
            _arg_("inp0", datatypes::s32, {100, 200, 10}),
            _arg_("inp1", datatypes::s32, {100, 300, 10}),
            _arg_("inp2", datatypes::s32, {100, 400, 10}),
            _arg_("out0", datatypes::s32, {100, 900, 10})) {
        _bind_(inp0, inp1, inp2, out0);
        auto out0_ptr = builder::tensor_ptr(out0, {0, 0, 0}, {}, true);
        auto inp0_ptr = builder::tensor_ptr(inp0, {0, 0, 0}, {}, true);
        auto inp1_ptr = builder::tensor_ptr(inp1, {0, 0, 0}, {}, true);
        auto inp2_ptr = builder::tensor_ptr(inp2, {0, 0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj0, 0, 200) {
                _for_(hh0, 0, 6, int(lanes)) {
                    auto mask = last_dim_generate_mask(
                            hh0, 0, 6, int(lanes), true);
                    out0_ptr[span_t({ii, jj0 + expr(0), hh0}, lanes, mask)]
                            = inp0_ptr[span_t({ii, jj0, hh0}, lanes, mask)];
                }
            }
            _for_(jj1, 0, 300) {
                _for_(hh1, 0, 6, int(lanes)) {
                    auto mask = last_dim_generate_mask(
                            hh1, 0, 6, int(lanes), true);
                    out0_ptr[span_t({ii, jj1 + (expr(0) + expr(200)), hh1},
                            lanes, mask)]
                            = inp1_ptr[span_t({ii, jj1, hh1}, lanes, mask)];
                }
            }
            _for_(jj2, 0, 400) {
                _for_(hh2, 0, 6, int(lanes)) {
                    auto mask = last_dim_generate_mask(
                            hh2, 0, 6, int(lanes), true);
                    out0_ptr[span_t(
                            {ii, jj2 + (expr(0) + expr(200) + expr(300)), hh2},
                            lanes, mask)]
                            = inp2_ptr[span_t({ii, jj2, hh2}, lanes, mask)];
                }
            }
        }
        _return_(123);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestSplitOP) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput = fusion.make<input_op>(make_tsr({100, 200, 10}));
    auto fsplit = fusion.make<split_op_t>(
            finput->get_outputs()[0], 1, sc_dims {40, 60, 100});
    auto foutput0 = fusion.make<output_op>(fsplit->get_outputs()[0]);
    auto foutput1 = fusion.make<output_op>(fsplit->get_outputs()[1]);
    auto foutput2 = fusion.make<output_op>(fsplit->get_outputs()[2]);
    _function_(datatypes::s32, aaa,
            _arg_("inp", datatypes::s32, {100, 200, 10}),
            _arg_("out0", datatypes::s32, {100, 100, 10}),
            _arg_("out1", datatypes::s32, {100, 100, 10}),
            _arg_("out2", datatypes::s32, {100, 200, 20})) {
        _bind_(inp, out0, out1, out2);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp,
                        {/*dim1*/ {10, 20}, /*dim2*/ {0, 200},
                                /*dim3*/ {0, 6}})},
                {tensor_slice(out0,
                         {/*dim1*/ {40, 20}, /*dim2*/ {0, 40},
                                 /*dim3*/ {2, 6}}),
                        tensor_slice(out1,
                                {/*dim1*/ {62, 20}, /*dim2*/ {30, 60},
                                        /*dim3*/ {4, 6}}),
                        tensor_slice(out2,
                                {/*dim1*/ {81, 20}, /*dim2*/ {100, 100},
                                        /*dim3*/ {8, 6}})});

        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1], aaa->params_[2], aaa->params_[3]});

    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("inp", datatypes::s32, {100, 200, 10}),
            _arg_("out0", datatypes::s32, {100, 100, 10}),
            _arg_("out1", datatypes::s32, {100, 100, 10}),
            _arg_("out2", datatypes::s32, {100, 200, 20})) {
        _bind_(inp, out0, out1, out2);
        auto inp_ptr = builder::tensor_ptr(inp, {10, 0, 0}, {}, true);
        auto out0_ptr = builder::tensor_ptr(out0, {40, 0, 2}, {}, true);
        auto out1_ptr = builder::tensor_ptr(out1, {62, 30, 4}, {}, true);
        auto out2_ptr = builder::tensor_ptr(out2, {81, 100, 8}, {}, true);
        _for_(ii, 0, 20) {
            _for_(jj, 0, 40) {
                _for_(hh, 0, 6) {
                    out0_ptr[{ii, jj, hh}] = inp_ptr[{ii, jj + expr(0), hh}];
                }
            }
            _for_(jj, 0, 60) {
                _for_(hh, 0, 6) {
                    out1_ptr[{ii, jj, hh}]
                            = inp_ptr[{ii, jj + (expr(0) + expr(40)), hh}];
                }
            }
            _for_(jj, 0, 100) {
                _for_(hh, 0, 6) {
                    out2_ptr[{ii, jj, hh}] = inp_ptr[{
                            ii, jj + (expr(0) + expr(40) + expr(60)), hh}];
                }
            }
        }
        _return_(123);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

#ifdef __AVX512F__

TEST(GCCore_CPU_fuse_mgr_cpp, TestLeakyReluOP) {
    auto check_leaky_relu = [&](sc_data_type_t type, const int M, const int K,
                                    float alpha) {
        builder::ir_builder_t builder;
        fusion_manager fusion;
        auto fin = fusion.make<input_op>(make_tsr({M, K}, type));
        auto flkrelu
                = fusion.make<leaky_relu_op_t>(fin->get_outputs()[0], alpha);
        auto fout = fusion.make<output_op>(flkrelu->get_outputs()[0]);

        _function_(datatypes::void_t, testf, _arg_("in", type, {M, K}),
                _arg_("out", type, {M, K})) {
            _bind_(in, out);

            fusion.create_output_fusion_anchor(
                    {tensor_slice(in, {{0, M}, {0, K}})},
                    {tensor_slice(out, {{0, M}, {0, K}})});
        }
        do_commit(fusion, {testf->params_[1]});

        auto testf_mod
                = ir_module_t::from_entry_func(get_default_context(), testf);
        auto testf_ptr = jit_engine_t::make(get_test_ctx())
                                 ->get_entry_func(testf_mod, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);

        if (type == datatypes::bf16) {
            std::vector<bf16_t> cast_in_buf(M * K);
            std::vector<bf16_t> cast_out_buf(M * K);
            std::vector<bf16_t> ref_cast_out_buf(M * K);

            for (int i = 0; i < M * K; i++) {
                cast_in_buf[i] = bf16_t(in_buf[i]);
            }
            testf_ptr->call_default(cast_in_buf.data(), cast_out_buf.data());
            for (int i = 0; i < M * K; i++) {
                out_buf[i] = cast_out_buf[i];
            }
            ref_leaky_relu(
                    ref_cast_out_buf.data(), cast_in_buf.data(), M * K, alpha);
            for (int i = 0; i < M * K; i++) {
                ref_out_buf[i] = ref_cast_out_buf[i];
            }
        } else {
            testf_ptr->call_default(in_buf.data(), out_buf.data());
            ref_leaky_relu(ref_out_buf.data(), in_buf.data(), M * K, alpha);
        }

        test_utils::compare_data(out_buf, ref_out_buf, 1e-4f, 1e-4f);
    };

    check_leaky_relu(datatypes::f32, 100, 200, 0.01);
    check_leaky_relu(datatypes::f32, 100, 256, 0.01);
    check_leaky_relu(datatypes::f32, 100, 200, 0.5f);
    check_leaky_relu(datatypes::f32, 100, 256, 0.5f);

    check_leaky_relu(datatypes::bf16, 100, 200, 0.01);
    check_leaky_relu(datatypes::bf16, 100, 256, 0.01);
    check_leaky_relu(datatypes::bf16, 100, 200, 0.5f);
    check_leaky_relu(datatypes::bf16, 100, 256, 0.5f);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestTanhOP) {
    auto check_tanh = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_manager fusion;
        auto fin = fusion.make<input_op>(make_tsr({M, K}, datatypes::f32));
        auto ftanh = fusion.make<tanh_op_t>(fin->get_outputs()[0]);
        auto fout = fusion.make<output_op>(ftanh->get_outputs()[0]);

        _function_(datatypes::void_t, aaa, _arg_("in", datatypes::f32, {M, K}),
                _arg_("out", datatypes::f32, {M, K})) {
            _bind_(in, out);

            fusion.create_output_fusion_anchor(
                    {tensor_slice(in, {{0, M}, {0, K}})},
                    {tensor_slice(out, {{0, M}, {0, K}})});
        }
        do_commit(fusion, {aaa->params_[1]});

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);
        // verify the upper/under bound values
        in_buf[0] = -10.f;
        in_buf[1] = -100.f;
        in_buf[2] = -500.f;
        in_buf[3] = 10.f;
        in_buf[4] = 100.f;
        in_buf[5] = 500.f;

        fptr->call_default(in_buf.data(), out_buf.data());
        ref_tanh(ref_out_buf.data(), in_buf.data(), M * K);

        test_utils::compare_data(out_buf, ref_out_buf, 5e-3f, 1e-4f);
    };
    // scalar version
    check_tanh(100, 200);

    // vectorization version
    check_tanh(100, 256);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestErfOP) {
    auto check_erf = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_manager fusion;
        auto fin = fusion.make<input_op>(make_tsr({M, K}, datatypes::f32));
        auto ffwd = fusion.make<erf_op_t>(fin->get_outputs()[0]);
        auto fout = fusion.make<output_op>(ffwd->get_outputs()[0]);

        _function_(datatypes::void_t, aaa, _arg_("in", datatypes::f32, {M, K}),
                _arg_("out", datatypes::f32, {M, K})) {
            _bind_(in, out);

            fusion.create_output_fusion_anchor(
                    {tensor_slice(in, {{0, M}, {0, K}})},
                    {tensor_slice(out, {{0, M}, {0, K}})});
        }

        do_commit(fusion, {aaa->params_[1]});

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);
        // verify the upper/under bound values
        in_buf[0] = -10.f;
        in_buf[1] = -100.f;
        in_buf[2] = -500.f;
        in_buf[3] = 10.f;
        in_buf[4] = 100.f;
        in_buf[5] = 500.f;

        fptr->call_default(in_buf.data(), out_buf.data());
        ref_erf(ref_out_buf.data(), in_buf.data(), M * K);

        test_utils::compare_data(out_buf, ref_out_buf, 5e-3f, 1e-4f);
    };
    // scalar version
    check_erf(128, 200);

    // vectorization version
    check_erf(128, 256);
}
#endif

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerBroadcast1) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    int bc_block = 64;
    auto finput = fusion.make<input_op>(make_tsr({50, 100, 200}));
    auto finput_add
            = fusion.make<input_op>(sc_dims {1, 100, 200}, datatypes::s32);
    auto fadd = fusion.make<add_op_t>(
            finput->get_outputs()[0], finput_add->get_outputs()[0]);
    auto foutput = fusion.make<output_op>(fadd->get_outputs()[0]);
    EXPECT_EQ(fadd->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::s32, {50, 100, 200}),
            _arg_("out", datatypes::s32, {50, 100, 200}),
            _arg_("bc_args_add", datatypes::s32, {1, 100, 200})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{0, 50}, {0, bc_block}, {0, bc_block}})},
                {tensor_slice(out, {{0, 50}, {0, bc_block}, {0, bc_block}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]}, {{aaa->params_[2]}});

    ///// Expected func:
    uint32_t lanes = fadd->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::s32, {50, 100, 200}),
            _arg_("out", datatypes::s32, {50, 100, 200}),
            _arg_("bc_args_add", datatypes::s32, {1, 100, 200})) {
        _bind_(buf, out, bc_args_add);
        auto bc_args_add_tptr
                = builder::tensor_ptr(bc_args_add, {0, 0, 0}, {}, true);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0, 0}, {}, true);
        _for_(n, 0, 50) {
            _for_(i, 0, bc_block, 1) {
                _for_(j, 0, bc_block, int(lanes)) {
                    out_tptr[span_t({n, i, j}, lanes)]
                            = buf_tptr[span_t({n, i, j}, lanes)]
                            + bc_args_add_tptr[span_t({0, i, j}, lanes)];
                }
            }
        }
        _return_(123);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerBroadcast2) {
    bool is_builtin = is_builtin_test_ctx();
    builder::ir_builder_t builder;
    fusion_manager fusion;
    int bc_block = 64;
    auto finput = fusion.make<input_op>(make_tsr({50, 100, 200}));
    auto finput_add
            = fusion.make<input_op>(sc_dims {1, 100, 1}, datatypes::s32);
    auto fadd = fusion.make<add_op_t>(
            finput->get_outputs()[0], finput_add->get_outputs()[0]);
    auto foutput = fusion.make<output_op>(fadd->get_outputs()[0]);
    EXPECT_EQ(fadd->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::s32, {50, 100, 200}),
            _arg_("out", datatypes::s32, {50, 100, 200}),
            _arg_("bc_args_add", datatypes::s32, {1, 100, 1})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{0, 50}, {0, bc_block}, {0, 50}})},
                {tensor_slice(out, {{0, 50}, {0, bc_block}, {0, 50}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]}, {{aaa->params_[2]}});

    ///// Expected func:
    uint32_t lanes = fadd->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::s32, {50, 100, 200}),
            _arg_("out", datatypes::s32, {50, 100, 200}),
            _arg_("bc_args_add", datatypes::s32, {1, 100, 1})) {
        _bind_(buf, out, bc_args_add);
        auto bc_args_add_tptr
                = builder::tensor_ptr(bc_args_add, {0, 0, 0}, {}, true);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0, 0}, {}, true);
        _for_(n, 0, 50) {
            _for_(i, 0, bc_block) {
                _for_(j, 0, 48, int(lanes)) {
                    out_tptr[span_t({n, i, j}, lanes)]
                            = buf_tptr[span_t({n, i, j}, lanes)]
                            + builder::make_broadcast(
                                    bc_args_add_tptr[{0, i, 0}],
                                    static_cast<int>(lanes));
                }
                if (is_builtin) {
                    _for_(j, 48, 50, int(lanes)) {
                        auto mask = last_dim_generate_mask(
                                j, 48, 50, int(lanes), true);
                        out_tptr[span_t({n, i, j}, lanes, mask)]
                                = buf_tptr[span_t({n, i, j}, lanes, mask)]
                                + builder::make_broadcast(
                                        bc_args_add_tptr[{0, i, 0}],
                                        static_cast<int>(lanes));
                    }
                } else {
                    _for_(j, 48, 50, 1) {
                        out_tptr[span_t({n, i, j}, 1)]
                                = buf_tptr[span_t({n, i, j}, 1)]
                                + bc_args_add_tptr[{0, i, 0}];
                    }
                }
            }
        }
        _return_(123);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerBroadcast3) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    int bc_block = 64;
    auto finput = fusion.make<input_op>(make_tsr({50, 100, 200}));
    auto finput_add = fusion.make<input_op>(sc_dims {100}, datatypes::s32);
    auto fadd = fusion.make<add_op_t>(
            std::vector<graph_tensor_ptr> {
                    finput->get_outputs()[0], finput_add->get_outputs()[0]},
            std::vector<graph_tensor_ptr> {},
            any_map_t {{"bc_axis", std::vector<int> {1}}});
    auto foutput = fusion.make<output_op>(fadd->get_outputs()[0]);
    EXPECT_EQ(fadd->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::s32, {50, 100, 200}),
            _arg_("out", datatypes::s32, {50, 100, 200}),
            _arg_("bc_args_add", datatypes::s32, {100})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{0, 50}, {0, bc_block}, {0, bc_block}})},
                {tensor_slice(out, {{0, 50}, {0, bc_block}, {0, bc_block}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]}, {{aaa->params_[2]}});

    ///// Expected func:
    uint32_t lanes = fadd->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::s32, {50, 100, 200}),
            _arg_("out", datatypes::s32, {50, 100, 200}),
            _arg_("bc_args_add", datatypes::s32, {100})) {
        _bind_(buf, out, bc_args_add);
        auto bc_args_add_tptr = builder::tensor_ptr(bc_args_add, {0}, {}, true);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0, 0}, {}, true);
        _for_(n, 0, 50) {
            _for_(i, 0, bc_block, 1) {
                _for_(j, 0, bc_block, int(lanes)) {
                    out_tptr[span_t({n, i, j}, lanes)]
                            = buf_tptr[span_t({n, i, j}, lanes)]
                            + builder::make_broadcast(bc_args_add_tptr[i],
                                    static_cast<int>(lanes));
                }
            }
        }
        _return_(123);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerBroadcast4) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    int bc_block = 64;
    auto finput = fusion.make<input_op>(make_tsr({100}));
    auto finput_add = fusion.make<input_op>(make_tsr({50, 100, 200}));
    auto fadd = fusion.make<sub_op_t>(
            std::vector<graph_tensor_ptr> {
                    finput->get_outputs()[0], finput_add->get_outputs()[0]},
            std::vector<graph_tensor_ptr> {},
            any_map_t {{"bc_axis", std::vector<int> {1}}});
    auto foutput = fusion.make<output_op>(fadd->get_outputs()[0]);
    EXPECT_EQ(fadd->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::s32, {50, 100, 200}),
            _arg_("out", datatypes::s32, {50, 100, 200}),
            _arg_("bc_args_add", datatypes::s32, {100})) {
        _bind_(buf, out, bc_arg);
        fusion.create_output_fusion_anchor(
                {tensor_slice(bc_arg, {{0, bc_block}}),
                        tensor_slice(
                                buf, {{0, 50}, {0, bc_block}, {0, bc_block}})},
                {tensor_slice(out, {{0, 50}, {0, bc_block}, {0, bc_block}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    ///// Expected func:
    uint32_t lanes = fadd->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::s32, {50, 100, 200}),
            _arg_("out", datatypes::s32, {50, 100, 200}),
            _arg_("bc_args_add", datatypes::s32, {100})) {
        _bind_(buf, out, bc_args_add);
        auto bc_args_add_tptr = builder::tensor_ptr(bc_args_add, {0}, {}, true);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0, 0}, {}, true);
        _for_(n, 0, 50) {
            _for_(i, 0, bc_block, 1) {
                _for_(j, 0, bc_block, int(lanes)) {
                    out_tptr[span_t({n, i, j}, lanes)]
                            = builder::make_broadcast(bc_args_add_tptr[i],
                                      static_cast<int>(lanes))
                            - buf_tptr[span_t({n, i, j}, lanes)];
                }
            }
        }
        _return_(123);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerVectorizedReLU) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput = fusion.make<input_op>(make_tsr({100, 256}));
    auto frelu = fusion.make<relu_op_t>(finput->get_outputs()[0]);
    auto foutput = fusion.make<output_op>(frelu->get_outputs()[0]);
    EXPECT_EQ(frelu->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa, _arg_("buf", datatypes::s32, {100, 256}),
            _arg_("out", datatypes::s32, {100, 256})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{0, 100}, {0, 256}})},
                {tensor_slice(out, {{0, 100}, {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    ///// Expected func:
    uint32_t lanes = frelu->get_vx_info().lanes;
    _function_(datatypes::s32, bbb, _arg_("buf", datatypes::s32, {100, 256}),
            _arg_("out", datatypes::s32, {100, 256})) {
        _bind_(buf, out);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0}, {}, true);
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 256, int(lanes)) {
                out_tptr[span_t({i, j}, lanes)]
                        = builder::make_max(buf_tptr[span_t({i, j}, lanes)],
                                make_expr<constant_node>((int64_t)0,
                                        sc_data_type_t::s32(lanes)));
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerExp) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput = fusion.make<input_op>(make_tsr({100, 256}));
    auto fexp = fusion.make<exp_op_t>(finput->get_outputs()[0]);
    auto foutput = fusion.make<output_op>(fexp->get_outputs()[0]);
    EXPECT_EQ(fexp->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::f32, aaa, _arg_("buf", datatypes::f32, {100, 256}),
            _arg_("out", datatypes::f32, {100, 256})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{0, 100}, {0, 256}})},
                {tensor_slice(out, {{0, 100}, {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    uint32_t lanes = fexp->get_vx_info().lanes;
    _function_(datatypes::f32, bbb, _arg_("buf", datatypes::f32, {100, 256}),
            _arg_("out", datatypes::f32, {100, 256})) {
        _bind_(buf, out);
        auto buf_tptr = builder::tensor_ptr(buf, {0, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {0, 0}, {}, true);
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 256, int(lanes)) {
                out_tptr[span_t({i, j}, lanes)]
                        = builder::make_exp(buf_tptr[span_t({i, j}, lanes)]);
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestSigmoidOP) {
    REQUIRE_AVX2();
    auto check_sigmoid = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_manager fusion;
        auto fin = fusion.make<input_op>(make_tsr({M, K}));
        auto fsig = fusion.make<sigmoid_op_t>(fin->get_outputs()[0]);
        auto fout = fusion.make<output_op>(fsig->get_outputs()[0]);

        _function_(datatypes::void_t, aaa, _arg_("in", datatypes::f32, {M, K}),
                _arg_("out", datatypes::f32, {M, K})) {
            _bind_(in, out);

            fusion.create_output_fusion_anchor(
                    {tensor_slice(in, {{0, M}, {0, K}})},
                    {tensor_slice(out, {{0, M}, {0, K}})});
        }
        do_commit(fusion, {aaa->params_[1]});

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);
        // verify the upper/under bound values
        in_buf[0] = -10.f;
        in_buf[1] = -100.f;
        in_buf[2] = -std::numeric_limits<float>::infinity();
        in_buf[3] = 10.f;
        in_buf[4] = 100.f;
        in_buf[5] = std::numeric_limits<float>::infinity();

        in_buf[6] = std::numeric_limits<float>::min();
        in_buf[7] = std::numeric_limits<float>::max();

        in_buf[8] = -std::numeric_limits<float>::min();
        in_buf[9] = -std::numeric_limits<float>::max();

        fptr->call_default(in_buf.data(), out_buf.data());

        ref_sigmoid(ref_out_buf.data(), in_buf.data(), M * K);

        test_utils::compare_data(out_buf, ref_out_buf, 1e-5f, 1e-5f);
    };
    // scalar version
    check_sigmoid(100, 200);

    // vectorization version
    check_sigmoid(100, 256);
}

// #define BENCH_SIGMOID

#ifdef BENCH_SIGMOID
TEST(GCCore_CPU_fuse_mgr_cpp, BenchVectorizedSigmoidOP) {
    auto bench_sigmoid = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_manager fusion;
        auto fin = fusion.make<input_op>();
        auto fsig = fusion.make<sigmoid_op_t>(fin->get_outputs()[0], true);
        auto fout = fusion.make<output_op>(fsig->get_outputs()[0]);

        _function_(datatypes::void_t, aaa, _arg_("in", datatypes::f32, {M, K}),
                _arg_("out", datatypes::f32, {M, K})) {
            _bind_(in, out);

            fusion.create_output_fusion_anchor(
                    {tensor_slice(in, {{0, M}, {0, K}})},
                    {tensor_slice(out, {{0, M}, {0, K}})});
        }
        fusion.commit(ctx, aaa);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);

        auto start = high_resolution_clock::now();

        int count = 100;
        for (int i = 0; i < count; i++) {
            fptr->call_default(in_buf.data(), out_buf.data());
        }

        auto stop = high_resolution_clock::now();

        auto duration
                = duration_cast<microseconds>(stop - start) / (1000.f * count);
        std::cout << "Data shape: " << M << "x" << K << ". "
                  << "Average time taken by op: " << duration.count()
                  << " milliseconds\n";
    };
    // vectorization version
    bench_sigmoid(100, 256);

    // vectorization version
    bench_sigmoid(100, 2560);

    // vectorization version
    bench_sigmoid(100, 25600);

    // vectorization version
    bench_sigmoid(100, 512000);
}
#endif

//#define BENCH_ROUND

#ifdef BENCH_ROUND
TEST(GCCore_CPU_fuse_mgr_cpp, BenchVectorizedRoundOP) {
    auto bench_round = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_manager fusion;
        auto fin = fusion.make<input_op>();
        auto fround = fusion.make<round_op_t>(fin->get_outputs()[0]);
        auto fout = fusion.make<output_op>(fround->get_outputs()[0]);

        _function_(datatypes::void_t, aaa, _arg_("in", datatypes::f32, {M, K}),
                _arg_("out", datatypes::f32, {M, K})) {
            _bind_(in, out);

            fusion.create_output_fusion_anchor(
                    {tensor_slice(in, {{0, M}, {0, K}})},
                    {tensor_slice(out, {{0, M}, {0, K}})});
        }
        fusion.commit(ctx, aaa);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);

        auto start = high_resolution_clock::now();

        int count = 100;
        for (int i = 0; i < count; i++) {
            fptr->call_default(in_buf.data(), out_buf.data());
        }

        auto stop = high_resolution_clock::now();

        auto duration
                = duration_cast<microseconds>(stop - start) / (1000.f * count);
        std::cout << "Data shape: " << M << "x" << K << ". "
                  << "Average time taken by op: " << duration.count()
                  << " milliseconds\n";
    };
    // vectorization version
    bench_round(100, 256);

    // vectorization version
    bench_round(100, 2560);

    // vectorization version
    bench_round(100, 25600);

    // vectorization version
    bench_round(100, 512000);
}
#endif

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerReduceSum1) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 100, 256}, datatypes::f32));
    auto fsum = fusion.make<reduce_sum_op_t>(
            finput->get_outputs()[0], std::vector<int> {2}, false);
    auto foutput = fusion.make<output_op>(fsum->get_outputs()[0]);
    EXPECT_EQ(fsum->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::f32, {20, 100, 256}),
            _arg_("out", datatypes::f32, {20, 100})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{10, 1}, {30, 1}, {0, 256}})},
                {tensor_slice(out, {{10, 1}, {30, 1}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    ///// Expected func:
    uint32_t lanes = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::f32, {20, 100, 256}),
            _arg_("out", datatypes::f32, {20, 100})) {
        _bind_(buf, out);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 1) {
                _var_(reduce_value, sc_data_type_t::f32(lanes));
                reduce_value = make_expr<constant_node>(
                        0.0f, sc_data_type_t::f32(lanes));
                _for_(k, 0, 256, static_cast<int>(lanes)) {
                    reduce_value = builder::make_add(
                            buf_tptr[span_t({i, j, k}, lanes)], reduce_value);
                }
                out_tptr[{i, j}] = builder::make_reduce_add(reduce_value);
            }
        }
        _return_(123);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerReduceSum2) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 100, 256}, datatypes::f32));
    auto fsum = fusion.make<reduce_sum_op_t>(
            finput->get_outputs()[0], std::vector<int> {1}, true);
    auto foutput = fusion.make<output_op>(fsum->get_outputs()[0]);
    EXPECT_EQ(fsum->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::f32, {20, 100, 256}),
            _arg_("out", datatypes::f32, {20, 1, 256})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{10, 1}, {0, 100}, {20, 1}})},
                {tensor_slice(out, {{10, 1}, {0, 1}, {20, 1}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::f32, {20, 100, 256}),
            _arg_("out", datatypes::f32, {20, 1, 256})) {
        _bind_(buf, out);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 0, 20}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 0, 20}, {}, true);
        _for_(i, 0, 1) {
            _for_(k, 0, 1) {
                _var_(reduce_sum, sc_data_type_t::f32());
                reduce_sum = 0.f;
                _for_(j, 0, 100) {
                    reduce_sum = builder::make_add(
                            buf_tptr[{i, j, k}], reduce_sum);
                }
                out_tptr[{i, 0, k}] = reduce_sum;
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerReduceProd) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 100, 256}, datatypes::f32));
    auto fprod = fusion.make<reduce_prod_op_t>(
            finput->get_outputs()[0], std::vector<int> {2}, false);
    auto foutput = fusion.make<output_op>(fprod->get_outputs()[0]);
    EXPECT_EQ(fprod->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::f32, {20, 100, 256}),
            _arg_("out", datatypes::f32, {20, 100})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{10, 1}, {30, 1}, {0, 256}})},
                {tensor_slice(out, {{10, 1}, {30, 1}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    uint32_t lanes = fprod->get_lanes();
    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::f32, {20, 100, 256}),
            _arg_("out", datatypes::f32, {20, 100})) {
        _bind_(buf, out);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 1) {
                _var_(reduce_mul, sc_data_type_t::f32(lanes));
                reduce_mul = make_expr<constant_node>(
                        1.0f, sc_data_type_t::f32(lanes));
                _for_(k, 0, 256, static_cast<int>(lanes)) {
                    reduce_mul = builder::make_mul(
                            buf_tptr[span_t({i, j, k}, lanes)], reduce_mul);
                }
                out_tptr[{i, j}] = builder::make_reduce_mul(reduce_mul);
            }
        }
        _return_(123);
    }
    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerSquaredDiff1) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 100, 256}, datatypes::f32));
    auto fsum = fusion.make<reduce_sum_op_t>(
            finput->get_outputs()[0], std::vector<int> {2}, true);
    auto fsquare_diff = fusion.make<squared_diff_op_t>(
            finput->get_outputs()[0], fsum->get_outputs()[0], true, true);
    auto foutput = fusion.make<output_op>(fsquare_diff->get_outputs()[0]);
    EXPECT_EQ(fsquare_diff->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::f32, {20, 100, 256}),
            _arg_("out", datatypes::f32, {20, 100, 256})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{10, 1}, {30, 1}, {0, 256}})},
                {tensor_slice(out, {{10, 1}, {30, 1}, {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    ///// Expected func:
    uint32_t rd_lanes
            = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    uint32_t sd_lanes = fsquare_diff->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::f32, {20, 100, 256}),
            _arg_("out", datatypes::f32, {20, 100, 256})) {
        _bind_(buf, out);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30, 0}, {}, true);
        _tensor_(_fuse_buf_0, datatypes::f32, {20UL, 100UL, 1UL});
        auto fuse_buf_tptr
                = builder::tensor_ptr(_fuse_buf_0, {10, 30, 0}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 1) {
                _var_(reduce_value, sc_data_type_t::f32(rd_lanes));
                reduce_value = make_expr<constant_node>(
                        0.0f, sc_data_type_t::f32(rd_lanes));
                _for_(k, 0, 256, static_cast<int>(rd_lanes)) {
                    reduce_value = builder::make_add(
                            buf_tptr[span_t({i, j, k}, rd_lanes)],
                            reduce_value);
                }
                fuse_buf_tptr[{i, j, 0}]
                        = builder::make_reduce_add(reduce_value);
            }
        }

        _for_(i, 0, 1) {
            _for_(j, 0, 1) {
                _for_(k, 0, UINT64_C(256), static_cast<int>(sd_lanes)) {
                    out_tptr[span_t({i, j, k}, sd_lanes)]
                            = (buf_tptr[span_t({i, j, k}, sd_lanes)]
                                      - builder::make_broadcast(
                                              builder::tensor_ptr(_fuse_buf_0,
                                                      {10, 30, 0}, {},
                                                      true)[{i, j, 0}],
                                              static_cast<int>(sd_lanes)))
                            * (buf_tptr[span_t({i, j, k}, sd_lanes)]
                                    - builder::make_broadcast(
                                            builder::tensor_ptr(_fuse_buf_0,
                                                    {10, 30, 0}, {},
                                                    true)[{i, j, 0}],
                                            static_cast<int>(sd_lanes)));
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerSquaredDiff2) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 100, 256}, datatypes::f32));
    auto fmean
            = fusion.make<input_op>(make_tsr({20, 100, 256}, datatypes::f32));
    auto fsquare_diff = fusion.make<squared_diff_op_t>(
            finput->get_outputs()[0], fmean->get_outputs()[0], true, true);
    auto foutput = fusion.make<output_op>(fsquare_diff->get_outputs()[0]);
    EXPECT_EQ(fsquare_diff->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::s32, {20, 100, 256}),
            _arg_("sqd_mean", datatypes::s32, {20, 100, 256}),
            _arg_("out", datatypes::s32, {20, 100, 256})) {
        _bind_(buf, sqd_mean, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{10, 1}, {30, 50}, {0, 256}}),
                        tensor_slice(sqd_mean, {{10, 1}, {30, 50}, {0, 256}})},
                {tensor_slice(out, {{10, 1}, {30, 50}, {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[2]});

    ///// Expected func:
    uint32_t lanes = fsquare_diff->get_lanes();
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::s32, {20, 100, 256}),
            _arg_("sqd_mean", datatypes::s32, {20, 100, 256}),
            _arg_("out", datatypes::s32, {20, 100, 256})) {
        _bind_(buf, sqd_mean, out);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30, 0}, {}, true);
        auto sqd_mean_tptr
                = builder::tensor_ptr(sqd_mean, {10, 30, 0}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 50) {
                _for_(k, 0, 256, static_cast<int>(lanes)) {
                    out_tptr[span_t({i, j, k}, lanes)]
                            = (buf_tptr[span_t({i, j, k}, lanes)]
                                      - sqd_mean_tptr[span_t({i, j, k}, lanes)])
                            * (buf_tptr[span_t({i, j, k}, lanes)]
                                    - sqd_mean_tptr[span_t({i, j, k}, lanes)]);
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerSquaredRoot1) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 100, 256}, datatypes::f32));
    auto fsquare_root
            = fusion.make<squared_root_op_t>(finput->get_outputs()[0]);
    auto foutput = fusion.make<output_op>(fsquare_root->get_outputs()[0]);
    EXPECT_EQ(fsquare_root->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::s32, {20, 100, 256}),
            _arg_("out", datatypes::s32, {20, 100, 256})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{10, 1}, {30, 50}, {0, 256}})},
                {tensor_slice(out, {{10, 1}, {30, 50}, {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    ///// Expected func:
    uint32_t lanes = fsquare_root->get_vx_info().lanes;
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::s32, {20, 100, 256}),
            _arg_("out", datatypes::s32, {20, 100, 256})) {
        _bind_(buf, out);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30, 0}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 50) {
                _for_(k, 0, 256, static_cast<int>(lanes)) {
                    out_tptr[span_t({i, j, k}, lanes)]
                            = make_sqrt(buf_tptr[span_t({i, j, k}, lanes)]);
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerSquaredRoot2) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 100, 256}, datatypes::f32));
    auto fsquare_root
            = fusion.make<squared_root_op_t>(finput->get_outputs()[0], true);
    auto foutput = fusion.make<output_op>(fsquare_root->get_outputs()[0]);
    EXPECT_EQ(fsquare_root->get_outputs()[0], foutput->get_inputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("buf", datatypes::s32, {20, 100, 256}),
            _arg_("out", datatypes::s32, {20, 100, 256})) {
        _bind_(buf, out);
        fusion.create_output_fusion_anchor(
                {tensor_slice(buf, {{10, 1}, {30, 50}, {0, 256}})},
                {tensor_slice(out, {{10, 1}, {30, 50}, {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    ///// Expected func:
    uint32_t lanes = fsquare_root->get_vx_info().lanes;
    _function_(datatypes::s32, bbb,
            _arg_("buf", datatypes::s32, {20, 100, 256}),
            _arg_("out", datatypes::s32, {20, 100, 256})) {
        _bind_(buf, out);
        auto buf_tptr = builder::tensor_ptr(buf, {10, 30, 0}, {}, true);
        auto out_tptr = builder::tensor_ptr(out, {10, 30, 0}, {}, true);
        _for_(i, 0, 1) {
            _for_(j, 0, 50) {
                _for_(k, 0, 256, static_cast<int>(lanes)) {
                    out_tptr[span_t({i, j, k}, lanes)]
                            = make_rsqrt(buf_tptr[span_t({i, j, k}, lanes)]);
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerCastBF16) {
    REQUIRE_AVX512();
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finputf32 = fusion.make<input_op>(make_tsr({20, 259}, datatypes::f32));
    auto finputbf16
            = fusion.make<input_op>(make_tsr({20, 259}, datatypes::bf16));

    auto fcastf32tobf16 = fusion.make<cast_op_t>(
            finputf32->get_outputs()[0], datatypes::bf16);
    auto fcastbf16tof32 = fusion.make<cast_op_t>(
            finputbf16->get_outputs()[0], datatypes::f32);

    auto foutf32tobf16
            = fusion.make<output_op>(fcastf32tobf16->get_outputs()[0]);
    auto foutbf16tof32
            = fusion.make<output_op>(fcastbf16tof32->get_outputs()[0]);

    EXPECT_EQ(foutf32tobf16->get_inputs()[0], fcastf32tobf16->get_outputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("outf32tobf16", datatypes::bf16, {20, 259}),
            _arg_("outbf16tof32", datatypes::f32, {20, 259}),
            _arg_("inf32", datatypes::f32, {20, 259}),
            _arg_("inbf16", datatypes::bf16, {20, 259}), ) {
        _bind_(outf32tobf16, outbf16tof32, inf32, inbf16);
        fusion.create_output_fusion_anchor(
                {
                        tensor_slice(inf32, {{0, 20}, {0, 259}}),
                        tensor_slice(inbf16, {{0, 20}, {0, 259}}),
                },
                {
                        tensor_slice(outf32tobf16, {{0, 20}, {0, 259}}),
                        tensor_slice(outbf16tof32, {{0, 20}, {0, 259}}),
                });
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[0], aaa->params_[1]});

    std::vector<float> inf32(20 * 259), outbf16tof32(20 * 259);
    std::vector<bf16_t> inbf16(20 * 259), outf32tobf16(20 * 259);
    test_utils::fill_data<float>(inf32.data(), 20 * 259);
    for (int i = 0; i < 20 * 259; i++) {
        inbf16[i] = inf32[i];
    }

    auto fptr = jit_engine_t::make(get_test_ctx())
                        ->get_entry_func(ir_module_t::from_entry_func(
                                                 get_default_context(), aaa),
                                true);
    fptr->call_default(outf32tobf16.data(), outbf16tof32.data(), inf32.data(),
            inbf16.data());

    for (unsigned i = 0; i < 20 * 259; i++) {
        if (std::abs(outf32tobf16[i] - inbf16[i]) >= 1e-5f) {
            std::cout << outf32tobf16[i] << "\n";
        }
        EXPECT_TRUE(
                std::abs(outf32tobf16[i] - float(bf16_t(inf32[i]))) < 1e-5f);
        EXPECT_TRUE(std::abs(outbf16tof32[i] - inbf16[i]) < 1e-5f);
    }
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerCastU8S8) {
    BUILTIN_REQUIRE_AVX512(); // AVX2 no cast instruction
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finputf32tou8
            = fusion.make<input_op>(make_tsr({20, 259}, datatypes::f32));
    auto finputf32tos8
            = fusion.make<input_op>(make_tsr({20, 259}, datatypes::f32));
    auto finputs32tou8 = fusion.make<input_op>(make_tsr({20, 259}));
    auto finputs32tos8 = fusion.make<input_op>(make_tsr({20, 259}));
    auto finputu8 = fusion.make<input_op>(make_tsr({20, 259}, datatypes::u8));
    auto finputs8 = fusion.make<input_op>(make_tsr({20, 259}, datatypes::s8));

    auto fcastf32tos32 = fusion.make<cast_op_t>(
            finputf32tos8->get_outputs()[0], datatypes::s32);
    auto fcastf32tou8 = fusion.make<cast_op_t>(
            finputf32tou8->get_outputs()[0], datatypes::u8);
    auto fcastf32tos8 = fusion.make<cast_op_t>(
            finputf32tos8->get_outputs()[0], datatypes::s8);

    auto fcasts32tof32 = fusion.make<cast_op_t>(
            finputs32tos8->get_outputs()[0], datatypes::f32);
    auto fcasts32tou8 = fusion.make<cast_op_t>(
            finputs32tou8->get_outputs()[0], datatypes::u8);
    auto fcasts32tos8 = fusion.make<cast_op_t>(
            finputs32tos8->get_outputs()[0], datatypes::s8);

    auto fcastu8tos32 = fusion.make<cast_op_t>(
            finputu8->get_outputs()[0], datatypes::s32);
    auto fcastu8tof32 = fusion.make<cast_op_t>(
            finputu8->get_outputs()[0], datatypes::f32);

    auto fcasts8tos32 = fusion.make<cast_op_t>(
            finputs8->get_outputs()[0], datatypes::s32);
    auto fcasts8tof32 = fusion.make<cast_op_t>(
            finputs8->get_outputs()[0], datatypes::f32);

    auto foutf32tos32 = fusion.make<output_op>(fcastf32tos32->get_outputs()[0]);
    auto foutf32tou8 = fusion.make<output_op>(fcastf32tou8->get_outputs()[0]);
    auto foutf32tos8 = fusion.make<output_op>(fcastf32tos8->get_outputs()[0]);

    auto fouts32tof32 = fusion.make<output_op>(fcasts32tof32->get_outputs()[0]);
    auto fouts32tou8 = fusion.make<output_op>(fcasts32tou8->get_outputs()[0]);
    auto fouts32tos8 = fusion.make<output_op>(fcasts32tos8->get_outputs()[0]);

    auto foutu8tof32 = fusion.make<output_op>(fcastu8tof32->get_outputs()[0]);
    auto foutu8tos32 = fusion.make<output_op>(fcastu8tos32->get_outputs()[0]);

    auto fouts8tof32 = fusion.make<output_op>(fcasts8tof32->get_outputs()[0]);
    auto fouts8tos32 = fusion.make<output_op>(fcasts8tos32->get_outputs()[0]);

    EXPECT_EQ(fouts8tof32->get_inputs()[0], fcasts8tof32->get_outputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("outf32tos32", datatypes::s32, {20, 259}),
            _arg_("outf32tou8", datatypes::u8, {20, 259}),
            _arg_("outf32tos8", datatypes::s8, {20, 259}),
            _arg_("outs32tof32", datatypes::f32, {20, 259}),
            _arg_("outs32tou8", datatypes::u8, {20, 259}),
            _arg_("outs32tos8", datatypes::s8, {20, 259}),
            _arg_("outu8tof32", datatypes::f32, {20, 259}),
            _arg_("outu8tos32", datatypes::s32, {20, 259}),
            _arg_("outs8tof32", datatypes::f32, {20, 259}),
            _arg_("outs8tos32", datatypes::s32, {20, 259}),
            _arg_("inf32tou8", datatypes::f32, {20, 259}),
            _arg_("inf32tos8", datatypes::f32, {20, 259}),
            _arg_("ins32tou8", datatypes::s32, {20, 259}),
            _arg_("ins32tos8", datatypes::s32, {20, 259}),
            _arg_("inu8", datatypes::u8, {20, 259}),
            _arg_("ins8", datatypes::s8, {20, 259}), ) {
        _bind_(outf32tos32, outf32tou8, outf32tos8, outs32tof32, outs32tou8,
                outs32tos8, outu8tof32, outu8tos32, outs8tof32, outs8tos32,
                inf32tou8, inf32tos8, ins32tou8, ins32tos8, inu8, ins8);
        fusion.create_output_fusion_anchor(
                {
                        tensor_slice(inf32tou8, {{0, 20}, {0, 259}}),
                        tensor_slice(inf32tos8, {{0, 20}, {0, 259}}),
                        tensor_slice(ins32tou8, {{0, 20}, {0, 259}}),
                        tensor_slice(ins32tos8, {{0, 20}, {0, 259}}),
                        tensor_slice(inu8, {{0, 20}, {0, 259}}),
                        tensor_slice(ins8, {{0, 20}, {0, 259}}),
                },
                {
                        tensor_slice(outf32tos32, {{0, 20}, {0, 259}}),
                        tensor_slice(outf32tou8, {{0, 20}, {0, 259}}),
                        tensor_slice(outf32tos8, {{0, 20}, {0, 259}}),
                        tensor_slice(outs32tof32, {{0, 20}, {0, 259}}),
                        tensor_slice(outs32tou8, {{0, 20}, {0, 259}}),
                        tensor_slice(outs32tos8, {{0, 20}, {0, 259}}),
                        tensor_slice(outu8tof32, {{0, 20}, {0, 259}}),
                        tensor_slice(outu8tos32, {{0, 20}, {0, 259}}),
                        tensor_slice(outs8tof32, {{0, 20}, {0, 259}}),
                        tensor_slice(outs8tos32, {{0, 20}, {0, 259}}),
                });
        _return_(123);
    }
    std::vector<expr> outs;
    outs.reserve(10);
    for (int i = 0; i < 10; i++) {
        outs.emplace_back(aaa->params_[i]);
    }
    do_commit(fusion, outs);

    std::vector<float> outs32tof32(20 * 259), outu8tof32(20 * 259),
            outs8tof32(20 * 259);
    std::vector<int32_t> outf32tos32(20 * 259, 1), outu8tos32(20 * 259),
            outs8tos32(20 * 259);
    std::vector<uint8_t> inu8(20 * 259), outf32tou8(20 * 259),
            outs32tou8(20 * 259);
    std::vector<int8_t> ins8(20 * 259), outf32tos8(20 * 259),
            outs32tos8(20 * 259);
    test_utils::fill_data<uint8_t>(inu8.data(), 20 * 259);
    test_utils::fill_data<int8_t>(ins8.data(), 20 * 259);
    std::vector<float> inf32tou8(inu8.begin(), inu8.end()),
            inf32tos8(ins8.begin(), ins8.end());
    std::vector<int32_t> ins32tou8(inu8.begin(), inu8.end()),
            ins32tos8(ins8.begin(), ins8.end());

    auto fptr = jit_engine_t::make(get_test_ctx())
                        ->get_entry_func(ir_module_t::from_entry_func(
                                                 get_default_context(), aaa),
                                true);
    fptr->call_default(outf32tos32.data(), outf32tou8.data(), outf32tos8.data(),
            outs32tof32.data(), outs32tou8.data(), outs32tos8.data(),
            outu8tof32.data(), outu8tos32.data(), outs8tof32.data(),
            outs8tos32.data(), inf32tou8.data(), inf32tos8.data(),
            ins32tou8.data(), ins32tos8.data(), inu8.data(), ins8.data());
    for (unsigned i = 0; i < 20 * 259; i++) {
        EXPECT_TRUE(outf32tos32[i] == static_cast<int>(ins8[i]));
        EXPECT_TRUE(outf32tou8[i] == static_cast<int>(inu8[i]));
        EXPECT_TRUE(outf32tos8[i] == static_cast<int>(ins8[i]));
        EXPECT_TRUE(
                static_cast<int>(outs32tof32[i]) == static_cast<int>(ins8[i]));
        EXPECT_TRUE(outs32tou8[i] == static_cast<int>(inu8[i]));
        EXPECT_TRUE(outs32tos8[i] == static_cast<int>(ins8[i]));
        EXPECT_TRUE(
                static_cast<int>(outu8tof32[i]) == static_cast<int>(inu8[i]));
        EXPECT_TRUE(outu8tos32[i] == static_cast<int>(inu8[i]));
        EXPECT_TRUE(
                static_cast<int>(outs8tof32[i]) == static_cast<int>(ins8[i]));
        EXPECT_TRUE(outs8tos32[i] == static_cast<int>(ins8[i]));
    }
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerMultiAnchor) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 30, 40, 64}, datatypes::f32));
    auto fexp = fusion.make<exp_op_t>(finput->get_outputs()[0]);
    auto fsum = fusion.make<reduce_sum_op_t>(
            fexp->get_outputs()[0], std::vector<int> {1, 3}, true);
    auto fadd = fusion.make<add_op_t>(
            finput->get_outputs()[0], fsum->get_outputs()[0], false);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 64}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 64})) {
        _bind_(inp0, out0);
        _for_(m_o, 0, 20) {
            _for_(n_o, 0, 30) {
                // Anchor 0
                fusion.create_output_fusion_anchor(
                        {tensor_slice(
                                inp0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 64}})},
                        {tensor_slice(
                                out0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 64}})});
            }
            // Anchor 1
            fusion.create_output_fusion_anchor(
                    {tensor_slice(inp0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 64}})},
                    {tensor_slice(
                            out0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 64}})});
        }
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    ///// Expected func:
    uint32_t lanes = ctx->get_max_vector_lanes(sc_data_etype::F32);
    _function_(datatypes::s32, bbb,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 64}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 64})) {
        _bind_(inp0, out0);
        _for_(m_o, 0, 20) {
            _tensor_(_fuse_buf_0, datatypes::f32, {20UL, 30UL, 40UL, 64UL});
            _tensor_(_fuse_buf_1, datatypes::f32, {20UL, 1UL, 40UL, 1UL});
            _for_(n_o, 0, 30) {
                // fuse exp
                _for_(i, 0, 1) {
                    _for_(j, 0, 1) {
                        _for_(ii, 0, 40) {
                            _for_(jj, 0, 64, static_cast<int>(lanes)) {
                                builder::tensor_ptr(_fuse_buf_0,
                                        {m_o, n_o, 0, 0}, {},
                                        true)[span_t({i, j, ii, jj}, lanes)]
                                        = make_exp(tensor_ptr(inp0,
                                                {m_o, n_o, 0, 0}, {},
                                                true)[span_t(
                                                {i, j, ii, jj}, lanes)]);
                            }
                        }
                    }
                }
            }
            // fuse reduce
            _for_(i, 0, 1) {
                _for_(ii, 0, 40) {
                    _var_(reduce_sum, sc_data_type_t::f32(lanes));
                    reduce_sum = make_expr<constant_node>(
                            0.0f, sc_data_type_t::f32(lanes));
                    _for_(j, 0, 30) {
                        _for_(jj, 0, 64, static_cast<int>(lanes)) {
                            reduce_sum = builder::make_add(
                                    builder::tensor_ptr(_fuse_buf_0,
                                            {m_o, 0, 0, 0}, {}, true)[span_t(
                                            {i, j, ii, jj}, lanes)],
                                    reduce_sum);
                        }
                    }
                    builder::tensor_ptr(_fuse_buf_1, {m_o, 0, 0, 0}, {},
                            true)[{i, 0, ii, 0}]
                            = builder::make_reduce_add(reduce_sum);
                }
            }
            // fuse add
            _for_(i, 0, 1) {
                _for_(j, 0, 30) {
                    _for_(ii, 0, 40) {
                        _for_(jj, 0, 64, static_cast<int>(lanes)) {
                            builder::tensor_ptr(out0, {m_o, 0, 0, 0}, {},
                                    true)[span_t({i, j, ii, jj}, lanes)]
                                    = builder::tensor_ptr(inp0, {m_o, 0, 0, 0},
                                              {}, true)[span_t(
                                              {i, j, ii, jj}, lanes)]
                                    + builder::make_broadcast(
                                            builder::tensor_ptr(_fuse_buf_1,
                                                    {m_o, 0, 0, 0}, {},
                                                    true)[{i, 0, ii, 0}],
                                            static_cast<int>(lanes));
                        }
                    }
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerMultiAnchorShrink) {
    sc_graph_t mgr;
    SET_THREADS_OR_SKIP(56);

    // gemm + exp + reduce fusion pattern
    auto x = graph_tensor::make({8192, 256});
    auto w = graph_tensor::make({256, 128});
    auto input = mgr.make_input({x, w});
    input->attrs_.set("constant", const_kind::local_const);

    const ops::matmul_core_config_t gemm_cfg = {32, 32, 32};
    auto gemm = mgr.make("matmul_core", input->get_outputs(), {}, {});
    gemm->stc_cast<tunable_op_t>()->set_config(
            reflection::general_object_t::make(gemm_cfg));
    auto exp = mgr.make("exp", {gemm->get_outputs()[0]}, {}, {});
    auto reduce = mgr.make("reduce", {exp->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    auto add = mgr.make(
            "add", {gemm->get_outputs()[0], reduce->get_outputs()[0]}, {}, {});
    mgr.make_output({add->get_outputs()[0]});
    layout_propagation(mgr, get_test_ctx());
    fuse_ops(mgr, get_test_ctx());

    auto ir_mod = lower_graph(get_test_ctx(), mgr, {});
    tensor_shrinker_t pass;
    auto func = ir_mod->get_func("matmul_core_exp_reduce_add_reorder__4");
    COMPILE_ASSERT(func, "no function got");
    auto ss = pass(func->body_).checked_as<stmts>()->seq_;
    COMPILE_ASSERT(ss.size() > 1, "Unexpected stmts size found");
    auto cur_loop = ss.at(1).checked_as<for_loop>().get();
    while (true) {
        auto nextloop = get_inner_for_loop(cur_loop).get();
        if (!nextloop) break;
        cur_loop = nextloop;
    }
    ss = cur_loop->body_.checked_as<stmts>()->seq_;
    COMPILE_ASSERT(ss.size(), "Unexpected stmts size found");
    auto tsr = ss.at(0).checked_as<define>()->var_.checked_as<tensor>();
    bool name_eq = (tsr->name_.find("__origouts_") != std::string::npos
            && tsr->name_.find("shr") != std::string::npos);
    EXPECT_TRUE(name_eq);
    int N_num_block = 128 / gemm_cfg.N_block;
    bool dim_eq = (get_expr_to_dims(tsr->dims_)
            == sc_dims {1, N_num_block, gemm_cfg.M_block, gemm_cfg.N_block});
    EXPECT_TRUE(dim_eq);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerInnerAnchorElemOp1) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 30, 40, 64}, datatypes::f32));
    auto fexp = fusion.make<exp_op_t>(finput->get_outputs()[0]);
    auto frelu = fusion.make<relu_op_t>(fexp->get_outputs()[0]);
    auto fred = fusion.make<reduce_sum_op_t>(
            frelu->get_outputs()[0], std::vector<int> {1, 3}, true);
    auto fadd = fusion.make<add_op_t>(
            finput->get_outputs()[0], fred->get_outputs()[0], false);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 64}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 64})) {
        _bind_(inp0, out0);
        _for_(m_o, 0, 20) {
            _for_(n_o, 0, 30) {
                // Anchor 0
                fusion.create_output_fusion_anchor(
                        {tensor_slice(
                                inp0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 64}})},
                        {tensor_slice(
                                out0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 64}})});
            }
            // Anchor 1
            fusion.create_output_fusion_anchor(
                    {tensor_slice(inp0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 64}})},
                    {tensor_slice(
                            out0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 64}})});
        }
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    int lanes = fexp->get_vx_info().lanes;
    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 64}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 64})) {
        _bind_(inp0, out0);
        auto out0_ptr = builder::tensor_ptr(out0, {30, 10, 3}, {}, true);

        _for_(m_o, 0, 20) {
            // Anchor 0
            _tensor_(_fuse_buf_0, datatypes::f32, {20UL, 30UL, 40UL, 64UL});
            _tensor_(_fuse_buf_1, datatypes::f32, {20UL, 1UL, 40UL, 1UL});
            _for_(n_o, 0, 30) {
                _for_(i, 0, 40) {
                    _for_(j, 0, 64, lanes) {
                        auto _fuse_buf_0_ptr = builder::tensor_ptr(
                                _fuse_buf_0, {m_o, n_o, 0, 0}, {}, true);
                        auto inp0_ptr = builder::tensor_ptr(
                                inp0, {m_o, n_o, 0, 0}, {}, true);
                        _fuse_buf_0_ptr[span_t({0, 0, i, j}, lanes)]
                                = builder::make_exp(
                                        inp0_ptr[span_t({0, 0, i, j}, lanes)]);
                        _fuse_buf_0_ptr = builder::tensor_ptr(
                                _fuse_buf_0, {m_o, n_o, i, j}, {}, true);
                        _fuse_buf_0_ptr[span_t({0, 0, 0, 0}, lanes)]
                                = builder::make_max(
                                        _fuse_buf_0_ptr[span_t(
                                                {0, 0, 0, 0}, lanes)],
                                        make_expr<constant_node>(0.f,
                                                sc_data_type_t::f32(lanes)));
                    }
                }
            }
            // Anchor 1
            // fuse reduce
            _for_(i, 0, 1) {
                _for_(ii, 0, 40) {
                    _var_(reduce_sum, sc_data_type_t::f32(lanes));
                    reduce_sum = make_expr<constant_node>(
                            0.0f, sc_data_type_t::f32(lanes));
                    _for_(j, 0, 30) {
                        _for_(jj, 0, 64, static_cast<int>(lanes)) {
                            reduce_sum = builder::make_add(
                                    builder::tensor_ptr(_fuse_buf_0,
                                            {m_o, 0, 0, 0}, {}, true)[span_t(
                                            {i, j, ii, jj}, lanes)],
                                    reduce_sum);
                        }
                    }
                    builder::tensor_ptr(_fuse_buf_1, {m_o, 0, 0, 0}, {},
                            true)[{i, 0, ii, 0}]
                            = builder::make_reduce_add(reduce_sum);
                }
            }
            // fuse add
            _for_(i, 0, 1) {
                _for_(j, 0, 30) {
                    _for_(ii, 0, 40) {
                        _for_(jj, 0, 64, static_cast<int>(lanes)) {
                            builder::tensor_ptr(out0, {m_o, 0, 0, 0}, {},
                                    true)[span_t({i, j, ii, jj}, lanes)]
                                    = builder::tensor_ptr(inp0, {m_o, 0, 0, 0},
                                              {}, true)[span_t(
                                              {i, j, ii, jj}, lanes)]
                                    + builder::make_broadcast(
                                            builder::tensor_ptr(_fuse_buf_1,
                                                    {m_o, 0, 0, 0}, {},
                                                    true)[{i, 0, ii, 0}],
                                            static_cast<int>(lanes));
                        }
                    }
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

// This test shows that Elementwise op will create larger inner anchor when tail
// computation occurs.
TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerInnerAnchorElemOp2) {
    REQUIRE_AVX();
    bool is_builtin = false;
    is_builtin = is_builtin_test_ctx();
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 30, 40, 50}, datatypes::f32));
    auto fexp = fusion.make<exp_op_t>(finput->get_outputs()[0]);
    auto frelu = fusion.make<relu_op_t>(fexp->get_outputs()[0]);
    auto fred = fusion.make<reduce_sum_op_t>(
            frelu->get_outputs()[0], std::vector<int> {1, 3}, true);
    auto fadd = fusion.make<add_op_t>(
            finput->get_outputs()[0], fred->get_outputs()[0], false);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 50}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 50})) {
        _bind_(inp0, out0);
        _for_(m_o, 0, 20) {
            _for_(n_o, 0, 30) {
                // Anchor 0
                fusion.create_output_fusion_anchor(
                        {tensor_slice(
                                inp0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 50}})},
                        {tensor_slice(
                                out0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 50}})});
            }
            // Anchor 1
            fusion.create_output_fusion_anchor(
                    {tensor_slice(inp0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 50}})},
                    {tensor_slice(
                            out0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 50}})});
        }
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    int lanes = fexp->get_vx_info().lanes;
    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 50}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 50})) {
        _bind_(inp0, out0);
        auto out0_ptr = builder::tensor_ptr(out0, {30, 10, 3}, {}, true);

        _for_(m_o, 0, 20) {
            // Anchor 0
            _tensor_(_fuse_buf_0, datatypes::f32, {20UL, 30UL, 40UL, 50UL});
            _tensor_(_fuse_buf_1, datatypes::f32, {20UL, 1UL, 40UL, 1UL});
            _for_(n_o, 0, 30) {
                _for_(i, 0, 40) {
                    _for_(j, 0, 48, lanes) {
                        auto _fuse_buf_0_ptr = builder::tensor_ptr(
                                _fuse_buf_0, {m_o, n_o, 0, 0}, {}, true);
                        auto inp0_ptr = builder::tensor_ptr(
                                inp0, {m_o, n_o, 0, 0}, {}, true);
                        _fuse_buf_0_ptr[span_t({0, 0, i, j}, lanes)]
                                = builder::make_exp(
                                        inp0_ptr[span_t({0, 0, i, j}, lanes)]);
                    }
                    if (is_builtin) {
                        _for_(j, 48, 50, lanes) {
                            auto _fuse_buf_0_ptr = builder::tensor_ptr(
                                    _fuse_buf_0, {m_o, n_o, 0, 0}, {}, true);
                            auto inp0_ptr = builder::tensor_ptr(
                                    inp0, {m_o, n_o, 0, 0}, {}, true);
                            auto mask = last_dim_generate_mask(
                                    j, 48, 50, lanes, true);
                            _fuse_buf_0_ptr[span_t({0, 0, i, j}, lanes, mask)]
                                    = builder::make_exp(inp0_ptr[span_t(
                                            {0, 0, i, j}, lanes, mask)]);
                        }
                    } else {
                        _for_(j, 48, 50, 1) {
                            auto _fuse_buf_0_ptr = builder::tensor_ptr(
                                    _fuse_buf_0, {m_o, n_o, 0, 0}, {}, true);
                            auto inp0_ptr = builder::tensor_ptr(
                                    inp0, {m_o, n_o, 0, 0}, {}, true);
                            _fuse_buf_0_ptr[span_t({0, 0, i, j}, 1)]
                                    = builder::make_exp(
                                            inp0_ptr[span_t({0, 0, i, j}, 1)]);
                        }
                    }

                    _for_(j, 0, 48, lanes) {
                        auto _fuse_buf_0_ptr = builder::tensor_ptr(
                                _fuse_buf_0, {m_o, n_o, i, 0}, {}, true);

                        _fuse_buf_0_ptr[span_t({0, 0, 0, j}, lanes)]
                                = builder::make_max(
                                        _fuse_buf_0_ptr[span_t(
                                                {0, 0, 0, j}, lanes)],
                                        make_expr<constant_node>(0.f,
                                                sc_data_type_t::f32(lanes)));
                    }
                    if (is_builtin) {
                        _for_(j, 48, 50, lanes) {
                            auto _fuse_buf_0_ptr = builder::tensor_ptr(
                                    _fuse_buf_0, {m_o, n_o, i, 0}, {}, true);
                            auto mask = last_dim_generate_mask(
                                    j, 48, 50, lanes, true);

                            _fuse_buf_0_ptr[span_t({0, 0, 0, j}, lanes, mask)]
                                    = builder::make_max(
                                            _fuse_buf_0_ptr[span_t(
                                                    {0, 0, 0, j}, lanes, mask)],
                                            make_expr<constant_node>(0.f,
                                                    sc_data_type_t::f32(
                                                            lanes)));
                        }
                    } else {
                        _for_(j, 48, 50, 1) {
                            auto _fuse_buf_0_ptr = builder::tensor_ptr(
                                    _fuse_buf_0, {m_o, n_o, i, 0}, {}, true);

                            _fuse_buf_0_ptr[span_t({0, 0, 0, j}, 1)]
                                    = builder::make_max(
                                            _fuse_buf_0_ptr[span_t(
                                                    {0, 0, 0, j}, 1)],
                                            make_expr<constant_node>(0.f,
                                                    sc_data_type_t::f32(1)));
                        }
                    }
                }
            }
            // Anchor 1
            // fuse reduce
            _for_(i, 0, 1) {
                _for_(ii, 0, 40) {
                    _var_(reduce_sum, sc_data_type_t::f32(lanes));
                    reduce_sum = make_expr<constant_node>(
                            0.0f, sc_data_type_t::f32(lanes));
                    _for_(j, 0, 30) {
                        _for_(jj, 0, 50, lanes) {
                            assert(lanes == 16 || lanes == 8);
                            auto mask
                                    = last_dim_generate_mask(jj, 48, 50, lanes);

                            reduce_sum = builder::make_add(
                                    builder::tensor_ptr(_fuse_buf_0,
                                            {m_o, 0, 0, 0}, {}, true)[span_t(
                                            {i, j, ii, jj}, lanes, mask)],
                                    reduce_sum);
                        }
                    }
                    builder::tensor_ptr(_fuse_buf_1, {m_o, 0, 0, 0}, {},
                            true)[{i, 0, ii, 0}]
                            = builder::make_reduce_add(reduce_sum);
                }
            }
            // fuse add
            _for_(i, 0, 1) {
                _for_(j, 0, 30) {
                    _for_(ii, 0, 40) {
                        _for_(jj, 0, 48, lanes) {
                            builder::tensor_ptr(out0, {m_o, 0, 0, 0}, {},
                                    true)[span_t({i, j, ii, jj}, lanes)]
                                    = builder::tensor_ptr(inp0, {m_o, 0, 0, 0},
                                              {}, true)[span_t(
                                              {i, j, ii, jj}, lanes)]
                                    + builder::make_broadcast(
                                            builder::tensor_ptr(_fuse_buf_1,
                                                    {m_o, 0, 0, 0}, {},
                                                    true)[{i, 0, ii, 0}],
                                            static_cast<int>(lanes));
                        }
                        if (is_builtin) {
                            _for_(jj, 48, 50, lanes) {
                                auto mask = last_dim_generate_mask(
                                        jj, 48, 50, lanes, true);
                                builder::tensor_ptr(
                                        out0, {m_o, 0, 0, 0}, {}, true)[span_t(
                                        {i, j, ii, jj}, lanes, mask)]
                                        = builder::tensor_ptr(inp0,
                                                  {m_o, 0, 0, 0}, {},
                                                  true)[span_t(
                                                  {i, j, ii, jj}, lanes, mask)]
                                        + builder::make_broadcast(
                                                builder::tensor_ptr(_fuse_buf_1,
                                                        {m_o, 0, 0, 0}, {},
                                                        true)[{i, 0, ii, 0}],
                                                static_cast<int>(lanes));
                            }
                        } else {
                            _for_(jj, 48, 50, 1) {
                                builder::tensor_ptr(out0, {m_o, 0, 0, 0}, {},
                                        true)[span_t({i, j, ii, jj}, 1)]
                                        = builder::tensor_ptr(inp0,
                                                  {m_o, 0, 0, 0}, {},
                                                  true)[span_t(
                                                  {i, j, ii, jj}, 1)]
                                        + builder::tensor_ptr(_fuse_buf_1,
                                                {m_o, 0, 0, 0}, {},
                                                true)[{i, 0, ii, 0}];
                            }
                        }
                    }
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerInnerAnchorReduce1) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 30, 40, 64}, datatypes::f32));
    auto fexp = fusion.make<exp_op_t>(finput->get_outputs()[0]);
    auto fred = fusion.make<reduce_sum_op_t>(
            fexp->get_outputs()[0], std::vector<int> {1, 3}, true);
    auto frelu = fusion.make<relu_op_t>(fred->get_outputs()[0]);
    auto fadd = fusion.make<add_op_t>(
            fred->get_outputs()[0], frelu->get_outputs()[0], false);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 64}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 64})) {
        _bind_(inp0, out0);
        _for_(m_o, 0, 20) {
            _for_(n_o, 0, 30) {
                // Anchor 0
                fusion.create_output_fusion_anchor(
                        {tensor_slice(
                                inp0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 64}})},
                        {tensor_slice(
                                out0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 64}})});
            }
            // Anchor 1
            fusion.create_output_fusion_anchor(
                    {tensor_slice(inp0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 64}})},
                    {tensor_slice(
                            out0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 64}})});
        }
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    int lanes = fexp->get_vx_info().lanes;
    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 64}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 64})) {
        _bind_(inp0, out0);
        auto out0_ptr = builder::tensor_ptr(out0, {30, 10, 3}, {}, true);

        _for_(m_o, 0, 20) {
            // Anchor 0
            _tensor_(_fuse_buf_2, datatypes::f32, {20UL, 1UL, 40UL, 1UL});
            _tensor_(_fuse_buf_1, datatypes::f32, {20UL, 1UL, 40UL, 1UL});
            _for_(n_o, 0, 30) {
                _for_(i, 0, 40) {
                    _for_(j, 0, 64, lanes) {
                        auto inp0_ptr = builder::tensor_ptr(
                                inp0, {m_o, n_o, 0, 0}, {}, true);
                        inp0_ptr[span_t({0, 0, i, j}, lanes)]
                                = builder::make_exp(
                                        inp0_ptr[span_t({0, 0, i, j}, lanes)]);
                    }
                }
            }
            // Anchor 1
            // fuse reduce
            _for_(i, 0, 1) {
                _for_(ii, 0, 40) {
                    _var_(reduce_sum, sc_data_type_t::f32(lanes));
                    reduce_sum = make_expr<constant_node>(
                            0.0f, sc_data_type_t::f32(lanes));
                    _for_(j, 0, 30) {
                        _for_(jj, 0, 64, static_cast<int>(lanes)) {
                            reduce_sum = builder::make_add(
                                    builder::tensor_ptr(inp0, {m_o, 0, 0, 0},
                                            {}, true)[span_t(
                                            {i, j, ii, jj}, lanes)],
                                    reduce_sum);
                        }
                    }
                    auto _fuse_buf_1_ptr = builder::tensor_ptr(
                            _fuse_buf_1, {m_o, 0, 0, 0}, {}, true);
                    _fuse_buf_1_ptr[{i, 0, ii, 0}]
                            = builder::make_reduce_add(reduce_sum);
                    auto _fuse_buf_2_ptr = builder::tensor_ptr(
                            _fuse_buf_2, {m_o, 0, ii, 0}, {}, true);
                    _fuse_buf_1_ptr = builder::tensor_ptr(
                            _fuse_buf_1, {m_o, 0, ii, 0}, {}, true);
                    _fuse_buf_2_ptr[{0, 0, 0, 0}]
                            = builder::make_max(_fuse_buf_1_ptr[{0, 0, 0, 0}],
                                    make_expr<constant_node>(0.f));
                    builder::tensor_ptr(
                            out0, {m_o, 0, ii, 0}, {}, true)[{0, 0, 0, 0}]
                            = _fuse_buf_1_ptr[{0, 0, 0, 0}]
                            + _fuse_buf_2_ptr[{0, 0, 0, 0}];
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

// This test shows that reduce op will not generate inner anchor due to
// performance consideration
TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerInnerAnchorReduce2) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput
            = fusion.make<input_op>(make_tsr({20, 30, 40, 64}, datatypes::f32));
    auto fexp = fusion.make<exp_op_t>(finput->get_outputs()[0]);
    auto fred = fusion.make<reduce_sum_op_t>(
            fexp->get_outputs()[0], std::vector<int> {1, 3}, true);
    auto frelu = fusion.make<relu_op_t>(fred->get_outputs()[0]);
    auto fadd = fusion.make<add_op_t>(
            fexp->get_outputs()[0], frelu->get_outputs()[0], false);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);
    _function_(datatypes::s32, aaa,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 64}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 64})) {
        _bind_(inp0, out0);
        _for_(m_o, 0, 20) {
            _for_(n_o, 0, 30) {
                // Anchor 0
                fusion.create_output_fusion_anchor(
                        {tensor_slice(
                                inp0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 64}})},
                        {tensor_slice(
                                out0, {{m_o, 1}, {n_o, 1}, {0, 40}, {0, 64}})});
            }
            // Anchor 1
            fusion.create_output_fusion_anchor(
                    {tensor_slice(inp0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 64}})},
                    {tensor_slice(
                            out0, {{m_o, 1}, {0, 30}, {0, 40}, {0, 64}})});
        }
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    int lanes = fexp->get_vx_info().lanes;
    ///// Expected func:
    _function_(datatypes::s32, bbb,
            _arg_("inp0", datatypes::f32, {20, 30, 40, 64}),
            _arg_("out0", datatypes::f32, {20, 30, 40, 64})) {
        _bind_(inp0, out0);
        auto out0_ptr = builder::tensor_ptr(out0, {30, 10, 3}, {}, true);

        _for_(m_o, 0, 20) {
            // Anchor 0
            _tensor_(_fuse_buf_1, datatypes::f32, {20UL, 1UL, 40UL, 1UL});
            _for_(n_o, 0, 30) {
                _for_(i, 0, 40) {
                    _for_(j, 0, 64, lanes) {
                        auto inp0_ptr = builder::tensor_ptr(
                                inp0, {m_o, n_o, 0, 0}, {}, true);
                        inp0_ptr[span_t({0, 0, i, j}, lanes)]
                                = builder::make_exp(
                                        inp0_ptr[span_t({0, 0, i, j}, lanes)]);
                    }
                }
            }
            // Anchor 1
            // fuse reduce
            _for_(i, 0, 1) {
                _for_(ii, 0, 40) {
                    _var_(reduce_sum, sc_data_type_t::f32(lanes));
                    reduce_sum = make_expr<constant_node>(
                            0.0f, sc_data_type_t::f32(lanes));
                    _for_(j, 0, 30) {
                        _for_(jj, 0, 64, static_cast<int>(lanes)) {
                            reduce_sum = builder::make_add(
                                    builder::tensor_ptr(inp0, {m_o, 0, 0, 0},
                                            {}, true)[span_t(
                                            {i, j, ii, jj}, lanes)],
                                    reduce_sum);
                        }
                    }
                    auto _fuse_buf_1_ptr = builder::tensor_ptr(
                            _fuse_buf_1, {m_o, 0, 0, 0}, {}, true);
                    _fuse_buf_1_ptr[{i, 0, ii, 0}]
                            = builder::make_reduce_add(reduce_sum);
                }
            }
            _for_(i, 0, 1) {
                _for_(ii, 0, 40) {
                    auto _fuse_buf_1_ptr = builder::tensor_ptr(
                            _fuse_buf_1, {m_o, 0, 0, 0}, {}, true);
                    _fuse_buf_1_ptr[{i, 0, ii, 0}]
                            = builder::make_max(_fuse_buf_1_ptr[{i, 0, ii, 0}],
                                    make_expr<constant_node>(0.f));
                }
            }
            // fuse add
            _for_(i, 0, 1) {
                _for_(j, 0, 30) {
                    _for_(ii, 0, 40) {
                        _for_(jj, 0, 64, static_cast<int>(lanes)) {
                            builder::tensor_ptr(out0, {m_o, 0, 0, 0}, {},
                                    true)[span_t({i, j, ii, jj}, lanes)]
                                    = builder::tensor_ptr(inp0, {m_o, 0, 0, 0},
                                              {}, true)[span_t(
                                              {i, j, ii, jj}, lanes)]
                                    + builder::make_broadcast(
                                            builder::tensor_ptr(_fuse_buf_1,
                                                    {m_o, 0, 0, 0}, {},
                                                    true)[{i, 0, ii, 0}],
                                            static_cast<int>(lanes));
                        }
                    }
                }
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestBinaryElementwiseOp) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput0 = fusion.make<input_op>(make_tsr({100, 200}));
    auto finput1 = fusion.make<input_op>(make_tsr({100, 200}));
    auto fmin = fusion.make<min_op_t>(
            finput0->get_outputs()[0], finput1->get_outputs()[0], 0, 1);
    auto fmax = fusion.make<max_op_t>(
            finput0->get_outputs()[0], fmin->get_outputs()[0]);
    auto fout = fusion.make<output_op>(fmax->get_outputs()[0]);

    EXPECT_EQ(fmax->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::s32, aaa, _arg_("inp0", datatypes::s32, {100, 200}),
            _arg_("inp1", datatypes::s32, {100, 200}),
            _arg_("out0", datatypes::s32, {100, 200})) {
        _bind_(inp0, inp1, out0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 128}}),
                        tensor_slice(
                                inp1, {/*dim1*/ {10, 10}, /*dim2*/ {0, 128}})},
                {tensor_slice(out0, {/*dim1*/ {20, 10}, /*dim2*/ {0, 128}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[2]});

    int lanes = ctx->get_max_vector_lanes(sc_data_etype::F32);
    ///// Expected func:
    _function_(datatypes::s32, bbb, _arg_("inp0", datatypes::s32, {100, 200}),
            _arg_("inp1", datatypes::s32, {100, 200}),
            _arg_("out0", datatypes::s32, {100, 200})) {
        _bind_(inp0, inp1, out0);
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto inp1_tptr = builder::tensor_ptr(inp1, {10, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj, 0, 128, lanes) {
                inp1_tptr[span_t({ii, jj}, lanes)]
                        = builder::make_min(inp0_tptr[span_t({ii, jj}, lanes)],
                                inp1_tptr[span_t({ii, jj}, lanes)]);
                auto out0_tptr = builder::tensor_ptr(
                        out0, {ii + UINT64_C(20), jj}, {}, true);
                inp0_tptr = builder::tensor_ptr(inp0, {ii, jj}, {}, true);
                inp1_tptr = builder::tensor_ptr(inp1, {ii, jj}, {}, true);
                out0_tptr[span_t({0, 0}, lanes)]
                        = builder::make_max(inp0_tptr[span_t({0, 0}, lanes)],
                                inp1_tptr[span_t({0, 0}, lanes)]);
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestReshapeCopyOp) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput0 = fusion.make<input_op>(make_tsr({100, 200}));
    auto freshape = fusion.make<reshape_op_t>(finput0->get_outputs(),
            std::vector<graph_tensor_ptr>(),
            any_map_t {{"shape", sc_dims {10, 10, 20, 10}}});
    auto fout = fusion.make<output_op>(freshape->get_outputs()[0]);

    _function_(datatypes::s32, aaa, _arg_("inp0", datatypes::s32, {100, 200}),
            _arg_("out0", datatypes::s32, {10, 10, 20, 10})) {
        _bind_(inp0, out0);
        fusion.create_output_fusion_anchor({tensor_slice(
                inp0, {/*dim1*/ {0, 100UL}, /*dim2*/ {0, 200UL}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[1]});

    ///// Expected func:
    _function_(datatypes::s32, bbb, _arg_("inp0", datatypes::s32, {100, 200}),
            _arg_("out0", datatypes::s32, {10, 10, 20, 10})) {
        _bind_(inp0, out0);
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto out0_tptr = builder::tensor_ptr(out0, {0, 0, 0, 0}, {}, true);
        _for_(ii, 0, 10 * 10 * 20 * UINT64_C(10)) {
            out0_tptr[{ii / (10 * 20 * UINT64_C(10)),
                    ii % (10 * 20 * UINT64_C(10)) / (20 * UINT64_C(10)),
                    ii % (20 * UINT64_C(10)) / UINT64_C(10), ii % UINT64_C(10)}]
                    = inp0_tptr[{ii / UINT64_C(200), ii % UINT64_C(200)}];
        }
        _return_(123);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestVecterizedClampOP) {
    REQUIRE_AVX2();
    auto check_clamp = [&](const int M, const int K, bool vectorized,
                               const float clamp_min, const float clamp_max) {
        builder::ir_builder_t builder;
        fusion_manager fusion;
        auto fin = fusion.make<input_op>(make_tsr({M, K}, datatypes::f32));
        auto fclamp = fusion.make<clamp_op_t>(
                fin->get_outputs()[0], clamp_min, clamp_max);
        auto fout = fusion.make<output_op>(fclamp->get_outputs()[0]);

        _function_(datatypes::void_t, aaa, _arg_("in", datatypes::f32, {M, K}),
                _arg_("out", datatypes::f32, {M, K})) {
            _bind_(in, out);

            fusion.create_output_fusion_anchor(
                    {tensor_slice(in, {{0, M}, {0, K}})},
                    {tensor_slice(out, {{0, M}, {0, K}})});
        }
        do_commit(fusion, {aaa->params_[1]});

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);
        // verify the upper/under bound values
        in_buf[0] = -10.f;
        in_buf[1] = -100.f;
        in_buf[2] = -std::numeric_limits<float>::infinity();
        in_buf[3] = 10.f;
        in_buf[4] = 100.f;
        in_buf[5] = std::numeric_limits<float>::infinity();

        in_buf[6] = std::numeric_limits<float>::min();
        in_buf[7] = std::numeric_limits<float>::max();

        in_buf[8] = -std::numeric_limits<float>::min();
        in_buf[9] = -std::numeric_limits<float>::max();

        fptr->call_default(in_buf.data(), out_buf.data());

        ref_clamp(
                ref_out_buf.data(), in_buf.data(), M * K, clamp_min, clamp_max);

        test_utils::compare_data(out_buf, ref_out_buf, 1e-5f, 1e-5f);
    };
    // scalar version
    // check_clamp(100, 200, false, 0.1f, 0.5f);

    // vectorization version
    check_clamp(100, 256, true, 0.1f, 0.5f);
}

class check_hint_visitor_t : public ir_visitor_t {
public:
    int64_t default_tick = special_ticks::HINT_IN_LOOP;
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> &guide_ticks_;
    check_hint_visitor_t(
            std::unordered_map<std::string, std::pair<int64_t, int64_t>>
                    &guide_ticks)
        : guide_ticks_(guide_ticks) {}
    expr_c visit(tensor_c v) override {
        auto itr = guide_ticks_.find(v->name_);
        if (itr != guide_ticks_.end()) {
            EXPECT_EQ(v.remove_const()->attr().get_or_else(
                              attr_keys::hint_first_access_tick, default_tick),
                    itr->second.first);
            EXPECT_EQ(v.remove_const()->attr().get_or_else(
                              attr_keys::hint_last_access_tick, default_tick),
                    itr->second.second);
        }
        return ir_visitor_t::visit(v);
    }
};

TEST(GCCore_CPU_fuse_mgr_cpp, TestTensorHintForBufferSchedule1) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput0 = fusion.make<input_op>(make_tsr({100, 200}));
    auto finput1 = fusion.make<input_op>(make_tsr({100, 200}));
    auto finput2 = fusion.make<input_op>(make_tsr({100}));
    auto frelu0 = fusion.make<relu_op_t>(finput0->get_outputs()[0]);
    auto fcast1
            = fusion.make<cast_op_t>(finput1->get_outputs()[0], datatypes::s8);
    auto fcast2
            = fusion.make<cast_op_t>(fcast1->get_outputs()[0], datatypes::s32);
    auto fadd = fusion.make<add_op_t>(
            frelu0->get_outputs()[0], fcast2->get_outputs()[0]);
    auto freduce = fusion.make<reduce_op_t>(
            fadd->get_outputs()[0], std::vector<int> {1});
    auto frelu1 = fusion.make<relu_op_t>(freduce->get_outputs()[0]);
    auto fadd2 = fusion.make<add_op_t>(
            frelu1->get_outputs()[0], finput2->get_outputs()[0]);
    auto fout = fusion.make<output_op>(fadd2->get_outputs()[0]);

    _function_(datatypes::s32, aaa, _arg_("inp0", datatypes::s32, {100, 200}),
            _arg_("inp1", datatypes::s32, {100, 200}),
            _arg_("inp2", datatypes::s32, {100}),
            _arg_("out0", datatypes::s32, {100})) {
        _bind_(inp0, inp1, inp2, out0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 200}}),
                        tensor_slice(
                                inp1, {/*dim1*/ {10, 10}, /*dim2*/ {0, 200}}),
                        tensor_slice(inp2, {/*dim1*/ {10, 10}})},
                {tensor_slice(out0, {/*dim1*/ {20, 10}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[3]});
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> guide_ticks {
            {"inp0", {0, 8}}, {"inp1", {0, 3}},
            {"_cast_buf_0" /*cast1 out*/, {-3, 5}},
            {"_cast_buf_1" /*cast2 out*/, {-3, 7}},
            {"_reduce_buf_2" /*reduce_buf_2*/, {-3, 10}},
            {"out0" /*reduce out*/, {-3, 11}}};
    check_hint_visitor_t vis(guide_ticks);
    vis.dispatch(aaa);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestTensorHintForBufferSchedule2) {
    // mha training pattern
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput0 = fusion.make<input_op>(make_tsr({100, 200}));
    auto finput1 = fusion.make<input_op>(make_tsr({100, 200}));
    auto finput2 = fusion.make<input_op>(make_tsr({100, 200}));
    auto finput3 = fusion.make<input_op>(make_tsr({100, 200}));
    auto finput4 = fusion.make<input_op>(make_tsr({100, 200}));

    auto fcast0
            = fusion.make<cast_op_t>(finput0->get_outputs()[0], datatypes::f32);
    auto fmul0 = fusion.make<mul_op_t>(
            fcast0->get_outputs()[0], finput1->get_outputs()[0]);
    auto fmul1 = fusion.make<mul_op_t>(
            fmul0->get_outputs()[0], finput2->get_outputs()[0]);
    auto freduce = fusion.make<reduce_sum_op_t>(
            fmul1->get_outputs()[0], std::vector<int> {1}, true);
    auto fsub = fusion.make<sub_op_t>(
            fmul0->get_outputs()[0], freduce->get_outputs()[0]);
    auto fmul2 = fusion.make<mul_op_t>(
            fsub->get_outputs()[0], finput3->get_outputs()[0]);
    auto fdiv = fusion.make<div_op_t>(
            fmul2->get_outputs()[0], finput4->get_outputs()[0]);
    auto fout = fusion.make<output_op>(fdiv->get_outputs()[0]);

    _function_(datatypes::s32, aaa, _arg_("inp0", datatypes::s32, {100, 200}),
            _arg_("inp1", datatypes::f32, {100, 200}),
            _arg_("inp2", datatypes::f32, {100, 200}),
            _arg_("inp3", datatypes::f32, {100, 200}),
            _arg_("inp4", datatypes::f32, {100, 200}),
            _arg_("out0", datatypes::f32, {100, 200})) {
        _bind_(inp0, inp1, inp2, inp3, inp4, out0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 200}}),
                        tensor_slice(
                                inp1, {/*dim1*/ {10, 10}, /*dim2*/ {0, 200}}),
                        tensor_slice(
                                inp2, {/*dim1*/ {10, 10}, /*dim2*/ {0, 200}}),
                        tensor_slice(
                                inp3, {/*dim1*/ {10, 10}, /*dim2*/ {0, 200}}),
                        tensor_slice(
                                inp4, {/*dim1*/ {10, 10}, /*dim2*/ {0, 200}})},
                {tensor_slice(out0, {/*dim1*/ {20, 10}, {0, 200}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[5]});
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> guide_ticks {
            {"inp0" /*dont schedule, s32*/, {0, 6}}, {"inp1", {0, 7}},
            {"inp2", {0, 8}}, {"inp3", {0, 11}}, {"inp4", {0, 12}},
            {"_cast_buf_0" /*cast out*/, {-3, 10}},
            {"_mul_buf_1" /*mul1 out*/, {8, 9}},
            {"_reduce_buf_2" /*reduce_buf_2*/, {-3, 10}},
            {"_sub_buf_3" /*sub_buf_3*/, {10, 12}},
            {"out0" /*div out*/, {-3, 13}}};
    check_hint_visitor_t vis(guide_ticks);
    vis.dispatch(aaa);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestVecterizedRoundOP) {
    REQUIRE_AVX2();
    auto check_round = [&](const int M, const int K) {
        builder::ir_builder_t builder;
        fusion_manager fusion;
        auto fin = fusion.make<input_op>(make_tsr({M, K}, datatypes::f32));
        auto fround = fusion.make<round_op_t>(fin->get_outputs()[0]);
        auto fout = fusion.make<output_op>(fround->get_outputs()[0]);

        _function_(datatypes::void_t, aaa, _arg_("in", datatypes::f32, {M, K}),
                _arg_("out", datatypes::f32, {M, K})) {
            _bind_(in, out);

            fusion.create_output_fusion_anchor(
                    {tensor_slice(in, {{0, M}, {0, K}})},
                    {tensor_slice(out, {{0, M}, {0, K}})});
        }
        do_commit(fusion, {aaa->params_[1]});

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);
        // verify the upper/under bound values
        in_buf[0] = -10.5f;
        in_buf[1] = -10.4f;

        in_buf[2] = 10.455f;
        in_buf[3] = 100.5f;

        in_buf[4] = std::numeric_limits<float>::min();
        in_buf[5] = std::numeric_limits<float>::max();

        in_buf[6] = -std::numeric_limits<float>::min();
        in_buf[7] = -std::numeric_limits<float>::max();

        fptr->call_default(in_buf.data(), out_buf.data());

        ref_round(ref_out_buf.data(), in_buf.data(), M * K);

        test_utils::compare_data(out_buf, ref_out_buf, 1e-5f, 1e-5f);
    };

    // vectorization version
    check_round(100, 256);
}

// #define BENCH_CLAMP

#ifdef BENCH_CLAMP
TEST(GCCore_CPU_fuse_mgr_cpp, BenchVectorizedClampOP) {
    auto bench_clamp = [&](const int M, const int K, bool vectorized,
                               const float clamp_min, const float clamp_max) {
        builder::ir_builder_t builder;
        fusion_manager fusion;
        auto fin = fusion.make<input_op>();
        auto fclamp = fusion.make<clamp_op_t>(
                fin->get_outputs()[0], clamp_min, clamp_max);
        auto fout = fusion.make<output_op>(fclamp->get_outputs()[0]);

        _function_(datatypes::void_t, aaa, _arg_("in", datatypes::f32, {M, K}),
                _arg_("out", datatypes::f32, {M, K})) {
            _bind_(in, out);

            fusion.create_output_fusion_anchor(
                    {tensor_slice(in, {{0, M}, {0, K}})},
                    {tensor_slice(out, {{0, M}, {0, K}})});
        }
        fusion.commit(ctx, aaa);

        auto aaa1 = ir_module_t::from_entry_func(get_default_context(), aaa);
        auto fptr = jit_engine_t::make(get_test_ctx())
                            ->get_entry_func(aaa1, true);

        std::vector<float> in_buf(M * K);
        std::vector<float> out_buf(M * K);
        std::vector<float> ref_out_buf(M * K);

        test_utils::fill_data(in_buf.data(), M * K);

        auto start = high_resolution_clock::now();

        int count = 100;
        for (int i = 0; i < count; i++) {
            fptr->call_default(in_buf.data(), out_buf.data());
        }

        auto stop = high_resolution_clock::now();

        auto duration
                = duration_cast<microseconds>(stop - start) / (1000.f * count);
        std::cout << "Data shape: " << M << "x" << K << ". "
                  << "Average time taken by op: " << duration.count()
                  << " milliseconds\n";
    };
    // vectorization version
    bench_clamp(100, 256, true, 0.1f, 0.5f);

    // vectorization version
    bench_clamp(100, 2560, true, 0.1f, 0.5f);

    // vectorization version
    bench_clamp(100, 25600, true, 0.1f, 0.5f);

    // vectorization version
    bench_clamp(100, 512000, true, 0.1f, 0.5f);
}

#endif

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionUnaryOp) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    sc_dims arg_dims = {100, 256};
    auto finput0 = fusion.make<input_op>(make_tsr(arg_dims, datatypes::s32));
    auto finput1 = fusion.make<input_op>(std::move(arg_dims), datatypes::s32);
    auto frelu = fusion.make<relu_op_t>(finput1->get_outputs()[0]);
    auto fadd = fusion.make<add_op_t>(
            finput0->get_outputs()[0], frelu->get_outputs()[0], true);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);

    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::s32, aaa, _arg_("out0", datatypes::s32, {100, 256}),
            _arg_("inp0", datatypes::s32, {100, 256}),
            _arg_("inp1", datatypes::s32, {100, 256})) {
        _bind_(out0, inp0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 256}})},
                {tensor_slice(out0, {/*dim1*/ {20, 10}, /*dim2*/ {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[0]}, {aaa->params_[2]});

    ///// Expected func:
    auto lanes = frelu->get_vx_info().lanes;
    _function_(datatypes::s32, bbb, _arg_("out0", datatypes::s32, {100, 256}),
            _arg_("inp0", datatypes::s32, {100, 256}),
            _arg_("inp1", datatypes::s32, {100, 256})) {
        _bind_(out0, inp0, inp1);

        auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0}, {}, true);

        _tensor_(_fuse_buf, datatypes::s32, {100UL, 256UL});
        auto fuse_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                fuse_tptr[span_t({ii, jj}, lanes)]
                        = builder::make_max(inp1_tptr[span_t({ii, jj}, lanes)],
                                make_expr<constant_node>((uint64_t)0,
                                        sc_data_type_t::s32(lanes)));
            }
        }
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                auto out0_tptr = builder::tensor_ptr(out0, {20, 0}, {}, true);
                auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
                fuse_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
                out0_tptr[span_t({ii, jj}, lanes)]
                        = inp0_tptr[span_t({ii, jj}, lanes)]
                        + fuse_tptr[span_t({ii, jj}, lanes)];
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionBinaryOp) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    sc_dims arg_dims_1 = {100, 256};
    auto finput0 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::s32));
    auto finput1 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::s32));
    auto finput2 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::s32));
    auto fsub = fusion.make<sub_op_t>(
            finput1->get_outputs()[0], finput2->get_outputs()[0], true);
    auto fadd = fusion.make<add_op_t>(
            finput0->get_outputs()[0], fsub->get_outputs()[0], true);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);

    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::s32, aaa, _arg_("out0", datatypes::s32, {100, 256}),
            _arg_("inp0", datatypes::s32, {100, 256}),
            _arg_("inp1", datatypes::s32, {100, 256}),
            _arg_("inp2", datatypes::s32, {100, 256})) {
        _bind_(out0, inp0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 256}})},
                {tensor_slice(out0, {/*dim1*/ {20, 10}, /*dim2*/ {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[0]}, {aaa->params_[2], aaa->params_[3]});

    ///// Expected func:
    auto lanes = fsub->get_vx_info().lanes;
    _function_(datatypes::s32, bbb, _arg_("out0", datatypes::s32, {100, 256}),
            _arg_("inp0", datatypes::s32, {100, 256}),
            _arg_("inp1", datatypes::s32, {100, 256}),
            _arg_("inp2", datatypes::s32, {100, 256})) {
        _bind_(out0, inp0, inp1, inp2);
        auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0}, {}, true);
        auto inp2_tptr = builder::tensor_ptr(inp2, {0, 0}, {}, true);
        _tensor_(_fuse_buf, datatypes::s32, {100UL, 256UL});
        auto fuse_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                fuse_tptr[span_t({ii, jj}, lanes)]
                        = inp1_tptr[span_t({ii, jj}, lanes)]
                        - inp2_tptr[span_t({ii, jj}, lanes)];
            }
        }
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                auto out0_tptr = builder::tensor_ptr(out0, {20, 0}, {}, true);
                auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
                fuse_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
                out0_tptr[span_t({ii, jj}, lanes)]
                        = inp0_tptr[span_t({ii, jj}, lanes)]
                        + fuse_tptr[span_t({ii, jj}, lanes)];
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionBinaryOpWithBroadCast) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    sc_dims arg_dims_1 = {100, 256};
    sc_dims arg_dims_2 = {1, 256};
    auto finput0 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::s32));
    auto finput1 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::s32));
    auto finput2 = fusion.make<input_op>(make_tsr(arg_dims_2, datatypes::s32));
    auto fmul = fusion.make<mul_op_t>(
            finput1->get_outputs()[0], finput2->get_outputs()[0], true);
    auto fadd = fusion.make<add_op_t>(
            finput0->get_outputs()[0], fmul->get_outputs()[0], true);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);

    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::s32, aaa, _arg_("out0", datatypes::s32, {100, 256}),
            _arg_("inp0", datatypes::s32, {100, 256}),
            _arg_("inp1", datatypes::s32, {100, 256}),
            _arg_("inp2", datatypes::s32, {1, 256})) {
        _bind_(out0, inp0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 256}})},
                {tensor_slice(out0, {/*dim1*/ {20, 10}, /*dim2*/ {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[0]}, {aaa->params_[2], aaa->params_[3]});

    ///// Expected func:
    auto lanes = fmul->get_vx_info().lanes;
    _function_(datatypes::s32, bbb, _arg_("out0", datatypes::s32, {100, 256}),
            _arg_("inp0", datatypes::s32, {100, 256}),
            _arg_("inp1", datatypes::s32, {100, 256}),
            _arg_("inp2", datatypes::s32, {1, 256})) {
        _bind_(out0, inp0, inp1, inp2);
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0}, {}, true);
        auto inp2_tptr = builder::tensor_ptr(inp2, {0, 0}, {}, true);
        auto out0_tptr = builder::tensor_ptr(out0, {20, 0}, {}, true);
        _tensor_(_fuse_buf, datatypes::s32, {100UL, 256UL});
        auto fuse_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                fuse_tptr[span_t({ii, jj}, lanes)]
                        = inp1_tptr[span_t({ii, jj}, lanes)]
                        * builder::make_indexing(inp2_tptr, {0, jj}, lanes);
            }
        }
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                out0_tptr[span_t({ii, jj}, lanes)]
                        = inp0_tptr[span_t({ii, jj}, lanes)]
                        + fuse_tptr[span_t({ii, jj}, lanes)];
            }
        }
        _return_(123);
    }
    //     std::cout << aaa << std::endl;
    //     std::cout << bbb << std::endl;
    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
    //     std::cout << cmper;
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionBroadcastOp) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    sc_dims arg_dims_1 = {100, 256};
    // Do broadcast
    sc_dims arg_dims_2 = {256};
    auto finput0 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::s32));
    auto finput1 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::s32));
    auto finput2 = fusion.make<input_op>(make_tsr(arg_dims_2, datatypes::s32));
    auto fbroadcast = fusion.make<mul_op_t>(
            finput1->get_outputs()[0], finput2->get_outputs()[0]);
    auto fadd = fusion.make<add_op_t>(
            finput0->get_outputs()[0], fbroadcast->get_outputs()[0], true);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);

    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::s32, aaa, _arg_("out0", datatypes::s32, {100, 256}),
            _arg_("inp0", datatypes::s32, {100, 256}),
            _arg_("inp1", datatypes::s32, {100, 256}),
            _arg_("inp2", datatypes::s32, {256})) {
        _bind_(out0, inp0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 256}})},
                {tensor_slice(out0, {/*dim1*/ {20, 10}, /*dim2*/ {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[0]}, {aaa->params_[2], aaa->params_[3]});

    ///// Expected func:
    auto lanes = fbroadcast->get_lanes();
    _function_(datatypes::s32, bbb, _arg_("out0", datatypes::s32, {100, 256}),
            _arg_("inp0", datatypes::s32, {100, 256}),
            _arg_("inp1", datatypes::s32, {100, 256}),
            _arg_("inp2", datatypes::s32, {256})) {
        _bind_(out0, inp0, inp1, inp2);
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0}, {}, true);
        auto inp2_tptr = builder::tensor_ptr(inp2, {0}, {}, true);
        auto out0_tptr = builder::tensor_ptr(out0, {20, 0}, {}, true);
        _tensor_(_fuse_buf, datatypes::s32, {100UL, 256UL});
        auto fuse_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                fuse_tptr[span_t({ii, jj}, lanes)]
                        = inp1_tptr[span_t({ii, jj}, lanes)]
                        * builder::make_indexing(inp2_tptr, {jj}, lanes);
            }
        }
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(lanes)) {
                out0_tptr[span_t({ii, jj}, lanes)]
                        = inp0_tptr[span_t({ii, jj}, lanes)]
                        + fuse_tptr[span_t({ii, jj}, lanes)];
            }
        }
        _return_(123);
    }
    //     std::cout << aaa << std::endl;
    //     std::cout << bbb << std::endl;
    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
    //     std::cout << cmper;
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionReduceOp) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    auto finput0 = fusion.make<input_op>(make_tsr({16, 16}, datatypes::f32));
    sc_dims arg_dims_1 = {10, 16, 10, 16};
    auto finput1 = fusion.make<input_op>(std::move(arg_dims_1), datatypes::f32);
    auto freduce = fusion.make<reduce_sum_op_t>(
            finput1->get_outputs()[0], std::vector<int> {0, 2}, false);
    auto fadd = fusion.make<add_op_t>(
            finput0->get_outputs()[0], freduce->get_outputs()[0], true);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);
    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::f32, aaa, _arg_("out0", datatypes::f32, {16, 16}),
            _arg_("inp0", datatypes::f32, {16, 16}),
            _arg_("inp1", datatypes::f32, {10, 16, 10, 16})) {
        _bind_(out0, inp0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 16}, /*dim2*/ {0, 16}})},
                {tensor_slice(out0, {/*dim1*/ {0, 16}, /*dim2*/ {0, 16}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[0]}, {aaa->params_[2]});

    ///// Expected func:
    auto rd_lanes = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    auto add_lanes = fadd->get_lanes();
    _function_(datatypes::f32, bbb, _arg_("out0", datatypes::f32, {16, 16}),
            _arg_("inp0", datatypes::f32, {16, 16}),
            _arg_("inp1", datatypes::f32, {10, 16, 10, 16})) {
        _bind_(out0, inp0, inp1);
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0, 0, 0}, {}, true);
        auto out0_tptr = builder::tensor_ptr(out0, {0, 0}, {}, true);
        _tensor_(_fuse_buf, datatypes::f32, {16UL, 16UL});
        auto fuse_buf_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(j, 0, 16) {
            _for_(jj, 0, 16, static_cast<int>(rd_lanes)) {
                _var_(reduce_value, sc_data_type_t::f32(rd_lanes));
                reduce_value = make_expr<constant_node>(
                        0.0f, sc_data_type_t::f32(rd_lanes));
                _for_(i, 0, 10UL) {
                    _for_(ii, 0, 10UL) {
                        reduce_value = builder::make_add(
                                inp1_tptr[span_t({i, j, ii, jj}, rd_lanes)],
                                reduce_value);
                    }
                }
                fuse_buf_tptr[span_t({j, jj}, rd_lanes)] = reduce_value;
            }
        }
        _for_(ii, 0, 16) {
            _for_(jj, 0, 16, static_cast<int>(add_lanes)) {
                out0_tptr[span_t({ii, jj}, add_lanes)]
                        = inp0_tptr[span_t({ii, jj}, add_lanes)]
                        + fuse_buf_tptr[span_t({ii, jj}, add_lanes)];
            }
        }
        _return_(123);
    }

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(aaa, bbb, false));
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestPreOpFusionReduceOpKeepDims) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    sc_dims arg_dims_1 = {100, 256};
    auto finput0 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::f32));
    auto finput1 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::f32));
    auto freduce = fusion.make<reduce_sum_op_t>(
            finput1->get_outputs()[0], std::vector<int> {1}, true);
    auto fadd = fusion.make<add_op_t>(
            finput0->get_outputs()[0], freduce->get_outputs()[0], true);
    auto fout = fusion.make<output_op>(fadd->get_outputs()[0]);

    EXPECT_EQ(fadd->get_outputs()[0], fout->get_inputs()[0]);

    _function_(datatypes::f32, aaa, _arg_("out0", datatypes::f32, {100, 256}),
            _arg_("inp0", datatypes::f32, {100, 256}),
            _arg_("inp1", datatypes::f32, {100, 256})) {
        _bind_(out0, inp0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 256}})},
                {tensor_slice(out0, {/*dim1*/ {20, 10}, /*dim2*/ {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[0]}, {aaa->params_[2]});

    ///// Expected func:
    auto rd_lanes = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    auto add_lanes = fadd->get_lanes();
    _function_(datatypes::f32, bbb, _arg_("out0", datatypes::f32, {100, 256}),
            _arg_("inp0", datatypes::f32, {100, 256}),
            _arg_("inp1", datatypes::f32, {100, 256})) {
        _bind_(out0, inp0, inp1);
        auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
        auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0}, {}, true);
        _tensor_(_fuse_buf, datatypes::f32, {100UL, 1UL});
        auto fuse_buf_tptr = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
        _for_(ii, 0, 10) {
            _var_(reduce_value, sc_data_type_t::f32(rd_lanes));
            reduce_value = make_expr<constant_node>(
                    0.0f, sc_data_type_t::f32(rd_lanes));
            _for_(jj, 0, 256UL, static_cast<int>(rd_lanes)) {
                reduce_value = builder::make_add(
                        inp1_tptr[span_t({ii, jj}, rd_lanes)], reduce_value);
            }
            fuse_buf_tptr[{ii, 0}] = builder::make_reduce_add(reduce_value);
        }
        _for_(ii, 0, 10) {
            _for_(jj, 0, 256, static_cast<int>(add_lanes)) {
                fuse_buf_tptr
                        = builder::tensor_ptr(_fuse_buf, {0, 0}, {}, true);
                inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
                auto out0_tptr = builder::tensor_ptr(out0, {20, 0}, {}, true);
                out0_tptr[span_t({ii, jj}, add_lanes)]
                        = inp0_tptr[span_t({ii, jj}, add_lanes)]
                        + builder::make_broadcast(fuse_buf_tptr[{ii, 0}],
                                static_cast<int>(add_lanes));
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestMultiOutputUser) {
    builder::ir_builder_t builder;
    fusion_manager fusion;
    sc_dims arg_dims_1 = {100, 256};
    auto finput0 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::f32));
    auto finput1 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::f32));
    auto finput2 = fusion.make<input_op>(make_tsr(arg_dims_1, datatypes::f32));

    auto fadd = fusion.make<add_op_t>(
            finput0->get_outputs()[0], finput1->get_outputs()[0], true, 0);
    auto fout0 = fusion.make<output_op>(fadd->get_outputs()[0]);
    auto fmul = fusion.make<mul_op_t>(
            fadd->get_outputs()[0], finput2->get_outputs()[0], true);

    auto fout1 = fusion.make<output_op>(fmul->get_outputs()[0]);

    _function_(datatypes::f32, aaa, _arg_("out0", datatypes::f32, {100, 256}),
            _arg_("out1", datatypes::f32, {100, 256}),
            _arg_("inp0", datatypes::f32, {100, 256}),
            _arg_("inp1", datatypes::f32, {100, 256}),
            _arg_("inp2", datatypes::f32, {100, 256})) {
        _bind_(out0, out1, inp0);
        fusion.create_output_fusion_anchor(
                {tensor_slice(inp0, {/*dim1*/ {0, 10}, /*dim2*/ {0, 256}})});
        _return_(123);
    }
    do_commit(fusion, {aaa->params_[0], aaa->params_[1]},
            {aaa->params_[3], aaa->params_[4]});

    int lanes = fadd->get_lanes();
    _function_(datatypes::f32, bbb, _arg_("out0", datatypes::f32, {100, 256}),
            _arg_("out1", datatypes::f32, {100, 256}),
            _arg_("inp0", datatypes::f32, {100, 256}),
            _arg_("inp1", datatypes::f32, {100, 256}),
            _arg_("inp2", datatypes::f32, {100, 256})) {
        _bind_(out0, out1, inp0, inp1, inp2);
        _for_(i, 0, 10, 1) {
            _for_(j, 0, 256, lanes) {
                auto inp0_tptr = builder::tensor_ptr(inp0, {0, 0}, {}, true);
                auto inp1_tptr = builder::tensor_ptr(inp1, {0, 0}, {}, true);
                auto out0_tptr = builder::tensor_ptr(out0, {0, 0}, {}, true);
                out0_tptr[span_t({i, j}, lanes)]
                        = inp0_tptr[span_t({i, j}, lanes)]
                        + inp1_tptr[span_t({i, j}, lanes)];
                out0_tptr = builder::tensor_ptr(out0, {i, j}, {}, true);
                auto out1_tptr = builder::tensor_ptr(out1, {i, j}, {}, true);
                auto inp2_tptr = builder::tensor_ptr(inp2, {i, j}, {}, true);
                out1_tptr[span_t({0, 0}, lanes)]
                        = out0_tptr[span_t({0, 0}, lanes)]
                        * inp2_tptr[span_t({0, 0}, lanes)];
            }
        }
        _return_(123);
    }

    CMP_SIMPLIFIED_IR(aaa, bbb);
}

TEST(GCCore_CPU_fuse_mgr_cpp, TestFusionManagerDeclareAndShrinkForTensorPtr) {
    sc_graph_t mgr;
    // gemm + exp + reduce fusion pattern
    auto input_A = mgr.make_input({graph_tensor::make({16, 384, 64})});
    auto input_B = mgr.make_input({graph_tensor::make({16, 64, 384})});
    auto matmul0 = mgr.make("matmul_core",
            {input_A->get_outputs()[0], input_B->get_outputs()[0]}, {}, {});
    auto cast0 = mgr.make("cast", {matmul0->get_outputs()[0]}, {},
            {{"dtype", datatypes::u8}});
    auto cast1 = mgr.make(
            "cast", {cast0->get_outputs()[0]}, {}, {{"dtype", datatypes::f32}});
    auto tv0 = mgr.make("tensor_view", {cast1->get_outputs()[0]}, {},
            {{"shape", sc_dims {16 * 384, 384}}});
    auto softmax = mgr.make("reduce", {tv0->get_outputs()[0]}, {},
            {{"rd_axis", std::vector<int> {1}}, {"rd_op", 0}});
    auto output = mgr.make_output(softmax->get_outputs());
    layout_propagation(mgr, get_test_ctx());
    fuse_ops(mgr, get_test_ctx());

    auto ir_mod = lower_graph(get_test_ctx(), mgr, {});
    tensor_shrinker_t pass;
    auto func = ir_mod->get_func(
            "matmul_core_cast_cast_tensor_view_reduce_reorder__3");
    ASSERT_TRUE(func);
    auto newbody = pass(func->body_).checked_as<stmts>();
    auto ss = &newbody->seq_;
    ASSERT_TRUE(ss->size() > 1);
    auto cur_loop = ss->at(1).checked_as<for_loop>().get();
    ss = &cur_loop->body_.checked_as<stmts>()->seq_;
    ASSERT_TRUE(!ss->empty());
    auto tsr = ss->at(0).as<define>()->var_.as<tensor>();
    bool name_eq = (tsr->name_.find("reduce_compute") != std::string::npos
            && tsr->name_.find("shr") != std::string::npos);
    EXPECT_TRUE(name_eq);
    bool dim_eq = (get_expr_to_dims(tsr->dims_)
            == sc_dims {1, 1, 1, 64, 1,
                    get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32)});
    EXPECT_TRUE(dim_eq);

    cur_loop = ss->at(2).as<for_loop>().get();
    ASSERT_TRUE(cur_loop);
    ss = &cur_loop->body_.checked_as<stmts>()->seq_;

    auto tsr2 = ss->at(2).as<define>()->var_.as<tensor>();
    bool name_eq2 = (tsr2->name_.find("_cast_") != std::string::npos
            && tsr2->name_.find("shr") != std::string::npos);
    EXPECT_TRUE(name_eq2);
    bool dim_eq2 = (get_expr_to_dims(tsr2->dims_)
            == sc_dims {1, 1, 1, 1,
                    vectorize_step(get_test_ctx(), sc_data_etype::F32)});
    EXPECT_TRUE(dim_eq2);
}

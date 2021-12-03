/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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
#include "batch_matmul.hpp"
#include <memory>
#include <numeric>
#include "templates/batch_matmul.hpp"
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <util/utils.hpp>

namespace sc {
namespace ops {

template <typename T>
static std::vector<T> merge_vec(
        const std::vector<T> &a, const std::vector<T> &b) {
    std::vector<T> result(a);
    for (auto it : b) {
        result.push_back(it);
    }
    return result;
}

static sc_data_type_t infer_out_dtype(
        const std::vector<graph_tensor_ptr> &ins) {
    if (ins.at(0)->details_.dtype_ == datatypes::u8
            || ins.at(0)->details_.dtype_ == datatypes::s8) {
        assert(ins.at(1)->details_.dtype_ == datatypes::s8);
        return datatypes::s32;
    }
    return datatypes::f32;
}
batch_matmul_op_t::batch_matmul_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("batch_matmul", ins, outs, attrs) {
    COMPILE_ASSERT(info_.inputs_.size() == 2, "batch_matmul expects 2 inputs");
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    COMPILE_ASSERT(A_dims.size() == B_dims.size() && A_dims.size() > 2,
            "batch_matmul expects 2 inputs with same size or bigger than 2, "
            "but got "
                    << A_dims.size() << " vs " << B_dims.size());
    sc_dims expected_out_shape = {merge_vec(get_batch_dims(),
            {A_dims[A_dims.size() - 2], B_dims[B_dims.size() - 1]})};

    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this,
                sc_data_format_t(), expected_out_shape, infer_out_dtype(ins)));
    } else {
        COMPILE_ASSERT(
                info_.outputs_.size() == 1, "batch_matmul expects 1 output");
        COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims()
                        == expected_out_shape,
                "Bad out dims");
    }
}

body_generator_ptr batch_matmul_op_t::create_generator() {
    return utils::make_unique<gen_batch_matmul_t>(
            graph::extract_detail_from_tensors(get_inputs()),
            graph::extract_detail_from_tensors(get_outputs()));
}

float batch_matmul_op_t::get_gflop() {
    return create_generator()->get_gflop();
}
sc_dims batch_matmul_op_t::get_batch_dims() {
    return {info_.inputs_[0]->details_.get_plain_dims().begin(),
            info_.inputs_[0]->details_.get_plain_dims().end() - 2};
}

void batch_matmul_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    if (!config_data_) {
        config_data_ = create_generator()->get_default_config(ctx);
    }
    int M_block, N_block, K_block;
    const batch_matmul_config &tcfg
            = *reinterpret_cast<batch_matmul_config *>(config_data_.get());
    M_block = tcfg.M_block;
    N_block = tcfg.N_block;
    K_block = tcfg.K_block;
    in_formats.reserve(2);
    sc_data_type_t B_dtype = info_.inputs_[1]->details_.dtype_;
    if (info_.inputs_[1]
                    ->details_.get_format()
                    .format_code_.is_batch_format()) {
        in_formats.push_back({sc_data_format_t::BMKmk(M_block, K_block)});
        if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
            in_formats.push_back({sc_data_format_t::BNKkn4k(K_block, N_block)});
        } else if (B_dtype == datatypes::bf16) {
            in_formats.push_back({sc_data_format_t::BNKkn2k(K_block, N_block)});
        } else {
            in_formats.push_back({sc_data_format_t::BNKkn(K_block, N_block)});
        }
        out_formats.push_back(
                {sc_data_format_t(format_kinds::BMNmn, {M_block, N_block})});
    } else {
        // runs into special process in bert.
        // ACBDcd * ABDCcd/ABDCcdc = ACBDcd for QK
        // ACBDcd * ACBDcd/ACBDcdc = ACBDcd for V
        in_formats.push_back(
                {sc_data_format_t(format_kinds::ACBDcd, {M_block, K_block})});
        if (info_.inputs_[1]->details_.get_format().format_code_
                == format_kinds::ACBD) {
            if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
                in_formats.push_back({sc_data_format_t(
                        format_kinds::ACBDcdc, {K_block, N_block, 4})});
            } else if (B_dtype == datatypes::bf16) {
                in_formats.push_back({sc_data_format_t(
                        format_kinds::ACBDcdc, {K_block, N_block, 2})});
            } else {
                in_formats.push_back({sc_data_format_t(
                        format_kinds::ACBDcd, {K_block, N_block})});
            }
        } else {
            if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
                in_formats.push_back({sc_data_format_t(
                        format_kinds::ABDCcdc, {K_block, N_block, 4})});
            } else if (B_dtype == datatypes::bf16) {
                in_formats.push_back({sc_data_format_t(
                        format_kinds::ABDCcdc, {K_block, N_block, 2})});
            } else {
                in_formats.push_back({sc_data_format_t(
                        format_kinds::ABDCcd, {K_block, N_block})});
            }
        }
        out_formats.push_back(
                {sc_data_format_t(format_kinds::ACBDcd, {M_block, N_block})});
    }
}

sc_op_ptr batch_matmul_op_t::do_compensations(
        sc_graph_t &mgr, const context_ptr &ctx) {
    need_compensation_ = false;
    // whether we need special compensation for microkernel.
    bool s8s8_compensation = ctx->machine_.cpu_flags_.fAVX512VNNI
            && info_.inputs_[0]->details_.dtype_ == datatypes::s8
            && (!ctx->flags_.brgemm_use_amx_
                    || (ctx->flags_.brgemm_use_amx_
                            && !ctx->machine_.cpu_flags_.fAVX512AMXINT8));

    int bds = get_batch_dims().size();

    auto cur_node = shared_from_this();
    auto s8s8_com = s8s8_compensation ? get_s8s8_compensation(mgr)
                                      : sc_op_ptr(nullptr);
    auto data_com = get_data_compensation(mgr);
    auto weight_com = get_weight_compensation(mgr);
    auto const_com = get_constant_compensation(mgr);

    if (s8s8_com) {
        cur_node = mgr.make("sub",
                {cur_node->get_outputs()[0], s8s8_com->get_outputs()[0]}, {},
                {});
    }
    if (data_com) {
        cur_node = mgr.make("sub",
                {cur_node->get_outputs()[0], data_com->get_outputs()[0]}, {},
                {});
    }
    if (weight_com) {
        cur_node = mgr.make("sub",
                {cur_node->get_outputs()[0], weight_com->get_outputs()[0]}, {},
                {});
    }
    if (const_com) {
        cur_node = mgr.make("add",
                {cur_node->get_outputs()[0], const_com->get_outputs()[0]}, {},
                {});
    }

    return cur_node;
}

sc_op_ptr batch_matmul_op_t::get_s8s8_compensation(sc_graph_t &mgr) {
    auto data = info_.inputs_[0];
    auto weight = info_.inputs_[1];
    auto out = info_.outputs_[0];
    auto cast_node
            = mgr.make("cast", {weight}, {}, {{"dtype", datatypes::s32}});

    auto reshape_dest = out->details_.get_blocking_dims();
    auto reshape_fmt = out->details_.get_format();
    reshape_fmt.blocks_[0] = 1;

    std::vector<int> rdaxis;
    // the blocking judgement should take permuted format like ACBD into
    // consideration, here just use original logic, need refactor in the futrue.
    // (Same in following compensation.)
    bool is_blocking;

    if (data->details_.get_format().format_code_.is_batch_format()) {
        assert(weight->details_.get_format().format_code_.is_batch_format());
        auto weight_blocking_dims = weight->details_.get_blocking_dims();
        int bds = get_batch_dims().size();

        is_blocking = (weight_blocking_dims.size() - bds) == 5;

        assert(weight_blocking_dims.size() - bds == 2
                || weight_blocking_dims.size() - bds == 5);

        rdaxis = std::vector<int> {bds};

        if (is_blocking) {
            reshape_dest[bds] = 1;
            reshape_dest[bds + 2] = 1;
        } else {
            reshape_dest = merge_vec(
                    get_batch_dims(), {1, weight_blocking_dims[bds + 1]});
        }
    } else {
        is_blocking = weight->details_.get_blocking_dims().size() != 4;
        // weight format = ABCD
        rdaxis = {2};
        if (!is_blocking) {
            reshape_dest[2] = 1;
            reshape_fmt.blocks_[0] = 0;
        } else {
            // out format = ACBD (plain out) or ACBDcd (blocking out)
            std::vector<int> rsaxis
                    = out->details_.get_format()
                              .format_code_.collect_blocking_index(
                                      reshape_dest.size()
                                      - get_batch_dims().size());
            for (auto v : rsaxis) {
                reshape_dest[v] = 1;
            }
        }
    }
    auto reduce_node = mgr.make("reduce", cast_node->get_outputs(), {},
            {{"rd_axis", rdaxis}, {"rd_op", 0}, {"keep_dims", true}});
    auto constant_node = mgr.make("constant", {}, {},
            {{"values",
                     std::make_shared<static_data_t>(std::vector<int> {128})},
                    {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}},
                    {"format", sc_data_format_t()}});
    auto mul_node = mgr.make("mul",
            {reduce_node->get_outputs()[0], constant_node->get_outputs()[0]},
            {}, {});
    if (reshape_dest
            == mul_node->get_outputs()[0]->details_.get_blocking_dims()) {
        mul_node->get_outputs()[0]->details_.set_format(
                out->details_.get_format());
        return mul_node;
    }
    auto reshape_node = mgr.make("tensor_view", mul_node->get_outputs(), {},
            {{"shape", reshape_dest}, {"format", reshape_fmt}});
    return reshape_node;
}

sc_op_ptr batch_matmul_op_t::get_data_compensation(sc_graph_t &mgr) {
    auto weight_zero_points
            = attrs_.get_or_else("weight_zero_points", std::vector<int> {0});
    if (weight_zero_points.empty()
            || (std::all_of(weight_zero_points.begin(),
                    weight_zero_points.end(), [](int i) { return i == 0; }))) {
        return nullptr;
    }
    assert(weight_zero_points.size() == 1);
    auto data = info_.inputs_[0];
    auto weight = info_.inputs_[1];
    auto out = info_.outputs_[0];

    auto cast_node = mgr.make("cast", {data}, {}, {{"dtype", datatypes::s32}});
    auto constant_node = mgr.make("constant", {}, {},
            {{"values",
                     std::make_shared<static_data_t>(
                             std::vector<int> {weight_zero_points[0]})},
                    {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}},
                    {"format", sc_data_format_t()}});
    std::vector<int> rdaxis;

    if (data->details_.get_format().format_code_.is_batch_format()) {
        // BMKmk or BMK
        auto data_blocking_dims = data->details_.get_blocking_dims();
        int bds = get_batch_dims().size();
        assert(data_blocking_dims.size() - bds == 2
                || data_blocking_dims.size() - bds == 4);
        rdaxis = std::vector<int> {bds + 1};
    } else {
        // data format = ACBD or ACBDcd
        rdaxis = std::vector<int> {3};
    }

    auto reduce_node = mgr.make("reduce", cast_node->get_outputs(), {},
            {{"rd_axis", rdaxis}, {"rd_op", 0}, {"keep_dims", true}});
    auto mul_node = mgr.make("mul",
            {reduce_node->get_outputs()[0], constant_node->get_outputs()[0]},
            {}, {});
    return mul_node;
}

sc_op_ptr batch_matmul_op_t::get_weight_compensation(sc_graph_t &mgr) {
    auto data_zero_points
            = attrs_.get_or_else("data_zero_points", std::vector<int> {0});
    if (data_zero_points.empty()
            || (std::all_of(data_zero_points.begin(), data_zero_points.end(),
                    [](int i) { return i == 0; }))) {
        return nullptr;
    }
    assert(data_zero_points.size() == 1);
    auto data = info_.inputs_[0];
    auto weight = info_.inputs_[1];
    auto out = info_.outputs_[0];
    auto cast_node
            = mgr.make("cast", {weight}, {}, {{"dtype", datatypes::s32}});

    auto reshape_dest = out->details_.get_blocking_dims();
    auto reshape_fmt = out->details_.get_format();
    reshape_fmt.blocks_[0] = 1;

    std::vector<int> rdaxis;
    bool is_blocking;

    if (data->details_.get_format().format_code_.is_batch_format()) {
        auto data_blocking_dims = data->details_.get_blocking_dims();
        auto weight_blocking_dims = weight->details_.get_blocking_dims();
        int bds = get_batch_dims().size();

        is_blocking = (weight_blocking_dims.size() - bds) == 5;

        assert(weight_blocking_dims.size() - bds == 2
                || weight_blocking_dims.size() - bds == 5);

        rdaxis = std::vector<int> {bds};

        if (is_blocking) {
            reshape_dest[bds] = 1;
            reshape_dest[bds + 2] = 1;
        } else {
            reshape_dest = merge_vec(
                    get_batch_dims(), {1, weight_blocking_dims[bds + 1]});
        }
    } else {
        is_blocking = weight->details_.get_blocking_dims().size() != 4;
        // weight format = ABCD
        rdaxis = {2};
        if (!is_blocking) {
            reshape_dest[2] = 1;
            reshape_fmt.blocks_[0] = 0;
        } else {
            // out format = ACBD (plain out) or ACBDcd (blocking out)
            std::vector<int> rsaxis
                    = out->details_.get_format()
                              .format_code_.collect_blocking_index(
                                      reshape_dest.size()
                                      - get_batch_dims().size());
            for (auto v : rsaxis) {
                reshape_dest[v] = 1;
            }
        }
    }
    auto reduce_node = mgr.make("reduce", cast_node->get_outputs(), {},
            {{"rd_axis", rdaxis}, {"rd_op", 0}, {"keep_dims", true}});
    auto constant_node = mgr.make("constant", {}, {},
            {{"values",
                     std::make_shared<static_data_t>(
                             std::vector<int> {data_zero_points[0]})},
                    {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}},
                    {"format", sc_data_format_t()}});
    auto mul_node = mgr.make("mul",
            {reduce_node->get_outputs()[0], constant_node->get_outputs()[0]},
            {}, {});
    if (reshape_dest
            == mul_node->get_outputs()[0]->details_.get_blocking_dims()) {
        mul_node->get_outputs()[0]->details_.set_format(
                out->details_.get_format());
        return mul_node;
    }
    auto reshape_node = mgr.make("tensor_view", mul_node->get_outputs(), {},
            {{"shape", reshape_dest}, {"format", reshape_fmt}});
    return reshape_node;
}

sc_op_ptr batch_matmul_op_t::get_constant_compensation(sc_graph_t &mgr) {
    auto data_zero_points
            = attrs_.get_or_else("data_zero_points", std::vector<int> {0});
    auto weight_zero_points
            = attrs_.get_or_else("weight_zero_points", std::vector<int> {0});
    if (data_zero_points.empty() || weight_zero_points.empty()) {
        return nullptr;
    }
    if ((std::all_of(data_zero_points.begin(), data_zero_points.end(),
                [](int i) { return i == 0; }))
            || (std::all_of(weight_zero_points.begin(),
                    weight_zero_points.end(), [](int i) { return i == 0; }))) {
        return nullptr;
    }
    assert(data_zero_points.size() == 1 && weight_zero_points.size() == 1);

    auto K_orig = info_.inputs_[0]->details_.get_plain_dims().at(
            info_.inputs_[0]->details_.get_plain_dims().size() - 1);

    int K = static_cast<int>(K_orig);
    auto constant_node = mgr.make("constant", {}, {},
            {{"values",
                     std::make_shared<static_data_t>(std::vector<int> {
                             data_zero_points[0] * weight_zero_points[0] * K})},
                    {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}},
                    {"format", sc_data_format_t()}});

    return constant_node;
}

} // namespace ops
// fix-me: (lowering) remove "new"
OP_REGISTER(::sc::ops::batch_matmul_op_t, batch_matmul)
} // namespace sc

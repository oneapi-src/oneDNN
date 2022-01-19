/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#include "matmul_core.hpp"
#include <memory>
#include <numeric>
#include "templates/matmul_core.hpp"
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
matmul_core_op_t::matmul_core_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("matmul_core", ins, outs, attrs) {
    COMPILE_ASSERT(info_.inputs_.size() == 2, "matmul_core expects 2 inputs");
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    COMPILE_ASSERT(A_dims.size() >= 2 && B_dims.size() >= 2,
            "matmul_core expects each input size equal or bigger than 2 , but "
            "got " << A_dims.size());
    sc_dims expected_out_shape = {merge_vec(get_batch_dims(),
            {A_dims[A_dims.size() - 2], B_dims[B_dims.size() - 1]})};

    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this,
                sc_data_format_t(), expected_out_shape, infer_out_dtype(ins)));
    } else {
        COMPILE_ASSERT(
                info_.outputs_.size() == 1, "matmul_core expects 1 output");
        COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims()
                        == expected_out_shape,
                "Bad out dims");
    }
    // record padded_K of input A for matmul_core
    attrs_["temp.padded_A_K"] = std::make_shared<VConst>();
}

body_generator_ptr matmul_core_op_t::create_generator() {
    return utils::make_unique<gen_matmul_core_t>(
            graph::extract_detail_from_tensors(get_inputs()),
            graph::extract_detail_from_tensors(get_outputs()));
}

float matmul_core_op_t::get_gflop() {
    return create_generator()->get_gflop();
}

sc_dims matmul_core_op_t::get_batch_dims() {
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    return A_dims.size() > B_dims.size()
            ? sc_dims {A_dims.begin(), A_dims.end() - 2}
            : sc_dims {B_dims.begin(), B_dims.end() - 2};
}

void matmul_core_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    if (!config_data_) {
        config_data_ = create_generator()->get_default_config(ctx);
    }
    int M_block, N_block, K_block;
    const matmul_core_config_t &tcfg
            = *reinterpret_cast<matmul_core_config_t *>(config_data_.get());
    M_block = tcfg.M_block;
    N_block = tcfg.N_block;
    K_block = tcfg.K_block;
    in_formats.reserve(2);
    sc_data_type_t B_dtype = info_.inputs_[1]->details_.dtype_;
    if (info_.inputs_[0]->details_.get_plain_dims().size() == 2) {
        in_formats.push_back({sc_data_format_t::MKmk(M_block, K_block)});
        if (info_.inputs_[1]->details_.get_plain_dims().size() == 2) {
            // 2dx2d matmul
            if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
                in_formats.push_back(
                        {sc_data_format_t::NKkn4k(K_block, N_block)});
            } else if (B_dtype == datatypes::bf16) {
                in_formats.push_back(
                        {sc_data_format_t::NKkn2k(K_block, N_block)});
            } else {
                in_formats.push_back(
                        {sc_data_format_t::NKkn(K_block, N_block)});
            }
            out_formats.push_back(
                    {sc_data_format_t(format_kinds::MNmn, {M_block, N_block})});
        } else {
            // 2dxNd matmul (N>2)
            if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
                in_formats.push_back(
                        {sc_data_format_t::BNKkn4k(K_block, N_block)});
            } else if (B_dtype == datatypes::bf16) {
                in_formats.push_back(
                        {sc_data_format_t::BNKkn2k(K_block, N_block)});
            } else {
                in_formats.push_back(
                        {sc_data_format_t::BNKkn(K_block, N_block)});
            }
            out_formats.push_back({sc_data_format_t(
                    format_kinds::BMNmn, {M_block, N_block})});
        }
    } else {
        if (info_.inputs_[0]
                        ->details_.get_format()
                        .format_code_.is_batch_format()) {
            in_formats.push_back({sc_data_format_t::BMKmk(M_block, K_block)});
            if (info_.inputs_[1]->details_.get_plain_dims().size() == 2) {
                // Ndx2d matmul (N>2)
                if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
                    in_formats.push_back(
                            {sc_data_format_t::NKkn4k(K_block, N_block)});
                } else if (B_dtype == datatypes::bf16) {
                    in_formats.push_back(
                            {sc_data_format_t::NKkn2k(K_block, N_block)});
                } else {
                    in_formats.push_back(
                            {sc_data_format_t::NKkn(K_block, N_block)});
                }
            } else {
                // NdxMd matmul (N>2, M>2)
                if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
                    in_formats.push_back(
                            {sc_data_format_t::BNKkn4k(K_block, N_block)});
                } else if (B_dtype == datatypes::bf16) {
                    in_formats.push_back(
                            {sc_data_format_t::BNKkn2k(K_block, N_block)});
                } else {
                    in_formats.push_back(
                            {sc_data_format_t::BNKkn(K_block, N_block)});
                }
            }
            out_formats.push_back({sc_data_format_t(
                    format_kinds::BMNmn, {M_block, N_block})});
        } else {
            // runs into special process in bert.
            // ACBDcd * ABDCcd/ABDCcdc = ACBDcd for QK
            // ACBDcd * ACBDcd/ACBDcdc = ACBDcd for V
            in_formats.push_back({sc_data_format_t(
                    format_kinds::ACBDcd, {M_block, K_block})});
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
            out_formats.push_back({sc_data_format_t(
                    format_kinds::ACBDcd, {M_block, N_block})});
        }
    }

    // To calculate padded K of input A
    auto pad_K_num = utils::divide_and_ceil(
            info_.inputs_[0]->details_.get_plain_dims().back(), K_block);
    attrs_["temp.padded_A_K"].get<std::shared_ptr<VConst>>()->var_
            = pad_K_num * K_block;
}

sc_op_ptr matmul_core_op_t::do_compensations(
        sc_graph_t &mgr, const context_ptr &ctx) {
    need_compensation_ = false;
    // whether we need special compensation for microkernel.
    bool s8s8_compensation = ctx->machine_.cpu_flags_.fAVX512VNNI
            && info_.inputs_[0]->details_.dtype_ == datatypes::s8
            && (!ctx->flags_.brgemm_use_amx_
                    || (ctx->flags_.brgemm_use_amx_
                            && !ctx->machine_.cpu_flags_.fAVX512AMXINT8));

    auto cur_node = shared_from_this();

    auto data_com = get_data_compensation(mgr);
    auto s8s8_weight_com
            = get_s8s8_and_weight_compensation(mgr, s8s8_compensation);
    auto const_com = get_constant_compensation(mgr);

    if (data_com) {
        cur_node = mgr.make("sub",
                {cur_node->get_outputs()[0], data_com->get_outputs()[0]}, {},
                {});
    }

    if (s8s8_weight_com[0]) {
        cur_node = mgr.make("sub",
                {cur_node->get_outputs()[0],
                        s8s8_weight_com[0]->get_outputs()[0]},
                {}, {});
    }
    if (s8s8_weight_com[1]) {
        cur_node = mgr.make("sub",
                {cur_node->get_outputs()[0],
                        s8s8_weight_com[1]->get_outputs()[0]},
                {}, {});
    }
    if (const_com) {
        cur_node = mgr.make("add",
                {cur_node->get_outputs()[0], const_com->get_outputs()[0]}, {},
                {});
    }

    return cur_node;
}

sc_op_ptr matmul_core_op_t::get_data_compensation(sc_graph_t &mgr) {
    auto weight_zero_points
            = attrs_.get_or_else("weight_zero_points", std::vector<int> {0});
    if (weight_zero_points.empty()
            || (std::all_of(weight_zero_points.begin(),
                    weight_zero_points.end(), [](int i) { return i == 0; }))) {
        return nullptr;
    }
    auto data = info_.inputs_[0];
    auto cast_node = mgr.make("cast", {data}, {}, {{"dtype", datatypes::s32}});

    std::shared_ptr<static_data_t> weight_zero_points_ptr
            = std::make_shared<static_data_t>(weight_zero_points);
    sc_dims const_plain_dims;
    sc_data_format_t const_format;
    if (weight_zero_points.size() == 1) {
        // per tensor
        const_plain_dims = {1};
    } else {
        // per channel
        COMPILE_ASSERT(0,
                "matmul_core does not support per channel weight zero points "
                "compensation yet");
        auto weight = info_.inputs_[1];
        auto weight_plain_dims = weight->details_.get_plain_dims();
        assert(weight_plain_dims.back()
                == static_cast<int64_t>(weight_zero_points.size()));
        const_plain_dims = {1, weight_plain_dims.back()};
        const_format = info_.inputs_[1]->details_.get_format();
    }
    auto constant_node = mgr.make("constant", {}, {},
            {{"values", weight_zero_points_ptr}, {"dtype", datatypes::s32},
                    {"plain_dims", const_plain_dims},
                    {"format", const_format}});
    // K is reduce axis
    std::vector<int> rdaxis
            = {static_cast<int>(data->details_.get_plain_dims().size()) - 1};

    auto reduce_node = mgr.make("reduce", cast_node->get_outputs(), {},
            {{"rd_axis", rdaxis}, {"rd_op", 0}, {"keep_dims", true}});
    auto mul_node = mgr.make("mul",
            {reduce_node->get_outputs()[0], constant_node->get_outputs()[0]},
            {}, {});
    if (data->details_.get_plain_dims().size() < get_batch_dims().size() + 2) {
        sc_dims unsqueeze_shape(get_batch_dims().size() + 2
                        - data->details_.get_plain_dims().size(),
                1);
        sc_dims reshape_dest
                = merge_vec(unsqueeze_shape, data->details_.get_plain_dims());
        reshape_dest.at(reshape_dest.size() - 1) = 1;
        auto reshape_fmt = info_.outputs_[0]->details_.get_format();
        auto reshape_node = mgr.make("tensor_view", mul_node->get_outputs(),
                {graph_tensor::make(reshape_dest, sc_data_format_t::BMK(),
                        mul_node->get_outputs()[0]->details_.dtype_)},
                {{"shape", reshape_dest}, {"format", reshape_fmt}});
        return reshape_node;
    }
    return mul_node;
}

std::vector<sc_op_ptr> matmul_core_op_t::get_s8s8_and_weight_compensation(
        sc_graph_t &mgr, bool s8s8_compensation) {
    auto data_zero_points
            = attrs_.get_or_else("data_zero_points", std::vector<int> {0});
    bool weight_compensation = !data_zero_points.empty()
            && !(std::all_of(data_zero_points.begin(), data_zero_points.end(),
                    [](int i) { return i == 0; }));
    std::vector<sc_op_ptr> nodes = {nullptr, nullptr};
    if (!s8s8_compensation && !weight_compensation) { return nodes; }

    auto weight = info_.inputs_[1];
    auto cast_node
            = mgr.make("cast", {weight}, {}, {{"dtype", datatypes::s32}});

    // K is reduce axis
    std::vector<int> rdaxis
            = {static_cast<int>(weight->details_.get_plain_dims().size()) - 2};
    auto reduce_node = mgr.make("reduce", cast_node->get_outputs(), {},
            {{"rd_axis", rdaxis}, {"rd_op", 0}, {"keep_dims", true}});

    if (weight_compensation) {
        std::shared_ptr<static_data_t> data_zero_points_ptr
                = std::make_shared<static_data_t>(data_zero_points);
        sc_dims const_plain_dims;
        sc_data_format_t const_format;
        if (data_zero_points.size() == 1) {
            // per tensor
            const_plain_dims = {1};
        } else {
            // per channel
            COMPILE_ASSERT(0,
                    "matmul_core does not support per channel data zero points "
                    "compensation yet");
            auto data = info_.inputs_[0];
            auto data_plain_dims = data->details_.get_plain_dims();
            size_t bds = get_batch_dims().size();
            assert(data_plain_dims[bds]
                    == static_cast<int64_t>(data_zero_points.size()));
            const_plain_dims = {data_plain_dims[bds], 1};
            const_format = info_.inputs_[0]->details_.get_format();
        }
        auto constant_node = mgr.make("constant", {}, {},
                {{"values", data_zero_points_ptr}, {"dtype", datatypes::s32},
                        {"plain_dims", const_plain_dims},
                        {"format", const_format}});
        nodes[0] = mgr.make("mul",
                {reduce_node->get_outputs()[0],
                        constant_node->get_outputs()[0]},
                {}, {});
        if (weight->details_.get_plain_dims().size()
                < get_batch_dims().size() + 2) {
            sc_dims unsqueeze_shape(get_batch_dims().size() + 2
                            - weight->details_.get_plain_dims().size(),
                    1);
            sc_dims reshape_dest = merge_vec(
                    unsqueeze_shape, weight->details_.get_plain_dims());
            reshape_dest.at(reshape_dest.size() - 2) = 1;
            auto reshape_fmt = info_.outputs_[0]->details_.get_format();
            nodes[0] = mgr.make("tensor_view", nodes[0]->get_outputs(),
                    {graph_tensor::make(reshape_dest, sc_data_format_t::BMK(),
                            nodes[0]->get_outputs()[0]->details_.dtype_)},
                    {{"shape", reshape_dest}, {"format", reshape_fmt}});
        }
    }

    if (s8s8_compensation) {
        auto s8_constant_node = mgr.make("constant", {}, {},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<int> {128})},
                        {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}},
                        {"format", sc_data_format_t()}});
        nodes[1] = mgr.make("mul",
                {reduce_node->get_outputs()[0],
                        s8_constant_node->get_outputs()[0]},
                {}, {});
        if (weight->details_.get_plain_dims().size()
                < get_batch_dims().size() + 2) {
            sc_dims unsqueeze_shape(get_batch_dims().size() + 2
                            - weight->details_.get_plain_dims().size(),
                    1);
            sc_dims reshape_dest = merge_vec(
                    unsqueeze_shape, weight->details_.get_plain_dims());
            reshape_dest.at(reshape_dest.size() - 2) = 1;
            auto reshape_fmt = info_.outputs_[0]->details_.get_format();
            nodes[1] = mgr.make("tensor_view", nodes[1]->get_outputs(),
                    {graph_tensor::make(reshape_dest, sc_data_format_t::BMK(),
                            nodes[1]->get_outputs()[0]->details_.dtype_)},
                    {{"shape", reshape_dest}, {"format", reshape_fmt}});
        }
    }
    return nodes;
}

sc_op_ptr matmul_core_op_t::get_constant_compensation(sc_graph_t &mgr) {
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
    COMPILE_ASSERT(
            data_zero_points.size() == 1 && weight_zero_points.size() == 1,
            "matmul_core does not support per channel data/weight zero points "
            "compensation yet");

    auto K_orig = info_.inputs_[0]->details_.get_plain_dims().at(
            info_.inputs_[0]->details_.get_plain_dims().size() - 1);

    int K = static_cast<int>(K_orig);
    COMPILE_ASSERT(attrs_.has_key("temp.padded_A_K"),
            "No related VConst set, which maybe cause correctness error")
    auto constant_node = mgr.make("constant", {}, {},
            {{"values",
                     std::make_shared<static_data_t>(std::vector<int> {
                             data_zero_points[0] * weight_zero_points[0] * K})},
                    {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}},
                    {"format", sc_data_format_t()},
                    {"temp.val/var",
                            data_zero_points[0] * weight_zero_points[0]},
                    {"temp.var", attrs_["temp.padded_A_K"]}});
    return constant_node;
}

} // namespace ops
OP_REGISTER(::sc::ops::matmul_core_op_t, matmul_core)
} // namespace sc

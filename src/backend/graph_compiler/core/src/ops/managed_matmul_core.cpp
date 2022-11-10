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
#include "managed_matmul_core.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include "matmul_core.hpp"
#include "templates/managed_matmul_core.hpp"
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/graph_map.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <runtime/config.hpp>
#include <util/reflection.hpp>
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

managed_matmul_core_op_t::managed_matmul_core_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("managed_matmul_core", ins, outs, attrs) {
    COMPILE_ASSERT(
            info_.inputs_.size() == 2, "managed_matmul_core expects 2 inputs");
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    COMPILE_ASSERT(A_dims.size() == 2 && B_dims.size() == 2,
            "managed_matmul_core only supports 2d cases yet");
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

std::vector<int> managed_matmul_core_op_t::query_prefetch(
        const context_ptr &ctx, bool is_global,
        const std::vector<tensor_slice> &ins) {
    if (!is_global) { return {0, 1}; }
    auto gen = create_generator();
    auto gen_ptr = static_cast<gen_managed_matmul_core_t *>(gen.get());
    if (gen_ptr->is_okay_to_prefetch(
                *config_data_.get_as<managed_matmul_core_config_t>(),
                is_global)) {
        return {1};
    } else {
        return {};
    }
}

void managed_matmul_core_op_t::generate_prefetcher_body_for_tensor(
        const context_ptr &ctx, const std::vector<expr> &func_args,
        const std::vector<expr> &ins, const std::vector<int> &indices) {
    auto gen = create_generator();
    static_cast<gen_managed_matmul_core_t *>(gen.get())
            ->generate_prefetcher_body_for_tensor(ctx,
                    *config_data_.get_as<managed_matmul_core_config_t>(),
                    func_args, ins, indices);
}

body_generator_ptr managed_matmul_core_op_t::create_generator() {
    auto mat_gen = utils::make_unique<gen_managed_matmul_core_t>(this,
            graph::extract_detail_from_tensors(get_inputs()),
            graph::extract_detail_from_tensors(get_outputs()));
    mat_gen->bwise_fusion_ = attrs_.get_or_else(op_attr_key::bwise_fuse, false);
    return std::move(mat_gen);
}

float managed_matmul_core_op_t::get_gflop() {
    return create_generator()->get_gflop();
}

sc_dims managed_matmul_core_op_t::get_batch_dims() const {
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    return A_dims.size() > B_dims.size()
            ? sc_dims {A_dims.begin(), A_dims.end() - 2}
            : sc_dims {B_dims.begin(), B_dims.end() - 2};
}

void managed_matmul_core_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    const sc_dims &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &A_blocking_dims
            = info_.inputs_[0]->details_.get_blocking_dims();
    const sc_dims &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &B_blocking_dims
            = info_.inputs_[1]->details_.get_blocking_dims();
    const sc_dims &C_dims = info_.outputs_[0]->details_.get_plain_dims();
    const sc_dim M = A_dims[A_dims.size() - 2];
    const sc_dim K = A_dims.back();
    const sc_dim N = B_dims.back();

    in_formats.reserve(2);
    sc_data_type_t A_dtype = info_.inputs_[0]->details_.dtype_;
    sc_data_type_t B_dtype = info_.inputs_[1]->details_.dtype_;
    sc_data_format_t A_format = info_.inputs_[0]->details_.get_format();
    sc_data_format_t B_format = info_.inputs_[1]->details_.get_format();
    auto gen_ptr = create_generator();
    auto gen = static_cast<gen_managed_matmul_core_t *>(gen_ptr.get());
    int iim_block = gen->iim_block_;
    int iin_block = gen->iin_block_;
    int iik_block = gen->iik_block_;
    if (!config_data_) {
        if (attrs_.get_or_else("transposed_a", false)) {
            config_data_ = gen->get_default_transposed_a_config(ctx);
        } else {
            config_data_ = create_generator()->get_default_config(ctx);
        }
    }

    // constant check
    bool constant_A = false, constant_B = false;
    if (info_.inputs_[0]->producer_owner_->isa<constant_op_t>()
            || info_.inputs_[0]->producer_owner_->attrs_.get_or_else(
                    "constant", const_kind::not_const)) {
        constant_A = true;
    } else {
        bool constant_A_parents = true;
        for (const auto &input :
                info_.inputs_[0]->producer_owner_->get_inputs()) {
            auto parent_node = input->producer_owner_;
            constant_A_parents &= (parent_node->attrs_.get_or_else(
                                           "constant", const_kind::not_const)
                    || parent_node->isa<constant_op_t>());
        }
        constant_A = constant_A_parents
                && !info_.inputs_[0]->producer_owner_->get_inputs().empty();
    }

    if (info_.inputs_[1]->producer_owner_->isa<constant_op_t>()
            || info_.inputs_[1]->producer_owner_->attrs_.get_or_else(
                    "constant", const_kind::not_const)) {
        constant_B = true;
    } else {
        bool constant_B_parents = true;
        for (const auto &input :
                info_.inputs_[1]->producer_owner_->get_inputs()) {
            auto parent_node = input->producer_owner_;
            constant_B_parents &= (parent_node->attrs_.get_or_else(
                                           "constant", const_kind::not_const)
                    || parent_node->isa<constant_op_t>());
        }
        constant_B = constant_B_parents
                && !info_.inputs_[1]->producer_owner_->get_inputs().empty();
    }

    if (A_dims.size() == 2) {
        if (A_dtype == datatypes::bf16 && A_format == sc_data_format_t::NK()) {
            in_formats.push_back({sc_data_format_t::NK()});
        } else {
            if (constant_A || A_format.is_blocking() || M % iim_block
                    || K % iik_block) {
                in_formats.push_back(
                        {sc_data_format_t::MKmk(iim_block, iik_block)});
            } else {
                in_formats.push_back({sc_data_format_t::MK()});
            }
        }
    } else {
        COMPILE_ASSERT(0, "managed_matmul_core only supports 2d yet");
    }
    if (B_dims.size() == 2) {
        if (utils::is_one_of(B_dtype, datatypes::u8, datatypes::s8)) {
            in_formats.push_back(
                    {sc_data_format_t::NKkn4k(iik_block, iin_block)});
        } else if (B_dtype == datatypes::bf16) {
            // do vnni reorder in template for transposed matmul
            if ((B_format == sc_data_format_t::MK()
                        && attrs_.get_or_else("transposed_a", false))
                    || (B_format == sc_data_format_t::NK()
                            && attrs_.get_or_else("transposed_b", false)
                            && M <= 512)) {
                // do pre-op fusion for NK -> NKkn2k only when shapes are small.
                in_formats.push_back({B_format});
            } else {
                in_formats.push_back(
                        {sc_data_format_t::NKkn2k(iik_block, iin_block)});
            }
        } else {
            if (constant_B || B_format.is_blocking() || K % iik_block
                    || N % iin_block) {
                in_formats.push_back(
                        {sc_data_format_t::NKkn(iik_block, iin_block)});
            } else {
                in_formats.push_back({sc_data_format_t::KN()});
            }
        }
    } else {
        COMPILE_ASSERT(0, "managed_matmul_core only supports 2d yet");
    }
    if (constant_B || M % iim_block || N % iin_block) {
        out_formats.push_back(
                {sc_data_format_t(sc_data_format_kind_t::get_2dblocking_by_dims(
                                          C_dims.size(), false),
                        {iim_block, iin_block})});
    } else {
        if (attrs_.get_or_else("transposed_b", false)) {
            out_formats.push_back(
                    {sc_data_format_t::get_plain_by_dims(C_dims.size())});
        } else {
            out_formats.push_back(
                    {sc_data_format_t(
                             sc_data_format_kind_t::get_2dblocking_by_dims(
                                     C_dims.size(), false),
                             {iim_block, iin_block}),
                            sc_data_format_t::get_plain_by_dims(
                                    C_dims.size())});
            in_formats[0].push_back(in_formats[0][0]);
            in_formats[1].push_back(in_formats[1][0]);
        }
    }

    // To calculate padded K of input A
    auto pad_K_num = utils::divide_and_ceil(
            info_.inputs_[0]->details_.get_plain_dims().back(), iik_block);
    attrs_["temp.padded_A_K"].get<std::shared_ptr<VConst>>()->var_
            = pad_K_num * iik_block;
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

sc_op_ptr managed_matmul_core_op_t::do_compensations(
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

sc_op_ptr managed_matmul_core_op_t::get_data_compensation(sc_graph_t &mgr) {
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
                {graph_tensor::make(reshape_dest, sc_data_format_t(),
                        mul_node->get_outputs()[0]->details_.dtype_)},
                {{"shape", reshape_dest}, {"format", reshape_fmt}});
        return reshape_node;
    }
    return mul_node;
}

std::vector<sc_op_ptr>
managed_matmul_core_op_t::get_s8s8_and_weight_compensation(
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
                    {graph_tensor::make(reshape_dest, sc_data_format_t(),
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
                    {graph_tensor::make(reshape_dest, sc_data_format_t(),
                            nodes[1]->get_outputs()[0]->details_.dtype_)},
                    {{"shape", reshape_dest}, {"format", reshape_fmt}});
        }
    }
    return nodes;
}

sc_op_ptr managed_matmul_core_op_t::get_constant_compensation(sc_graph_t &mgr) {
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

shape_rl_vec managed_matmul_core_op_t::get_dynamic_shape_relations() const {
    return matmul_core_op_t::get_shape_relations_impl(
            get_inputs()[0]->details_.get_plain_dims(),
            get_inputs()[1]->details_.get_plain_dims(),
            get_outputs()[0]->details_.get_plain_dims());
}

sc_dims managed_matmul_core_op_t::get_bwise_fuse_shrink_dims() {
    // Currently fordbid N-axis fuse, skip check weight
    auto out_fmt = info_.outputs_[0]->details_.get_format(),
         inp_fmt = info_.inputs_[0]->details_.get_format();
    auto output_dims = info_.outputs_[0]->details_.get_blocking_dims();
    int bs_size = get_batch_dims().size();

    auto out_p2b_map = out_fmt.format_code_.collect_p2b_mapping(),
         inp_p2b_map = inp_fmt.format_code_.collect_p2b_mapping();

    COMPILE_ASSERT(out_p2b_map.size() >= 2,
            "Matmul core output should at least have MN dimension")
    // validate input according shrinked output graph tensor
    int cnt = 0;
    for (; cnt < bs_size; cnt++) {
        auto plain_pos = out_fmt.format_code_.get(cnt);
        if (out_p2b_map[plain_pos].front() != cnt
                || inp_p2b_map[plain_pos].front() != cnt)
            break;
    }
    return {output_dims.begin(), output_dims.begin() + cnt};
}

void managed_matmul_core_op_t::collect_shrinked_lt_map(
        int bw_size, gt2gt_map &bw_lt_map) {
    // set output
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_outputs()[0], bw_size);
    auto &out_plain_dims
            = bw_lt_map.get(get_outputs()[0])->details_.get_plain_dims();
    auto old_inp_dims = get_inputs()[0]->details_.get_plain_dims();
    auto old_wei_dims = get_inputs()[1]->details_.get_plain_dims();
    // MK
    sc_dims inp_plain_dims = {
            out_plain_dims.at(out_plain_dims.size() - 2), old_inp_dims.back()};
    // KN
    sc_dims wei_plain_dims
            = {old_wei_dims.at(old_wei_dims.size() - 2), out_plain_dims.back()};

    int bs_out = out_plain_dims.size() - 2;
    int bs_inp = old_inp_dims.size() - 2;
    int bs_wei = old_wei_dims.size() - 2;

    for (int i = 1; i <= bs_out; i++) {
        if (i <= bs_inp) {
            inp_plain_dims.insert(inp_plain_dims.begin(),
                    out_plain_dims.at(out_plain_dims.size() - 2 - i));
        }
        if (i <= bs_wei) {
            wei_plain_dims.insert(wei_plain_dims.begin(),
                    out_plain_dims.at(out_plain_dims.size() - 2 - i));
        }
    }
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_inputs()[0], inp_plain_dims);
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_inputs()[1], wei_plain_dims);
}

void managed_matmul_core_op_t::collect_shrinked_axis_map(
        int bw_size, gt2axis_map &bw_axis_map) {
    auto ins = get_inputs()[0], wei = get_inputs()[1], out = get_outputs()[0];
    int bs_inp = get_inputs()[0]->details_.get_plain_dims().size() - 2;
    int bs_wei = get_inputs()[1]->details_.get_plain_dims().size() - 2;
    int bs_out = get_outputs()[0]->details_.get_plain_dims().size() - 2;
    auto get_idx = [](const graph_tensor_ptr &gt) {
        std::vector<int> batch;
        auto fmt = gt->details_.get_format();
        auto p2b_map = fmt.format_code_.collect_p2b_mapping();
        for (size_t i = 0; i < p2b_map.size() - 2; i++) {
            batch.insert(batch.end(), p2b_map[i].begin(), p2b_map[i].end());
        }
        std::vector<std::vector<int>> ret;
        ret.emplace_back(batch);
        ret.emplace_back(p2b_map[p2b_map.size() - 2]);
        ret.emplace_back(p2b_map[p2b_map.size() - 1]);
        return ret;
    };

    auto BMK = get_idx(ins), BKN = get_idx(wei), BMN = get_idx(out);

    auto get_idx_type = [](const std::vector<std::vector<int>> &map, int idx) {
        for (size_t i = 0; i < map.size(); i++) {
            if (std::find(map[i].begin(), map[i].end(), idx) != map[i].end())
                return static_cast<int>(i);
        }
        assert(0); // should never goto here
        return -1;
    };
    std::vector<int> BMK_idx, BKN_idx;
    for (int i = 0; i < bw_size; i++) {
        int idx_type = get_idx_type(BMN, i);
        if (idx_type == 0) {
            auto find_iter = std::find(BMN[0].begin(), BMN[0].end(), i);
            int batch_idx = std::distance(BMN[0].begin(), find_iter);
            // reversed position
            int batch_idx_rev = bs_out - batch_idx;
            if (batch_idx_rev <= bs_inp) {
                BMK_idx.emplace_back(BMK[idx_type][bs_inp - batch_idx_rev]);
            } else {
                BMK_idx.emplace_back(-1);
            }
            if (batch_idx_rev <= bs_wei) {
                BKN_idx.emplace_back(BKN[idx_type][bs_wei - batch_idx_rev]);
            } else {
                BKN_idx.emplace_back(-1);
            }
        } else if (idx_type == 1) {
            BMK_idx.emplace_back(BMK[idx_type][0]);
            BKN_idx.emplace_back(-1);
        } else if (idx_type == 2) {
            BMK_idx.emplace_back(-1);
            BKN_idx.emplace_back(BKN[idx_type][0]);
        }
    }

    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, ins, BMK_idx);
    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, wei, BKN_idx);
    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, out, bw_size);
}

void managed_matmul_core_op_t::infer_binding_axis(bound_axis_map &bdax_map) {
    infer_matmul_binding_axis(this, bdax_map);
}
void managed_matmul_core_op_t::pre_binding_axis(bound_axis_map &bdax_map) {
    pre_matmul_binding_axis(this, bdax_map);
}

} // namespace ops
OP_REGISTER(::sc::ops::managed_matmul_core_op_t, managed_matmul_core)
} // namespace sc

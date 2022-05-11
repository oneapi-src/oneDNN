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
#include "convolution.hpp"
#include <memory>
#include "templates/conv_bwd.hpp"
#include "templates/conv_fwd.hpp"
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

namespace sc {
namespace ops {

sc_data_type_t conv_fwd_op_t::infer_out_dtype(
        const sc_data_type_t &input_dtype, const sc_data_type_t &weight_dtype) {
    if (utils::is_one_of(input_dtype, datatypes::u8, datatypes::s8)
            && weight_dtype == datatypes::s8) {
        return datatypes::s32;
    } else {
        // both f32 and bf16 inputs generate f32 output
        return datatypes::f32;
    }
}

sc_dims conv_fwd_op_t::infer_out_dims(const sc_dims &input_dims,
        const sc_dims &weight_dims, const sc_dims &padding,
        const sc_dims &stride) {
    int ndims = input_dims.size();
    const bool is_3d = (ndims == 5);
    COMPILE_ASSERT(utils::is_one_of(static_cast<int>(input_dims.size()), 4, 5),
            "wrong input dims, expected to be 4D or 5D input, but got "
                    << input_dims.size() << "D.");
    COMPILE_ASSERT(utils::is_one_of(static_cast<int>(weight_dims.size()), 4, 5)
                    && (weight_dims.size() == input_dims.size()),
            "wrong weight dims, only support 4D or 5D weights, but got "
                    << weight_dims.size() << "D.");
    COMPILE_ASSERT(is_3d
                    ? utils::is_one_of(static_cast<int>(padding.size()), 1, 3)
                    : utils::is_one_of(static_cast<int>(padding.size()), 1, 2),
            "wrong padding dims, should be 1D or 2D for 2D conv, and 1D or 3D "
            "for 3D conv, but got "
                    << padding.size() << "D for in " << (is_3d ? 3 : 2)
                    << "D conv.");
    COMPILE_ASSERT(is_3d
                    ? utils::is_one_of(static_cast<int>(stride.size()), 1, 3)
                    : utils::is_one_of(static_cast<int>(stride.size()), 1, 2),
            "wrong stride dims, should be 1D or 2D for 2D conv, and 1D or 3D "
            "for 3D conv, but got "
                    << stride.size() << "D for in " << (is_3d ? 3 : 2)
                    << "D conv.");
    sc_dims pad_dims(ndims - 2, padding[0]);
    if (padding.size() > 1) {
        pad_dims[ndims - 4] = padding[ndims - 4];
        pad_dims[ndims - 3] = padding[ndims - 3];
    }
    sc_dims stride_dims(ndims - 2, stride[0]);
    if (stride.size() > 1) {
        stride_dims[ndims - 4] = stride[ndims - 4];
        stride_dims[ndims - 3] = stride[ndims - 3];
    }
    auto calc_out_shapes = [](int i, int k, int p, int s) {
        auto r = (i + 2 * p - k) / s + 1;
        return r;
    };

    sc_dims out_dims(ndims);
    out_dims[0] = input_dims[0];
    out_dims[1] = weight_dims[0];
    for (int i = 2; i < ndims; ++i) {
        out_dims[i] = calc_out_shapes(input_dims[i], weight_dims[i],
                pad_dims[i - 2], stride_dims[i - 2]);
    }

    return out_dims;
}

void conv_fwd_op_t::check_dtypes(const sc_data_type_t &data_dtype,
        const sc_data_type_t &weight_dtype, const sc_data_type_t &out_dtype) {
    if (utils::is_one_of(data_dtype, datatypes::u8, datatypes::s8)) {
        COMPILE_ASSERT((weight_dtype == datatypes::s8),
                "weight_dtype expected to be s8 when data_dtype is u8/s8, but "
                "got " << weight_dtype
                       << ".");
        if (out_dtype != datatypes::undef) {
            COMPILE_ASSERT((out_dtype == datatypes::s32),
                    "out_dtype expected to be s32 when data and weights are in "
                    "u8|s8, but got "
                            << out_dtype << ".");
        }
    } else if (data_dtype == datatypes::bf16) {
        COMPILE_ASSERT((weight_dtype == datatypes::bf16),
                "weight_dtype expected to be bf16 when data_dtype is bf16, but "
                "got " << weight_dtype
                       << ".");
    } else {
        COMPILE_ASSERT(((data_dtype == datatypes::f32)
                               && (weight_dtype == datatypes::f32)
                               && (out_dtype == datatypes::undef
                                       || out_dtype == datatypes::f32)),
                "All datatypes are expected to be f32, but got data_dtype: "
                        << data_dtype << ", weight_dtype: " << weight_dtype
                        << ", out_dtype: " << out_dtype << ".");
    }
}

conv_fwd_op_t::conv_fwd_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("conv_fwd", ins, outs, attrs) {
    COMPILE_ASSERT(info_.inputs_.size() == 2, "conv expects 2 inputs");
    auto &indims = info_.inputs_[0]->details_.get_plain_dims();
    auto &weightdims = info_.inputs_[1]->details_.get_plain_dims();
    ndims_ = indims.size();
    auto expected_out_shape = infer_out_dims(indims, weightdims,
            attrs_.get<sc_dims>("paddings"), attrs_.get<sc_dims>("strides"));
    auto &data_dtype = info_.inputs_[0]->details_.dtype_;
    auto &weight_dtype = info_.inputs_[1]->details_.dtype_;
    if (info_.outputs_.empty()) {
        check_dtypes(data_dtype, weight_dtype);
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this,
                sc_data_format_t(), expected_out_shape,
                infer_out_dtype(data_dtype, weight_dtype)));
    } else {
        COMPILE_ASSERT(info_.outputs_.size() == 1, "conv expects 1 output");
        COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims()
                        == expected_out_shape,
                "Bad output shape for conv");
        check_dtypes(
                data_dtype, weight_dtype, info_.outputs_[0]->details_.dtype_);
    }
}

body_generator_ptr conv_fwd_op_t::create_generator() {
    auto &stride = attrs_.get<sc_dims>("strides");
    auto &padding = attrs_.get<sc_dims>("paddings");
    return utils::make_unique<gen_conv_fwd_t>(stride, padding,
            graph::extract_detail_from_tensors(get_inputs()),
            graph::extract_detail_from_tensors(get_outputs()));
}

float conv_fwd_op_t::get_gflop() {
    return create_generator()->get_gflop();
}

void conv_fwd_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    if (!config_data_) {
        config_data_ = create_generator()->get_default_config(ctx);
    }
    const conv_fwd_config_t &tcfg = *config_data_.get_as<conv_fwd_config_t>();
    in_formats.reserve(2);
    int C_block = tcfg.C_block;
    int K_block = tcfg.K_block;
    const bool is_3d = ndims_ == 5;
    in_formats.push_back({is_3d ? sc_data_format_t::NCDHWc(C_block)
                                : sc_data_format_t::NCHWc(C_block)});
    COMPILE_ASSERT(info_.inputs_.size() == 2,
            "conv expects 2 inputs, but got " << info_.inputs_.size()
                                              << " inputs.");
    const auto src_dtype = info_.inputs_[0]->details_.dtype_;
    const auto wei_dtype = info_.inputs_[1]->details_.dtype_;
    if (utils::is_one_of(src_dtype, datatypes::u8, datatypes::s8)
            && wei_dtype == datatypes::s8) {
        in_formats.push_back(
                {is_3d ? sc_data_format_t::KCDRSck4c(C_block, K_block)
                       : sc_data_format_t::KCRSck4c(C_block, K_block)});
    } else if (src_dtype == datatypes::bf16 && wei_dtype == datatypes::bf16) {
        in_formats.push_back(
                {is_3d ? sc_data_format_t::KCDRSck2c(C_block, K_block)
                       : sc_data_format_t::KCRSck2c(C_block, K_block)});
    } else {
        in_formats.push_back(
                {is_3d ? sc_data_format_t::KCDRSck(C_block, K_block)
                       : sc_data_format_t::KCRSck(C_block, K_block)});
    }
    // for output format
    out_formats.push_back({is_3d ? sc_data_format_t::NCDHWc(K_block)
                                 : sc_data_format_t::NCHWc(K_block)});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

conv_bwd_op_t::conv_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("conv_bwd", ins, outs, attrs) {
    COMPILE_ASSERT(info_.inputs_.size() == 2, "conv expects 2 inputs");
    COMPILE_ASSERT(info_.outputs_.size() == 1, "conv expects 1 output");
}

body_generator_ptr conv_bwd_op_t::create_generator() {
    auto &stride = attrs_.get<sc_dims>("strides");
    auto &padding = attrs_.get<sc_dims>("paddings");
    return utils::make_unique<gen_conv_bwd>(stride, padding,
            graph::extract_detail_from_tensors(get_inputs()),
            graph::extract_detail_from_tensors(get_outputs()));
}

float conv_bwd_op_t::get_gflop() {
    return create_generator()->get_gflop();
}

void conv_bwd_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    if (!config_data_) {
        config_data_ = create_generator()->get_default_config(ctx);
    }
    const conv_fwd_config_t &tcfg = *config_data_.get_as<conv_fwd_config_t>();
    in_formats.reserve(get_inputs().size());
    in_formats.push_back({sc_data_format_t::NCHWc(tcfg.K_block)});
    in_formats.push_back(
            {sc_data_format_t::KCRSck(tcfg.C_block, tcfg.K_block)});
    out_formats.push_back(
            {sc_data_format_t(format_kinds::NKHWk, {tcfg.C_block})});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

sc_op_ptr conv_fwd_op_t::do_compensations(
        sc_graph_t &mgr, const context_ptr &ctx) {
    need_compensation_ = false;
    return shared_from_this();
}

} // namespace ops
OP_REGISTER(::sc::ops::conv_fwd_op_t, conv_fwd)
OP_REGISTER(::sc::ops::conv_bwd_op_t, conv_bwd)
} // namespace sc

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
#include "convolution.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "templates/conv1x1_backprop_data.hpp"
#include "templates/conv1x1_backprop_weight.hpp"
#include "templates/convNxN_backprop_data.hpp"
#include "templates/convNxN_backprop_weight.hpp"
#include "templates/conv_bwd.hpp"
#include "templates/conv_fwd.hpp"
#include "templates/conv_rl.hpp"
#include "templates/nested_conv1x1_backprop_data.hpp"
#include "templates/nested_conv1x1_backprop_weight.hpp"
#include "templates/nested_convNxN_backprop_data.hpp"
#include "templates/nested_convNxN_backprop_weight.hpp"
#include "templates/nested_conv_fwd.hpp"
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/mixed_partition.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/transform/loop_transform.hpp>
#include <ops/templates/utils.hpp>
#include <runtime/config.hpp>
#include <runtime/dynamic_dispatch/ops/config.hpp>
#include <runtime/dynamic_dispatch/ops/runtime_op_info.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/math_utils.hpp>
#include <util/reflection.hpp>
#include <util/simple_math.hpp>
#include <util/utils.hpp>

SC_MODULE(ops.convolution);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

sc_data_type_t conv_fwd_core_op_t::infer_out_dtype(
        const sc_data_type_t &input_dtype, const sc_data_type_t &weight_dtype) {
    if (utils::is_one_of(input_dtype, datatypes::u8, datatypes::s8)
            && weight_dtype == datatypes::s8) {
        return datatypes::s32;
    } else {
        // both f32 and bf16 inputs generate f32 output
        return datatypes::f32;
    }
}

void conv_fwd_core_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    bool is_weight_constant
            = info_.inputs_[1]->producer_owner_->isa<constant_op_t>()
            || info_.inputs_[1]->producer_owner_->attrs_.get_or_else(
                    "constant", const_kind::not_const)
            || info_.inputs_[1]->attrs_.get_or_else(
                    "constant", const_kind::not_const);
    if (attrs_.has_key("inverse_filter")
            || !attrs_.get_or_else("image_affinity", true)
            || !is_weight_constant || is_dynamic()) {
        stat_map.append_ops_by_status(this, infer_status_code::FAIL);
        return;
    }
    int C_block = 1;
    int K_block = 1;
    int tile_p = 1;
    const auto use_rl = attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING);

    if (config_data_) {
        if (use_nested_conv_fwd_generator()) {
            const nested_conv_fwd_config_t &tcfg
                    = *config_data_.get_as<nested_conv_fwd_config_t>();
            tile_p = tcfg.im_h_block;
        } else if (use_rl == ops::rl_kind::FULL_LOWERING) {
            const conv_fwd_rl_config_t &tcfg
                    = *config_data_.get_as<conv_fwd_rl_config_t>();
            tile_p = tcfg.brgemm_m;
        } else {
            const conv_fwd_config_t &tcfg
                    = *config_data_.get_as<conv_fwd_config_t>();
            tile_p = tcfg.tile_p;
        }
    }
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);
    // assume input is known
    if (known_ranges_map[0].empty() && known_ranges_map[1].empty()) {
        stat_map.append_ops_by_status(this, infer_status_code::RETRY);
        return;
    }
    auto inp_plain_size = get_inputs()[0]->details_.get_plain_dims().size();
    auto wei_plain_dim = use_rl != ops::rl_kind::NO_LOWERING
            ? attrs_.get<sc_dims>("origin_wei_plain_dims")
            : get_inputs()[1]->details_.get_plain_dims();

    auto wei_plain_size = wei_plain_dim.size();
    auto inp_dims = get_inputs()[0]->details_.get_blocking_dims(),
         wei_dims = get_inputs()[1]->details_.get_blocking_dims(),
         out_dims = get_outputs()[0]->details_.get_blocking_dims();
    const int num_threads = runtime_config_t::get().get_num_threads();

    auto &data_dtype = info_.inputs_[0]->details_.dtype_;
    auto &weight_dtype = info_.inputs_[1]->details_.dtype_;
    auto is_int8 = utils::is_one_of(data_dtype, datatypes::u8, datatypes::s8);
    auto L2_cache_size
            = get_default_context()->machine_.cpu_flags_.getDCacheSize(2);

    auto get_slice_size = [](const slice_range &ranges,
                                  const int dtype_size = 1) {
        auto total_size = dtype_size;
        for (auto &range : ranges) {
            auto second = do_cast_and_fold(range.second);
            if (second.isa<constant>()) {
                total_size *= get_const_as_int(second.checked_as<constant>());
            } else {
                return -1;
            }
        }
        return total_size;
    };
    auto input_slice_size = known_ranges_map[0].empty()
            ? -1
            : get_slice_size(
                    known_ranges_map[0][0], utils::get_sizeof_type(data_dtype));
    auto weight_size = math_utils::get_dims_product(wei_dims)
            * utils::get_sizeof_type(weight_dtype);
    auto can_fit_in_L2_cache = input_slice_size > 0
            && (input_slice_size + weight_size < L2_cache_size);
    if (config_data_ && inp_dims[0] % num_threads == 0
            && wei_plain_dim.size() == 4 && wei_plain_dim[2] == 1
            && wei_plain_dim[3] == 1
            && can_fit_in_L2_cache) { // 1x1 NH-wise fusion
        auto in_p2b_map = get_inputs()[0]
                                  ->details_.get_format()
                                  .format_code_.collect_p2b_mapping();
        auto out_p2b_map = get_outputs()[0]
                                   ->details_.get_format()
                                   .format_code_.collect_p2b_mapping();
        slice_range inp_slice, wei_slice, out_slice;
        inp_slice.resize(inp_dims.size());
        wei_slice.resize(wei_dims.size());
        out_slice.resize(out_dims.size());
        if (!known_ranges_map[0].empty()) {
            slice_range inp_tmp;
            inp_tmp = known_ranges_map[0][0];
            if (!slice_full_on_axis(inp_dims, inp_tmp,
                        in_p2b_map[1])) { // full on C
                stat_map.append_ops_by_status(this, infer_status_code::RETRY);
                return;
            }
            if (!slice_full_on_axis(inp_dims, inp_tmp,
                        in_p2b_map[3])) { // full on W
                stat_map.append_ops_by_status(this, infer_status_code::RETRY);
                return;
            }
            if (!slice_divisible_by_factor(
                        inp_tmp, {in_p2b_map[2].back()}, tile_p)
                    || !slice_larger_than_bound_on_axis(inp_tmp, in_p2b_map[2],
                            tile_p,
                            data_dtype == datatypes::f32
                                    ? 1
                                    : 2)) { // dividable on H
                stat_map.append_ops_by_status(this, infer_status_code::RETRY);
                return;
            }

            for (auto i = 0UL; i < in_p2b_map.size(); i++) {
                if (i != 1 && i != 3) {
                    auto blocking_axis = in_p2b_map[i];
                    for (auto &ax : blocking_axis) {
                        inp_slice[ax] = known_ranges_map[0][0][ax];
                        out_slice[ax] = known_ranges_map[0][0][ax];
                    }
                }
            }
        } else {
            std::vector<int> plain_axis_required = {0, 2}; // N,H
            for (auto plain_ax : plain_axis_required) {
                for (unsigned i = 0; i < in_p2b_map[plain_ax].size(); i++) {
                    auto ax = in_p2b_map[plain_ax][i];
                    inp_slice[ax] = std::make_pair(
                            expr(0), dim2unsigned(inp_dims[ax]));
                }
                for (unsigned i = 0; i < out_p2b_map[plain_ax].size(); i++) {
                    auto ax = out_p2b_map[plain_ax][i];
                    out_slice[ax] = std::make_pair(
                            expr(0), dim2unsigned(out_dims[ax]));
                }
            }
        }
        if (!known_ranges_map[1].empty()) {
            auto wei_slice = known_ranges_map[1][0];
            std::vector<int> required_axis;
            for (unsigned i = 0; i < wei_dims.size(); i++) {
                required_axis.emplace_back(i);
            }
            if (!slice_full_on_axis(wei_dims, wei_slice, required_axis)) {
                stat_map.append_ops_by_status(this, infer_status_code::RETRY);
                return;
            }
        }
        for (unsigned i = 0; i < wei_dims.size(); i++) {
            wei_slice[i] = std::make_pair(expr(0), dim2unsigned(wei_dims[i]));
        }
        std::vector<int> plain_axis_required = {1, 3}; // C, W
        for (auto plain_ax : plain_axis_required) {
            for (unsigned i = 0; i < in_p2b_map[plain_ax].size(); i++) {
                auto ax = in_p2b_map[plain_ax][i];
                inp_slice[ax]
                        = std::make_pair(expr(0), dim2unsigned(inp_dims[ax]));
            }
            for (unsigned i = 0; i < out_p2b_map[plain_ax].size(); i++) {
                auto ax = out_p2b_map[plain_ax][i];
                out_slice[ax]
                        = std::make_pair(expr(0), dim2unsigned(out_dims[ax]));
            }
        }
        fsmap.get(get_inputs()[0]) = slice_range_list {inp_slice};
        fsmap.get(get_inputs()[1]) = slice_range_list {wei_slice};
        fsmap.get(get_outputs()[0]) = slice_range_list {out_slice};
    } else {
        slice_range inp_slice, wei_slice, out_slice;
        if (!known_ranges_map[0].empty()) {
            slice_range inp_tmp;
            inp_tmp = known_ranges_map[0][0];
            std::vector<int> required_axis;
            for (unsigned i = 1; i < inp_dims.size(); i++) {
                required_axis.emplace_back(i);
            }
            if (!slice_full_on_axis(inp_dims, inp_tmp, required_axis)) {
                stat_map.append_ops_by_status(this, infer_status_code::RETRY);
                return;
            }
            inp_slice.emplace_back(known_ranges_map[0][0][0]);
            out_slice.emplace_back(known_ranges_map[0][0][0]);
        } else {
            inp_slice.emplace_back(
                    std::make_pair(expr(0), dim2unsigned(inp_dims[0])));
        }
        if (!known_ranges_map[1].empty()) {
            auto wei_slice = known_ranges_map[1][0];
            std::vector<int> required_axis;
            for (unsigned i = 0; i < wei_dims.size(); i++) {
                required_axis.emplace_back(i);
            }
            if (!slice_full_on_axis(wei_dims, wei_slice, required_axis)) {
                stat_map.append_ops_by_status(this, infer_status_code::RETRY);
                return;
            }
        }
        for (unsigned i = 1; i < inp_dims.size(); i++) {
            inp_slice.emplace_back(
                    std::make_pair(expr(0), dim2unsigned(inp_dims[i])));
        }
        for (unsigned i = 0; i < wei_dims.size(); i++) {
            wei_slice.emplace_back(
                    std::make_pair(expr(0), dim2unsigned(wei_dims[i])));
        }
        for (unsigned i = 1; i < out_dims.size(); i++) {
            out_slice.emplace_back(
                    std::make_pair(expr(0), dim2unsigned(out_dims[i])));
        }
        fsmap.get(get_inputs()[0]) = slice_range_list {inp_slice};
        fsmap.get(get_inputs()[1]) = slice_range_list {wei_slice};
        fsmap.get(get_outputs()[0]) = slice_range_list {out_slice};
    }
}

void conv_fwd_core_op_t::infer_out_tensor_details() {
    if (!info_.outputs_[0]->details_.get_plain_dims().empty()) return;
    auto &cur_plain_dims = info_.outputs_[0]->details_.get_plain_dims();
    auto &indims = info_.inputs_[0]->details_.get_plain_dims();
    auto weightdims = info_.inputs_[1]->details_.get_plain_dims();
    if (attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING)
            > ops::rl_kind::NO_LOWERING) {
        weightdims = attrs_.get<sc_dims>("origin_wei_plain_dims");
    }

    auto &pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    auto &pads_end = attrs_.has_key("pads_end")
            ? attrs_.get<sc_dims>("pads_end")
            : attrs_.get<sc_dims>("paddings");
    auto expected_out_shape = infer_out_dims(get_owner_graph(), indims,
            weightdims, pads_begin, pads_end, attrs_.get<sc_dims>("strides"),
            get_dilations(attrs_), attrs_);
    if (!cur_plain_dims.empty() && !is_dynamic()) {
        COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims()
                        == expected_out_shape,
                "Bad output shape for conv");
    } else {
        info_.outputs_[0]->details_.set_plain_dims(expected_out_shape);
    }
}

sc_dims conv_fwd_core_op_t::infer_out_dims(sc_graph_t &owner_graph,
        const sc_dims &input_dims, const sc_dims &weight_dims,
        const sc_dims &pads_begin, const sc_dims &pads_end,
        const sc_dims &stride, const sc_dims &dilation,
        const any_map_t &attrs) {
    int ndims = input_dims.size();
    const bool is_1d = (ndims == 3);
    const bool is_3d = (ndims == 5);
    sc_dims wei_dims = weight_dims;
    COMPILE_ASSERT(
            utils::is_one_of(static_cast<int>(input_dims.size()), 3, 4, 5),
            "wrong input dims, expected to be 3D, 4D or 5D input, but got "
                    << input_dims.size() << "D.");
    if (attrs.get_or_else("use_rl", ops::rl_kind::NO_LOWERING)
            > ops::rl_kind::NO_LOWERING) {
        wei_dims = attrs.get<sc_dims>("origin_wei_plain_dims");
    }

    COMPILE_ASSERT(utils::is_one_of(static_cast<int>(wei_dims.size()), 3, 4, 5)
                    && (wei_dims.size() == input_dims.size()),
            "wrong weight dims, only support 4D or 5D weights, but got "
                    << wei_dims.size() << "D.");
    COMPILE_ASSERT(
            is_3d ? utils::is_one_of(static_cast<int>(pads_begin.size()), 1, 3)
                    : is_1d ? utils::is_one_of(
                              static_cast<int>(pads_begin.size()), 1, 1)
                            : utils::is_one_of(
                                    static_cast<int>(pads_begin.size()), 1, 2),
            "wrong pads_begin dims, should be 1D or 2D for 2D conv, and 1D or "
            "3D for 3D conv, but got "
                    << pads_begin.size() << "D for in " << (is_3d ? 3 : 2)
                    << "D conv.");
    COMPILE_ASSERT(is_3d
                    ? utils::is_one_of(static_cast<int>(pads_end.size()), 1, 3)
                    : is_1d
                    ? utils::is_one_of(static_cast<int>(pads_end.size()), 1, 1)
                    : utils::is_one_of(static_cast<int>(pads_end.size()), 1, 2),
            "wrong pads_end dims, should be 1D or 2D for 2D conv, and 1D or 3D "
            "for 3D conv, but got "
                    << pads_end.size() << "D for in "
                    << (is_3d                  ? 3
                                       : is_1d ? 1
                                               : 2)
                    << "D conv.");
    COMPILE_ASSERT(is_3d
                    ? utils::is_one_of(static_cast<int>(stride.size()), 1, 3)
                    : is_1d
                    ? utils::is_one_of(static_cast<int>(stride.size()), 1, 2)
                    : utils::is_one_of(static_cast<int>(stride.size()), 1, 2),
            "wrong stride dims, should be 1D or 2D for 2D conv, and 1D or 3D "
            "for 3D conv, but got "
                    << stride.size() << "D for in "
                    << (is_3d                  ? 3
                                       : is_1d ? 1
                                               : 2)
                    << "D conv.");
    COMPILE_ASSERT(is_3d
                    ? utils::is_one_of(static_cast<int>(dilation.size()), 1, 3)
                    : is_1d
                    ? utils::is_one_of(static_cast<int>(dilation.size()), 1, 2)
                    : utils::is_one_of(static_cast<int>(dilation.size()), 1, 2),
            "wrong dilation dims, should be 1D or 2D for 2D conv, and 1D or 3D "
            "for 3D conv, but got "
                    << dilation.size() << "D for in "
                    << (is_3d                  ? 3
                                       : is_1d ? 1
                                               : 2)
                    << "D conv.");
    sc_dims pads_begin_dims(ndims - 2, pads_begin[0]);
    if (pads_begin.size() > 1) { pads_begin_dims = pads_begin; }
    sc_dims pads_end_dims(ndims - 2, pads_end[0]);
    if (pads_end.size() > 1) { pads_end_dims = pads_end; }
    sc_dims stride_dims(ndims - 2, stride[0]);
    if (stride.size() > 1) { stride_dims = stride; }
    sc_dims dilation_dims(ndims - 2, dilation[0]);
    if (dilation.size() > 1) { dilation_dims = dilation; }
    auto calc_out_shapes = [](int i, int k, int pb, int pe, int s, int d) {
        auto r = (i + pb + pe - d * (k - 1) - 1) / s + 1;
        return r;
    };
    sc_dims out_dims(ndims);
    out_dims[0] = input_dims[0];
    out_dims[1] = wei_dims[0];
    for (int i = 2; i < ndims; ++i) {
        if (is_dynamic_dim(input_dims[i]) || is_dynamic_dim(wei_dims[i])
                || is_dynamic_dim(pads_begin_dims[i - 2])
                || is_dynamic_dim(pads_end_dims[i - 2])
                || is_dynamic_dim(stride_dims[i - 2])) {
            out_dims[i] = owner_graph.get_next_dynamic_placeholder();
        } else {
            out_dims[i] = calc_out_shapes(input_dims[i], wei_dims[i],
                    pads_begin_dims[i - 2], pads_end_dims[i - 2],
                    stride_dims[i - 2], dilation_dims[i - 2]);
        }
    }
    if (is_1d && stride.size() > 1) {
        out_dims[2] = attrs.get_or_else("origin_oh", sc_dim(1))
                * attrs.get_or_else("origin_ow", sc_dim(1))
                * (input_dims[2] / attrs.get_or_else("origin_ih", sc_dim(1))
                        / attrs.get_or_else("origin_iw", sc_dim(1)));
    }
    return out_dims;
}

void conv_fwd_core_op_t::infer_auto_pad(sc_graph_t &owner_graph,
        const sc_dims &input_dims, const sc_dims &weight_dims,
        const sc_dims &stride, const sc_dims &dilation, any_map_t &attrs,
        bool is_same_upper) {
    int ndims = input_dims.size();
    sc_dims stride_dims(ndims - 2, stride[0]);
    if (stride.size() > 1) { stride_dims = stride; }
    sc_dims dilation_dims(ndims - 2, dilation[0]);
    if (dilation.size() > 1) { dilation_dims = dilation; }
    sc_dims pads_begin(ndims - 2, 0);
    sc_dims pads_end(ndims - 2, 0);
    auto calc_total_padding = [](int i, int k, int o, int s, int d) {
        return std::max((o - 1) * s + (d * (k - 1) + 1) - i, 0);
    };
    for (int i = 2; i < ndims; ++i) {
        if (is_dynamic_dim(input_dims[i]) || is_dynamic_dim(weight_dims[i])
                || is_dynamic_dim(stride_dims[i - 2])) {
            // pads_begin not necessarily equal to pads_end
            pads_begin[i - 2] = owner_graph.get_next_dynamic_placeholder();
            pads_end[i - 2] = owner_graph.get_next_dynamic_placeholder();
        } else {
            sc_dim output_dim
                    = utils::divide_and_ceil(input_dims[i], stride_dims[i - 2]);
            sc_dim total_pad = calc_total_padding(input_dims[i], weight_dims[i],
                    output_dim, stride_dims[i - 2], dilation_dims[i - 2]);
            if (total_pad % 2 == 0) {
                pads_begin[i - 2] = pads_end[i - 2] = total_pad / 2;
            } else {
                pads_begin[i - 2]
                        = is_same_upper ? total_pad / 2 : total_pad / 2 + 1;
                pads_end[i - 2]
                        = is_same_upper ? total_pad / 2 + 1 : total_pad / 2;
            }
        }
    }
    attrs.set<sc_dims>("pads_begin", pads_begin);
    attrs.set<sc_dims>("pads_end", pads_end);
}

void conv_fwd_core_op_t::check_dtypes(const sc_data_type_t &data_dtype,
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

conv_fwd_core_op_t::conv_fwd_core_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("conv_fwd_core", ins, outs, attrs) {
    COMPILE_ASSERT(info_.inputs_.size() == 2, "conv expects 2 inputs");
    auto &indims = info_.inputs_[0]->details_.get_plain_dims();
    auto weightdims = info_.inputs_[1]->details_.get_plain_dims();
    if (attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING)
            > ops::rl_kind::NO_LOWERING) {
        weightdims = attrs.get<sc_dims>("origin_wei_plain_dims");
    }
    ndims_ = indims.size();
    auto strides = attrs_.get<sc_dims>("strides");
    auto dilations = get_dilations(attrs_);
    // processing padding info
    // if auto_pad is set, original pads_begin/pads_end values will be omitted
    // so we directly overwrite attrs_
    if (attrs_.has_key("auto_pad")) {
        auto pad_type = attrs_.get<std::string>("auto_pad");
        if (pad_type == "VALID") {
            attrs_.set<sc_dims>("pads_begin", sc_dims(ndims_ - 2, 0));
            attrs_.set<sc_dims>("pads_end", sc_dims(ndims_ - 2, 0));
        } else if (pad_type == "SAME_UPPER" || pad_type == "SAME_LOWER") {
            // output spatial dims are equal to input spatial dims
            infer_auto_pad(get_owner_graph(), indims, weightdims, strides,
                    dilations, attrs_, pad_type == "SAME_UPPER");
        }
        attrs_.set<std::string>("auto_pad", "none");
    }
    sc_dims pads_begin, pads_end;
    if (attrs_.has_key("pads_begin")) {
        COMPILE_ASSERT(attrs_.has_key("pads_end"),
                "convolution op shall have pads_begin & pads_end attributes.");
        pads_begin = attrs_.get<sc_dims>("pads_begin");
        pads_end = attrs_.get<sc_dims>("pads_end");
    } else {
        pads_begin = attrs_.get<sc_dims>("paddings");
        pads_end = pads_begin;
    }
    sc_dim groups = attrs_.get_or_else("groups", 1);
    auto ic = indims[1];
    auto oc = weightdims[0];
    COMPILE_ASSERT(ic % groups == 0 && oc % groups == 0,
            "input channel and output channel must both be divisible by "
            "groups, but got ic("
                    << ic << "), oc(" << oc << "), groups(" << groups << ").");
    COMPILE_ASSERT(ic / groups == weightdims[1],
            "ic/g should be equal to filter_dims[1], but got "
                    << ic / groups << " vs " << weightdims[1] << ".");
    COMPILE_ASSERT((groups == 1) || (groups > 1 && ic != groups),
            "depthwise conv is not support yet!");
    auto &data_dtype = info_.inputs_[0]->details_.dtype_;
    auto &weight_dtype = info_.inputs_[1]->details_.dtype_;
    if (info_.outputs_.empty()) {
        check_dtypes(data_dtype, weight_dtype);
        info_.outputs_.emplace_back(
                std::make_shared<graph_tensor>(this, sc_data_format_t(),
                        sc_dims {}, infer_out_dtype(data_dtype, weight_dtype)));
    } else {
        COMPILE_ASSERT(info_.outputs_.size() == 1, "conv expects 1 output");
        check_dtypes(
                data_dtype, weight_dtype, info_.outputs_[0]->details_.dtype_);
    }
}

bool conv_fwd_core_op_t::use_nested_conv_fwd_generator() {
    if (is_dynamic()) return true;
    sc_dim groups = attrs_.get_or_else("groups", 1);
    if (groups > 1) { return false; }
    if (attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING)
            > ops::rl_kind::NO_LOWERING) {
        return false;
    }
    bool use_1d = info_.inputs_[0]->details_.get_plain_dims().size() == 3;
    bool use_nested = attrs_.get_or_else("use_nested", true);
    if (!use_nested && !use_1d) { return false; }
    const sc_dims &pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    const sc_dims &weight_shape = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &data_shape = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &output_shape = info_.outputs_[0]->details_.get_plain_dims();
    auto dilations = get_dilations(attrs_);
    auto has_dilation = std::any_of(
            dilations.begin(), dilations.end(), [](int x) { return x > 1; });
    auto has_pad = std::any_of(
            pads_begin.begin(), pads_begin.end(), [](int x) { return x > 0; });
    auto is_1x1 = std::all_of(weight_shape.begin() + 2, weight_shape.end(),
            [](int x) { return x == 1; });
    auto is_int8 = utils::is_one_of(
            info_.inputs_[0]->details_.dtype_, datatypes::u8, datatypes::s8);
    const int num_threads = runtime_config_t::get().get_num_threads();
    auto dtype_size = utils::get_sizeof_type(info_.inputs_[1]->details_.dtype_);
    bool os_blocking_with_oc_threads = !has_pad
            && output_shape.back() * dtype_size < 32
            && utils::is_one_of(info_.inputs_[0]->details_.dtype_,
                    datatypes::u8, datatypes::s8, datatypes::bf16)
            && !is_1x1 && weight_shape[0] >= 32
            && is_parallel_space_enough(data_shape[0], num_threads);
    // Only support conv 3x3 with os blocking currently
    // TODO(zhicong): the config of nested conv 3x3 with big
    // shape(150x150,300x300, 7x7 oc split) needs to be further tuned
    // only used in throughput mode or real time mode in which the config is
    // well tuned
    auto use_nested_conv = (ndims_ == 4 && !has_pad && !has_dilation && is_int8
                                   && !(data_shape.back() > 56 && !is_1x1)
                                   && !(output_shape.back() <= 7 && !is_1x1)
                                   && !(data_shape[1] % 32 != 0 && is_1x1)
                                   && num_threads / data_shape[0] <= 4
                                   && !os_blocking_with_oc_threads
                                   && !attrs_.get_or_else("use_rl", false))
            || use_1d;
    return use_nested_conv;
}

bool conv_fwd_core_op_t::use_conv1d() {
    // should be 2d case
    sc_dim groups = attrs_.get_or_else("groups", 1);
    if (groups > 1) { return false; }
    const sc_dims &weight_shape = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &data_shape = info_.inputs_[0]->details_.get_plain_dims();
    if (weight_shape.size() != 4UL) { return false; }

    // not support 1x1 with padding case
    const sc_dims &paddings = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    for (auto &p : paddings) {
        if (p != 0) { return false; }
    }

    // only support 1x1 conv
    sc_dim kh = weight_shape[2], kw = weight_shape[3];
    if (kh != 1 || kw != 1) { return false; }

    // flatten pass cannot handle other format
    const auto &format = get_inputs()[0]->details_.get_format();
    if (format != sc_data_format_t::NCHW()
            && format != sc_data_format_t::NHWC()) {
        return false;
    }

    // training case disable
    bool is_weight_constant
            = get_inputs()[1]->producer_owner_->isa<constant_op_t>()
            || get_inputs()[1]->producer_owner_->attrs_.get_or_else(
                    "constant", const_kind::not_const)
            || get_inputs()[1]->attrs_.get_or_else(
                    "constant", const_kind::not_const);
    if (!is_weight_constant) {
        // TODO(zhicong): improve f32/bf16 training fwd config
        return false;
    }

    // big data and small weight
    auto stride = attrs_.get<sc_dims>("strides");
    auto weight_size = math_utils::get_dims_product(weight_shape)
            * utils::get_sizeof_type(info_.inputs_[1]->details_.dtype_);
    auto image_size = math_utils::get_dims_product(data_shape) / data_shape[0]
            * utils::get_sizeof_type(info_.inputs_[0]->details_.dtype_);
    int num_threads = runtime_config_t::get().get_num_threads();
    auto boundry = 5UL;
    bool has_stride = !std::all_of(
            stride.begin(), stride.end(), [](int x) { return x == 1; });
    auto is_int8 = utils::is_one_of(
            info_.inputs_[0]->details_.dtype_, datatypes::u8, datatypes::s8);
    if (image_size / weight_size > boundry && !has_stride
            && data_shape[0] % num_threads == 0) {
        // disable conv1d to use NH fusion
        return false;
    }
    // only used in throughput mode or real time mode in which the config is
    // well tuned
    // TODO(zhicong): further confirm the constraint 32 in other data types
    if ((data_shape[0] == 1 && has_stride && is_int8) || data_shape[1] % 32 != 0
            || weight_shape[0] % 32 != 0) {
        return false;
    }
    return true;
}

body_generator_ptr conv_fwd_core_op_t::create_generator() {
    auto &stride = attrs_.get<sc_dims>("strides");
    auto dilations = get_dilations(attrs_);
    auto &pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    auto &pads_end = attrs_.has_key("pads_end")
            ? attrs_.get<sc_dims>("pads_end")
            : attrs_.get<sc_dims>("paddings");

#define CREATE_GENERATOR(type) \
    utils::make_unique<type>(this, stride, dilations, pads_begin, pads_end, \
            graph::extract_detail_from_tensors(get_inputs()), \
            graph::extract_detail_from_tensors(get_outputs()))
    if (use_nested_conv_fwd_generator()) {
        return CREATE_GENERATOR(gen_nested_conv_fwd_t);
    } else if (attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING)
            == ops::rl_kind::FULL_LOWERING) {
        return CREATE_GENERATOR(gen_conv_fwd_rl_t);
    } else {
        auto ret = CREATE_GENERATOR(gen_conv_fwd_t);
        if (attrs_.get_or_else("inverse_filter", false)) {
            ret->inverse_filter_ = true;
        }
        return std::move(ret);
    }
#undef CREATE_GENERATOR
}

float conv_fwd_core_op_t::get_gflop() {
    return create_generator()->get_gflop();
}

void conv_fwd_core_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    sc_data_format_t data_format, weight_format, out_format;
    sc_data_format_t raw_weight_format
            = info_.inputs_[1]->details_.get_format();
    bool dynamic = is_dynamic();
    COMPILE_ASSERT(info_.inputs_.size() == 2,
            "conv expects 2 inputs, but got " << info_.inputs_.size()
                                              << " inputs.");
    ndims_ = info_.inputs_[0]->details_.get_plain_dims().size();
    auto use_rl = attrs_.get_or_else("use_rl", ops::rl_kind::NO_LOWERING);
    sc_dim groups = attrs_.get_or_else("groups", 1);
    auto weight_plain_dims = use_rl > ops::rl_kind::NO_LOWERING
            ? attrs_.get<sc_dims>("origin_wei_plain_dims")
            : info_.inputs_[1]->details_.get_plain_dims();
    auto input_plain_dims = info_.inputs_[0]->details_.get_plain_dims();
    int ic = input_plain_dims[1] / groups, oc = weight_plain_dims[0] / groups;
    const bool is_data_blocking
            = info_.inputs_[0]->details_.get_format().is_blocking();
    const bool is_weight_blocking
            = info_.inputs_[1]->details_.get_format().is_blocking();

    // nested os blocking conv 3x3 works when use_amx is true
    if (!ctx->use_amx()) { attrs_.set("use_nested", false); }
    if (!config_data_) {
        config_data_ = create_generator()->get_default_config(ctx);
    }
    int C_block = 1;
    int K_block = 1;
    if (use_nested_conv_fwd_generator()) {
        const nested_conv_fwd_config_t &tcfg
                = *config_data_.get_as<nested_conv_fwd_config_t>();
        auto body_gen = create_generator();
        auto gen = static_cast<gen_nested_conv_fwd_t *>(body_gen.get());
        in_formats.reserve(2);
        C_block = tcfg.im_ic_block;
        K_block = tcfg.im_oc_block;
        if (gen->use_conv1d) {
            C_block = gen->im_ic_block_;
            K_block = gen->im_oc_block_;
        }
    } else if (use_rl == ops::rl_kind::FULL_LOWERING) {
        const conv_fwd_rl_config_t &tcfg
                = *config_data_.get_as<conv_fwd_rl_config_t>();
        in_formats.reserve(2);
        K_block = tcfg.brgemm_n;
    } else {
        const conv_fwd_config_t &tcfg
                = *config_data_.get_as<conv_fwd_config_t>();
        in_formats.reserve(2);
        C_block = tcfg.C_block;
        K_block = tcfg.K_block;
    }
    in_formats.resize(2);
    out_formats.resize(1);
    const bool is_3d = ndims_ == 5;
    const bool is_1d = ndims_ == 3;
    const auto src_dtype = info_.inputs_[0]->details_.dtype_;
    const auto wei_dtype = info_.inputs_[1]->details_.dtype_;
    auto dilations = get_dilations(attrs_);
    sc_dims dilation_dims(ndims_ - 2, dilations[0]);
    if (dilations.size() > 1) { dilation_dims = dilations; }
    auto &pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    auto &pads_end = attrs_.has_key("pads_end")
            ? attrs_.get<sc_dims>("pads_end")
            : attrs_.get<sc_dims>("paddings");
    bool has_pad = std::any_of(pads_begin.begin(), pads_begin.end(),
                           [](sc_dim p) { return p > 0; })
            || std::any_of(pads_end.begin(), pads_end.end(),
                    [](sc_dim d) { return d > 0; });
    bool is_weight_constant
            = info_.inputs_[1]->producer_owner_->isa<constant_op_t>()
            || info_.inputs_[1]->producer_owner_->attrs_.get_or_else(
                    "constant", const_kind::not_const)
            || info_.inputs_[1]->attrs_.get_or_else(
                    "constant", const_kind::not_const);

    bool channel_last_support = false;
    auto kh = weight_plain_dims[ndims_ - 2];
    auto kw = weight_plain_dims[ndims_ - 1];
    auto is_1x1 = std::all_of(weight_plain_dims.begin() + 2,
            weight_plain_dims.end(), [](int x) { return x == 1; });
    if (!is_1d) {
        channel_last_support = is_1x1 || ops::is_amx_dtype(ctx, src_dtype)
                || (has_pad && attrs_.get_or_else("inverse_filter", false))
                || use_rl == ops::rl_kind::FULL_LOWERING;
        if (use_rl != ops::rl_kind::NO_LOWERING && groups > 1)
            channel_last_support = false;
    }

    std::string test_format;
    if (attrs_.has_key("temp.test_format")) {
        test_format = attrs_.get<std::string>("temp.test_format");
    }
    bool force_channel_last = test_format == "NHWC" || test_format == "NDHWC"
            || test_format == "NSC";
    bool force_blocking = test_format == "NCHWc" || test_format == "NCDHWc"
            || test_format == "NCSc";
    auto cur_format_set = std::unordered_set<std::vector<sc_data_format_t>>();
    auto cur_dispatch_key_set = dispatch_key_set_t();
    assert(in_formats.size() == 2);
    bool is_first_format = true;
    auto default_block = get_dyn_conv_default_block(is_1x1,
            utils::get_sizeof_type(src_dtype), has_pad,
            src_dtype == datatypes::f32);
    C_block = !dynamic ? C_block
                       : utils::get_blocks(ic, 1, default_block).back();
    K_block = !dynamic ? K_block
                       : utils::get_blocks(oc, 1, default_block).back();
    bool use_channel_last
            = (((channel_last_support && !force_blocking)
                       || (channel_last_support && force_channel_last))
                      && ic % C_block == 0 && oc % K_block == 0)
            || is_dynamic();
    // data layout
    if (use_channel_last) {
        data_format = is_3d ? sc_data_format_t::NDHWC()
                : is_1d     ? sc_data_format_t::NSC()
                            : sc_data_format_t::NHWC();
    } else {
        data_format = is_3d ? sc_data_format_t::NCDHWc(C_block)
                : is_1d
                ? sc_data_format_t::NSC()
                : sc_data_format_t::NCHWc(
                        (use_rl == ops::rl_kind::FULL_LOWERING && groups > 1)
                                ? ic
                                : C_block);
    }
    // weight layout
    if (use_rl == ops::rl_kind::FULL_LOWERING && !dynamic) {
        int brgemm_k = attrs_.get<int>("brgemm_k");
        int brgemm_n = K_block;
        if (utils::is_one_of(src_dtype, datatypes::u8, datatypes::s8)
                && wei_dtype == datatypes::s8) {
            weight_format = sc_data_format_t::KNkn4k(brgemm_k, brgemm_n);
        } else if (src_dtype == datatypes::bf16
                && wei_dtype == datatypes::bf16) {
            weight_format = sc_data_format_t::KNkn2k(brgemm_k, brgemm_n);
        } else {
            COMPILE_ASSERT(0, "Invalid datatype for reduce lowering!");
        }
    } else {
        if (utils::is_one_of(src_dtype, datatypes::u8, datatypes::s8)
                && wei_dtype == datatypes::s8) {
            weight_format = is_3d
                    ? sc_data_format_t::KCDRSck4c(C_block, K_block)
                    : is_1d ? sc_data_format_t::KCSck4c(C_block, K_block)
                            : sc_data_format_t::KCRSck4c(C_block, K_block);
        } else if (src_dtype == datatypes::bf16
                && wei_dtype == datatypes::bf16) {
            weight_format = is_3d
                    ? sc_data_format_t::KCDRSck2c(C_block, K_block)
                    : is_1d ? sc_data_format_t::KCSck2c(C_block, K_block)
                            : sc_data_format_t::KCRSck2c(C_block, K_block);
        } else {
            weight_format = is_3d ? sc_data_format_t::KCDRSck(C_block, K_block)
                    : is_1d       ? sc_data_format_t::KCSck(C_block, K_block)
                                  : sc_data_format_t::KCRSck(C_block, K_block);
        }
    }

    if (is_weight_blocking && dynamic) {
        weight_format = raw_weight_format;
        // follow last layer's config
        if (raw_weight_format.blocks_[0]) {
            C_block = raw_weight_format.blocks_[0];
        }
        if (raw_weight_format.blocks_[1]) {
            K_block = raw_weight_format.blocks_[1];
        }
    }

    // out layout
    if (use_channel_last) {
        out_format = is_3d ? sc_data_format_t::NDHWC()
                : is_1d    ? sc_data_format_t::NSC()
                           : sc_data_format_t::NHWC();
    } else {
        out_format = is_3d ? sc_data_format_t::NCDHWc(K_block)
                : is_1d
                ? sc_data_format_t::NSC()
                : sc_data_format_t::NCHWc(
                        (use_rl == ops::rl_kind::NO_LOWERING && groups > 1)
                                ? oc
                                : K_block);
    }

    std::vector<sc_data_format_t> ret_formats
            = {data_format, weight_format, out_format};
    if (cur_format_set.find(ret_formats) == cur_format_set.end()) {
        in_formats[0].emplace_back(data_format);
        in_formats[1].emplace_back(weight_format);
        out_formats[0].emplace_back(out_format);
        cur_format_set.insert(ret_formats);
    }

    if (dynamic) {
        if (is_first_format) {
            nested_conv_fwd_config_t &tcfg
                    = *config_data_.get_as<nested_conv_fwd_config_t>();
            tcfg.im_ic_block = C_block;
            tcfg.im_oc_block = K_block;
            is_first_format = false;
        }
        std::vector<std::vector<sc_dim>> var_block
                = {{}, {C_block, K_block}, {}};
        op_dispatch_key_t ret_key(var_block, ret_formats);
        cur_dispatch_key_set.set_.insert(ret_key);
        auto &dispatch_key_set = get_dispatch_key_set();
        dispatch_key_set->get_inner_set().insert(
                cur_dispatch_key_set.set_.begin(),
                cur_dispatch_key_set.set_.end());
    }
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void conv_fwd_core_op_t::set_config_by_key(
        const op_dispatch_key_t &key, const context_ptr &ctx) {
    assert(key.var_block_.size() == 3);
    if (use_nested_conv_fwd_generator()) {
        config_data_ = dyn_config_candidates_[key.impl_];
        nested_conv_fwd_config_t &tcfg
                = *config_data_.get_as<nested_conv_fwd_config_t>();
        tcfg.im_ic_block = key.var_block_[1][0];
        tcfg.im_oc_block = key.var_block_[1][1];
        auto cfg = dyn_config_candidates_[key.impl_]
                           .unchecked_get_as<nested_conv_fwd_config_t>();
        dynamic_conv_param.h_threads = cfg->h_threads;
        dynamic_conv_param.oc_threads = cfg->oc_threads;
        dynamic_conv_param.im_h_block = cfg->im_h_block;
        dynamic_conv_param.im_w_block = cfg->im_w_block;
    }
}

sc_op_ptr conv_fwd_core_op_t::copy(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto ret = tunable_op_t::copy(ins, outs, mgr);
    auto conv_fwd = ret->dyn_cast<conv_fwd_core_op_t>();
    conv_fwd->dynamic_conv_param.h_threads = dynamic_conv_param.h_threads;
    conv_fwd->dynamic_conv_param.oc_threads = dynamic_conv_param.oc_threads;
    conv_fwd->dynamic_conv_param.im_h_block = dynamic_conv_param.im_h_block;
    conv_fwd->dynamic_conv_param.im_w_block = dynamic_conv_param.im_w_block;
    return ret;
}

std::vector<int> conv_fwd_core_op_t::get_impl_dispatch_candidates(
        const context_ptr &ctx) {
    return get_dynamic_impl_dispatch_candidates(this, ctx);
}

shape_rl_vec conv_fwd_core_op_t::get_dynamic_shape_relations() const {
    return get_shape_relations_impl(get_inputs()[0]->details_.get_plain_dims(),
            get_inputs()[1]->details_.get_plain_dims(),
            get_outputs()[0]->details_.get_plain_dims(), attrs_);
}

reflection::shared_general_object_t
conv_fwd_core_op_t::get_dynamic_runtime_info() {
    sc_dims pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get_or_else<sc_dims>("paddings", sc_dims(ndims_ - 2, 0));
    sc_dims pads_end = attrs_.has_key("pads_end")
            ? attrs_.get<sc_dims>("pads_end")
            : attrs_.get_or_else<sc_dims>("paddings", sc_dims(ndims_ - 2, 0));

    sc_dims stride = attrs_.get<sc_dims>("strides");
    sc_dims stride_dims(ndims_ - 2, stride[0]);
    if (stride.size() > 1) { stride_dims = stride; }

    auto dyn_info = ndims_ == 5
            ? dyn_conv_fwd_runtime_info_t(stride[0], stride[1], stride[2],
                    pads_begin[0], pads_begin[1], pads_begin[2], pads_end[0],
                    pads_end[1], pads_end[2])
            : dyn_conv_fwd_runtime_info_t(stride[0], stride[1], pads_begin[0],
                    pads_begin[1], pads_end[0], pads_end[1]);
    reflection::shared_general_object_t info
            = reflection::general_object_t::make(dyn_info);
    return info;
}

shape_rl_vec conv_fwd_core_op_t::get_shape_relations_impl(
        const std::vector<sc_dim> &data_plain_dims,
        const std::vector<sc_dim> &weight_plain_dims,
        const std::vector<sc_dim> &out_plain_dims, const any_map_t &attrs) {
    auto ndims = data_plain_dims.size();
    sc_dims pads_begin = attrs.has_key("pads_begin")
            ? attrs.get<sc_dims>("pads_begin")
            : attrs.get_or_else<sc_dims>("paddings", sc_dims(ndims - 2, 0));
    sc_dims pads_end = attrs.has_key("pads_end")
            ? attrs.get<sc_dims>("pads_end")
            : attrs.get_or_else<sc_dims>("paddings", sc_dims(ndims - 2, 0));

    sc_dims stride = attrs.get<sc_dims>("strides");
    sc_dims stride_dims(ndims - 2, stride[0]);
    if (stride.size() > 1) { stride_dims = stride; }

    shape_rl_vec ret;
    auto is_1x1 = std::all_of(weight_plain_dims.begin() + 2,
            weight_plain_dims.end(), [](int x) { return x == 1; });
    assert(data_plain_dims.size() == weight_plain_dims.size()
            && data_plain_dims.size() == 4 && weight_plain_dims.size() == 4);
    auto data_BS = data_plain_dims[0];
    auto data_H = data_plain_dims[data_plain_dims.size() - 2];
    auto data_W = data_plain_dims[data_plain_dims.size() - 1];
    auto out_BS = out_plain_dims[0];
    auto out_H = out_plain_dims[data_plain_dims.size() - 2];
    auto out_W = out_plain_dims[data_plain_dims.size() - 1];
    if (is_dynamic_dim(data_BS)) { ret.emplace_back(data_BS, out_BS); }
    if (is_dynamic_dim(data_H) && stride[0] == 1
            && (pads_begin[0] + pads_end[0] - weight_plain_dims[2] == -1)) {
        ret.emplace_back(data_H, out_H);
    }
    if (is_dynamic_dim(data_W) && stride[1] == 1
            && (pads_begin[1] + pads_end[1] - weight_plain_dims[3] == -1)) {
        ret.emplace_back(data_W, out_W);
    }
    return ret;
}

sc_op_ptr conv_fwd_core_op_t::do_compensations(
        sc_graph_t &mgr, const context_ptr &ctx) {
    need_compensation_ = false;
    return shared_from_this();
}

// TODO(baihui): this is only for 2d conv
void conv_fwd_core_op_t::collect_shrinked_lt_map(
        int bw_size, gt2gt_map &bw_lt_map) {
    auto data_plain_dims = get_inputs()[0]->details_.get_plain_dims();
    auto weight_plain_dims = get_inputs()[1]->details_.get_plain_dims();
    auto out_plain_dims = get_outputs()[0]->details_.get_plain_dims();
    auto data_blocking_dims = get_inputs()[0]->details_.get_blocking_dims();
    auto out_blocking_dims = get_outputs()[0]->details_.get_blocking_dims();
    sc_dims input_dims
            = {1, data_plain_dims[1], data_plain_dims[2], data_plain_dims[3]};
    sc_dims out_dims
            = {1, out_plain_dims[1], out_plain_dims[2], out_plain_dims[3]};
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_outputs()[0], out_dims);
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_inputs()[0], input_dims);
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_inputs()[1], weight_plain_dims);
}

void conv_fwd_core_op_t::collect_shrinked_axis_map(
        int bw_size, gt2axis_map &bw_axis_map) {
    auto data = get_inputs()[0], weight = get_inputs()[1],
         out = get_outputs()[0];
    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, data, std::vector<int> {0});
    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, weight, std::vector<int> {-1});
    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, out, std::vector<int> {0});
}

sc_dims conv_fwd_core_op_t::get_bwise_fuse_shrink_dims() {
    const int L2_size = 1024 * 1024 * 2;
    auto weight_dims = get_inputs()[1]->details_.get_plain_dims();
    auto data_dims = get_inputs()[0]->details_.get_plain_dims();
    auto out_dims = get_outputs()[0]->details_.get_plain_dims();
    const int dtype_sz = get_inputs()[1]->details_.dtype_ == datatypes::f32 ? 4
            : get_inputs()[1]->details_.dtype_ == datatypes::bf16           ? 2
                                                                            : 1;
    const int weight_size_byte = weight_dims[0] * weight_dims[1]
            * weight_dims[2] * weight_dims[3] * dtype_sz;
    const int data_size_byte
            = data_dims[1] * data_dims[2] * data_dims[3] * dtype_sz;
    const int output_size_byte
            = out_dims[1] * out_dims[2] * out_dims[3] * dtype_sz;
    bool enable_bwise = true;
    if (weight_size_byte >= L2_size / 4) {
        if (weight_size_byte >= L2_size / 2 || data_dims[1] >= 1024) {
            enable_bwise = false;
        }
    }
    sc_dims ret = {0};
    if (enable_bwise) {
        auto out_blocking_dims = get_outputs()[0]->details_.get_blocking_dims();
        ret = {out_blocking_dims[0]};
    }
    return ret;
}

conv_bwd_data_core_op_t::conv_bwd_data_core_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("conv_bwd_data_core", ins, outs, attrs) {
    COMPILE_ASSERT(info_.inputs_.size() == 2 || info_.inputs_.size() == 3,
            "conv_bwd_data expects 2 or 3 inputs");
    auto output_shape = attrs_.get<sc_dims>("dst_shape");
    ndims_ = info_.inputs_[0]->details_.get_plain_dims().size();
    auto &weightdims = info_.inputs_[1]->details_.get_plain_dims();
    is_1x1_ = std::all_of(weightdims.begin() + 2, weightdims.end(),
            [](int x) { return x == 1; });
    auto strides = attrs_.get<sc_dims>("strides");
    auto dilations = get_dilations(attrs_);
    COMPILE_ASSERT(std::all_of(dilations.begin(), dilations.end(),
                           [](int x) { return x == 1; }),
            "conv_bwd_data_core does not support dilation > 1 now");
    if (attrs_.has_key("auto_pad")) {
        auto pad_type = attrs_.get<std::string>("auto_pad");
        if (pad_type == "VALID") {
            attrs_.set<sc_dims>("pads_begin", sc_dims(ndims_ - 2, 0));
            attrs_.set<sc_dims>("pads_end", sc_dims(ndims_ - 2, 0));
        } else if (pad_type == "SAME_UPPER" || pad_type == "SAME_LOWER") {
            // output spatial dims are equal to input spatial dims
            conv_fwd_core_op_t::infer_auto_pad(get_owner_graph(), output_shape,
                    weightdims, strides, dilations, attrs_,
                    pad_type == "SAME_UPPER");
        }
        attrs_.set<std::string>("auto_pad", "none");
    }
    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, sc_data_format_t(), output_shape, datatypes::f32));
    } else {
        COMPILE_ASSERT(info_.outputs_.size() == 1,
                "conv_bwd_data_core expects 1 output");
        COMPILE_ASSERT(
                info_.outputs_[0]->details_.get_plain_dims() == output_shape,
                "conv_bwd_data_core's out dims not correct");
    }
}

bool conv_bwd_data_core_op_t::use_nested_generator() {
    bool use_nested = attrs_.get_or_else("use_nested", true);
    if (!use_nested) { return false; }
    const sc_dims &stride = attrs_.get<sc_dims>("strides");
    const sc_dims &pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    int num_threads = runtime_config_t::get().get_num_threads();
    const sc_dims &input_shape = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &weight_shape = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &output_shape = info_.outputs_[0]->details_.get_plain_dims();
    if (is_1x1_) {
        // nested generator constraints for 1x1 case
        // ToDo(zhangyan): improve following constraints
        if (num_threads % 7 != 0) { return false; }
        if (ndims_ != 4) { return false; }
        auto tmp_kernel
                = utils::make_unique<gen_nested_conv1x1_backprop_data_t>(this,
                        stride, pads_begin,
                        graph::extract_detail_from_tensors(get_inputs()),
                        graph::extract_detail_from_tensors(get_outputs()));
        int im_oc_block = tmp_kernel->im_oc_block_;
        int im_ic_block = tmp_kernel->im_ic_block_;
        int im_ow_block = tmp_kernel->im_ow_block_;
        int IC = weight_shape[1], OC = weight_shape[0],
            OS = input_shape[2] * input_shape[3], BS = input_shape[0];
        if (IC % im_ic_block || OC % im_oc_block || OS % im_ow_block) {
            return false;
        }
        // we need to check whether we can fully utilize all threads
        int possible_parallel_space
                = BS * (OS / im_ow_block) * (IC / im_ic_block);
        if (possible_parallel_space < num_threads) { return false; }
        return true;
    } else {
        // nested generator constraints for NxN case
        if (ndims_ != 4) { return false; }
        auto tmp_kernel
                = utils::make_unique<gen_nested_convNxN_backprop_data_t>(this,
                        stride, pads_begin,
                        graph::extract_detail_from_tensors(get_inputs()),
                        graph::extract_detail_from_tensors(get_outputs()));
        int im_ic_block = tmp_kernel->im_ic_block_;
        int BS = input_shape[0], IC = output_shape[1], IH = output_shape[2];
        int OW = input_shape[3], IW = output_shape[3];
        if (IC % im_ic_block) { return false; }
        // TODO(yifei): fix this restriction
        // currently we force im_ow_block_ = OW
        // this only holds if OW * stride_w == IW
        int stride_w = stride.back();
        if (OW * stride_w != IW) { return false; }
        // we need to check whether we can fully utilize all threads
        // int possible_parallel_space = BS * IH * (IC / im_ic_block);
        // TODO(yifei): loosen the restriction here
        // avoid the possibility that BS < bs_threads
        if (BS < num_threads) { return false; }
        return true;
    }
    return false;
}

body_generator_ptr conv_bwd_data_core_op_t::create_generator() {
    auto &stride = attrs_.get<sc_dims>("strides");
    const auto &pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    const bool is_3d = ndims_ == 5;
    int D = is_3d ? info_.inputs_[1]->details_.get_plain_dims()[2] : 1;
    int R = is_3d ? info_.inputs_[1]->details_.get_plain_dims()[3]
                  : info_.inputs_[1]->details_.get_plain_dims()[2];
    int S = is_3d ? info_.inputs_[1]->details_.get_plain_dims()[4]
                  : info_.inputs_[1]->details_.get_plain_dims()[3];
    if (D == 1 && R == 1 && S == 1) {
        if (use_nested_generator()) {
            return utils::make_unique<gen_nested_conv1x1_backprop_data_t>(this,
                    stride, pads_begin,
                    graph::extract_detail_from_tensors(get_inputs()),
                    graph::extract_detail_from_tensors(get_outputs()));
        } else {
            SC_MODULE_WARN << "Fall-back to non-nested conv1x1 backprop data.";
            return utils::make_unique<gen_conv1x1_backprop_data_t>(this, stride,
                    pads_begin,
                    graph::extract_detail_from_tensors(get_inputs()),
                    graph::extract_detail_from_tensors(get_outputs()));
        }
    } else {
        if (use_nested_generator()) {
            return utils::make_unique<gen_nested_convNxN_backprop_data_t>(this,
                    stride, pads_begin,
                    graph::extract_detail_from_tensors(get_inputs()),
                    graph::extract_detail_from_tensors(get_outputs()));
        } else {
            SC_MODULE_WARN << "Fall-back to non-nested convNxN backprop data.";
            return utils::make_unique<gen_convNxN_backprop_data>(this, stride,
                    pads_begin,
                    graph::extract_detail_from_tensors(get_inputs()),
                    graph::extract_detail_from_tensors(get_outputs()));
        }
    }
}

float conv_bwd_data_core_op_t::get_gflop() {
    return create_generator()->get_gflop();
}

void conv_bwd_data_core_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    if (!config_data_) {
        config_data_ = create_generator()->get_default_config(ctx);
    }
    int oc_block, ic_block;
    if (use_nested_generator()) {
        auto temp_generator = create_generator();
        auto gen_1x1 = dynamic_cast<gen_nested_conv1x1_backprop_data_t *>(
                temp_generator.get());
        auto gen_NxN = dynamic_cast<gen_nested_convNxN_backprop_data_t *>(
                temp_generator.get());
        ic_block = is_1x1_ ? gen_1x1->im_ic_block_ : gen_NxN->im_ic_block_;
        oc_block = is_1x1_ ? gen_1x1->im_oc_block_ : gen_NxN->im_oc_block_;
    } else {
        const conv_bwd_data_config_t &tcfg
                = *config_data_.get_as<conv_bwd_data_config_t>();
        ic_block = tcfg.C_block, oc_block = tcfg.K_block;
    }
    const bool is_3d = ndims_ == 5;
    in_formats.reserve(get_inputs().size());
    bool is_bf16 = info_.inputs_[0]->details_.dtype_ == datatypes::bf16;
    // plain input format
    in_formats.push_back(
            {is_3d ? sc_data_format_t::NDHWC() : sc_data_format_t::NHWC()});
    if (is_bf16) {
        COMPILE_ASSERT(info_.inputs_[1]->details_.dtype_ == datatypes::bf16,
                "The two inputs of conv_bwd_data_op_t should have the same "
                "data "
                "format");
        // CKRSkc2k or CKDRSkc2k
        in_formats.push_back(
                {is_3d ? sc_data_format_t::CKDRSkc2k(oc_block, ic_block)
                       : sc_data_format_t::CKRSkc2k(oc_block, ic_block)});
    } else {
        // CKRSkc or CKDRSkc
        in_formats.push_back(
                {is_3d ? sc_data_format_t::CKDRSkc(oc_block, ic_block)
                       : sc_data_format_t::CKRSkc(oc_block, ic_block)});
    }
    // plain output format
    out_formats.push_back(
            {is_3d ? sc_data_format_t::NDHWC() : sc_data_format_t::NHWC()});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

conv_bwd_weight_core_op_t::conv_bwd_weight_core_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("conv_bwd_weight_core", ins, outs, attrs) {
    COMPILE_ASSERT(info_.inputs_.size() == 2 || info_.inputs_.size() == 3,
            "conv_bwd_weight_core expects 2 or 3 inputs");
    auto &in_data_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &in_fwd_output_dims = info_.inputs_[1]->details_.get_plain_dims();
    auto &weight_shape = attrs_.get<sc_dims>("weights_shape");
    is_1x1_ = std::all_of(weight_shape.begin() + 2, weight_shape.end(),
            [](int x) { return x == 1; });
    COMPILE_ASSERT(in_data_dims[0] == in_fwd_output_dims[0],
            "The two inputs of conv_bwd_weight_core should have the same batch "
            "size.");
    COMPILE_ASSERT(info_.inputs_[0]->details_.dtype_
                    == info_.inputs_[1]->details_.dtype_,
            "The two inputs of conv_bwd_weight_core should have the "
            "same datatype");
    ndims_ = in_data_dims.size();
    auto strides = attrs_.get<sc_dims>("strides");
    auto dilations = get_dilations(attrs_);
    COMPILE_ASSERT(std::all_of(dilations.begin(), dilations.end(),
                           [](int x) { return x == 1; }),
            "conv_bwd_data_core does not support dilation > 1 now");
    if (attrs_.has_key("auto_pad")) {
        auto pad_type = attrs_.get<std::string>("auto_pad");
        if (pad_type == "VALID") {
            attrs_.set<sc_dims>("pads_begin", sc_dims(ndims_ - 2, 0));
            attrs_.set<sc_dims>("pads_end", sc_dims(ndims_ - 2, 0));
        } else if (pad_type == "SAME_UPPER" || pad_type == "SAME_LOWER") {
            // output spatial dims are equal to input spatial dims
            conv_fwd_core_op_t::infer_auto_pad(get_owner_graph(), in_data_dims,
                    weight_shape, strides, dilations, attrs_,
                    pad_type == "SAME_UPPER");
        }
        attrs_.set<std::string>("auto_pad", "none");
    }
    const auto &pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    const auto &pads_end = attrs_.has_key("pads_end")
            ? attrs_.get<sc_dims>("pads_end")
            : attrs_.get<sc_dims>("paddings");
    bool has_pad = std::any_of(pads_begin.begin(), pads_begin.end(),
                           [](sc_dim p) { return p > 0; })
            || std::any_of(pads_end.begin(), pads_end.end(),
                    [](sc_dim p) { return p > 0; });

    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, sc_data_format_t(), weight_shape, datatypes::f32));
    } else {
        COMPILE_ASSERT(info_.outputs_.size() == 1,
                "conv_bwd_weight_core expects 1 output");
        COMPILE_ASSERT(
                info_.outputs_[0]->details_.get_plain_dims() == weight_shape,
                "conv_bwd_weight_core's out dims not correct");
    }
}

bool conv_bwd_weight_core_op_t::use_nested_generator() {
    bool use_nested = attrs_.get_or_else("use_nested", true);
    if (!use_nested) { return false; }
    const sc_dims &stride = attrs_.get<sc_dims>("strides");
    const sc_dims &pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    int num_threads = runtime_config_t::get().get_num_threads();
    const sc_dims &weight_shape = info_.outputs_[0]->details_.get_plain_dims();
    const sc_dims &input_shape = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &delta_shape = info_.inputs_[1]->details_.get_plain_dims();
    if (!is_1x1_) {
        // nested generator constraints for NxN case
        if (num_threads % 7 != 0) { return false; }
        if (ndims_ != 4) { return false; }
        int R = weight_shape[ndims_ - 2];
        int S = weight_shape[ndims_ - 1];
        int stride_h = stride[0], stride_w = stride[0];
        if (stride.size() > 1) { stride_w = stride[1]; }
        if (stride_h > R || stride_w > S) { return false; }
        auto tmp_kernel = utils::make_unique<gen_nested_convNXN_bwd_weight_t>(
                this, stride, pads_begin,
                graph::extract_detail_from_tensors(get_inputs()),
                graph::extract_detail_from_tensors(get_outputs()));
        int im_oc_block = tmp_kernel->im_oc_block_;
        int im_ic_block = tmp_kernel->im_ic_block_;
        int im_bs_block = tmp_kernel->im_bs_block_;
        int BS = input_shape[0], IC = input_shape[1], OC = delta_shape[1],
            OH = delta_shape[2];
        if (BS % im_bs_block || IC % im_ic_block || OC % im_oc_block
                || OH % 7) {
            return false;
        }
        // we need to check whether we can fully utilize all threads
        int possible_parallel_space = (BS / im_bs_block) * (IC / im_ic_block)
                * (OC / im_oc_block) * (OH / 7);
        if (possible_parallel_space < num_threads) { return false; }
        return true;
    } else {
        // nested generator constraints for 1x1 case
        if (num_threads % 7 != 0) { return false; }
        if (ndims_ != 4) { return false; }
        auto tmp_kernel
                = utils::make_unique<gen_nested_conv1x1_backprop_weight_t>(this,
                        stride, pads_begin,
                        graph::extract_detail_from_tensors(get_inputs()),
                        graph::extract_detail_from_tensors(get_outputs()));
        int im_oc_block = tmp_kernel->im_oc_block_;
        int im_ic_block = tmp_kernel->im_ic_block_;
        int im_bs_block = tmp_kernel->im_bs_block_;
        int BS = input_shape[0], IC = input_shape[1], OC = delta_shape[1],
            OH = delta_shape[2];
        if (BS % im_bs_block || IC % im_ic_block || OC % im_oc_block
                || OH % 7) {
            return false;
        }
        // we need to check whether we can fully utilize all threads
        int possible_parallel_space = (BS / im_bs_block) * (IC / im_ic_block)
                * (OC / im_oc_block) * (OH / 7);
        if (possible_parallel_space < num_threads) { return false; }
        return true;
    }
    return false;
}

body_generator_ptr conv_bwd_weight_core_op_t::create_generator() {
    auto &stride = attrs_.get<sc_dims>("strides");
    auto &pads_begin = attrs_.has_key("pads_begin")
            ? attrs_.get<sc_dims>("pads_begin")
            : attrs_.get<sc_dims>("paddings");
    auto &weight_shape = attrs_.get<sc_dims>("weights_shape");
    sc_dims input_dims = info_.inputs_[0]->details_.get_plain_dims();
    if (is_1x1_) {
        if (use_nested_generator()) {
            return utils::make_unique<gen_nested_conv1x1_backprop_weight_t>(
                    this, stride, pads_begin,
                    graph::extract_detail_from_tensors(get_inputs()),
                    graph::extract_detail_from_tensors(get_outputs()));
        } else {
            SC_MODULE_WARN
                    << "Fall-back to non-nested conv1x1 backprop weight.";
            // tested for reduce on ALL
            int block_size = 64;
            if (weight_shape[0] * weight_shape[1] * input_dims[0]
                            / (block_size * block_size * block_size)
                    < runtime_config_t::get().get_num_threads()) {
                return utils::make_unique<gen_conv1x1_backprop_weight_t>(this,
                        stride, pads_begin,
                        graph::extract_detail_from_tensors(get_inputs()),
                        graph::extract_detail_from_tensors(get_outputs()),
                        gen_conv1x1_backprop_weight_t::generator_type_t::
                                REDUCE_ALL2);
            } else {
                return utils::make_unique<gen_conv1x1_backprop_weight_t>(this,
                        stride, pads_begin,
                        graph::extract_detail_from_tensors(get_inputs()),
                        graph::extract_detail_from_tensors(get_outputs()),
                        gen_conv1x1_backprop_weight_t::generator_type_t::
                                REDUCE_N);
            }
        }
    } else {
        if (use_nested_generator()) {
            return utils::make_unique<gen_nested_convNXN_bwd_weight_t>(this,
                    stride, pads_begin,
                    graph::extract_detail_from_tensors(get_inputs()),
                    graph::extract_detail_from_tensors(get_outputs()));
        }
        SC_MODULE_WARN << "Fall-back to non-nested convNxN backprop weight.";
        return utils::make_unique<gen_convNxN_backprop_weight>(this, stride,
                pads_begin, graph::extract_detail_from_tensors(get_inputs()),
                graph::extract_detail_from_tensors(get_outputs()),
                gen_convNxN_backprop_weight::generator_type_t::REDUCE_N);
    }
}

float conv_bwd_weight_core_op_t::get_gflop() {
    return create_generator()->get_gflop();
}

void conv_bwd_weight_core_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    if (!config_data_) {
        config_data_ = create_generator()->get_default_config(ctx);
    }
    if (use_nested_generator()) {
        auto temp_generator = create_generator();
        auto gen_1x1 = dynamic_cast<gen_nested_conv1x1_backprop_weight_t *>(
                temp_generator.get());
        auto gen_NxN = dynamic_cast<gen_nested_convNXN_bwd_weight_t *>(
                temp_generator.get());
        int im_bs_block
                = is_1x1_ ? gen_1x1->im_bs_block_ : gen_NxN->im_bs_block_;
        int im_ic_block
                = is_1x1_ ? gen_1x1->im_ic_block_ : gen_NxN->im_ic_block_;
        int im_oc_block
                = is_1x1_ ? gen_1x1->im_oc_block_ : gen_NxN->im_oc_block_;
        const bool is_3d = ndims_ == 5;
        in_formats.reserve(get_inputs().size());
        if (!is_1x1_) {
            in_formats.push_back(
                    {is_3d ? sc_data_format_t::NDHWCn(im_bs_block)
                           : sc_data_format_t::NHWCn(im_bs_block)});
        } else {
            in_formats.push_back({is_3d ? sc_data_format_t::NDHWC()
                                        : sc_data_format_t::NHWC()});
        }
        // N(D)HWK
        in_formats.push_back(
                {is_3d ? sc_data_format_t::NDHWC() : sc_data_format_t::NHWC()});
        out_formats.push_back(
                {is_3d ? sc_data_format_t::CKDRSck(im_ic_block, im_oc_block)
                       : sc_data_format_t::CKRSck(im_ic_block, im_oc_block)});
        format_to_dense_format_stride_pair(
                in_formats, out_formats, supported_ins, supported_outs);
        return;
    }
    const conv_bwd_weight_config_t &tcfg
            = *config_data_.get_as<conv_bwd_weight_config_t>();
    const bool is_3d = ndims_ == 5;
    in_formats.reserve(get_inputs().size());

    // NC(D)HWnc or NC(D)HWnc2n
    if (info_.inputs_[0]->details_.dtype_ == datatypes::bf16) {
        in_formats.push_back(
                {is_3d ? sc_data_format_t(
                         sc_data_format_kind_t(0, 1, 2, 3, 4, 0, 1, 0),
                         {tcfg.N_block, tcfg.C_block, 2})
                       : sc_data_format_t(
                               sc_data_format_kind_t(0, 1, 2, 3, 0, 1, 0),
                               {tcfg.N_block, tcfg.C_block, 2})});
    } else {
        // NC(D)HWnc
        in_formats.push_back(
                {is_3d ? sc_data_format_t(
                         sc_data_format_kind_t(0, 1, 2, 3, 4, 0, 1),
                         {tcfg.N_block, tcfg.C_block})
                       : sc_data_format_t(
                               sc_data_format_kind_t(0, 1, 2, 3, 0, 1),
                               {tcfg.N_block, tcfg.C_block})});
    }
    // NK(D)HWkn
    in_formats.push_back(
            {is_3d ? sc_data_format_t(
                     sc_data_format_kind_t(0, 1, 2, 3, 4, 1, 0),
                     {tcfg.K_block, tcfg.N_block})
                   : sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3, 1, 0),
                           {tcfg.K_block, tcfg.N_block})});
    // KC(D)RSkc
    out_formats.push_back(
            {is_3d ? sc_data_format_t(
                     sc_data_format_kind_t(0, 1, 2, 3, 4, 0, 1),
                     {tcfg.K_block, tcfg.C_block})
                   : sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3, 0, 1),
                           {tcfg.K_block, tcfg.C_block})});

    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

} // namespace ops
OP_REGISTER(ops::conv_fwd_core_op_t, conv_fwd_core)
OP_REGISTER(ops::conv_bwd_data_core_op_t, conv_bwd_data_core)
OP_REGISTER(ops::conv_bwd_weight_core_op_t, conv_bwd_weight_core)

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

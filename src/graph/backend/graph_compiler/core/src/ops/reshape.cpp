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
#include "reshape.hpp"
#include <memory>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
static void get_output_shape(sc_dims &outshape, const sc_dims &input_dims,
        const int32_t *shape, int dim, bool special_zero) {
    // we allow one dim value to be -1, which is automatically calculated to
    // keep the number of elements of the out tensor = in tensor.
    int auto_cal_dim = -1;
    size_t total_shape = 1;
    for (int i = 0; i < dim; i++) {
        int shape_v = shape[i];
        if (shape_v == -1) {
            COMPILE_ASSERT(
                    auto_cal_dim == -1, "reshape only support one -1 shape");
            auto_cal_dim = i;
        } else {
            if (special_zero && shape_v == 0) {
                COMPILE_ASSERT(static_cast<size_t>(i) < input_dims.size(),
                        "The special zero at "
                                << i
                                << " dimension is out of range in input shape");
                shape_v = input_dims[i];
            }
            total_shape *= shape_v;
        }
        outshape.emplace_back(shape_v);
    }
    size_t input_total_shape = 1;
    for (auto v : input_dims) {
        input_total_shape *= v;
    }
    const char *error_msg
            = "Reshape: The input tensor size does not match the given shape";
    if (auto_cal_dim != -1) {
        COMPILE_ASSERT(input_total_shape >= total_shape, error_msg);
        outshape[auto_cal_dim] = input_total_shape / total_shape;
    } else {
        COMPILE_ASSERT(input_total_shape == total_shape, error_msg);
    }
}

static static_data_t *validate_and_get_static_shape(sc_op *ths) {
    COMPILE_ASSERT(
            ths->get_inputs().size() == 2, "dynamic reshape op takes 2 inputs");
    auto in1 = ths->get_inputs()[1]->producer_owner_;
    auto &in1_detail = ths->get_inputs()[1]->details_;
    COMPILE_ASSERT(in1_detail.get_format().is_plain()
                    || in1_detail.get_format().is_any(),
            "Expecting plain format for input 2 of " << ths->op_name_);
    COMPILE_ASSERT(in1_detail.get_blocking_dims().size() == 1
                    && utils::is_one_of(in1_detail.dtype_, datatypes::index,
                            datatypes::s32),
            "Expecting 1D and int32/int64 tensor for input 2 of "
                    << ths->op_name_);
    if (!in1->isa<input_op>() && !in1->isa<constant_op_t>()) { return nullptr; }
    if (ths->is_dynamic()) { return nullptr; }
    auto ret = in1->attrs_.get_or_else<std::shared_ptr<static_data_t>>(
            "values", nullptr);
    COMPILE_ASSERT(ret,
            "Since dynamic shape is not supported yet, we are expecting the "
            "constant value data from the inputs as the shape "
            "info for the dynamic shaped op: "
                    << ths->op_name_);
    return ret.get();
}

dynamic_reshape_op::dynamic_reshape_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : sc_op("dynamic_reshape", ins, outs, attrs) {
    auto shape_data = validate_and_get_static_shape(this);
    sc_dims outshape;
    auto dim = get_inputs()[1]->details_.get_plain_dims()[0];
    if (!ins[0]->is_dynamic()) {
        COMPILE_ASSERT(shape_data,
                "Reshape requires compile-time constant shape for now");
        auto input_dims = get_inputs()[0]->details_.get_plain_dims();
        bool special_zero = attrs.get<bool>("special_zero");
        int32_t *shape = reinterpret_cast<int32_t *>(shape_data->data_);
        COMPILE_ASSERT(
                dim * sizeof(int32_t) == shape_data->size_, "Bad shape data");
        outshape.reserve(dim);
        get_output_shape(outshape, input_dims, shape, dim, special_zero);
    } else {
        outshape = sc_dims(dim, dimensions::dynamic_any);
    }
    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(graph_tensor::make(outshape,
                sc_data_format_t(), get_inputs()[0]->details_.dtype_));
    } else {
        COMPILE_ASSERT(
                info_.outputs_.size() == 1, "Expecting 1 output for reshape");
        auto &details = info_.outputs_[0]->details_;
        COMPILE_ASSERT(details.dtype_ == info_.inputs_[0]->details_.dtype_,
                "Reshape: input/output dtype does not match");
        COMPILE_ASSERT(details.get_plain_dims() == outshape,
                "Reshape: Expecting output shape = "
                        << utils::print_vector(outshape) << ", given: "
                        << utils::print_vector(details.get_plain_dims()));
    }
}
void dynamic_reshape_op::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    out_formats.push_back({sc_data_format_kind_t::get_plain_by_dims(
            get_outputs()[0]->details_.get_plain_dims().size())});
    in_formats.push_back({sc_data_format_kind_t::get_plain_by_dims(
            get_inputs()[0]->details_.get_plain_dims().size())});
    in_formats.push_back({sc_data_format_kind_t::get_plain_by_dims(1)});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}
ir_module_ptr dynamic_reshape_op::get_func(context_ptr ctx) {
    throw std::runtime_error("Not implemented");
}
sc_op_ptr dynamic_reshape_op::constant_optimize(sc_graph_t &graph) {
    if (is_dynamic()) { return nullptr; }
    auto shape_data = validate_and_get_static_shape(this);
    // if input shape is not constant, return
    if (!shape_data) { return nullptr; }
    auto new_input = graph.make("tensor_view", {get_inputs()[0]}, {},
            {{"shape", get_outputs()[0]->details_.get_plain_dims()}});
    this->replace_uses_with_and_remove(new_input);
    return new_input;
}

static_reshape_op::static_reshape_op(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : sc_op("static_reshape", ins, outs, attrs) {
    COMPILE_ASSERT(
            attrs.has_key("shape"), "Static reshape requires shape attributes");
    auto shape = attrs.get<sc_dims>("shape");
    std::vector<int32_t> shape_s32(shape.begin(), shape.end());
    bool special_zero = attrs.get<bool>("special_zero");
    auto input_dims = get_inputs()[0]->details_.get_plain_dims();
    auto dim = static_cast<int>(shape.size());
    sc_dims outshape;
    outshape.reserve(dim);
    get_output_shape(outshape, input_dims, shape_s32.data(), dim, special_zero);
    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(graph_tensor::make(outshape,
                sc_data_format_t(), get_inputs()[0]->details_.dtype_));
    } else {
        COMPILE_ASSERT(
                info_.outputs_.size() == 1, "Expecting 1 output for reshape");
        auto &details = info_.outputs_[0]->details_;
        COMPILE_ASSERT(details.dtype_ == info_.inputs_[0]->details_.dtype_,
                "Reshape: input/output dtype does not match");
        COMPILE_ASSERT(details.get_plain_dims() == outshape,
                "Reshape: Expecting output shape = "
                        << utils::print_vector(outshape) << ", given: "
                        << utils::print_vector(details.get_plain_dims()));
    }
}
void static_reshape_op::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    throw std::runtime_error("Not implemented");
}

// for single op generate
ir_module_ptr static_reshape_op::get_func(context_ptr ctx) {
    throw std::runtime_error("Not implemented");
}
sc_op_ptr static_reshape_op::constant_optimize(sc_graph_t &graph) {
    auto new_input = graph.make("tensor_view", {get_inputs()[0]}, {},
            {{"shape", get_outputs()[0]->details_.get_plain_dims()}});
    this->replace_uses_with_and_remove(new_input);
    return new_input;
}
} // namespace ops
OP_REGISTER(ops::dynamic_reshape_op, dynamic_reshape);
OP_REGISTER(ops::static_reshape_op, static_reshape);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

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

#include <assert.h>
#include <atomic>
#include <numeric>
#include <unordered_map>

#include <algorithm>
#include <utility>
#include "fused_op.hpp"
#include "fusible_op.hpp"
#include "fusion_mgr.hpp"
#include "outer_loop_generator.hpp"
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

namespace sc {

ir_module_ptr fusible_op_t::get_func(context_ptr ctx) {
    int base_idx = 0;
    if (auto binary_node = this->dyn_cast<binary_elementwise_op_t>()) {
        // if bc side (smaller side) is the lhs, we need to set base_idx to 1
        if (!binary_node->get_broadcast_input()) { base_idx = 1; }
    }
    outer_loop_generator_t gen(base_idx);
    return fusible_op_get_func(this, gen, ctx, true);
}

void fusible_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<sc_data_format_t>> &in_formats,
        std::vector<std::vector<sc_data_format_t>> &out_formats) {
    if (this->isa<constant_op_t>()) {
        out_formats.push_back({info_.outputs_[0]->details_.get_format()});
    } else {
        out_formats.push_back({info_.inputs_[0]->details_.get_format()});
    }
}

size_t fusible_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    size_t wkld = 0UL;
    auto accumulate_workload
            = [&wkld](size_t weight, const shape_dtype_pair &v) {
                  auto &dtype = v.second;
                  wkld += utils::get_sizeof_type(dtype) * weight;
              };
    std::for_each(ins.begin(), ins.end(),
            std::bind(accumulate_workload,
                    static_cast<size_t>(
                            op_traits::workload_computable_t::read_weight),
                    std::placeholders::_1));
    std::for_each(outs.begin(), outs.end(),
            std::bind(accumulate_workload,
                    static_cast<size_t>(
                            op_traits::workload_computable_t::write_weight),
                    std::placeholders::_1));
    return wkld;
}

size_t fusible_op_t::compute_fusible_workload(const context_ptr &ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    std::vector<shape_dtype_pair> wkld_ins, wkld_outs;
    wkld_ins.resize(inputs.size());
    wkld_outs.resize(dst.size());
    auto get_shape_dtype_pair = [](const tensor_slice *v) {
        return std::make_pair(get_expr_to_dims(v->shape_), v->get_base_dtype());
    };
    std::transform(inputs.begin(), inputs.end(), wkld_ins.begin(),
            get_shape_dtype_pair);
    std::transform(
            dst.begin(), dst.end(), wkld_outs.begin(), get_shape_dtype_pair);
    return compute_workload(wkld_ins, wkld_outs);
}

input_op::input_op(const sc_dims &dims, sc_data_type_t dtype) {
    op_name_ = "input";
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
            this, sc_data_format_t(), dims, dtype));
}

input_op::input_op(const logical_tensor_t &lt) {
    op_name_ = "input";
    info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this, lt));
}

input_op::input_op(const std::vector<graph_tensor_ptr> &outs) {
    info_.outputs_ = outs;
    for (auto &out : info_.outputs_) {
        out->producer_owner_ = this;
    }
    op_name_ = "input";
}

void input_op::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
}

output_op::output_op(const graph_tensor_ptr &v) {
    info_.inputs_.emplace_back(v);
    op_name_ = "output";
}

output_op::output_op(const std::vector<graph_tensor_ptr> &in) {
    info_.inputs_ = in;
    op_name_ = "output";
}

void output_op::prepare_fusion_data(fdata_map &fdmap) {
    assert(info_.outputs_.empty() && "Wrong op output size.\n");
    auto &inputs = info_.inputs_[0];
    auto &outdetail = fdmap.get(inputs);
    outdetail.need_alloc_ = false;
}

// special handling for union values
bool constant_op_t::compare_contents(const sc_op *other) const {
    COMPILE_ASSERT(attrs_.has_key("values") && attrs_.has_key("dtype"),
            "expecting values and dtype in attr");
    COMPILE_ASSERT(
            other->attrs_.has_key("values") && other->attrs_.has_key("dtype"),
            "expecting values and dtype in attr");
    auto dtype = attrs_.get<sc_data_type_t>("dtype");
    if (other->attrs_.get<sc_data_type_t>("dtype") != dtype) { return false; }
    if (attrs_.has_key("format")) {
        if (!other->attrs_.has_key("format")) { return false; }
        if (other->attrs_.get<sc_data_format_t>("format")
                != attrs_.get<sc_data_format_t>("format")) {
            return false;
        }
    }
    auto &vals = attrs_.get<std::shared_ptr<static_data_t>>("values");
    auto &vals2 = other->attrs_.get<std::shared_ptr<static_data_t>>("values");
    if (vals->size_ != vals2->size_) { return false; }

    switch (get_type_category_nothrow(dtype)) {
        case CATE_FLOAT:
            for (size_t i = 0; i < vals->size_ / 4; i++) {
                if (static_cast<float *>(vals->data_)[i]
                        != static_cast<float *>(vals2->data_)[i]) {
                    return false;
                }
            }
            break;
        case CATE_INT:
        case CATE_UINT:
            for (size_t i = 0; i < vals->size_ / 4; i++) {
                if (static_cast<uint32_t *>(vals->data_)[i]
                        != static_cast<uint32_t *>(vals2->data_)[i]) {
                    return false;
                }
            }
            break;
        default:
            throw std::runtime_error("Met unexpected dtype for constant");
            break;
    }
    return true;
}

size_t constant_op_t::hash_contents() const {
    size_t seed = 0;
    COMPILE_ASSERT(attrs_.has_key("values") && attrs_.has_key("dtype"),
            "expecting values and dtype in attr");
    if (attrs_.has_key("format")) {
        hash_combine(seed, attrs_.get<sc_data_format_t>("format"));
    }
    auto &vals = attrs_.get<std::shared_ptr<static_data_t>>("values");

    for (size_t i = 0; i < vals->size_; i++) {
        hash_combine(seed, static_cast<char *>(vals->data_)[i]);
    }

    return seed;
}

void constant_op_t::reset_const_values() {
    if (attrs_.has_key("temp.var") && attrs_.has_key("temp.val/var")) {
        int K = static_cast<int>(
                attrs_.get<std::shared_ptr<VConst>>("temp.var")->var_);
        int base_val = attrs_.get<int>("temp.val/var");
        // update private member
        const_values_ = std::make_shared<static_data_t>(
                std::vector<int> {base_val * K});
        // update attr
        attrs_.set("values", const_values_);
    }
}

constant_op_t::constant_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    COMPILE_ASSERT(ins.empty(), "No op input.\n");
    COMPILE_ASSERT(attrs.has_key("values") && attrs.has_key("dtype")
                    && attrs.has_key("plain_dims"),
            "expecting values, format and dtype in attr");
    op_name_ = "constant";
    sc_data_format_t format
            = attrs.get_or_else("format", sc_data_format_t(format_kinds::A));
    attrs_ = attrs;
    const_values_ = attrs.get<std::shared_ptr<static_data_t>>("values");
    sc_data_type_t dtype = attrs.get<sc_data_type_t>("dtype");
    sc_dims plain_dims = attrs.get<sc_dims>("plain_dims");

    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(
                this, logical_tensor_t {format, plain_dims, dtype}));
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
    }
}

// todo: support tensor expr
constant_op_t::constant_op_t(std::shared_ptr<static_data_t> v,
        sc_data_type_t dtype, const sc_dims &plain_dims,
        const sc_data_format_t &format) {
    const_values_ = std::move(v);
    info_.outputs_.emplace_back(
            std::make_shared<graph_tensor>(this, format, plain_dims, dtype));
    info_.outputs_[0]->details_.dtype_ = dtype;
    info_.outputs_[0]->details_.set_plain_dims(plain_dims);
    attrs_.set("dtype", dtype);
    attrs_.set("values", const_values_);
    attrs_.set("plain_dims", plain_dims);
    attrs_.set("format", format);
    op_name_ = "constant";
}

void constant_op_t::prepare_fusion_data(fdata_map &fdmap) {
    COMPILE_ASSERT(info_.outputs_.size() == 1, "Wrong op output size.\n");
    auto &output = info_.outputs_[0];
    auto &outdetail = fdmap.get(output);
    auto blocking_dims = get_constant_blocking_dims();
    outdetail.need_alloc_ = true;
}

OP_REGISTER(constant_op_t, constant)

} // namespace sc

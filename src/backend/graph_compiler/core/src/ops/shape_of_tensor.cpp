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

#include "shape_of_tensor.hpp"
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <compiler/ir/builder.hpp>
#include <microkernel/builtin.hpp>
namespace sc {
namespace ops {

shape_of_tensor_op_t::shape_of_tensor_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    op_name_ = "shape_of_tensor";
    COMPILE_ASSERT(ins.size() == 1, "Shape of tensor op takes 1 input");
    info_.inputs_ = ins;
    attrs_ = attrs;
    auto &shape_idxs = attrs_.get<std::vector<int>>("shape_idxs");
    auto shape_size
            = static_cast<int>(ins[0]->details_.get_plain_dims().size());
    for (auto &idx : shape_idxs) {
        COMPILE_ASSERT(idx < shape_size,
                "Shape index: " << idx
                                << " should not be large than input shape "
                                   "size: "
                                << shape_size << "!");
    }
    shape_idxs_ = shape_idxs;
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this));
        // shrinked from sc_dim
        info_.outputs_[0]->details_.dtype_ = datatypes::s32;
        info_.outputs_[0]->details_.set_plain_dims(
                {static_cast<int64_t>(shape_idxs.size())});
    } else {
        COMPILE_ASSERT(outs.size() == 1, "Wrong op output size.\n");
        info_.outputs_ = outs;
    }
}

void shape_of_tensor_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    out_formats.reserve(info_.outputs_.size());
    out_formats.push_back({sc_data_format_t(format_kinds::A)});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

runtime_extra_info_t shape_of_tensor_op_t::get_extra_lower_infos(
        sc_graph_t &graph, ir_module_ptr &m) {
    std::vector<graph_tensor_ptr> in_ltsrs, out_ltsrs;
    out_ltsrs = info_.outputs_;
    if (find_ltsr_func_ != nullptr) {
        std::vector<graph_tensor_ptr> ins;
        ins = find_ltsr_func_(this);
        in_ltsrs = ins;
    } else {
        in_ltsrs = info_.inputs_;
    }
    auto shape_tsr = builder::make_tensor("shape_idxs", {shape_idxs_.size()},
            datatypes::s32, address_space::automatic,
            std::make_shared<static_data_t>(shape_idxs_));
    m->add_global_var(builder::make_var_tensor_def_unattached(
            shape_tsr, linkage::private_global)
                              .checked_as<define>());
    auto shape_size = builder::make_var(datatypes::s32, "shape_idxs_size");
    m->add_global_var(builder::make_var_tensor_def_unattached(shape_size,
            linkage::private_global, expr(static_cast<int>(shape_idxs_.size())))
                              .checked_as<define>());
    std::vector<expr> attrs = {shape_tsr, shape_size};
    return runtime_extra_info_t {in_ltsrs, out_ltsrs, attrs};
}

ir_module_ptr shape_of_tensor_op_t::get_func(context_ptr ctx) {
    auto modu = std::make_shared<ir_module_t>(ctx);
    auto f = builtin::get_cal_shape_of_tensor_op_func();
    modu->attr_.set("temp.runtime_func", f);
    return modu;
}
} // namespace ops
OP_REGISTER(ops::shape_of_tensor_op_t, shape_of_tensor);

} // namespace sc

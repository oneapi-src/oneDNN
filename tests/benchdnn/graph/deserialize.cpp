/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>

#include "deserialize.hpp"

namespace graph {

using namespace dnnl::graph;
using namespace dnnl::impl::graph;

void deserialized_attr::load(utils::json::json_reader_t *reader) {
    reader->begin_object();
    std::string key_entry;
    std::string value_entry;
    reader->next_object_item(&key_entry);
    if (key_entry != "type") return;

    reader->read<std::string>(&type_);
    if (type_ == "string") {
        reader->next_object_item(&value_entry);
        if (value_entry == "value") { reader->read<std::string>(&str_value_); }
    } else if (type_ == "bool") {
        reader->next_object_item(&value_entry);
        if (value_entry == "value") { reader->read<bool>(&bool_value_); }
    } else if (type_ == "s64") {
        reader->next_object_item(&value_entry);
        if (value_entry == "value") { reader->read<int64_t>(&s64_value_); }
    } else if (type_ == "s64[]") {
        reader->next_object_item(&value_entry);
        if (value_entry == "value") {
            reader->read<std::vector<int64_t>>(&s64_vector_);
        }
    } else if (type_ == "f32") {
        reader->next_object_item(&value_entry);
        if (value_entry == "value") { reader->read<float>(&f32_value_); }
    } else if (type_ == "f32[]") {
        reader->next_object_item(&value_entry);
        if (value_entry == "value") {
            reader->read<std::vector<float>>(&f32_vector_);
        }
    }
    reader->next_object_item(&value_entry);
}

logical_tensor::data_type deserialized_lt::get_data_type() const {
    if (data_type_ == "f32") {
        return logical_tensor::data_type::f32;
    } else if (data_type_ == "f16") {
        return logical_tensor::data_type::f16;
    } else if (data_type_ == "s8") {
        return logical_tensor::data_type::s8;
    } else if (data_type_ == "u8") {
        return logical_tensor::data_type::u8;
    } else if (data_type_ == "bf16") {
        return logical_tensor::data_type::bf16;
    } else if (data_type_ == "s32") {
        return logical_tensor::data_type::s32;
    } else if (data_type_ == "boolean") {
        return logical_tensor::data_type::boolean;
    } else if (data_type_ == "f8_e5m2") {
        return logical_tensor::data_type::f8_e5m2;
    } else if (data_type_ == "f8_e4m3") {
        return logical_tensor::data_type::f8_e4m3;
    } else if (data_type_ == "s4") {
        return logical_tensor::data_type::s4;
    } else if (data_type_ == "u4") {
        return logical_tensor::data_type::u4;
    } else {
        return logical_tensor::data_type::undef;
    }
}

logical_tensor::property_type deserialized_lt::get_property_type() const {
    if (property_type_ == "constant") {
        return logical_tensor::property_type::constant;
    } else if (property_type_ == "variable") {
        return logical_tensor::property_type::variable;
    } else {
        return logical_tensor::property_type::undef;
    }
}

logical_tensor deserialized_lt::create() const {
    if (layout_type_ == "any") {
        return logical_tensor(id_, get_data_type(), shape_,
                logical_tensor::layout_type::any, get_property_type());
    } else {
        return logical_tensor(
                id_, get_data_type(), shape_, stride_, get_property_type());
    }
}

void deserialized_lt::load(utils::json::json_reader_t *reader) {
    utils::json::read_helper_t helper;

    helper.declare_field("id", &id_);
    helper.declare_field("dtype", &data_type_);
    helper.declare_field("shape", &shape_);
    helper.declare_field("stride", &stride_);
    helper.declare_field("layout_type", &layout_type_);
    helper.declare_field("property_type", &property_type_);
    helper.read_fields(reader);
}

void deserialized_op::load(utils::json::json_reader_t *reader) {
    utils::json::read_helper_t helper;

    helper.declare_field("id", &id_);
    helper.declare_field("name", &name_);
    helper.declare_field("kind", &kind_);
    helper.declare_field("attrs", &attrs_);
    helper.declare_field("inputs", &in_lts_);
    helper.declare_field("outputs", &out_lts_);
    helper.read_fields(reader);
}

op deserialized_op::create() const {
    op aop(id_, opstr2kind(kind_), name_);
    for (auto it = attrs_.begin(); it != attrs_.end(); ++it) {
        const auto &attr = attrstr2kind(it->first);
        const auto &attr_value = it->second;
        const auto &type = attr_value.type_;
        if (type == "string") {
            const auto value = attr_value.str_value_;
            aop.set_attr(attr, value);
        } else if (type == "bool") {
            const auto value = attr_value.bool_value_;
            aop.set_attr(attr, value);
        } else if (type == "s64") {
            const auto value = attr_value.s64_value_;
            aop.set_attr(attr, value);
        } else if (type == "s64[]") {
            const auto value = attr_value.s64_vector_;
            aop.set_attr(attr, value);
        } else if (type == "f32") {
            const auto value = attr_value.f32_value_;
            aop.set_attr(attr, value);
        } else if (type == "f32[]") {
            const auto value = attr_value.f32_vector_;
            aop.set_attr(attr, value);
        }
    }

    for (const auto &lt : in_lts_) {
        aop.add_input(lt.create());
    }
    for (const auto &lt : out_lts_) {
        aop.add_output(lt.create());
    }

    return aop;
}

bool deserialized_op::get_attr_string(
        std::string &attr, const std::string &attr_name) const {
    auto it = attrs_.find(attr_name);
    if (it == attrs_.end()) return false;
    return attr = it->second.str_value_, true;
}

bool deserialized_op::get_attr_bool(
        bool &attr, const std::string &attr_name) const {
    auto it = attrs_.find(attr_name);
    if (it == attrs_.end()) return false;
    return attr = it->second.bool_value_, true;
}

bool deserialized_op::get_attr_f32(
        float &attr, const std::string &attr_name) const {
    auto it = attrs_.find(attr_name);
    if (it == attrs_.end()) return false;
    return attr = it->second.f32_value_, true;
}

bool deserialized_op::get_attr_s64(
        int64_t &attr, const std::string &attr_name) const {
    auto it = attrs_.find(attr_name);
    if (it == attrs_.end()) return false;
    return attr = it->second.s64_value_, true;
}

bool deserialized_op::get_attr_f32_vector(
        std::vector<float> &attr, const std::string &attr_name) const {
    auto it = attrs_.find(attr_name);
    if (it == attrs_.end()) return false;
    return attr = it->second.f32_vector_, true;
}

bool deserialized_op::get_attr_s64_vector(
        std::vector<int64_t> &attr, const std::string &attr_name) const {
    auto it = attrs_.find(attr_name);
    if (it == attrs_.end()) return false;
    return attr = it->second.s64_vector_, true;
}

bool deserialized_op::has_NXC_format() const {
    std::string data_format;
    if (get_attr_string(data_format, "data_format")) {
        return data_format == "NXC";
    } else {
        // these op has default data format nxc, as data_format is optional
        // attribute, if not found, use default data format: nxc
        static const std::unordered_set<std::string> op_has_dflt_nxc_attr {
                "AvgPool", "AvgPoolBackward", "BatchNormForwardTraining",
                "BatchNormInference", "BatchNormTrainingBackward", "BiasAdd",
                "BiasAddBackward", "Convolution", "ConvolutionBackwardData",
                "ConvolutionBackwardWeights", "ConvTranspose",
                "ConvTransposeBackwardData", "ConvTransposeBackwardWeights",
                "Interpolate", "InterpolateBackward", "MaxPool",
                "MaxPoolBackward", "PReLU", "PReLUBackward"};
        return op_has_dflt_nxc_attr.find(name_) != op_has_dflt_nxc_attr.end();
    }
}

logical_tensor::dims deserialized_op::get_NCX_shape(
        size_t idx, bool input) const {
    auto src_dims = input ? in_lts_.at(idx).shape_ : out_lts_.at(idx).shape_;
    if (has_NXC_format()) { change_format_to_ncx(src_dims); }
    return src_dims;
}

void deserialized_graph::load(const std::string &pass_config_json) {
    std::ifstream fs(pass_config_json.c_str());
    utils::json::json_reader_t read(&fs);
    utils::json::read_helper_t helper;
    helper.declare_field("graph", &ops_);
    helper.declare_field("version", &version_);
    helper.declare_field("engine_kind", &engine_kind_);
    helper.declare_field("fpmath_mode", &fpmath_mode_);
    helper.declare_field(
            "fpmath_mode_apply_to_int", &fpmath_mode_apply_to_int_);
    helper.declare_field("input_ports", &input_ports_);
    helper.declare_field("output_ports", &output_ports_);
    helper.read_fields(&read);

    if (ops_.empty()) {
        BENCHDNN_PRINT(
                0, "Error: Graph %s is empty.\n", pass_config_json.c_str());
        SAFE_V(FAIL);
    }

    // Original graph int8 cases come with close-to-real-world values of scales.
    // They are typically f32 random real numbers which doesn't help to validate
    // correctness due to rounding issues in lower precision data types.
    // To work around potential rounding, the code below rounds scales from the
    // case to the nearest pow2 value. E.g. 0.2438485 -> 0.25.
    // Note: it must be done before `ops` are put into `ops_map`.
    for (auto &aop : ops_) {
        if (aop.kind_ != "Dequantize" && aop.kind_ != "Quantize") continue;

        const auto it_attr_scales = aop.attrs_.find("scales");
        const bool has_scales = it_attr_scales != aop.attrs_.end();
        if (!has_scales) continue;

        auto &f32_vector = it_attr_scales->second.f32_vector_;
        for (size_t i = 0; i < f32_vector.size(); i++) {
            const int64_t p
                    = static_cast<int64_t>(std::ceil(std::log2(f32_vector[i])));
            const float new_scale = p >= 0 ? (1LL << p) : (1.f / (1LL << -p));
            f32_vector[i] = new_scale;
        }
    }

    std::map<size_t, size_t> deg; // record indegree for each op
    std::map<size_t, deserialized_op> ops_map; // op_id -> op
    for (const auto &aop : ops_) {
        ops_map[aop.id_] = aop;
        deg[aop.id_] = 0;
        for (const auto &lt : aop.in_lts_) {
            in_lt_2_ops_[lt.id_].push_back(aop);
        }
        for (const auto &lt : aop.out_lts_) {
            out_lt_2_op_[lt.id_] = aop;
            // collect graph internal and output tensors memory layout
            std::string mtag
                    = strides2memory_tag(lt.shape_.size(), lt.stride_, false);
            lt_2_mtag_[lt.id_] = mtag;
        }
    }

    for (const auto &item : in_lt_2_ops_) {
        // count indegree for each op
        // do not count an input if it is an external one since it does not
        // contain an output.
        if (out_lt_2_op_.find(item.first) != out_lt_2_op_.end()) {
            for (const auto &aop : item.second) {
                deg[aop.id_]++;
            }
        }
    }

    ops_.clear();

    for (const auto &item : deg) {
        if (item.second == 0) { ops_.push_back(ops_map[item.first]); }
    }
    for (size_t idx = 0; idx < ops_.size(); idx++) {
        const auto &op = ops_[idx];
        // for each output id of the op, find the ops with the same input id
        // check the input
        for (const auto &out : op.out_lts_) {
            for (const auto &aop : in_lt_2_ops_[out.id_]) {
                deg[aop.id_]--;
                if (deg[aop.id_] == 0) { ops_.push_back(ops_map[aop.id_]); }
            }
        }
    }
    if (ops_map.size() != ops_.size()) {
        BENCHDNN_PRINT(0, "FAIL: the graph %s is not a DAG.\n",
                pass_config_json.c_str());
        SAFE_V(FAIL);
    }

    for (const auto &in_lt : in_lt_2_ops_) {
        if (out_lt_2_op_.find(in_lt.first) != out_lt_2_op_.end()) continue;

        const auto &aop = in_lt_2_ops_[in_lt.first][0];
        for (const auto &lt : aop.in_lts_) {
            if (lt.id_ != in_lt.first) continue;

            graph_tensors_.emplace(in_lt.first, lt.shape_);
            // collect graph input tensors memory layout
            std::string mtag
                    = strides2memory_tag(lt.shape_.size(), lt.stride_, false);
            lt_2_mtag_[lt.id_] = mtag;
        }
    }

    // Keep the object out of the call due to recursion inside the call.
    // Accumulates the state of mb rewrite of nested ops.
    std::unordered_map<size_t, bool> mb_rewrite_ret;
    for (const auto &graph_in : graph_tensors_) {
        if (check_tensor_with_mb(graph_in.first, mb_rewrite_ret)) {
            graph_inputs_with_mb_.push_back(graph_in.first);
        }
    }

    std::cout << "mb rewrite tensors: ";
    for (const auto &id : graph_inputs_with_mb_) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // at this very stage, put all graph_tensors_ id to input_ports_ if
    // even if input_ports_ is not empty
    for (const auto &item : graph_tensors_) {
        input_ports_.emplace_back(item.first);
    }
}

// Prints the lt in the plain string format: `(id):dt:shape`.
std::ostream &operator<<(std::ostream &s, const deserialized_lt &dlt) {
    s << "(" << dlt.id_ << "):" << dlt.data_type_ << ":"
      << lt_dims2str(dlt.shape_);
    return s;
}

std::string deserialized_lt::get_string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

// Prints the op in the plain string format:
// {(id) OpKind}
//     In: { lt0, lt1, ... }
//     Out: { lt0, lt1, ... }
//     Attrs: { Scales: { val0, ... } }  // <-- if any available.
std::ostream &operator<<(std::ostream &s, const deserialized_op &dop) {
    s << "{(" << dop.id_ << ") " << dop.kind_ << "}\n";

    s << "    In: { ";
    for (size_t i = 0; i < dop.in_lts_.size(); i++) {
        s << dop.in_lts_[i];
        if (i != dop.in_lts_.size() - 1) s << ",";
        s << " ";
    }
    s << "}\n";

    s << "    Out: { ";
    for (size_t i = 0; i < dop.out_lts_.size(); i++) {
        s << dop.out_lts_[i];
        if (i != dop.out_lts_.size() - 1) s << ",";
        s << " ";
    }
    s << "}\n";

    const auto it_attr_scales = dop.attrs_.find("scales");
    const auto it_attr_group_shape = dop.attrs_.find("group_shape");
    const bool has_scales = it_attr_scales != dop.attrs_.end();
    const bool has_group_shape = it_attr_group_shape != dop.attrs_.end();

    if (has_scales || has_group_shape) {
        s << "    Attrs: { ";

        if (has_scales) {
            const auto &scales_v = it_attr_scales->second.f32_vector_;
            const auto size = scales_v.size();
            std::string size_str = " (" + std::to_string(size) + ")";
            s << "Scales" << (size > 1 ? size_str : "") << ": { ";
            for (size_t i = 0; i < size; i++) {
                s << scales_v[i];
                if (i != size - 1) s << ",";
                s << " ";
            }
            s << "} "; // Scales
        }

        if (has_group_shape) {
            const auto &group_shape_v = it_attr_group_shape->second.s64_vector_;
            const auto size = group_shape_v.size();
            std::string size_str = " (" + std::to_string(size) + ")";
            s << "Group shape:" << (size > 1 ? size_str : "") << ": { ";
            for (size_t i = 0; i < size; i++) {
                s << group_shape_v[i];
                if (i != size - 1) s << ",";
                s << " ";
            }
            s << "} "; // Group Shape
        }

        s << "}\n"; // Attrs
    }

    return s;
}

std::string deserialized_op::get_string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &operator<<(std::ostream &s, const deserialized_graph &dg) {
    for (const auto &op : dg.ops_) {
        s << op;
    }
    return s;
}

std::string deserialized_graph::get_string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

dnnl::graph::graph deserialized_graph::to_graph(
        const graph_fpmath_mode_t &fpmath_mode) const {
    const auto &engine = get_graph_engine();
    dnnl::graph::graph g(engine.get_kind());
    g.set_fpmath_mode(static_cast<dnnl::fpmath_mode>(
                              str2fpmath_mode(fpmath_mode.mode_.c_str())),
            fpmath_mode.apply_to_int_);

    for (const auto &aop : ops_) {
        try {
            g.add_op(aop.create());
        } catch (const dnnl::error &e) {
            BENCHDNN_PRINT(0, "Error: Adding op %s failed: %s\n",
                    aop.name_.c_str(), e.message);
            SAFE_V(FAIL);
        }
    }
    return g;
}

const deserialized_op &deserialized_graph::get_op(size_t id) const {
    for (const auto &op : ops_) {
        if (op.id_ == id) return op;
    }
    assert(!"Given id was not found in the deserialized graph.");
    static deserialized_op dummy;
    return dummy;
}

const deserialized_op &deserialized_graph::get_op_by_out_lt(
        size_t out_lt_id) const {
    for_(const auto &op : ops_)
    for (const auto &out_lt : op.out_lts_) {
        if (out_lt.id_ == out_lt_id) return op;
    }

    static deserialized_op dummy;
    return dummy;
}

const deserialized_op &deserialized_graph::get_op_by_in_lt(
        size_t in_lt_id) const {
    for_(const auto &op : ops_)
    for (const auto &in_lt : op.in_lts_) {
        if (in_lt.id_ == in_lt_id) return op;
    }

    static deserialized_op dummy;
    return dummy;
}

bool deserialized_graph::check_tensor_with_mb(size_t tensor_id,
        std::unordered_map<size_t, bool> &mb_rewrite_ret) const {
    if (in_lt_2_ops_.find(tensor_id) == in_lt_2_ops_.end()) return true;
    if (mb_rewrite_ret.find(tensor_id) != mb_rewrite_ret.end())
        return mb_rewrite_ret.at(tensor_id);

    const auto broadcast_mb_rewrite
            = [tensor_id](const deserialized_op &aop) -> bool {
        const deserialized_lt *target_lt = nullptr;
        int max_rank = 0;
        for (const auto &lt : aop.in_lts_) {
            max_rank = MAX2(max_rank, static_cast<int>(lt.shape_.size()));
            if (lt.id_ == tensor_id) { target_lt = &lt; }
        }
        return static_cast<int>(target_lt->shape_.size()) == max_rank;
    };

    bool ret = true;
    // TODO: initialize with false to avoid multiple re-initialization
    for (const auto &aop : in_lt_2_ops_.at(tensor_id)) {
        const bool matmul_mb_rewrite = (aop.kind_ == "MatMul")
                && aop.in_lts_[0].shape_.size() > 2
                && (tensor_id == aop.in_lts_[0].id_
                        || tensor_id == aop.in_lts_[1].id_);
        // The second and third inputs of dynamic dequantize are allowed to
        // rewrite md only when the sebsequent op supports mb rewriting.
        std::string qtype;
        const bool dynamicdq_mb_rewrite = (aop.kind_ == "DynamicDequantize")
                && aop.in_lts_[0].shape_.size() > 2
                && check_tensor_with_mb(aop.out_lts_[0].id_, mb_rewrite_ret)
                && aop.get_attr_string(qtype, "qtype")
                && ((qtype == "per_group"
                            && (tensor_id == aop.in_lts_[0].id_
                                    || tensor_id == aop.in_lts_[1].id_
                                    || tensor_id == aop.in_lts_[2].id_))
                        || ((qtype == "per_channel" || qtype == "per_tensor")
                                && tensor_id == aop.in_lts_[0].id_));

        if (std::find(unsupport_mb_rewrite_ops_.begin(),
                    unsupport_mb_rewrite_ops_.end(), aop.kind_)
                != unsupport_mb_rewrite_ops_.end()) {
            // those unsupport op need rewrite dst_shape / weight_shape also
            ret = false;
        } else if (std::find(bwd_ops_.begin(), bwd_ops_.end(), aop.kind_)
                != bwd_ops_.end()) {
            // bwd ops have multiple inputs with mb
            ret = false;
            if (tensor_id == aop.in_lts_[0].id_
                    || tensor_id == aop.in_lts_[1].id_) {
                ret = check_tensor_with_mb(aop.out_lts_[0].id_, mb_rewrite_ret);
                // deal with LayerNormBackward
            } else if (aop.kind_ == "LayerNormBackward"
                    && ((tensor_id == aop.in_lts_[2].id_
                                && aop.in_lts_[2].shape_[0]
                                        == aop.in_lts_[0].shape_[0])
                            || (tensor_id == aop.in_lts_[3].id_
                                    && aop.in_lts_[3].shape_[0]
                                            == aop.in_lts_[0].shape_[0]))) {
                ret = check_tensor_with_mb(aop.out_lts_[0].id_, mb_rewrite_ret);
            }
        } else if (std::find(binary_ops_.begin(), binary_ops_.end(), aop.kind_)
                        != binary_ops_.end()
                || aop.kind_ == "Select") {
            // binary ops need consider rank of 2 inputs
            ret = false;
            if (broadcast_mb_rewrite(aop)) {
                ret = check_tensor_with_mb(aop.out_lts_[0].id_, mb_rewrite_ret);
            }
        } else if (aop.kind_ == "PReLU" && tensor_id == aop.in_lts_[1].id_) {
            // prelu input1 may has same shape with input0
            ret = false;
            if (aop.in_lts_[0].shape_.size() == aop.in_lts_[1].shape_.size()) {
                ret = check_tensor_with_mb(aop.out_lts_[0].id_, mb_rewrite_ret);
            }
        } else if (!(matmul_mb_rewrite || dynamicdq_mb_rewrite
                           || aop.kind_ == "Concat")
                && tensor_id != aop.in_lts_[0].id_) {
            // Do not rewrite if the given tensor is not the first input of the
            // op, except matmul, dynamic dequantize and concat.
            ret = false;
        } else if (aop.kind_ == "End") {
            ret = true;
        } else {
            ret = check_tensor_with_mb(aop.out_lts_[0].id_, mb_rewrite_ret);
        }
    }

    mb_rewrite_ret.emplace(tensor_id, ret);
    return ret;
}

} // namespace graph

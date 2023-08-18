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

#include <vector>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <gtest/gtest.h>
#include <unordered_map>
#include <util/any_map.hpp>

#include "context.hpp"

using namespace dnnl::impl::graph::gc;

namespace llga_fake {
enum class op_kind_t { kconv, kmatmul };
enum class data_type {
    undef = 0,
    f16 = 1,
    bf16 = 2,
    f32 = 3,
    s32 = 4,
    s8 = 5,
    u8 = 6,
};

enum class layout_type {
    undef = 0,
    any = 1,
    strided = 2,
    opaque = 3,
};

struct logical_tensor_t {
    int64_t id;
    int32_t ndims;
    std::vector<int64_t> dims;
    data_type dtype;
    layout_type ltype;
};

class node;
class value_t : public logical_tensor_t {
public:
    class consumer_t {
    public:
        consumer_t(node &op, size_t offset) : op_(&op), offset_(offset) {}
        consumer_t(const consumer_t &c) = default;
        bool operator==(const consumer_t &c) const {
            return op_ == c.op_ && offset_ == c.offset_;
        };
        node &get_op() const { return *op_; }
        size_t get_offset() const { return offset_; }

    private:
        node *op_ = nullptr;
        size_t offset_ = 0;
    };

    value_t(const logical_tensor_t &logical_tensor)
        : logical_tensor_t(logical_tensor) {}
    value_t(node &producer, size_t offset,
            const logical_tensor_t &logical_tensor)
        : logical_tensor_t(logical_tensor)
        , producer_(&producer)
        , offset_(offset) {}
    node *get_producer() const { return producer_; }
    size_t get_offset() const { return offset_; }
    std::vector<consumer_t> get_consumers() const { return consumers_; }
    void set_producer(node &anode) { producer_ = &anode; }
    void set_offset(size_t aoffset) { offset_ = aoffset; }

    void add_consumer(node &op, size_t offset) {
        if (std::find(consumers_.begin(), consumers_.end(),
                    consumer_t(op, offset))
                == consumers_.end()) {
            consumers_.emplace_back(consumer_t {op, offset});
        }
    }

private:
    node *producer_ {nullptr};
    size_t offset_;
    std::vector<consumer_t> consumers_;
};

using value_ptr_t = std::shared_ptr<value_t>;
using value_ptrs_t = std::vector<value_ptr_t>;
class node {
private:
    int64_t id_;
    std::string name_;
    op_kind_t op_kind_;
    std::unordered_map<std::string, any_t> attrs_;

public:
    value_ptrs_t outputs_ {};
    value_ptrs_t inputs_ {};
    node(int64_t id, const std::string &name, op_kind_t op_kind)
        : id_(id), name_(name), op_kind_(op_kind) {
        SC_UNUSED(id_);
    }
    template <typename T>
    void set_attr(const std::string &name, T value) {
        attrs_[name] = value;
    }
    const std::unordered_map<std::string, any_t> &get_attrs() { return attrs_; }
    op_kind_t get_op_kind() const { return op_kind_; }
    const std::string &get_name() { return name_; }

    value_ptr_t get_input_value(size_t offset) const {
        return inputs_.at(offset);
    }
    value_ptr_t get_output_value(size_t offset) const {
        return outputs_.at(offset);
    }

    void add_input(const logical_tensor_t &t) {
        add_input(std::make_shared<value_t>(t));
    }

    void add_input(const value_ptr_t &v) { inputs_.push_back(v); }

    value_ptr_t add_output(const logical_tensor_t &t) {
        value_ptr_t v = std::make_shared<value_t>(*this, outputs_.size(), t);
        outputs_.push_back(v);
        return v;
    }

    void add_output(const value_ptr_t &v) {
        v->set_producer(*this);
        v->set_offset(outputs_.size());
        outputs_.push_back(v);
    }
    /*!
     * \brief Set input of this node.
     * \param offset The index of this node's inputs.
     * \param input_node The input node to this node.
     * \param input_offset The index of the input node's outputs.
     * \return
     */
    void set_input(size_t offset, node &input_node, size_t input_offset) {
        set_input(offset, input_node.get_output_value(input_offset));
    }
    void set_input(size_t offset, const value_ptr_t &output) {
        output->add_consumer(*this, offset);
        if (inputs_.size() <= offset) { inputs_.resize(offset + 1); }
        inputs_[offset] = output;
    }
    value_ptr_t get_output_value(size_t offset) {
        if (offset < outputs_.size()) {
            return outputs_[offset];
        } else {
            value_ptr_t v;
            while (offset >= outputs_.size()) {
                v = std::make_shared<value_t>(*this, outputs_.size(),
                        logical_tensor_t {0, -1, {-1}, data_type::undef,
                                layout_type::undef});
                outputs_.push_back(v);
            }
            return v;
        }
    }

    /*! \brief get number of output tensors of this op */
    size_t num_output_users(size_t offset) const {
        return get_output_value(offset)->get_consumers().size();
    }

    size_t num_output_ops() const {
        size_t num = 0;
        for (const auto &out : outputs_) {
            num += out->get_consumers().size();
        }
        return num;
    }

    size_t num_input_ops() const {
        size_t num = 0;
        for (const auto &in : inputs_) {
            if (in->get_producer() != nullptr) { num++; }
        }
        return num;
    }
};

} // namespace llga_fake

TEST(GCCore_CPU_sc_op_test, add_template_op) {
    // using namespace llga_fake;
    llga_fake::node conv(0, "conv0", llga_fake::op_kind_t::kconv);
    llga_fake::logical_tensor_t conv_src {0, 4,
            std::vector<int64_t> {4, 5, 64, 64}, llga_fake::data_type::f32,
            llga_fake::layout_type::any};
    llga_fake::logical_tensor_t conv_weight {1, 4,
            std::vector<int64_t> {5, 5, 1, 1}, llga_fake::data_type::f32,
            llga_fake::layout_type::any};
    llga_fake::logical_tensor_t conv_dst {2, 4,
            std::vector<int64_t> {4, 5, 64, 64}, llga_fake::data_type::f32,
            llga_fake::layout_type::any};
    conv.add_input(conv_src);
    conv.add_input(conv_weight);
    conv.add_output(conv_dst);
    conv.set_attr<sc_dims>("strides", {1, 1});
    conv.set_attr<sc_dims>("paddings", {0, 0});
    conv.set_attr<sc_dims>("dilations", {1, 1});
    conv.set_attr<std::string>("data_format", "NCX");

    llga_fake::node mm(1, "matmul0", llga_fake::op_kind_t::kmatmul);
    llga_fake::logical_tensor_t mm_src {3, 4,
            std::vector<int64_t> {4, 5, 64, 64}, llga_fake::data_type::f32,
            llga_fake::layout_type::any};
    llga_fake::logical_tensor_t mm_dst {4, 4,
            std::vector<int64_t> {4, 5, 64, 64}, llga_fake::data_type::f32,
            llga_fake::layout_type::any};

    // inputs should be matched or can be broadcasted.
    mm.set_input(0, conv, 0);
    mm.add_input(mm_src);
    mm.add_output(mm_dst);
    mm.set_attr<bool>("transpose_a", true);
    mm.set_attr<bool>("transpose_b", true);

    // llga fake subgraph
    std::vector<llga_fake::node *> node_lists {&conv, &mm};

    // create  llga_fake_node->sc_op mapping
    using sc_op_ptr = std::shared_ptr<sc_op>;
    std::unordered_map<llga_fake::node *, sc_op_ptr> op_mapping;

    // create sc_graph_t
    sc_graph_t opmg;

    // TODO(zhichen): 1. llga and sc logical_tensor order mapping
    // convert llga fake node to sc op
    // should pre-order traverse.
    for (auto llga_node : node_lists) {
        // add connection and create sc op
        std::shared_ptr<sc_op> ret;
        std::vector<graph_tensor_ptr> producer_lt, consumer_lt;
        producer_lt.reserve(llga_node->inputs_.size());
        consumer_lt.reserve(llga_node->outputs_.size());
        for (auto &in_value : llga_node->inputs_) {
            auto scop = in_value->get_producer() == nullptr
                    ? nullptr
                    : op_mapping[in_value->get_producer()];
            if (scop) {
                producer_lt.emplace_back(
                        scop->get_info().outputs_[in_value->get_offset()]);
            } else {
                auto lrt = std::make_shared<graph_tensor>(scop.get(),
                        sc_data_format_t(),
                        sc_dims {in_value->dims.begin(), in_value->dims.end()},
                        sc_data_type_t(sc_data_etype::F32, 1));
                producer_lt.emplace_back(lrt);
            }
        }
        for (auto &out_value : llga_node->outputs_) {
            auto scop = out_value->get_producer() == nullptr
                    ? nullptr
                    : op_mapping[out_value->get_producer()];
            auto lrt = std::make_shared<graph_tensor>(scop.get(),
                    sc_data_format_t(),
                    sc_dims {out_value->dims.begin(), out_value->dims.end()},
                    sc_data_type_t(sc_data_etype::F32, 1));
            consumer_lt.emplace_back(lrt);
        }
        if (llga_node->get_op_kind() == llga_fake::op_kind_t::kconv) {
            ret = opmg.make("conv_fwd_core", producer_lt, consumer_lt,
                    any_map_t(llga_node->get_attrs()));
        } else if (llga_node->get_op_kind() == llga_fake::op_kind_t::kmatmul) {
            ret = opmg.make("matmul", producer_lt, consumer_lt,
                    any_map_t(llga_node->get_attrs()));
        }

        op_mapping[llga_node] = ret;
        if (!llga_node->num_output_ops()) {
            auto foutput = opmg.make_output(consumer_lt);
        }
    }

    ASSERT_TRUE(opmg.ops_.size() == 6);
    for (auto &op : opmg.ops_) {
        if (op->op_name_ == "conv_fwd_core") {
            ASSERT_EQ(op->is_fusible_, false);
            ASSERT_TRUE(op->get_inputs().size() == 2);
            ASSERT_TRUE(op->get_outputs().size() == 1);
            // conv's output op(matmul)'s input_op is "conv_fwd_core"
            ASSERT_EQ(op->get_outputs()[0]->producer_owner_->op_name_,
                    "conv_fwd_core");
            ASSERT_EQ(op->get_info().inputs_[0]->producer_owner_->op_name_,
                    "input");
            ASSERT_EQ(op->get_info().inputs_[1]->producer_owner_->op_name_,
                    "input");
        } else if (op->op_name_ == "matmul") {
            ASSERT_EQ(op->is_fusible_, false);
            ASSERT_TRUE(op->get_inputs().size() == 2);
            ASSERT_TRUE(op->get_outputs().size() == 1);
            ASSERT_EQ(op->get_info().outputs_[0]->producer_owner_->op_name_,
                    "matmul");
            ASSERT_EQ(op->get_info().inputs_[0]->producer_owner_->op_name_,
                    "conv_fwd_core");
            ASSERT_EQ(op->get_info().inputs_[1]->producer_owner_->op_name_,
                    "input");
        } else if (op->op_name_ == "input") {
            ASSERT_TRUE(op->get_inputs().empty());
            ASSERT_TRUE(op->get_outputs().size() == 1);
        } else if (op->op_name_ == "output") {
            ASSERT_TRUE(op->get_inputs().size() == 1);
            ASSERT_TRUE(op->get_outputs().empty());
        } else {
            FAIL();
        }
    }
}

TEST(GCCore_CPU_sc_op_test, add_fusible_op) {
    sc_graph_t opmg;
    std::unordered_map<std::string, any_t> transpose_attrs;
    auto transpose_in1
            = std::make_shared<graph_tensor>(nullptr, sc_data_format_t(),
                    sc_dims {4, 3}, sc_data_type_t(sc_data_etype::F32, 1));
    auto transpose_out
            = std::make_shared<graph_tensor>(nullptr, sc_data_format_t(),
                    sc_dims {3, 4}, sc_data_type_t(sc_data_etype::F32, 1));
    transpose_attrs["order"] = std::vector<int> {1, 0};
    auto mm_op = opmg.make(
            "transpose", {transpose_in1}, {transpose_out}, transpose_attrs);
    ASSERT_TRUE(opmg.ops_.size() == 2);
    for (auto &op : opmg.ops_) {
        if (op->op_name_ == "transpose") {
            ASSERT_EQ(op->isa<fusible_op_t>(), true);
            ASSERT_TRUE(op->get_inputs().size() == 1);
            ASSERT_TRUE(op->get_outputs().size() == 1);
        }
    }
}

#if 0
// TODO(xxx): we should check the result instead of check IR. we may generate
// optimial IR.
TEST(GCCore_CPU_sc_op_test, check_get_func) {
    sc_graph_t opmg;
    auto conv_src = std::make_shared<logical_tensor_t>(nullptr,
            sc_data_format_t(), sc_dims {28, 64, 16, 16},
            sc_dims {28, 64, 16, 16},
            sc_data_type_t(sc_data_etype::F32, 1));
    auto conv_weight = std::make_shared<logical_tensor_t>(nullptr,
            sc_data_format_t(), sc_dims {64, 64, 1, 1},
            sc_dims {64, 64, 1, 1},
            sc_data_type_t(sc_data_etype::F32, 1));
    auto conv_out = std::make_shared<logical_tensor_t>(nullptr,
            sc_data_format_t(), sc_dims {64, 64, 1, 1},
            sc_dims {64, 64, 1, 1},
            sc_data_type_t(sc_data_etype::F32, 1));
    auto relu_out = std::make_shared<logical_tensor_t>(nullptr,
            sc_data_format_t(), sc_dims {64, 64, 1, 1},
            sc_dims {64, 64, 1, 1},
            sc_data_type_t(sc_data_etype::F32, 1));
    std::unordered_map<std::string, any_t> attrs {
            {"strides", sc_dims {1, 1}},
            {"paddings", sc_dims {0, 0}}};
    auto ret = opmg.make("conv_fwd_core", {conv_src, conv_weight}, {conv_out},
            any_map_t(attrs));
    auto f1 = ret->get_func(get_test_ctx());

    ret = opmg.make("relu", {conv_out}, {relu_out}, any_map_t());
    auto f2 = ret->get_func(get_test_ctx());

    builder::ir_builder_t bld;
    _function_(datatypes::void_t, relu_f,
            _arg_("out", datatypes::f32, {64, 64, 1, 1}),
            _arg_("in", datatypes::f32, {64, 64, 1, 1})) {
        _bind_(out, in);
        _for_(i, 0, 64) {
            _for_(j, 0, 64) {
                _for_(k, 0, 1) {
                    _for_(p, 0, 1) {
                        out[{i + 0, j + 0, k + 0, p + 0}] = builder::make_max(
                                in[{i + 0, j + 0, k + 0, p + 0}], 0);
                    }
                }
            }
        }
    }
    ir_comparer ircmp;
    EXPECT_TRUE(ircmp.compare(f2->get_entry_func(), relu_f));
}
#endif

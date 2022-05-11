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

#include "../visitor.hpp"
#include "pass.hpp"
#include <util/json.hpp>
#include <util/reflection.hpp>

namespace sc {
// clang-format off
SC_CLASS(logical_tensor_t)
    SC_FIELD(format_)
    SC_FIELD(plain_dims_)
    SC_FIELD(dtype_)
    SC_FIELD(strides_)
    SC_INITIALIZER(internal_update)
SC_CLASS_END();
// clang-format on

struct graph_ltsr_info_t {
    //<user_op_id_,user_op_pos_> pair
    std::vector<std::pair<int, int>> uses_;
    int tsr_id_ = -1;

    logical_tensor_t &details() const {
        assert(details_ && "Details should not be empty");
        return *details_;
    }

    void init_owned() {
        assert(!owned_);
        owned_ = true;
        details_ = new logical_tensor_t();
    }

    graph_ltsr_info_t() = default;

    graph_ltsr_info_t(int tsr_id, logical_tensor_t &detail) {
        tsr_id_ = tsr_id;
        details_ = &detail;
    }

    graph_ltsr_info_t(graph_ltsr_info_t &&other) { *this = std::move(other); }
    graph_ltsr_info_t &operator=(graph_ltsr_info_t &&other) {
        COMPILE_ASSERT(!owned_, "Should not own the data");
        details_ = other.details_;
        uses_ = std::move(other.uses_);
        tsr_id_ = other.tsr_id_;
        owned_ = other.owned_;
        other.details_ = nullptr;
        return *this;
    }
    ~graph_ltsr_info_t() {
        if (owned_) { delete details_; }
    }

private:
    logical_tensor_t *details_ = nullptr;
    // if owned_, this object is responsible to free the pointers (details_).
    // Otherwise, the pointer is a borrow of existing data
    bool owned_ = false;
};

struct graph_op_info_t {
    int id_ = -1;
    std::vector<int> inputs_;
    std::vector<int> outputs_;

    void init_owned() {
        assert(!owned_);
        owned_ = true;
        // will be freed by ~graph_op_info()
        attrs_ = new any_map_t();
        op_name_ = new std::string();
    }

    std::string &op_name() const {
        assert(op_name_ && "op_name_ should not be null");
        return *op_name_;
    }

    any_map_t &attrs() const {
        assert(attrs_ && "attrs_ should not be null");
        return *attrs_;
    }

    // init an empty op_info
    graph_op_info_t() = default;

    // makes a graph_op_info with references of existing data. It does not has
    // ownership on the pointers. Pass the references to ensure the opname and
    // attrs are not null.
    graph_op_info_t(int id, std::string &opname, any_map_t &attrs) {
        id_ = id;
        owned_ = false;
        op_name_ = &opname;
        attrs_ = &attrs;
    }

    graph_op_info_t(graph_op_info_t &&other) { *this = std::move(other); }
    graph_op_info_t &operator=(graph_op_info_t &&other) {
        COMPILE_ASSERT(!owned_, "Should not own the data");
        id_ = other.id_;
        inputs_ = std::move(other.inputs_);
        outputs_ = std::move(other.outputs_);
        op_name_ = other.op_name_;
        attrs_ = other.attrs_;
        owned_ = other.owned_;
        other.op_name_ = nullptr;
        other.attrs_ = nullptr;
        return *this;
    }
    ~graph_op_info_t() {
        if (owned_) {
            delete op_name_;
            delete attrs_;
        }
    }

private:
    std::string *op_name_ = nullptr;
    any_map_t *attrs_ = nullptr;
    // if owned_, this object is responsible to free the pointers (op_name_,
    // attrs_). Otherwise, the pointer is a borrow of existing data
    bool owned_ = false;
};

namespace json {
template <>
struct handler<graph_ltsr_info_t> {
    static void write(json_writer *writer, const graph_ltsr_info_t &data) {
        writer->begin_object();
        auto ref = reflection::general_ref_t::from(data.details());
        writer->write_keyvalue("detail", ref);
        writer->write_keyvalue("uses", data.uses_);
        writer->end_object();
    }
    static graph_ltsr_info_t read(json_reader *reader) {
        graph_ltsr_info_t data;
        data.init_owned();
        reader->begin_object();
        auto ref = reflection::general_ref_t::from(data.details());
        reader->expect_next_object_key("detail");
        json::handler<reflection::general_ref_t>::read(reader, &ref);
        reader->expect_next_object_key("uses");
        reader->read(&data.uses_);
        reader->expect_object_ends();
        return data;
    }
};

template <>
struct handler<graph_ltsr_info_t *> {
    using ptr_t = graph_ltsr_info_t *;
    static void write(json_writer *writer, const ptr_t &data) {
        handler<graph_ltsr_info_t>::write(writer, *data);
    }
};

template <>
struct handler<graph_op_info_t> {
    static void write(json_writer *writer, const graph_op_info_t &data) {
        writer->begin_object();
        writer->write_keyvalue("op", data.op_name());
        writer->write_keyvalue("inputs", data.inputs_);
        writer->write_keyvalue("outputs", data.outputs_);
        writer->write_keyvalue("attrs", data.attrs().as_map());
        writer->end_object();
    }
    static graph_op_info_t read(json_reader *reader) {
        graph_op_info_t data;
        // let graph_op_info take the ownership of the data
        data.init_owned();
        reader->begin_object();
        reader->expect_next_object_key("op");
        reader->read(&data.op_name());
        reader->expect_next_object_key("inputs");
        reader->read(&data.inputs_);
        reader->expect_next_object_key("outputs");
        reader->read(&data.outputs_);
        reader->expect_next_object_key("attrs");
        reader->read(&data.attrs().as_map());
        reader->expect_object_ends();
        return data;
    }
};
} // namespace json

static void dump_info_to_json(json::json_writer &writer,
        std::vector<graph_ltsr_info_t *> &id_ltsr,
        std::vector<graph_op_info_t> &id_op, const any_map_t &graph_attrs) {
    writer.begin_object();
    writer.write_keyvalue("tensors", id_ltsr);
    writer.write_keyvalue("ops", id_op);
    writer.write_keyvalue("graph_attrs", graph_attrs.as_map());
    writer.end_object();
}

static void load_info_from_json(json::json_reader &reader,
        std::vector<graph_ltsr_info_t> &id_ltsr,
        std::vector<graph_op_info_t> &id_op, any_map_t &graph_attrs) {
    reader.begin_object();
    reader.expect_next_object_key("tensors");
    reader.read(&id_ltsr);
    reader.expect_next_object_key("ops");
    reader.read(&id_op);
    reader.expect_next_object_key("graph_attrs");
    reader.read(&graph_attrs.as_map());
    reader.expect_object_ends();
}

void save_graph_to_json(const sc_graph_t &graph, json::json_writer &writer) {
    // todo: logical tensor should also have an unique id
    std::unordered_map<graph_tensor_ptr, graph_ltsr_info_t> ltsr_id;
    std::vector<graph_ltsr_info_t *> id_ltsr;
    std::vector<graph_op_info_t> id_op;
    auto get_tensor_id = [&](const graph_tensor_ptr &t) -> graph_ltsr_info_t * {
        auto itr = ltsr_id.find(t);
        if (itr != ltsr_id.end()) { return &itr->second; }
        int id = id_ltsr.size();
        auto &ret = ltsr_id[t];
        ret = graph_ltsr_info_t(id, t->details_);
        for (auto &use : t->uses_) {
            ret.uses_.emplace_back(
                    std::make_pair(use.second->logical_op_id_, use.first));
        }
        id_ltsr.emplace_back(&ret);
        return &ret;
    };

    for (size_t i = 0; i < graph.ops_.size(); i++) {
        auto &op = graph.ops_[i];
        COMPILE_ASSERT((int)i == op->logical_op_id_, "Bad Op id");
        id_op.emplace_back(op->logical_op_id_, op->op_name_, op->attrs_);
        auto &out_info = id_op.back();
        if (!op->is_removed_) {
            for (auto &t : op->get_inputs()) {
                out_info.inputs_.push_back(get_tensor_id(t)->tsr_id_);
            }
            for (auto &t : op->get_outputs()) {
                out_info.outputs_.push_back(get_tensor_id(t)->tsr_id_);
            }
        }
    }
    dump_info_to_json(writer, id_ltsr, id_op, graph.attrs_);
}

void save_graph_to_json(const sc_graph_t &graph, std::ostream &os) {
    json::json_writer writer {&os, /*pretty_print*/ true, /*skip_temp*/ true};
    save_graph_to_json(graph, writer);
}

sc_graph_t load_graph_from_json(json::json_reader &reader) {
    std::vector<graph_ltsr_info_t> id_ltsr;
    std::vector<graph_op_info_t> id_op;
    std::vector<graph_tensor_ptr> ltsrs;
    any_map_t graph_attrs;
    load_info_from_json(reader, id_ltsr, id_op, graph_attrs);
    sc_graph_t graph;
    for (size_t i = 0; i < id_ltsr.size(); i++) {
        auto &tsr_info = id_ltsr[i];
        ltsrs.emplace_back(
                std::make_shared<graph_tensor>(nullptr, tsr_info.details()));
    }
    for (size_t i = 0; i < id_op.size(); i++) {
        auto &op_info = id_op[i];
        std::vector<graph_tensor_ptr> ins;
        std::vector<graph_tensor_ptr> outs;
        for (auto idx : op_info.inputs_) {
            ins.emplace_back(ltsrs.at(idx));
        }
        for (auto idx : op_info.outputs_) {
            outs.emplace_back(ltsrs.at(idx));
        }

        if (op_info.op_name() == "input") {
            auto ins = graph.make_input(outs);
            ins->attrs_ = op_info.attrs();
        } else if (op_info.op_name() == "output") {
            auto outs = graph.make_output(ins);
            outs->attrs_ = op_info.attrs();
        } else {
            graph.make(op_info.op_name(), ins, outs, op_info.attrs());
        }
    }
    for (size_t i = 0; i < id_ltsr.size(); i++) {
        auto &tsr_info = id_ltsr[i];
        auto &tsr = ltsrs[i];
        COMPILE_ASSERT(tsr->uses_.size() == tsr_info.uses_.size(),
                "Expecting tsr->uses_.size()==tsr_info.uses_.size()");
        for (size_t j = 0; j < tsr_info.uses_.size(); j++) {
            auto &kv = tsr_info.uses_[j];
            //<user_op_id_,user_op_pos_> pair
            tsr->uses_[j].first = kv.second;
            tsr->uses_[j].second = graph.ops_.at(kv.first);
        }
    }
    graph.attrs_ = std::move(graph_attrs);
    return graph;
}

sc_graph_t load_graph_from_json(std::istream &is) {
    json::json_reader reader {&is};
    return load_graph_from_json(reader);
}
} // namespace sc

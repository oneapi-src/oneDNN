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
#include "graph_repository.hpp"
#include <compiler/ir/graph/pass/load_store.hpp>
#include <util/reflection.hpp>

namespace sc {
namespace graph {

repository repository::load(const context_ptr &ctx, std::istream &is) {
    repository ret;
    if (!is) { return ret; }
    json::json_reader reader {&is};
    reader.begin_object();
    reader.expect_next_object_key("graphs");
    // begin parsing the graphs
    reader.begin_array();
    while (reader.next_array_item()) {
        // begin parsing a single repository_entry
        reader.begin_object();
        auto entry = utils::make_unique<repository_entry>();

        reader.expect_next_object_key("name");
        entry->name_ = reader.read_string();
        reader.expect_next_object_key("graph");
        sc_graph_t g = reader.read<sc_graph_t>();
        reader.expect_next_object_key("cost");
        reader.read(&entry->cost_);
        reader.expect_next_object_key("config");
        auto ref = reflection::general_ref_t::from(entry->config_);
        reader.read(&ref);
        // end parsing a single repository_entry
        reader.expect_object_ends();
        ret.entries_.emplace(std::move(g), std::move(entry));
    }
    // end parsing the graphs
    reader.expect_object_ends();
    return ret;
}

void repository::store(const context_ptr &ctx, std::ostream &os) const {
    json::json_writer writer {&os, /*pretty_print*/ true, /*skip_temp*/ true};
    writer.begin_object();
    writer.write_key("graphs");
    // begin parsing the graphs
    writer.begin_array();
    for (auto &entry : entries_) {
        writer.write_array_seperator();
        // begin parsing a single repository_entry
        writer.begin_object();
        writer.write_key("name");
        writer.write_string(entry.second->name_);
        writer.write_key("graph");
        writer.write(entry.first);
        writer.write_key("cost");
        writer.write(entry.second->cost_);
        writer.write_key("config");
        auto ref = reflection::general_ref_t::from(entry.second->config_);
        writer.write(ref);
        writer.end_object();
    }
    writer.end_array();

    // end parsing the graphs
    writer.end_object();
}

void repository::add(const sc_graph_t &g, const std::string &name, float cost,
        const graph_config &config) {
    auto ent = utils::make_unique<repository_entry>();
    ent->name_ = name;
    ent->cost_ = cost;
    ent->config_ = config;
    entries_.emplace(copy_graph(g), std::move(ent));
}

repository_entry *repository::find(const sc_graph_t &g) const {
    repository_entry *ret = nullptr;
    auto pair_iter = entries_.find(g);
    if (pair_iter != entries_.end()) { ret = (pair_iter->second).get(); }
    return ret;
}

} // namespace graph

} // namespace sc

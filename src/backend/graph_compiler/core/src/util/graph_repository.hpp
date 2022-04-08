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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_GRAPH_REPOSITORY_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_UTIL_GRAPH_REPOSITORY_HPP
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/graph_config.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <unordered_map>

// custom specialization of std::hash can be injected in namespace std
namespace std {
template <>
struct hash<sc::sc_graph_t> {
    size_t operator()(sc::sc_graph_t const &g) const noexcept {
        return g.hash_contents();
    }
};

template <>
struct equal_to<sc::sc_graph_t> {
    bool operator()(
            sc::sc_graph_t const &lhs, sc::sc_graph_t const &rhs) const {
        return compare_graph(lhs, rhs);
    }
};
} // namespace std

namespace sc {
namespace graph {

struct repository_entry {
    // optional human-readable name of the graph. Can be empty
    std::string name_;
    // the cost of the config. Lower is better
    float cost_;
    // the best config
    graph_config config_;
};

// the data structure to manage the graphs and their best configs
struct SC_INTERNAL_API repository {
    std::unordered_map<sc_graph_t, std::unique_ptr<repository_entry>,
            std::hash<sc_graph_t>, std::equal_to<sc::sc_graph_t>>
            entries_;

    void add(const sc_graph_t &g, const std::string &name, float cost,
            const graph_config &config);

    repository_entry *find(const sc_graph_t &g) const;

    static repository load(const context_ptr &ctx, std::istream &is);
    void store(const context_ptr &ctx, std::ostream &os) const;

    repository() = default;
    repository(repository &&) = default;
    repository &operator=(repository &&other) {
        entries_ = std::move(other.entries_);
        return *this;
    }
};
} // namespace graph
} // namespace sc
#endif

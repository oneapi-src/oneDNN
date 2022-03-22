/*******************************************************************************
 * Copyright 2021-2022 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_PASS_LOAD_STORE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_PASS_LOAD_STORE_HPP

#include "../graph.hpp"
#include <util/json.hpp>

namespace sc {
sc_graph_t load_graph_from_json(json::json_reader &reader);
void save_graph_to_json(const sc_graph_t &graph, json::json_writer &writer);

namespace json {
template <>
struct handler<sc_graph_t> {
    inline static void write(json_writer *writer, const sc_graph_t &data) {
        save_graph_to_json(data, *writer);
    }
    inline static void read(json_reader *reader, sc_graph_t *data) {
        *data = load_graph_from_json(*reader);
    }
};
} // namespace json

} // namespace sc
#endif

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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_CONFIG_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_CONFIG_HPP

#include <memory>
#include <string>
#include <vector>
#include "graph.hpp"
#include "util/general_object.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// todo(zhichen): replaced by any map
struct graph_config {
    std::vector<reflection::shared_general_object_t> op_cfgs_;
    // maybe anther config item in the future
};
namespace tuner {
struct config_space;
} // namespace tuner

namespace graph {
SC_INTERNAL_API graph_config get_graph_default_config(
        context_ptr ctx, const sc_graph_t &g);
SC_INTERNAL_API void set_graph_config(
        sc_graph_t &g, const graph_config &config);
} // namespace graph
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

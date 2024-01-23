/*******************************************************************************
 * Copyright 2023-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_BINDING_AXIS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_BINDING_AXIS_HPP

#include <memory>
#include <string>
#include <vector>
#include <compiler/ir/sc_stmt.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace binding_axis_attr {
// global binding axis map cache, attached on the graph
constexpr const char *global_map_cache = "global_map_cache";
// loop binding axis hint, attached on the for loop node
constexpr const char *loop_hint = "loop_hint";
}; // namespace binding_axis_attr

class sc_graph_t;
struct graph_tensor;
struct logical_tensor_t;

using binding_axis = std::vector<std::vector<int>>;
template <typename valT>
struct gt_map_t;
using binding_axis_map = gt_map_t<binding_axis>;

using global_binding_axis_map = std::unordered_map<uintptr_t,
        std::unordered_map<uintptr_t, binding_axis>>;

// The identity based on the topology of graph
struct graph_identity {
    std::vector<std::string> op_names_;
    std::vector<logical_tensor_t> op_lts_;
    size_t hash_;
    // compare graph identity
    bool operator==(const graph_identity &other) const;
};

struct global_map_cache {
    // global map ptr
    std::shared_ptr<global_binding_axis_map> global_map_ptr_;
    // ensure consistency of global mapping ptr
    graph_identity identity_;
};

struct loop_binding_axis_hint {
    // global map ptr
    std::shared_ptr<global_binding_axis_map> global_map_ptr_;
    // key: the address of graph tensor that generate loops
    uintptr_t key_;
    // axis of graph tensor's plain dims
    std::vector<int> axis_;
};

// query binding axis based on give graph and set result on the attr of graph
void query_binding_axis(sc_graph_t &g);

// the standart binding axis interface
void bind_loop_axis(const std::shared_ptr<graph_tensor> &gt,
        const for_loop &loop, const std::vector<int> &axis,
        bool is_block = false);
// friendly to development
void bind_loop_axis(const std::shared_ptr<graph_tensor> &gt, const stmt &loop,
        int axis, bool is_block = false);
void bind_loop_axis(const std::shared_ptr<graph_tensor> &gt,
        const std::vector<for_loop> &loops, const std::vector<int> &axis,
        bool is_block = false);

// compare loop with binding axis
bool check_loop_binding_axis(
        const for_loop_node_t *loop_a, const for_loop_node_t *loop_b);
bool check_loop_binding_axis(const for_loop &loop_a, const for_loop &loop_b);
// compare loop with binding axis and return aligned loop num
int check_loop_binding_axis(const std::vector<for_loop> &loop_a,
        const std::vector<for_loop> &loop_b, int64_t check_loop_size = -1);

// check whether the given loop has axis of graph tensor
bool check_loop_has_axis(const for_loop &loop,
        const std::shared_ptr<graph_tensor> &gt, const std::vector<int> &axis);

/* used for loop transform */
// copy binding axis hint when loop split
void copy_binding_axis_hint(const for_loop_node_t *ths, for_loop_node_t *other);
// merge binding axis hint when loop fuse
void fuse_binding_axis_hint(
        const for_loop_node_t *ths, const for_loop_node_t *other);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

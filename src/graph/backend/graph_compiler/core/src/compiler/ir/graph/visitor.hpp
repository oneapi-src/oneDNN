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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_VISITOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_VISITOR_HPP

#include <functional>
#include <list>
#include <memory>
#include <numeric>
#include <vector>
#include <compiler/ir/graph/graph.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * The util for traversing the OP graph. It contains key functions: selector and
 * updater. Selector will pick the nodes to visit from the "queue": `to_visit_`.
 * Updater is called after visiting a node, which "pushes" the nodes to visit in
 * the `to_visit_` "queue". Different selectors and updaters results in
 * different visiting orders. This class has some pre-defined selectors and
 * updaters.
 * Note: `op_visitor_t::visit_graph()` can be called only once. Please create a
 * new object for each visit.
 * */
class SC_INTERNAL_API op_visitor_t {
public:
    // the queue/stack for the nodes to visit
    std::list<sc_op_ptr> to_visit_;
    // the array to memorize the nodes that we have visited, indexed by the op
    // id
    std::vector<bool> visited_;

    using visitor_func = std::function<void(op_visitor_t *, sc_op_ptr)>;
    using updater_func = std::function<void(op_visitor_t *, sc_op_ptr)>;
    using selector_func = std::function<sc_op_ptr(op_visitor_t *)>;
    // the selector to return the next node to visit in `to_visit_` list. It
    // should also remove the node from the list. It can return null if it finds
    // a node that has been visited. The visitor will try to call it again
    selector_func select_next_node;
    // will be called after a node has been visited. Usually it should update
    // the `visited_`, and push/enqueue the sub-nodes to the `to_visit_`
    updater_func update_visit_list;
    // if true, when the user call `visit_graph()` or `post_visit_graph()`, we
    // will check whether all ops in the graph are visited (not including the
    // ops which are newly created and inserted into the graph during the
    // visit). If any ops are not visited, an exception will be thrown
    const bool check_all_ops_visited_;

    void visit(const visitor_func &f);

    // set a node as visited
    void set_visited(int id);
    // returns if an id is in the visited node set
    bool has_visited(int id);

    void visit_graph(const sc_graph_t &mgr, const visitor_func &f);

    op_visitor_t(selector_func select_next_node_func,
            updater_func update_visit_list_func, bool check_all_ops_visited);

    // updates the visitor states after a node is visited. It can be also used
    // when a new node replaces an old one. Users should call this function with
    // the new node
    void update_state_for_visited(sc_op_ptr node);

    // the updater which pushes all uses of all output logical tensors
    // to the back of the to_visit list
    static void push_back_updater(op_visitor_t *, const sc_op_ptr &sc_op_ptr);

    // For DAG updater, its user visit order can be specified as demand, the
    // return vector is the index of multi-users
    using user_sort_func
            = std::function<std::vector<int>(const graph_tensor_ptr &gt)>;
    // the updater which pushes all nodes whose dependencies have already been
    // visited. Used in topology sort. The default user order is sorted by
    // ascend
    static updater_func create_DAG_updater(
            size_t total_nodes_hint,
            const user_sort_func &func
            = [](const graph_tensor_ptr &gt) -> std::vector<int> {
                std::vector<int> usr_vis_ord(gt->uses_.size());
                std::iota(usr_vis_ord.begin(), usr_vis_ord.end(), 0);
                return usr_vis_ord;
            });
    static updater_func create_DAG_updater_post(size_t total_nodes_hint);
    // Different from default create_DAG_updater, it has specific user sort func
    static updater_func create_DAG_updater_speculative(size_t total_nodes_hint);
    // create_DAG_updatater_post;
    // post order traversing
    void post_visit_graph(const sc_graph_t &mgr, const visitor_func &f);

    // the selector which pops a node in `to_visit_` from back
    static sc_op_ptr pop_back_selector(op_visitor_t *v);

    // the selector which pops a node in `to_visit_` from front
    static sc_op_ptr dequeue_selector(op_visitor_t *v);

    // constructs a DFS visitor, using push_back_updater and
    // pop_back_selector
    static op_visitor_t dfs();
    // constructs a BFS visitor, using push_back_updater and
    // dequeue_selector
    static op_visitor_t bfs();
    // constructs a topology sort visitor in DFS order, using
    // create_DAG_updater and pop_back_selector
    static op_visitor_t dfs_topology_sort(size_t total_nodes_hint = 30);
    static op_visitor_t dfs_topology_speculative_sort(
            size_t total_nodes_hint = 30);
    // constructs a topology sort visitor in BFS order, using
    // create_DAG_updater and dequeue_selector
    static op_visitor_t bfs_topology_sort(size_t total_nodes_hint = 30);
    // constructs a BFS visitor, and do not check whether all ops are visited
    static op_visitor_t bfs_unchecked();
    // constructs a topology sort visitor in BFS order, and do not check whether
    // all ops are visited
    static op_visitor_t dfs_topology_sort_unchecked(
            size_t total_nodes_hint = 30);

private:
    // asserts that all ops in the graph whose index is within
    // [0, concerned_size) have been visited
    void assert_all_ops_visited(const sc_graph_t &mgr, size_t concerned_size);
};

/** Op Depenency Matrix
 * This class is used to record depenency relationship between all graph
 * ops. For n Ops in graph, it will create nxn matrix, in which:
 * 1. <i,j> = 1 represents the j-th OP depends on i-th OP
 * 2. <i,j> = -1 represents the i-th OP depends on j-th OP
 * 3. <i,j> = 0 represents i-th OP and j-th OP have no depenency
 * @note both i and j are logical_op_id_ of original
 * whole graph.
 * */
class op_dep_matrix_t {
private:
    std::vector<std::vector<int>> matrix_;
    int op_size_;

public:
    op_dep_matrix_t(const sc_graph_t &);
    // initlize op_size x op_size matrix by default value of zero
    op_dep_matrix_t(int op_size)
        : matrix_(op_size, std::vector<int>(op_size, 0)), op_size_(op_size) {}
    // update depenency matrix according topology sort
    void update(const sc_op_ptr &cur);
    /** lookup function for matrix, it will return depenency relationship
     * return 1 represents j-th op depends on i-th op.
     * return -1 represents i-th op depends on j-th op
     * return  0 represents i-th op and j-th op have no depenency
     * */
    int lookup(int i, int j) const {
        COMPILE_ASSERT(i >= 0 && i < op_size_ && j >= 0 && j < op_size_,
                "illegal lookup index for depenency matrix.");
        return matrix_[i][j];
    }

    int lookup(sc_op *op_i, sc_op *op_j) const {
        int i = op_i->logical_op_id_, j = op_j->logical_op_id_;
        return lookup(i, j);
    }

    int lookup(const sc_op_ptr &op_i, const sc_op_ptr &op_j) const {
        return lookup(op_i.get(), op_j.get());
    }

    // look up all op id which depend on given op id
    std::vector<int> lookup_ops_depend_on(int depend_i) const {
        COMPILE_ASSERT(depend_i >= 0 && depend_i < op_size_,
                "illegal lookup index for depenency matrix.");
        std::vector<int> ret;
        for (int j = 0; j < op_size_; j++) {
            if (matrix_[depend_i][j] == 1) { ret.emplace_back(j); }
        }
        return ret;
    }
};

using dep_mat_ptr = std::shared_ptr<op_dep_matrix_t>;

/**
 * op_sorting_visitor_t is sort based, and visiting order is defined by the user
 * sort function or rules. op_visitor_t is queue/iteration based.
 * */
class op_sorting_visitor_t {
public:
    /**
     * pre-defined optimize rules are listed as enumurate type, users can
     * directly use this enumuate kind rules. Actually, it will call pre-defined
     * create_xxx_rule function. @note multiple optimzied rules are supported,
     * so these rules would be sorted by priority if more than one rules are
     * used.
     * */
    enum class sort_rule : int {
        same_kind = 0,
        // TODO(xxx): Add other pre-defined rules here.
    };

    /**
     * user-defined optimize rules are the lambda function, which will accept
     * two arguements. This function is expected to reorder op_seq according
     * given rules and dep_matrix.
     * @param op_seq: op sequence generated by op_visitor_t.
     * @param dep_matrix: op depenency matrix.
     * */
    using rule_func = std::function<void(
            std::vector<sc_op_ptr> &op_seq, const op_dep_matrix_t &dep_matrix)>;

    /** Visit topology sequence by given rules.
     * This function will split this into three stages. First, it will visit all
     * ops in graph.ops and generate initial sequence. Meanwhile, it will record
     * its dependency in a Adjacent Matrix. Secondly, it may change the op
     * visiting order by the given visit rules. Finally, all ops will be excuted
     * by new order.
     * @note This visit function have two versions: one is suitable for
     * pre-defined rules, and another is more suitable for user-defined rule.
     * @param graph: sc_graph object
     * @param f: visiting function for each op in graph
     * @param rules: (Version 1) the list of pre-defined rules, it will be
     * reordered by priority inside.
     * @param f_fule (Version 2) user-defined lambda rule function.
     * */
    static void visit_by_rules(sc_graph_t &graph,
            const std::function<void(sc_op_ptr)> &f,
            std::vector<sort_rule> rules);
    static void visit_by_rules(sc_graph_t &graph,
            const std::function<void(sc_op_ptr)> &f, const rule_func &f_rule);

    static std::vector<sc_op_ptr> sort_by_rules(
            sc_graph_t &graph, std::vector<sort_rule> rules);
    static std::vector<sc_op_ptr> sort_by_rules(
            sc_graph_t &graph, const rule_func &f_rule);

    /** create pre-defined same_kind rules
     * Due to same_kind op like elementwise_op, broadcast_op_t or reduce_op_t
     * may have generate similar loop in IR, which will expose more opportunity
     * for Loop Merge Pass, we can make them as neighboring as possible.
     * */
    static rule_func create_same_kind_rule();
};

// count amount of tuneable op
int count_tuneop_linearly(const sc_op_ptr &start_node, int step);

// search first tunable op linearly
sc_op_ptr search_tuneop_linearly(const sc_op_ptr &start_node, int max_step = 5);

/**
 * What is bypass: it starts from the certain op which has more than one user
 * ops, and one of them is tunable op. it means fuse op pass will reparition
 * fused graph starting that tunable op.
 * E.g.
 *       in_a  in_b
 *         \    /
 *        matmul2d   in_c
 *           |      /
 *          bias
 *           |
 *    in_d  quan
 *      \    |       \
 *        matmul2d   dequan
 *           |       /
 *          add
 *           |
 *         output
 *
 * In graph above, `deq` or `cast+sub+mul`(after graph inline) is the bypass
 * what we want to search. For each op found in bypsas, it can be fused either
 * previous or post op, global reschedule is aimed to mark suitable fuse attr
 * (break pre/post fuse or even no fused) for each op them according several
 * different rules.
 * */
std::vector<sc_op_ptr> search_tuneop_bypass(const context_ptr &ctx,
        const sc_op_ptr &tuneop, const sc_op_ptr &start_node,
        const op_dep_matrix_t &dep, int max_step = 10);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

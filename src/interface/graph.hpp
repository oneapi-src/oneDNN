/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_INTERFACE_GRAPH_HPP
#define LLGA_INTERFACE_GRAPH_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.h"

#include "common.hpp"
#include "engine.hpp"
#include "ir.hpp"
#include "op_schema.hpp"
#include "utils/compatible.hpp"

struct dnnl_graph_graph : public dnnl_graph_id, public llga::impl::attributes {
    using node_ptr = std::unique_ptr<llga::impl::node_t>;

private:
    /*! \brief nodes in this graph */
    std::vector<node_ptr> nodes_;

    /*! \brief added llga ops*/
    std::vector<llga::impl::op_t> ops_;

    /*! \brief The engine kind on which the operator will be evaluated */
    llga::impl::engine_kind_t engine_kind_;

public:
    dnnl_graph_graph(
            llga::impl::engine_kind_t kind = llga::impl::engine_kind::cpu)
        : engine_kind_(kind) {};

    dnnl_graph_graph(const dnnl_graph_graph &other) = delete;
    dnnl_graph_graph &operator=(const dnnl_graph_graph &other) = delete;

    ~dnnl_graph_graph() = default;

    /*!
     * \brief Check whether a llga operator can be added
     * \param l_n A llga operator in frameworks' graph.
     * \return Whether the operator is supported
     */
    llga::impl::status_t add_op(llga::impl::op_t *l_n) {
        if (std::none_of(ops_.begin(), ops_.end(),
                    [&l_n](const std::vector<llga::impl::op_t>::value_type
                                    &op) { return op.id() == l_n->id(); })) {
            const llga::impl::op_schema *opm
                    = llga::impl::op_schema_registry::get_op_schema(
                            l_n->kind());
            llga::impl::op_t tmp_ln = *l_n;
            if (opm != nullptr) {
                opm->set_default_attribute(&tmp_ln);
                if (!opm->verify(&tmp_ln)) {
                    return llga::impl::status::invalid_op;
                }
            }
            ops_.emplace_back(tmp_ln);
        }
        return llga::impl::status::success;
    }

    /*!
     * \brief Create and add a node to this graph.
     * \param aop_kind The operator used to create the node
     * \return node* created node
     */
    llga::impl::node_t *create_node(llga::impl::op_kind_t aop_kind) {
        nodes_.push_back(
                llga::impl::utils::make_unique<llga::impl::node_t>(aop_kind));
        return nodes_.back().get();
    }

    /*!
     * \brief Create and add a node to this graph.
     * \param lop The llga op used to create the node
     * \return node* created node
     */
    llga::impl::node_t *create_node(const llga::impl::op_t &lop) {
        for (const node_ptr &n : nodes_) {
            // there must be only one llga op id
            // while building graph
            if (n->get_op_ids().front() == lop.id()) {
                n->parse_op_attr(&lop);
                return n.get();
            }
        }
        node_ptr anode = llga::impl::utils::make_unique<llga::impl::node_t>(
                lop.id(), lop.kind());
        anode->parse_op_attr(&lop);
        anode->add_op_ids(lop.id());
        anode->add_input_tensors(lop.inputs());
        anode->add_output_tensors(lop.outputs());
        const auto &it = nodes_.insert(nodes_.end(), std::move(anode));
        return it->get();
    }

    /*!
     * \brief Delete a node of this graph.
     * \param anode The node to be deleted
     * \return void
     */
    void delete_node(llga::impl::node_t *anode) {
        std::vector<node_ptr>::iterator pos = std::find_if(nodes_.begin(),
                nodes_.end(), [anode](const node_ptr &n) -> bool {
                    return n.get() == anode;
                });
        if (pos != nodes_.end()) nodes_.erase(pos);
    }

    /*!
     * \brief Get all the nodes of this graph.
     * \return vector of nodes pointers
     */
    const std::vector<node_ptr> &get_nodes() const { return nodes_; }

    /*! \brief get num_inputs of this node */
    size_t num_nodes() const { return nodes_.size(); }

    /*!
     * \brief Get the input nodes of this graph.
     * \return vector of input nodes pointers
     */
    std::vector<llga::impl::node_t *> get_inputs() {
        std::vector<llga::impl::node_t *> inputs;
        for (const node_ptr &n : nodes_) {
            if (n->num_inputs() == 0) { inputs.push_back(n.get()); }
        }
        return inputs;
    }

    /*!
     * \brief Get the output nodes of this graph.
     * \return vector of output nodes pointers
     */
    std::vector<llga::impl::node_t *> get_outputs() {
        std::vector<llga::impl::node_t *> outputs;
        for (const node_ptr &n : nodes_) {
            if (n->num_outputs() == 0) { outputs.push_back(n.get()); }
        }
        return outputs;
    }

    /*!
     * \brief execute graph pass
     * \param policy Partition policy
     * \return llga result
     */
    llga::impl::status_t run_pass(llga::impl::partition_policy_t policy);

    /*!
     * \brief Get partition numbers
     * \return partition numbers
     */
    size_t get_num_partitions() const {
        return static_cast<size_t>(std::count_if(begin(nodes_), end(nodes_),
                [](const node_ptr &n) { return n->has_attr("backend"); }));
    };

    /*!
     * \brief get list of partitions
     * \param list of partitions
     */
    void get_partitions(std::vector<dnnl_graph_partition *> &partitions);

    /*!
     * \brief Build backend graph after add op is done
     */
    llga::impl::status_t build_graph();

    void visualize(std::string filename);
};

#endif

/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef INTERFACE_GRAPH_HPP
#define INTERFACE_GRAPH_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "oneapi/dnnl/dnnl_graph.h"

#include "c_types_map.hpp"
#include "common.hpp"
#include "engine.hpp"
#include "id.hpp"
#include "op.hpp"
#include "op_schema.hpp"
#include "utils/compatible.hpp"

struct dnnl_graph_graph : public dnnl_graph_id {
    using op_t = dnnl::graph::impl::op_t;
    using op_ptr = std::shared_ptr<op_t>;

private:
    /*! \brief added ops*/
    std::vector<op_ptr> ops_ {};

    /*! \brief The engine kind on which the operator will be evaluated */
    dnnl::graph::impl::engine_kind_t engine_kind_;

    bool is_built_ {false};

public:
    dnnl_graph_graph(dnnl::graph::impl::engine_kind_t kind
            = dnnl::graph::impl::engine_kind::cpu)
        : engine_kind_(kind) {};

    dnnl_graph_graph(const dnnl_graph_graph &other) = delete;
    dnnl_graph_graph &operator=(const dnnl_graph_graph &other) = delete;

    ~dnnl_graph_graph() = default;

    dnnl::graph::impl::engine_kind_t get_engine_kind() const {
        return engine_kind_;
    }

    /*!
     * \brief Check whether an operator can be added
     * \param l_n An operator in frameworks' graph.
     * \return Whether the operator is supported
     */
    dnnl::graph::impl::status_t add_op(const op_t *l_n) {
        if (!l_n) return dnnl::graph::impl::status::invalid_op;

        if (std::none_of(ops_.begin(), ops_.end(),
                    [&l_n](const std::vector<op_ptr>::value_type &op) {
                        return op->get_id() == l_n->get_id();
                    })) {
            const dnnl::graph::impl::op_schema *opm
                    = dnnl::graph::impl::op_schema_registry::get_op_schema(
                            l_n->get_kind());
            op_t tmp_ln = *l_n;
            if (opm != nullptr) {
                opm->set_default_attribute(&tmp_ln);
                if (!opm->verify(&tmp_ln)) {
                    return dnnl::graph::impl::status::invalid_op;
                }
            }
            ops_.push_back(std::make_shared<op_t>(tmp_ln));
            auto back_op = ops_.back().get();
            for (size_t i = 0; i < back_op->num_outputs(); i++)
                back_op->get_output_value(i)->set_producer(*back_op);
        }
        return dnnl::graph::impl::status::success;
    }

    op_t *create_op(dnnl_graph_op_kind_t kind, std::string name = "") {
        ops_.push_back(std::make_shared<op_t>(kind, std::move(name)));
        return ops_.back().get();
    }

    void delete_op(op_t *op) {
        if (!op) return;

        auto pos = std::find_if(ops_.begin(), ops_.end(),
                [op](const op_ptr &n) -> bool { return *n == *op; });
        if (pos != ops_.end()) ops_.erase(pos);
    }

    /*!
     * \brief Get all the ops of this graph, inlcuding original ops and fused.
     * \return vector of ops pointers
     */
    const std::vector<op_ptr> &get_ops() const { return ops_; }

    /*! \brief how many ops in the graph */
    size_t num_ops() const { return ops_.size(); }

    /*!
     * \brief Get the output ops of this graph.
     * \return vector of output op pointers
     */
    std::vector<op_t *> get_output_ops() {
        std::vector<op_t *> outputs;
        for (const op_ptr &n : ops_) {
            size_t num_consumers = 0;
            for (size_t i = 0; i < n->num_outputs(); i++) {
                num_consumers += n->num_output_consumers(i);
            }

            if (num_consumers == 0) { outputs.push_back(n.get()); }
        }
        return outputs;
    }

    /*!
     * \brief execute graph pass
     * \param policy Partition policy
     * \return result
     */
    dnnl::graph::impl::status_t run_pass(
            dnnl::graph::impl::partition_policy_t policy);

    /*!
     * \brief Get partition numbers
     * \return partition numbers
     */
    size_t get_num_partitions() const {
        return static_cast<size_t>(std::count_if(begin(ops_), end(ops_),
                [](const op_ptr &n) { return n->has_attr("backend"); }));
    };

    /*!
     * \brief get list of partitions
     * \param list of partitions
     */
    void get_partitions(
            std::vector<dnnl::graph::impl::partition_t *> &partitions);

    /*!
     * \brief Build backend graph after add op is done
     */
    dnnl::graph::impl::status_t build_graph();

    void visualize(const std::string &filename);
};

#endif

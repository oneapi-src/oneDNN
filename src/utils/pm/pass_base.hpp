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

#ifndef UTILS_PM_PASS_BASE_HPP
#define UTILS_PM_PASS_BASE_HPP

#include <functional>
#include <list>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "interface/graph.hpp"
#include "utils/json.hpp"
#include "utils/pm/pbuilder.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace pass {

using pb_graph_t = utils::pm::pb_graph_t;
using graph = impl::graph_t;

/*! \brief pass type */
enum class pass_type { kAnalysis = 0, kTransformation = 1 };

class pass_base;
class pattern;
using pass_base_ptr = std::shared_ptr<pass_base>;

// FCreatePattern: a function for defining pattern.
// One pass can have several FCreatePattern functions.
using FCreatePattern = std::function<void(pattern *apattern)>;

// FCreateV2Pattern: a function for defining pattern.
// One pass can have several FCreateV2Pattern functions.
using FCreateV2Pattern
        = std::function<void(const std::shared_ptr<pb_graph_t> &pattern_graph)>;

// FCreateOptPattern: a function for defining optimized pattern,
// which is used in the graph rewriting part to rewrite the pattern
// to optimized pattern.
// One pass can only have one FCreateOptPattern function.
using FCreateOptPattern = std::function<void(pattern *opt_pattern)>;

// FCreateV2FusedOp: a function for defining fused op,
// which is used in the graph rewriting part to rewrite the pattern
// to a fused op.
// One pass can only have one FCreateV2FusedOp function.
using FCreateV2FusedOp = std::function<std::shared_ptr<op_t>()>;

// FRequirement: a function to check if a graph op can meet the
// requirement of a pattern op
// One pattern op can have several FRequirement function.
using FRequirement = std::function<bool(op_t *graph_op)>;

class pattern {
private:
    /*! \brief ops in this pattern */
    std::vector<std::shared_ptr<op_t>> ops_;

public:
    /*!
     * \brief Create and add a op to this pattern.
     * \param aop_kind The operator used to create the op
     * \return op* created op
     */
    op_t *create_op(op_kind_t aop_kind) {
        ops_.push_back(std::make_shared<op_t>(aop_kind));
        return ops_.back().get();
    }

    /*!
     * \brief Get the starter op of this pattern.
     *        Any op in the vector of ops_ can be the starter
     *        Any op except "any" in the vector of ops_ can be the starter
     *        By default we use the first op in the vector
     * \return op* starter op
     */
    op_t *get_starter_op() {
        auto it = std::find_if(ops_.begin(), ops_.end(),
                [&](std::vector<std::shared_ptr<op_t>>::value_type &op)
                        -> bool {
                    return op->get_kind() != impl::op_kind::Wildcard;
                });
        return it->get();
    }
};

/*!
 * \brief pass_base provides a base class for pass creation.
 *        A pass is used to do pattern matching on a given graph,
 *        and reconstruct a new graph based on optimized patterns.
 */
class pass_base {
public:
    pass_base(pass_type ptype, std::string pbackend, std::string pname)
        : type_(ptype)
        , backend_(std::move(pbackend))
        , name_(std::move(pname)) {}

    pass_base() = default;

    // the criteria of pass execution
    virtual void run(graph &agraph) { UNUSED(agraph); }
    // save pass basic information into json
    virtual void save(utils::json::json_writer_t *writer) {
        writer->begin_object();
        writer->write_keyvalue("pass_name", name_);
        if (type_ == pass_type::kTransformation) {
            writer->write_keyvalue("pass_type", std::string("Transformation"));
        } else {
            writer->write_keyvalue("pass_type", std::string("Analysis"));
        }
        writer->write_keyvalue("pass_backend", backend_);
        writer->write_keyvalue("priority", priority_);
        writer->write_keyvalue("enable", enable_);
        writer->end_object();
    }
    // load pass basic information from json
    virtual void load(utils::json::json_reader_t *reader) {
        utils::json::read_helper_t helper;
        std::string type;
        helper.declare_field("pass_name", &name_);
        helper.declare_field("pass_type", &type);
        helper.declare_field("pass_backend", &backend_);
        helper.declare_field("priority", &priority_);
        helper.declare_field("enable", &enable_);
        helper.read_fields(reader);
    }

    virtual ~pass_base() = default;

    pass_type get_pass_type() { return type_; }

    std::string get_pass_backend() { return backend_; }

    std::string get_pass_name() { return name_; }

    // set pass priority, passes with high priority
    // will be executed before passes with low priority
    pass_base &set_priority(float priority) {
        priority_ = priority;
        return *this;
    }
    float get_priority() const { return priority_; }

    // set enable status
    // can be used for override default value of enable_
    pass_base &set_enable(bool enable) {
        enable_ = enable;
        return *this;
    }

    bool get_enable() const { return enable_; }
    /*!
    * \brief Register additional attributes.
    * \param attr_name The name of the attribute.
    * \param value The value to be set.
    * \tparam value_type The type of the value to be set.
    */
    template <typename value_type>
    pass_base &set_attr(const std::string &attr_name, // NOLINT(*)
            const value_type &value) {
        attrs_.insert(make_pair(attr_name, value));
        return *this;
    }

    /*!
    * \brief Get additional registered attribute.
    * \param attr_name The name of the attribute.
    * \return An attribute of specified attr_name.
    * \tparam value_type The type of the attribute.
    */
    template <typename value_type>
    std::vector<value_type> get_attr(const std::string &attr_name) {
        std::vector<value_type> attr_vec;
        for (auto it = attrs_.begin(); it != attrs_.end(); ++it) {
            if (it->first == attr_name) {
                attr_vec.push_back(utils::any_cast<value_type>(it->second));
            }
        }
        return attr_vec;
    }

    /*!
    * \brief check if pass has a specific attribute.
    * \param attr_name The name of the attribute.
    * \return bool: if this attribute exist in pass.
    */
    bool has_attr(const std::string &attr_name) {
        return attrs_.find(attr_name) != attrs_.end();
    }

protected:
    std::unordered_multimap<std::string, impl::utils::any_t> attrs_;

private:
    pass_type type_ {};
    std::string backend_ {};
    std::string name_ {};
    float priority_ {5.0f};
    bool enable_ {true};
};

} // namespace pass
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

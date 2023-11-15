/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_COMPILER_BACKEND_HPP
#define BACKEND_GRAPH_COMPILER_COMPILER_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "graph/backend/graph_compiler/utils.hpp"
#include "graph/interface/backend.hpp"
#include "graph/utils/pm/pass_manager.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {

class compiler_backend_t : public backend_t {
    friend class compiler_partition_impl_t;

public:
    static compiler_backend_t &get_singleton() {
        static compiler_backend_t ins("compiler_backend", /*priority*/ 2.f);
        return ins;
    }

    bool support_engine_kind(engine_kind_t kind) const override {
        return kind == engine_kind_t::dnnl_cpu;
    }

    /*! \brief Register defined patterns that can be processed with compiler backend
     */
    graph::pass::pass_registry_t &get_pass_registry() { return pass_registry_; }

    /*! \brief Get the size of logical tensor in the unit of bytes
     */
    size_t get_mem_size(const logical_tensor_t &lt) const override;

    /*! \brief Get the partition that can be processed by compiler backend
     */
    status_t get_partitions(
            graph_t &agraph, partition_policy_t policy) override;

    // /*! \brief Return the support status for a specific engine kine
    //  */
    // bool support_engine_kind(engine_kind_t kind) const override {
    //     static const std::unordered_set<engine_kind_t, utils::enum_hash_t>
    //             supported_kind = {engine_kind::cpu};
    //     return supported_kind.count(kind);
    // }

private:
    compiler_backend_t(const std::string &backend_name, float priority)
        : backend_t(backend_name, priority) {};

    static graph::pass::pass_registry_t register_passes();

    static graph::pass::pass_registry_t pass_registry_;
};

} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

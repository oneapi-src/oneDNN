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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_OP_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_GRAPH_OP_HPP

#include <memory>
#include <utility>
#include <vector>
#include "graph.hpp"
#include "util/general_object.hpp"
#include <compiler/ir/graph/graph_config.hpp>
#include <compiler/ir/graph/trait/configurable.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

namespace sc {
class fusion_manager;

class graph_op_t : public sc_op {
public:
    ir_module_ptr get_func(context_ptr ctx) override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats)
            override {};
    virtual std::shared_ptr<sc_graph_t> get_graph_impl() = 0;

    virtual std::shared_ptr<sc_graph_t> get_graph();

    static std::vector<graph_tensor_ptr> remake_logical_tensors(
            const std::vector<graph_tensor_ptr> &flts);
};

class configurable_graph_op_t : public graph_op_t,
                                public op_traits::configurable_t {
public:
    std::shared_ptr<sc_graph_t> get_graph() override;

    config_ptr get_config() override;

    void set_config(const config_ptr &config) override;

    reflection::shared_general_object_t get_default_config(
            context_ptr ctx) override;

protected:
    sc::graph_config config_data_;
};

} // namespace sc

#endif

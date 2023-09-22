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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DRIVER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DRIVER_HPP

#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "analysis/analysis.hpp"
#include "transform/transform.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

enum class pass_type { analysis, pre_tune, post_tune };

using pass_func = void (*)(sc_graph_t &, const context_ptr &);
struct basic_graph_pass_t {
public:
    std::string name_;
    std::vector<std::string> requires_;
    pass_func func_;
    pass_type type_;
    sc_opt_level opt_level_; // allowed minimum opt level.
    bool enabled_; // for debug and tuning
    basic_graph_pass_t(pass_func func, const std::string &name,
            const std::vector<std::string> &required, pass_type type,
            sc_opt_level opt_level = sc_opt_level::lv3, bool enabled = true)
        : name_(name)
        , requires_(required)
        , func_(func)
        , type_(type)
        , opt_level_(opt_level)
        , enabled_(enabled) {}
};

using basic_graph_pass_ptr = std::shared_ptr<basic_graph_pass_t>;

basic_graph_pass_ptr create_graph_pass(const std::string &name, pass_func func,
        const std::vector<std::string> &required, pass_type type,
        sc_opt_level opt_level = sc_opt_level::lv3, bool enabled = true);

// Return: std::vector<std::shared_ptr>, represents all passes run order.
// 1. If adding a new pass, developer need to define this pass will be put which
// pos in vector container.
// 2. If this pass has dependent passes, please set`requires_`, for example
// `create_pass("pass_0", pass_0_func, {"pass_1"}, pass_type::xxxx)`
// 3. Each pass's enabled_ field is defaulted true. In general, the pass's
// enabled_ is true. If pass has dependencies and executed by sepcial situations
// like: assuming having analaysis_xxx pass return tue, a pass will be opened
// (enabled_ sets true) and the passes it depends on also need to be opened.
SC_API std::tuple<std::vector<basic_graph_pass_ptr>,
        std::vector<basic_graph_pass_ptr>>
get_graph_passes(const context_ptr &ctx);

/**
 * @param graph orginal graph
 * @param ctx the context
 * Return: a deep copy of the original graph that runs preprocess passes for
 * dynamic_infer_shape
 * */
SC_API sc_graph_t dynamic_shape_infer_preprocess(const sc_graph_t &graph,
        const context_ptr &ctx = get_default_context());

/**
 * @param graph orginal graph
 * @param tuner_batch the number of configs which is
 * generated from a call to get_next_config_batch(). The executor may
 * generate executables from a batch of configs in parallel.
 * @param repeat the times to repeatedly run the same executable.
 * @param ctx if developer doesn't set context, use default context.
 * The param: tuner_batch and repeat may be replaced by `timed out`.
 * */
SC_API void graph_driver(sc_graph_t &graph, int tuner_batch, int repeat,
        const context_ptr &ctx = get_default_context());

struct tuner_creator;
struct graph_config;

/**
 * Runs post tune passes, tunes the graph and runs post tune passes
 * @param graph orginal graph
 * @param ctx the context
 * @param in_cfg  if tuning is off and if not null, takes this parameter as the
 * config for the graph. If is null, will use default configs
 * @param out_cfg if tuning is on and if not null, returns the tuned graph
 * config to this pointer.
 * @param tuner_batch the number of configs which is
 * generated from a call to get_next_config_batch(). The executor may
 * generate executables from a batch of configs in parallel.
 * @param repeat the times to repeatedly run the same executable.
 * @param timeout the timeout of the tuning. Will check if time runs out after
 * each tuner batch. If set to negative, there will be no time limit. If set to
 * 0, then the tuning will be turned off
 * @param tune_creator the creator for the tuner. if null, use default tuner
 * settings
 * @param pre_tune_pass the graph passes before running tuner. if null, use
 * default passes
 * @param post_tune_pass the graph passes after running tuner. if null, use
 * default passes
 * @param allow_cache allow reusing cached code for the graph
 * */
SC_API void graph_driver(sc_graph_t &graph,
        const context_ptr &ctx = get_default_context(),
        const graph_config *in_cfg = nullptr, graph_config *out_cfg = nullptr,
        int tuner_batch = 0, int repeat = 0, int64_t timeout = 0,
        tuner_creator *tune_creator = nullptr,
        std::vector<basic_graph_pass_ptr> *pre_tune_pass = nullptr,
        std::vector<basic_graph_pass_ptr> *post_tune_pass = nullptr,
        bool allow_cache = false);

// util function to create mapping of ops in the copied graph
std::unordered_map<sc_op_ptr, std::vector<sc_op_ptr>> create_op_map(
        sc_graph_t &lg, sc_graph_t &rg);
void run_graph_passes(sc_graph_t &graph, const context_ptr &ctx,
        const std::vector<basic_graph_pass_ptr> &passes,
        bool allow_cache = false);

// get graph driver result before fusion, usually used for unit test
void graph_driver_before_fusion(sc_graph_t &graph, const context_ptr &ctx);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

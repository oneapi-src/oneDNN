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

#include <algorithm>
#include <atomic>
#include <fstream>
#include <tuple>
#include <utility>
#include "driver.hpp"
#include "pass/pass.hpp"
#include <runtime/env_vars.hpp>
#include <unordered_map>
#include <util/exceptions.hpp>
#include <util/graph_repository.hpp>
#include <util/scoped_timer.hpp>

#ifdef _MSC_VER
#include <Windows.h>
#define getprocessid GetCurrentProcessId
#else
#include <unistd.h>
#define getprocessid getpid
#endif

namespace sc {

SC_MODULE(graph.driver)

basic_graph_pass_ptr create_graph_pass(const std::string &name,
        pass_func func_t, const std::vector<std::string> &requires,
        pass_type type, bool enabled) {
    return std::make_shared<basic_graph_pass_t>(
            func_t, name, requires, type, enabled);
}

static std::vector<basic_graph_pass_ptr> create_default_graph_flow() {
    std::vector<basic_graph_pass_ptr> passes;
    passes.push_back(create_graph_pass("analysis_quantized", analysis_quantized,
            {}, pass_type::analysis, true));
    passes.push_back(create_graph_pass(
            "graph_inline", graph_inline, {}, pass_type::pre_tune, true));
    passes.push_back(create_graph_pass("constant_optimization",
            constant_optimization, {}, pass_type::pre_tune, true));
    passes.push_back(create_graph_pass("quantized_info_propagation",
            quantize::quantize_info_propagation, {}, pass_type::pre_tune,
            true));
    passes.push_back(create_graph_pass("quantized_graph_reschedule",
            quantize::graph_reschedule, {}, pass_type::pre_tune, true));
    passes.push_back(create_graph_pass("quantize_inline",
            quantize::quantize_inline, {}, pass_type::pre_tune, true));
    passes.push_back(create_graph_pass("elemtwise_bcast_swap",
            elemwise_bcast_swap, {}, pass_type::pre_tune, true));
    passes.push_back(create_graph_pass("permute_propagation",
            permute_propagation, {}, pass_type::pre_tune, true));

    // ------------------ post_tune -------------------------------------------
    passes.push_back(create_graph_pass("quantize_op_compensation",
            quantize::calculate_op_compensation, {}, pass_type::post_tune,
            true));
    passes.push_back(create_graph_pass("elemwise_dimension_alignment",
            elemwise_dimension_alignment, {}, pass_type::post_tune, true));
    passes.push_back(create_graph_pass("layout_propagation", layout_propagation,
            {}, pass_type::post_tune, true));
    passes.push_back(create_graph_pass("tensor_view_transform",
            tensor_view_transform, {}, pass_type::post_tune, true));
    passes.push_back(create_graph_pass(
            "graph_simplify", graph_simplify, {}, pass_type::post_tune, true));
    passes.push_back(create_graph_pass("global_reschedule", global_reschedule,
            {}, pass_type::post_tune, true));
    passes.push_back(create_graph_pass("const_folding",
            graph_constant_input_folding, {}, pass_type::post_tune, true));
    passes.push_back(create_graph_pass(
            "fuse_ops", fuse_ops, {}, pass_type::post_tune, true));
    passes.push_back(create_graph_pass("horizontal_merge", horizontal_merge, {},
            pass_type::post_tune, true));
    passes.push_back(create_graph_pass("const_folding",
            graph_constant_input_folding, {}, pass_type::post_tune, true));
    passes.push_back(create_graph_pass("inplace_transform", inplace_transform,
            {}, pass_type::post_tune, true));
    passes.push_back(create_graph_pass("batchwise_merge", batchwise_merge, {},
            pass_type::post_tune, true));

    // get passes map
    std::unordered_map<std::string, basic_graph_pass_ptr> passes_map;
    std::transform(passes.begin(), passes.end(),
            std::inserter(passes_map, passes_map.end()),
            [](const basic_graph_pass_ptr &pass) {
                return std::make_pair(pass->name_, pass);
            });

    // get pass's dependies and reset enabled_.
    for (auto &kv : passes_map) {
        if (kv.second->enabled_) {
            for (const std::string &require : kv.second->requires_) {
                passes_map[require]->enabled_ = true;
            }
        }
    }
    return passes;
}

const std::vector<basic_graph_pass_ptr> &get_graph_passes() {
    static auto passes = create_default_graph_flow();
    return passes;
}

static void run_passes(sc_graph_t &graph, const context_ptr &ctx,
        const std::vector<basic_graph_pass_ptr> &passes) {
    bool need_time = utils::compiler_configs_t::get().print_pass_time_;
    bool need_result = utils::compiler_configs_t::get().print_pass_result_;
    for (auto &pass : passes) {
        if (pass->enabled_) {
            auto timer = utils::create_scoped_timer(
                    need_time, [&pass](utils::time_duration dur) {
                        std::string name = std::string("graph.driver.time.")
                                + pass->name_;
                        SC_MODULE_INFO2(name.c_str())
                                << "took "
                                << std::chrono::duration_cast<
                                           std::chrono::microseconds>(dur)
                                           .count()
                                << " us";
                    });
            pass->func_(graph, ctx);
            if (need_result) {
                std::string name
                        = std::string("graph.driver.debug.") + pass->name_;
                if (auto stream
                        = runtime::get_info_logging_stream(name.c_str())) {
                    *stream.stream_ << "IR after this pass:\n";
                    print_graph(graph, *stream.stream_, true, true, true, true);
                }
            }
        }
    }
}

void dump_graph_to_json(const sc_graph_t &graph) {
    static std::atomic<int> file_counter = {0};
    static std::string export_path
            = utils::getenv_string(env_names[env_key::SC_DUMP_GRAPH_JSON]);

    if (!export_path.empty()) {
        // construct a file name through file_counter
        bool file_exist = true;
        std::string filename;
        while (file_exist) {
            std::ifstream infile;
            std::stringstream ss;
            ss << export_path << '/' << ++file_counter << ".json";
            filename = ss.str();
            infile.open(filename);
            file_exist = infile.good();
        }
        // save graph
        std::ofstream outfile(filename);
        if (!outfile.good()) {
            SC_MODULE_WARN << "Could not write to " << export_path
                           << ", the directory may not exist" << std::endl;
        }
        save_graph_to_json(graph, outfile);
    }
}

void graph_driver(sc_graph_t &graph, const context_ptr &ctx,
        const graph_config *in_cfg, graph_config *out_cfg, int batch_size,
        int repeat, int64_t timeout, tuner_creator *tune_creator,
        std::vector<basic_graph_pass_ptr> *passes) {
    auto all_pass = passes ? passes : &get_graph_passes();
    dump_graph_to_json(graph);
    // run passes
    run_passes(graph, ctx, *all_pass);
}

namespace graph {
std::unique_ptr<graph::repository> &get_driver_import_repo(
        const context_ptr &ctx) {
    auto make_repo = [](const context_ptr &ctx) {
        std::string import_path
                = utils::getenv_string(env_names[env_key::SC_TUNING_IMPORT]);
        if (import_path.empty()) {
            import_path = "sctune.json";
            SC_MODULE_WARN << "The environment variable SC_TUNING_IMPORT "
                              "is not set, using the graph config file in the "
                              "current dir: sctune.json.";
        }
        std::ifstream ifs(import_path);
        if (!ifs) {
            SC_MODULE_WARN << "Cannot open graph config file: " << import_path;
            return std::unique_ptr<graph::repository> {};
        } else {
            try {
                return utils::make_unique<graph::repository>(
                        graph::repository::load(ctx, ifs));
            } catch (json_error &je) {
                SC_MODULE_WARN << "Ignored graph config file: " << import_path
                               << ", error = " << je.what();
                return std::unique_ptr<graph::repository> {};
            }
        }
    };
    static std::unique_ptr<graph::repository> repo = make_repo(ctx);
    return repo;
}
} // namespace graph

void graph_driver(
        sc_graph_t &graph, int batch_size, int repeat, const context_ptr &ctx) {
    graph_config *pincfg = nullptr;
    graph_config *poutcfg = nullptr;
    graph_config incfg;
    auto repo = graph::get_driver_import_repo(ctx).get();
    if (repo) {
        auto entry = repo->find(graph);
        if (!entry) {
            SC_MODULE_WARN << "Cannot open find the graph in the config file";
        } else {
            incfg = entry->config_;
            pincfg = &incfg;
        }
    }
    tuner_creator *ptun_creator = nullptr;
    int64_t real_timeout = 0;
    sc_graph_t orig_graph;
    if (poutcfg) { orig_graph = copy_graph(graph); }
    graph_driver(graph, ctx, pincfg, poutcfg, batch_size, repeat, real_timeout,
            ptun_creator);
}

} // namespace sc

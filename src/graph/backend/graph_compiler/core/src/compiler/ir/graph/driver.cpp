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
#include <util/scoped_timer.hpp>

#ifdef _MSC_VER
#include <Windows.h>
#define getprocessid GetCurrentProcessId
#else
#include <unistd.h>
#define getprocessid getpid
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(graph.driver)

basic_graph_pass_ptr create_graph_pass(const std::string &name,
        pass_func func_t, const std::vector<std::string> &required,
        pass_type type, sc_opt_level opt_level, bool enabled) {
    return std::make_shared<basic_graph_pass_t>(
            func_t, name, required, type, opt_level, enabled);
}

static std::vector<basic_graph_pass_ptr> filter_passes_by_opt_level(
        const std::vector<basic_graph_pass_ptr> &passes,
        sc_opt_level opt_level) {
    std::vector<basic_graph_pass_ptr> ret;
    for (auto &p : passes) {
        if (p->opt_level_ <= opt_level) { ret.emplace_back(p); }
    }
    return ret;
}

static std::tuple<std::vector<basic_graph_pass_ptr>,
        std::vector<basic_graph_pass_ptr>>
create_default_graph_flow(const context_ptr &ctx) {
    std::vector<basic_graph_pass_ptr> pre_tune_passes, post_tune_passes;
    pre_tune_passes.push_back(create_graph_pass("eliminate_zero_shaped_tensors",
            eliminate_zero_shaped_tensors, {}, pass_type::pre_tune,
            sc_opt_level::lv0, true));
    pre_tune_passes.push_back(
            create_graph_pass("analysis_quantized", analysis_quantized, {},
                    pass_type::analysis, sc_opt_level::lv0, true));
    pre_tune_passes.push_back(create_graph_pass("annotate_fusion_break",
            quantize::annotate_fusion_break, {}, pass_type::pre_tune,
            sc_opt_level::lv2, true));
    pre_tune_passes.push_back(create_graph_pass("annotate_config",
            annotate_config, {}, pass_type::pre_tune, sc_opt_level::lv2, true));
    pre_tune_passes.push_back(create_graph_pass("graph_inline", graph_inline,
            {}, pass_type::pre_tune, sc_opt_level::lv0, true));
    pre_tune_passes.push_back(
            create_graph_pass("constant_optimization", constant_optimization,
                    {}, pass_type::pre_tune, sc_opt_level::lv0, true));
    pre_tune_passes.push_back(create_graph_pass("quantized_info_propagation",
            quantize::quantize_info_propagation, {}, pass_type::pre_tune,
            sc_opt_level::lv0, true));
    pre_tune_passes.push_back(create_graph_pass("quantized_graph_reschedule",
            quantize::graph_reschedule, {}, pass_type::pre_tune,
            sc_opt_level::lv0, true));
    pre_tune_passes.push_back(create_graph_pass("fpmath_mode", fpmath_mode, {},
            pass_type::pre_tune, sc_opt_level::lv0, true));
    if (ctx->flags_.mixed_fusion_) {
        // should be executed after graph reschedule, and before quantize_inline
        pre_tune_passes.push_back(create_graph_pass("rl_conv_weight_transform",
                rl_conv_weight_transform, {}, pass_type::pre_tune,
                sc_opt_level::lv2, true));
        pre_tune_passes.push_back(
                create_graph_pass("const_folding", graph_constant_input_folding,
                        {}, pass_type::pre_tune, sc_opt_level::lv2, true));
        pre_tune_passes.push_back(
                create_graph_pass("flatten_conv", flatten_conv, {},
                        pass_type::pre_tune, sc_opt_level::lv2, true));
    }
    pre_tune_passes.push_back(create_graph_pass("dynamic_graph_transform",
            dynamic_graph_transform, {}, pass_type::pre_tune, sc_opt_level::lv2,
            true));
    pre_tune_passes.push_back(
            create_graph_pass("quantize_inline", quantize::quantize_inline, {},
                    pass_type::pre_tune, sc_opt_level::lv0, true));
    pre_tune_passes.push_back(
            create_graph_pass("elemtwise_bcast_swap", elemwise_bcast_swap, {},
                    pass_type::pre_tune, sc_opt_level::lv1, true));
    pre_tune_passes.push_back(
            create_graph_pass("permute_propagation", permute_propagation, {},
                    pass_type::pre_tune, sc_opt_level::lv0, true));
    pre_tune_passes.push_back(create_graph_pass("quantize_op_compensation",
            quantize::calculate_op_compensation, {}, pass_type::pre_tune,
            sc_opt_level::lv0, true));
    pre_tune_passes.push_back(
            create_graph_pass("broadcast_transform", broadcast_transform, {},
                    pass_type::pre_tune, sc_opt_level::lv0, true));
    pre_tune_passes.push_back(create_graph_pass("elemwise_dimension_alignment",
            elemwise_dimension_alignment, {}, pass_type::pre_tune,
            sc_opt_level::lv0, true));
    pre_tune_passes.push_back(create_graph_pass("shape_relationship_binding",
            shape_relationship_binding, {}, pass_type::pre_tune,
            sc_opt_level::lv0, true));

    // ------------------ post_tune -------------------------------------------
    post_tune_passes.push_back(
            create_graph_pass("const_folding", graph_constant_input_folding, {},
                    pass_type::post_tune, sc_opt_level::lv0, true));
    post_tune_passes.push_back(
            create_graph_pass("div_bcast_transform", div_bcast_transform, {},
                    pass_type::post_tune, sc_opt_level::lv2, true));
    if (ctx->flags_.mixed_fusion_) {
        post_tune_passes.push_back(create_graph_pass("pre_padding", pre_padding,
                {}, pass_type::post_tune, sc_opt_level::lv2, true));
    }
    post_tune_passes.push_back(
            create_graph_pass("layout_propagation", layout_propagation, {},
                    pass_type::post_tune, sc_opt_level::lv0, true));
    post_tune_passes.push_back(
            create_graph_pass("tensor_view_transform", tensor_view_transform,
                    {}, pass_type::post_tune, sc_opt_level::lv0, true));
    post_tune_passes.push_back(
            create_graph_pass("const_folding", graph_constant_input_folding, {},
                    pass_type::post_tune, sc_opt_level::lv0, true));
    post_tune_passes.push_back(create_graph_pass("graph_simplify",
            graph_simplify, {}, pass_type::post_tune, sc_opt_level::lv2, true));
    post_tune_passes.push_back(
            create_graph_pass("global_reschedule", global_reschedule, {},
                    pass_type::post_tune, sc_opt_level::lv1, true));
    post_tune_passes.push_back(
            create_graph_pass("intrusive_opt_level", intrusive_opt_level, {},
                    pass_type::post_tune, sc_opt_level::lv0, true));
    post_tune_passes.push_back(
            create_graph_pass("partial_reduce_replace", partial_reduce_replace,
                    {}, pass_type::post_tune, sc_opt_level::lv2, true));
    // fix-me(brgemm-fuse): recover the following when postop is fixed
#if 0
    post_tune_passes.push_back(create_graph_pass("brgemm_fusion_transform",
            brgemm_fusion_transform, {}, pass_type::post_tune, true));
#endif
    post_tune_passes.push_back(
            create_graph_pass("const_folding", graph_constant_input_folding, {},
                    pass_type::post_tune, sc_opt_level::lv0, true));
    if (ctx->flags_.concat_optimization_) {
        post_tune_passes.push_back(
                create_graph_pass("merge_concats", merge_concats, {},
                        pass_type::post_tune, sc_opt_level::lv2, true));
    }
    if (!ctx->flags_.mixed_fusion_) {
        post_tune_passes.push_back(create_graph_pass("fuse_ops", fuse_ops, {},
                pass_type::post_tune, sc_opt_level::lv1, true));
        post_tune_passes.push_back(
                create_graph_pass("horizontal_merge", horizontal_merge, {},
                        pass_type::post_tune, sc_opt_level::lv1, true));
    }
    post_tune_passes.push_back(create_graph_pass("const_folding_and_share",
            graph_constant_input_folding_and_share_constants, {},
            pass_type::post_tune, sc_opt_level::lv1, true));
    post_tune_passes.push_back(
            create_graph_pass("graph_code_cache", graph_code_cache, {},
                    pass_type::post_tune, sc_opt_level::lv1, true));
    post_tune_passes.push_back(
            create_graph_pass("inplace_transform", inplace_transform, {},
                    pass_type::post_tune, sc_opt_level::lv0, true));
    post_tune_passes.push_back(
            create_graph_pass("padded_mask_mark", padded_mask_mark, {},
                    pass_type::post_tune, sc_opt_level::lv0, true));
    if (!ctx->flags_.mixed_fusion_) {
        post_tune_passes.push_back(
                create_graph_pass("batchwise_merge", batchwise_merge, {},
                        pass_type::post_tune, sc_opt_level::lv1, true));
    } else {
        post_tune_passes.push_back(
                create_graph_pass("mixed_partition", mixed_partition, {},
                        pass_type::post_tune, sc_opt_level::lv1, true));
    }
    if (ctx->flags_.concat_optimization_) {
        post_tune_passes.push_back(create_graph_pass("graph_concat_optimize",
                graph_concat_memory_planning, {}, pass_type::post_tune,
                sc_opt_level::lv2, true));
    }
    // filter passes by opt level
    pre_tune_passes = filter_passes_by_opt_level(
            pre_tune_passes, ctx->flags_.opt_level_);
    post_tune_passes = filter_passes_by_opt_level(
            post_tune_passes, ctx->flags_.opt_level_);

    // get passes map
    std::unordered_map<std::string, basic_graph_pass_ptr> passes_map;
    std::transform(pre_tune_passes.begin(), pre_tune_passes.end(),
            std::inserter(passes_map, passes_map.end()),
            [](const basic_graph_pass_ptr &pass) {
                return std::make_pair(pass->name_, pass);
            });

    std::transform(post_tune_passes.begin(), post_tune_passes.end(),
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
    return std::make_tuple(pre_tune_passes, post_tune_passes);
}

std::tuple<std::vector<basic_graph_pass_ptr>, std::vector<basic_graph_pass_ptr>>
get_graph_passes(const context_ptr &ctx) {
    return create_default_graph_flow(ctx);
}

void run_graph_passes(sc_graph_t &graph, const context_ptr &ctx,
        const std::vector<basic_graph_pass_ptr> &passes, bool allow_cache) {
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
            if (allow_cache && pass->type_ == pass_type::post_tune
                    && pass->name_ == "graph_code_cache") {
                if (graph.attrs_.has_key("graph_code_cache")) { break; }
            }
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

std::unordered_map<sc_op_ptr, std::vector<sc_op_ptr>> create_op_map(
        sc_graph_t &lg, sc_graph_t &rg) {
    assert(lg.ops_.size() == rg.ops_.size());
    auto op_size = lg.ops_.size();
    std::unordered_map<sc_op_ptr, std::vector<sc_op_ptr>> op_map;
    for (auto i = 0UL; i < op_size; i++) {
        op_map[lg.ops_[i]] = std::vector<sc_op_ptr>({rg.ops_[i]});
        op_map[rg.ops_[i]] = std::vector<sc_op_ptr>({lg.ops_[i]});
    }
    return op_map;
}

sc_graph_t dynamic_shape_infer_preprocess(
        const sc_graph_t &graph, const context_ptr &ctx) {
    sc_graph_t copy = copy_graph(graph);
    graph_inline(copy, ctx);
    constant_optimization(copy, ctx);
    return copy;
}

void graph_driver(sc_graph_t &graph, const context_ptr &ctx,
        const graph_config *in_cfg, graph_config *out_cfg, int batch_size,
        int repeat, int64_t timeout, tuner_creator *tune_creator,
        std::vector<basic_graph_pass_ptr> *pre_tune_pass,
        std::vector<basic_graph_pass_ptr> *post_tune_pass, bool allow_cache) {
    bool need_tuning = timeout != 0;
    sc_graph_t graph_cpy;

    // save origin graph(tuning) / load config(no tune)
    if (need_tuning) {
        graph_cpy = copy_graph(graph);
        graph.attrs_["temp.op_map"] = create_op_map(graph, graph_cpy);
    } else {
        SC_MODULE_INFO << "Use default config";
    }

    auto passes_tuple = get_graph_passes(ctx);
    const std::vector<basic_graph_pass_ptr> *prepass
            = pre_tune_pass ? pre_tune_pass : &std::get<0>(passes_tuple);
    const std::vector<basic_graph_pass_ptr> *postpass
            = post_tune_pass ? post_tune_pass : &std::get<1>(passes_tuple);
    // run pre_processing passes
    run_graph_passes(graph, ctx, *prepass, true);

    // run post tune passes
    run_graph_passes(graph, ctx, *postpass, true);
}

void graph_driver(
        sc_graph_t &graph, int batch_size, int repeat, const context_ptr &ctx) {
    graph_config *pincfg = nullptr;
    graph_config *poutcfg = nullptr;
    tuner_creator *ptun_creator = nullptr;
    int64_t real_timeout = 0;
    sc_graph_t orig_graph;
    if (poutcfg) { orig_graph = copy_graph(graph); }
    graph_driver(graph, ctx, pincfg, poutcfg, batch_size, repeat, real_timeout,
            ptun_creator);
}

void graph_driver_before_fusion(sc_graph_t &graph, const context_ptr &ctx) {
    analysis_quantized(graph, ctx);
    graph_inline(graph, ctx);
    constant_optimization(graph, ctx);
    quantize::quantize_info_propagation(graph, ctx);

    quantize::graph_reschedule(graph, ctx);
    quantize::quantize_inline(graph, ctx);

    elemwise_bcast_swap(graph, ctx);
    shape_relationship_binding(graph, ctx);
    permute_propagation(graph, ctx);

    quantize::calculate_op_compensation(graph, ctx);
    elemwise_dimension_alignment(graph, ctx);
    layout_propagation(graph, ctx);

    tensor_view_transform(graph, ctx);
    graph_simplify(graph, ctx);
    global_reschedule(graph, ctx);
    partial_reduce_replace(graph, ctx);
    graph_constant_input_folding(graph, ctx);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

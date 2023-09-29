/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "dnnl_common.hpp"

#include "utils/cold_cache.hpp"
#include "utils/fill.hpp"

cold_cache_mode_t default_cold_cache_mode {cold_cache_mode_t::none};
cold_cache_mode_t cold_cache_mode {default_cold_cache_mode};

namespace cold_cache_utils {
// Returns `arg` index in `dnnl_args` since they packed in random order.
int get_arg_idx(const std::vector<dnnl_exec_arg_t> &dnnl_args, int arg) {
    for (int i = 0; i < static_cast<int>(dnnl_args.size()); ++i)
        if (dnnl_args[i].arg == arg) return i;
    return -1;
}

size_t get_arg_size(const std::vector<dnnl_exec_arg_t> &dnnl_args, int arg) {
    const int arg_idx = get_arg_idx(dnnl_args, arg);
    if (arg_idx < 0) return 0; // `arg` was not found, return empty size.

    const auto &mem = dnnl_args[arg_idx].memory;
    return dnnl_memory_desc_get_size(query_md(mem));
}
} // namespace cold_cache_utils

cold_cache_t::cold_cache_t(const std::vector<dnnl_exec_arg_t> &dnnl_args)
    : enabled_(use_cold_cache(dnnl_args))
    , n_buffers_top_limit_(is_gpu() ? gpu_n_buffers_top_limit_ : SIZE_MAX)
    , n_buffers_bottom_limit_(1)
    , n_buffers_(0)
    , override_n_buffers_(false) {

    if (!enabled_) return;

    static size_t cpu_cache_capacity = 0;
    SAFE_V(get_cpu_cache_size(cpu_cache_capacity));
    // `3` potentially to cover both one and two socket scenarios.
    static const size_t cpu_cache_size_upper_bound = cpu_cache_capacity * 3;

    static size_t gpu_cache_capacity = 0;
    SAFE_V(get_gpu_cache_size(gpu_cache_capacity));
    static const size_t gpu_cache_size_upper_bound = gpu_cache_capacity * 2;

    const auto cache_capacity
            = is_gpu() ? gpu_cache_capacity : cpu_cache_capacity;
    const auto cache_size_upper_bound = is_gpu() ? gpu_cache_size_upper_bound
                                                 : cpu_cache_size_upper_bound;

    size_t full_args_size = 0;
    for (auto &e : dnnl_args) {
        if (!e.memory) continue;
        full_args_size += dnnl_memory_desc_get_size(query_md(e.memory));
    }
    size_t hot_args_size = full_args_size;
    size_t cold_args_size = 0;

    std::vector<int> cc_args; // future keys for cold_cache object.
    if (cold_cache_mode == cold_cache_mode_t::wei) {
        cc_args = {DNNL_ARG_WEIGHTS};
        const auto wei_size
                = cold_cache_utils::get_arg_size(dnnl_args, DNNL_ARG_WEIGHTS);
        hot_args_size -= wei_size;
        cold_args_size += wei_size;
    } else if (cold_cache_mode == cold_cache_mode_t::all) {
        cc_args.resize(dnnl_args.size());
        for (size_t i = 0; i < dnnl_args.size(); i++) {
            cc_args[i] = dnnl_args[i].arg;
        }
        hot_args_size = 0;
        cold_args_size = full_args_size;
    } else if (cold_cache_mode == cold_cache_mode_t::custom) {
        const std::vector<int> user_args = {/* DNNL_ARG_WEIGHTS, ... */};
        cc_args = user_args;
        if (cc_args.empty()) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: execution args for custom cold cache weren't "
                    "specified.");
            SAFE_V(FAIL);
        }
        for (int arg : cc_args) {
            const auto arg_size
                    = cold_cache_utils::get_arg_size(dnnl_args, arg);
            hot_args_size -= arg_size;
            cold_args_size += arg_size;
        }
    } else {
        assert(!"unknown cold cache mode!");
    }

    const auto MB = [](size_t bytes) {
        return static_cast<double>(bytes) / powf(2, 20);
    };
    BENCHDNN_PRINT(3,
            "[COLD_CACHE]%s Size: %.3g MB; Limit: %.3g MB; Hot args: "
            "%.3g MB; Cold args: %.3g MB;\n",
            (is_gpu() ? "[GPU]" : "[CPU]"), MB(cache_capacity),
            MB(cache_size_upper_bound), MB(hot_args_size), MB(cold_args_size));

    const size_t cold_mem_pool_size = hot_args_size > cache_size_upper_bound
            ? 0
            : cache_size_upper_bound - hot_args_size;
    const size_t n_mem_pool_buffers
            = div_up(cold_mem_pool_size, cold_args_size);
    n_buffers_ = MIN2(MAX2(n_mem_pool_buffers, n_buffers_bottom_limit_),
            n_buffers_top_limit_);
    override_n_buffers_ = n_mem_pool_buffers > n_buffers_top_limit_;

    BENCHDNN_PRINT(3,
            "[COLD_CACHE] n_buffer_limits: [%zu, %s]; n_mem_pool_buffers: "
            "%zu; n_buffers: %zu.\n",
            n_buffers_bottom_limit_,
            (n_buffers_top_limit_ == SIZE_MAX
                            ? "SIZE_MAX"
                            : std::to_string(n_buffers_top_limit_).c_str()),
            n_mem_pool_buffers, n_buffers_);

    for (auto arg : cc_args) {
        const int idx = cold_cache_utils::get_arg_idx(dnnl_args, arg);
        if (idx < 0) {
            BENCHDNN_PRINT(0, "%s \'%d\' %s\n", "Error: execution argument",
                    idx, "requested for cold caching was not found!");
            SAFE_V(FAIL);
        }

        // Empty memories don't get their cold cache entry.
        if (!dnnl_args[idx].memory) continue;

        auto &cc_entry = cache_[arg];
        cc_entry.resize(n_buffers_);
        auto orig_cc_mem_md = query_md(dnnl_args[idx].memory);

        for (size_t i = 0; i < n_buffers_; i++) {
            cc_entry[i] = dnn_mem_t(orig_cc_mem_md, get_test_engine());
            if (has_bench_mode_modifier(mode_modifier_t::no_host_memory))
                continue;

            fill_random_real(cc_entry[i]);
            if (cc_entry[i].is_mapped()) cc_entry[i].unmap();
        }
    }

    // Refer to `gpu_n_buffers_top_limit_` comment.
    // Exact cache size for src is needed to secure from potential non-temporal
    // dst stores.
    // Run reorder only if memory heuristic was overrided.
    if (override_n_buffers_) {
        dnnl_primitive_desc_t r_pd {};
        dnnl_primitive_t prim {};
        dnnl_primitive_attr_t attr {};

        const auto &engine = get_test_engine();
        const dnnl_dim_t nelems = static_cast<dnnl_dim_t>(
                div_up(cache_capacity, sizeof(float)));
        dnnl_dims_t dims {nelems};
        dnn_mem_t src_m(1, dims, dnnl_f32, tag::abx, engine);
        dnn_mem_t dst_m(1, dims, dnnl_f32, tag::abx, engine);

        DNN_SAFE_V(dnnl_reorder_primitive_desc_create(
                &r_pd, src_m.md_, engine, dst_m.md_, engine, attr));
        auto r_pd_w = make_benchdnn_dnnl_wrapper(r_pd);

        DNN_SAFE_V(dnnl_primitive_create(&prim, r_pd));
        auto prim_w = make_benchdnn_dnnl_wrapper(prim);

        args_t args;
        args.set(DNNL_ARG_FROM, src_m);
        args.set(DNNL_ARG_TO, dst_m);
        SAFE_V(execute_and_wait(prim, args));
    }
}

cold_cache_t::~cold_cache_t() {
    if (has_bench_mode_modifier(mode_modifier_t::no_host_memory)) return;

    // Mapping memories after execution to have them destroyed gracefully.
    for (auto &e : cache_) {
        auto &cc_entry = e.second;
        for (size_t i = 0; i < cc_entry.size(); i++) {
            if (!cc_entry[i].is_mapped()) cc_entry[i].map();
        }
    }
}

bool cold_cache_t::update_dnnl_args(std::vector<dnnl_exec_arg_t> &dnnl_args) {
    if (!enabled_) return true;
    if (should_stop()) return false;

    for (const auto &cc_entry : cache_) {
        const int arg = cc_entry.first;
        const int dnnl_args_idx = cold_cache_utils::get_arg_idx(dnnl_args, arg);
        if (dnnl_args_idx < 0) return false;

        const auto &e = cc_entry.second;
        // Assumption that cache entries of the same size.
        if (cc_counter_ >= e.size()) cc_counter_ = 0;
        dnnl_args[dnnl_args_idx].memory = e[cc_counter_].m_;
    }
    // Update counter outside of the loop to make **all** arguments use same
    // order element from the cache.
    cc_counter_++;

    return true;
}

bool cold_cache_t::should_stop() const {
    return override_n_buffers_ && cc_counter_ == n_buffers_;
}

bool cold_cache_t::use_cold_cache(
        const std::vector<dnnl_exec_arg_t> &dnnl_args) {
    const bool cc_wei = cold_cache_mode == cold_cache_mode_t::wei;
    const bool cc_all = cold_cache_mode == cold_cache_mode_t::all;
    const bool cc_custom = cold_cache_mode == cold_cache_mode_t::custom;
    const bool has_weights
            = cold_cache_utils::get_arg_idx(dnnl_args, DNNL_ARG_WEIGHTS) >= 0;
    static int warning_printed = 0;
    if (cc_wei && !has_weights && !warning_printed) {
        BENCHDNN_PRINT(0, "%s\n",
                "Warning: cold cache for weights was requested but weights "
                "were not identified in execution arguments. Cold cache will "
                "not be enabled.");
        warning_printed = 1;
    }

    return (cc_wei && has_weights) || cc_all || cc_custom;
}

std::ostream &operator<<(std::ostream &s, cold_cache_mode_t cold_cache_mode) {
    if (cold_cache_mode == cold_cache_mode_t::none)
        s << "";
    else if (cold_cache_mode == cold_cache_mode_t::wei)
        s << "wei";
    else if (cold_cache_mode == cold_cache_mode_t::all)
        s << "all";
    else if (cold_cache_mode == cold_cache_mode_t::custom)
        s << "custom";
    else {
        assert(!"unsupported cold cache mode");
    }
    return s;
}

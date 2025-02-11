/*******************************************************************************
* Copyright 2017-2025 Intel Corporation
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

#include <algorithm> // for std::reverse and std::copy
#include <functional> // for std::bind and std::placeholders
#include <list>
#include <string> // for std::string
#include <utility> // for std::pair
#include <vector> // for std::vector

#include <assert.h>

#include "oneapi/dnnl/dnnl.hpp"
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.h"
#endif

#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
#include "src/common/primitive_cache.hpp"
#endif

#include "cpu/platform.hpp"

#include "tests/test_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "utils/cold_cache.hpp"
#include "utils/dnnl_query.hpp"
#include "utils/fill.hpp"
#include "utils/stream_kind.hpp"

extern "C" dnnl_status_t dnnl_impl_notify_profiling_complete(
        dnnl_stream_t stream);

int check_pd_cache(const_dnnl_primitive_desc_t pd, res_t *res) {
    // Disable this check for threadpool. A threadpool is always defined in
    // validation infrastructure but creates primitives in a separate
    // environment that doesn't have threadpool enabled. Thus, it utilizes a
    // specified number of threads in a testing environment that is different
    // from the number of cores per socket (internal logic for primitive
    // creation). This will cause this check to fail as the number of threads
    // is used in the primitive cache key.
#if !defined(DNNL_DISABLE_PRIMITIVE_CACHE) \
        && DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL
    int capacity = 0;
    DNN_SAFE(dnnl_get_primitive_cache_capacity(&capacity), FAIL);
    if (capacity && !dnnl::impl::is_pd_in_cache(pd)) {
        res->state = FAILED;
        BENCHDNN_PRINT(0, "%s\n",
                "Error: primitive descriptor is expected to be fetched from "
                "the primitive cache");
        return FAIL;
    }
#endif
    return OK;
}

int check_primitive_cache(dnnl_primitive_t p, res_t *res) {
    // See the comment in `check_pd_cache`.
#if !defined(DNNL_DISABLE_PRIMITIVE_CACHE) \
        && DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL
    int capacity = 0;
    DNN_SAFE(dnnl_get_primitive_cache_capacity(&capacity), WARN);
    if (capacity && !dnnl::impl::is_primitive_in_cache(p)) {
        res->state = FAILED;
        BENCHDNN_PRINT(0, "%s\n",
                "Error: primitive is expected to be fetched from the primitive "
                "cache");
        return FAIL;
    }
#endif
    return OK;
}

size_t set_primitive_cache_capacity_without_clearing(size_t capacity) {
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    return dnnl::impl::set_primitive_cache_capacity_without_clearing(capacity);
#endif
    return size_t(0);
}

int get_cache_blob_id(
        std::vector<uint8_t> &cache_blob_id, const_dnnl_primitive_desc_t pd) {
    dnnl_dim_t count;
    const uint8_t *c_id;
    DNN_SAFE(dnnl_primitive_desc_query(
                     pd, dnnl_query_cache_blob_id_size_s64, 0, (void *)&count),
            WARN);
    DNN_SAFE(dnnl_primitive_desc_query(
                     pd, dnnl_query_cache_blob_id, 0, (void **)&c_id),
            WARN);
    cache_blob_id = {c_id, c_id + count};
    return OK;
}

int get_cache_blob(std::vector<uint8_t> &cache_blob, dnnl_primitive_t prim) {
    size_t size = 0;
    DNN_SAFE(dnnl_primitive_get_cache_blob(prim, &size, nullptr), WARN);

    cache_blob.resize(size);
    DNN_SAFE(dnnl_primitive_get_cache_blob(prim, &size, cache_blob.data()),
            WARN);
    return OK;
}

struct lru_cache_t {
    lru_cache_t(size_t capacity) : capacity_(capacity) {}

    const std::vector<uint8_t> &get(const std::vector<uint8_t> &key) {
        auto it = cache_mapper_.find(key);
        if (it == cache_mapper_.end()) { return dummy_; }
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        return cache_list_.front().value_;
    }

    void add(const std::vector<uint8_t> &key,
            const std::vector<uint8_t> &value) {
        assert(!cache_mapper_.count(key));
        if (cache_mapper_.size() >= capacity_) {
            cache_mapper_.erase(cache_list_.back().key_);
            cache_list_.pop_back();
        }
        cache_list_.emplace_front(key, value);
        cache_mapper_.insert({key, cache_list_.begin()});
    }

private:
    lru_cache_t(const lru_cache_t &other) = delete;
    lru_cache_t(lru_cache_t &&other) = delete;
    lru_cache_t &operator=(const lru_cache_t &other) = delete;
    lru_cache_t &operator=(lru_cache_t &&other) = delete;

    struct entry_t {
        entry_t(const std::vector<uint8_t> &key,
                const std::vector<uint8_t> &value)
            : key_(key), value_(value) {}
        std::vector<uint8_t> key_;
        std::vector<uint8_t> value_;
    };

    size_t capacity_;
    std::list<entry_t> cache_list_;
    std::map<std::vector<uint8_t>, std::list<entry_t>::iterator> cache_mapper_;

    const std::vector<uint8_t> dummy_;
};

lru_cache_t &get_test_cache() {
    static lru_cache_t cache(1024);
    return cache;
}

int test_persistent_cache_api(
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim, res_t *res) {

    auto pd = query_pd(prim);

    // 0. check memory descriptor serialization API
    // Limiting to weights md should have good md kind coverage and
    // limit overhead
    {
        dnnl_memory_desc_t new_md;
        size_t sz = 0;
        auto wei_md = query_md(pd, dnnl_query_weights_md);
        dnnl_memory_desc_get_blob(nullptr, &sz, wei_md);

        std::vector<uint8_t> md_blob(sz);
        dnnl_memory_desc_get_blob(md_blob.data(), &sz, wei_md);
        dnnl_memory_desc_create_with_blob(&new_md, md_blob.data());
        auto mew_mdw = make_benchdnn_dnnl_wrapper(new_md);

        if (dnnl_memory_desc_equal(wei_md, new_md) == 0)
            return res->state = FAILED, FAIL;
    }

    // Start testing persistent cache API.
    if (!is_gpu() || (is_gpu() && DNNL_GPU_RUNTIME != DNNL_RUNTIME_OCL)) {
        return OK;
    }

    // 1. Disable primitive cache to make sure that the next primitive will
    // be created from the cache blob and not fetched from the primitive cache.
    const auto old_capacity = set_primitive_cache_capacity_without_clearing(0);
    // 2. Get cache blob ID to use it as a key for the `test_cache`.
    std::vector<uint8_t> cache_blob_id;
    auto st = get_cache_blob_id(cache_blob_id, pd);
    if (st != OK) return res->state = FAILED, FAIL;
    // 3. Check if a cache blob for the obtained cache blob ID is present in the
    //    `test_cache`.
    //    a) If the cache blob is found the primitive is created from it.
    //    b) If the cache blob is not found then get it from the previously
    //       created primitive, store it in the cache and create the primitive
    //       from it.
    dnnl_primitive_t p {};
    auto &cache = get_test_cache();
    auto cache_value = cache.get(cache_blob_id);
    if (!cache_value.empty()) {
        const size_t size = cache_value.size();
        const uint8_t *cache_blob = cache_value.data();
        auto dnnl_st = dnnl_primitive_create_from_cache_blob(
                &p, pd, size, cache_blob);
        if (dnnl_st != dnnl_success) return res->state = FAILED, FAIL;
    } else {
        std::vector<uint8_t> cache_blob;
        st = get_cache_blob(cache_blob, prim);
        if (st != OK) return res->state = FAILED, FAIL;

        // The cross-engine and direct copy reorders are special primitives that
        // may contain no kernels therefore the cache blob will always be empty,
        // which is the correct behavior.
        if (cache_blob.empty()) {
            set_primitive_cache_capacity_without_clearing(old_capacity);
            if (query_prim_kind(pd) == dnnl_reorder
                    && (res->impl_name.find("cross_engine") != std::string::npos
                            || res->impl_name.find("direct_copy")
                                    != std::string::npos))
                return OK;

            BENCHDNN_PRINT(
                    0, "error: %s\n", "cache blob is not expected to be empty");
            res->state = FAILED;
            return FAIL;
        }

        auto dnnl_st = dnnl_primitive_create_from_cache_blob(
                &p, pd, cache_blob.size(), cache_blob.data());
        if (dnnl_st != dnnl_success) return res->state = FAILED, FAIL;
        cache.add(cache_blob_id, cache_blob);
    }
    prim.reset(p);

    // 4. Restore the original primitive cache capacity to make it functional.
    set_primitive_cache_capacity_without_clearing(old_capacity);

    return OK;
}

// Engine kind used to run oneDNN primitives for testing
dnnl_engine_kind_t engine_tgt_kind = dnnl_cpu;
// Engine index used to run oneDNN primitives for testing
size_t engine_index = 0;
// CPU ISA specific hints : none by default
isa_hints_t hints {isa_hints_t::none};

memory_kind_ext_t memory_kind {default_memory_kind};

int default_num_streams = 1;
int num_streams = default_num_streams;

void init_isa_settings() {
    if (hints.get() == isa_hints_t::no_hints) {
        DNN_SAFE_V(dnnl_set_cpu_isa_hints(dnnl_cpu_isa_no_hints));
    } else if (hints.get() == isa_hints_t::prefer_ymm) {
        auto status = dnnl_set_cpu_isa_hints(dnnl_cpu_isa_prefer_ymm);
        // Unimplemented is a valid status for non-x64
        if (status == dnnl_success || status == dnnl_unimplemented) return;
        DNN_SAFE_V(status);
    } else {
        // Do nothing when hints == none
        assert(hints.get() == isa_hints_t::none);
    }
}

// This ctor is responsible to provide proper pointers to memory objects for
// correspondent arguments. It is important for in-place cases when a single
// object should be used as SRC and DST.
// `mem_map` object is an owner of memory objects and can't use the same object
// for SRC and DST while `args` is a proxy with pointers to memories and may
// easily change what to pick for a specific arg.
args_t::args_t(const dnn_mem_map_t &mem_map) {
    for (const auto &map_entry : mem_map) {
        const dnn_mem_t *mem_ptr = &map_entry.second;
        for (int inplace_arg : {DNNL_ARG_DST, DNNL_ARG_DIFF_SRC}) {
            if (map_entry.first != inplace_arg || map_entry.second) continue;

            auto it = mem_map.begin();
            switch (inplace_arg) {
                case DNNL_ARG_DST:
                    it = mem_map.find(DNNL_ARG_SRC);
                    // May happen that source argument is different.
                    if (it == mem_map.end())
                        it = mem_map.find(DNNL_ARG_MULTIPLE_SRC);
                    break;
                case DNNL_ARG_DIFF_SRC:
                    it = mem_map.find(DNNL_ARG_DIFF_DST);
                    break;
                default: assert(!"unsupported arg"); break;
            }
            if (it == mem_map.end()) {
                BENCHDNN_PRINT(0, "%s\n", "Inplace substitution failed.");
                SAFE_V(FAIL);
            }

            mem_ptr = &((*it).second); // Update reference with in-place memory.
            break;
        }

        args_.emplace_back(map_entry.first, mem_ptr);
    }
}

args_t &args_t::set(int arg, const dnn_mem_t &mem) {
    args_.emplace_back(arg, &mem);
    return *this;
}

const dnn_mem_t &args_t::find(int arg) const {
    static dnn_mem_t empty_stub;
    for (const auto &e : args_) {
        if (e.first == arg) return *(e.second);
    }
    return empty_stub;
}

void args_t::replace(int arg, const dnn_mem_t *mem) {
    for (auto &e : args_) {
        if (e.first == arg) {
            e.second = mem;
            break;
        }
    }
}

// Unmap before passing the memory to execute
void execute_unmap_args(
        const args_t &args, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    dnnl_args.resize(args.size());
    for (int i = 0; i < args.size(); ++i) {
        if (args.dnn_mem(i).is_mapped()) args.dnn_mem(i).unmap();

        dnnl_args[i].arg = args.arg(i);
        dnnl_args[i].memory = args.dnn_mem(i).m_;
    }
}

// Map the memory back after execute
void execute_map_args(const args_t &args) {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return;

    for (int i = 0; i < args.size(); ++i)
        if (!args.dnn_mem(i).is_mapped()) args.dnn_mem(i).map();
}

int execute_and_wait(perf_function_t &exec_func, const dnnl_engine_t &engine,
        const args_t &args, res_t *res) {
    stream_t stream(engine);
    std::vector<dnnl_exec_arg_t> dnnl_args;

    execute_unmap_args(args, dnnl_args);

    dnnl_status_t status = dnnl_runtime_error;
    bool run_regular_exec = true;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    while (execution_mode == execution_mode_t::graph && is_gpu(engine)) {
        void *queue_ptr;
        DNN_SAFE(dnnl_sycl_interop_stream_get_queue(stream, &queue_ptr), CRIT);
        sycl::queue queue = *static_cast<sycl::queue *>(queue_ptr);
        const bool can_run_sycl_graph = queue.get_device().get_backend()
                == sycl::backend::ext_oneapi_level_zero;
        if (!can_run_sycl_graph) break;

        BENCHDNN_PRINT(
                2, "%s\n", "[INFO] Using experimental SYCL graph execution.");
        sycl::ext::oneapi::experimental::command_graph graph {
                queue.get_context(), queue.get_device()};

        graph.begin_recording(queue);
        status = exec_func(stream, dnnl_args);
        graph.end_recording(queue);
        DNN_SAFE(dnnl_stream_wait(stream), CRIT);

        auto exec = graph.finalize();
        queue.ext_oneapi_graph(exec).wait();

        // SYCL graph feature completed submission and execution, no need to
        // have a regular run.
        run_regular_exec = false;
        break;
    }
#endif
    if (run_regular_exec) {
        status = exec_func(stream, dnnl_args);
        DNN_SAFE(dnnl_stream_wait(stream), CRIT);
    }
    if (res) res->state = EXECUTED;

    execute_map_args(args);
    if (status != dnnl_success) {
        if (res) res->state = FAILED;
        return FAIL;
    }

    return OK;
}

dnnl_status_t primitive_executor(dnnl_primitive_t prim,
        const dnnl_stream_t &stream,
        const std::vector<dnnl_exec_arg_t> &dnnl_args) {
    return dnnl_primitive_execute(
            prim, stream, (int)dnnl_args.size(), dnnl_args.data());
}

int execute_and_wait(dnnl_primitive_t prim, const args_t &args, res_t *res) {
    perf_function_t exec_func = std::bind(&primitive_executor, prim,
            std::placeholders::_1, std::placeholders::_2);
    auto pd = query_pd(prim);
    auto engine = query_engine(pd);
    return execute_and_wait(exec_func, engine, args, res);
}

void reset_gpu_profiling(dnnl_stream_t stream) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    DNN_SAFE_V(dnnl_reset_profiling(stream));
#endif
}

int get_gpu_profiling_info(dnnl_stream_t stream, std::vector<uint64_t> &nsecs,
        std::vector<uint64_t> &cycles, int expected_num_entries) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    dnnl_profiling_data_kind_t undef_kind {};
    dnnl_profiling_data_kind_t time_kind {};

    // This is an internal data kind.
    dnnl_profiling_data_kind_t cycles_kind
            = dnnl::impl::profiling_data_kind::cycles;
#ifndef DNNL_EXPERIMENTAL_PROFILING
    undef_kind = 0;
    time_kind = 1;
#else
    undef_kind = dnnl_profiling_data_kind_undef;
    time_kind = dnnl_profiling_data_kind_time;
#endif

    int num_entries = 0;
    DNN_SAFE(dnnl_query_profiling_data(
                     stream, undef_kind, &num_entries, nullptr),
            CRIT);
    if (expected_num_entries != -1 && num_entries != expected_num_entries) {
        BENCHDNN_PRINT(0,
                "ERROR: profiling entries mismatch, expected: %d entries but "
                "got %d entries\n",
                expected_num_entries, num_entries);
        return FAIL;
    }
    DNN_SAFE(dnnl_query_profiling_data(
                     stream, time_kind, &num_entries, nsecs.data()),
            CRIT);
    nsecs.resize(num_entries);
    cycles.resize(num_entries);
    DNN_SAFE(dnnl_query_profiling_data(
                     stream, time_kind, &num_entries, nsecs.data()),
            CRIT);
    DNN_SAFE(dnnl_query_profiling_data(
                     stream, cycles_kind, &num_entries, cycles.data()),
            CRIT);
#endif
    return OK;
}

void notify_gpu_profiling_complete(dnnl_stream_t stream) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
    DNN_SAFE_V(dnnl_impl_notify_profiling_complete(stream));
#endif
}

void finalize() {
    finalize_tbb();
}

inline int measure_perf_individual(timer::timer_t &t, dnnl_stream_t stream,
        perf_function_t &perf_func, std::vector<dnnl_exec_arg_t> &dnnl_args) {
    cold_cache_t cold_cache(dnnl_args, stream);

    t.reset();
    while (true) {
        if (!cold_cache.update_dnnl_args(dnnl_args)) break;
        t.start();
        DNN_SAFE(perf_func(stream, dnnl_args), WARN);
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

inline int measure_perf_aggregate(timer::timer_t &t,
        const std::vector<stream_t> &v_stream, perf_function_t &perf_func,
        std::vector<std::vector<dnnl_exec_arg_t>> &dnnl_args) {
    // There seems to be some limit to how many kernels can be queued in OCL
    // builds and 4096 seems to be a nice number under that limit.
    // Otherwise, hangs in perf validation are observed due to many kernels
    // being queued at once.
    static constexpr int max_batch_times = 4096;

    std::vector<cold_cache_t> cold_cache(num_streams);

    // Nvidia/AMD don't support profiling.
    const bool use_profiling = is_gpu() && !is_nvidia_gpu() && !is_amd_gpu();

    for (size_t j = 0; j < v_stream.size(); j++) {
        // Warm-up run, this is not measured due to possibility the associated
        // kernel has not been built and skews the results.
        DNN_SAFE(perf_func(v_stream[j], dnnl_args[j]), WARN);
        DNN_SAFE(dnnl_stream_wait(v_stream[j]), CRIT);
        cold_cache[j] = cold_cache_t(dnnl_args[j], v_stream[j]);
        if (use_profiling) reset_gpu_profiling(v_stream[j]);
    }

    bool is_first_loop = true;
    int cur_batch_times
            = fix_times_per_prb ? fix_times_per_prb : min_times_per_prb;

    t.reset();
    while (true) {
        // Keep separate var due to a `break` inside the loop.
        int execute_count = 0;
        // Keep inner loop over streams for better submission overlapping.
        for_(int i = 0; i < cur_batch_times; i++)
        for (size_t j = 0; j < v_stream.size(); j++) {
            if (!cold_cache[j].update_dnnl_args(dnnl_args[j])) break;
            DNN_SAFE(perf_func(v_stream[j], dnnl_args[j]), WARN);
            execute_count++;
        }

        for (size_t j = 0; j < v_stream.size(); j++) {
            DNN_SAFE(dnnl_stream_wait(v_stream[j]), CRIT);
        }

        if (use_profiling) {
            std::vector<std::vector<uint64_t>> v_nsecs(num_streams);
            std::vector<std::vector<uint64_t>> v_cycles(num_streams);
            bool nsecs_is_empty = false;
            for (size_t j = 0; j < v_stream.size(); j++) {
                SAFE(get_gpu_profiling_info(v_stream[j], v_nsecs[j],
                             v_cycles[j], execute_count),
                        CRIT);
                reset_gpu_profiling(v_stream[j]);

                // Profiling should have information to report, otherwise, stop.
                if (v_nsecs[j].empty()) {
                    nsecs_is_empty = true;
                    BENCHDNN_PRINT(0, "%s\n",
                            "WARNING: no counters were found during "
                            "profiling.");
                    break;
                }
            }
            if (nsecs_is_empty) break;

            for_(size_t j = 0; j < v_stream.size(); j++)
            for (size_t i = 0; i < v_nsecs[j].size(); i++) {
                t.stop(1, (int64_t)v_cycles[j][i], v_nsecs[j][i] / 1e6);
            }
        } else {
            t.stamp(cur_batch_times * num_streams);
        }

        // Assumption that for each stream cold_cache acts same.
        if (should_stop(t) || cold_cache[0].should_stop()) break;

        // Adjust cur_batch_times after the first batch run
        if (is_first_loop) {
            double ms_min = t.ms(timer::timer_t::min);
            // Heuristic: try to use ~5 batch runs for the whole benchmark
            int batch_times_heuristic = (ms_min == 0.0)
                    ? INT_MAX
                    : MAX2(1,
                            (int)((max_ms_per_prb - t.total_ms()) / ms_min
                                    / 5));
            cur_batch_times = MIN2(max_batch_times, batch_times_heuristic);
            is_first_loop = false;
        }
    }

    if (use_profiling) {
        for (size_t j = 0; j < v_stream.size(); j++) {
            notify_gpu_profiling_complete(v_stream[j]);
        }
    }

    return OK;
}

int measure_perf(const thr_ctx_t &ctx, res_t *res, perf_function_t &perf_func,
        args_t &args) {
    if (!has_bench_mode_bit(mode_bit_t::perf)) return OK;

    const auto &engine = get_test_engine();
    std::vector<stream_t> v_stream(num_streams);
    for (int i = 0; i < num_streams; i++)
        v_stream[i] = stream_t(engine, ctx.get_interop_obj());

    std::vector<std::vector<dnnl_exec_arg_t>> dnnl_args(num_streams);
    std::vector<dnn_mem_map_t> mem_map(num_streams);
    std::vector<args_t> v_args(num_streams);
    v_args[0] = args;
    for (int j = 1; j < num_streams; j++) {
        for (int i = 0; i < args.size(); i++) {
            int arg = args.arg(i);
            const auto &m = args.dnn_mem(i);
            mem_map[j].emplace(arg, dnn_mem_t(m.md_, engine));
            SAFE(mem_map[j].at(arg).reorder(m), WARN);
        }
        v_args[j] = args_t(mem_map[j]);
        execute_unmap_args(v_args[j], dnnl_args[j]);
    }
    execute_unmap_args(args, dnnl_args[0]);

    auto &t = res->timer_map.perf_timer();
    // For non-DPCPP CPU: measure individual iterations.
    // For DPCPP CPU and GPU: measure iterations in batches to hide driver
    // overhead. DPCPP CPU follows the model of GPU, thus, handled similar.
    int ret = OK;
    if (is_cpu() && !is_sycl_engine(engine)) {
        ret = execute_in_thr_ctx(ctx, measure_perf_individual, t, v_stream[0],
                perf_func, dnnl_args[0]);
    } else {
        ret = execute_in_thr_ctx(
                ctx, measure_perf_aggregate, t, v_stream, perf_func, dnnl_args);
    }

    if (ret != OK) res->state = FAILED;
    execute_map_args(args);
    for (int j = 1; j < num_streams; j++) {
        execute_map_args(v_args[j]);
    }

    return ret;
}

int measure_perf(
        const thr_ctx_t &ctx, res_t *res, dnnl_primitive_t prim, args_t &args) {
    perf_function_t perf_func = std::bind(&primitive_executor, prim,
            std::placeholders::_1, std::placeholders::_2);

    return measure_perf(ctx, res, perf_func, args);
}

std::vector<float> prepare_po_vals(const dnn_mem_t &dst_m, const args_t &args,
        const std::vector<std::pair<int, int>> &v_po_masks,
        const size_t dst_off) {
    if (v_po_masks.empty()) return std::vector<float>();

    std::vector<float> v_vals(v_po_masks.size());

    for (size_t d = 0; d < v_po_masks.size(); ++d) {
        const auto po_offset = dst_m.get_idx(dst_off, v_po_masks[d].second);
        const float val = args.find(v_po_masks[d].first).get_elem(po_offset);
        v_vals[d] = val;
    }
    return v_vals;
}

bool check_md_consistency_with_tag(
        const_dnnl_memory_desc_t md, const std::string &tag) {
    auto md_new_tag = dnn_mem_t::init_md(
            query_md_ndims(md), query_md_dims(md), query_md_data_type(md), tag);
    return dnnl_memory_desc_equal(md_new_tag, md);
}

void skip_unimplemented_data_type(
        const std::vector<dnnl_data_type_t> &v_dt, dir_t dir, res_t *res) {
    const bool has_f64_support = is_f64_supported();
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    using namespace dnnl::impl::cpu::platform;
    // bf16 is supported on AVX512-CORE+
    const bool has_bf16_support = is_gpu()
            || (is_cpu() && has_data_type_support(dnnl_bf16)
                    && IMPLICATION(!(dir & FLAG_INF),
                            has_training_support(dnnl_bf16)));
    const bool has_f16_support = is_gpu()
            || (is_cpu() && has_data_type_support(dnnl_f16)
                    && IMPLICATION(
                            !(dir & FLAG_INF), has_training_support(dnnl_f16)));
    const bool has_e8m0_support
            = is_gpu() || (is_cpu() && has_data_type_support(dnnl_e8m0));
    const bool has_f4_e2m1_support
            = is_gpu() || (is_cpu() && has_data_type_support(dnnl_f4_e2m1));
    const bool has_f4_e3m0_support
            = is_gpu() || (is_cpu() && has_data_type_support(dnnl_f4_e3m0));
    const bool has_f8_e5m2_support = is_gpu()
            || (is_cpu() && has_data_type_support(dnnl_f8_e5m2)
                    && (dir & FLAG_INF));
    const bool has_f8_e4m3_support = is_gpu()
            || (is_cpu() && has_data_type_support(dnnl_f8_e4m3)
                    && (dir & FLAG_INF));
#else
    const bool has_bf16_support = is_gpu();
    // f16 is supported on GPU for inference only.
    const bool has_f16_support = is_gpu() && (dir & FLAG_FWD);
    const bool has_f4_e2m1_support = is_gpu();
    const bool has_f4_e3m0_support = false;
    const bool has_e8m0_support = is_gpu();
    const bool has_f8_e5m2_support = is_gpu();
    const bool has_f8_e4m3_support = is_gpu();
#endif

    for (const auto &i_dt : v_dt) {
        bool need_skip = false;
        switch (i_dt) {
            case dnnl_bf16: need_skip = !has_bf16_support; break;
            case dnnl_f16: need_skip = !has_f16_support; break;
            case dnnl_f64: need_skip = !has_f64_support; break;
            case dnnl_e8m0: need_skip = !has_e8m0_support; break;
            case dnnl_f4_e2m1: need_skip = !has_f4_e2m1_support; break;
            case dnnl_f4_e3m0: need_skip = !has_f4_e3m0_support; break;
            case dnnl_f8_e5m2: need_skip = !has_f8_e5m2_support; break;
            case dnnl_f8_e4m3: need_skip = !has_f8_e4m3_support; break;
            default: break;
        }
        if (need_skip) {
            res->state = SKIPPED;
            res->reason = skip_reason::data_type_not_supported;
            return;
        }
    }
}

void skip_unimplemented_sum_po(const attr_t &attr, res_t *res,
        dnnl_primitive_kind_t pkind, dnnl_data_type_t src_dt,
        dnnl_data_type_t dst_dt) {
    const auto &po = attr.post_ops;
    if (po.is_def()) return;

    const int first_sum_idx = po.find(attr_t::post_ops_t::SUM);
    if (first_sum_idx == -1) return;

    const auto sum_dt = po.entry[first_sum_idx].sum.dt;

    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        if (e.is_sum_kind()) {
            // API requirements
            if (e.sum.zero_point != 0) {
                // Sum with zero-point is only supported for int8
                if (!is_integral_dt(src_dt)) {
                    res->state = SKIPPED;
                    res->reason = skip_reason::case_not_supported;
                    return;
                } else {
                    // Only quantized sum operand can have zero point
                    const dnnl_data_type_t e_sum_dt
                            = e.sum.dt == dnnl_data_type_undef ? dst_dt
                                                               : e.sum.dt;
                    if (!is_integral_dt(e_sum_dt)) {
                        res->state = SKIPPED;
                        res->reason = skip_reason::case_not_supported;
                        return;
                    }
                }
            }

            // Sum with zero-point is not supported on GPU
            if (is_gpu() && e.sum.zero_point != 0) {
                res->state = SKIPPED;
                res->reason = skip_reason::case_not_supported;
                break;
            }
            // Each sum must have same data on CPU
            if (is_cpu() && e.sum.dt != sum_dt) {
                res->state = SKIPPED;
                res->reason = skip_reason::case_not_supported;
                break;
            }
            // Sum must have data type with the same size like dst on both
            if (dst_dt != dnnl_data_type_undef && sum_dt != dnnl_data_type_undef
                    && dnnl_data_type_size(dst_dt)
                            != dnnl_data_type_size(e.sum.dt)) {
                res->state = SKIPPED;
                res->reason = skip_reason::case_not_supported;
                return;
            }
        }
    }
}

void skip_unimplemented_prelu_po(
        const attr_t &attr, res_t *res, dnnl_primitive_kind_t pkind) {
    const auto &po = attr.post_ops;
    if (po.is_def()) return;

    const int first_prelu_idx = po.find(attr_t::post_ops_t::PRELU);
    if (first_prelu_idx == -1) return;

    switch (pkind) {
        case dnnl_convolution:
        case dnnl_deconvolution:
        case dnnl_inner_product:
        case dnnl_matmul: return; break;
        default:
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            break;
    }
}

void skip_unimplemented_arg_scale(const attr_t &attr, res_t *res) {
    for (const auto &arg_s : attr.scales.scales) {
        if (arg_s.second.policy != policy_t::COMMON) {
            res->state = SKIPPED;
            res->reason = skip_reason::case_not_supported;
            return;
        }
    }
}

void skip_invalid_inplace(res_t *res, dnnl_data_type_t sdt,
        dnnl_data_type_t ddt, const std::string &stag,
        const std::string &dtag) {
    // Note: existing implementation of dnn_mem_t doesn't allow to track the
    // fact that two different objects pointing on the same SYCL memory should
    // not map/unmap both objects.
    // This leads to the restriction that memory descriptors should coincide,
    // thus, a single memory object would be used for in-place validation.
    // General limitation of in-place mode is having same amount of memory on
    // input and output.
    if (sdt != ddt) {
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }

    if (dtag == tag::any) return;
    if (stag != dtag) {
        res->state = SKIPPED;
        res->reason = skip_reason::invalid_case;
        return;
    }
}

// Check ensures that attributes don't cause implementation fallback
int check_same_pd(const dnnl_primitive_desc_t &pd_no_attr, res_t *res) {
    const auto pd_no_attr_name = query_impl_info(pd_no_attr);
    if (res->impl_name == pd_no_attr_name) return OK;

    res->state = FAILED;
    BENCHDNN_PRINT(0,
            "ERROR: attributes caused impl fallback from [%s] to [%s]\n",
            pd_no_attr_name.c_str(), res->impl_name.c_str());
    return FAIL;
}

// Checks if unexpected reference implementation was hit.
int check_ref_impl_hit(res_t *res) {
    if (!check_ref_impl) return OK;

    // Nvidia, AMD and Generic backends use reference implementations to fill
    // gaps in feature support.
    if (is_nvidia_gpu() || is_amd_gpu() || is_generic_gpu()) return OK;

    const auto &impl_name = res->impl_name;
    if (impl_name.find("ref") != std::string::npos) {
        res->state = FAILED;
        res->reason = "Ref Impl Not Expected";
        return FAIL;
    }
    return OK;
}

bool is_cpu(const dnnl_engine_t &engine) {
    return query_engine_kind(engine) == dnnl_cpu;
}

bool is_gpu(const dnnl_engine_t &engine) {
    return query_engine_kind(engine) == dnnl_gpu;
}

bool is_sycl_engine(const dnnl_engine_t &engine) {
    if (is_cpu(engine)) return DNNL_CPU_RUNTIME == DNNL_RUNTIME_DPCPP;
    if (is_gpu(engine)) return DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP;
    return false;
}

bool is_opencl_engine(const dnnl_engine_t &engine) {
    if (is_gpu(engine)) return DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL;
    return false;
}

bool is_nvidia_gpu(const dnnl_engine_t &engine) {
#if defined(DNNL_WITH_SYCL) && DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    if (!is_gpu(engine)) return false;
    constexpr int nvidia_vendor_id = 0x10DE;
    auto eng = dnnl::engine(engine, true);
    auto device = dnnl::sycl_interop::get_device(eng);
    const auto eng_vendor_id
            = device.get_info<::sycl::info::device::vendor_id>();
    return eng_vendor_id == nvidia_vendor_id;
#endif
    return false;
}

bool is_amd_gpu(const dnnl_engine_t &engine) {
#if defined(DNNL_WITH_SYCL) && DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
    if (!is_gpu(engine)) return false;
    constexpr int amd_vendor_id = 0x1002;
    auto eng = dnnl::engine(engine, true);
    auto device = dnnl::sycl_interop::get_device(eng);
    const auto eng_vendor_id
            = device.get_info<::sycl::info::device::vendor_id>();
    return eng_vendor_id == amd_vendor_id;
#endif
    return false;
}

bool is_generic_gpu(const dnnl_engine_t &engine) {
#if defined(DNNL_WITH_SYCL) && DNNL_GPU_VENDOR == DNNL_VENDOR_GENERIC
    return is_gpu(engine);
#endif

    return false;
}

bool is_f64_supported(const dnnl_engine_t &engine) {
    if (!is_gpu(engine)) return false;
    if (is_nvidia_gpu(engine) || is_amd_gpu(engine)) return false;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    if (is_sycl_engine(engine)) {
        auto eng = dnnl::engine(engine, true);
        auto dev = dnnl::sycl_interop::get_device(eng);
        return dev.has(::sycl::aspect::fp64);
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (is_opencl_engine(engine)) {
        auto eng = dnnl::engine(engine, true);
        cl_device_id dev = dnnl::ocl_interop::get_device(eng);
        size_t param_size = 0;
        cl_int err = clGetDeviceInfo(
                dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &param_size);
        if (err != CL_SUCCESS) return false;

        std::string extension_string(param_size, '\0');
        err = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, param_size,
                &extension_string[0], &param_size);
        if (err != CL_SUCCESS) return false;

        return extension_string.find("cl_khr_fp64") != std::string::npos;
    }
#endif
    return false;
}

#if defined(_WIN32)
#include "windows.h"

size_t get_cpu_ram_size() {
    MEMORYSTATUSEX s {};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    return s.ullTotalPhys;
}
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__QNXNTO__)
#include <unistd.h>
#include <sys/sysctl.h>

size_t get_cpu_ram_size() {
#ifdef __APPLE__
    int query_ram[] = {CTL_HW, HW_MEMSIZE};
#else
    int query_ram[] = {CTL_HW, HW_PHYSMEM};
#endif
    int query_ram_len = sizeof(query_ram) / sizeof(*query_ram);
    size_t totalram = 0;
    size_t length = sizeof(totalram);

    sysctl(query_ram, query_ram_len, &totalram, &length, NULL, 0);
    return totalram;
}
#else
#include <sys/sysinfo.h>

size_t get_cpu_ram_size() {
    struct sysinfo s {};
    sysinfo(&s);
    return s.totalram;
}
#endif

int get_gpu_ram_sizes(size_t &ram_size, size_t &max_alloc_size) {
    if (!is_gpu()) return OK;
    if (ram_size > 0 && max_alloc_size > 0) return OK;

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    auto eng = dnnl::engine(get_test_engine(), true);
    cl_int status = CL_SUCCESS;
    cl_device_id ocl_device = dnnl::ocl_interop::get_device(eng);

    cl_ulong ram_sz = 0;
    status = clGetDeviceInfo(ocl_device, CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(cl_ulong), &ram_sz, nullptr);
    if (status != CL_SUCCESS) return FAIL;

    cl_ulong max_alloc_sz = 0;
    status = clGetDeviceInfo(ocl_device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
            sizeof(cl_ulong), &max_alloc_sz, nullptr);
    if (status != CL_SUCCESS) return FAIL;

    ram_size = (size_t)ram_sz;
    max_alloc_size = (size_t)max_alloc_sz;
    return OK;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto eng = dnnl::engine(get_test_engine(), true);
    auto sycl_dev = dnnl::sycl_interop::get_device(eng);
    ram_size = (size_t)sycl_dev
                       .get_info<::sycl::info::device::global_mem_size>();
    max_alloc_size
            = (size_t)sycl_dev
                      .get_info<::sycl::info::device::max_mem_alloc_size>();
    return OK;
#endif
    ram_size = 0;
    max_alloc_size = 0;
    return OK;
}

int get_cpu_cache_size(cpu_cache_args_t &cache_args) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    using namespace dnnl::impl::cpu::platform;
    cache_args.L2_size = get_per_core_cache_size(2);
    cache_args.L3_size = get_per_core_cache_size(3);
    cache_args.num_cores = get_num_cores();
    cache_args.total_socket_size
            = (cache_args.L2_size + cache_args.L3_size) * cache_args.num_cores;
#else
    // If functions are not available, just use 150 MiB.
    cache_args.total_socket_size = 150 * 1024 * 1024;
#endif
    return OK;
}

int get_gpu_cache_size(size_t &cache_size) {
    if (!is_gpu()) return OK;

    static size_t _cache_size = 0;
    if (_cache_size > 0) {
        cache_size = _cache_size;
        return OK;
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    auto eng = dnnl::engine(get_test_engine(), true);
    cl_int status = CL_SUCCESS;
    cl_device_id ocl_device = dnnl::ocl_interop::get_device(eng);

    cl_ulong cache_sz = 0;
    status = clGetDeviceInfo(ocl_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
            sizeof(cl_ulong), &cache_sz, nullptr);
    if (status != CL_SUCCESS) return FAIL;

    _cache_size = (size_t)cache_sz;
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
    auto eng = dnnl::engine(get_test_engine(), true);
    auto sycl_dev = dnnl::sycl_interop::get_device(eng);
    _cache_size
            = (size_t)sycl_dev
                      .get_info<::sycl::info::device::global_mem_cache_size>();
#endif
    cache_size = _cache_size;
    return OK;
}

static int check_total_size(
        const check_mem_size_args_t &check_mem_size_args, res_t *res) {
    static size_t cpu_device_capacity = get_cpu_ram_size();
    static size_t gpu_device_capacity = 0;
    static size_t gpu_max_alloc_capacity = 0;
    SAFE(get_gpu_ram_sizes(gpu_device_capacity, gpu_max_alloc_capacity), WARN);

    const size_t device_max_capacity
            = is_cpu() ? cpu_device_capacity : gpu_device_capacity;
    const size_t cpu_max_capacity = cpu_device_capacity;

    // 0.75f is taken randomly and is subject to change in future.
    const double capacity_factor = 0.75;
    const double benchdnn_device_limit = capacity_factor * device_max_capacity;
    const double benchdnn_cpu_limit = capacity_factor * cpu_max_capacity;
    assert(benchdnn_device_limit > 0 && benchdnn_cpu_limit > 0);

    auto GB = [](double bytes) { return bytes / powf(2, 30); };
    auto dir_c_str = [&res]() {
        return (res->mem_check_dir & FLAG_FWD) ? "FWD" : "BWD";
    };

    if (is_gpu()) {
        const bool fits_device_ram = check_mem_size_args.total_size_device
                <= benchdnn_device_limit;
        if (!fits_device_ram) {
            BENCHDNN_PRINT(2,
                    "[CHECK_MEM][%s]: Not enough device RAM for a problem.\n",
                    dir_c_str());
            res->state = SKIPPED;
            res->reason = skip_reason::not_enough_ram;
        }

        const bool all_allocation_fit_limit = std::all_of(
                check_mem_size_args.sizes.cbegin(),
                check_mem_size_args.sizes.cend(), [&](size_t s) {
                    const bool fit = s < gpu_max_alloc_capacity;
                    if (!fit) {
                        BENCHDNN_PRINT(2,
                                "[CHECK_MEM][%s]: Allocation of size %g GB "
                                "doesn't fit allocation limit of %g GB.\n",
                                dir_c_str(), GB(s), GB(gpu_max_alloc_capacity));
                    }
                    return fit;
                });
        if (!all_allocation_fit_limit) {
            res->state = SKIPPED;
            res->reason = skip_reason::not_enough_ram;
        }

        BENCHDNN_PRINT((!fits_device_ram ? 2 : 6),
                "[CHECK_MEM][%s]: Requested: %g GB; benchdnn_device_limit: %g "
                "GB; device_RAM_capacity: %g GB; gpu_max_alloc: %g GB;\n",
                dir_c_str(), GB(check_mem_size_args.total_size_device),
                GB(benchdnn_device_limit), GB(gpu_device_capacity),
                GB(gpu_max_alloc_capacity));
    }

    size_t total_size_cpu = check_mem_size_args.total_size_cpu;

    // Add device size as a simple method to account for integrated devices and
    // mapping/unmapping memory
    total_size_cpu += check_mem_size_args.total_size_device;
    bool fits_cpu_ram = total_size_cpu <= benchdnn_cpu_limit;

    if (!fits_cpu_ram) {
        BENCHDNN_PRINT(2,
                "[CHECK_MEM][%s]: Not enough CPU RAM for a problem.\n",
                dir_c_str());
        // Try to catch a huge scratchpad size requested by the library.
        // Use following logic:
        //     scratch_size
        // ---------------------- <= 0.75 (pre-defined threshold).
        // io_size + scratch_size
        //
        // 0.75 value supposed to be experimental and might be adjusted.
        static constexpr float scratch_trh = 0.75f;
        if (is_cpu()
                && check_mem_size_args.scratchpad_size
                        > scratch_trh * check_mem_size_args.total_size_device) {
            BENCHDNN_PRINT(2,
                    "[CHECK_MEM][%s]: CPU scratchpad size `%zu` exceeded a "
                    "given threshold `%zu`.\n",
                    dir_c_str(), check_mem_size_args.scratchpad_size,
                    (size_t)(scratch_trh
                            * check_mem_size_args.total_size_device));
            res->state = FAILED;
        } else {
            res->state = SKIPPED;
        }
        res->reason = skip_reason::not_enough_ram;
    }

    BENCHDNN_PRINT((!fits_cpu_ram ? 2 : 6),
            "[CHECK_MEM][%s]: Requested: %g GB; benchdnn_CPU_limit: %g GB; "
            "CPU_RAM_capacity: %g GB;\n",
            dir_c_str(), GB(total_size_cpu), GB(benchdnn_cpu_limit),
            GB(cpu_device_capacity));

    return res->state == FAILED ? FAIL : OK;
}

void add_md_size(const_dnnl_memory_desc_t md,
        check_mem_size_args_t &check_mem_size_args) {
    const auto mem_size = dnnl_memory_desc_get_size(md);
    // Runtime mem size is not defined.
    if (mem_size == 0 || mem_size == DNNL_RUNTIME_SIZE_VAL) return;

    check_mem_size_args.sizes.push_back(mem_size);

    // Original memory size.
    check_mem_size_args.total_size_device += mem_size;

    // GPU mapped memory factor.
    // All memory is mapped once it is created and unmapped only before
    // primitive execution. Device memory requires additional buffer for mapped
    // memory allocated on host (CPU).
    // Note: In DPC++ build oneDNN uses USM memory, which shouldn't require an
    // additional buffer, so map factor should be equal to 0 for DPC++.
    // However due to a driver issue oneDNN pretends that shared USM is not
    // accessible on the host, hence map will allocate an extra memory.
    const bool mapped_mem_factor = !is_cpu()
            && !has_bench_mode_modifier(mode_modifier_t::no_ref_memory);

    // Mapped memory for GPU backend on CPU.
    check_mem_size_args.total_size_cpu += mapped_mem_factor * mem_size;

    if (check_mem_size_args.is_scratchpad) {
        check_mem_size_args.scratchpad_size += mem_size;
    } else {
        const bool is_corr = has_bench_mode_bit(mode_bit_t::corr);
        const bool is_bitwise = has_bench_mode_bit(mode_bit_t::bitwise);
        // Reference memories are always tag::abx fp32, hence need re-creating
        // memory descriptor and take its size.
        auto ref_md = dnn_mem_t::init_md(
                query_md_ndims(md), query_md_dims(md), dnnl_f32, tag::abx);
        const auto ref_md_size = dnnl_memory_desc_get_size(ref_md);

        const size_t ref_mem_idx = check_mem_size_args.want_input ? 0 : 1;
        check_mem_size_args.total_ref_md_size[ref_mem_idx] = ref_md_size;

        // A memory copy for ref_compute, happens only in correctness.
        check_mem_size_args.total_size_cpu += is_corr * ref_md_size;

        // Comparison function allocates an additional tag::abx f32 memory.
        // This allocation holds for correctness and bitwise modes.
        const bool compare_mem_factor
                = !check_mem_size_args.want_input && (is_corr || is_bitwise);
        check_mem_size_args.total_size_cpu += compare_mem_factor * ref_md_size;

        // Bitwise comparison allocates an additional tag::abx f32 memory from
        // the first run to compare results against it.
        const bool bitwise_compare_mem_factor
                = !check_mem_size_args.want_input && is_bitwise;
        check_mem_size_args.total_size_cpu
                += bitwise_compare_mem_factor * ref_md_size;
    }
}

bool is_fwd_training(dnnl_prop_kind_t prop_kind) {
    return prop_kind == dnnl_forward_training
            || prop_kind == dnnl_prop_kind_undef;
}

bool is_fwd_prop_kind(dnnl_prop_kind_t prop_kind) {
    return prop_kind == dnnl_forward_training
            || prop_kind == dnnl_forward_inference
            || prop_kind == dnnl_prop_kind_undef;
}

void get_memory_bytes(check_mem_size_args_t &check_mem_size_args) {
    auto const_pd = check_mem_size_args.pd;
    const int n_idx = check_mem_size_args.want_input
            ? query_n_inputs(const_pd)
            : query_n_outputs(const_pd);
    const auto prop_kind = query_prop_kind(const_pd);
    const bool is_fwd = is_fwd_prop_kind(prop_kind);

#define MD(name) dnnl_query_##name##_md
    std::vector<dnnl_query_t> query_fwd_in_mds {MD(src), MD(weights)};
    std::vector<dnnl_query_t> query_fwd_out_mds {MD(dst), MD(workspace)};

    std::vector<dnnl_query_t> query_bwd_in_mds {
            MD(src), MD(weights), MD(dst), MD(diff_dst), MD(workspace)};
    std::vector<dnnl_query_t> query_bwd_out_mds {
            MD(diff_src), MD(diff_weights)};
#undef MD

    const auto &query_in_mds = is_fwd ? query_fwd_in_mds : query_bwd_in_mds;
    const auto &query_out_mds = is_fwd ? query_fwd_out_mds : query_bwd_out_mds;
    const auto &query_mds
            = check_mem_size_args.want_input ? query_in_mds : query_out_mds;

    for_(const auto query : query_mds)
    for (int idx = 0; idx < n_idx; ++idx) {
        const auto &md = query_md(const_pd, query, idx);
        add_md_size(md, check_mem_size_args);
    }

    // Binary post-op memories counted as input.
    if (check_mem_size_args.want_input) {
        auto const_attr_po = query_post_ops(const_pd);
        auto po_len = dnnl_post_ops_len(const_attr_po);
        for (int idx = 0; idx < po_len; ++idx) {
            const auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
            if (kind == dnnl_binary) {
                int po_arg
                        = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1;
                const auto &po_md = query_md(const_pd, po_arg);
                add_md_size(po_md, check_mem_size_args);
            }
        }
    }
}

int check_mem_size(const_dnnl_memory_desc_t md, res_t *res) {
    if (!mem_check) return OK;

    check_mem_size_args_t check_mem_size_args(nullptr, false);
    const auto md_size = dnnl_memory_desc_get_size(md);
    check_mem_size_args.total_size_device = md_size;
    check_mem_size_args.sizes.push_back(md_size);

    return check_total_size(check_mem_size_args, res);
}

int check_mem_size(const_dnnl_primitive_desc_t const_pd, res_t *res, dir_t dir,
        bool need_skip) {
    // Skip the check if it is disabled.
    if (!mem_check) return OK;

    // Skip the check if the test object won't be executed.
    if (!has_bench_mode_bit(mode_bit_t::exec)) return OK;

    // Skip the check if it has already happened for provided `dir`. Saves from
    // repreated run when the second test object is created to test the
    // primitive cache, but allows to verify both objects when a double-run
    // driver executes fwd-for-bwd first and bwd after.
    if (need_skip && res->mem_check_dir == dir) return OK;
    res->mem_check_dir = dir;

    // Get input sizes.
    check_mem_size_args_t check_mem_size_args(const_pd, /* input = */ true);
    get_memory_bytes(check_mem_size_args);

    // Get scratchpad size.
    // Since scratchpad modes are mutually excluded, it takes sizes of both
    // modes as either of them will report 0 size depending on the mode.
    const auto library_scratchpad_size = query_mem_consumption(const_pd);
    if (library_scratchpad_size > 0) {
        // Update same fields as `add_md_size` would. See details there.
        check_mem_size_args.sizes.push_back(library_scratchpad_size);
        check_mem_size_args.total_size_device += library_scratchpad_size;
        check_mem_size_args.scratchpad_size += library_scratchpad_size;
    } else {
        check_mem_size_args.is_scratchpad = true;
        const auto &scratchpad_md = query_md(const_pd, DNNL_ARG_SCRATCHPAD);
        add_md_size(scratchpad_md, check_mem_size_args);
        check_mem_size_args.is_scratchpad = false;
    }

    // Get output sizes.
    check_mem_size_args.want_input = false;
    get_memory_bytes(check_mem_size_args);

    // Save the mem size args for graph driver check.
    res->mem_size_args = check_mem_size_args;

    return check_total_size(check_mem_size_args, res);
}

int get_memory_footprint(const_dnnl_primitive_desc_t const_pd, res_t *res) {
    check_mem_size_args_t check_mem_in_size_args(
            const_pd, /* want_input = */ true);
    get_memory_bytes(check_mem_in_size_args); // Get input bytes.
    check_mem_size_args_t check_mem_out_size_args(
            const_pd, /* want_input = */ false);
    get_memory_bytes(check_mem_out_size_args); // Get output bytes.

    // Sum post-ops include dst bytes as an input. Not included in get_memory_bytes
    // since it would cause check_mem_size to double-count dst bytes.
    auto const_attr_po = query_post_ops(const_pd);
    auto po_len = dnnl_post_ops_len(const_attr_po);
    for (int idx = 0; idx < po_len; ++idx) {
        const auto kind = dnnl_post_ops_get_kind(const_attr_po, idx);
        if (kind == dnnl_sum) {
            const auto &dst_md = query_md(const_pd, DNNL_ARG_DST);
            add_md_size(dst_md, check_mem_in_size_args);
        }
    }

    // Forward nearest neighbor down sampling (one or more bigger source
    // dimensions than in destination) cases by the nature of the operation
    // read only a part of a source tensor. Here we adjust the `ibytes` variable
    // so that reported bandwidth numbers reflect the reality, otherwise
    // final numbers are much higher than they should be.
    dnnl_primitive_kind_t kind = query_prim_kind(const_pd);
    dnnl_alg_kind_t alg = query_alg_kind(const_pd);
    dnnl_prop_kind_t prop = query_prop_kind(const_pd);
    if (is_fwd_prop_kind(prop) && kind == dnnl_resampling
            && alg == dnnl_resampling_nearest) {
        auto src_md = query_md(const_pd, DNNL_ARG_SRC);
        auto dst_md = query_md(const_pd, DNNL_ARG_DST);
        auto ndims = query_md_ndims(src_md);
        auto src_pdims = query_md_padded_dims(src_md);
        auto dst_pdims = query_md_padded_dims(dst_md);
        dnnl_dim_t total_elems = 1;
        dnnl_dim_t read_elems = 1;
        for (int i = 0; i < ndims; i++) {
            if (dst_pdims[i] < src_pdims[i]) {
                total_elems *= src_pdims[i];
                read_elems *= dst_pdims[i];
            }
        }
        const auto src_size = dnnl_memory_desc_get_size(src_md);
        const auto fixed_src_size = src_size / static_cast<size_t>(total_elems)
                * static_cast<size_t>(read_elems);
        check_mem_in_size_args.total_size_device += fixed_src_size - src_size;
    }

    res->ibytes = check_mem_in_size_args.total_size_device;
    res->obytes = check_mem_out_size_args.total_size_device;

    return OK;
}

memory_kind_ext_t str2memory_kind(const char *str) {
#define CASE(param) \
    if (!strcasecmp(#param, str)) return memory_kind_ext_t::param

    CASE(usm);
    CASE(buffer);
    CASE(usm_device);
    CASE(usm_shared);

#undef CASE

    assert(!"not expected");
    return memory_kind_ext_t::usm;
}

const char *execution_mode2str(execution_mode_t mode) {
#define EXECUTION_MODE_TO_STR(name, ...) \
    if (execution_mode_t::name == mode) return #name;

    EXECUTION_MODE_TO_STR(direct);
    EXECUTION_MODE_TO_STR(graph);
#undef EXECUTION_MODE_STR

    BENCHDNN_PRINT(0, "%s", "Error: execution mode value is not recognized.\n");
    SAFE_V(FAIL);
    return "";
}

execution_mode_t str2execution_mode(const char *str) {
#define STR_TO_EXECUTION_MODE(name, ...) \
    if (!strcasecmp(#name, str)) return execution_mode_t::name;

    STR_TO_EXECUTION_MODE(direct);
    STR_TO_EXECUTION_MODE(graph);
#undef STR_TO_EXECUTION_MODE

    BENCHDNN_PRINT(0, "%s", "Error: execution mode value is not recognized.\n");
    SAFE_V(FAIL);
    return execution_mode_t::direct;
}

static void maybe_print_cpu_engine_error_message() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
    fprintf(stderr,
            "ERROR: can't create CPU engine. Possible reasons for this error:\n"
            "- Incorrect SYCL_DEVICE_FILTER. The filter must be either unset "
            "or include 'opencl:cpu' devices.\n"
            "- Missing TBB library which is required for OpenCL CPU runtime. "
            "Check that TBB library is available in the system.\n"
            "- Missing OpenCL CPU runtime or other issues with OpenCL CPU "
            "runtime. Check that output from `sycl-ls` or `clinfo -l` commands "
            "include any CPU devices.\n");
#endif
}

engine_t::engine_t(dnnl_engine_kind_t engine_kind) : is_owner_(true) {
    size_t idx = engine_kind == dnnl_cpu ? 0 : engine_index;
    dnnl_status_t status = dnnl_engine_create(&engine_, engine_kind, idx);
    if (engine_kind == dnnl_cpu && status != dnnl_success)
        maybe_print_cpu_engine_error_message();
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) {
        if (has_bench_mode_bit(mode_bit_t::corr)) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: the modifier to disable host memory usage "
                    "cannot be used for correctness testing.");
            status = dnnl_invalid_arguments;
        }
    }
    DNN_SAFE_V(status);
}

engine_t::engine_t(dnnl_engine_t engine) : engine_(engine), is_owner_(false) {}

engine_t::engine_t(const engine_t &other) : is_owner_(other.is_owner_) {
    if (!is_owner_) {
        engine_ = other.engine_;
        return;
    }

    dnnl_engine_kind_t engine_kind;
    DNN_SAFE_V(dnnl_engine_get_kind(other.engine_, &engine_kind));

    if (engine_kind == dnnl_cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        void *dev;
        void *ctx;
        DNN_SAFE_V(dnnl_sycl_interop_engine_get_device(other.engine_, &dev));
        DNN_SAFE_V(dnnl_sycl_interop_engine_get_context(other.engine_, &ctx));
        DNN_SAFE_V(dnnl_sycl_interop_engine_create(&engine_, dev, ctx));
#else
        DNN_SAFE_V(dnnl_engine_create(&engine_, dnnl_cpu, 0));
#endif
    } else if (engine_kind == dnnl_gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        cl_device_id dev;
        cl_context ctx;
        DNN_SAFE_V(dnnl_ocl_interop_get_device(other.engine_, &dev));
        DNN_SAFE_V(dnnl_ocl_interop_engine_get_context(other.engine_, &ctx));
        DNN_SAFE_V(dnnl_ocl_interop_engine_create(&engine_, dev, ctx));
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        void *dev;
        void *ctx;
        DNN_SAFE_V(dnnl_sycl_interop_engine_get_device(other.engine_, &dev));
        DNN_SAFE_V(dnnl_sycl_interop_engine_get_context(other.engine_, &ctx));
        DNN_SAFE_V(dnnl_sycl_interop_engine_create(&engine_, dev, ctx));
#endif
    } else {
        assert(!"unsupported engine kind");
    }
}

engine_t::~engine_t() {
    if (is_owner_) DNN_SAFE_V(dnnl_engine_destroy(engine_));
}

stream_t::stream_t(dnnl_engine_t engine, void *interop_obj) : is_owner_(true) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    if (is_cpu(engine)) {
        auto tp = static_cast<dnnl::threadpool_interop::threadpool_iface *>(
                interop_obj);
        if (tp == nullptr) tp = dnnl::testing::get_threadpool();
        SAFE_V(dnnl_threadpool_interop_stream_create(&stream_, engine, tp));
        return;
    }
#endif

    const bool use_profiling = has_bench_mode_bit(mode_bit_t::perf)
            && is_gpu(engine) && !is_nvidia_gpu(engine) && !is_amd_gpu(engine);
    dnnl_stream_flags_t flags
            = stream_kind2stream_flags(stream_kind, use_profiling);
    DNN_SAFE_V(dnnl_stream_create(&stream_, engine, flags));
}

stream_t::~stream_t() {
    if (is_owner_) DNN_SAFE_V(dnnl_stream_destroy(stream_));
}

stream_t &stream_t::operator=(stream_t &&rhs) {
    if (&rhs == this) return *this;

    stream_ = rhs.stream_;
    is_owner_ = rhs.is_owner_;

    rhs.is_owner_ = false;
    return *this;
}

float reorder_rescale_factor() {
    float factor = 1.f;
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    if (is_cpu(get_test_engine()))
        factor = dnnl::impl::cpu::platform::s8s8_weights_scale_factor();
#endif
    return factor;
}

dims_t md2dims(const_dnnl_memory_desc_t md, int mask, bool extend_by_ones,
        const std::vector<int64_t> &groups) {
    auto ndims = query_md_ndims(md);
    dims_t dims;
    for (int d = 0; d < ndims; ++d) {
        if (mask & (1 << d)) {
            dims.push_back(query_md_dims(md)[d]);
            // Note: groups are done for matmul's last two dimensions.
            const auto group_dim = d - (ndims - 2);
            if (!groups.empty() && group_dim >= 0) {
                // If groups are passed, divide dims on the correspondent group
                // size. It's needed to pass proper memory objects.
                assert(dims.back() % groups[group_dim] == 0);
                dims.back() /= groups[group_dim];
            }
        } else if (extend_by_ones) {
            dims.push_back(1);
        }
    }
    return dims;
}

dims_t md2dims(const_dnnl_memory_desc_t md, int mask, bool extend_by_ones) {
    return md2dims(md, mask, extend_by_ones, {});
}

dnnl_data_type_t deduce_cfg_data_type(
        dnnl_data_type_t in_dt, const attr_t &attr, data_kind_t dk) {
    dnnl_data_type_t dt_ = in_dt;

    if ((dk == SRC || dk == WEI) && dt_ == dnnl_f32) {
        // Update data type based on fpmath-mode attribute
        switch (attr.fpmath_mode.mode) {
            case dnnl_fpmath_mode_strict: break;
            case dnnl_fpmath_mode_bf16: dt_ = dnnl_bf16; break;
            case dnnl_fpmath_mode_f16: dt_ = dnnl_f16; break;
            case dnnl_fpmath_mode_tf32: dt_ = dnnl_bf16; break;
            default: assert(!"unsupported_fpmath_mode"); SAFE_V(CRIT);
        }
    } else if (dk == DST) {
        // Sum post-op defines the type of filling destination.
        const int sum_idx = attr.post_ops.find(attr_t::post_ops_t::SUM);
        if (sum_idx >= 0) {
            auto sum_dt = attr.post_ops.entry[sum_idx].sum.dt;
            if (sum_dt != dnnl_data_type_undef) dt_ = sum_dt;
        }
    }

    return dt_;
}

// This function handles cases when optimized CPU primitive is used as a
// reference for a problem. Optimized primitive means custom memory formats
// which require reorder to them. Since `ref_mem_map` is passed to optimized
// primitive, it's required to replace correspondent memory objects and update
// them with proper values to get the matched output.
// The function also handles cases when a target primitive doesn't need a
// scratchpad while the reference one does.
// Note: the last argument `swapped_dt` is a property of `driver::cfg_t`. `cfg`
// could be passed directly, but since it's tied to a `prb_t`, the function will
// requires a template argument. If more members from `cfg` would be needed,
// consider passing `cfg` directly.
int update_ref_mem_map_from_prim(dnnl_primitive_t prim_ref,
        const dnn_mem_t &library_mem, dnn_mem_map_t &ref_mem_map, int exec_arg,
        dnnl_data_type_t swapped_dt) {
    if (!prim_ref) return OK;

    const auto &ref_mem = ref_mem_map.at(exec_arg);
    const bool is_scratchpad = exec_arg == DNNL_ARG_SCRATCHPAD;

    // If `ref_mem` is empty (unless scratchpad since GPU may not have one
    // while CPU may), it means there's nothing to update.
    if (!ref_mem && !is_scratchpad) return OK;

    bool skip_replace = false;
    auto const_ref_pd = query_pd(prim_ref);
    const auto &ref_md = query_md(const_ref_pd, exec_arg);
    const auto &ref_engine = get_cpu_engine();
    dnn_mem_t prim_ref_mem(ref_md, ref_engine);

    // When queried memory comes empty, it may be attributes as library doesn't
    // have dedicated query mechanism for those. Process potential outcomes:
    while (query_md_ndims(ref_md) == 0) {
        bool is_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);
        // Scales received data type support in the library. The reference
        // primitive expects them in the same data type.
        if (is_scales_arg) {
            prim_ref_mem = dnn_mem_t(
                    library_mem.md_, library_mem.dt(), tag::abx, ref_engine);
            break;
        }

        bool is_zero_point_arg = (exec_arg & DNNL_ARG_ATTR_ZERO_POINTS);
        // Zero-points received data type support in the library. The reference
        // primitive expects them in the same data type.
        if (is_zero_point_arg) {
            prim_ref_mem = dnn_mem_t(
                    library_mem.md_, library_mem.dt(), tag::abx, ref_engine);
            break;
        }

        const int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
                - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
        const bool is_post_ops_arg = (exec_arg & post_ops_range);
        const bool is_prelu_arg
                = is_post_ops_arg && (exec_arg & DNNL_ARG_WEIGHTS);
        // The library doesn't return a memory desc for prelu post-op. Prelu
        // requires `tag::axb` format, thus, need to put a desc into ref prim.
        if (is_prelu_arg) {
            prim_ref_mem = dnn_mem_t(
                    library_mem.md_, dnnl_f32, tag::axb, ref_engine);
            break;
        }

        // Rest arguments don't need special handling and should be
        // skipped as empty.
        skip_replace = true;
        break;
    }

    // Avoid reordering on empty memories;
    // Avoid replacing memories that have already been properly filled.
    if (skip_replace) return OK;

    if (!is_scratchpad) SAFE(prim_ref_mem.reorder(ref_mem, swapped_dt), WARN);
    ref_mem_map[exec_arg] = std::move(prim_ref_mem);

    return OK;
}

// This function provides a general filling for atributes across all drivers.
//
// It provides a default filling config for both binary and prelu post-op.
// The user has an option to override it by passing `fill_cfg_map` with
// correspondent argument and attached `fill_cfg_t` object to it.
// Default filling configs are simple to avoid floating-point rounding effects,
// but not cancellation effects. For latter ones, it's the user's responsibility
// to avoid them by supplying a proper fill_cfg for a given driver/problem.
int init_ref_memory_args_default_case(int exec_arg, dnn_mem_t &mem,
        dnn_mem_t &ref_mem, const attr_t &attr, res_t *res,
        const std::unordered_map<int, fill_cfg_t> &fill_cfg_map) {
    assert(exec_arg > 0); // Negative values will produce false-positive `true`.

    const int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
            - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
    const bool is_post_ops_arg = (exec_arg & post_ops_range);
    const bool is_scales_arg = (exec_arg & DNNL_ARG_ATTR_SCALES);
    const bool is_zero_point_arg = (exec_arg & DNNL_ARG_ATTR_ZERO_POINTS);
    const bool is_dropout_p = (exec_arg == DNNL_ARG_ATTR_DROPOUT_PROBABILITY);
    const bool is_dropout_seed = (exec_arg == DNNL_ARG_ATTR_DROPOUT_SEED);
    const bool is_rounding_seed = (exec_arg == DNNL_ARG_ATTR_ROUNDING_SEED);

    if (is_post_ops_arg) {
        if (exec_arg & DNNL_ARG_SRC_1) {
            const int bin_po_idx
                    = exec_arg / DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE - 1;
            assert(bin_po_idx < attr.post_ops.len());
            const auto alg = attr.post_ops.entry[bin_po_idx].kind;
            // Binary post-op filling.
            fill_cfg_t def_binary_cfg(mem.dt(), -16.f, 16.f, /* int = */ true,
                    alg, "def_binary_post_op");
            const auto it = fill_cfg_map.find(DNNL_ARG_SRC_1);
            const bool has_external_cfg = it != fill_cfg_map.end();
            const fill_cfg_t &binary_fill_cfg
                    = has_external_cfg ? (*it).second : def_binary_cfg;
            TIME_FILL(SAFE(fill_random_real(mem, ref_mem, res, binary_fill_cfg),
                    WARN));
        } else if (exec_arg & DNNL_ARG_WEIGHTS) {
            // Prelu post-op filling.
            fill_cfg_t def_prelu_fill_cfg(mem.dt(), -2.f, 2.f, /* int = */ true,
                    attr_t::post_ops_t::kind_t::PRELU, "def_prelu_post_op");
            const auto it = fill_cfg_map.find(DNNL_ARG_WEIGHTS);
            const bool has_external_cfg = it != fill_cfg_map.end();
            const fill_cfg_t &prelu_fill_cfg
                    = has_external_cfg ? (*it).second : def_prelu_fill_cfg;
            TIME_FILL(SAFE(
                    fill_random_real(mem, ref_mem, res, prelu_fill_cfg), WARN));
        }
    } else if (is_scales_arg) {
        int local_exec_arg = exec_arg ^ DNNL_ARG_ATTR_SCALES;
        TIME_FILL(SAFE(fill_scales(attr, local_exec_arg, mem, ref_mem), WARN));
    } else if (is_zero_point_arg) {
        int local_exec_arg = exec_arg ^ DNNL_ARG_ATTR_ZERO_POINTS;
        TIME_FILL(SAFE(
                fill_zero_points(attr, local_exec_arg, mem, ref_mem), WARN));
    } else if (is_dropout_p) {
        ref_mem.set_elem(0, attr.dropout.p);
        TIME_FILL(SAFE(mem.reorder(ref_mem), WARN));
    } else if (is_dropout_seed) {
        ref_mem.set_elem(0, attr.dropout.seed);
        TIME_FILL(SAFE(mem.reorder(ref_mem), WARN));
    } else if (is_rounding_seed) {
        ref_mem.set_elem(0, attr.rounding_mode.seed);
        TIME_FILL(SAFE(mem.reorder(ref_mem), WARN));
    }

    return OK;
}

// This function is responsible for performing the bitwise validation:
// * Saves the output of the first run for further comparison.
// * Refreshes the input data when the original data was corrupted, e.g.,
//   inplace mode or sum post-op.
// * Performs a second run of the original primitive and its inputs.
// * Compares both outputs on bitwise exactness.
//
// The function takes the following arguments:
// * A `prim` object, the same one used for the first run.
// * A vector of `kinds` to validate each output from the primitive. The exactly
//   same one is used for correctness validation.
// * `args` arguments with all memory objects used for the first run and also
//   with stashed memories when they got overwritten during execution.
// * `inplace` flags to help identify if data refresh is needed.
// * `res` object to save the state of the validation result.
//
int check_bitwise(dnnl_primitive_t prim, const std::vector<data_kind_t> &kinds,
        const args_t &args, const attr_t &attr, bool inplace, res_t *res) {
    // Fast exit for any modes but bitwise.
    if (!has_bench_mode_bit(mode_bit_t::bitwise)) return OK;

    // Forward-for-backward service primitives define `kinds` as empty to skip
    // validation. This is to avoid extra checks on higher level.
    if (kinds.empty()) return OK;

    // Collect copies of outputs.
    dnn_mem_map_t run1_mem_map;
    for (const auto &kind : kinds) {
        const int arg = data_kind2exec_arg(kind);
        assert(arg > 0);

        auto &mem = args.find(arg);
        if (!mem) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: output memory was not found among arguments.");
            res->state = FAILED;
            return FAIL;
        }
        // A memory used as reference for comparison, must be allocated on the
        // CPU engine.
        run1_mem_map.emplace(
                arg, dnn_mem_t(mem.md_, dnnl_f32, tag::abx, get_cpu_engine()));
        SAFE(run1_mem_map.at(arg).reorder(mem), WARN);
    }

    // Put original data into DST tensor if sum post-op is present.
    if (query_post_ops_has_kind(prim, dnnl_sum)) {
        const int query_arg = DNNL_ARG_DST;
        auto &dst_mem = const_cast<dnn_mem_t &>(args.find(query_arg));
        const auto &orig_dst_mem = args.find(-query_arg);
        SAFE_V(bool(orig_dst_mem) && bool(dst_mem) ? OK : FAIL);
        SAFE(dst_mem.reorder(orig_dst_mem), WARN);
    }

    // Put original data into SRC if inplace mode was specified.
    if (inplace) {
        const bool has_multiple_args = bool(args.find(DNNL_ARG_MULTIPLE_SRC));
        const auto prop_kind = query_prop_kind(query_pd(prim));
        const auto query_arg = is_fwd_prop_kind(prop_kind)
                ? (has_multiple_args ? DNNL_ARG_MULTIPLE_SRC : DNNL_ARG_SRC)
                : DNNL_ARG_DIFF_DST;
        auto &in_mem = const_cast<dnn_mem_t &>(args.find(query_arg));
        if (!in_mem) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: input memory was not found among arguments.");
            res->state = FAILED;
            return FAIL;
        }
        const auto &orig_in_mem = args.find(-query_arg);
        if (!orig_in_mem) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: original input memory was not found among "
                    "arguments.");
            res->state = FAILED;
            return FAIL;
        }
        SAFE(in_mem.reorder(orig_in_mem), WARN);
    }

    // Perform a second run.
    SAFE(execute_and_wait(prim, args, res), WARN);

    // `args_t` has an interface to retrieve a memory object by the `arg`.
    args_t run1_args(run1_mem_map);
    compare::compare_t cmp;
    for (const auto &kind : kinds) {
        cmp.set_data_kind(kind);

        const int arg = data_kind2exec_arg(kind);
        assert(arg > 0);

        auto &mem = args.find(arg);
        auto &run1_mem = run1_args.find(arg);

        TIME_COMPARE(cmp.compare(run1_mem, mem, attr, res));
    }

    return OK;
}

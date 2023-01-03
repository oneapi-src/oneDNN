/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef TEST_THREAD_HPP
#define TEST_THREAD_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl_config.h"

#ifdef COMMON_DNNL_THREAD_HPP
#error "src/common/dnnl_thread.hpp" was already included
#endif

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE

#if DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_SEQ
#error "DNNL_CPU_THREADING_RUNTIME is expected to be SEQ for GPU only configurations."
#endif

#undef DNNL_CPU_THREADING_RUNTIME

// Enable CPU threading layer for testing:
// - DPCPP: TBB
// - OCL: OpenMP
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_TBB
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_OMP
#endif

#endif

// Here we define some types in global namespace to handle customized
// threading context for creation and execution
struct thr_ctx_t {
    int max_concurrency;
    int core_type;
    int nthr_per_core;

    bool operator==(const thr_ctx_t &rhs) const {
        return max_concurrency == rhs.max_concurrency
                && core_type == rhs.core_type
                && nthr_per_core == rhs.nthr_per_core;
    }
    bool operator!=(const thr_ctx_t &rhs) const { return !(*this == rhs); }
    void *get_interop_obj() const;
};

// tbb constraints on core type appear in 2021.2
// tbb constraints on max_concurrency appear in 2020
// we check only for 2021.2 to enable thread context knobs
#ifdef TBB_INTERFACE_VERSION
#define DNNL_TBB_CONSTRAINTS_ENABLED (TBB_INTERFACE_VERSION >= 12020)
#else
#define DNNL_TBB_CONSTRAINTS_ENABLED 0
#endif

#define DNNL_TBB_THREADING_WITH_CONSTRAINTS \
    (DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB) \
            && DNNL_TBB_CONSTRAINTS_ENABLED
#define DNNL_TBB_THREADING_WITHOUT_CONSTRAINTS \
    (DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB) \
            && !DNNL_TBB_CONSTRAINTS_ENABLED

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ \
        || DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL \
        || DNNL_TBB_THREADING_WITHOUT_CONSTRAINTS
const thr_ctx_t default_thr_ctx = {0, -1, 0};
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
#include "omp.h"
const thr_ctx_t default_thr_ctx = {omp_get_max_threads(), -1, 0};
#elif DNNL_TBB_THREADING_WITH_CONSTRAINTS
#include "oneapi/tbb/task_arena.h"
const thr_ctx_t default_thr_ctx = {tbb::task_arena::automatic,
        tbb::task_arena::automatic, tbb::task_arena::automatic};
#endif

std::ostream &operator<<(std::ostream &os, const thr_ctx_t &ctx);

#define THR_CTX_ASSERT(check, msg_fmt, ...) \
    do { \
        if (!(check)) { \
            fprintf(stderr, msg_fmt, __VA_ARGS__); \
            exit(1); \
        } \
    } while (0)

// This hack renames the namespaces used by threading functions for
// threadpool-related functions so that the calls to dnnl::impl::parallel*()
// from the test use a special testing threadpool.
//
// At the same time, the calls to dnnl::impl::parallel*() from within the
// library continue using the library version of these functions.
#define threadpool_utils testing_threadpool_utils
#include "src/common/dnnl_thread.hpp"
#undef threadpool_utils

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
// Restore the original DNNL_CPU_THREADING_RUNTIME value.
#undef DNNL_CPU_THREADING_RUNTIME
#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_SEQ
#endif

#ifndef COMMON_DNNL_THREAD_HPP
#error "src/common/dnnl_thread.hpp" has an unexpected header guard
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
namespace dnnl {

// Original threadpool utils are used by the scoped_tp_activation_t and thus
// need to be re-declared because of the hack above.
namespace impl {
namespace threadpool_utils {
void activate_threadpool(dnnl::threadpool_interop::threadpool_iface *tp);
void deactivate_threadpool();
dnnl::threadpool_interop::threadpool_iface *get_active_threadpool();
} // namespace threadpool_utils
} // namespace impl

namespace testing {

dnnl::threadpool_interop::threadpool_iface *get_threadpool(
        const thr_ctx_t &ctx = default_thr_ctx);

// Sets the testing threadpool as active for the lifetime of the object.
// Required for the tests that throw to work.
struct scoped_tp_activation_t {
    scoped_tp_activation_t(dnnl::threadpool_interop::threadpool_iface *tp_
            = get_threadpool()) {
        impl::threadpool_utils::activate_threadpool(tp_);
    }
    ~scoped_tp_activation_t() {
        impl::threadpool_utils::deactivate_threadpool();
    }
};

struct scoped_tp_deactivation_t {
    scoped_tp_deactivation_t() {
        impl::threadpool_utils::deactivate_threadpool();
    }
    ~scoped_tp_deactivation_t() {
        // we always use the same threadpool that is returned by `get_threadpool()`
        impl::threadpool_utils::activate_threadpool(get_threadpool());
    }
};

} // namespace testing
} // namespace dnnl
#endif

// These are free functions to allow running a function in a given threading context.
// A threading context is defined by:
// - number of threads
// - type of cores (TBB only)
// - threads per core (TBB only)

// Note: we have to differentiate creation and execution in thread
// context because of threadpool as it uses different mecanisms in
// both (in execution, tp is passed in stream)

#define ALIAS_TO_RUN_IN_THR_CTX(name) \
    template <typename F, class... Args_t> \
    auto name(const thr_ctx_t &ctx, F &&f, Args_t &...args) \
            ->decltype(run_in_thr_ctx<F, Args_t...>(ctx, f, args...)) { \
        return run_in_thr_ctx<F, Args_t...>(ctx, f, args...); \
    }

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ \
        || DNNL_TBB_THREADING_WITHOUT_CONSTRAINTS
template <typename F, class... Args_t>
auto run_in_thr_ctx(const thr_ctx_t &ctx, F &&f, Args_t &...args)
        -> decltype(f(args...)) {

    THR_CTX_ASSERT(ctx.core_type == default_thr_ctx.core_type
                    && ctx.max_concurrency == default_thr_ctx.max_concurrency
                    && ctx.nthr_per_core == default_thr_ctx.nthr_per_core,
            "Threading knobs not supported for this runtime: %s\n",
            DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
                    ? "sequential runtime has no threading"
                    : "TBB version is too old (>=2021.2 required)");

    return f(args...);
}

ALIAS_TO_RUN_IN_THR_CTX(create_in_thr_ctx)
ALIAS_TO_RUN_IN_THR_CTX(execute_in_thr_ctx)

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
template <typename F, class... Args_t>
auto run_in_thr_ctx(const thr_ctx_t &ctx, F &&f, Args_t &...args)
        -> decltype(f(args...)) {

    THR_CTX_ASSERT(ctx.core_type == default_thr_ctx.core_type,
            "core type %d is not supported for OMP runtime\n", ctx.core_type);

    auto max_nthr = omp_get_max_threads();
    omp_set_num_threads(ctx.max_concurrency);
    auto st = f(args...);
    omp_set_num_threads(max_nthr);
    return st;
}

ALIAS_TO_RUN_IN_THR_CTX(create_in_thr_ctx)
ALIAS_TO_RUN_IN_THR_CTX(execute_in_thr_ctx)

#elif DNNL_TBB_THREADING_WITH_CONSTRAINTS
#include "oneapi/tbb/info.h"

template <typename F, class... Args_t>
auto run_in_thr_ctx(const thr_ctx_t &ctx, F &&f, Args_t &...args)
        -> decltype(f(args...)) {
    static auto core_types
            = tbb::info::core_types(); // sorted by the relative strength

    if ((ctx.core_type != default_thr_ctx.core_type)
            && (ctx.core_type >= core_types.size()))
        printf("WARNING: TBB smallest core has index %lu. Using this "
               "instead of %d.\n",
                core_types.size() - 1, ctx.core_type);
    size_t core_type_id = ctx.core_type < core_types.size()
            ? ctx.core_type
            : core_types.size() - 1;
    static auto core_type = ctx.core_type == tbb::task_arena::automatic
            ? tbb::task_arena::automatic
            : core_types[core_type_id];
    static auto arena = tbb::task_arena {
            tbb::task_arena::constraints {}
                    .set_core_type(core_type)
                    .set_max_threads_per_core(ctx.nthr_per_core)
                    .set_max_concurrency(ctx.max_concurrency)};
    return arena.execute([&] { return f(args...); });
}

ALIAS_TO_RUN_IN_THR_CTX(create_in_thr_ctx)
ALIAS_TO_RUN_IN_THR_CTX(execute_in_thr_ctx)

#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
template <typename F, class... Args_t>
auto create_in_thr_ctx(const thr_ctx_t &ctx, F &&f, Args_t &...args)
        -> decltype(f(args...)) {
    THR_CTX_ASSERT(ctx.core_type == default_thr_ctx.core_type,
            "core type %d is not supported for TP runtime\n", ctx.core_type);

    auto tp = dnnl::testing::get_threadpool(ctx);
    auto stp = dnnl::testing::scoped_tp_activation_t(tp);
    return f(args...);
}

// The function f shall take an interop obj as last argument
template <typename F, class... Args_t>
auto execute_in_thr_ctx(const thr_ctx_t &ctx, F &&f, Args_t &...args)
        -> decltype(f(args...)) {
    THR_CTX_ASSERT(ctx.core_type == default_thr_ctx.core_type,
            "core type %d is not supported for TP runtime\n", ctx.core_type);
    return f(args...);
}

#else
#error __FILE__"(" __LINE__ ")" "unsupported threading runtime!"
#endif

#undef ALIAS_TO_RUN_IN_THR_CTX
#undef THR_CTX_ASSERT
#undef DNNL_TBB_THREADING_WITHOUT_CONSTRAINTS
#undef DNNL_TBB_THREADING_WITH_CONSTRAINTS
#undef DNNL_TBB_CONSTRAINTS_ENABLED

#endif

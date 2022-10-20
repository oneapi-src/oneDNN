/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#ifndef DNNL_COMMON_HPP
#define DNNL_COMMON_HPP

#include <functional>
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "oneapi/dnnl/dnnl.h"
#include "src/common/bfloat16.hpp"
#include "src/common/float16.hpp"
#include "src/common/nstl.hpp"

int check_pd_cache(const_dnnl_primitive_desc_t pd);
int check_primitive_cache(dnnl_primitive_t p);

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_debug.hpp"
#include "dnnl_memory.hpp"
#include "utils/compare.hpp"
#include "utils/dims.hpp"
#include "utils/dnnl_query.hpp"

#include "tests/test_thread.hpp"

#define for_ for

#define DNN_SAFE(f, s) \
    do { \
        dnnl_status_t status__ = f; \
        if (status__ != dnnl_success) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, \
                        status2str(status__), (int)status__); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            return FAIL; \
        } \
    } while (0)

#define DNN_SAFE_V(f) \
    do { \
        dnnl_status_t status__ = f; \
        if (status__ != dnnl_success) { \
            BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                    __PRETTY_FUNCTION__, __LINE__, STRINGIFY(f), \
                    status2str(status__), (int)status__); \
            fflush(0); \
            exit(2); \
        } \
    } while (0)

#define DNN_SAFE_STATUS(f) \
    do { \
        dnnl_status_t status__ = f; \
        if (status__ != dnnl_success) { return status__; } \
    } while (0)

/* aux */
using bfloat16_t = dnnl::impl::bfloat16_t;
using float16_t = dnnl::impl::float16_t;
template <dnnl_data_type_t>
struct prec_traits;
template <>
struct prec_traits<dnnl_bf16> {
    typedef bfloat16_t type;
};
template <>
struct prec_traits<dnnl_f16> {
    typedef float16_t type;
};
template <>
struct prec_traits<dnnl_f32> {
    typedef float type;
};

// XXX: benchdnn infra doesn't support double yet.
// Use float's max/min/epsilon values to avoid following build warnings:
// warning C4756: overflow in constant arithmetic.
// This should be fixed once cpu reference in f64 is added.
template <>
struct prec_traits<dnnl_f64> {
    typedef float type;
};
template <>
struct prec_traits<dnnl_s32> {
    typedef int32_t type;
};
template <>
struct prec_traits<dnnl_s8> {
    typedef int8_t type;
};
template <>
struct prec_traits<dnnl_u8> {
    typedef uint8_t type;
};

#define CASE_ALL(dt) \
    switch (dt) { \
        CASE(dnnl_bf16); \
        CASE(dnnl_f16); \
        CASE(dnnl_f32); \
        CASE(dnnl_f64); \
        CASE(dnnl_s32); \
        CASE(dnnl_s8); \
        CASE(dnnl_u8); \
        default: assert(!"bad data_type"); \
    }

/* std::numeric_limits::digits functionality */
inline int digits_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::digits;

    CASE_ALL(dt);

#undef CASE
    return 0;
}

inline float epsilon_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::epsilon();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

inline float lowest_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::lowest();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

inline float max_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::max();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

#undef CASE_ALL

#define BENCHDNN_S32_TO_F32_SAT_CONST 2147483520.f

template <dnnl_data_type_t dt>
inline float saturate_and_round(float val) {
    const float dt_max = max_dt(dt);
    const float dt_min = (float)dnnl::impl::nstl::numeric_limits<
            typename prec_traits<dt>::type>::lowest();
    if (dt == dnnl_s32 && val >= max_dt(dnnl_s32)) return max_dt(dnnl_s32);
    if (val > dt_max) val = dt_max;
    if (val < dt_min) val = dt_min;
    return mxcsr_cvt(val);
}

inline bool is_integral_dt(dnnl_data_type_t dt) {
    return dt == dnnl_s32 || dt == dnnl_s8 || dt == dnnl_u8;
}

inline float maybe_saturate(dnnl_data_type_t dt, float value) {
    if (!is_integral_dt(dt)) return value;

    switch (dt) {
#define CASE(dt) \
    case dt: return saturate_and_round<dt>(value);
        CASE(dnnl_s32);
        CASE(dnnl_s8);
        CASE(dnnl_u8);
#undef CASE
        default: assert(!"bad data_type");
    }
    return 0;
}

float round_to_nearest_representable(dnnl_data_type_t dt, float value);

extern dnnl_engine_kind_t engine_tgt_kind;
extern size_t engine_index;
extern isa_hints_t hints;

struct engine_t {
    engine_t(dnnl_engine_kind_t engine_kind);
    engine_t(dnnl_engine_t engine);
    engine_t(const engine_t &other);
    ~engine_t();
    operator dnnl_engine_t() const { return engine_; }

private:
    engine_t &operator=(engine_t &other) = delete;
    dnnl_engine_t engine_;
    bool is_owner_;
};

struct stream_t {
    stream_t(dnnl_engine_t engine, void *interop_obj = nullptr);
    ~stream_t();
    operator dnnl_stream_t() const { return stream_; }

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(stream_t);
    dnnl_stream_t stream_;
};

// Engine used to run oneDNN primitives for testing.
inline const engine_t &get_test_engine() {
    if (is_bench_mode(PROF)) {
        bool is_profiling_supported = false;
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL \
        || DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        is_profiling_supported = (engine_tgt_kind == dnnl_gpu);
#endif

        if (!is_profiling_supported) {
            fprintf(stderr,
                    "Profiling-based performance mode is supported for OpenCL "
                    "and DPC++ only.\n");
            exit(2);
        }
    }
    static const engine_t instance(engine_tgt_kind);
    return instance;
}

// Engine used to run all reference native implementations and CPU
// implementations used by `--fast-ref-gpu` option.
inline const engine_t &get_cpu_engine() {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE
    // In case of lacking CPU engine, just re-use testing one.
    return get_test_engine();
#else
    static const engine_t instance(dnnl_cpu);
    return instance;
#endif
}

bool is_cpu(const dnnl_engine_t &engine = get_test_engine());
bool is_gpu(const dnnl_engine_t &engine = get_test_engine());
bool is_sycl_engine(const dnnl_engine_t &engine = get_test_engine());
bool is_opencl_engine(const dnnl_engine_t &engine = get_test_engine());
bool is_nvidia_gpu(const dnnl_engine_t &engine = get_test_engine());
bool is_f64_supported(const dnnl_engine_t &engine = get_test_engine());
bool is_amd_gpu(const dnnl_engine_t &engine = get_test_engine());

// Extended version of dnnl_sycl_interop_memory_kind_t enumeration.
enum class memory_kind_ext_t {
    usm, // Same as dnnl_sycl_interop_usm
    buffer, // Same as dnnl_sycl_interop_buffer
    usm_device, // USM allocated via malloc_device()
    usm_shared, // USM allocated via malloc_shared()
};

const memory_kind_ext_t default_memory_kind = memory_kind_ext_t::usm;

extern memory_kind_ext_t memory_kind;

void init_isa_settings();

struct args_t {
    args_t &set(int arg, const dnn_mem_t &mem);
    args_t &set(
            const std::vector<int> &args, const std::vector<dnn_mem_t> &mems);
    void clear() { args_.clear(); }

    int size() const { return (int)args_.size(); }

    const dnn_mem_t &find(int arg) const;

    int arg(int index) const { return args_[index].first; }
    const dnn_mem_t &dnn_mem(int index) const { return *args_[index].second; }

private:
    std::vector<std::pair<int, const dnn_mem_t *>> args_;
};

template <typename prb_t>
struct init_pd_args_t {
    init_pd_args_t(res_t *res, dnnl_engine_t engine, const prb_t *prb,
            dir_t dir, const_dnnl_primitive_desc_t hint)
        : pd(nullptr)
        , is_iterator_supported(true)
        , res(res)
        , engine(engine)
        , prb(prb)
        , dir(dir)
        , hint(hint) {}

    // Output members
    dnnl_primitive_desc_t pd;

    bool is_iterator_supported;

    // Input members
    res_t *res;
    dnnl_engine_t engine;
    const prb_t *prb;
    dir_t dir;
    const_dnnl_primitive_desc_t hint;
};

bool is_fwd_prop_kind(dnnl_prop_kind_t prop_kind);
int get_memory_footprint(const_dnnl_primitive_desc_t pd, res_t *res);
int check_same_pd(const dnnl_primitive_desc_t &pd_no_attr, res_t *res);
int test_persistent_cache_api(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim,
        const_dnnl_primitive_desc_t pd, res_t *res);
int check_mem_size(const_dnnl_memory_desc_t md, res_t *res);
int check_mem_size(const_dnnl_primitive_desc_t const_pd, res_t *res);

void skip_start(res_t *res);
void skip_unimplemented_data_type(
        const std::vector<dnnl_data_type_t> &v_dt, dir_t dir, res_t *res);
void skip_unimplemented_sum_po(const attr_t &attr, res_t *res,
        dnnl_data_type_t dst_dt = dnnl_data_type_undef);
void skip_invalid_inplace(res_t *res, dnnl_data_type_t sdt,
        dnnl_data_type_t ddt, const std::string &stag, const std::string &dtag);
void skip_unimplemented_arg_scale(const attr_t &attr, res_t *res);

// `check_dnnl_status` function is called to validate the result of primitive
// descriptor creation. Based on the status, it produces additional checks:
// * For `invalid_arguments` it just updates the `res` object with it.
// * For `unimplemented` it checks whether the lack of support is expected or
//   not. It relies on `skip_unimplemented_prb` function declared and defined
//   at every driver and expects it to find in correspondent namespace from
//   where `prb_t` was picked up. If the case is unknown, `UNIMPLEMENTED` status
//   will be returned.
template <typename prb_t>
int check_dnnl_status(dnnl_status_t status, const prb_t *prb, res_t *res) {
    if (!res || status == dnnl_success) return OK;

    switch (status) {
        case dnnl_invalid_arguments: res->state = INVALID_ARGUMENTS; break;
        case dnnl_unimplemented: {
            // Unconditionally set all Nvidia backend unimplemented cases as
            // not supported.
            if (is_nvidia_gpu() || is_amd_gpu()) {
                res->state = SKIPPED;
                res->reason = CASE_NOT_SUPPORTED;
                return OK;
            }

            // Check driver specific cases of unimplemented functionality.
            skip_unimplemented_prb(prb, res);
            if (res->state == SKIPPED) return OK;

            // If the case is not known to be skipped, it is unimplemented.
            res->state = UNIMPLEMENTED;
        } break;
        default: assert(!"unexpected");
    }
    return FAIL;
}

// `fetch_impl` is responsible to provide a valid `pd` under certain conditions:
// 1. Either valid `pd` or `pd_it` were provided.
// 2a. It's a service primitive (fwd-for-bwd or cpu-for-gpu or
//     simple-prims-of-complex-prim).
// 2b. It's a tested primitive and not all implementations hit skip-impl option
//     values.
template <typename prb_t>
int fetch_impl(benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> &pdw,
        init_pd_args_t<prb_t> &init_pd_args, res_t *res, bool is_service_prim) {
    if (!init_pd_args.pd) return FAIL;

    // Wrapper is expected to come empty.
    assert(!pdw);

    pdw.reset(init_pd_args.pd);

    // Service primitive is not supposed to utilize further logic.
    if (is_service_prim) return OK;

    while (true) {
        const auto impl_name = query_impl_info(pdw);
        // Skip-impl is not requested or hit. Latest pd already fetched.
        if (!maybe_skip(impl_name)) return OK;

        BENCHDNN_PRINT(6, "Implementation skipped: %s\n", impl_name.c_str());

        // Iterator is not supported, further logic is not applicable.
        if (!init_pd_args.is_iterator_supported) {
            res->state = SKIPPED;
            res->reason = SKIP_IMPL_HIT;
            return OK;
        }

        auto status = dnnl_primitive_desc_next_impl(pdw);
        if (status == dnnl_last_impl_reached) {
            BENCHDNN_PRINT(2, "%s\n", "All implementations were skipped!");
            res->state = SKIPPED;
            res->reason = SKIP_IMPL_HIT;
            pdw.reset(nullptr);
            return OK;
        } else if (status == dnnl_success) {
            continue;
        } else {
            BENCHDNN_PRINT(0, "%s\n", "Unexpected status from pd iterator.");
            return FAIL;
        }
    }

    // Unreached fail status.
    return FAIL;
}

// This is an internal to `init_prim` function that utilizes the logic of
// creating a `pd` and `prim` and assign them to input wrappers. It allows to
// remove code duplication and keep all the logic in a single place.
template <typename func_t, typename prb_t>
int create_primitive(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &primw,
        dnnl_engine_t engine, const func_t &init_pd_func, const prb_t *prb,
        res_t *res, dir_t dir, const_dnnl_primitive_desc_t hint,
        bool is_service_prim) {
    dnnl_status_t status = dnnl_success;
    dnnl_primitive_t prim {};

    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw;

    init_pd_args_t<prb_t> init_pd_args(res, engine, prb, dir, hint);
    status = init_pd_func(init_pd_args);

    SAFE(check_dnnl_status(status, prb, res), WARN);
    if (res->state == SKIPPED) return OK;

    // Fetch also checks if user requested to skip certain implementations.
    SAFE(fetch_impl(pdw, init_pd_args, res, is_service_prim), WARN);
    if (res->state == SKIPPED) return OK;

    DNN_SAFE(dnnl_primitive_create(&prim, pdw), WARN);
    primw.reset(prim);

    return OK;
}

template <typename func_t, typename prb_t>
int check_pd_w_and_wo_attr(dnnl_engine_t engine, const func_t &init_pd_func,
        const prb_t *prb, res_t *res, dir_t dir,
        const_dnnl_primitive_desc_t hint) {

    if (!attr_same_pd_check || prb->attr.is_def()) return OK;

    if (prb->attr.post_ops.convolution_index() != -1) return OK;

    // Check that adding attributes doesn't cause a fall back to another impl.
    auto *prb_mutable = const_cast<prb_t *>(prb);
    auto old_attr = prb_mutable->attr;
    prb_mutable->attr = attr_t();
    init_pd_args_t<prb_t> init_pd_args_without_attr(
            res, engine, prb_mutable, dir, hint);
    DNN_SAFE(init_pd_func(init_pd_args_without_attr), WARN);
    benchdnn_dnnl_wrapper_t<dnnl_primitive_desc_t> pdw(
            init_pd_args_without_attr.pd);
    prb_mutable->attr = old_attr;
    SAFE(check_same_pd(pdw, res), WARN);
    return OK;
}

template <typename func_t, typename prb_t>
int init_prim(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &user_prim,
        const func_t &init_pd_func, const prb_t *prb, res_t *res,
        dir_t dir = FLAG_FWD, const_dnnl_primitive_desc_t hint = nullptr,
        bool is_service_prim = false) {
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> primw;

    skip_start(res);
    if (res->state == SKIPPED) return OK;
    skip_invalid_prb(prb, res);
    if (res->state == SKIPPED) return OK;
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE

    // The idea is to create the requested primitive twice using different
    // engines but the same device and context in the case of OpenCL and DPCPP.
    // Rationale: make sure that the primitive cache is robust in the case
    // where CPU and GPU engines are re-created because this is a commonly
    // used scenario in the frameworks.
    engine_t engine(get_test_engine());

    // The first primitive creation using a temporary engine.
    SAFE(create_primitive(primw, engine, init_pd_func, prb, res, dir, hint,
                 is_service_prim),
            WARN);
    if (res->state == SKIPPED) return OK;

#endif
    // The second (if the cache is enabled) primitive creation using the global
    // test engine. This primitive is expected to come from the cache.
    SAFE(create_primitive(primw, get_test_engine(), init_pd_func, prb, res, dir,
                 hint, is_service_prim),
            WARN);
    if (res->state == SKIPPED) return OK;

    auto pd = query_pd(primw);
    SAFE(check_mem_size(pd, res), WARN);
    if (res->state == SKIPPED) return OK;

    // Further checks are only for tested primitives.
    if (is_service_prim) {
        user_prim.reset(primw.release());
        return OK;
    }

    res->impl_name = query_impl_info(pd);
    BENCHDNN_PRINT(5, "oneDNN implementation: %s\n", res->impl_name.c_str());
    // Check that adding attributes doesn't cause a fall back to another impl.
    SAFE(check_pd_w_and_wo_attr(
                 get_test_engine(), init_pd_func, prb, res, dir, hint),
            WARN);
    // Check primitive descriptor is picked up from the cache, if applicable.
    SAFE(check_pd_cache(pd), WARN);
    // Check primitive is picked up from the cache, if applicable.
    SAFE(check_primitive_cache(primw), WARN);
    // Collect memory footprint for a given primitive descriptor.
    SAFE(get_memory_footprint(pd, res), WARN);

    SAFE(test_persistent_cache_api(primw, pd, res), WARN);

    user_prim.reset(primw.release());
    return OK;
}

template <typename func_t, typename prb_t>
int init_prim(const thr_ctx_t &thr_ctx,
        benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &user_prim,
        const func_t &init_pd_func, prb_t *prb, res_t *res,
        dir_t dir = FLAG_FWD, const_dnnl_primitive_desc_t hint = nullptr,
        bool is_service_prim = false) {
    int (*f)(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &, func_t &, prb_t *,
            res_t *, dir_t, const_dnnl_primitive_desc_t, bool)
            = &init_prim<func_t, prb_t>;
    return create_in_thr_ctx(thr_ctx, f, user_prim, init_pd_func, prb, res, dir,
            hint, is_service_prim);
}

// `check_correctness` function is designed to be called from every driver where
// correctness validation is needed. It takes:
// * A pointer to a `prb_t` problem.
// * A vector of kinds to compare, to validate several outputs, if applicable.
// * Backend arguments to compare the output.
// * Driver's reference memory arguments to compute the reference path, then
//   setup a compare object, and, finally, compare the output.
// * A reference to function that sets up the compare object, see description
//   below.
// * A pointer to a `res_t` structure to update validation status.
// * An optional pointer to CPU primitive for speeding up reference path
//   computation on GPU.
//
// The function doesn't return status since we rely on `res` to contain all
// necessary information about validation results.
//
// The function performs several validation steps:
// * Checks that padded area of all memories are properly zeroed.
// * Checks that GPU backend haven't modified out-of-boundary memory regions.
// * Executes driver's reference path, using the problem, driver reference
//   arguments, and CPU primitive for GPU backend, if available.
// * For each kind to validate it:
//   - Creates and sets up the compare object. Setting is done with
//     `setup_cmp_func`.
//   - Finds correspondent memory arguments from backend and reference and
//     compares them.
//   - Result of comparison is saved into `res` object.
//
// `setup_cmp_func` is a function that supposed to be defined in every driver's
// namespace. Its interface is:
// `void (compare::compare_t &, const prb_t *, data_kind_t, const args_t &);`
// It takes:
// * A reference to a `compare_t` object which the function modifies based on
//   driver's needs.
// * A pointer to a `prb_t` problem.
// * `data_kind` value to help to setup threshold depending on output argument.
// * Driver's reference memory arguments since some drivers can't validate
//   certain scenarios for sure without additional memory arguments.
// Returns nothing since the object is modified by reference due to lifetime of
// the compare object is controlled by `check_correctness`.
//
// Note: a dedicated non-templated type for `setup_cmp_func_t` could be used but
// since it relies on a `prb_t` type which is individual for each driver,
// it is'nt possible without a template.
template <typename setup_cmp_func_t, typename prb_t>
void check_correctness(const prb_t *prb, const std::vector<data_kind_t> &kinds,
        const args_t &args, const args_t &ref_args,
        const setup_cmp_func_t &setup_cmp_func, res_t *res,
        dnnl_primitive_t prim_ref = nullptr) {

    for (int i = 0; i < args.size(); ++i) {
        check_zero_padding(args.dnn_mem(i), args.arg(i), res);
        check_buffer_overwrite(args.dnn_mem(i), args.arg(i), res);
    }

    TIME_REF(compute_ref(prb, ref_args, prim_ref));

    for (const auto &kind : kinds) {
        compare::compare_t cmp;
        cmp.set_data_kind(kind);
        setup_cmp_func(cmp, prb, kind, ref_args);

        int arg = 0;
        switch (kind) {
            case DST: arg = DNNL_ARG_DST; break;
            case SRC: arg = DNNL_ARG_DIFF_SRC; break;
            case SRC_1: arg = DNNL_ARG_DIFF_SRC_1; break;
            case WEI: arg = DNNL_ARG_DIFF_WEIGHTS; break;
            case BIA: arg = DNNL_ARG_DIFF_BIAS; break;
            case MEAN: arg = DNNL_ARG_MEAN; break;
            case VAR: arg = DNNL_ARG_VARIANCE; break;
            case SC: arg = DNNL_ARG_DIFF_SCALE; break;
            case SH: arg = DNNL_ARG_DIFF_SHIFT; break;
            case DST_ITER: arg = DNNL_ARG_DST_ITER; break;
            case DST_ITER_C: arg = DNNL_ARG_DST_ITER_C; break;
            case AUGRU_ATTENTION: arg = DNNL_ARG_DIFF_AUGRU_ATTENTION; break;
            case SRC_ITER: arg = DNNL_ARG_DIFF_SRC_ITER; break;
            case SRC_ITER_C: arg = DNNL_ARG_DIFF_SRC_ITER_C; break;
            case WEI_ITER: arg = DNNL_ARG_DIFF_WEIGHTS_ITER; break;
            case WEI_PEEPHOLE: arg = DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE; break;
            case WEI_PROJECTION: arg = DNNL_ARG_DIFF_WEIGHTS_PROJECTION; break;
            default: assert(!"unsupported kind"); SAFE_V(FAIL);
        }
        const auto &mem_dt = args.find(arg);
        const auto &mem_fp = ref_args.find(arg);

        cmp.compare(mem_fp, mem_dt, prb->attr, res);
    }
}

typedef std::function<dnnl_status_t(
        const dnnl_stream_t &, const std::vector<dnnl_exec_arg_t> &)>
        perf_function_t;

int execute_and_wait(perf_function_t &exec_func, const dnnl_engine_t &engine,
        const args_t &args, res_t *res = nullptr);
int execute_and_wait(
        dnnl_primitive_t prim, const args_t &args, res_t *res = nullptr);

void reset_gpu_profiling();
int measure_perf(const thr_ctx_t &ctx, res_t *res, perf_function_t &perf_func,
        args_t &args);
int measure_perf(
        const thr_ctx_t &ctx, res_t *res, dnnl_primitive_t prim, args_t &args);

void maybe_prepare_runtime_scales(dnn_mem_t &scales_m,
        const attr_t::scale_t &scale, int64_t scale_cnt, const float *scales);

void maybe_prepare_runtime_scales_v2(dnn_mem_t &scales_dt, dnn_mem_t &scales_fp,
        const attr_t::scale_t &scale, int64_t scale_cnt, const float *scales);

void maybe_prepare_runtime_zero_points(dnn_mem_t &zero_points_m,
        const attr_t &attr, int arg, int64_t count, const int32_t *zero_points);

void maybe_prepare_runtime_zero_points_v2(dnn_mem_t &zero_points_dt,
        dnn_mem_t &zero_points_fp, const attr_t &attr, int arg, int64_t count,
        const int32_t *zero_points);

std::vector<float> prepare_po_vals(const dnn_mem_t &dst_m, const args_t &args,
        const std::vector<std::pair<int, int>> &v_po_masks,
        const size_t dst_off);

bool check_md_consistency_with_tag(
        const_dnnl_memory_desc_t md, const std::string &tag);

memory_kind_ext_t str2memory_kind(const char *str);

float reorder_rescale_factor();
dims_t md2dims(const dnnl_memory_desc_t &md);

// Function adjusts data type if fpmath mode is present or sum_dt is different
// from destination_dt. It is used in `cfg` objects that regulate filling.
dnnl_data_type_t deduce_cfg_data_type(
        dnnl_data_type_t in_dt, const attr_t &attr, data_kind_t dk);

#endif

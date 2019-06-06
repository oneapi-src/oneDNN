/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef _MKLDNN_COMMON_HPP
#define _MKLDNN_COMMON_HPP

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "mkldnn.h"
#include "src/common/bfloat16.hpp"
#include "src/common/float16.hpp"
#include "src/common/nstl.hpp"

#include "common.hpp"
#include "dnn_types.hpp"
#include "mkldnn_debug.hpp"

#define DNN_SAFE(f, s) do { \
    mkldnn_status_t status = f; \
    if (status != mkldnn_success) { \
        if (s == CRIT || s == WARN) { \
            print(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                    __PRETTY_FUNCTION__, __LINE__, \
                    #f, status2str(status), (int)status); \
            fflush(0); \
            if (s == CRIT) exit(2); \
        } \
        return FAIL; \
    } \
} while(0)

#define DNN_SAFE_V(f) do { \
    mkldnn_status_t status = f; \
    if (status != mkldnn_success) { \
        print(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                __PRETTY_FUNCTION__, __LINE__, \
                STRINGIFY(f), status2str(status), (int)status); \
        fflush(0); \
        exit(2); \
    } \
} while(0)

/* aux */
using bfloat16_t = mkldnn::impl::bfloat16_t;
using float16_t = mkldnn::impl::float16_t;
template <mkldnn_data_type_t> struct prec_traits;
template <> struct prec_traits<mkldnn_bf16> { typedef bfloat16_t type; };
template <> struct prec_traits<mkldnn_f16> { typedef float16_t type; };
template <> struct prec_traits<mkldnn_f32> { typedef float type; };
template <> struct prec_traits<mkldnn_s32> { typedef int32_t type; };
template <> struct prec_traits<mkldnn_s8> { typedef int8_t type; };
template <> struct prec_traits<mkldnn_u8> { typedef uint8_t type; };

#define CASE_ALL(dt)                   \
    switch (dt) {                      \
        CASE(mkldnn_bf16);             \
        CASE(mkldnn_f16);              \
        CASE(mkldnn_f32);              \
        CASE(mkldnn_s32);              \
        CASE(mkldnn_s8);               \
        CASE(mkldnn_u8);               \
    default: assert(!"bad data_type"); \
    }

inline size_t sizeof_dt(mkldnn_data_type_t dt) {
#   define CASE(dt) \
    case dt: return sizeof(typename prec_traits<dt>::type);

    CASE_ALL(dt);

#   undef CASE
    return 0;
}

/* std::numeric_limits::digits functionality */
inline int digits_dt(mkldnn_data_type_t dt) {
#   define CASE(dt)                                \
    case dt:                                       \
        return mkldnn::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::digits;

    CASE_ALL(dt);

#   undef CASE
    return 0;
}

#undef CASE_ALL

template <mkldnn_data_type_t dt> inline float saturate(float val) {
    return MAX2(mkldnn::impl::nstl::numeric_limits<
            typename prec_traits<dt>::type>::lowest(),
            MIN2(mkldnn::impl::nstl::numeric_limits<
                typename prec_traits<dt>::type>::max(), mxcsr_round(val)));
}

inline float maybe_saturate(mkldnn_data_type_t dt, float value) {
    if (dt == mkldnn_s32 || dt == mkldnn_s8 || dt == mkldnn_u8) {
        switch (dt) {
#       define CASE(dt) case dt: { return saturate<dt>(value); }
        CASE(mkldnn_s32);
        CASE(mkldnn_s8);
        CASE(mkldnn_u8);
#       undef CASE
        default: assert(!"bad data_type");
        }
        return 0;
    }
    return value;
}

/* simplification */
extern mkldnn_engine_kind_t engine_tgt_kind;

extern mkldnn_engine_t engine_ref;
extern mkldnn_engine_t engine_tgt;

extern mkldnn_stream_t stream_ref;
extern mkldnn_stream_t stream_tgt;

extern "C" mkldnn_status_t mkldnn_engine_create_with_backend(
        mkldnn_engine_t *engine, mkldnn_engine_kind_t kind, int backend_kind,
        size_t index);
extern "C" mkldnn_status_t mkldnn_engine_get_backend_kind(
        mkldnn_engine_t engine, int *backend_kind);

inline int init() {
    /* Create engine with CPU native backend: backend_kind == 0 */
    DNN_SAFE(mkldnn_engine_create_with_backend(&engine_ref, mkldnn_cpu, 0, 0),
            CRIT);
    DNN_SAFE(mkldnn_stream_create(
                     &stream_ref, engine_ref, mkldnn_stream_default_flags),
            CRIT);

    if (!engine_tgt) {
        DNN_SAFE(mkldnn_engine_create(&engine_tgt, engine_tgt_kind, 0), CRIT);
        DNN_SAFE(mkldnn_stream_create(
                         &stream_tgt, engine_tgt, mkldnn_stream_default_flags),
                CRIT);
    }
    return OK;
}

inline int finalize() {
    DNN_SAFE(mkldnn_stream_destroy(stream_ref), CRIT);
    DNN_SAFE(mkldnn_stream_destroy(stream_tgt), CRIT);

    DNN_SAFE(mkldnn_engine_destroy(engine_ref), CRIT);
    DNN_SAFE(mkldnn_engine_destroy(engine_tgt), CRIT);
    return OK;
}

inline const char *query_impl_info(const_mkldnn_primitive_desc_t pd) {
    const char *str;
    mkldnn_primitive_desc_query(pd, mkldnn_query_impl_info_str, 0, &str);
    return str;
}

struct args_t {
    args_t &set(int arg, mkldnn_memory_t memory) {
        mkldnn_exec_arg_t a = {arg, memory};
        args_.push_back(a);
        return *this;
    }
    void clear() { args_.clear(); }

    int size() const { return (int)args_.size(); }
    const mkldnn_exec_arg_t *args() const { return args_.data(); }
    operator const mkldnn_exec_arg_t *() const { return args(); }
private:
    std::vector<mkldnn_exec_arg_t> args_;
};

inline mkldnn_status_t execute_and_wait(mkldnn_primitive_t prim,
        mkldnn_stream_t stream, int nargs, const mkldnn_exec_arg_t *args) {
    mkldnn_status_t status
            = mkldnn_primitive_execute(prim, stream, nargs, args);
    if (status != mkldnn_success)
        return status;
    return mkldnn_stream_wait(stream);
}

inline int measure_perf(benchdnn_timer_t &t, mkldnn_primitive_t prim,
        args_t &args) {
    if (bench_mode & PERF) {
        t.reset();
        while (true) {
            DNN_SAFE(execute_and_wait(prim, stream_tgt, args.size(), args),
                    WARN);
            t.stamp();
            const bool stop = false
                || (fix_times_per_prb && t.times() >= fix_times_per_prb)
                || (!fix_times_per_prb
                        && t.total_ms() >= max_ms_per_prb
                        && t.times() >= min_times_per_prb);
            if (stop) break;
        }
    }
    return OK;
}

#endif

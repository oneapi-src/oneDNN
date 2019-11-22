/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "dnnl.h"
#include "src/common/bfloat16.hpp"
#include "src/common/float16.hpp"
#include "src/common/nstl.hpp"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_debug.hpp"

#define for_ for

#define DNN_SAFE(f, s) \
    do { \
        dnnl_status_t status = f; \
        if (status != dnnl_success) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, status2str(status), \
                        (int)status); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            return FAIL; \
        } \
    } while (0)

#define DNN_SAFE_V(f) \
    do { \
        dnnl_status_t status = f; \
        if (status != dnnl_success) { \
            BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                    __PRETTY_FUNCTION__, __LINE__, STRINGIFY(f), \
                    status2str(status), (int)status); \
            fflush(0); \
            exit(2); \
        } \
    } while (0)

#define DNN_SAFE_CLEAN(f, s, clean) \
    do { \
        dnnl_status_t status = f; \
        if (status != dnnl_success) { \
            if (s == CRIT || s == WARN) { \
                BENCHDNN_PRINT(0, "error [%s:%d]: '%s' -> %s(%d)\n", \
                        __PRETTY_FUNCTION__, __LINE__, #f, status2str(status), \
                        (int)status); \
                fflush(0); \
                if (s == CRIT) exit(2); \
            } \
            clean(); \
            return FAIL; \
        } \
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
        CASE(dnnl_s32); \
        CASE(dnnl_s8); \
        CASE(dnnl_u8); \
        default: assert(!"bad data_type"); \
    }

inline size_t sizeof_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: return sizeof(typename prec_traits<dt>::type);

    CASE_ALL(dt);

#undef CASE
    return 0;
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

#undef CASE_ALL

template <dnnl_data_type_t dt>
inline float saturate(float val) {
    return MAX2((float)dnnl::impl::nstl::numeric_limits<
                        typename prec_traits<dt>::type>::lowest(),
            MIN2((float)dnnl::impl::nstl::numeric_limits<
                         typename prec_traits<dt>::type>::max(),
                    mxcsr_round(val)));
}

inline float maybe_saturate(dnnl_data_type_t dt, float value) {
    if (dt == dnnl_s32 || dt == dnnl_s8 || dt == dnnl_u8) {
        switch (dt) {
#define CASE(dt) \
    case dt: { \
        return saturate<dt>(value); \
    }
            CASE(dnnl_s32);
            CASE(dnnl_s8);
            CASE(dnnl_u8);
#undef CASE
            default: assert(!"bad data_type");
        }
        return 0;
    }
    return value;
}

/* simplification */
extern dnnl_engine_kind_t engine_tgt_kind;

extern dnnl_engine_t engine_tgt;
extern dnnl_stream_t stream_tgt;

/* for fast-ref-gpu support */
extern dnnl_engine_t engine_ref;

inline int init() {
    if (!engine_tgt) {
        DNN_SAFE(dnnl_engine_create(&engine_tgt, engine_tgt_kind, 0), CRIT);
        DNN_SAFE(dnnl_stream_create(
                         &stream_tgt, engine_tgt, dnnl_stream_default_flags),
                CRIT);
    }
    if (!engine_ref) {
        DNN_SAFE(dnnl_engine_create(&engine_ref, dnnl_cpu, 0), CRIT);
    }

    return OK;
}

inline int finalize() {
    DNN_SAFE(dnnl_stream_destroy(stream_tgt), CRIT);
    DNN_SAFE(dnnl_engine_destroy(engine_tgt), CRIT);
    DNN_SAFE(dnnl_engine_destroy(engine_ref), CRIT);
    return OK;
}

inline const char *query_impl_info(const_dnnl_primitive_desc_t pd) {
    const char *str;
    dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &str);
    return str;
}

struct dnn_mem_t;
struct attr_bundle_t;

struct args_t {
    args_t &set(int arg, const dnn_mem_t &mem);
    void clear() { args_.clear(); }

    int size() const { return (int)args_.size(); }

    int arg(int index) const { return args_[index].first; }
    const dnn_mem_t &dnn_mem(int index) const { return *args_[index].second; }

private:
    std::vector<std::pair<int, const dnn_mem_t *>> args_;
};

dnnl_status_t execute_and_wait(
        dnnl_primitive_t prim, dnnl_stream_t stream, const args_t &args);

int measure_perf(benchdnn_timer_t &t, dnnl_primitive_t prim, args_t &args);

void maybe_prepare_runtime_scales(dnn_mem_t &scales_m, const attr_t &attr,
        int64_t scale_cnt, const float *scales, dnnl_engine_t engine);
void maybe_prepare_runtime_scales(dnn_mem_t &scales_m,
        const attr_bundle_t &attr_bundle, dnnl_engine_t engine);

void maybe_prepare_runtime_zero_points(dnn_mem_t &zero_points_m,
        const attr_t &attr, int arg, dnnl_engine_t engine);

#endif

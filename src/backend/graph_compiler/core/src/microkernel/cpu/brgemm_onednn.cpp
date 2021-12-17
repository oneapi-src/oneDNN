/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#include <cstring>
#include <dnnl.h>
#include <iostream>
#include <math.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include "microkernel.hpp"
#include <compiler/codegen/x86simd/vec_u32x4.hpp>
#include <compiler/ir/sc_data_type.hpp>
#include <cpu/x64/amx_tile_configure.hpp>
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/brgemm/brgemm_types.hpp>
#include <runtime/config.hpp>
#include <runtime/context.hpp>
#include <runtime/thread_locals.hpp>
#include <unordered_map>
#include <util/hash_utils.hpp>

#ifdef SC_KERNEL_PROFILE
#include <atomic>
#include <chrono>
extern std::atomic<uint64_t> mkernel_init;
extern std::atomic<uint64_t> mkernel_exec;
#endif

using namespace dnnl::impl::cpu::x64;
typedef sc::sc_data_etype sc_dtype;

static dnnl_data_type_t convert_dnnl_dtype(int dtype) {
    switch (sc_dtype(dtype)) {
        case sc_dtype::F32: return dnnl_f32;
        case sc_dtype::S32: return dnnl_s32;
        case sc_dtype::BF16: return dnnl_bf16;
        case sc_dtype::S8: return dnnl_s8;
        case sc_dtype::U8: return dnnl_u8;
        default:
            throw std::runtime_error(
                    "convert_dnnl_dtype error, currently only support datatype "
                    "f32/s32/bf16/s8/u8");
    }
}

static size_t get_dtype_sizeof(int dtype) {
    switch (sc_dtype(dtype)) {
        case sc_dtype::F32: return sizeof(float);
        case sc_dtype::S32: return sizeof(int32_t);
        case sc_dtype::BF16: return sizeof(uint16_t);
        case sc_dtype::S8: return sizeof(int8_t);
        case sc_dtype::U8: return sizeof(uint8_t);
        default:
            throw std::runtime_error(
                    "Get dtype size error, currently only support datatype "
                    "f32/s32/bf16/s8/u8");
    }
}

struct alignas(64) brg_arg_t {
    float alpha;
    float beta;
    int LDA;
    int LDB;
    int LDC;
    int M;
    int N;
    int K;
    int stride_a;
    int stride_b;
    brgemm_batch_kind_t brg_type;
    int dtypeA;
    int dtypeB;
    int pad1 = 0;
    int pad2 = 0;
    int pad3 = 0;

    brg_arg_t(float alpha, float beta, int LDA, int LDB, int LDC, int M, int N,
            int K, int stride_a, int stride_b, brgemm_batch_kind_t brg_type,
            int dtypeA, int dtypeB)
        : alpha(alpha)
        , beta(beta)
        , LDA(LDA)
        , LDB(LDB)
        , LDC(LDC)
        , M(M)
        , N(N)
        , K(K)
        , stride_a(stride_a)
        , stride_b(stride_b)
        , brg_type(brg_type)
        , dtypeA(dtypeA)
        , dtypeB(dtypeB) {}

#define LOAD(name, ptr) \
    vec_u32x4 name##1 \
            = vec_u32x4::load(reinterpret_cast<const uint32_t *>(ptr)); \
    vec_u32x4 name##2 \
            = vec_u32x4::load(reinterpret_cast<const uint32_t *>(ptr) + 4); \
    vec_u32x4 name##3 = vec_u32x4::load( \
            reinterpret_cast<const uint32_t *>(ptr) + 4 * 2); \
    vec_u32x4 name##4 = vec_u32x4::load( \
            reinterpret_cast<const uint32_t *>(ptr) + 4 * 3);

    bool operator==(const brg_arg_t &v) const {
        return alpha == v.alpha && beta == v.beta && LDA == v.LDA
                && LDB == v.LDB && LDC == v.LDC && M == v.M && N == v.N
                && K == v.K && stride_a == v.stride_a && stride_b == v.stride_b
                && brg_type == v.brg_type && dtypeA == v.dtypeA
                && dtypeB == v.dtypeB;
    }

    size_t get_hash() const {
        static_assert(sizeof(brg_arg_t) == 16 * 4,
                "expecting 64-byte size for brg_arg");
        LOAD(v, this);

        v1 = v1 ^ (_mm_srli_si128(v2.v, 3));
        v1 = v2 ^ (_mm_srli_si128(v1.v, 2));
        v3 = v3 ^ (_mm_srli_si128(v4.v, 3));
        v3 = v4 ^ (_mm_srli_si128(v3.v, 2));

        v1 = v1 ^ v3;
        size_t ret = 0;
        for (int i = 0; i < 2; i++) {
            ret ^= reinterpret_cast<uint64_t *>(v1.raw)[i];
        }
        return ret;
    }
};

namespace std {
template <>
struct hash<brg_arg_t> {
    std::size_t operator()(const brg_arg_t &k) const { return k.get_hash(); }
};
} // namespace std

struct brg_arg_ptr_hash_t {
    std::size_t operator()(const brg_arg_t *k) const { return k->get_hash(); }
};

struct brg_arg_ptr_eq_to_t {
    bool operator()(const brg_arg_t *k, const brg_arg_t *k2) const {
        return *k == *k2;
    }
};

struct brgemm_kernel_info {
    alignas(64) char palette_[64];
    brgemm_kernel_t *brg_kernel_;
    bool is_amx_;
    ~brgemm_kernel_info() {
        if (brg_kernel_) {
            brgemm_kernel_destroy(brg_kernel_);
            brg_kernel_ = nullptr;
        }
    }
};

struct brg_desc_safe_t {
    brgemm_kernel_info *get(const brg_arg_t &arg) {
        auto found_kernel = brg_desc_vec_local_.find(&arg);
        if (found_kernel != brg_desc_vec_local_.end()) {
            return found_kernel->second;
        }
        return nullptr;
    }

    brgemm_kernel_info *getInstance(const brg_arg_t &arg) {
        brgemm_kernel_info *found_kernel = get(arg);
        // check if the brg_arg is in thread local cache (lock free)
        if (found_kernel) { return found_kernel; }
        // if the brg_arg is not found in thread local cache
        std::lock_guard<std::mutex> guard(lock_);
        auto itr = brg_desc_vec_.find(arg);
        if (itr != brg_desc_vec_.end()) {
            // double check if it is global kernel cache. If so, update the
            // thread local cache and return
            brg_desc_vec_local_.insert(
                    std::make_pair(&itr->first, &itr->second));
            return &itr->second;
        }
        // If we go here, the kernel is not yet created.
        brgemm_t desc;
        brgemm_strides_t stride_info = {arg.stride_a, arg.stride_b};
        auto dnnl_dtypeA = convert_dnnl_dtype(arg.dtypeA);
        auto dnnl_dtypeB = convert_dnnl_dtype(arg.dtypeB);
        size_t dtype_size = get_dtype_sizeof(arg.dtypeA);
        // todo: this type assignment is caused by lack of tail processing
        // in oneDNN (mkl-dnn/src/cpu/x64/brgemm/brgemm.cpp:305)
        auto choose_isa_type = [&]() {
            if (dnnl_dtypeA != dnnl_f32 && (arg.K < (4 / (int)dtype_size)))
                return avx512_core_vnni;
            int rd_block = dnnl_dtypeA == dnnl_bf16
                    ? 32
                    : (dnnl_dtypeA == dnnl_s8 || dnnl_dtypeA == dnnl_u8) ? 64
                                                                         : -1;
            // when no need for amx:
            if (rd_block == -1) { return isa_any; }
            int rdb = arg.K / rd_block;
            int rdb_tail = arg.K % rd_block;
            int dtype_block = rd_block == 32 ? 2 : 4;
            // if somehow invalid config for amx was generated anyway, make sure
            // it runs on vnni, which has less constraints
            if (rdb > 0 && rdb_tail > 0) { return avx512_core_vnni; }
            if (rdb_tail % dtype_block) { return avx512_core_vnni; }
            return isa_any;
        };
        cpu_isa_t isa_type = choose_isa_type();

        auto status = brgemm_desc_init(&desc, isa_type, arg.brg_type,
                dnnl_dtypeA, dnnl_dtypeB, false, false, brgemm_row_major,
                arg.alpha, arg.beta, arg.LDA, arg.LDB, arg.LDC, arg.M, arg.N,
                arg.K, &stride_info);
        assert(status == dnnl::impl::status::success);

        // create an entry in kernel cache
        auto new_itr = brg_desc_vec_.insert(
                std::make_pair(arg, brgemm_kernel_info()));
        // check that the insertion happens
        assert(new_itr.second);
        found_kernel = &new_itr.first->second;
        // insert the kernel to thread local cache
        brg_desc_vec_local_.insert(
                std::make_pair(&new_itr.first->first, found_kernel));

        found_kernel->is_amx_ = false;
        status = brgemm_init_tiles(desc, found_kernel->palette_);
        if (status == dnnl::impl::status::success) {
            amx_tile_configure(found_kernel->palette_);
            found_kernel->is_amx_ = true;
        }
        // TODO(xxx): Add those attributes into our brgemm interface.
        // brgemm_attr_t brgattr;
        // brgattr.max_top_vpad = 0;
        // brgattr.max_bottom_vpad = 0;
        // brgemm_desc_set_attr(&desc, brgattr);
        status = brgemm_kernel_create(&found_kernel->brg_kernel_, desc);
        assert(status == dnnl::impl::status::success);

        return found_kernel;
    }

    std::mutex lock_;
    // the table of brgemm argument => brgemm kernel map. It is shared by
    // threads of the same process
    std::unordered_map<brg_arg_t, brgemm_kernel_info> brg_desc_vec_;

    using thread_local_cache = std::unordered_map<const brg_arg_t *,
            brgemm_kernel_info *, brg_arg_ptr_hash_t, brg_arg_ptr_eq_to_t>;
    // the thread local cache of brgemm kernel. The cached key-values are
    // pointers to the key-values in the map above
    static thread_local thread_local_cache brg_desc_vec_local_;
};

static brg_desc_safe_t g_brg_desc_s;
thread_local brg_desc_safe_t::thread_local_cache
        brg_desc_safe_t::brg_desc_vec_local_;

namespace sc {
namespace runtime {

amx_buffer_t::~amx_buffer_t() {
    release();
}
void amx_buffer_t::reset(sc::runtime::stream_t *stream) {
    if (!stream) stream = sc::runtime::get_default_stream();
    stream_ = stream;
    ptr_ = stream->vtable_->persistent_alloc(stream, 1024);
}
void amx_buffer_t::release() {
    if (ptr_) {
        assert(stream_);
        stream_->vtable_->persistent_dealloc(stream_, ptr_);
        ptr_ = nullptr;
        stream_ = nullptr;
    }
}
} // namespace runtime
} // namespace sc

static thread_local char *cur_palette = nullptr;
static void *get_amx_tile_buf(brgemm_kernel_info *brg_desc,
        sc::runtime::stream_t *stream, bool &amx_exclusive) {
    void *tmp_amx_tile_buf = nullptr;
    if (brg_desc->is_amx_) {
        amx_exclusive = sc::runtime_config_t::get().amx_exclusive_;
        if (!amx_exclusive || cur_palette != brg_desc->palette_) {
            amx_tile_configure(brg_desc->palette_);
            cur_palette = brg_desc->palette_;
            auto &amx_tile_buf = sc::runtime::tls_buffer.amx_buffer;
            if (!amx_tile_buf.ptr_) { amx_tile_buf.reset(stream); }
            tmp_amx_tile_buf = amx_tile_buf.ptr_;
        }
    }
    return tmp_amx_tile_buf;
}

extern "C" {
SC_API void *dnnl_brgemm_func(int M, int N, int K, int LDA, int LDB, int LDC,
        int stride_a, int stride_b, float beta, int dtypeA, int dtypeB) {
    float alpha = 1.0;
    return g_brg_desc_s.getInstance({alpha, beta, LDA, LDB, LDC, M, N, K,
            static_cast<int>(stride_a * get_dtype_sizeof(dtypeA)),
            static_cast<int>(stride_b * get_dtype_sizeof(dtypeB)), brgemm_strd,
            dtypeA, dtypeB});
}

SC_API void dnnl_brgemm_call(brgemm_kernel_info *brg_desc, const void *A,
        const void *B, void *C, int num, sc::runtime::stream_t *stream) {
    bool amx_exclusive = false;
    void *tmp_amx_tile_buf = get_amx_tile_buf(brg_desc, stream, amx_exclusive);
    brgemm_kernel_execute(brg_desc->brg_kernel_, num, (void **)A, (void **)B,
            nullptr, (void *)C, tmp_amx_tile_buf);
    if (!amx_exclusive && brg_desc->is_amx_) { amx_tile_release(); }
}

SC_API void *dnnl_brgemm_list_func(int M, int N, int K, int LDA, int LDB,
        int LDC, float beta, int dtypeA, int dtypeB) {
    float alpha = 1.0;
    if (M <= 0) { return nullptr; }
    return g_brg_desc_s.getInstance({alpha, beta, LDA, LDB, LDC, M, N, K, 0, 0,
            brgemm_addr, dtypeA, dtypeB});
}

SC_API void dnnl_brgemm_list_call(brgemm_kernel_info *brg_desc,
        const void **A_list, const void **B_list, void *C, int num,
        int stride_a, int stride_b, int len, int dtypeA, int dtypeB,
        sc::runtime::stream_t *stream) {
    const int batch_num = num * len;
#ifdef _MSC_VER
    brgemm_batch_element_t *batch
            = (brgemm_batch_element_t *)_malloca(batch_num);
#else
#if CLANGVERSION <= 3
    std::unique_ptr<brgemm_batch_element_t[]> batch_v(
            new brgemm_batch_element_t[batch_num]);
    brgemm_batch_element_t *batch = batch_v.get();
#else
    brgemm_batch_element_t batch[batch_num]; // NOLINT
#endif
#endif

    int sizeofA = get_dtype_sizeof(dtypeA);
    int sizeofB = get_dtype_sizeof(dtypeB);
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < num; ++j) {
            batch[i * num + j].ptr.A
                    = (((char **)A_list)[i] + (j * stride_a * sizeofA));
            batch[i * num + j].ptr.B
                    = (((char **)B_list)[i] + (j * stride_b * sizeofB));
        }
    }
    bool amx_exclusive = false;
    void *tmp_amx_tile_buf = get_amx_tile_buf(brg_desc, stream, amx_exclusive);
    brgemm_kernel_execute(brg_desc->brg_kernel_, batch_num, batch, (void *)C,
            tmp_amx_tile_buf);
#ifdef _MSC_VER
    _freea(batch);
#endif
    if (!amx_exclusive && brg_desc->is_amx_) { amx_tile_release(); }
}

#define BRGEMM_DTYPE_INIT(dtype) \
    if (LDC == N) { \
        memset(C, (dtype)value, M *N *get_dtype_sizeof(dtypeC)); \
    } else { \
        for (int i = 0; i < M; ++i) { \
            for (int j = 0; j < N; ++j) { \
                ((dtype *)C)[i + LDC + j] = (dtype)value; \
            } \
        } \
    }

SC_API int dnnl_brgemm_init(
        void *C, int M, int N, int LDC, int dtypeC, float value) {
    if (get_dtype_sizeof(dtypeC) == 1) {
        BRGEMM_DTYPE_INIT(uint8_t);
    } else {
        BRGEMM_DTYPE_INIT(int32_t);
    }
    return 0;
}

SC_API int dnnl_brgemm_init_update(const void *A, const void *B, void *C,
        int num, int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b, int dtypeA, int dtypeB, sc::runtime::stream_t *stream) {
#ifdef SC_KERNEL_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    float alpha = 1.0, beta = 0.0;
    auto brg_desc = g_brg_desc_s.getInstance({alpha, beta, LDA, LDB, LDC, M, N,
            K, static_cast<int>(stride_a * get_dtype_sizeof(dtypeA)),
            static_cast<int>(stride_b * get_dtype_sizeof(dtypeB)), brgemm_strd,
            dtypeA, dtypeB});
#ifdef SC_KERNEL_PROFILE
    auto init_stop = std::chrono::high_resolution_clock::now();
#endif
    bool amx_exclusive = false;
    void *tmp_amx_tile_buf = get_amx_tile_buf(brg_desc, stream, amx_exclusive);
    brgemm_kernel_execute(brg_desc->brg_kernel_, num, (void **)A, (void **)B,
            nullptr, (void *)C, tmp_amx_tile_buf);
    if (!amx_exclusive && brg_desc->is_amx_) { amx_tile_release(); }

#ifdef SC_KERNEL_PROFILE
    auto exec_stop = std::chrono::high_resolution_clock::now();
    mkernel_init += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    init_stop - start)
                    .count());
    mkernel_exec += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    exec_stop - init_stop)
                    .count());
#endif
    return 0;
}

SC_API int dnnl_brgemm_update(const void *A, const void *B, void *C, int num,
        int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b, int dtypeA, int dtypeB, sc::runtime::stream_t *stream) {
#ifdef SC_KERNEL_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    float alpha = 1.0, beta = 1.0;
    auto brg_desc = g_brg_desc_s.getInstance({alpha, beta, LDA, LDB, LDC, M, N,
            K, static_cast<int>(stride_a * get_dtype_sizeof(dtypeA)),
            static_cast<int>(stride_b * get_dtype_sizeof(dtypeB)), brgemm_strd,
            dtypeA, dtypeB});
#ifdef SC_KERNEL_PROFILE
    auto init_stop = std::chrono::high_resolution_clock::now();
#endif
    bool amx_exclusive = false;
    void *tmp_amx_tile_buf = get_amx_tile_buf(brg_desc, stream, amx_exclusive);
    brgemm_kernel_execute(brg_desc->brg_kernel_, num, (void **)A, (void **)B,
            nullptr, (void *)C, tmp_amx_tile_buf);
    if (!amx_exclusive && brg_desc->is_amx_) { amx_tile_release(); }

#ifdef SC_KERNEL_PROFILE
    auto exec_stop = std::chrono::high_resolution_clock::now();
    mkernel_init += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    init_stop - start)
                    .count());
    mkernel_exec += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    exec_stop - init_stop)
                    .count());
#endif
    return 0;
}

SC_API int dnnl_brgemm_list_update(const void **A_list, const void **B_list,
        void *C, int num, int M, int N, int K, int LDA, int LDB, int LDC,
        int stride_a, int stride_b, int len, int dtypeA, int dtypeB,
        sc::runtime::stream_t *stream) {
#ifdef SC_KERNEL_PROFILE
    auto start = std::chrono::high_resolution_clock::now();
#endif
    float alpha = 1.0, beta = 1.0;
    const int batch_num = num * len;
#ifdef _MSC_VER
    brgemm_batch_element_t *batch
            = (brgemm_batch_element_t *)_malloca(batch_num);
#else
#if CLANGVERSION <= 3
    std::unique_ptr<brgemm_batch_element_t[]> batch_v(
            new brgemm_batch_element_t[batch_num]);
    brgemm_batch_element_t *batch = batch_v.get();
#else
    brgemm_batch_element_t batch[batch_num]; // NOLINT
#endif
#endif
    int sizeofA = get_dtype_sizeof(dtypeA);
    int sizeofB = get_dtype_sizeof(dtypeB);
    for (int i = 0; i < len; ++i) {
        for (int j = 0; j < num; ++j) {
            batch[i * num + j].ptr.A
                    = (((char **)A_list)[i] + (j * stride_a * sizeofA));
            batch[i * num + j].ptr.B
                    = (((char **)B_list)[i] + (j * stride_b * sizeofB));
        }
    }
    auto brg_desc = g_brg_desc_s.getInstance({alpha, beta, LDA, LDB, LDC, M, N,
            K, 0, 0, brgemm_addr, dtypeA, dtypeB});
#ifdef SC_KERNEL_PROFILE
    auto init_stop = std::chrono::high_resolution_clock::now();
#endif
    bool amx_exclusive = false;
    void *tmp_amx_tile_buf = get_amx_tile_buf(brg_desc, stream, amx_exclusive);
    brgemm_kernel_execute(brg_desc->brg_kernel_, batch_num, batch, (void *)C,
            tmp_amx_tile_buf);
#ifdef _MSC_VER
    _freea(batch);
#endif
    if (!amx_exclusive && brg_desc->is_amx_) { amx_tile_release(); }

#ifdef SC_KERNEL_PROFILE
    auto exec_stop = std::chrono::high_resolution_clock::now();
    mkernel_init += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    init_stop - start)
                    .count());
    mkernel_exec += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                    exec_stop - init_stop)
                    .count());
#endif
    return 0;
}
}

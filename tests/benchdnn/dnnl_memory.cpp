/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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
#include <cctype>
#include <memory>
#include <numeric>
#include <string>

#include "oneapi/dnnl/dnnl.h"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_ocl.hpp"
#include "src/xpu/ocl/usm_utils.hpp"
#endif

#include "tests/test_thread.hpp"

#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/cold_cache.hpp"
#include "utils/dnnl_query.hpp"
#include "utils/parallel.hpp"

extern "C" dnnl_status_t dnnl_memory_desc_create_with_string_tag(
        dnnl_memory_desc_t *, int, const dnnl_dims_t, dnnl_data_type_t,
        const char *);

extern "C" dnnl_status_t dnnl_memory_desc_set_data_type(
        dnnl_memory_desc_t memory_desc, dnnl_data_type_t data_type);

dnn_mem_t::dnn_mem_t(const_dnnl_memory_desc_t md, dnnl_engine_t engine,
        const handle_info_t &handle_info) {
    if (query_md_ndims(md) > 0) {
        auto status = dnnl_memory_desc_clone(&md_, md);
        (void)status;
        assert(status == dnnl_success);
        active_ = (initialize(engine, handle_info) == OK);
    }
}

dnn_mem_t::dnn_mem_t(const_dnnl_memory_desc_t md, dnnl_data_type_t dt,
        const std::string &tag, dnnl_engine_t engine) {
    const int ndims = query_md_ndims(md);
    if (ndims > 0) {
        auto md_wrapper = dnn_mem_t::init_md(ndims, query_md_dims(md), dt, tag);
        md_ = md_wrapper.release();
        active_ = (initialize(engine) == OK);
    }
}

dnn_mem_t::dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
        const std::string &tag, dnnl_engine_t engine) {
    if (ndims > 0) {
        auto md_wrapper = dnn_mem_t::init_md(ndims, dims, dt, tag);
        md_ = md_wrapper.release();
        active_ = (initialize(engine) == OK);
    }
}

dnn_mem_t::dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
        const dnnl_dims_t strides, dnnl_engine_t engine) {
    if (ndims > 0) {
        auto status = dnnl_memory_desc_create_with_strides(
                &md_, ndims, dims, dt, strides);
        (void)status;
        assert(status == dnnl_success);
        active_ = (initialize(engine) == OK);
    }
}

dnn_mem_t::dnn_mem_t(const dnn_mem_t &rhs, dnnl_data_type_t dt,
        const std::string &tag, dnnl_engine_t engine)
    : dnn_mem_t(rhs.md_, dt, tag, engine) {
    if (active_) {
        int status = reorder(rhs);
        if (status != OK) {
            BENCHDNN_PRINT(
                    0, "%s\n", "Error: reorder in memory constructor failed.");
        }
    }
}

int execute_reorder(const dnn_mem_t &src, dnn_mem_t &dst,
        const_dnnl_primitive_attr_t attr) {
    std::shared_ptr<const dnn_mem_t> r_src(&src, [](const dnn_mem_t *) {});
    std::shared_ptr<dnn_mem_t> r_dst(&dst, [](dnn_mem_t *) {});

    dnnl_primitive_desc_t r_pd_ {};
    dnnl_primitive_t prim_ {};

    // Optimization to reduce testing time for GPU.
    //
    // For CPU <-> GPU reorders, the library creates GPU-side kernels.
    // Benchdnn heavily relies on reorders and this greatly increases execution
    // time because of big overhead on building OpenCL kernels.
    //
    // First, try to create CPU reorder for the requested GPU reorder. If
    // succeeded, then create CPU memory object wrapping mapped pointers of
    // source and destination and execute CPU reorder. If CPU reorder can't be
    // create, then just execute a regular GPU reorder.
#if ((DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL) \
        || (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL)) \
        && DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    const auto &cpu_engine = get_cpu_engine();
    if (src.engine_kind() == dnnl_gpu || dst.engine_kind() == dnnl_gpu) {

        dnnl_status_t status = dnnl_reorder_primitive_desc_create(
                &r_pd_, src.md_, cpu_engine, dst.md_, cpu_engine, attr);
        if (status == dnnl_success) {
            // Create CPU memory objects wrapping mapped pointers of source and
            // destination
            r_src = std::make_shared<dnn_mem_t>(dnn_mem_t::create_from_host_ptr(
                    src.md_, cpu_engine, (void *)src));
            r_dst = std::make_shared<dnn_mem_t>(dnn_mem_t::create_from_host_ptr(
                    dst.md_, cpu_engine, (void *)dst));
        }
    }
#endif

    if (!r_pd_) {
        DNN_SAFE(dnnl_reorder_primitive_desc_create(&r_pd_, src.md_,
                         src.engine(), dst.md_, dst.engine(), attr),
                WARN);
    }
    auto r_pd = make_benchdnn_dnnl_wrapper(r_pd_);
    const auto &scratchpad_md = query_md(r_pd, DNNL_ARG_SCRATCHPAD);
    const auto &scratchpad_engine
            = dst.engine_kind() == dnnl_gpu ? dst.engine() : src.engine();
    dnn_mem_t scratchpad(scratchpad_md, scratchpad_engine);

    DNN_SAFE(dnnl_primitive_create(&prim_, r_pd), CRIT);
    auto prim = make_benchdnn_dnnl_wrapper(prim_);

    args_t args;
    args.set(DNNL_ARG_FROM, *r_src);
    args.set(DNNL_ARG_TO, *r_dst);
    args.set(DNNL_ARG_SCRATCHPAD, scratchpad);

    return execute_and_wait(prim, args);
}

// `swap_dt` changes `this` data type which may be needed for
// different sum data type or fpmath mode specified.
int dnn_mem_t::reorder(const dnn_mem_t &rhs, const_dnnl_primitive_attr_t attr,
        dnnl_data_type_t swap_dt) {
    if (this == &rhs) return OK;

    // When `rhs` object is empty, it's illigal to execute a reorder over it.
    // Do nothing, return a good status. Keep here to avoid guarding externally.
    if (query_md_ndims(rhs.md_) == 0) return OK;

    // Assumption is `no_ref_memory` assigned values at construction, and no
    // actual reorder needed. This check is to avoid extra code outside of
    // reorder interface.
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const bool do_swap_dt = swap_dt != dnnl_data_type_undef;
    dnnl_data_type_t orig_dt = this->dt();
    if (do_swap_dt) this->set_dt(swap_dt);
    auto status = execute_reorder(rhs, *this, attr);
    if (do_swap_dt) this->set_dt(orig_dt);
    return status;
}

size_t dnn_mem_t::size() const {
    return dnnl_memory_desc_get_size(md_);
}

size_t dnn_mem_t::sizeof_dt() const {
    return dnnl_data_type_size(dt());
}

float dnn_mem_t::get_elem(int64_t idx, int buffer_index) const {
    void *data = get_mapped_pointer<void>(buffer_index);
    float elem = 0.0;

    switch (dt(buffer_index)) {
        case dnnl_s8: elem = static_cast<int8_t *>(data)[idx]; break;
        case dnnl_u8: elem = static_cast<uint8_t *>(data)[idx]; break;
        case dnnl_s32: elem = static_cast<int32_t *>(data)[idx]; break;
        case dnnl_f32: elem = static_cast<float *>(data)[idx]; break;
        case dnnl_f64: elem = static_cast<double *>(data)[idx]; break;
        case dnnl_f16:
            elem = static_cast<dnnl::impl::float16_t *>(data)[idx];
            break;
        case dnnl_bf16:
            elem = static_cast<dnnl::impl::bfloat16_t *>(data)[idx];
            break;
        case dnnl_e8m0:
            elem = static_cast<dnnl::impl::float8_e8m0_t *>(data)[idx];
            break;
        case dnnl_f8_e5m2:
            elem = static_cast<dnnl::impl::float8_e5m2_t *>(data)[idx];
            break;
        case dnnl_f8_e4m3:
            elem = static_cast<dnnl::impl::float8_e4m3_t *>(data)[idx];
            break;
        case dnnl_s4: {
            dnnl::impl::nibble2_t nibble_pair(
                    reinterpret_cast<uint8_t *>(data)[idx / 2]);
            elem = dnnl::impl::int4_t(nibble_pair.get(idx % 2));
            break;
        }
        case dnnl_u4: {
            dnnl::impl::nibble2_t nibble_pair(
                    reinterpret_cast<uint8_t *>(data)[idx / 2]);
            elem = dnnl::impl::uint4_t(nibble_pair.get(idx % 2));
            break;
        }
        case dnnl_f4_e2m1: {
            dnnl::impl::nibble2_t nibble_pair(
                    reinterpret_cast<uint8_t *>(data)[idx / 2]);
            elem = dnnl::impl::float4_e2m1_t(nibble_pair.get(idx % 2));
            break;
        }
        case dnnl_f4_e3m0: {
            dnnl::impl::nibble2_t nibble_pair(
                    reinterpret_cast<uint8_t *>(data)[idx / 2]);
            elem = dnnl::impl::float4_e3m0_t(nibble_pair.get(idx % 2));
            break;
        }
        default: assert(!"bad data type");
    }
    return elem;
}

void dnn_mem_t::set_elem(int64_t idx, float value, int buffer_index) const {
    void *data = get_mapped_pointer<void>(buffer_index);

    switch (dt(buffer_index)) {
        case dnnl_s8: ((int8_t *)data)[idx] = value; break;
        case dnnl_u8: ((uint8_t *)data)[idx] = value; break;
        case dnnl_s32: ((int32_t *)data)[idx] = value; break;
        case dnnl_f32: ((float *)data)[idx] = value; break;
        case dnnl_f64: ((double *)data)[idx] = value; break;
        case dnnl_f16: ((dnnl::impl::float16_t *)data)[idx] = value; break;
        case dnnl_bf16: ((dnnl::impl::bfloat16_t *)data)[idx] = value; break;
        case dnnl_e8m0: ((dnnl::impl::float8_e8m0_t *)data)[idx] = value; break;
        case dnnl_f8_e5m2:
            ((dnnl::impl::float8_e5m2_t *)data)[idx] = value;
            break;
        case dnnl_f8_e4m3:
            ((dnnl::impl::float8_e4m3_t *)data)[idx] = value;
            break;
        case dnnl_s4: {
            auto dst_val = ((dnnl::impl::nibble2_t *)data)[idx / 2];
            dst_val.set(dnnl::impl::int4_t(value).raw_bits_, idx % 2);
            ((dnnl::impl::nibble2_t *)data)[idx / 2] = dst_val;
            break;
        }
        case dnnl_u4: {
            auto dst_val = ((dnnl::impl::nibble2_t *)data)[idx / 2];
            dst_val.set(dnnl::impl::uint4_t(value).raw_bits_, idx % 2);
            ((dnnl::impl::nibble2_t *)data)[idx / 2] = dst_val;
            break;
        }
        case dnnl_f4_e2m1: {
            auto dst_val = ((dnnl::impl::nibble2_t *)data)[idx / 2];
            dst_val.set(dnnl::impl::float4_e2m1_t(value).raw_bits_, idx % 2);
            ((dnnl::impl::nibble2_t *)data)[idx / 2] = dst_val;
            break;
        }
        case dnnl_f4_e3m0: {
            auto dst_val = ((dnnl::impl::nibble2_t *)data)[idx / 2];
            dst_val.set(dnnl::impl::float4_e3m0_t(value).raw_bits_, idx % 2);
            ((dnnl::impl::nibble2_t *)data)[idx / 2] = dst_val;
            break;
        }
        default: assert(!"bad data type");
    }
}

// Returns an updated logical index based on input `logical_index` and
// `dims_mask`.
// `logical_idx` is a generally composed index where dims[ndims - 1] is most
// dense and dims[0] is least dense dimensions, as tensor in `abx` format.
// `dims_mask` represents dimensions to keep or remove. A value is composed of
// number of bits equal to `ndims` and value of `1` indicates the dimension to
// present in final calculation. E.g., mask=0 means a single value, and mask=2,
// or 1 << 1, means to keep dims[1] only.
// `ndims` allows to reduce the number of dimensions from innermost direction.
// It is used to find an index in batched dimensions. E.g., if ndims() returns
// `4`, but `ndims` argument is passed as `2`, it will count only first two
// dimensions instead of all 4.
// `groups` is an extension of scales when their dimensions are different.
// In this case, it's required to adjust dimensions values according to group
// values to properly compute stride in a smaller tensor. The definition is
// aligned with the API and expects groups of size 2 or empty.
// When `ndims()/ndims` are bigger than 2, groups are applied only to two most
// dense dimensions, which is aligned with 2D matmul definition. E.g., dims=2x6,
// mask=3, ndims=2 and groups=1x3; in this case indices 0,1,2 will return 0 as
// those indices represent the first group. 3,4,5 will return 1. Changing the
// example like dims=6x2, mask=3, ndims=2 and groups=3x1 will change ouput as
// well due to groups are dense not over the last dimension. The match will look
// like this: 0->0, 1->1, 2->0, 3->1, ..., 6->2, ..., 11->4.
//
// Helps to find an index of smaller tensor in bigger one, e.g., scales. Scales
// are usually represented by a 1D array and have a mask indicating dims to be
// applied for. It coincides with `dims_mask`.
// Another usage is to identify an index for operations that support broadcast
// over different dimensions, e.g., matmul batch dimensions, or binary
// primitive.
//
// Example: there's a tensor 3x4 and scales with mask=2, which means to apply
// scales over the dimension of 4. Passing indices from 0 to 3 will return
// values from 0 to 3 correspondently. However, passing `logical_idx` of 4 will
// return 0, as 4 is a logical representation of point 1x0.
int64_t dnn_mem_t::get_idx(int64_t logical_idx, int dims_mask, const int ndims,
        const dims_t &groups) const {
    if (dims_mask == 0) return 0;

    const auto &dims = this->dims();
    int64_t stride = 1;
    int64_t offset = 0;

    assert(groups.empty() || groups.size() == 2);
    assert(groups.size() <= static_cast<size_t>(ndims));
    dims_t groups_ext(ndims, 1);
    if (!groups.empty()) {
        groups_ext[ndims - 2] = groups[0];
        groups_ext[ndims - 1] = groups[1];
    }

    for (int i = 0; i < ndims; ++i) {
        int d = ndims - 1 - i;
        auto pos = logical_idx % dims[d];
        logical_idx /= dims[d];
        if (dims_mask & (1 << d)) {
            offset += (pos / groups_ext[d]) * stride;
            stride *= (dims[d] / groups_ext[d]);
        }
    }

    return offset;
}

// Creates a memory object from the underlying buffer of an existing memory
// object `mem`. The size of `mem` must not be less than the size of `md`.
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL || defined(DNNL_WITH_SYCL)
static int init_memory(
        dnnl_memory_t *ret, const dnnl_memory_desc_t &md, dnnl_memory_t mem) {

    dnnl_engine_t engine;
    DNN_SAFE(dnnl_memory_get_engine(mem, &engine), CRIT);

    bool is_sycl = is_sycl_engine(engine);
    bool is_opencl = is_opencl_engine(engine);

    *ret = nullptr;

    const int nhandles = query_md_num_handles(md);
    std::vector<void *> handles(nhandles);
    for (int i = 0; i < nhandles; i++)
        DNN_SAFE(dnnl_memory_get_data_handle_v2(mem, &handles[i], i), CRIT);

    if (is_opencl) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        dnnl_ocl_interop_memory_kind_t mem_kind;
        DNN_SAFE(dnnl_ocl_interop_memory_get_memory_kind(mem, &mem_kind), CRIT);
        DNN_SAFE(dnnl_ocl_interop_memory_create_v2(ret, md, engine, mem_kind,
                         (int)handles.size(), handles.data()),
                CRIT);
#endif
    } else if (is_sycl) {
#ifdef DNNL_WITH_SYCL
        dnnl_sycl_interop_memory_kind_t mem_kind;
        DNN_SAFE(
                dnnl_sycl_interop_memory_get_memory_kind(mem, &mem_kind), CRIT);
        DNN_SAFE(dnnl_sycl_interop_memory_create_v2(ret, md, engine, mem_kind,
                         (int)handles.size(), handles.data()),
                CRIT);
#endif
    }

    // Memory must be initialized at this point in some of the branches above.
    if (!*ret) assert(!"not expected");

    return OK;
}
#endif

void dnn_mem_t::map() const {
    assert(!is_mapped_ && "memory is already mapped");
    is_mapped_ = true;

    if (!m_) return;
    auto mem = m_padded_ ? m_padded_ : m_;
    const int nhandles = query_md_num_handles(md_);
    mapped_ptrs_.resize(nhandles);
    for (int i = 0; i < nhandles; i++) {
        auto st = dnnl_memory_map_data_v2(mem, &mapped_ptrs_[i], i);
        if (st != dnnl_success) {
            for (int j = 0; j < i; j++)
                DNN_SAFE_V(dnnl_memory_unmap_data_v2(mem, mapped_ptrs_[i], i));
            DNN_SAFE_V(st);
        }
    }
}

void dnn_mem_t::unmap() const {
    assert(is_mapped_ && "memory is not mapped");
    is_mapped_ = false;

    if (!m_) return;
    auto mem = m_padded_ ? m_padded_ : m_;
    const int nhandles = query_md_num_handles(md_);
    for (int i = 0; i < nhandles; i++) {
        DNN_SAFE_V(dnnl_memory_unmap_data_v2(mem, mapped_ptrs_[i], i));
        mapped_ptrs_[i] = nullptr;
    }
}

void dnn_mem_t::memset(int value, size_t size) const {
    bool is_opencl = is_opencl_engine(engine_);
    bool is_sycl = is_sycl_engine(engine_);
    auto mem = m_padded_ ? m_padded_ : m_;
    void *mem_handle;
    DNN_SAFE_V(dnnl_memory_get_data_handle(mem, &mem_handle));
    if (is_opencl) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        stream_t stream(engine_);
        switch (memory_kind) {
            case memory_kind_ext_t::buffer: {
                auto buf = static_cast<cl_mem>(mem_handle);
                cl_command_queue queue;
                DNN_SAFE_V(dnnl_ocl_interop_stream_get_command_queue(
                        stream, &queue));
                cl_int err = clEnqueueFillBuffer(queue, buf, &value,
                        sizeof(uint8_t), 0, size, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) SAFE_V(FAIL);
                DNN_SAFE_V(dnnl_stream_wait(stream));
                return;
            }
            case memory_kind_ext_t::usm:
            case memory_kind_ext_t::usm_device:
            case memory_kind_ext_t::usm_shared: {
                DNN_SAFE_V(dnnl::impl::xpu::ocl::usm::memset(
                        stream, mem_handle, value, size));
                DNN_SAFE_V(dnnl_stream_wait(stream));
                return;
            }
        }
#endif
    } else if (is_sycl) {
#ifdef DNNL_WITH_SYCL
        stream_t stream(engine_);
        void *queue_ptr;
        DNN_SAFE_V(dnnl_sycl_interop_stream_get_queue(stream, &queue_ptr));
        auto &queue = *static_cast<::sycl::queue *>(queue_ptr);
        switch (memory_kind) {
            case memory_kind_ext_t::buffer: {
                auto &buf = *static_cast<::sycl::buffer<uint8_t, 1> *>(
                        mem_handle);
                queue.submit([&](::sycl::handler &cgh) {
                    constexpr auto target_device = ::sycl::target::device;
                    ::sycl::accessor<uint8_t, 1, ::sycl::access::mode::write,
                            target_device>
                            acc(buf, cgh);
                    cgh.fill(acc, static_cast<uint8_t>(value));
                });
                DNN_SAFE_V(dnnl_stream_wait(stream));
                return;
            }
            case memory_kind_ext_t::usm:
            case memory_kind_ext_t::usm_device:
            case memory_kind_ext_t::usm_shared: {
                queue.submit([&](::sycl::handler &cgh) {
                    cgh.memset(mem_handle, value, size);
                });
                DNN_SAFE_V(dnnl_stream_wait(stream));
                return;
            }
        }
#endif
    }
    if (is_cpu(engine_)) {
        ::memset(mem_handle, value, size);
        return;
    }
    SAFE_V(FAIL);
}

dnn_mem_t dnn_mem_t::create_from_host_ptr(
        const dnnl_memory_desc_t &md, dnnl_engine_t engine, void *host_ptr) {
    return dnn_mem_t(md, engine, {true, host_ptr});
}

size_t dnn_mem_t::pad_memory_size(
        size_t sz, dnnl_engine_kind_t engine_kind, bool *was_padded) {
    if (was_padded) *was_padded = false;
    if (sz == 0 || !has_bench_mode_bit(mode_bit_t::corr)
            || engine_kind == dnnl_cpu)
        return sz;

    const size_t page_size = 4096;
    auto padded_sz = rnd_up(sz, page_size);
    if (was_padded) *was_padded = padded_sz != sz;
    return padded_sz;
}

dnnl_memory_desc_t dnn_mem_t::pad_memory_desc(const_dnnl_memory_desc_t md,
        dnnl_engine_kind_t engine_kind, bool *was_padded) {
    if (was_padded) *was_padded = false;
    // TODO: add padded memory descriptor support for sparse memory.
    if (query_md_format_kind(md) == dnnl_format_kind_sparse) return nullptr;
    size_t old_sz = dnnl_memory_desc_get_size(md);
    if (old_sz == 0 || !has_bench_mode_bit(mode_bit_t::corr)
            || engine_kind == dnnl_cpu)
        return nullptr;

    size_t sz = pad_memory_size(old_sz, engine_kind, was_padded);
    if (sz == old_sz) return nullptr;

    dnnl_memory_desc_t ret;
    dnnl_dims_t dims = {(dnnl_dim_t)sz};
    DNN_SAFE_V(
            dnnl_memory_desc_create_with_tag(&ret, 1, dims, dnnl_u8, dnnl_x));
    return ret;
}

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dnn_mem_t::init_md(int ndims,
        const dnnl_dims_t dims, dnnl_data_type_t data_type,
        const std::string &tag_, const dims_t &strides_) {
    dnnl_memory_desc_t md {};
    const bool use_strides = !strides_.empty();
    // Ignore tag_ in case strides_ are explicitly provided
    if (use_strides) {
        std::vector<dnnl_dim_t> strides(strides_);
        DNN_SAFE_V(dnnl_memory_desc_create_with_strides(
                &md, ndims, dims, data_type, strides.data()));
        return md;
    }

    auto tag = normalize_tag(tag_, ndims);
    if (tag == tag::undef || tag == tag::any || ndims == 0) {
        dnnl_format_tag_t enum_tag = (tag == tag::undef || ndims == 0)
                ? dnnl_format_tag_undef
                : dnnl_format_tag_any;
        DNN_SAFE_V(dnnl_memory_desc_create_with_tag(
                &md, ndims, dims, data_type, enum_tag));
        return md;
    }

    DNN_SAFE_V(dnnl_memory_desc_create_with_string_tag(
            &md, ndims, dims, data_type, tag.data()));

    return md;
}

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dnn_mem_t::init_csr_md(int ndims,
        const dnnl_dims_t dims, dnnl_data_type_t data_type, dnnl_dim_t nnz,
        dnnl_data_type_t indices_dt, dnnl_data_type_t pointers_dt) {
    dnnl_memory_desc_t md {};
    DNN_SAFE_V(dnnl_memory_desc_create_with_csr_encoding(
            &md, ndims, dims, data_type, nnz, indices_dt, pointers_dt));
    return md;
}

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dnn_mem_t::init_coo_md(int ndims,
        const dnnl_dims_t dims, dnnl_data_type_t data_type, dnnl_dim_t nnz,
        dnnl_data_type_t indices_dt) {
    dnnl_memory_desc_t md {};
    DNN_SAFE_V(dnnl_memory_desc_create_with_coo_encoding(
            &md, ndims, dims, data_type, nnz, indices_dt));
    return md;
}

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dnn_mem_t::init_sparse_packed_md(
        int ndims, const dnnl_dims_t dims, dnnl_data_type_t data_type,
        dnnl_dim_t nnz) {
    dnnl_memory_desc_t md {};
    DNN_SAFE_V(dnnl_memory_desc_create_with_packed_encoding(
            &md, ndims, dims, data_type, nnz));
    return md;
}

int dnn_mem_t::initialize_memory_create_sycl(const handle_info_t &handle_info) {
#ifdef DNNL_WITH_SYCL
    if (handle_info.is_host_ptr) {
        // Ignore memory_kind with host pointers and force USM.
        const int nhandles = query_md_num_handles(md_);
        std::vector<void *> handles(nhandles, handle_info.ptr);
        DNN_SAFE(dnnl_sycl_interop_memory_create_v2(&m_, md_, engine_,
                         dnnl_sycl_interop_usm, (int)handles.size(),
                         handles.data()),
                CRIT);
        return OK;
    }

    auto md_padded = pad_memory_desc(md_, engine_kind_, &is_canary_protected_);
    if (!md_padded) md_padded = md_;

    switch (memory_kind) {
        case memory_kind_ext_t::usm:
        case memory_kind_ext_t::buffer: {
            dnnl_sycl_interop_memory_kind_t mem_kind
                    = (memory_kind == memory_kind_ext_t::usm
                                    ? dnnl_sycl_interop_usm
                                    : dnnl_sycl_interop_buffer);
            const int nhandles = query_md_num_handles(md_);
            std::vector<void *> handles(nhandles, handle_info.ptr);
            DNN_SAFE(dnnl_sycl_interop_memory_create_v2(&m_padded_, md_padded,
                             engine_, mem_kind, (int)handles.size(),
                             handles.data()),
                    CRIT);
            SAFE(init_memory(&m_, md_, m_padded_), CRIT);
            break;
        }
        case memory_kind_ext_t::usm_device:
        case memory_kind_ext_t::usm_shared: {
            SAFE(handle_info.is_allocate() ? OK : FAIL, CRIT);
            is_data_owner_ = true;

            auto eng = dnnl::engine(engine_, true);
            auto dev = dnnl::sycl_interop::get_device(eng);
            auto ctx = dnnl::sycl_interop::get_context(eng);

            const int nhandles = query_md_num_handles(md_);
            for (int i = 0; i < nhandles; i++) {
                size_t sz = dnnl_memory_desc_get_size_v2(md_padded, i);
                if (memory_kind == memory_kind_ext_t::usm_device) {
                    data_.push_back(::sycl::malloc_device(sz, dev, ctx));
                } else {
                    data_.push_back(::sycl::malloc_shared(sz, dev, ctx));
                }
                if (sz > 0 && !data_[i]) {
                    for (void *p : data_)
                        ::sycl::free(p, ctx);
                    DNN_SAFE(dnnl_out_of_memory, CRIT);
                }
            }
            DNN_SAFE(dnnl_sycl_interop_memory_create_v2(&m_padded_, md_padded,
                             engine_, dnnl_sycl_interop_usm, (int)data_.size(),
                             data_.data()),
                    CRIT);
            SAFE(init_memory(&m_, md_, m_padded_), CRIT);
            break;
        }
        default: assert(!"not expected");
    }
    if (md_padded != md_) DNN_SAFE(dnnl_memory_desc_destroy(md_padded), CRIT);

#else
    (void)handle_info;
#endif
    return OK;
}

int dnn_mem_t::initialize_memory_create_opencl(
        const handle_info_t &handle_info) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    if (handle_info.is_host_ptr) {
        // Ignore memory_kind with host pointers and force USM.
        const int nhandles = query_md_num_handles(md_);
        std::vector<void *> handles(nhandles, handle_info.ptr);
        DNN_SAFE(dnnl_ocl_interop_memory_create_v2(&m_, md_, engine_,
                         dnnl_ocl_interop_usm, (int)handles.size(),
                         handles.data()),
                CRIT);
        return OK;
    }

    SAFE(handle_info.is_allocate() ? OK : FAIL, CRIT);

    auto md_padded = pad_memory_desc(md_, engine_kind_, &is_canary_protected_);
    if (!md_padded) md_padded = md_;

    switch (memory_kind) {
        case memory_kind_ext_t::usm:
        case memory_kind_ext_t::buffer: {
            dnnl_ocl_interop_memory_kind_t mem_kind
                    = (memory_kind == memory_kind_ext_t::usm
                                    ? dnnl_ocl_interop_usm
                                    : dnnl_ocl_interop_buffer);
            const int nhandles = query_md_num_handles(md_);
            std::vector<void *> handles(nhandles, handle_info.ptr);
            DNN_SAFE(dnnl_ocl_interop_memory_create_v2(&m_padded_, md_padded,
                             engine_, mem_kind, (int)handles.size(),
                             handles.data()),
                    CRIT);
            SAFE(init_memory(&m_, md_, m_padded_), CRIT);
            break;
        }
        case memory_kind_ext_t::usm_device:
        case memory_kind_ext_t::usm_shared: {
            is_data_owner_ = true;

            const int nhandles = query_md_num_handles(md_);
            for (int i = 0; i < nhandles; i++) {
                size_t sz = dnnl_memory_desc_get_size_v2(md_padded, i);
                if (memory_kind == memory_kind_ext_t::usm_device) {
                    data_.push_back(dnnl::impl::xpu::ocl::usm::malloc_device(
                            engine_, sz));
                } else {
                    data_.push_back(dnnl::impl::xpu::ocl::usm::malloc_shared(
                            engine_, sz));
                }

                if (sz > 0 && !data_[i]) {
                    for (void *p : data_)
                        dnnl::impl::xpu::ocl::usm::free(engine_, p);
                    DNN_SAFE(dnnl_out_of_memory, CRIT);
                }
            }
            DNN_SAFE(dnnl_ocl_interop_memory_create_v2(&m_padded_, md_padded,
                             engine_, dnnl_ocl_interop_usm, (int)data_.size(),
                             data_.data()),
                    CRIT);
            SAFE(init_memory(&m_, md_, m_padded_), CRIT);
            break;
        }
        default: assert(!"not expected");
    }
    if (md_padded != md_) DNN_SAFE(dnnl_memory_desc_destroy(md_padded), CRIT);
#else
    (void)handle_info;
#endif
    return OK;
}

int dnn_mem_t::initialize_memory_create(const handle_info_t &handle_info) {
    bool is_sycl = is_sycl_engine(engine_);
    bool is_opencl = is_opencl_engine(engine_);

    if (handle_info.is_host_ptr) {
        // Host pointer can be used with CPU memory only.
        // XXX: assumption is that SYCL can work with native host pointers.
        SAFE(is_cpu(engine_) ? OK : FAIL, CRIT);
    }

    if (is_cpu(engine_) && handle_info.is_allocate() && !is_sycl) {
        // Allocate memory for native runtime directly.
        is_data_owner_ = true;
        const size_t alignment = 2 * 1024 * 1024;

        const int nhandles = query_md_num_handles(md_);
        for (int i = 0; i < nhandles; i++) {
            size_t sz = dnnl_memory_desc_get_size_v2(md_, i);
            data_.push_back(zmalloc(sz, alignment));
        }
        if (std::any_of(
                    data_.cbegin(), data_.cend(), [](void *p) { return !p; })) {
            for (void *p : data_)
                zfree(p);
            DNN_SAFE(dnnl_out_of_memory, CRIT);
        }
        DNN_SAFE(dnnl_memory_create_v2(
                         &m_, md_, engine_, (int)data_.size(), data_.data()),
                CRIT);

    } else if (is_sycl) {
        SAFE(initialize_memory_create_sycl(handle_info), CRIT);
    } else if (is_opencl) {
        SAFE(initialize_memory_create_opencl(handle_info), CRIT);
    } else {
        is_data_owner_ = false;
        const int nhandles = query_md_num_handles(md_);
        std::vector<void *> handles(nhandles, handle_info.ptr);
        DNN_SAFE(dnnl_memory_create_v2(&m_, md_, engine_, (int)handles.size(),
                         handles.data()),
                CRIT);
    }
    return OK;
}

int dnn_mem_t::initialize(
        dnnl_engine_t engine, const handle_info_t &handle_info) {
    is_mapped_ = false;
    engine_ = engine;
    engine_kind_ = query_engine_kind(engine_);

    SAFE(initialize_memory_create(handle_info), CRIT);

    if (handle_info.is_allocate()) {
        if (!has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) map();

        const int nhandles = query_md_num_handles(md_);
        for (int i = 0; i < nhandles; i++) {
            size_t sz = dnnl_memory_desc_get_size_v2(md_, i);
            if (is_canary_protected_) sz = pad_memory_size(sz, engine_kind_);
            // Do not fill a memory if its size is zero. Moreover, memset
            // expects defined pointer, nullptr is not allowed.
            if (sz != 0) {
                // Avoid costy data reorders for cold cache mode when
                // initializing cold cache buffers.
                // TODO: consider enabling broadly for perf mode.
                if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)
                        || cold_cache_mode != default_cold_cache_mode) {
                    // Fill memory directly with 0x3F3F3F3F (0.747059f) number.
                    this->memset(dnnl_mem_default_perf_test_value, sz);
                } else {
                    // Fill memory with a magic number (NAN for fp data types)
                    // to catch possible uninitialized access.
                    ::memset(mapped_ptrs_[i], dnnl_mem_default_value, sz);
                }
            }
        }
    }

    return OK;
}

static int cleanup_sycl(
        const dnnl_engine_t &engine, const std::vector<void *> &data) {
#ifdef DNNL_WITH_SYCL
    switch (memory_kind) {
        case memory_kind_ext_t::usm_device:
        case memory_kind_ext_t::usm_shared: {
            auto eng = dnnl::engine(engine, true);
            auto ctx = dnnl::sycl_interop::get_context(eng);
            for (void *p : data)
                ::sycl::free(p, ctx);
            break;
        }
        default: break;
    }
#endif
    return OK;
}

static int cleanup_opencl(
        const dnnl_engine_t &engine, const std::vector<void *> &data) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    switch (memory_kind) {
        case memory_kind_ext_t::usm_device:
        case memory_kind_ext_t::usm_shared:
            for (void *p : data)
                dnnl::impl::xpu::ocl::usm::free(engine, p);
            break;
        default: break;
    }
#endif
    return OK;
}

int dnn_mem_t::cleanup() {
    if (!active_) return OK;
    if (!has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) unmap();
    DNN_SAFE(dnnl_memory_desc_destroy(md_), CRIT);
    DNN_SAFE(dnnl_memory_destroy(m_), CRIT);
    if (is_data_owner_) {
        if (is_sycl_engine(engine_)) {
            SAFE(cleanup_sycl(engine_, data_), CRIT);
        } else if (is_opencl_engine(engine_)) {
            SAFE(cleanup_opencl(engine_, data_), CRIT);
        } else {
            for (void *p : data_)
                zfree(p);
        }
    }
    DNN_SAFE(dnnl_memory_destroy(m_padded_), CRIT);
    return OK;
}

void dnn_mem_t::set_dt(dnnl_data_type_t dt) const {
    // NOLINTNEXTLINE(readability-make-member-function-const)
    dnnl_memory_desc_set_data_type(md_, dt);
}

// Queries from memory descriptor.
int dnn_mem_t::ndims() const {
    return query_md_ndims(md_);
}

// Can't merge two below because compiler doesn't like conversion from
// pointer to reference type.
const dnnl_dims_t &dnn_mem_t::dims() const {
    return query_md_dims(md_);
}

const dnnl_dims_t &dnn_mem_t::padded_dims() const {
    return query_md_padded_dims(md_);
}

dnnl_data_type_t dnn_mem_t::dt(int buffer_index) const {
    return query_md_data_type(md_, buffer_index);
}

const dnnl_dims_t &dnn_mem_t::padded_offsets() const {
    return query_md_padded_offsets(md_);
}

dnnl_dim_t dnn_mem_t::offset0() const {
    return query_md_submemory_offset(md_);
}

dnnl_format_kind_t dnn_mem_t::format_kind() const {
    return query_md_format_kind(md_);
}

const dnnl_dims_t &dnn_mem_t::strides() const {
    return query_md_strides(md_);
}

int dnn_mem_t::inner_nblks() const {
    return query_md_inner_nblks(md_);
}

const dnnl_dims_t &dnn_mem_t::inner_blks() const {
    return query_md_inner_blks(md_);
}

const dnnl_dims_t &dnn_mem_t::inner_idxs() const {
    return query_md_inner_idxs(md_);
}

// Returns physical offset by logical one. logical offset is represented by a
// scalar l_offset. If is_pos_padded is true, l_offset represents logical
// offset in already padded area.
static dnnl_dim_t md_off_l(dnnl_dims_t _pos, const dnn_mem_t &mem,
        dnnl_dim_t l_offset, bool is_pos_padded = false) {
    dnnl_dims_t pos;
    const auto &_dims = is_pos_padded ? mem.padded_dims() : mem.dims();
    for (int rd = 0; rd < mem.ndims(); ++rd) {
        const int d = mem.ndims() - 1 - rd;
        const dnnl_dim_t cur_dim = _dims[d];
        pos[d] = l_offset % cur_dim;
        if (_pos) _pos[d] = pos[d];
        l_offset /= cur_dim;
    }
    return md_off_v(mem, pos, is_pos_padded);
}

template <typename T>
static int check_zero_padding_impl(
        const dnn_mem_t &mem, int arg, res_t *res, int *error_count) {
    const int ndims = mem.ndims();
    const auto &dims = mem.dims();
    const auto &pdims = mem.padded_dims();

    if (ndims == 0) return OK;
    if (mem.format_kind() != dnnl_blocked) return OK;

    auto product = [](const dnnl_dim_t *beg, const dnnl_dim_t *end) {
        return std::accumulate(
                beg, end, (dnnl_dim_t)1, std::multiplies<dnnl_dim_t>());
    };

    int errors = 0;
    std::atomic<int> ok(true);

    for (int dim_m_idx = 0; dim_m_idx < ndims; ++dim_m_idx) {
        if (dims[dim_m_idx] == pdims[dim_m_idx]) continue;

        auto dim_l = product(pdims, pdims + dim_m_idx);
        auto dim_r = product(pdims + dim_m_idx + 1, pdims + ndims);

        benchdnn_parallel_nd(dim_l, dim_r, [&](dnnl_dim_t l, dnnl_dim_t r) {
            for (dnnl_dim_t m = dims[dim_m_idx]; m < pdims[dim_m_idx]; ++m) {
                auto l_idx = (l * pdims[dim_m_idx] + m) * dim_r + r;
                auto idx = md_off_l(nullptr, mem, l_idx, true);
                if (!(mem.get_elem(idx) == 0)) ok = false;
            }
        });

        // Run the check one more time to report incorrect elements. This check
        // is sequential.
        if (!ok) {
            for_(dnnl_dim_t l = 0; l < dim_l; ++l)
            for_(dnnl_dim_t m = dims[dim_m_idx]; m < pdims[dim_m_idx]; ++m)
            for (dnnl_dim_t r = 0; r < dim_r; ++r) {
                auto l_idx = (l * pdims[dim_m_idx] + m) * dim_r + r;
                dnnl_dims_t pos = {};
                auto idx = md_off_l(pos, mem, l_idx, true);

                bool idx_ok = (mem.get_elem(idx) == 0);
                if (!idx_ok) errors++;

                const bool dump = (!idx_ok && (errors < 10 || verbose >= 10))
                        || (verbose >= 99);
                if (dump) {
                    BENCHDNN_PRINT(0,
                            "[%4ld][arg:%d]"
                            "[" IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                            "," IFMT "] fp:  0.f dt:% 9.6g \n",
                            (long)idx, arg, pos[0], pos[1], pos[2], pos[3],
                            pos[4], pos[5], mem.get_elem(idx));
                }
            }
        }
    }

    if (!ok) {
        BENCHDNN_PRINT(0, "@@@ [arg:%d] check_zero_padding failed\n", arg);
        if (res) res->state = FAILED;
    }

    if (error_count != nullptr) *error_count = errors;

    return ok ? OK : FAIL;
}

int check_zero_padding(
        const dnn_mem_t &mem, int arg, res_t *res, int *error_count) {
#define CASE(dt, type) \
    case dt: return check_zero_padding_impl<type>(mem, arg, res, error_count);

    switch (mem.dt()) {
        case dnnl_data_type_undef:
            return OK;

            CASE(dnnl_e8m0, dnnl::impl::float8_e8m0_t);
            CASE(dnnl_f8_e5m2, dnnl::impl::float8_e5m2_t);
            CASE(dnnl_f8_e4m3, dnnl::impl::float8_e4m3_t);
            CASE(dnnl_bf16, dnnl::impl::bfloat16_t);
            CASE(dnnl_f16, dnnl::impl::float16_t);
            CASE(dnnl_f32, float);
            CASE(dnnl_f64, double);
            CASE(dnnl_s32, int32_t);
            CASE(dnnl_s8, int8_t);
            CASE(dnnl_u8, uint8_t);
            CASE(dnnl_s4, dnnl::impl::int4_t);
            CASE(dnnl_u4, dnnl::impl::uint4_t);
            CASE(dnnl_f4_e2m1, dnnl::impl::float4_e2m1_t);
            CASE(dnnl_f4_e3m0, dnnl::impl::float4_e3m0_t);
        default: assert(!"bad data_type");
    };
#undef CASE

    return FAIL;
}

int check_buffer_overwrite(const dnn_mem_t &mem, int arg, res_t *res) {
    if (!mem.is_canary_protected()) return OK;

    size_t sz = mem.size();
    size_t sz_padded = dnn_mem_t::pad_memory_size(sz, mem.engine_kind());

    auto *mem_ptr = (const uint8_t *)mem;
    for (size_t i = sz; i < sz_padded; i++) {
        if (mem_ptr[i] == dnnl_mem_default_value) continue;

        BENCHDNN_PRINT(0,
                "@@@ [arg:%d] check_buffer_overwrite failed. Expected: %d at "
                "byte: %lld but found: %d\n",
                arg, dnnl_mem_default_value, (long long)i, mem_ptr[i]);
        if (res) res->state = FAILED;
        return FAIL;
    }
    return OK;
}

// Returns physical offset by logical one. Logical offset is represented by an
// array pos. If is_pos_padded is true pos represents the position in already
// padded area.
dnnl_dim_t md_off_v(
        const dnn_mem_t &mem, const dnnl_dims_t pos, bool is_pos_padded) {
    assert(mem.format_kind() == dnnl_blocked);

    dnnl_dims_t pos_copy = {0};
    for (int d = 0; d < mem.ndims(); ++d)
        pos_copy[d] = pos[d] + (is_pos_padded ? 0 : mem.padded_offsets()[d]);

    dnnl_dim_t phys_offset = mem.offset0();

    const int nblks = mem.inner_nblks();
    if (nblks > 0) {
        const auto &inner_idxs = mem.inner_idxs();
        const auto &inner_blks = mem.inner_blks();
        dnnl_dim_t blk_stride = 1;
        for (int iblk = nblks - 1; iblk >= 0; --iblk) {
            const int d = inner_idxs[iblk];

            dnnl_dim_t p = pos_copy[d] % inner_blks[iblk];
            pos_copy[d] /= inner_blks[iblk];

            phys_offset += p * blk_stride;
            blk_stride *= inner_blks[iblk];
        }
    }

    for (int d = 0; d < mem.ndims(); ++d) {
        const dnnl_dim_t p = pos_copy[d];
        phys_offset += p * mem.strides()[d];
    }

    return phys_offset;
}

dnnl_memory_desc_t clone_md(const_dnnl_memory_desc_t md) {
    dnnl_memory_desc_t cloned_md;
    auto status = dnnl_memory_desc_clone(&cloned_md, md);
    if (status != dnnl_success) return nullptr;
    return cloned_md;
}

/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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

#ifndef DNNL_MEMORY_HPP
#define DNNL_MEMORY_HPP

#include <unordered_map>

#include "oneapi/dnnl/dnnl.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_DPCPP
#include "oneapi/dnnl/dnnl_sycl.h"
#endif

#include "common.hpp"
#include "utils/dims.hpp"
#include "utils/wrapper.hpp"

#define dnnl_mem_default_value 0xFF
#define dnnl_mem_default_perf_test_value 0x3F

struct dnn_mem_t {
    struct handle_info_t {
        bool is_host_ptr;
        void *ptr;

        bool is_allocate() const { return ptr == DNNL_MEMORY_ALLOCATE; }

        static handle_info_t allocate() {
            return {false, DNNL_MEMORY_ALLOCATE};
        }
    };

    dnn_mem_t() { map(); }
    dnn_mem_t(const_dnnl_memory_desc_t md, dnnl_engine_t engine,
            const handle_info_t &handle_info = handle_info_t::allocate());
    dnn_mem_t(const_dnnl_memory_desc_t md, dnnl_data_type_t dt,
            const std::string &tag, dnnl_engine_t engine);

    dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
            const std::string &tag, dnnl_engine_t engine);
    dnn_mem_t(int ndims, const dnnl_dims_t dims, dnnl_data_type_t dt,
            const dnnl_dims_t strides, dnnl_engine_t engine);

    dnn_mem_t(const dnn_mem_t &rhs, dnnl_data_type_t dt, const std::string &tag,
            dnnl_engine_t engine);

    dnn_mem_t(const dnn_mem_t &rhs) = delete;
    dnn_mem_t &operator=(const dnn_mem_t &rhs) = delete;

    dnn_mem_t &operator=(dnn_mem_t &&rhs) {
        if (&rhs == this) return *this;
        cleanup();

        md_ = rhs.md_;
        m_ = rhs.m_;
        m_padded_ = rhs.m_padded_;
        data_ = std::move(rhs.data_);
        is_data_owner_ = rhs.is_data_owner_;
        active_ = rhs.active_;
        engine_kind_ = rhs.engine_kind_;
        engine_ = rhs.engine_;
        is_mapped_ = (bool)rhs.is_mapped_;
        mapped_ptrs_ = std::move(rhs.mapped_ptrs_);

        rhs.active_ = false;
        return *this;
    }
    dnn_mem_t(dnn_mem_t &&rhs) : dnn_mem_t() { *this = std::move(rhs); }

    ~dnn_mem_t() { cleanup(); }

    int reorder(const dnn_mem_t &rhs, const_dnnl_primitive_attr_t attr);
    int reorder(const dnn_mem_t &rhs) { return reorder(rhs, nullptr); }

    size_t size() const;

    int64_t nelems(bool with_padded_dims = false) const {
        const auto &_dims = with_padded_dims ? padded_dims() : dims();
        if (ndims() == 0) return 0;

        int64_t n = 1;
        for (int i = 0; i < ndims(); ++i)
            n *= _dims[i];
        return n;
    }

    // Queries from memory descriptor.
    int ndims() const;
    const dnnl_dims_t &dims() const;
    const dnnl_dims_t &padded_dims() const;
    dnnl_data_type_t dt(int buffer_index = 0) const;
    const dnnl_dims_t &padded_offsets() const;
    dnnl_dim_t offset0() const;
    dnnl_format_kind_t format_kind() const;
    const dnnl_dims_t &strides() const;
    int inner_nblks() const;
    const dnnl_dims_t &inner_blks() const;
    const dnnl_dims_t &inner_idxs() const;

    size_t sizeof_dt() const;

    void set_dt(dnnl_data_type_t dt) const;

    template <typename T>
    explicit operator T *() const {
        assert(is_mapped_);
        // Always return 0th ptr.
        return static_cast<T *>(get_mapped_pointer<T>(0));
    }

    template <typename T>
    T *get_mapped_pointer(int index = 0) const {
        assert(is_mapped_);
        return static_cast<T *>(mapped_ptrs_[index]);
    }

    explicit operator bool() const { return active_; }

    float get_elem(int64_t idx, int buffer_index = 0) const;
    void set_elem(int64_t idx, float value, int buffer_index = 0) const;

    int64_t get_scale_idx(
            int64_t data_idx, int scale_mask, const int ndims) const {
        const auto &_dims = dims();
        int64_t stride = 1;
        int64_t offset = 0;

        if (scale_mask != 0) {
            for (int i = 0; i < ndims; ++i) {
                int d = ndims - 1 - i;
                auto pos = data_idx % _dims[d];
                data_idx /= _dims[d];
                if (scale_mask & (1 << d)) {
                    offset += pos * stride;
                    stride *= _dims[d];
                }
            }
        }

        return offset;
    }

    int64_t get_scale_idx(int64_t data_idx, int scale_mask) const {
        return get_scale_idx(data_idx, scale_mask, ndims());
    }

    dnnl_engine_t engine() const { return engine_; }
    dnnl_engine_kind_t engine_kind() const { return engine_kind_; }

    bool is_mapped() const { return is_mapped_; }

    bool is_canary_protected() const { return is_canary_protected_; }

    void map() const;
    void unmap() const;
    void memset(int value, size_t size) const;

    static dnn_mem_t create_from_host_ptr(
            const dnnl_memory_desc_t &md, dnnl_engine_t engine, void *host_ptr);

    // Increases memory size to catch potential buffer overreads and
    // overwrites. The padded area is filled with a canary value.
    static size_t pad_memory_size(size_t sz, dnnl_engine_kind_t engine_kind,
            bool *was_padded = nullptr);
    // Increases memory descriptor size to catch potential buffer overreads and
    // overwrites. The padded area is filled with a canary value.
    static dnnl_memory_desc_t pad_memory_desc(const_dnnl_memory_desc_t md,
            dnnl_engine_kind_t engine_kind, bool *was_padded = nullptr);
    // Initializes memory descriptor from sporadic tag or strides.
    static benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> init_md(int ndims,
            const dnnl_dims_t dims, dnnl_data_type_t data_type,
            const std::string &tag, const dims_t &strides_ = {});
#ifdef DNNL_EXPERIMENTAL_SPARSE
    // Initializes memory descriptor for CSR encoding.
    static benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> init_csr_md(int ndims,
            const dnnl_dims_t dims, dnnl_data_type_t data_type, dnnl_dim_t nnz,
            dnnl_data_type_t indices_dt, dnnl_data_type_t pointers_dt);
#endif

    /* fields */
    dnnl_memory_desc_t md_ {};
    dnnl_memory_t m_ {};

    // "Base" memory with a canary-padded buffer for buffer overflow
    // protection.
    dnnl_memory_t m_padded_ {};
    bool is_canary_protected_ = false;

private:
    std::vector<void *> data_;
    bool is_data_owner_ = false;
    bool active_ = false;

    dnnl_engine_kind_t engine_kind_ = dnnl_any_engine;
    dnnl_engine_t engine_ = NULL;

    mutable bool is_mapped_ = false;
    mutable std::vector<void *> mapped_ptrs_;

    int initialize_memory_create_sycl(const handle_info_t &handle_info);
    int initialize_memory_create_opencl(const handle_info_t &handle_info);
    int initialize_memory_create(const handle_info_t &handle_info);

    int initialize(dnnl_engine_t engine,
            const handle_info_t &handle_info = handle_info_t::allocate());

    int cleanup();
};

using dnn_mem_map_t = std::unordered_map<int, dnn_mem_t>;

dnnl_memory_desc_t clone_md(const_dnnl_memory_desc_t md);

// Checks that zero padding is preserved.
int check_zero_padding(const dnn_mem_t &mem, int arg, res_t *res = nullptr,
        int *error_count = nullptr);

// Checks that the buffer is not overrun if it was protected by a canary.
int check_buffer_overwrite(const dnn_mem_t &mem, int arg, res_t *res = nullptr);

// Returns physical offset by logical one. Logical offset is represented by an
// array pos. If is_pos_padded is true pos represents the position in already
// padded area.
dnnl_dim_t md_off_v(const dnn_mem_t &mem, const dnnl_dims_t pos,
        bool is_pos_padded = false);

#endif

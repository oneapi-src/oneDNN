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

#ifndef _MKLDNN_MEMORY_HPP
#define _MKLDNN_MEMORY_HPP

#include "mkldnn_common.hpp"

struct dnn_mem_t {
    dnn_mem_t() {}

    dnn_mem_t(const mkldnn_memory_desc_t &md, mkldnn_engine_t engine) {
        active_ = (initialize(md, engine) == OK);
    }

    dnn_mem_t(int ndims, const mkldnn_dims_t dims, mkldnn_data_type_t dt,
            mkldnn_format_tag_t tag, mkldnn_engine_t engine) {
        active_ = (initialize(ndims, dims, dt, tag, engine) == OK);
    }

    dnn_mem_t(int ndims, const mkldnn_dims_t dims, mkldnn_data_type_t dt,
            mkldnn_format_tag_t tag, const mkldnn_memory_extra_desc_t &extra,
            mkldnn_engine_t engine) {
        active_ = (initialize(ndims, dims, dt, tag, extra, engine) == OK);
    }

    dnn_mem_t(int ndims, const mkldnn_dims_t dims, mkldnn_data_type_t dt,
            const mkldnn_dims_t strides, mkldnn_engine_t engine) {
        active_ = (initialize(ndims, dims, dt, strides, engine) == OK);
    }

    dnn_mem_t(const mkldnn_memory_desc_t &md, mkldnn_data_type_t dt,
            mkldnn_format_tag_t tag = mkldnn_format_tag_undef,
            mkldnn_engine_t engine = engine_ref) {
        active_ = (initialize(md, dt, tag, engine) == OK);
    }

    dnn_mem_t(const mkldnn_memory_desc_t &md, mkldnn_data_type_t dt,
            mkldnn_engine_t engine = engine_ref) {
        active_ = (initialize(md, dt, mkldnn_format_tag_undef, engine) == OK);
    }

    dnn_mem_t(const dnn_mem_t &rhs, mkldnn_data_type_t dt,
            mkldnn_format_tag_t tag = mkldnn_format_tag_undef,
            mkldnn_engine_t engine = engine_ref)
        : dnn_mem_t(rhs.md_, dt, tag, engine) {
        if (active_)
            reorder(rhs);
    }

    dnn_mem_t(const dnn_mem_t &rhs) = delete;
    dnn_mem_t &operator=(const dnn_mem_t &rhs) = delete;

    dnn_mem_t &operator=(dnn_mem_t &&rhs) {
        if (&rhs == this) return *this;
        cleanup();

        md_ = rhs.md_;
        m_ = rhs.m_;
        data_ = rhs.data_;
        is_data_owner_ = rhs.is_data_owner_;
        active_ = rhs.active_;
        engine_kind_ = rhs.engine_kind_;
        engine_ = rhs.engine_;
        is_cpu_native_ = rhs.is_cpu_native_;
        is_mapped_ = rhs.is_mapped_;
        mapped_ptr_ = rhs.mapped_ptr_;

        rhs.active_ = false;
        return *this;
    }
    dnn_mem_t(dnn_mem_t &&rhs) : dnn_mem_t() {
        *this = std::move(rhs);
    }


    ~dnn_mem_t() { cleanup(); }

    int reorder(const dnn_mem_t &rhs) { return reorder(rhs, NULL); }
    int reorder(const dnn_mem_t &rhs, const mkldnn_primitive_attr_t &attr) {
        if (this == &rhs) return OK;

        mkldnn_primitive_desc_t rpd;
        DNN_SAFE(mkldnn_reorder_primitive_desc_create(&rpd,
                    &rhs.md_, rhs.engine_, &md_, engine_, attr), WARN);

        mkldnn_primitive_t r;
        DNN_SAFE(mkldnn_primitive_create(&r, rpd), WARN);
        mkldnn_engine_t reorder_engine;
        DNN_SAFE(mkldnn_primitive_desc_query(
                         rpd, mkldnn_query_engine, 0, &reorder_engine),
                CRIT);
        DNN_SAFE(mkldnn_primitive_desc_destroy(rpd), CRIT);

        mkldnn_exec_arg_t args[] = {
            {MKLDNN_ARG_FROM, rhs.m_},
            {MKLDNN_ARG_TO, m_},
        };

        mkldnn_stream_t reorder_stream
                = (reorder_engine == engine_ref) ? stream_ref : stream_tgt;

        DNN_SAFE(execute_and_wait(r, reorder_stream, 2, args), WARN);
        DNN_SAFE(mkldnn_primitive_destroy(r), CRIT);

        return OK;
    }

    size_t size() const { return mkldnn_memory_desc_get_size(&md_); }

    int64_t nelems(bool with_padded_dims = false) const {
        auto dims = with_padded_dims
            ? md_.padded_dims
            : md_.dims;
        int64_t n = 1;
        for (int i = 0; i < md_.ndims; ++i)
            n *= dims[i];
        return n;
    }

    mkldnn_data_type_t dt() const { return md_.data_type; }
    size_t sizeof_dt() const { return ::sizeof_dt(dt()); }

    template <typename T>
    explicit operator T *() const {
        if (engine_ == engine_ref) {
            // Assume that the reference engine supports direct memory access
            // without map/unmap
            return static_cast<T *>(data_);
        }

        assert(is_mapped_ && "direct access only for mapped memory");
        return static_cast<T *>(mapped_ptr_);
    }

    float get_elem(int64_t idx) const {
        void *data = (void *)*this;
        float elem = 0.0;
        switch (dt()) {
        case mkldnn_s8: elem = static_cast<int8_t *>(data)[idx]; break;
        case mkldnn_u8: elem = static_cast<uint8_t *>(data)[idx]; break;
        case mkldnn_s32: elem = static_cast<int32_t *>(data)[idx]; break;
        case mkldnn_f32: elem = static_cast<float *>(data)[idx]; break;
        case mkldnn_f16: elem = static_cast<float16_t *>(data)[idx]; break;
        case mkldnn_bf16: elem = static_cast<bfloat16_t *>(data)[idx]; break;
        default: assert(!"bad data type");
        }
        return elem;
    }

    void set_elem(int64_t idx, float value) {
        void *data = (void *)*this;
        switch (dt()) {
            case mkldnn_s8: ((int8_t *)data)[idx] = value; break;
            case mkldnn_u8: ((uint8_t *)data)[idx] = value; break;
            case mkldnn_s32: ((int32_t *)data)[idx] = value; break;
            case mkldnn_f32: ((float *)data)[idx] = value; break;
            case mkldnn_f16: ((float16_t *)data)[idx] = value; break;
            case mkldnn_bf16: ((bfloat16_t *)data)[idx] = value; break;
            default: assert(!"bad data type");
        }
    }

    int64_t get_scale_idx(int64_t data_idx, int scale_mask) const {
        const int ndims = md_.ndims;
        const auto &dims = md_.dims;
        int64_t stride = 1;
        int64_t offset = 0;

        if (scale_mask != 0) {
            for (int i = 0; i < ndims; ++i) {
                int d = md_.ndims - 1 - i;
                auto pos = data_idx % dims[d];
                data_idx /= dims[d];
                if (scale_mask & (1 << d)) {
                    offset += pos * stride;
                    stride *= dims[d];
                }
            }
        }

        return offset;
    }

    void map() {
        assert(!is_mapped_ && "memory is already mapped");

        DNN_SAFE_V(mkldnn_memory_map_data(m_, &mapped_ptr_));
        is_mapped_ = true;
    }

    void unmap() {
        assert(is_mapped_ && "memory is not mapped");

        DNN_SAFE_V(mkldnn_memory_unmap_data(m_, mapped_ptr_));
        is_mapped_ = false;
        mapped_ptr_ = NULL;
    }

    /* fields */

    mkldnn_memory_desc_t md_{};
    mkldnn_memory_t m_{};

private:
    void *data_ = NULL;
    bool is_data_owner_ = false;
    bool active_ = false;

    mkldnn_engine_kind_t engine_kind_ = mkldnn_any_engine;
    mkldnn_engine_t engine_ = NULL;

    bool is_cpu_native_ = false;

    bool is_mapped_ = false;
    void *mapped_ptr_ = NULL;

    int initialize(const mkldnn_memory_desc_t &md, mkldnn_data_type_t dt,
            mkldnn_format_tag_t tag, mkldnn_engine_t engine) {
        if (tag == mkldnn_format_tag_undef) {
            md_ = md;
            md_.data_type = dt;
        } else {
            DNN_SAFE(mkldnn_memory_desc_init_by_tag(
                        &md_, md.ndims, md.dims, dt, tag), CRIT);
        }
        engine_ = engine;
        DNN_SAFE_V(mkldnn_engine_get_kind(engine_, &engine_kind_));

        int backend_kind;
        DNN_SAFE_V(mkldnn_engine_get_backend_kind(engine_, &backend_kind));
        is_cpu_native_ = (engine_kind_ == mkldnn_cpu) && (backend_kind == 0);

        if (is_cpu_native_) {
            // Allocate memory for native backend directly
            is_data_owner_ = true;
            const size_t alignment = 64;
            size_t sz = mkldnn_memory_desc_get_size(&md_);
            data_ = zmalloc(sz, alignment);
            DNN_SAFE(data_ == NULL ? mkldnn_out_of_memory : mkldnn_success,
                    CRIT);
            DNN_SAFE(mkldnn_memory_create(&m_, &md_, engine, data_), CRIT);

            // Init reference float type memory with NANs
            if (engine == engine_ref && dt == mkldnn_f32)
                for (int64_t i = 0; i < (int64_t)(sz / sizeof(float)); i++)
                    ((float *)data_)[i] = NAN;
        } else {
            is_data_owner_ = false;
            data_ = NULL;
            DNN_SAFE(mkldnn_memory_create(
                             &m_, &md_, engine, MKLDNN_MEMORY_ALLOCATE),
                    CRIT);
        }

        is_mapped_ = false;
        mapped_ptr_ = NULL;

        return OK;
    }

    int initialize(const mkldnn_memory_desc_t &md, mkldnn_engine_t engine) {
        return initialize(md, md.data_type, mkldnn_format_tag_undef, engine);
    }

    int initialize(int ndims, const mkldnn_dims_t dims, mkldnn_data_type_t dt,
            mkldnn_format_tag_t tag, mkldnn_engine_t engine) {
        mkldnn_memory_desc_t xmd;
        DNN_SAFE(mkldnn_memory_desc_init_by_tag(&xmd, ndims, dims, dt, tag), CRIT);
        SAFE(initialize(xmd, engine), CRIT);
        return OK;
    }

    int initialize(int ndims, const mkldnn_dims_t dims, mkldnn_data_type_t dt,
            mkldnn_format_tag_t tag, const mkldnn_memory_extra_desc_t &extra,
            mkldnn_engine_t engine) {
        mkldnn_memory_desc_t xmd;
        DNN_SAFE(mkldnn_memory_desc_init_by_tag(&xmd, ndims, dims, dt, tag), CRIT);
        xmd.extra = extra;
        SAFE(initialize(xmd, engine), CRIT);
        return OK;
    }

    int initialize(int ndims, const mkldnn_dims_t dims, mkldnn_data_type_t dt,
            const mkldnn_dims_t strides, mkldnn_engine_t engine) {
        mkldnn_memory_desc_t xmd;
        DNN_SAFE(mkldnn_memory_desc_init_by_strides(
                    &xmd, ndims, dims, dt, strides), CRIT);
        SAFE(initialize(xmd, engine), CRIT);
        return OK;
    }

    int cleanup() {
        if (!active_) return OK;
        DNN_SAFE(mkldnn_memory_destroy(m_), CRIT);
        if (is_data_owner_) zfree(data_);
        return OK;
    }
};

#endif

/*******************************************************************************
* Copyright 2017 Intel Corporation
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
    dnn_mem_t(): active_(false) {}

    dnn_mem_t(const mkldnn_memory_desc_t &md, void *data = NULL): active_(true)
    { initialize(md, data); }

    dnn_mem_t(int ndims, mkldnn_dims_t dims, mkldnn_data_type_t dt,
            mkldnn_memory_format_t fmt, void *data = NULL): active_(true) {
        mkldnn_memory_desc_t md;
        /* is it ugly enough? */
        [&](){
            DNN_SAFE(mkldnn_memory_desc_init(&md, ndims, dims, dt, fmt),
                    CRIT);
            SAFE(initialize(md, data), CRIT);
            return OK;
        }();
    }

    dnn_mem_t(const mkldnn_memory_desc_t &md, mkldnn_data_type_t dt,
            mkldnn_memory_format_t fmt = mkldnn_format_undef,
            void *data = NULL): active_(true) {
        mkldnn_memory_desc_t xmd;
        [&](){
            DNN_SAFE(mkldnn_memory_desc_init(&xmd, md.ndims, md.dims, dt,
                        fmt != mkldnn_format_undef ? fmt : md.format), CRIT);
            SAFE(initialize(xmd, data), CRIT);
            return OK;
        }();
    }

    dnn_mem_t(const dnn_mem_t &rhs, mkldnn_data_type_t dt,
            mkldnn_memory_format_t fmt = mkldnn_format_undef,
            void *data = NULL): dnn_mem_t(rhs.md_, dt, fmt, data)
    { reorder(rhs); }

    /* FIXME: ugly RT assert... need better mkldnn memory handling */
    dnn_mem_t &operator=(const dnn_mem_t &rhs)
    { []() { SAFE(FAIL, CRIT); return 0; }(); return *this; }
    dnn_mem_t(const dnn_mem_t &rhs)
    { []() { SAFE(FAIL, CRIT); return 0; }(); }

    ~dnn_mem_t() { cleanup(); }

#if 0
    /* I don't know c++ :( */
    dnn_mem_t &operator=(dnn_mem_t &rhs) {
        if (&rhs == this) return *this;
        cleanup();
        md_ = rhs.md_;
        mpd_ = rhs.mpd_; rhs.mpd_ = NULL;
        p_ = rhs.p_; rhs.p_ = NULL;
        data_ = rhs.data_; rhs.data_ = NULL;
        is_data_owner_ = rhs.is_data_owner_; rhs.is_data_owner_ = false;
        active_ = !(rhs.active_ = false);
        return *this;
    }
#endif

    int reorder(const dnn_mem_t &rhs) {
        if (this == &rhs) return OK;

        mkldnn_primitive_desc_t rpd;
        mkldnn_primitive_t r;
        DNN_SAFE(mkldnn_reorder_primitive_desc_create(&rpd, rhs.mpd_, mpd_),
                WARN);
        mkldnn_primitive_at_t i = {rhs.p_, 0};
        const_mkldnn_primitive_t o = p_;
        DNN_SAFE(mkldnn_primitive_create(&r, rpd, &i, &o), WARN);
        SAFE(execute(r), WARN);
        DNN_SAFE(mkldnn_primitive_desc_destroy(rpd), CRIT);
        DNN_SAFE(mkldnn_primitive_destroy(r), CRIT);

        return OK;
    }

    int N() { return md_.dims[0]; }
    int with_G() { return md_.ndims == 5; }
    int G() { return md_.ndims == 5 ? md_.dims[0] : 1; }

    int C() { return md_.ndims == 1 ? md_.dims[0] : md_.dims[1]; }
    int OC() { return md_.dims[with_G() + 0]; }
    int IC() { return md_.dims[with_G() + 1]; }
    int H() { return md_.dims[with_G() + 2]; } // works for both IH and KH
    int W() { return md_.dims[with_G() + 3]; } // works for both IW and KW

    size_t size() { return mkldnn_memory_primitive_desc_get_size(mpd_); }
    size_t nelems() {
        DNN_SAFE(md_.data_type != mkldnn_f32
                ? mkldnn_invalid_arguments : mkldnn_success, CRIT);
        return size() / sizeof(float);
    }

    mkldnn_data_type_t dt() const { return md_.data_type; }

    template <typename T>
    explicit operator T*() const { return static_cast<T*>(data_); }

    /* fields */

    mkldnn_memory_desc_t md_;
    mkldnn_primitive_desc_t mpd_;
    mkldnn_primitive_t p_;
    void *data_;
    bool is_data_owner_, active_;

private:
    int initialize(const mkldnn_memory_desc_t &md, void *data) {
        md_ = md;
        DNN_SAFE(mkldnn_memory_primitive_desc_create(&mpd_, &md_, engine),
                CRIT);
        DNN_SAFE(mkldnn_primitive_create(&p_, mpd_, NULL, NULL), CRIT);
        is_data_owner_ = data == NULL;
        if (data == NULL) {
            size_t sz = mkldnn_memory_primitive_desc_get_size(mpd_);
            data_ = zmalloc(sz, 64);
            DNN_SAFE(data_ == NULL ? mkldnn_out_of_memory : mkldnn_success,
                    WARN);
        } else {
            data_ = data;
        }
        DNN_SAFE(mkldnn_memory_set_data_handle(p_, data_), CRIT);

        return OK;
    }

    int cleanup() {
        if (!active_) return OK;
        DNN_SAFE(mkldnn_primitive_desc_destroy(mpd_), CRIT);
        DNN_SAFE(mkldnn_primitive_destroy(p_), CRIT);
        if (is_data_owner_) zfree(data_);
        return OK;
    }
};

#endif

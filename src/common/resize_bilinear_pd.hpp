/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef RESIZE_BILINEAR_PD_HPP
#define RESIZE_BILINEAR_PD_HPP

#include "mkldnn.h"
#include "patch_mkldnn.hpp"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {

struct CachedInterpolation {
    int lower;  // Lower source index used in the interpolation
    int upper;  // Upper source index used in the interpolation
    float lerp;
};

struct resize_bilinear_fwd_pd_t;

struct resize_bilinear_pd_t: public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::resize_bilinear;

    resize_bilinear_pd_t(engine_t *engine,
            const resize_bilinear_desc_t *adesc, 
            const primitive_attr_t *attr,
            const resize_bilinear_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , ws_md_()
    {}

    const resize_bilinear_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { impl::init_info(this, this->info_); }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::resize_bilinear_d:
            *(const resize_bilinear_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common resize_bilinear aux functions */

    inline int MB() const { return src_desc().dims[0]; }
    inline int C() const { return src_desc().dims[1]; }

    inline int ID() const { return ndims() >= 5 ? src_desc().dims[ndims() - 3] : 1; }
    inline int IH() const { return ndims() >= 4 ? src_desc().dims[ndims() - 2] : 1; }
    inline int IW() const { return src_desc().dims[ndims() - 1]; }

    inline int OD() const { return ndims() >= 5 ? dst_desc().dims[ndims() - 3] : 1; }
    inline int OH() const { return ndims() >= 4 ? dst_desc().dims[ndims() - 2] : 1; }
    inline int OW() const { return dst_desc().dims[ndims() - 1]; }

    inline int alignCorners() const { return align_corners(); }
    inline float RATE(const int _in, const int _out) const {
        float float_in = _in * 1.0;
        float scale = 0.0;
        if (align_corners() && _out > 1)
            scale = (float_in - 1) / (_out - 1);
        else
            scale = float_in / _out;

        return scale;
    }
    inline void interpolationWeights(const int in_size, 
                                    const int out_size, 
                                    const float scale, 
                                    CachedInterpolation* interpolation) const {
        interpolation[out_size].lower = 0;
        interpolation[out_size].upper = 0;
        for (int i = out_size - 1; i >= 0; --i) {
            const float in = i * scale;
            interpolation[i].lower = static_cast<int>(in);
            interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
            interpolation[i].lerp = in - interpolation[i].lower;
        }
    }

    int ndims() const { return src_desc().ndims; }
    int spatial_ndims() const { return ndims() - 2; }
    bool is_3d() const { return ndims() == 5; }

    bool has_zero_dim_memory() const
    { return memory_desc_wrapper(src_desc()).has_zero_dim(); }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

protected:
    resize_bilinear_desc_t desc_;
    const resize_bilinear_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t ws_md_;

    void init_default_ws(data_type_t dt = data_type::undef) {
        ws_md_ = is_fwd() ? *dst_md() : *diff_dst_md();
        ws_md_.data_type = (dt != data_type::undef) ? dt : indices_data_type();
    }

    data_type_t indices_data_type() const {
#if 0
        /* the simplest way to express 256... */
        const int u8_max = nstl::numeric_limits<
            typename prec_traits<data_type::u8>::type>::max();
        return utils::array_product(desc()->kernel, spatial_ndims()) <= u8_max
            ? data_type::u8 : data_type::s32;
#else
        return data_type::s32;
#endif
    }

private:
    const memory_desc_t &src_desc() const { return desc_.src_desc; }
    const memory_desc_t &dst_desc() const { return desc_.dst_desc; }
    const int &align_corners() const { return desc_.align_corners; }
};

struct resize_bilinear_fwd_pd_t: public resize_bilinear_pd_t {
    typedef resize_bilinear_fwd_pd_t base_class;
    typedef resize_bilinear_fwd_pd_t hint_class;

    resize_bilinear_fwd_pd_t(engine_t *engine,
            const resize_bilinear_desc_t *adesc, 
            const primitive_attr_t *attr,
            const resize_bilinear_fwd_pd_t *hint_fwd_pd)
        : resize_bilinear_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , dst_md_(desc_.dst_desc)
    {}

    virtual arg_usage_t arg_usage(int arg) const override {
        if (arg == MKLDNN_ARG_SRC)
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DST)
            return arg_usage_t::output;

        if (arg == MKLDNN_ARG_WORKSPACE && (workspace_md() != nullptr))
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *src_md(int index = 0) const override
    { return index == 0 ? &src_md_ : nullptr; }
    virtual const memory_desc_t *dst_md(int index = 0) const override
    { return index == 0 ? &dst_md_ : nullptr; }
    virtual const memory_desc_t *workspace_md(int index = 0) const override
    { return index == 0 && !types::is_zero_md(&ws_md_) ? &ws_md_ : nullptr; }

    virtual int n_inputs() const override { return 1; }
    virtual int n_outputs() const override
    { return 1 + (workspace_md() != nullptr); }

protected:
    memory_desc_t src_md_;
    memory_desc_t dst_md_;

    virtual status_t set_default_params() {
        if (dst_md()->format_kind != format_kind::any)
            return status::success;

        if (src_md()->format_kind != format_kind::blocked)
            return status::unimplemented;

        return memory_desc_init_by_blocking_desc(dst_md_,
                src_md_.format_desc.blocking);
    }
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s


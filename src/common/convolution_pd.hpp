/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CONVOLUTION_PD_HPP
#define CONVOLUTION_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

status_t conv_desc_init(convolution_desc_t *conv_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t dilates,
        const dims_t padding_l, const dims_t padding_r,
        padding_kind_t padding_kind);

memory_desc_t *conv_prop_agnostic_src_d(convolution_desc_t *desc);
memory_desc_t *conv_prop_agnostic_wei_d(convolution_desc_t *desc);
memory_desc_t *conv_prop_agnostic_bia_d(convolution_desc_t *desc);
memory_desc_t *conv_prop_agnostic_dst_d(convolution_desc_t *desc);
const memory_desc_t *conv_prop_agnostic_src_d(const convolution_desc_t *desc);
const memory_desc_t *conv_prop_agnostic_wei_d(const convolution_desc_t *desc);
const memory_desc_t *conv_prop_agnostic_bia_d(const convolution_desc_t *desc);
const memory_desc_t *conv_prop_agnostic_dst_d(const convolution_desc_t *desc);

struct convolution_fwd_pd_t;

struct convolution_pd_t: public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::convolution;

    convolution_pd_t(engine_t *engine,
            const convolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
    {}

    const convolution_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_conv(this, this->info_); }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case pkind_traits<base_pkind>::query_d:
            *(const convolution_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common conv aux functions */

    int MB() const { return _src_md()->dims[0]; }

    int IC() const { return _src_md()->dims[1]; }
    int OC() const { return _dst_md()->dims[1]; }
    int G() const { return with_groups() ? _wei_md()->dims[0] : 1; }

    int ID() const { return ndims() >= 5 ? _src_md()->dims[ndims() - 3] : 1; }
    int IH() const { return ndims() >= 4 ? _src_md()->dims[ndims() - 2] : 1; }
    int IW() const { return _src_md()->dims[ndims() - 1]; }

    int OD() const { return ndims() >= 5 ? _dst_md()->dims[ndims() - 3] : 1; }
    int OH() const { return ndims() >= 4 ? _dst_md()->dims[ndims() - 2] : 1; }
    int OW() const { return _dst_md()->dims[ndims() - 1]; }

    int KD() const { return ndims() >= 5 ? _wei_md()->dims[ndims() + with_groups() - 3] : 1; }
    int KH() const { return ndims() >= 4 ? _wei_md()->dims[ndims() + with_groups() - 2] : 1; }
    int KW() const { return _wei_md()->dims[ndims() + with_groups() - 1]; }

    int KSD() const { return ndims() >= 5 ? desc_.strides[ndims() - 5] : 1; }
    int KSH() const { return ndims() >= 4 ? desc_.strides[ndims() - 4] : 1; }
    int KSW() const { return desc_.strides[ndims() - 3]; }

    int KDD() const { return ndims() >= 5 ? desc_.dilates[ndims() - 5] : 0; }
    int KDH() const { return ndims() >= 4 ? desc_.dilates[ndims() - 4] : 1; }
    int KDW() const { return desc_.dilates[ndims() - 3]; }

    int padFront() const { return ndims() >= 5 ? desc_.padding[0][ndims() - 5] : 0; }
    int padBack() const { return ndims() >= 5 ? desc_.padding[1][ndims() - 5] : 0; }
    int padT() const { return ndims() >= 4 ? desc_.padding[0][ndims() - 4] : 0; }
    int padB() const { return ndims() >= 4 ? desc_.padding[1][ndims() - 4] : 0; }
    int padL() const { return desc_.padding[0][ndims() - 3]; }
    int padR() const { return desc_.padding[1][ndims() - 3]; }

    int ndims() const { return _src_md()->ndims; }

    bool with_bias() const { return !memory_desc_wrapper(*_bia_md()).is_zero(); }
    bool with_groups() const { return _wei_md()->ndims == ndims() + 1; }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    bool has_zero_dim_memory() const {
        const auto s_d = memory_desc_wrapper(*_src_md());
        const auto d_d = memory_desc_wrapper(*_dst_md());
        return s_d.has_zero_dim() || d_d.has_zero_dim();
    }

protected:
    convolution_desc_t desc_;
    const convolution_fwd_pd_t *hint_fwd_pd_;

    bool set_default_alg_kind(alg_kind_t alg_kind) {
        assert(utils::one_of(alg_kind, alg_kind::convolution_direct,
                    alg_kind::convolution_winograd));
        if (desc_.alg_kind == alg_kind::convolution_auto)
            desc_.alg_kind = alg_kind;
        return desc_.alg_kind == alg_kind;
    }

private:
    const memory_desc_t *_src_md() const { return conv_prop_agnostic_src_d(&desc_); }
    const memory_desc_t *_wei_md() const { return conv_prop_agnostic_wei_d(&desc_); }
    const memory_desc_t *_bia_md() const { return conv_prop_agnostic_bia_d(&desc_); }
    const memory_desc_t *_dst_md() const { return conv_prop_agnostic_dst_d(&desc_); }
};

struct convolution_fwd_pd_t: public convolution_pd_t {
    typedef convolution_fwd_pd_t base_class;
    typedef convolution_fwd_pd_t hint_class;

    convolution_fwd_pd_t(engine_t *engine,
            const convolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : convolution_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , weights_md_(desc_.weights_desc)
        , bias_md_(desc_.bias_desc)
        , dst_md_(desc_.dst_desc)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (utils::one_of(arg, MKLDNN_ARG_SRC, MKLDNN_ARG_WEIGHTS))
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_BIAS && with_bias())
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DST)
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *src_md(int index = 0) const override
    { return index == 0 ? &src_md_ : nullptr; }
    virtual const memory_desc_t *dst_md(int index = 0) const override
    { return index == 0 ? &dst_md_ : nullptr; }
    virtual const memory_desc_t *weights_md(int index = 0) const override {
        if (index == 0) return &weights_md_;
        if (index == 1 && with_bias()) return &bias_md_;
        return nullptr;
    }

    virtual int n_inputs() const override { return 2 + with_bias(); }
    virtual int n_outputs() const override { return 1; }

protected:
    memory_desc_t src_md_;
    memory_desc_t weights_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_md_;
};

struct convolution_bwd_data_pd_t: public convolution_pd_t {
    typedef convolution_bwd_data_pd_t base_class;
    typedef convolution_fwd_pd_t hint_class;

    convolution_bwd_data_pd_t(engine_t *engine,
            const convolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : convolution_pd_t(engine, adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , weights_md_(desc_.weights_desc)
        , bias_md_(desc_.bias_desc)
        , diff_dst_md_(desc_.diff_dst_desc)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (utils::one_of(arg, MKLDNN_ARG_WEIGHTS, MKLDNN_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DIFF_SRC)
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *diff_src_md(int index = 0) const override
    { return index == 0 ? &diff_src_md_ : nullptr; }
    virtual const memory_desc_t *diff_dst_md(int index = 0) const override
    { return index == 0 ? &diff_dst_md_ : nullptr; }
    virtual const memory_desc_t *weights_md(int index = 0) const override {
        if (index == 0) return &weights_md_;
        if (index == 1 && with_bias()) return &bias_md_;
        return nullptr;
    }

    virtual int n_inputs() const override { return 2 + with_bias(); }
    virtual int n_outputs() const override { return 1; }

    virtual bool support_bias() const { return false; }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t weights_md_;
    memory_desc_t bias_md_;
    memory_desc_t diff_dst_md_;
};

struct convolution_bwd_weights_pd_t: public convolution_pd_t {
    typedef convolution_bwd_weights_pd_t base_class;
    typedef convolution_fwd_pd_t hint_class;

    convolution_bwd_weights_pd_t(engine_t *engine,
            const convolution_desc_t *adesc,
            const primitive_attr_t *attr,
            const convolution_fwd_pd_t *hint_fwd_pd)
        : convolution_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , diff_weights_md_(desc_.diff_weights_desc)
        , diff_bias_md_(desc_.diff_bias_desc)
        , diff_dst_md_(desc_.diff_dst_desc)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (utils::one_of(arg, MKLDNN_ARG_SRC, MKLDNN_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DIFF_WEIGHTS)
            return arg_usage_t::output;

        if (arg == MKLDNN_ARG_DIFF_BIAS && with_bias())
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *src_md(int index = 0) const override
    { return index == 0 ? &src_md_ : nullptr; }
    virtual const memory_desc_t *diff_dst_md(int index = 0) const override
    { return index == 0 ? &diff_dst_md_ : nullptr; }
    virtual const memory_desc_t *diff_weights_md(int index = 0) const override {
        if (index == 0) return &diff_weights_md_;
        if (index == 1 && with_bias()) return &diff_bias_md_;
        return nullptr;
    }

    virtual int n_inputs() const override { return 2; }
    virtual int n_outputs() const override { return 1 + with_bias(); }

protected:
    memory_desc_t src_md_;
    memory_desc_t diff_weights_md_;
    memory_desc_t diff_bias_md_;
    memory_desc_t diff_dst_md_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

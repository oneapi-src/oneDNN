/*******************************************************************************
* Copyright 2016-2022 Intel Corporation
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

#ifndef COMMON_SOFTMAX_PD_HPP
#define COMMON_SOFTMAX_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

namespace dnnl {
namespace impl {

struct softmax_fwd_pd_t;

struct softmax_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::softmax_v2;

    const softmax_v2_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::prop_kind:
                *(prop_kind_t *)result = desc()->prop_kind;
                break;
            case query::softmax_d:
                *(const softmax_desc_t **)result
                        = reinterpret_cast<const softmax_desc_t *>(desc());
                break;
            case query::logsoftmax_d:
                *(const logsoftmax_desc_t **)result
                        = reinterpret_cast<const logsoftmax_desc_t *>(desc());
                break;
            case query::softmax_v2_d:
                *(const softmax_v2_desc_t **)result = desc();
                break;
            case query::primitive_kind:
                if (desc()->primitive_kind == primitive_kind::softmax_v2)
                    *(primitive_kind_t *)result = desc()->primitive_kind;
                else
                    *(primitive_kind_t *)result = primitive_kind::softmax;
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common softmax aux functions */

    dim_t MB() const { return dst_desc().dims[0]; }
    dim_t C() const { return dst_desc().dims[1]; }
    dim_t D() const { return ndims() >= 5 ? dst_desc().dims[ndims() - 3] : 1; }
    dim_t H() const { return ndims() >= 4 ? dst_desc().dims[ndims() - 2] : 1; }
    dim_t W() const { return ndims() >= 3 ? dst_desc().dims[ndims() - 1] : 1; }

    dim_t outer_size() const {
        return utils::array_product(dst_desc().dims, axis());
    }
    dim_t axis_size(bool padded = false) const {
        return padded ? dst_desc().padded_dims[axis()]
                      : dst_desc().dims[axis()];
    }
    dim_t inner_size() const {
        return utils::array_product(
                dst_desc().dims + axis() + 1, ndims() - 1 - axis());
    }

    dim_t outer_stride() const {
        const memory_desc_wrapper dst_d(dst_desc());
        return axis() > 0 ? dst_d.blocking_desc().strides[axis() - 1] : 1;
    }

    int axis() const { return desc_.softmax_axis; }
    int ndims() const { return dst_desc().ndims; }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(dst_desc()).has_zero_dim();
    }

    alg_kind_t alg_kind() const { return desc()->alg_kind; }
    bool is_softmax() const { return alg_kind() == alg_kind::softmax_accurate; }
    bool is_logsoftmax() const { return alg_kind() == alg_kind::softmax_log; }

protected:
    softmax_v2_desc_t desc_;
    const softmax_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t dst_md_;

    softmax_pd_t(const softmax_v2_desc_t *adesc, const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(cast_softmax_v1_to_v2(*adesc))
        , hint_fwd_pd_(hint_fwd_pd)
        , dst_md_(desc_.dst_desc) {}

private:
    const memory_desc_t &dst_desc() const { return dst_md_; }

    softmax_v2_desc_t cast_softmax_v1_to_v2(
            const softmax_v2_desc_t &softmax_desc) const {
        if (softmax_desc.primitive_kind == primitive_kind::softmax_v2)
            return softmax_desc;

        softmax_v2_desc_t softmax_v2_desc;
        softmax_v2_desc.primitive_kind = softmax_desc.primitive_kind;
        softmax_v2_desc.prop_kind = softmax_desc.prop_kind;
        softmax_v2_desc.src_desc = softmax_desc.src_desc;
        softmax_v2_desc.diff_src_desc = softmax_desc.diff_src_desc;
        softmax_v2_desc.softmax_axis = softmax_desc.softmax_axis;
        softmax_v2_desc.alg_kind
                = softmax_desc.primitive_kind == primitive_kind::softmax
                ? alg_kind::softmax_accurate
                : alg_kind::softmax_log;
        softmax_v2_desc.dst_desc = softmax_desc.src_desc;
        softmax_v2_desc.diff_dst_desc = softmax_desc.diff_src_desc;

        return softmax_v2_desc;
    }
};

struct softmax_fwd_pd_t : public softmax_pd_t {
    typedef softmax_fwd_pd_t base_class;
    typedef softmax_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (arg == DNNL_ARG_SRC) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE && (!types::is_zero_md(workspace_md())))
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_DST: return dst_md(0);
            default: return softmax_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index = 0) const override {
        return index == 0 ? &src_md_ : &glob_zero_md;
    }
    const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &dst_md_ : &glob_zero_md;
    }

    int n_inputs() const override { return 1; }
    int n_outputs() const override {
        return 1 + (!types::is_zero_md(workspace_md()));
    }

protected:
    memory_desc_t src_md_;

    softmax_fwd_pd_t(const softmax_v2_desc_t *adesc,
            const primitive_attr_t *attr, const softmax_fwd_pd_t *hint_fwd_pd)
        : softmax_pd_t(adesc, attr, hint_fwd_pd), src_md_(desc_.src_desc) {}

    status_t set_default_formats() {
        if (dst_md()->format_kind != format_kind::any) return status::success;

        if (src_md()->format_kind != format_kind::blocked)
            return status::unimplemented;

        return memory_desc_init_by_blocking_desc(
                dst_md_, src_md_.format_desc.blocking);
    }

    bool attr_oscale_ok() const {
        const auto &oscale = attr()->output_scales_;
        const bool ok = IMPLICATION(desc()->primitive_kind != base_pkind,
                attr()->output_scales_.has_default_values());
        return ok && oscale.mask_ == 0;
    }
};

struct softmax_bwd_pd_t : public softmax_pd_t {
    typedef softmax_bwd_pd_t base_class;
    typedef softmax_fwd_pd_t hint_class;

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_DST, DNNL_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;

        if (arg == DNNL_ARG_WORKSPACE && (!types::is_zero_md(workspace_md())))
            return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_DST: return dst_md(0);
            case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
            case DNNL_ARG_DIFF_DST: return diff_dst_md(0);
            default: return softmax_pd_t::arg_md(arg);
        }
    }

    const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &dst_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_dst_md(int index = 0) const override {
        return index == 0 ? &diff_dst_md_ : &glob_zero_md;
    }
    const memory_desc_t *diff_src_md(int index = 0) const override {
        return index == 0 ? &diff_src_md_ : &glob_zero_md;
    }

    int n_inputs() const override {
        return 2 + (!types::is_zero_md(workspace_md()));
    }
    int n_outputs() const override { return 1; }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t diff_dst_md_;

    softmax_bwd_pd_t(const softmax_v2_desc_t *adesc,
            const primitive_attr_t *attr, const softmax_fwd_pd_t *hint_fwd_pd)
        : softmax_pd_t(adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , diff_dst_md_(desc_.diff_dst_desc) {}

    status_t set_default_formats() {
        status_t st = status::invalid_arguments;
        if (diff_dst_md_.format_kind == format_kind::any) {
            st = memory_desc_init_by_md_and_dt(
                    diff_dst_md_, dst_md_, diff_dst_md_.data_type);
            if (st != status::success) return st;
        }
        if (diff_src_md_.format_kind == format_kind::any) {
            st = memory_desc_init_by_md_and_dt(
                    diff_src_md_, diff_dst_md_, diff_src_md_.data_type);
            if (st != status::success) return st;
        }
        return status::success;
    }
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s

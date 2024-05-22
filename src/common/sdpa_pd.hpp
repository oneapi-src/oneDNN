/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef COMMON_SDPA_PD_HPP
#define COMMON_SDPA_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/primitive_desc.hpp"
#include "common/sdpa_utils.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define DNNL_ARG_QUERIES DNNL_ARG_SRC_0
#define DNNL_ARG_KEYS DNNL_ARG_SRC_1
#define DNNL_ARG_VALUES DNNL_ARG_SRC_2
#define DNNL_ARG_ATTN_MASK DNNL_ARG_SHIFT

#define VDISPATCH_SDPA(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, sdpa, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_SDPA_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, sdpa, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

struct sdpa_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::sdpa;

    typedef sdpa_pd_t base_class;
    typedef sdpa_pd_t hint_class;

    const sdpa_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_QUERIES, DNNL_ARG_KEYS, DNNL_ARG_VALUES,
                    DNNL_ARG_ATTN_MASK, DNNL_ARG_SCALE))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_QUERIES: return src_md(0);
            case DNNL_ARG_KEYS: return src_md(1);
            case DNNL_ARG_VALUES: return src_md(2);
            case DNNL_ARG_ATTN_MASK: return src_md(3);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        switch (index) {
            case 0: return &desc_.q_desc;
            case 1: return &desc_.k_desc;
            case 2: return &desc_.v_desc;
            case 3: return &desc_.attn_mask_desc;
            default: return &glob_zero_md;
        }
    }
    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &desc_.dst_desc : &glob_zero_md;
    }

    const memory_desc_t *qry_md() const { return &desc_.q_desc; }
    const memory_desc_t *key_md() const { return &desc_.k_desc; }
    const memory_desc_t *val_md() const { return &desc_.v_desc; }
    const memory_desc_t *attn_mask_md() const { return &desc_.attn_mask_desc; }

    int n_inputs() const override { return 3 + int(with_attn_mask()); }
    int n_outputs() const override { return 1; }

    bool with_attn_mask() const {
        return (attn_mask_md()->data_type != data_type::undef);
    }

protected:
    sdpa_desc_t desc_;

    sdpa_pd_t(const sdpa_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind), desc_(*adesc) {}

    bool set_default_format(memory_desc_t *md) {
        memory_desc_wrapper mdw(md);
        if (mdw.format_any()) return false;

        return true;
    }

    bool set_default_formats() {
        bool ok = true;

        for (auto md : {&desc_.q_desc, &desc_.k_desc, &desc_.v_desc,
                     &desc_.dst_desc}) {
            ok = ok && set_default_format(md);
        }

        auto status = attr_.post_ops_.set_default_formats(&desc_.dst_desc);
        ok = ok && (status == status::success);

        return ok;
    }
};

} // namespace impl
} // namespace dnnl

#endif

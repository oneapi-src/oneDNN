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

#ifndef COMMON_GATED_MLP_PD_HPP
#define COMMON_GATED_MLP_PD_HPP

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/primitive_desc.hpp"
#include "common/gated_mlp_utils.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

#define DNNL_ARG_SRC DNNL_ARG_SRC_0
#define DNNL_ARG_WTS_GATE DNNL_ARG_SRC_1
#define DNNL_ARG_WTS_UP DNNL_ARG_SRC_2
#define DNNL_ARG_WTS_DOWN DNNL_ARG_SRC_3

#define VDISPATCH_GATED_MLP(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, gated_mlp, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

#define VDISPATCH_GATED_MLP_SC(f, msg, ...) \
    VCHECK(primitive, create, dispatch, gated_mlp, (f), "%s," msg, \
            this->info(engine), ##__VA_ARGS__)

struct gated_mlp_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::gated_mlp;

    typedef gated_mlp_pd_t base_class;
    typedef gated_mlp_pd_t hint_class;

    const gated_mlp_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC, DNNL_ARG_WTS_GATE, DNNL_ARG_WTS_UP,
                    DNNL_ARG_WTS_DOWN)) //TODO: scale? zp?
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(
            int arg, bool user_input = false) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_WTS_GATE: return src_md(1);
            case DNNL_ARG_WTS_UP: return src_md(2);
            case DNNL_ARG_WTS_DOWN: return src_md(3);
            case DNNL_ARG_DST: return dst_md(0, user_input);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(
            int index = 0, bool user_input = false) const override {
        switch (index) {
            case 0: return &desc_.src_desc;
            case 1: return &desc_.W_gate_desc;
            case 2: return &desc_.W_up_desc;
            case 3: return &desc_.W_down_desc;
            default: return &glob_zero_md;
        }
    }

    const memory_desc_t *dst_md(
            int index = 0, bool user_input = false) const override {
        return index == 0 ? &desc_.dst_desc : &glob_zero_md;
    }

    const memory_desc_t *src0_md() const { return &desc_.src_desc; }
    const memory_desc_t *W_gate_md() const { return &desc_.W_gate_desc; }
    const memory_desc_t *W_up_md() const { return &desc_.W_up_desc; }
    const memory_desc_t *W_down_md() const { return &desc_.W_down_desc; }

    int n_inputs() const override {
        return 4 + int(0 /*with_attn_mask() TODO: add scale+zp?*/);
    }
    int n_outputs() const override { return 1; }

    /*TODO: w scale + zp?
    bool with_attn_scale() const {
        //return (desc_.scale_dt != data_type::undef);
        return false;
    }
    */

protected:
    gated_mlp_desc_t desc_;

    gated_mlp_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*op_desc_t::to_desc<gated_mlp_desc_t>(adesc)) {}

    bool set_default_format(memory_desc_t *md) {
        memory_desc_wrapper mdw(md);
        if (mdw.format_any()) return false;

        return true;
    }

    bool set_default_formats() {
        bool ok = true;
        for (auto md : {&desc_.src_desc, &desc_.W_gate_desc, &desc_.W_up_desc,
                     &desc_.W_down_desc, &desc_.dst_desc}) {
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

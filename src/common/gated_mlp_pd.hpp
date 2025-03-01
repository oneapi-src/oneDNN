/*******************************************************************************
* Copyright 2025 Intel Corporation
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
#include "common/gated_mlp_utils.hpp"
#include "common/primitive_desc.hpp"
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
                    DNNL_ARG_WTS_DOWN, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WTS_GATE,
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WTS_UP,
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_WTS_DOWN,
                    DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WTS_GATE,
                    DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WTS_UP,
                    DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WTS_DOWN))
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

    int n_inputs() const override { return 4; }
    int n_outputs() const override { return 1; }

    /// check scales enabled for each tensor
    /// If true, dequantize the wts_gate tensor using scaling
    bool with_wts_gate_scales() const {
        return (!desc()->wts_gate_scales.has_default_values());
    }

    /// If true, dequantize the wts_up tensor using scaling
    bool with_wts_up_scales() const {
        return (!desc()->wts_up_scales.has_default_values());
    }

    /// If true, dequantize the wts_down tensor using scaling
    bool with_wts_down_scales() const {
        return (!desc()->wts_down_scales.has_default_values());
    }

    /// check zero points enabled for each tensor
    /// If true, dequantize the wts_gate tensor with zero points
    bool with_wts_gate_zp() const {
        return (!desc()->wts_gate_zero_points.has_default_values(
                DNNL_ARG_WEIGHTS));
    }

    /// If true, dequantize the wts_up tensor with zero points
    bool with_wts_up_zp() const {
        return (!desc()->wts_up_zero_points.has_default_values(
                DNNL_ARG_WEIGHTS));
    }

    /// If true, dequantize the wts_down tensor with zero points
    bool with_wts_down_zp() const {
        return (!desc()->wts_down_zero_points.has_default_values(
                DNNL_ARG_WEIGHTS));
    }

    /// Scales data types for each tensor
    /// Returns the data type of the scales tensor for the wts_gate matmul
    data_type_t wts_gate_scales_dt() const {
        return desc()->wts_gate_scales.get_data_type();
    }

    /// Returns the data type of the scales tensor for the wts_up matmul
    data_type_t wts_up_scales_dt() const {
        return desc()->wts_up_scales.get_data_type();
    }

    /// Returns the data type of the scales tensor for the wts_down matmul
    data_type_t wts_down_scales_dt() const {
        return desc()->wts_down_scales.get_data_type();
    }

    /// Zero points data types for each tensor
    /// Returns the data type of the zero points tensor for the wts_gate matmul
    data_type_t wts_gate_zp_dt() const {
        return desc()->wts_gate_zero_points.get_data_type(DNNL_ARG_WEIGHTS);
    }

    /// Returns the data type of the zero points tensor for the wts_up matmul
    data_type_t wts_up_zp_dt() const {
        return desc()->wts_up_zero_points.get_data_type(DNNL_ARG_WEIGHTS);
    }

    /// Returns the data type of the zero points tensor for the wts_down matmul
    data_type_t wts_down_zp_dt() const {
        return desc()->wts_down_zero_points.get_data_type(DNNL_ARG_WEIGHTS);
    }

    // Returns the group size for the quantization parameters for the WTS_{up,gate} matmul
    int wts_gate_group_size() const {
        int out = 0;
        if (with_wts_gate_scales()) {
            out = scale_group_size(desc()->wts_gate_scales, *W_gate_md());
        } else if (with_wts_gate_zp()) {
            out = zp_group_size(desc()->wts_gate_zero_points, *W_gate_md());
        }
        return out;
    }

    // Returns the group size for the quantization parameters for the WTS_{up,gate} matmul
    int wts_up_group_size() const {
        int out = 0;
        if (with_wts_up_scales()) {
            out = scale_group_size(desc()->wts_up_scales, *W_up_md());
        } else if (with_wts_up_zp()) {
            out = zp_group_size(desc()->wts_up_zero_points, *W_up_md());
        }
        return out;
    }

    // Returns the group size for the quantization parameters for the WTS_{down} matmul
    int wts_down_group_size() const {
        int out = 0;
        if (with_wts_down_scales()) {
            out = scale_group_size(desc()->wts_down_scales, *W_down_md());
        } else if (with_wts_down_zp()) {
            out = zp_group_size(desc()->wts_down_zero_points, *W_down_md());
        }
        return out;
    }

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

private:
    static int scale_group_size(
            const quant_entry_t &scales, const memory_desc_t &desc) {
        dim_t out = utils::array_product(desc.dims, desc.ndims);
        if (scales.has_default_groups()) {
            for (int idx : mask_iterator(scales.get_mask())) {
                out /= desc.dims[idx];
            }
        } else {
            for (int idx : mask_iterator(scales.get_mask())) {
                out /= (desc.dims[idx] / scales.get_group(idx));
            }
        }
        return static_cast<int>(out);
    }

    static int zp_group_size(
            const zero_points_t &zp, const memory_desc_t &desc) {
        dim_t out = utils::array_product(desc.dims, desc.ndims);
        if (zp.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
            for (int idx : mask_iterator(zp.get_mask(DNNL_ARG_WEIGHTS))) {
                out /= desc.dims[idx];
            }
        } else {
            for (int idx : mask_iterator(zp.get_mask(DNNL_ARG_WEIGHTS))) {
                out /= (desc.dims[idx] / zp.get_group(DNNL_ARG_WEIGHTS, idx));
            }
        }
        return static_cast<int>(out);
    }
};

} // namespace impl
} // namespace dnnl

#endif

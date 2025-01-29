/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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
                    DNNL_ARG_ATTN_MASK, DNNL_ARG_SCALE,
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS,
                    DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES,
                    DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS,
                    DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES,
                    DNNL_ARG_PROMPT_LENS, DNNL_ARG_SUBSEQUENCE_BEGINS,
                    DNNL_ARG_BLOCK_INDICES, DNNL_ARG_BLOCK_INDICES_BEGINS))
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
            case DNNL_ARG_PROMPT_LENS: return src_md(4);
            case DNNL_ARG_SUBSEQUENCE_BEGINS: return src_md(5);
            case DNNL_ARG_BLOCK_INDICES: return src_md(6);
            case DNNL_ARG_BLOCK_INDICES_BEGINS: return src_md(7);
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
            case 4: return &desc_.prompt_lens_desc;
            case 5: return &desc_.subsequence_begins_desc;
            case 6: return &desc_.block_indices_desc;
            case 7: return &desc_.block_indices_begins_desc;
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

    const memory_desc_t *prompt_lens_md() const {
        return &desc_.prompt_lens_desc;
    }
    const memory_desc_t *subsequence_begins_md() const {
        return &desc_.subsequence_begins_desc;
    }
    const memory_desc_t *block_indices_md() const {
        return &desc_.block_indices_desc;
    }
    const memory_desc_t *block_indices_begins_md() const {
        return &desc_.block_indices_begins_desc;
    }

    int n_inputs() const override {
        return 7 + int(with_attn_mask()) + int(with_attn_scale());
    }
    int n_outputs() const override { return 1; }

    bool with_attn_scale() const {
        return (desc_.scale_dt != data_type::undef);
    }

    bool with_attn_mask() const {
        return (attn_mask_md()->data_type != data_type::undef);
    }

    /// If true, the attention mask is causal mask
    bool with_causal_mask() const { return desc_.causal_mask; }

    /// If true, dequantize the K tensor using scaling in the KQ matmul
    bool with_key_scales() const {
        return (!desc()->kq_scales.has_default_values());
    }

    /// If true, dequantize the V tensor using scaling in the VS matmul
    bool with_value_scales() const {
        return (!desc()->vs_scales.has_default_values());
    }

    /// If true, dequantize the K tensor with zero points in the KQ matmul
    bool with_key_zp() const {
        return (!desc()->kq_zero_points.has_default_values(DNNL_ARG_WEIGHTS));
    }

    /// If true, dequantize the V tensor with zero points in the VS matmul
    bool with_value_zp() const {
        return (!desc()->vs_zero_points.has_default_values(DNNL_ARG_WEIGHTS));
    }

    /// Returns the data type of the scales tensor for the KQ matmul
    data_type_t key_scales_dt() const {
        return desc()->kq_scales.get_data_type();
    }

    /// Returns the data type of the zero points tensor for the KQ matmul
    data_type_t key_zp_dt() const {
        return desc()->kq_zero_points.get_data_type(DNNL_ARG_WEIGHTS);
    }

    /// Returns the data type of the scales tensor for the VS matmul
    data_type_t value_scales_dt() const {
        return desc()->vs_scales.get_data_type();
    }

    /// Returns the data type of the zero points tensor for the VS matmul
    data_type_t value_zp_dt() const {
        return desc()->vs_zero_points.get_data_type(DNNL_ARG_WEIGHTS);
    }

    // Returns the group size for the quantization parameters for the KQ matmul
    int key_group_size() const {
        int out = 0;
        if (with_key_scales())
            out = scale_group_size(desc()->kq_scales, *key_md());
        else if (with_key_zp()) {
            out = zp_group_size(desc()->kq_zero_points, *key_md());
        }
        return out;
    }

    // Returns the group size for the quantization parameters for the VS matmul
    int value_group_size() const {
        int out = 0;
        if (with_value_scales())
            out = scale_group_size(desc()->vs_scales, *val_md());
        else if (with_value_zp()) {
            out = zp_group_size(desc()->vs_zero_points, *val_md());
        }
        return out;
    }

protected:
    sdpa_desc_t desc_;

    sdpa_pd_t(const op_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind)
        , desc_(*op_desc_t::to_desc<sdpa_desc_t>(adesc)) {}

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

private:
    static int scale_group_size(
            const quant_entry_t &scales, const memory_desc_t &desc) {
        dim_t out = utils::array_product(desc.dims, desc.ndims);
        const auto mask = scales.get_mask();
        if (scales.has_default_groups()) {
            for (int idx : mask_iterator(mask)) {
                out /= desc.dims[idx];
            }
        } else {
            for (int idx : mask_iterator(mask)) {
                if (idx < 2) {
                    out /= desc.dims[idx];
                } else {
                    out /= (desc.dims[idx] / scales.get_group(idx - 2));
                }
            }
        }
        return static_cast<int>(out);
    }

    static int zp_group_size(
            const zero_points_t &zp, const memory_desc_t &desc) {
        dim_t out = utils::array_product(desc.dims, desc.ndims);
        if (zp.has_default_groups(DNNL_ARG_WEIGHTS)) {
            for (int idx : mask_iterator(zp.get(DNNL_ARG_WEIGHTS))) {
                out /= desc.dims[idx];
            }
        } else {
            auto groups = zp.get_groups(DNNL_ARG_WEIGHTS);
            for (int idx : mask_iterator(zp.get(DNNL_ARG_WEIGHTS))) {
                if (idx < 2) {
                    out /= desc.dims[idx];
                } else {
                    out /= (desc.dims[idx] / groups[idx - 2]);
                }
            }
        }
        return static_cast<int>(out);
    }
};

} // namespace impl
} // namespace dnnl

#endif

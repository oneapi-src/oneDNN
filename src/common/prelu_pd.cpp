/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "prelu_pd.hpp"

namespace dnnl {
namespace impl {

void set_reduction_buffers(
        const dim_t work_amount, dim_t &group_size, dim_t &buf_size) {
    float sqrt = std::sqrt(work_amount);
    group_size = std::ceil(sqrt);
    buf_size = std::floor(sqrt);
    if (group_size * buf_size < work_amount) group_size++;
}

prelu_pd_t::prelu_pd_t(const prelu_desc_t *adesc, const primitive_attr_t *attr,
        const prelu_fwd_pd_t *hint_fwd_pd)
    : primitive_desc_t(attr, base_pkind)
    , desc_(*adesc)
    , hint_fwd_pd_(hint_fwd_pd)
    , data_md_(desc_.data_desc)
    , weights_md_(desc_.weights_desc) {}

const prelu_desc_t *prelu_pd_t::desc() const {
    return &desc_;
}

const op_desc_t *prelu_pd_t::op_desc() const {
    return reinterpret_cast<const op_desc_t *>(this->desc());
}

status_t prelu_pd_t::query(query_t what, int idx, void *result) const {
    switch (what) {
        case query::prop_kind:
            *(prop_kind_t *)result = desc()->prop_kind;
            break;
        case query::prelu_d: *(const prelu_desc_t **)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
    }
    return status::success;
}

dim_t prelu_pd_t::N() const {
    return data_desc().dims[0];
}
dim_t prelu_pd_t::C() const {
    return data_desc().dims[1];
}
dim_t prelu_pd_t::D() const {
    return ndims() >= 5 ? data_desc().dims[ndims() - 3] : 1;
}
dim_t prelu_pd_t::H() const {
    return ndims() >= 4 ? data_desc().dims[ndims() - 2] : 1;
}
dim_t prelu_pd_t::W() const {
    return ndims() >= 3 ? data_desc().dims[ndims() - 1] : 1;
}

int prelu_pd_t::ndims() const {
    return data_desc().ndims;
}

bool prelu_pd_t::has_zero_dim_memory() const {
    return memory_desc_wrapper(desc_.data_desc).has_zero_dim();
}

bool prelu_pd_t::is_fwd() const {
    return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
}

const memory_desc_t *prelu_pd_t::weights_md(int index) const {
    return index == 0 ? &weights_md_ : &glob_zero_md;
}

const memory_desc_t *prelu_pd_t::src_md(int index) const {
    return index == 0 ? &data_md_ : &glob_zero_md;
}

const memory_desc_t *prelu_pd_t::dst_md(int index) const {
    return index == 0 ? &data_md_ : &glob_zero_md;
}

size_t prelu_pd_t::dtype_size() const {
    return types::data_type_size(data_md_.data_type);
}

const memory_desc_t &prelu_pd_t::data_desc() const {
    return desc_.data_desc;
}

prelu_fwd_pd_t::prelu_fwd_pd_t(const prelu_desc_t *adesc,
        const primitive_attr_t *attr, const prelu_fwd_pd_t *hint_fwd_pd)
    : prelu_pd_t(adesc, attr, hint_fwd_pd) {}

primitive_desc_t::arg_usage_t prelu_fwd_pd_t::arg_usage(int arg) const {
    if (arg == DNNL_ARG_SRC) return arg_usage_t::input;
    if (arg == DNNL_ARG_WEIGHTS) return arg_usage_t::input;
    if (arg == DNNL_ARG_DST) return arg_usage_t::output;
    return primitive_desc_t::arg_usage(arg);
}

const memory_desc_t *prelu_fwd_pd_t::arg_md(int arg) const {
    switch (arg) {
        case DNNL_ARG_SRC: return src_md(0);
        case DNNL_ARG_WEIGHTS: return weights_md(0);
        case DNNL_ARG_DST: return dst_md(0);
        default: return prelu_pd_t::arg_md(arg);
    }
}

int prelu_fwd_pd_t::n_inputs() const {
    return 2;
}

int prelu_fwd_pd_t::n_outputs() const {
    return 1;
}

bool prelu_fwd_pd_t::set_default_formats() {
    if (weights_md_.format_kind == format_kind::any)
        if (memory_desc_init_by_blocking_desc(
                    weights_md_, data_md_.format_desc.blocking)
                != status::success)
            return false;
    return true;
}

prelu_bwd_pd_t::prelu_bwd_pd_t(const prelu_desc_t *adesc,
        const primitive_attr_t *attr, const prelu_fwd_pd_t *hint_fwd_pd)
    : prelu_pd_t(adesc, attr, hint_fwd_pd)
    , diff_data_md_(desc_.diff_data_desc)
    , diff_weights_md_(desc_.diff_weights_desc) {}

primitive_desc_t::arg_usage_t prelu_bwd_pd_t::arg_usage(int arg) const {
    if (arg == DNNL_ARG_SRC) return arg_usage_t::input;
    if (arg == DNNL_ARG_WEIGHTS) return arg_usage_t::input;
    if (arg == DNNL_ARG_DIFF_SRC) return arg_usage_t::output;
    if (arg == DNNL_ARG_DIFF_DST) return arg_usage_t::input;
    if (arg == DNNL_ARG_DIFF_WEIGHTS) return arg_usage_t::output;
    return primitive_desc_t::arg_usage(arg);
}

const memory_desc_t *prelu_bwd_pd_t::arg_md(int arg) const {
    switch (arg) {
        case DNNL_ARG_SRC: return src_md(0);
        case DNNL_ARG_WEIGHTS: return weights_md(0);
        case DNNL_ARG_DIFF_SRC: return diff_src_md(0);
        case DNNL_ARG_DIFF_DST: return diff_dst_md(0);
        case DNNL_ARG_DIFF_WEIGHTS: return diff_weights_md(0);
        default: return prelu_pd_t::arg_md(arg);
    }
}

const memory_desc_t *prelu_bwd_pd_t::diff_src_md(int index) const {
    return index == 0 ? &diff_data_md_ : &glob_zero_md;
}

const memory_desc_t *prelu_bwd_pd_t::diff_dst_md(int index) const {
    return index == 0 ? &diff_data_md_ : &glob_zero_md;
}

const memory_desc_t *prelu_bwd_pd_t::diff_weights_md(int index) const {
    return index == 0 ? &diff_weights_md_ : &glob_zero_md;
}

int prelu_bwd_pd_t::n_inputs() const {
    return 3;
}

int prelu_bwd_pd_t::n_outputs() const {
    return 2;
}

bool prelu_bwd_pd_t::set_default_formats() {
    if (weights_md_.format_kind == format_kind::any)
        if (memory_desc_init_by_blocking_desc(
                    weights_md_, data_md_.format_desc.blocking)
                != status::success)
            return false;
    if (diff_weights_md_.format_kind == format_kind::any)
        if (memory_desc_init_by_blocking_desc(
                    diff_weights_md_, data_md_.format_desc.blocking)
                != status::success)
            return false;
    return true;
}

} // namespace impl
} // namespace dnnl

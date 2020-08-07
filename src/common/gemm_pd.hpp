/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef COMMON_GEMM_PD_HPP
#define COMMON_GEMM_PD_HPP

#include "dnnl.h"

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/primitive_desc.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

struct gemm_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::gemm;

    typedef gemm_pd_t base_class;
    typedef gemm_pd_t hint_class;

    gemm_pd_t(const gemm_desc_t *adesc, const primitive_attr_t *attr,
            const hint_class *hint_fwd_pd)
        : primitive_desc_t(attr, base_pkind), desc_(*adesc) {}

    const gemm_desc_t *desc() const { return &desc_; }
    const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    arg_usage_t arg_usage(int arg) const override {
        if (utils::one_of(arg, DNNL_ARG_SRC_0, DNNL_ARG_SRC_1))
            return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC_0: return src_md(0);
            case DNNL_ARG_SRC_1: return src_md(1);
            case DNNL_ARG_DST: return dst_md(0);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    const memory_desc_t *src_md(int index = 0) const override {
        switch (index) {
            case 0: return &desc_.a_desc;
            case 1: return &desc_.b_desc;
            default: return &glob_zero_md;
        }
    }
    const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &desc_.c_desc : &glob_zero_md;
    }

    int n_inputs() const override { return 2; }
    int n_outputs() const override { return 1; }

private:

    gemm_desc_t desc_;
};

} // namespace impl
} // namespace dnnl

#endif

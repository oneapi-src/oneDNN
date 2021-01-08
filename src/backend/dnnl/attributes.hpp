/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_DNNL_ATTRIBUTES_HPP
#define BACKEND_DNNL_ATTRIBUTES_HPP

#include <tuple>
#include <utility>

#include "abstract_types.hpp"
#include "utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

using post_ops = dnnl::post_ops;

/// Attribute class for extra information into computations
struct attr_t : public dnnl::primitive_attr {
    attr_t() {}

    // Helper factory
    static attr_t fuse_sum(float scale = 1.0) {
        attr_t attr;
        post_ops po;
        po.append_sum(scale);
        attr.set_post_ops(po);
        return attr;
    }

    static attr_t fuse_eltwise(algorithm algo = algorithm::eltwise_relu,
            float scale = 1.0, float alpha = 0.f, float beta = 0.f) {
        attr_t attr;
        post_ops po;
        po.append_eltwise(scale, algo, alpha, beta);
        attr.set_post_ops(po);
        return attr;
    }

    static attr_t residual(algorithm algo = algorithm::eltwise_relu,
            float sum_scale = 1.0, float eltwise_scale = 1.0, float alpha = 0.f,
            float beta = 0.f) {
        attr_t attr;
        post_ops po;
        po.append_sum(sum_scale);
        po.append_eltwise(eltwise_scale, algo, alpha, beta);
        attr.set_post_ops(po);
        return attr;
    }

    static attr_t attr_post_ops(post_ops po) {
        attr_t attr;
        attr.set_post_ops(po);
        return attr;
    }

    bool has_op_kind(kind op_kind) const {
        auto po = get_post_ops();
        for (int i = 0; i < po.len(); i++)
            if (op_kind == po.kind(i)) return true;
        return false;
    }

    bool non_negitive_output() const {
        auto po = get_post_ops();
        auto last = po.len() - 1;
        if (last < 0) { return false; }

        auto params = get_params_(last);
        if (std::get<0>(params) != kind::eltwise || std::get<1>(params) <= 0.f
                || std::get<2>(params) != 0.f || std::get<3>(params) != 0.f
                || std::get<4>(params) != algorithm::eltwise_relu)
            return false;

        return true;
    }

private:
    std::tuple<kind, float, float, float, algorithm> get_params_(
            int index) const {
        auto po = get_post_ops();
        BACKEND_DNNL_ENFORCE(
                index < po.len(), "post_ops index is out of range");

        algorithm alg;
        float scale = 1.0, alpha = 1.0, beta = 0.0;

        auto akind = po.kind(index);
        switch (akind) {
            case kind::sum: po.get_params_sum(index, scale); break;
            case kind::eltwise:
                po.get_params_eltwise(index, scale, alg, alpha, beta);
                break;
            default:
                error::wrap_c_api(
                        dnnl_invalid_arguments, "could not get params");
                break;
        }

        return std::make_tuple(akind, scale, alpha, beta, alg);
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

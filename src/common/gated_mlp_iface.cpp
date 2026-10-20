/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "common/gated_mlp_iface.hpp"
#include "common/gated_mlp_pd.hpp"
#include "common/opdesc.hpp"
#include "common/primitive_desc_iface.hpp"

using namespace dnnl::impl;

status_t dnnl_gated_mlp_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *src_desc, const memory_desc_t *weights_gate_desc,
        const memory_desc_t *weights_up_desc,
        const memory_desc_t *weights_down_desc, const memory_desc_t *dst_desc,
        alg_kind_t activation, const primitive_attr_t *attr) {
    auto arg_md = [&](int arg) -> const memory_desc_t * {
        switch (arg) {
            default: return &glob_zero_md;
            case DNNL_ARG_DST: return dst_desc;
            case DNNL_ARG_SRC: return src_desc;
            case DNNL_ARG_WEIGHTS_GATE: return weights_gate_desc;
            case DNNL_ARG_WEIGHTS_UP: return weights_up_desc;
            case DNNL_ARG_WEIGHTS_DOWN: return weights_down_desc;
        }
    };
    auto dims_md = [&](int arg) {
        std::array<dim_t, DNNL_MAX_NDIMS> retn = {};
        auto md = arg_md(arg);
        switch (arg) {
            case DNNL_ARG_DST:
            case DNNL_ARG_SRC:
                if ((md->ndims < 3) || (md->dims[1] == 1))
                    retn = {md->dims[0], md->dims[md->ndims - 1]};
                break;
            case DNNL_ARG_WEIGHTS_GATE:
            case DNNL_ARG_WEIGHTS_UP:
            case DNNL_ARG_WEIGHTS_DOWN:
                if ((md->ndims < 3) || (md->dims[0] == 1))
                    retn = {md->dims[md->ndims - 2], md->dims[md->ndims - 1]};
                break;
        }
        return retn;
    };
    auto str_md = [](int arg) {
        switch (arg) {
            default: return "[unknown]";
            case DNNL_ARG_DST: return "dst";
            case DNNL_ARG_SRC: return "src";
            case DNNL_ARG_WEIGHTS_GATE: return "w_gate";
            case DNNL_ARG_WEIGHTS_UP: return "w_up";
            case DNNL_ARG_WEIGHTS_DOWN: return "w_down";
        }
    };
    VCHECK_GATED_MLP(
            utils::one_of(activation, alg_kind::eltwise_gelu_erf,
                    alg_kind::eltwise_gelu_tanh, alg_kind::eltwise_swish),
            "unsupported GMLP activation: %s", dnnl_alg_kind2str(activation));
    const auto ndims = arg_md(DNNL_ARG_DST)->ndims;
    VCHECK_GATED_MLP(utils::one_of(ndims, 2, 3), "invalid GMLP output dims");
    const auto &idxs = gated_mlp_pd_t::all_idxs();
    const auto dims = gated_mlp_pd_t::all_dims(dims_md(DNNL_ARG_SRC)[0],
            dims_md(DNNL_ARG_SRC)[1], dims_md(DNNL_ARG_WEIGHTS_GATE)[1]);
    for (int i = 0; i < int(dims.size()); i++) {
        const auto md = arg_md(idxs[i]);
        const auto dims_i = dims_md(idxs[i]);
        VCHECK_GATED_MLP(!memory_desc_wrapper(md).format_any(),
                "invalid GMLP %s format", str_md(idxs[i]));
        VCHECK_GATED_MLP((md->ndims == ndims) && dims_i[0] && dims_i[1]
                        && (dims_i[0] == dims[i][0])
                        && (dims_i[1] == dims[i][1]),
                "invalid GMLP %s dims", str_md(idxs[i]));
    }
    // Scales applied before zero-points (static_sbzp) are supported for
    // matmul weights decompression only.
    VCONDCHECK(primitive, create, check, gated_mlp,
            !attr->scales_.has_quantization_mode(
                    dnnl::impl::quantization_mode::static_sbzp)
                    && !attr->zero_points_.has_quantization_mode(
                            dnnl::impl::quantization_mode::static_sbzp),
            dnnl::impl::status::unimplemented, VERBOSE_UNSUPPORTED_ATTR);
    primitive_attr_t mlp_attr = *attr;
    auto po = mlp_attr.post_ops_.set_default_formats(arg_md(DNNL_ARG_DST));
    VCHECK_GATED_MLP(po == dnnl_success, "invalid GMLP post_ops formats");
    auto gated_mlp_desc
            = dnnl::impl::create_gated_mlp_desc(src_desc, weights_gate_desc,
                    weights_up_desc, weights_down_desc, dst_desc, activation);
    return dnnl::impl::primitive_desc_create(primitive_desc_iface, engine,
            (const dnnl::impl::op_desc_t *)&gated_mlp_desc, nullptr, &mlp_attr);
}

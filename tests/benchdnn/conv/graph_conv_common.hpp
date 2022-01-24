/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef GRAPH_CONV_COMMON_HPP
#define GRAPH_CONV_COMMON_HPP

#include <string>
#include <vector>

#include "conv/conv.hpp"
#include "dnnl_graph_common.hpp"

namespace benchdnnext {
namespace conv_common {

constexpr int CONV_3D_NDIMS = 5;
constexpr int CONV_2D_NDIMS = 4;
constexpr int CONV_1D_NDIMS = 3;
constexpr int CONV_MAX_NDIMS = CONV_3D_NDIMS;

struct spec_t {
    spec_t(const ::conv::prb_t *prb, bool is_deconv = false) noexcept {
        dir = prb->dir;
        groups = prb->has_groups ? (int64_t)prb->g : 1;
        has_groups = prb->has_groups;

        const dim_t src_1d_dims[] = {prb->mb, prb->ic, prb->iw};
        const dim_t src_2d_dims[] = {prb->mb, prb->ic, prb->ih, prb->iw};
        const dim_t src_3d_dims[]
                = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};

        const dim_t wei_1d_dims[]
                = {prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kw};
        const dim_t wei_2d_dims[] = {
                prb->g, prb->oc / prb->g, prb->ic / prb->g, prb->kh, prb->kw};
        const dim_t wei_3d_dims[] = {prb->g, prb->oc / prb->g, prb->ic / prb->g,
                prb->kd, prb->kh, prb->kw};

        bia_dims.assign({prb->oc});

        const dim_t dst_1d_dims[] = {prb->mb, prb->oc, prb->ow};
        const dim_t dst_2d_dims[] = {prb->mb, prb->oc, prb->oh, prb->ow};
        const dim_t dst_3d_dims[]
                = {prb->mb, prb->oc, prb->od, prb->oh, prb->ow};

        switch (prb->ndims) {
            case CONV_3D_NDIMS: {
                src_dims.assign(src_3d_dims, end(src_3d_dims));
                dst_dims.assign(dst_3d_dims, end(dst_3d_dims));

                wei_dims.assign(wei_3d_dims + (prb->has_groups ? 0 : 1),
                        end(wei_3d_dims));
            } break;

            case CONV_2D_NDIMS: {
                src_dims.assign(src_2d_dims, end(src_2d_dims));
                dst_dims.assign(dst_2d_dims, end(dst_2d_dims));

                wei_dims.assign(wei_2d_dims + (prb->has_groups ? 0 : 1),
                        end(wei_2d_dims));
            } break;

            case CONV_1D_NDIMS: {
                src_dims.assign(src_1d_dims, end(src_1d_dims));
                dst_dims.assign(dst_1d_dims, end(dst_1d_dims));

                wei_dims.assign(wei_1d_dims + (prb->has_groups ? 0 : 1),
                        end(wei_1d_dims));
            } break;

            default: break;
        }

        const dim_t strides_nd[] = {prb->sd, prb->sh, prb->sw};
        const dim_t dilates_nd[] = {prb->dd, prb->dh, prb->dw};
        const dim_t padding_nd[] = {prb->pd, prb->ph, prb->pw};
        const dim_t padding_r_nd[] = {prb->pd_r, prb->ph_r, prb->pw_r};

        const size_t spatial_offset = CONV_MAX_NDIMS - prb->ndims;
        strides.assign(strides_nd + spatial_offset, end(strides_nd));
        pads_begin.assign(padding_nd + spatial_offset, end(padding_nd));
        pads_end.assign(padding_r_nd + spatial_offset, end(padding_r_nd));
        dilations.assign(dilates_nd + spatial_offset, end(dilates_nd));
        std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                [](const dim_t d) { return d + 1; });

        src_dt = convert_dt(prb->cfg[SRC].dt);
        wei_dt = convert_dt(prb->cfg[WEI].dt);
        bia_dt = convert_dt(prb->cfg[BIA].dt);
        dst_dt = convert_dt(prb->cfg[DST].dt);

        data_format = "NCX";
        filter_format = "OIX";
        raw_src_tag = prb->stag;
        raw_wei_tag = prb->wtag;
        raw_dst_tag = prb->dtag;
    }

    dir_t dir;

    dims_t src_dims;
    dims_t wei_dims;
    dims_t bia_dims;
    dims_t dst_dims;

    dims_t strides;
    dims_t pads_begin;
    dims_t pads_end;
    dims_t dilations;

    std::string auto_pad {"None"};

    bool has_groups;
    int64_t groups;

    std::string data_format {"NCX"};
    std::string filter_format {"OIX"};
    std::string raw_src_tag;
    std::string raw_wei_tag;
    std::string raw_dst_tag;

    dt src_dt;
    dt wei_dt;
    dt bia_dt;
    dt dst_dt;
};

} // namespace conv_common
} // namespace benchdnnext

#endif

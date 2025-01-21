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

#include "gpu/intel/jit/ir/tensor_config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

void init_extra_tensors(const zero_points_config_t &zp_cfg,
        const primitive_attr_t &attr, const memory_desc_t *zp_src,
        const memory_desc_t &dst_md, dim_t ic, dim_t oc,
        tensor_config_t &tensor_cfg) {
    if (!attr.rounding_mode_.has_default_values()) {
        layout_t sround_seed_layout(type_t::u32(), 0, std::vector<dim_t> {1});
        tensor_cfg.add_tensor("sround_seed", DNNL_ARG_ATTR_ROUNDING_SEED,
                /*is_input=*/true, /*is_output=*/false, sround_seed_layout);
    }
    auto add_zp_buffer = [&](const std::string &name, type_t type, int arg_id,
                                 dim_t size) {
        layout_t zp_layout(type, 0, std::vector<dim_t> {size});
        tensor_cfg.add_tensor(name, DNNL_ARG_ATTR_ZERO_POINTS | arg_id,
                /*is_input=*/true, /*is_output=*/false, zp_layout);
    };
    if (zp_cfg.do_src_compensation && zp_cfg.is_runtime_src_zero_points) {
        if (zp_cfg.needs_src_conv_precalc) {
            gpu_assert(zp_src);
            int arg_key = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC;
            tensor_cfg.add_tensor("src_zero_points", arg_key, /*is_input=*/true,
                    /*is_output=*/false, layout_t(zp_src, false), layout_t());
        } else {
            add_zp_buffer("src_zero_points", zp_cfg.src_zp_type, DNNL_ARG_SRC,
                    (zp_cfg.is_common_src_zero_point) ? 1 : ic);
        }
    }
    if (zp_cfg.do_wei_compensation && zp_cfg.is_runtime_wei_zero_points) {
        gpu_assert(zp_cfg.is_common_wei_zero_point);
        add_zp_buffer(
                "wei_zero_points", zp_cfg.wei_zp_type, DNNL_ARG_WEIGHTS, 1);
    }
    if (zp_cfg.do_dst_compensation && zp_cfg.is_runtime_dst_zero_points) {
        add_zp_buffer("dst_zero_points", zp_cfg.dst_zp_type, DNNL_ARG_DST, oc);
    }
    auto scale_args = get_scale_args();
    for (int i = 0; i < (int)scale_args.size(); i++) {
        int arg = scale_args[i].second;
        if (attr.scales_.has_default_values(arg)) continue;
        std::vector<dim_t> dims = {(attr.scales_.get_mask(arg) == 0) ? 1 : oc};
        layout_t layout(type_t::f32(), 0, dims);
        int arg_key = DNNL_ARG_ATTR_SCALES | arg;
        tensor_cfg.add_tensor(scale_args[i].first, arg_key, /*is_input=*/true,
                /*is_output=*/false, layout);
    }
    for (int i = 0; i < attr.post_ops_.len(); i++) {
        auto &po = attr.post_ops_.entry_[i];
        if (po.is_eltwise()
                || po.is_sum(/*require_scale_one=*/false,
                        /*require_zp_zero=*/false)) {
            // No extra tensors.
        } else if (po.is_binary()) {
            auto layout = make_layout(po.binary.src1_desc);
            int arg_key = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1;
            tensor_cfg.add_tensor("binary_rhs_" + std::to_string(i), arg_key,
                    /*is_input=*/true,
                    /*is_output=*/false, layout);
        } else if (po.is_prelu()) {
            layout_t layout(type_t::f32(), 0,
                    get_prelu_weights_dims(po.prelu.mask, dst_md));
            int arg_key = DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_WEIGHTS;
            tensor_cfg.add_tensor("prelu_rhs_" + std::to_string(i), arg_key,
                    /*is_input=*/true, /*is_output=*/false, layout);
        } else {
            gpu_error_not_expected();
        }
    }
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

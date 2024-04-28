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
    if (zp_cfg.do_src_compensation && zp_cfg.is_runtime_src_zero_points) {
        int arg_key = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC;
        if (!zp_cfg.needs_src_precalc) {
            std::vector<dim_t> dim {(zp_cfg.is_common_src_zero_point) ? 1 : ic};
            layout_t zp_layout(type_t::s32(), 0, dim);
            tensor_cfg.add_tensor("src_zero_points", arg_key,
                    /*is_input=*/true, /*is_output=*/false, zp_layout);
        } else {
            ir_assert(zp_src);
            tensor_cfg.add_tensor("src_zero_points", arg_key, /*is_input=*/true,
                    /*is_output=*/false, layout_t(zp_src, false), layout_t());
        }
    }
    if (zp_cfg.do_dst_compensation && zp_cfg.is_runtime_dst_zero_points) {
        std::vector<dim_t> dims = {oc};
        layout_t zp_layout(type_t::s32(), 0, dims);
        int arg_key = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST;
        tensor_cfg.add_tensor("dst_zero_points", arg_key,
                /*is_input=*/true, /*is_output=*/false, zp_layout);
    }
    auto scale_args = get_scale_args();
    for (int i = 0; i < (int)scale_args.size(); i++) {
        int arg = scale_args[i].second;
        auto &s = attr.scales_.get(arg);
        if (s.has_default_values()) continue;
        std::vector<dim_t> dims = {(s.mask_ == 0) ? 1 : oc};
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
            ir_error_not_expected();
        }
    }
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

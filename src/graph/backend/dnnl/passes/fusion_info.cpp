/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/value.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/passes/fusion_info.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

dnnl::primitive_attr make_dnnl_primitive_attr(
        const std::shared_ptr<op_t> &op, const fusion_info_t &fusion_info) {
    dnnl::primitive_attr attr;

    // convert output scales
    if (fusion_info.output_scales_) {
        const op_t *oscales_op = fusion_info.output_scales_->get_op();
        auto oscales
                = oscales_op->get_attr<std::vector<float>>(op_attr::scales);
        int oscales_mask = oscales.size() == 1
                ? 0
                : 1 << oscales_op->get_attr<int64_t>(op_attr::axis);
        attr.set_output_scales(oscales_mask, oscales);
    }

    // convert input zps
    if (!fusion_info.input_zps_.empty()) {
        for (const auto &in_zps : fusion_info.input_zps_) {
            size_t in_zps_indice = in_zps.first;
            const op_t *in_zps_op = in_zps.second->get_op();
            auto zps = in_zps_op->get_attr<std::vector<int64_t>>(op_attr::zps);
            int mask = zps.size() == 1
                    ? 0
                    : 1 << in_zps_op->get_attr<int64_t>(op_attr::axis);
            std::vector<int32_t> neg_int32_zps = dnnl_impl::utils::fmap(
                    zps, [](int64_t zp) { return static_cast<int32_t>(-zp); });
            attr.set_zero_points(
                    in_zps_indice == 0 ? DNNL_ARG_SRC : DNNL_ARG_WEIGHTS, mask,
                    neg_int32_zps);
        }
    }

    // convert output zps
    if (fusion_info.output_zps_) {
        const op_t *out_zps_op = fusion_info.output_zps_->get_op();
        auto zps = out_zps_op->get_attr<std::vector<int64_t>>(op_attr::zps);
        int mask = zps.size() == 1
                ? 0
                : 1 << out_zps_op->get_attr<int64_t>(op_attr::axis);
        std::vector<int32_t> int32_zps = dnnl_impl::utils::fmap(
                zps, [](int64_t zp) { return static_cast<int32_t>(zp); });
        attr.set_zero_points(DNNL_ARG_DST, mask, int32_zps);
    }

    // convert post ops
    dnnl::post_ops dnnl_pops;
    for (auto &pop : fusion_info.get_post_ops()) {
        const op_t *fused_op = pop->get_op();
        const auto fused_op_kind = fused_op->get_kind();
        if (fused_op_kind == op_kind::dnnl_eltwise) {
            float scale = pop->get_scale();
            float alpha = 0.f;
            float beta = 0.f;
            if (fused_op->has_attr(op_attr::alpha)) {
                alpha = fused_op->get_attr<float>(op_attr::alpha);
            }
            if (fused_op->has_attr(op_attr::beta)) {
                beta = fused_op->get_attr<float>(op_attr::beta);
            }
            const auto alg = static_cast<dnnl::algorithm>(
                    fused_op->get_attr<int64_t>(op_attr::alg_kind));
            dnnl_pops.append_eltwise(scale, alg, alpha, beta);
        } else if (fused_op_kind == op_kind::dnnl_binary) {
            const auto &extra_inputs = pop->get_unfused_input_indices();
            if (extra_inputs.empty()) {
                // post-sum
                float scale = pop->get_scale();
                int32_t zp = pop->get_zp();
                dnnl_pops.append_sum(scale, zp);
            } else {
                // post-binary
                assertm(extra_inputs.size() == 1,
                        "post-binary only has 1 extra input");
                size_t src1_idx = extra_inputs[0];
                auto input = op->get_input_value(src1_idx);
                auto md = make_dnnl_memory_desc(input->get_logical_tensor());
                const auto alg = static_cast<dnnl::algorithm>(
                        fused_op->get_attr<int64_t>(op_attr::alg_kind));
                dnnl_pops.append_binary(alg, md);
            }
        } else if (fused_op_kind == op_kind::dnnl_convolution) {
            const auto &extra_input_indices = pop->get_unfused_input_indices();
            assertm(extra_input_indices.size() == 1,
                    "post-conv dosen't support extra bias inputs now");

            auto get_dnn_dt = [](const std::shared_ptr<value_t> &val) {
                using ltw = logical_tensor_wrapper_t;
                auto graph_dt = ltw(val->get_logical_tensor()).data_type();
                return static_cast<dnnl::memory::data_type>(graph_dt);
            };

            const size_t wei_idx = extra_input_indices[0];
            const auto wei_dt = get_dnn_dt(op->get_input_value(wei_idx));
            const auto dst_dt = get_dnn_dt(op->get_output_value(0));
            const auto bia_dt = dnnl::memory::data_type::undef;
            const int mask = 0;
            const bool is_k3s1p1 = fused_op->get_attr<std::vector<int64_t>>(
                                           op_attr::strides)[0]
                    == 1;
            if (is_k3s1p1) {
                dnnl_pops.append_dw_k3s1p1(wei_dt, bia_dt, dst_dt, mask, {});
            } else {
                dnnl_pops.append_dw_k3s2p1(wei_dt, bia_dt, dst_dt, mask, {});
            }
        } else {
            // not reachable
        }
    }
    attr.set_post_ops(dnnl_pops);

    return attr;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include "interface/c_types_map.hpp"
#include "interface/op.hpp"
#include "interface/value.hpp"
#include "utils/utils.hpp"

#include "backend/dnnl/fusion_info.hpp"
#include "backend/dnnl/internal_attrs.hpp"
#include "backend/dnnl/internal_ops.hpp"
#include "backend/dnnl/utils.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

dnnl::primitive_attr make_dnnl_primitive_attr(
        const std::shared_ptr<impl::op_t> &op,
        const fusion_info_t &fusion_info) {
    dnnl::primitive_attr attr;

    // convert output scales
    if (fusion_info.output_scales_) {
        const impl::op_t *oscales_op = fusion_info.output_scales_->get_op();
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
            const impl::op_t *in_zps_op = in_zps.second->get_op();
            auto zps = in_zps_op->get_attr<std::vector<int64_t>>(op_attr::zps);
            int mask = zps.size() == 1
                    ? 0
                    : 1 << in_zps_op->get_attr<int64_t>(op_attr::axis);
            std::vector<int32_t> int32_zps = utils::cast_to_int32(zps);
            attr.set_zero_points(
                    in_zps_indice == 0 ? DNNL_ARG_SRC : DNNL_ARG_WEIGHTS, mask,
                    int32_zps);
        }
    }

    // convert output zps
    if (fusion_info.output_zps_) {
        const impl::op_t *out_zps_op = fusion_info.output_zps_->get_op();
        auto zps = out_zps_op->get_attr<std::vector<int64_t>>(op_attr::zps);
        int mask = zps.size() == 1
                ? 0
                : 1 << out_zps_op->get_attr<int64_t>(op_attr::axis);
        std::vector<int32_t> int32_zps = utils::cast_to_int32(zps);
        attr.set_zero_points(DNNL_ARG_DST, mask, int32_zps);
    }

    // convert post ops
    dnnl::post_ops dnnl_pops;
    for (auto &pop : fusion_info.get_post_ops()) {
        const impl::op_t *fused_op = pop->get_op();
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
            const auto alg = static_cast<dnnl::algorithm>(
                    fused_op->get_attr<int64_t>(op_attr::alg_kind));
            const auto &extra_inputs = pop->get_unfused_input_indices();
            float scale = pop->get_scale();
            int32_t zp = pop->get_zp();
            const auto psrc = op->get_input_value(extra_inputs[0])
                                      ->get_logical_tensor();
            const auto dst = op->get_output_value(0)->get_logical_tensor();
            // check if can use post-sum, otherwise use binary post ops
            // algorithm should be binary_add
            bool is_post_sum = alg == dnnl::algorithm::binary_add;
            // base_op should not be eltwise or pool
            is_post_sum = is_post_sum
                    && !impl::utils::one_of(op->get_kind(),
                            op_kind::dnnl_eltwise, op_kind::dnnl_pool);
            // only support one post-sum
            is_post_sum = is_post_sum
                    && !(op->has_attr(op_attr::with_sum)
                            && op->get_attr<bool>(op_attr::with_sum));
            // post src and dst should have the same shape
            is_post_sum = is_post_sum
                    && impl::logical_tensor_wrapper_t(dst).vdims()
                            == impl::logical_tensor_wrapper_t(psrc).vdims();
            // dst should have equal or larger memory size than post src
            is_post_sum = is_post_sum
                    && (psrc.data_type == dst.data_type
                            || impl::utils::one_of(psrc.data_type,
                                    impl::data_type::u8, impl::data_type::s8));
            if (is_post_sum) {
                pop->set_post_sum();
                op->set_attr<bool>(op_attr::with_sum, true);
                dnnl::memory::data_type sum_dt = dnnl::memory::data_type::undef;
                if (psrc.data_type == impl::data_type::s8
                        && dst.data_type == impl::data_type::u8) {
                    sum_dt = dnnl::memory::data_type::s8;
                }
                dnnl_pops.append_sum(scale, zp, sum_dt);
            } else {
                // post-binary
                assertm(extra_inputs.size() == 1,
                        "post-binary only has 1 extra input");
                assertm(scale == 1.f && zp == 0,
                        "post-binary doesn't support input scale and zp");
                auto md = make_dnnl_memory_desc(psrc);
                dnnl_pops.append_binary(alg, md);
            }
        } else if (fused_op_kind == op_kind::dnnl_convolution) {
            const auto &extra_input_indices = pop->get_unfused_input_indices();
            assertm(extra_input_indices.size() == 1,
                    "post-conv dosen't support extra bias inputs now");

            auto get_dnn_dt = [](const std::shared_ptr<impl::value_t> &val) {
                using ltw = impl::logical_tensor_wrapper_t;
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
} // namespace impl
} // namespace graph
} // namespace dnnl

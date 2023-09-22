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

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/value.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/fusion_info.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"
#include "graph/backend/dnnl/internal_ops.hpp"
#include "graph/backend/dnnl/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

dnnl::primitive_attr make_dnnl_primitive_attr(
        const std::shared_ptr<op_t> &op, const fusion_info_t &fusion_info) {
    dnnl::primitive_attr attr;

    if (fusion_info.dst_scales_) {
        const op_t *dst_scales_op = fusion_info.dst_scales_->get_op();
        assertm(fusion_info.with_runtime_scales(false, 0),
                "only support runtime dst scales.\n");
        int mask = 0;
        if (dst_scales_op->has_attr(op_attr::axis)
                && dst_scales_op->has_attr(op_attr::qtype)) {
            int64_t axis = dst_scales_op->get_attr<int64_t>(op_attr::axis);
            std::string qtype
                    = dst_scales_op->get_attr<std::string>(op_attr::qtype);
            mask = qtype == "per_tensor" ? 0 : 1 << axis;
        }
        attr.set_scales_mask(DNNL_ARG_DST, mask);
    }

    // convert input scales
    if (!fusion_info.input_scales_.empty()) {
        for (const auto &in_scales : fusion_info.input_scales_) {
            size_t in_scales_indices = in_scales.first;
            const op_t *in_scales_op = in_scales.second->get_op();
            assertm(fusion_info.with_runtime_scales(true, in_scales_indices),
                    "only support runtime src scales.\n");
            int mask = 0;
            if (in_scales_op->has_attr(op_attr::axis)
                    && in_scales_op->has_attr(op_attr::qtype)) {
                int64_t axis = in_scales_op->get_attr<int64_t>(op_attr::axis);
                std::string qtype
                        = in_scales_op->get_attr<std::string>(op_attr::qtype);
                if (qtype == "per_tensor") {
                    mask = 0;
                } else {
                    if (impl::utils::one_of(op->get_kind(),
                                op_kind::dnnl_convolution,
                                op_kind::dnnl_convtranspose)
                            && in_scales_indices == 1) {
                        bool with_groups = false;
                        if (op->get_input_value(1)->has_producer()
                                && op->get_input_op(1)->get_kind()
                                        == op_kind::dnnl_to_group) {
                            const auto &to_group = op->get_input_op(1);
                            if (to_group->get_attr<int64_t>(op_attr::groups)
                                    > 1) {
                                with_groups = true;
                            }
                        }
                        mask = with_groups ? 3 : 1;
                    } else {
                        mask = 1 << axis;
                    }
                }
            }
            attr.set_scales_mask(
                    in_scales_indices == 0 ? DNNL_ARG_SRC : DNNL_ARG_WEIGHTS,
                    mask);
        }
    }

    // convert input zps
    if (!fusion_info.input_zps_.empty()) {
        for (const auto &in_zps : fusion_info.input_zps_) {
            size_t in_zps_indices = in_zps.first;
            assertm(fusion_info.with_runtime_zero_points(true, in_zps_indices),
                    "only support runtime src zero points.\n");
            int mask = 0;
            attr.set_zero_points_mask(
                    in_zps_indices == 0 ? DNNL_ARG_SRC : DNNL_ARG_WEIGHTS,
                    mask);
        }
    }

    // convert output zps
    if (fusion_info.output_zps_) {
        assertm(fusion_info.with_runtime_zero_points(false, 0),
                "only support runtime src zero points.\n");
        int mask = 0;
        attr.set_zero_points_mask(DNNL_ARG_DST, mask);
    }

    // convert post ops
    dnnl::post_ops dnnl_pops;
    for (auto &pop : fusion_info.get_post_ops()) {
        const op_t *fused_op = pop->get_op();
        const auto fused_op_kind = fused_op->get_kind();
        if (fused_op_kind == op_kind::dnnl_eltwise) {
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
            dnnl_pops.append_eltwise(alg, alpha, beta);
        } else if (fused_op_kind == op_kind::dnnl_binary) {
            const auto alg = static_cast<dnnl::algorithm>(
                    fused_op->get_attr<int64_t>(op_attr::alg_kind));
            const auto &extra_inputs = pop->get_unfused_input_indices();
            float scale = pop->get_scale();
            int32_t zp = pop->get_zp();
            const auto psrc_val = op->get_input_value(extra_inputs[0]);
            const auto psrc = psrc_val->get_logical_tensor();
            const auto dst = op->get_output_value(0)->get_logical_tensor();
            // check if can use post-sum, otherwise use binary post ops
            // algorithm should be binary_add
            bool is_post_sum = alg == dnnl::algorithm::binary_add;
            // base_op should not be eltwise, pool, or softmax.
            is_post_sum = is_post_sum
                    && !impl::utils::one_of(op->get_kind(),
                            op_kind::dnnl_eltwise, op_kind::dnnl_pool,
                            op_kind::dnnl_softmax, op_kind::dnnl_logsoftmax);
            // only support one post-sum
            is_post_sum = is_post_sum
                    && !(op->has_attr(op_attr::with_sum)
                            && op->get_attr<bool>(op_attr::with_sum));
            // post src and dst should have the same shape
            is_post_sum = is_post_sum
                    && logical_tensor_wrapper_t(dst).vdims()
                            == logical_tensor_wrapper_t(psrc).vdims();
            // dst should have equal or larger memory size than post src
            is_post_sum = is_post_sum
                    && (psrc.data_type == dst.data_type
                            || impl::utils::one_of(psrc.data_type,
                                    impl::data_type::u8, impl::data_type::s8));
            // post src should not have alias. Here the condition of alias_ins == 1
            // is to disable the inplace option for src = main_op(src) + src
            const auto get_external_id
                    = [](const std::shared_ptr<value_t> &val) {
                          auto tmp_val = val;
                          while (tmp_val->has_producer()) {
                              size_t lt_id = tmp_val->get_logical_tensor().id;
                              // check if lt_id is already a external id
                              if (lt_id != std::numeric_limits<size_t>::max())
                                  return lt_id;

                              const op_t &prod_op = tmp_val->get_producer();
                              // ops like Dnnl_constant_scales doesn't have external input
                              // return a internal id
                              if (prod_op.num_inputs() == 0) return lt_id;
                              tmp_val = prod_op.get_input_value(0);
                          }
                          return tmp_val->get_logical_tensor().id;
                      };
            size_t alias_ins = 0;
            size_t psrc_lt_id = get_external_id(psrc_val);
            for (size_t op_in_idx = 0; op_in_idx < op->num_inputs();
                    ++op_in_idx) {
                size_t op_in_lt_id
                        = get_external_id(op->get_input_value(op_in_idx));
                if (op_in_lt_id == psrc_lt_id) alias_ins++;
            }
            is_post_sum = is_post_sum && alias_ins == 1;
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
                    "post-conv doesn't support extra bias inputs now");

            auto get_dnn_dt = [](const std::shared_ptr<value_t> &val) {
                using ltw = logical_tensor_wrapper_t;
                auto graph_dt = ltw(val->get_logical_tensor()).data_type();
                return static_cast<dnnl::memory::data_type>(graph_dt);
            };

            const size_t wei_idx = extra_input_indices[0];
            auto wei_value = op->get_input_value(wei_idx);
            const auto wei_dt = get_dnn_dt(wei_value);
            const auto dst_dt = get_dnn_dt(op->get_output_value(0));
            const auto bia_dt = dnnl::memory::data_type::undef;
            const int64_t ks = wei_value->get_logical_tensor().dims[3];
            const int64_t stride = fused_op->get_attr<std::vector<int64_t>>(
                    op_attr::strides)[0];
            const int64_t pad_l = fused_op->get_attr<std::vector<int64_t>>(
                    op_attr::pads_begin)[0];
            dnnl_pops.append_dw(wei_dt, bia_dt, dst_dt, ks, stride, pad_l);
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

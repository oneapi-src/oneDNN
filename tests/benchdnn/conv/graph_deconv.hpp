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

#ifndef GRAPH_DECONV_HPP
#define GRAPH_DECONV_HPP

#include <vector>

#include "conv/deconv.hpp"
#include "conv/graph_conv_common.hpp"
#include "dnnl_graph_common.hpp"

namespace benchdnnext {

using namespace conv_common;

namespace deconv {

struct deconv_graph_prb_t : public graph_prb_t {
    deconv_graph_prb_t(const ::conv::prb_t *prb) : spec_(prb, true) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;

        if (prb->dir == FWD_B) {
            has_post_bia_ = true;
            ctor_status = handle_bia_();
            if (stop_work(ctor_status)) return;
        }

        if (benchdnnext::is_low_precision({spec_.src_dt, spec_.dst_dt}))
            // needs to be set before call of post-op handlers
            with_quantization_ = true;

        const std::vector<attr_t::post_ops_t::entry_t> &po_entry
                = prb->attr.post_ops.entry;

        for (const attr_t::post_ops_t::entry_t &po : po_entry) {
            if (po.is_eltwise_kind()) {
                has_post_eltwise_ = true;
                ctor_status = handle_elt_(po);
                if (stop_work(ctor_status)) return;
            } else if (po.is_sum_kind()) {
                has_post_sum_ = true;
                ctor_status = handle_sum_();
                if (stop_work(ctor_status)) return;
            } else if (po.is_binary_kind()) {
                has_post_bin_ = true;
                ctor_status = handle_bin_(po);
                if (stop_work(ctor_status)) return;
            }
        }

        if (with_quantization()) {
            ctor_status = handle_low_precision_(prb);
            if (stop_work(ctor_status)) return;
        }

        ctor_status = fill_status::DONE;
    };

    const struct spec_t &spec() const noexcept { return spec_; }
    std::vector<float> &get_oscales() noexcept { return oscales; }

private:
    const std::vector<dt> dt_constraints {dt::bf16, dt::f16, dt::f32};

    std::vector<float> oscales;
    std::vector<int64_t> src_zero_points;
    std::vector<int64_t> wei_zero_points;
    std::vector<int64_t> dst_zero_points;

    spec_t spec_;
    po_handlers_t po_handler;

    fill_status_t handle_main_op_();
    fill_status_t handle_bia_();
    fill_status_t handle_sum_();
    fill_status_t handle_bin_(const attr_t::post_ops_t::entry_t &po);
    fill_status_t handle_elt_(const attr_t::post_ops_t::entry_t &po);
    fill_status_t handle_low_precision_(const ::conv::prb_t *prb);

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return dnnl::graph::op::kind::ConvTranspose;
    }
};

int doit(const ::conv::prb_t *prb, res_t *res);

} // namespace deconv
} // namespace benchdnnext

#endif

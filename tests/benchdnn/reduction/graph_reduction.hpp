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

#ifndef GRAPH_REDUCTION_HPP
#define GRAPH_REDUCTION_HPP

#include "dnnl_graph_common.hpp"

#include "reduction/reduction.hpp"

namespace benchdnnext {
namespace reduction {

struct reduction_graph_prb_t : public graph_prb_t {
    reduction_graph_prb_t(const ::reduction::prb_t *prb) : spec_(prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;

        for (const auto &po : prb->attr.post_ops.entry) {
            if (po.is_eltwise_kind()) {
                ctor_status = handle_elt_(po);
                if (stop_work(ctor_status)) return;
            } else if (po.is_binary_kind()) {
                has_post_bin_ = true;
                ctor_status = handle_bin_(po);
                if (stop_work(ctor_status)) return;
            } else if (po.is_sum_kind()) {
                has_post_sum_ = true;
                ctor_status = handle_sum_();
                if (stop_work(ctor_status)) return;
            }
        }

        ctor_status = fill_status::DONE;
    };

private:
    struct spec_t {
        spec_t(const ::reduction::prb_t *prb);
        // in oneDNN we always set reduction dimensions to 1,
        // therefore keep_dims will default to true
        bool keep_dims {true};
        std::vector<int64_t> axes {};

        dims_t src_dims;
        dims_t dst_dims;

        dt src_dt;
        dt dst_dt;

        std::string raw_src_tag;
        std::string raw_dst_tag;

        dnnl::graph::op::kind alg;
    };

    spec_t spec_;
    po_handlers_t po_handler;

    fill_status_t handle_main_op_();
    fill_status_t handle_bin_(const attr_t::post_ops_t::entry_t &po);
    fill_status_t handle_elt_(const attr_t::post_ops_t::entry_t &po);
    fill_status_t handle_sum_();

    dnnl::graph::op::kind get_main_op_kind() const override {
        return spec_.alg;
    }
};

int doit(const ::reduction::prb_t *prb, res_t *res);

} // namespace reduction
} // namespace benchdnnext

#endif

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
    reduction_graph_prb_t(const ::reduction::prb_t *prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        op_kind = convert_alg_kind(::reduction::alg2alg_kind(prb->alg));

        ctor_status = handle_main_op_(prb);
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
    dnnl::graph::op::kind op_kind;
    po_handlers_t po_handler;

    fill_status_t handle_main_op_(const ::reduction::prb_t *prb);
    fill_status_t handle_bin_(const attr_t::post_ops_t::entry_t &po);
    fill_status_t handle_elt_(const attr_t::post_ops_t::entry_t &po);
    fill_status_t handle_sum_();

    dnnl::graph::op::kind get_main_op_kind() const override { return op_kind; }
};

int doit(const ::reduction::prb_t *prb, res_t *res);

} // namespace reduction
} // namespace benchdnnext

#endif

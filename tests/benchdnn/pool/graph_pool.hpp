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

#ifndef GRAPH_POOL_HPP
#define GRAPH_POOL_HPP

#include "dnn_graph_types.hpp"
#include "dnnl_graph_common.hpp"
#include "pool/pool.hpp"

namespace benchdnnext {
namespace pool {

struct pool_graph_prb_t : public graph_prb_t {
    pool_graph_prb_t(const ::pool::prb_t *prb) {
        using graph_op = dnnl::graph::op::kind;

        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        if (prb->dir & FLAG_FWD) {
            op_kind = (prb->alg == ::pool::max) ? graph_op::MaxPool
                                                : graph_op::AvgPool;
        } else {
            op_kind = (prb->alg == ::pool::max) ? graph_op::MaxPoolBackprop
                                                : graph_op::AvgPoolBackprop;
        }

        ctor_status = handle_main_op_(prb);
        if (stop_work(ctor_status)) return;

        auto dtypes
                = {convert_dt(prb->cfg[SRC].dt), convert_dt(prb->cfg[DST].dt)};
        if (benchdnnext::is_low_precision(dtypes))
            // needs to be set before call of post-op handlers
            with_quantization_ = true;

        for (const auto &po : prb->attr.post_ops.entry) {
            if (po.is_binary_kind()) {
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

private:
    dnnl::graph::op::kind op_kind {dnnl::graph::op::kind::LastSymbol};
    po_handlers_t po_handler;

    fill_status_t handle_main_op_(const ::pool::prb_t *prb);
    fill_status_t handle_low_precision_(const ::pool::prb_t *prb_);
    fill_status_t handle_bin_(const attr_t::post_ops_t::entry_t &po_entry);

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return op_kind;
    }
};

int doit(const ::pool::prb_t *prb, res_t *res);

} // namespace pool
} // namespace benchdnnext

#endif

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

#ifndef GRAPH_SOFTMAX_HPP
#define GRAPH_SOFTMAX_HPP

#include "dnnl_graph_common.hpp"
#include "softmax/softmax.hpp"

namespace benchdnnext {
namespace softmax {

struct softmax_graph_prb_t : public graph_prb_t {
    softmax_graph_prb_t(const ::softmax::prb_t *prb) {
        using graph_op = dnnl::graph::op::kind;

        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        switch (prb->alg) {
            case ::softmax::SOFTMAX:
                op_kind = prb->dir & FLAG_FWD ? graph_op::SoftMax
                                              : graph_op::SoftMaxBackprop;
                break;
            case ::softmax::LOGSOFTMAX:
                op_kind = prb->dir & FLAG_FWD ? graph_op::LogSoftmax
                                              : graph_op::LogSoftmaxBackprop;
                break;
            default: op_kind = graph_op::LastSymbol;
        }
        if (op_kind == graph_op::LastSymbol) {
            ctor_status = fill_status::UNSUPPORTED_OP;
            return;
        }

        ctor_status = handle_main_op_(prb);
        if (stop_work(ctor_status)) return;

        ctor_status = fill_status::DONE;
    };

private:
    dnnl::graph::op::kind op_kind {dnnl::graph::op::kind::LastSymbol};

    fill_status_t handle_main_op_(const ::softmax::prb_t *prb);
    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return op_kind;
    }
};

int doit(const ::softmax::prb_t *prb, res_t *res);

} // namespace softmax
} // namespace benchdnnext

#endif

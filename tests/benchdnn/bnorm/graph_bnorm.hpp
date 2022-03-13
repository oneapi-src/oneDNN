/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef GRAPH_BNOM_HPP
#define GRAPH_BNOM_HPP

#include "bnorm/bnorm.hpp"
#include "dnnl_graph_common.hpp"

namespace benchdnnext {
namespace bnorm {

struct bnorm_graph_prb_t : public graph_prb_t {
    bnorm_graph_prb_t(const ::bnorm::prb_t *prb) : spec_(prb) {
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
            }
        }

        ctor_status = fill_status::DONE;
    };

private:
    struct spec_t {
        spec_t(const ::bnorm::prb_t *prb) noexcept;
        float epsilon {0.00001f};

        dims_t dims;
        dims_t s_dims;
        dt bnorm_dt;
        std::string tag;
        dir_t dir;
    };

    spec_t spec_;
    po_handlers_t po_handler;

    fill_status_t handle_main_op_();
    fill_status_t handle_elt_(const attr_t::post_ops_t::entry_t &po_entry);

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return dnnl::graph::op::kind::BatchNormInference;
    }
};

int doit(const ::bnorm::prb_t *prb, res_t *res);

} // namespace bnorm
} // namespace benchdnnext

#endif

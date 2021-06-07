/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GRAPH_CONV_HPP
#define GRAPH_CONV_HPP

#include <vector>

#include "conv.hpp"

#include "dnnl_graph_common.hpp"

namespace benchdnnext {
namespace conv {

struct conv_graph_prb_t : public graph_prb_t {
    struct spec_t {
        spec_t(const ::conv::prb_t *prb);

        dims_t src_dim;
        dims_t wei_dim;
        dims_t bia_dim;
        dims_t dst_dim;

        dims_t strides;
        dims_t pads_begin;
        dims_t pads_end;
        dims_t dilations;

        std::string auto_pad {"None"};

        int64_t groups;

        std::string data_format {"NCX"};
        std::string filter_format {"OIX"};

        dt src_dtype;
        dt wei_dtype;
        dt bia_dtype;
        dt dst_dtype;
    };

    conv_graph_prb_t(const ::conv::prb_t *prb) : prb(prb), spec_(prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;
        if (prb->dir == FWD_B) {
            ctor_status = handle_bia_();
            if (stop_work(ctor_status)) return;
        }

        const std::vector<attr_t::post_ops_t::entry_t> &po_entry
                = prb->attr.post_ops.entry;

        for (attr_t::post_ops_t::entry_t po : po_entry) {
            if (po.is_eltwise_kind()) {
                ctor_status = handle_elt_(po);
                if (stop_work(ctor_status)) return;
            } else if (po.is_sum_kind()) {
                has_post_sum_ = true;
                ctor_status = handle_sum_();
                if (stop_work(ctor_status)) return;
            }
        }

        if (stop_work(ctor_status)) return;

        ctor_status = fill_status::DONE;
    };

    bool has_post_sum() const { return has_post_sum_; }
    const spec_t spec() const { return spec_; }

    std::vector<float> oscales;
    fill_status_t ctor_status;

private:
    const ::conv::prb_t *prb;
    spec_t spec_;
    po_handlers_t po_handler;

    bool has_post_sum_ {false};

    fill_status_t handle_main_op_();
    fill_status_t handle_bia_();
    fill_status_t handle_elt_(const attr_t::post_ops_t::entry_t &po);
    fill_status_t handle_sum_();
};

int doit(const ::conv::prb_t *prb, res_t *res);

} // namespace conv
} // namespace benchdnnext

#endif

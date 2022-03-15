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
    pool_graph_prb_t(const ::pool::prb_t *prb) : spec_(prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;

        auto dtypes = {spec_.src_dt, spec_.dst_dt};
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
    struct spec_t {
        spec_t(const ::pool::prb_t *prb) noexcept;

        dims_t strides;
        dims_t kernel;
        dims_t pads_begin;
        dims_t pads_end;
        std::string rounding_type;
        std::string data_format;
        std::string raw_data_format;
        // auto_pad will always be set to "None" due to benchdnn CLI limitations
        std::string auto_pad {"None"};

        // attributes specific to pooling type
        bool exclude_pad; // AvgPool
        dims_t dilations; // MaxPool

        dims_t src_dims;
        dims_t dst_dims;

        dt src_dt;
        dt dst_dt;

        dnnl::graph::op::kind op_kind;
        std::string op_name;
        bool is_fwd;
    };
    spec_t spec_;
    po_handlers_t po_handler;

    fill_status_t handle_main_op_();
    fill_status_t handle_low_precision_(const ::pool::prb_t *prb_);
    fill_status_t handle_bin_(const attr_t::post_ops_t::entry_t &po_entry);

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return spec_.op_kind;
    }
};

int doit(const ::pool::prb_t *prb, res_t *res);

} // namespace pool
} // namespace benchdnnext

#endif

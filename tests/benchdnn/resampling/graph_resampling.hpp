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

#ifndef GRAPH_RESAMPLING_HPP
#define GRAPH_RESAMPLING_HPP

#include "dnnl_graph_common.hpp"
#include "resampling/resampling.hpp"

namespace benchdnnext {
namespace resampling {

enum test_mode_t { SIZES_ATTR = 0, SCALES_ATTR, SIZES_INPUT_TENSOR };

struct resampling_graph_prb_t : public graph_prb_t {
    resampling_graph_prb_t(const ::resampling::prb_t *prb) : spec_(prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };
        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;
        for (const auto &po : prb->attr.post_ops.entry) {
            if (po.is_eltwise_kind()) {
                has_post_eltwise_ = true;
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
        spec_t(const ::resampling::prb_t *prb) noexcept;
        bool is_fwd_pass {true};
        dnnl::graph::op::kind op_kind;

        dims_t src_dims;
        dims_t dst_dims;

        dt src_dt;
        dt dst_dt;
        std::string mode, tag;
        std::vector<int64_t> sizes {0};
        std::vector<float> scales {0};
        std::string data_format = "NCX";
        //To program resampling sizes and scales
        //0 - sizes in input tensor
        //1 - sizes in attributes
        //2 - scales in attributes not supported for now in Graph
        int rand_testmode;
    };

    spec_t spec_;
    po_handlers_t po_handler;
    fill_status_t handle_main_op_();
    fill_status_t handle_sum_();
    fill_status_t handle_elt_(const attr_t::post_ops_t::entry_t &po_entry);
    fill_status_t handle_bin_(const attr_t::post_ops_t::entry_t &po_entry);
    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return spec_.op_kind;
    }

public:
    const spec_t &spec() const noexcept { return spec_; }
};

int doit(const ::resampling::prb_t *prb, res_t *res);

} // namespace resampling
} // namespace benchdnnext

#endif

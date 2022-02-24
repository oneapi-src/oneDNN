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

#ifndef GRAPH_CONCAT_HPP
#define GRAPH_CONCAT_HPP

#include "concat/concat.hpp"

#include "dnnl_graph_common.hpp"

namespace benchdnnext {
namespace concat {

inline int64_t permute_axis(const int64_t raw_axis, const dims_t &src0_dims,
        const std::string &raw_tag) {
    auto rank = src0_dims.size();

    const std::string dnnl_fmt_tag_str
            = normalize_tag(raw_tag, static_cast<int>(rank));
    const dnnl_format_tag_t fmt_tag = dnnl_fmt_str2tag(dnnl_fmt_tag_str);
    if (fmt_tag == dnnl_format_tag_undef) {
        []() {
            SAFE(FAIL, CRIT);
            return 0;
        }();
        return 0;
    }

    if (fmt_tag == dnnl_format_tag_any) { return raw_axis; }

    // Calculate axis based on a tag.
    const std::string ou_fmt_str = get_ou_format(dnnl_fmt_tag_str);
    for (int64_t d = 0; d < static_cast<int64_t>(rank); ++d) {
        const size_t coord = static_cast<size_t>(ou_fmt_str[d] - 'a');
        // If we permute the dimension with axis. We should also change the axis.
        if (coord == static_cast<size_t>(raw_axis)) { return d; }
    }
    return raw_axis;
}

struct concat_graph_prb_t : public graph_prb_t {
    concat_graph_prb_t(const ::concat::prb_t *prb) : spec_(prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;

        if (benchdnnext::is_low_precision({spec_.src_dt, spec_.dst_dt})) {
            ctor_status = handle_low_precision_(prb);
            if (stop_work(ctor_status)) return;
        }

        ctor_status = fill_status::DONE;
    };

private:
    struct spec_t {
        spec_t(const ::concat::prb_t *prb);

        int64_t axis {1};

        std::vector<dims_t> src_dims;
        dims_t dst_dims;

        dt src_dt;
        dt dst_dt;

        std::vector<std::string> raw_src_tag;
        std::string raw_dst_tag;

        int64_t n_inputs() const { return static_cast<int>(src_dims.size()); }
    };

    spec_t spec_;
    po_handlers_t po_handler;

    fill_status_t handle_main_op_();
    fill_status_t handle_low_precision_(const ::concat::prb_t *prb_);

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return dnnl::graph::op::kind::Concat;
    }

public:
    const spec_t &spec() const noexcept { return spec_; }
};

int doit(const ::concat::prb_t *prb, res_t *res);

} // namespace concat
} // namespace benchdnnext

#endif // GRAPH_CONCAT_HPP

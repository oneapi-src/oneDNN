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

#ifndef GRAPH_LNORM_HPP
#define GRAPH_LNORM_HPP

#include "dnnl_graph_common.hpp"
#include "lnorm/lnorm.hpp"

namespace benchdnnext {
namespace lnorm {

struct lnorm_graph_prb_t : public graph_prb_t {
    lnorm_graph_prb_t(const ::lnorm::prb_t *prb) : spec_(prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;

        ctor_status = fill_status::DONE;
    };

private:
    struct spec_t {
        spec_t(const ::lnorm::prb_t *prb) noexcept;
        bool is_fwd_pass {true};
        dnnl::graph::op::kind op_kind;

        bool keep_stats {true};
        int64_t begin_norm_axis {-1};
        bool use_affine {true};
        float epsilon {0.00001f};

        dims_t dims;
        dims_t stat_dims;
        dims_t ss_dims;
        dt lnorm_dt;
    };

    spec_t spec_;

    fill_status_t handle_main_op_();

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return spec_.op_kind;
    }

public:
    const spec_t &spec() const noexcept { return spec_; }
};

int doit(const ::lnorm::prb_t *prb, res_t *res);

} // namespace lnorm
} // namespace benchdnnext

#endif

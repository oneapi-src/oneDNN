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

#ifndef GRAPH_DECONV_HPP
#define GRAPH_DECONV_HPP

#include <vector>

#include "conv/deconv.hpp"
#include "conv/graph_conv_common.hpp"
#include "dnnl_graph_common.hpp"

namespace benchdnnext {

using namespace conv_common;

namespace deconv {

struct deconv_graph_prb_t : public graph_prb_t {
    deconv_graph_prb_t(const ::conv::prb_t *prb) : spec_(prb, true) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;

        ctor_status = fill_status::DONE;
    };

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return dnnl::graph::op::kind::ConvTranspose;
    }
    const struct spec_t spec() const noexcept { return spec_; }

    fill_status_t ctor_status;

private:
    spec_t spec_;

    fill_status_t handle_main_op_();
};

int doit(const ::conv::prb_t *prb, res_t *res);

} // namespace deconv
} // namespace benchdnnext

#endif

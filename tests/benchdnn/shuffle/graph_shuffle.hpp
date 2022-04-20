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

#ifndef GRAPH_SHUFFLE_HPP
#define GRAPH_SHUFFLE_HPP

#include "dnnl_graph_common.hpp"
#include "shuffle/shuffle.hpp"

namespace benchdnnext {
namespace shuffle {

struct shuffle_graph_prb_t : public graph_prb_t {
    shuffle_graph_prb_t(const ::shuffle::prb_t *prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_main_op_(prb);
        if (stop_work(ctor_status)) return;

        ctor_status = fill_status::DONE;
    };

private:
    fill_status_t handle_main_op_(const ::shuffle::prb_t *prb);

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return dnnl::graph::op::kind::StaticReshape;
    }
};

int doit(const ::shuffle::prb_t *prb, res_t *res);

} // namespace shuffle
} // namespace benchdnnext
#endif

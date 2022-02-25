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
    shuffle_graph_prb_t(const ::shuffle::prb_t *prb) : spec_(prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        ctor_status = handle_reshape_(0);
        if (stop_work(ctor_status)) return;

        ctor_status = handle_transpose_();
        if (stop_work(ctor_status)) return;

        ctor_status = handle_reshape_(1);
        if (stop_work(ctor_status)) return;

        ctor_status = fill_status::DONE;
    };

private:
    struct spec_t {
        spec_t(const ::shuffle::prb_t *prb);

        dims_t reshape0_src_dims;
        dims_t reshape0_dst_dims;

        dims_t transpose_dst_dims;
        dims_t transpose_order;

        dims_t reshape1_dst_dims;

        std::string raw_tag;
        dt dtype;
        int64_t group;
        int axis;
    };

    spec_t spec_;

    fill_status_t handle_reshape_(int id);
    fill_status_t handle_transpose_();

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return dnnl::graph::op::kind::StaticReshape;
    }
};

int doit(const ::shuffle::prb_t *prb, res_t *res);

} // namespace shuffle
} // namespace benchdnnext
#endif

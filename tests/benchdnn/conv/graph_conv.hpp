/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include "conv/conv.hpp"
#include "conv/graph_conv_dw_fusion.hpp"
#include "dnnl_graph_common.hpp"

namespace benchdnnext {
namespace conv {

fill_status_t append_graph_with_block(const ::conv::prb_t *prb);
int doit(const ::conv::prb_t *prb, res_t *res);

} // namespace conv
} // namespace benchdnnext

#endif

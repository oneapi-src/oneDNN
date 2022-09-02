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

#include "graph/interface/partition_impl.hpp"
#include "graph/interface/backend.hpp"
#include "graph/interface/graph.hpp"
#include "graph/interface/op_schema.hpp"

namespace dnnl {
namespace impl {
namespace graph {
status_t compiled_partition_impl_t::query_logical_tensor(
        size_t tid, logical_tensor_t *lt) const {
    auto pos_in = std::find_if(inputs_.begin(), inputs_.end(),
            [&](const logical_tensor_t &in_) -> bool { return in_.id == tid; });
    if (pos_in != inputs_.end()) {
        *lt = *pos_in;
        return status::success;
    }

    auto pos_out = std::find_if(outputs_.begin(), outputs_.end(),
            [&](const logical_tensor_t &out_) -> bool {
                return out_.id == tid;
            });
    if (pos_out != outputs_.end()) {
        *lt = *pos_out;
        return status::success;
    }

    // if we don't find the logical tensor in compiled partition's inputs_ and
    // outputs_, this means the logical tensor is not required by this compiled
    // partition. this will be a common situation if FWK gives arbitrary
    // connection, and shouldn't be regarded as an error
    *lt = empty_logical_tensor_with_default_id();
    return status::success;
}

} // namespace graph
} // namespace impl
} // namespace dnnl

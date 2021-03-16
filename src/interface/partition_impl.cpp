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

#include "partition_impl.hpp"
#include "backend.hpp"
#include "graph.hpp"
#include "op_schema.hpp"

namespace dnnl {
namespace graph {
namespace impl {
partition_impl_t::partition_impl_t(const partition_impl_t &other)
    : std::enable_shared_from_this<partition_impl_t>(other)
    , engine_kind_(other.engine_kind_)
    , ops_(impl::graph_t::deep_copy(other.ops_))
    , inputs_(other.inputs_)
    , outputs_(other.outputs_) {}

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

    // if we don't find the logical tensor in compiled partition's inputs_
    // and outputs_, this means the logical tensor is not required by this
    // compiled partition. this will be a common situation if FWK gives
    // arbitrary connection, and shouldn't be regarded as an error
    std::memset(lt, 0, sizeof(logical_tensor_t));
    return status::success;
}

} // namespace impl
} // namespace graph
} // namespace dnnl

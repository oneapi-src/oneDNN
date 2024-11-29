/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

// Common for RNN and LSTM cell execution

#include "gpu/generic/sycl/rnn/ref_rnn.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

using namespace dnnl::impl::utils;
using namespace rnn_utils;

status_t _ref_rnn_common_t::cell_execution(const cell_ctx_t &cell_struct) {

    auto cell_layer = cell_struct.workspace.states_range(cell_struct.lay - 1,
            cell_struct.lay - 1, cell_struct.dir, cell_struct.dir,
            cell_struct.iter - 1, cell_struct.iter);

    auto cell_iter = cell_struct.workspace.states_range(cell_struct.lay,
            cell_struct.lay, cell_struct.dir, cell_struct.dir,
            cell_struct.iter - 2, cell_struct.iter - 1);

    auto scratch_gates = cell_struct.scratch.gates(0);

    auto wei_layer
            = cell_struct.user_data.wei_layer(cell_struct.lay, cell_struct.dir);
    auto wei_iter
            = cell_struct.user_data.wei_iter(cell_struct.lay, cell_struct.dir);

    CHECK(gemm_primitive(cell_struct.engine, cell_struct.ctx, wei_layer,
            cell_layer, scratch_gates, gemm_layer_fwd));

    CHECK(gemm_primitive(cell_struct.engine, cell_struct.ctx, wei_iter,
            cell_iter, scratch_gates, gemm_iter_fwd));

    CHECK(rnn_bias(cell_struct.ctx, cell_struct.rnn.mb, cell_struct.rnn.dhc,
            cell_struct.iter, cell_struct.lay, cell_struct.dir,
            cell_struct.workspace, cell_struct.scratch, cell_struct.user_data));

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

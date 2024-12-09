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

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution)) {
    const conf_t &rnn = this->pd()->rnn_conf;

    auto cell_layer = workspace.states_range(
            lay - 1, lay - 1, dir, dir, iter - 1, iter - 1);

    auto cell_iter
            = workspace.states_range(lay, lay, dir, dir, iter - 2, iter - 2);

    auto scratch_gates = scratch.gates(0);

    auto wei_layer = user_data.wei_layer(lay, dir);
    auto wei_iter = user_data.wei_iter(lay, dir);

    CHECK(gemm_primitive(
            engine, ctx, wei_layer, cell_layer, scratch_gates, gemm_layer_fwd));

    CHECK(gemm_primitive(
            engine, ctx, wei_iter, cell_iter, scratch_gates, gemm_iter_fwd));

    CHECK(rnn_bias(ctx, rnn.mb, rnn.dhc, iter, lay, dir, workspace, scratch,
            user_data));

    return status::success;
}
template cell_execution_sig(ref_rnn_fwd_t::cell_execution);
} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

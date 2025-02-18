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

#include "gpu/generic/sycl/rnn/rnn_utils.hpp"

#include "common/c_types_map.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

using namespace dnnl::impl::utils;
using namespace prop_kind;
using namespace data_type;

void rnn_utils::init_rnn_conf(
        conf_t &rnn, const rnn_pd_t *rnn_pd, data_type_t acc_data_t) {

    rnn = utils::zero<decltype(rnn)>();
    rnn.is_fwd = utils::one_of(rnn_pd->desc()->prop_kind,
            prop_kind::forward_training, prop_kind::forward_inference);
    rnn.is_training = utils::one_of(rnn_pd->desc()->prop_kind,
            prop_kind::forward_training, prop_kind::backward);

    rnn.aux_data_type
            = acc_data_t == data_type::f16 ? data_type::f16 : data_type::f32;

    rnn.acc_data_type = acc_data_t;

    rnn.wei_layer_type = rnn_pd->weights_md(0)->data_type;
    rnn.wei_iter_type = rnn_pd->weights_md(1)->data_type;

    rnn.n_layer = rnn_pd->weights_md(0)->dims[0];
    rnn.n_iter = rnn_pd->src_md(0)->dims[0];
    rnn.n_dir = rnn_pd->weights_md(0)->dims[1];
    rnn.n_gates = rnn_pd->weights_md(0)->dims[3];
    rnn.n_states = rnn_pd->desc()->cell_kind == dnnl_vanilla_lstm ? 2 : 1;
    rnn.n_bias = rnn.n_gates + 1;
    rnn.mb = rnn_pd->src_md(0)->dims[1];
    rnn.sic = rnn_pd->weights_md(1)->dims[2];
    rnn.slc = rnn_pd->weights_md(0)->dims[2];
    rnn.dhc = rnn_pd->weights_md(0)->dims[4];
    rnn.dlc = rnn_pd->dst_md(0)->dims[2];

    rnn.gates_ld = rnn.dhc * rnn.n_gates;

    rnn.n_parts_bias = 1;
    rnn.parts_bias[0] = rnn.n_bias;
    rnn.parts_bias[1] = 0;
    rnn.iter_loop = 1;

    rnn.use_workspace = rnn.is_training;

    rnn.src_data_type = rnn_pd->src_md(0)->data_type;
    rnn.input_data_type = rnn_pd->src_md(1)->data_type;
    rnn.bias_data_type = rnn_pd->weights_md(2)->data_type;
    rnn.dst_data_type = rnn_pd->dst_md(0)->data_type;
    rnn.output_data_type = rnn_pd->dst_md(1)->data_type;

    // Assign types for optional parameters for improved kernel reuse.
    if (rnn.input_data_type == data_type::undef)
        rnn.input_data_type = rnn.src_data_type;
    if (rnn.output_data_type == data_type::undef)
        rnn.output_data_type = rnn.dst_data_type;
}

void rnn_utils::set_rnn_conf(conf_t &rnn, const rnn_desc_t &rd) {

    const bool is_fwd = rnn.is_fwd;

    dim_t aux_elsz
            = static_cast<dim_t>(types::data_type_size(rnn.aux_data_type));
    rnn.ws_states_elsz = types::data_type_size(rnn.src_data_type);

    rnn.scratch_gates_elsz = types::data_type_size(rnn.acc_data_type);

    // Set workspace sizes to store:
    // states to compute a pass
    // intermediate results from the gates
    rnn.states_ws_ld = nstl::max(rnn.slc, nstl::max(rnn.sic, rnn.dhc));
    rnn.gates_ws_ld = rnn.gates_ld;
    rnn.scratch_gates_ld = rnn.gates_ld;

    rnn.ws_states_cell_size = rnn.mb * rnn.states_ws_ld * rnn.ws_states_elsz;
    rnn.ws_states_size = (rnn.n_layer + 1) * rnn.n_dir * (rnn.n_iter + 1)
            * rnn.ws_states_cell_size;

    rnn.ws_gates_cell_size = rnn.mb * rnn.gates_ws_ld * aux_elsz;
    rnn.ws_gates_size = rnn.ws_gates_cell_size;
    rnn.scratch_gates_size
            = rnn.mb * rnn.scratch_gates_ld * rnn.scratch_gates_elsz;

    rnn.ws_bias_size
            = rnn.n_layer * rnn.n_dir * rnn.n_bias * rnn.dhc * aux_elsz;

    // For intermediate step in post-gemm fwd lbr gru
    rnn.scratch_cell_size = [&]() {
        if (is_fwd) {
            return rnn.mb * rnn.scratch_gates_ld * rnn.scratch_gates_elsz;
        } else {
            return static_cast<dim_t>(0);
        }
    }();

    // Used for storing the intermediate value from fwd pass in training lbr gru
    rnn.ws_per_cell = rnn.mb * rnn.dhc * aux_elsz;

    set_workspace_offsets(rnn, rnn.ws_gates_offset, rnn.ws_states_offset);
}

dim_t rnn_utils::set_workspace_offsets(
        const conf_t &rnn, dim_t &ws_gates_offset, dim_t &ws_states_offset) {

    const dim_t page_size = 4096;
    dim_t current_offset = 0;

#define register_space(a) \
    do { \
        current_offset = utils::rnd_up(current_offset, page_size); \
        CONCAT2(a, _offset) = current_offset; \
        current_offset += rnn.CONCAT2(a, _size); \
    } while (false)

    // Mandatory workspaces: go to workspace if use_workspace, scratchpad
    // otherwise assumes the workspace base pointer is page aligned
    register_space(ws_states);
    register_space(ws_gates);

    return current_offset;
}

dim_t rnn_utils::get_workspace_size(const conf_t &rnn) {
    dim_t ws_gates_offset, ws_states_offset;
    return set_workspace_offsets(rnn, ws_gates_offset, ws_states_offset);
}

status_t rnn_utils::set_good_strides(
        memory_desc_t &weights_md, format_tag_t tag) {
    auto &strides = weights_md.format_desc.blocking.strides;
    auto dims = weights_md.dims;
    using namespace format_tag;

    if (tag == ldigo) {
        strides[1] = dims[2] * strides[2];
        strides[0] = dims[1] * strides[1];
    } else if (tag == ldgoi) {
        strides[3] = dims[4] * strides[4];
        strides[1] = dims[3] * strides[3];
        strides[0] = dims[1] * strides[1];
    } else
        return status::unimplemented;

    return status::success;
}

status_t rnn_utils::set_weights_desc(
        memory_desc_t &weights_md, const conf_t &rnn) {
    using namespace format_tag;
    if (weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_md, rnn.is_fwd ? ldigo : ldgoi));

        // Adjust strides for good leading dimension in GEMM
        CHECK(set_good_strides(weights_md, rnn.is_fwd ? ldigo : ldgoi));

        return status::success;
    } else if (weights_md.format_kind != format_kind::blocked) {
        // This implementation only supports blocked memory
        return status::unimplemented;
    }
    return status::success;
}

const memory_storage_t &rnn_utils::get_storage(
        const memory_storage_t *storage) {
    return storage ? *storage : memory_storage_t::empty_storage();
}
const memory_storage_t &rnn_utils::get_storage(
        const std::unique_ptr<memory_storage_t> &storage) {
    return rnn_utils::get_storage(storage.get());
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

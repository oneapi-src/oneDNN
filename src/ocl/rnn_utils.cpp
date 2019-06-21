/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "c_types_map.hpp"

#include "ref_rnn.hpp"
#include "rnn_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace prop_kind;

bool rnn_utils::is_training(const rnn_pd_t &pd) {
    return utils::one_of(pd.desc()->prop_kind, forward_training, backward);
}

size_t rnn_utils::ws_states_size(const rnn_pd_t &pd) {
    int wic = nstl::max(pd.SLC(), nstl::max(pd.SIC(), pd.DIC()));
    int n_states = pd.cell_kind() == mkldnn_vanilla_lstm ? 2 : 1;
    return static_cast<size_t>((pd.L() + 1) * pd.D() * (pd.T() + 1)
            * n_states * pd.MB() * wic);
}

size_t rnn_utils::ws_diff_states_size(const rnn_pd_t &pd) {
    int wic = nstl::max(pd.SLC(), nstl::max(pd.SIC(), pd.DIC()));
    int n_states = pd.cell_kind() == mkldnn_vanilla_lstm ? 2 : 1;
    return static_cast<size_t>((pd.L() + 1) * pd.D() * (pd.T() + 1)
            * (n_states + 1) * pd.MB() * wic);
}

size_t rnn_utils::ws_gates_size(const rnn_pd_t &pd) {
    return static_cast<size_t>(pd.L() * pd.D() * pd.T() * pd.MB() * pd.G()
            * pd.DIC());
}

size_t rnn_utils::ws_cell_comp_size(const rnn_pd_t &pd) {
    return static_cast<size_t>(pd.is_lbr() * pd.G() * pd.MB() * pd.DIC());
}

size_t rnn_utils::ws_grid_comp_size(const rnn_pd_t &pd) {
    return static_cast<size_t>(pd.is_lbr() * is_training(pd) * pd.L() * pd.D()
            * pd.T() * pd.MB() * pd.DIC());
}

size_t rnn_utils::get_ws_size(const rnn_pd_t &pd) {
    size_t ws_gates_offset, ws_states_offset, ws_diff_states_offset,
        ws_grid_comp_offset, ws_cell_comp_offset;
    set_offsets(pd, ws_gates_offset, ws_states_offset, ws_diff_states_offset,
            ws_grid_comp_offset, ws_cell_comp_offset);
    return ws_grid_comp_offset + ws_grid_comp_size(pd);
}

size_t rnn_utils::get_scratchpad_size(const rnn_pd_t &pd) {
    size_t ws_gates_offset, ws_states_offset, ws_diff_states_offset,
        ws_grid_comp_offset, ws_cell_comp_offset;
    set_offsets(pd, ws_gates_offset, ws_states_offset, ws_diff_states_offset,
            ws_grid_comp_offset, ws_cell_comp_offset);
    if (pd.desc()->prop_kind == forward_inference)
        return ws_cell_comp_offset + ws_cell_comp_size(pd);
    else
        return ws_cell_comp_size(pd);
}

void rnn_utils::set_offsets(const rnn_pd_t &pd, size_t &ws_gates_offset,
        size_t &ws_states_offset, size_t &ws_diff_states_offset,
        size_t &ws_grid_comp_offset, size_t &ws_cell_comp_offset) {
    const size_t page_size = 4096;
    ws_gates_offset
            = 0; // assumes the workspace base pointer is page aligned
    ws_states_offset = utils::rnd_up(ws_gates_size(pd), page_size);
    ws_diff_states_offset
        = utils::rnd_up(ws_states_offset + ws_states_size(pd), page_size);
    ws_grid_comp_offset = utils::rnd_up(ws_diff_states_offset
            + ws_diff_states_size(pd), page_size);
    ws_cell_comp_offset = utils::rnd_up(ws_grid_comp_offset
            + ws_grid_comp_size(pd), page_size);
}


}
}
}

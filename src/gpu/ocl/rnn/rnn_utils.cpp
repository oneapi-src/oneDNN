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

#include "gpu/ocl/rnn/rnn_utils.hpp"

#include "common/c_types_map.hpp"
#include "gpu/ocl/rnn/ref_rnn.hpp"
#include "gpu/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
#define AOC array_offset_calculator

using namespace dnnl::impl::utils;
using namespace dnnl::impl::gpu::gpu_utils;
using namespace prop_kind;
using namespace data_type;

bool rnn_utils::is_ldigo(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked) return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[4] == 1
            && str[3] == dims[4] && str[1] == str[2] * dims[2]
            && str[0] == str[1] * dims[1];
};

bool rnn_utils::is_ldgoi(const memory_desc_wrapper &md) {
    if (md.format_kind() != format_kind::blocked) return false;

    auto blk = md.blocking_desc();
    auto str = blk.strides;
    auto dims = md.dims();
    return md.ndims() == 5 && blk.inner_nblks == 0 && str[2] == 1
            && str[3] == dims[4] * str[4] && str[1] == str[3] * dims[3]
            && str[0] == str[1] * dims[1];
};

void rnn_utils::init_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d, bool is_xe_hpc) {

    rnn = utils::zero<decltype(rnn)>();
    rnn.is_fwd = utils::one_of(rd.prop_kind, prop_kind::forward_training,
            prop_kind::forward_inference);
    rnn.is_training = utils::one_of(
            rd.prop_kind, prop_kind::forward_training, prop_kind::backward);
    rnn.is_lbr = rd.cell_kind == dnnl_lbr_gru;
    rnn.is_vanilla_gru = rd.cell_kind == dnnl_vanilla_gru;
    rnn.arch_ld = is_xe_hpc ? 128 : 64;

    switch (rd.direction) {
        case dnnl_unidirectional_left2right: rnn.exec_dir = l2r; break;
        case dnnl_unidirectional_right2left: rnn.exec_dir = r2l; break;
        case dnnl_bidirectional_concat: rnn.exec_dir = bi_concat; break;
        case dnnl_bidirectional_sum: rnn.exec_dir = bi_sum; break;
        default: break;
    }

    if (everyone_is(f32, src_layer_d.data_type(), dst_layer_d.data_type(),
                weights_layer_d.data_type()))
        rnn.dt_conf = all_f32;
    else if (everyone_is(bf16, src_layer_d.data_type(), dst_layer_d.data_type(),
                     weights_layer_d.data_type()))
        rnn.dt_conf = all_bf16;
    else if (everyone_is(f16, src_layer_d.data_type(), dst_layer_d.data_type(),
                     weights_layer_d.data_type()))
        rnn.dt_conf = all_f16;
    else if (dst_layer_d.data_type() == u8) {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            rnn.dt_conf = u8u8u8u8;
        else
            rnn.dt_conf = f32u8f32u8;
    } else {
        if (IMPLICATION(src_iter_d.md_, src_iter_d.data_type() == u8))
            rnn.dt_conf = u8u8u8f32;
        else
            rnn.dt_conf = f32u8f32f32;
    }
    rnn.is_int8 = !one_of(rnn.dt_conf, all_f32, all_f16, all_bf16);

    rnn.aux_data_type
            = rnn.dt_conf == all_f16 ? data_type::f16 : data_type::f32;
    rnn.diff_data_type = data_type::f32;

    rnn.n_layer = weights_layer_d.dims()[0];
    rnn.n_iter = src_layer_d.dims()[0];
    rnn.n_dir = weights_layer_d.dims()[1];
    rnn.n_gates = weights_layer_d.dims()[3];
    rnn.n_states = rd.cell_kind == dnnl_vanilla_lstm ? 2 : 1;
    rnn.n_bias = rnn.n_gates + rnn.is_lbr;
    rnn.mb = src_layer_d.dims()[1];
    rnn.sic = weights_iter_d.dims()[2];
    rnn.slc = weights_layer_d.dims()[2];
    rnn.dhc = weights_layer_d.dims()[4];
    rnn.dlc = dst_layer_d.dims()[2];
    rnn.wic = nstl::max(rnn.slc, nstl::max(rnn.sic, rnn.dhc));

    rnn.gates_ld = rnn.dhc * rnn.n_gates;

    // Set the correct number of weights parts
    rnn.n_parts_weights_layer = 1;
    rnn.parts_weights_layer[0] = rnn.n_gates;
    rnn.parts_weights_layer[1] = 0;

    //there are two parts for VANILLA GRU weights iteration
    rnn.n_parts_weights_iter = rnn.is_vanilla_gru ? 2 : 1;
    rnn.parts_weights_iter[0] = rnn.is_vanilla_gru ? 2 : rnn.n_gates;
    rnn.parts_weights_iter[1] = rnn.is_vanilla_gru ? 1 : 0;

    rnn.n_parts_bias = 1;
    rnn.parts_bias[0] = rnn.n_bias;
    rnn.parts_bias[1] = 0;

    bool is_gru = utils::one_of(
            rd.cell_kind, alg_kind::vanilla_gru, alg_kind::lbr_gru);

    // Decide if to merge gemm across iterations or layers
    auto dst_layer_is_trivial_stride = dst_layer_d.dims()[0] <= 1
            || dst_layer_d.dims()[1] <= 1
            || (dst_layer_d.blocking_desc().strides[0]
                    == (dst_layer_d.blocking_desc().strides[1] * rnn.mb));

    // Does not account for alignment striding
    dim_t merge_scratch_size_estimate = rnn.gates_ld * rnn.mb * rnn.n_iter;
    bool is_small_scratch = merge_scratch_size_estimate < 256 * 1024 * 1024;
    rnn.merge_gemm_layer = dev_getenv("merge_gemm_layer",
            is_small_scratch); // Avoid excessive memory usage
    rnn.merge_gemm_iter = dev_getenv("merge_gemm_iter",
            is_small_scratch && dst_layer_is_trivial_stride
                    && !(rnn.is_fwd || is_gru));

    // Decide to copy bias
    rnn.copy_bias = rnn.is_int8;

    rnn.use_workspace = rnn.is_training;

    switch (rnn.dt_conf) {
        case all_f32:
        case f32u8f32f32:
            rnn.input_data_type = f32;
            rnn.dst_data_type = f32;
            rnn.output_data_type = f32;
            break;
        case all_f16:
            rnn.input_data_type = f16;
            rnn.dst_data_type = f16;
            rnn.output_data_type = f16;
            break;
        case u8u8u8u8:
            rnn.input_data_type = u8;
            rnn.dst_data_type = u8;
            rnn.output_data_type = u8;
            break;
        case u8u8u8f32:
            rnn.input_data_type = u8;
            rnn.dst_data_type = f32;
            rnn.output_data_type = u8;
            break;
        case f32u8f32u8:
            rnn.input_data_type = f32;
            rnn.dst_data_type = u8;
            rnn.output_data_type = f32;
            break;
        case all_bf16:
            rnn.input_data_type = bf16;
            rnn.dst_data_type = bf16;
            rnn.output_data_type = bf16;
            break;
        default: assert(!"unimplemented");
    }
}

void rnn_utils::init_test_mode(conf_t &rnn, const primitive_attr_t &attr) {
    rnn.is_testmode = attr.rnn_tparams_.test_mode_;
    rnn.tm_ngates = attr.rnn_tparams_.ngates_;
    rnn.tm_cscale = attr.rnn_tparams_.cscale_;
}

void rnn_utils::set_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &diff_src_layer_d,
        const memory_desc_wrapper &diff_dst_layer_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d) {

    const bool is_fwd = rnn.is_fwd;
    const bool is_bwd = !rnn.is_fwd;

    //Set leading dimensions for input weights arrays depending on input format
    auto set_dims = [&](const memory_desc_wrapper &md, dim_t &ld, dim_t &nld) {
        ld = 0;
        nld = 0;
        if (md.is_blocking_desc()) {
            if (is_ldigo(md)) {
                ld = md.blocking_desc().strides[2];
                nld = md.dims()[2];
            } else if (is_ldgoi(md)) {
                ld = md.blocking_desc().strides[4];
                nld = md.dims()[3] * md.dims()[4];
            } else
                assert(!"unsupported weights format");
        }
    };
    set_dims(weights_layer_d, rnn.weights_layer_ld, rnn.weights_layer_nld);
    set_dims(weights_iter_d, rnn.weights_iter_ld, rnn.weights_iter_nld);
    if (is_bwd) {
        set_dims(diff_weights_layer_d, rnn.diff_weights_layer_ld,
                rnn.diff_weights_layer_nld);
        set_dims(diff_weights_iter_d, rnn.diff_weights_iter_ld,
                rnn.diff_weights_iter_nld);
    }

    int sizeof_states_dt
            = rnn.dt_conf == all_f32 ? sizeof(cl_float) : sizeof(cl_half);
    int aux_elsz = rnn.aux_data_type == data_type::f16 ? sizeof(cl_half)
                                                       : sizeof(float);
    rnn.ws_states_elsz = types::data_type_size(rd.src_layer_desc.data_type);

    rnn.scratch_gates_elsz = aux_elsz;
    rnn.scratch_diff_gates_elsz = is_bwd
            ? (rnn.dt_conf == all_bf16 ? sizeof(bfloat16_t) : aux_elsz)
            : 0;

    // Set workspace sizes to store:
    // states to copmute a pass
    // diff states to copmute bwd pass (training only)
    // intermediate results from the gates
    rnn.states_ws_ld = get_good_ld(rnn.arch_ld,
            nstl::max(rnn.slc, nstl::max(rnn.sic, rnn.dhc)), sizeof_states_dt);
    rnn.gates_ws_ld = get_good_ld(rnn.arch_ld, rnn.gates_ld,
            rnn.dt_conf == all_f16 ? sizeof(cl_half) : sizeof(cl_float));
    // Disable associativity check on some large problems to reduce memory
    // usage. Can be removed when further improvements are made to
    // copy_diff_src_layer
    rnn.scratch_diff_states_ld = get_good_ld(rnn.arch_ld,
            nstl::max(rnn.slc, nstl::max(rnn.sic, rnn.dhc)), sizeof(cl_float),
            utils::everyone_is(rnn.slc, rnn.sic, rnn.dhc)
                    && rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.mb
                            > 128 * 1024);

    rnn.scratch_gates_ld
            = get_good_ld(rnn.arch_ld, rnn.gates_ld, rnn.scratch_gates_elsz);
    rnn.scratch_diff_gates_ld = is_bwd ? get_good_ld(rnn.arch_ld, rnn.gates_ld,
                                        rnn.scratch_diff_gates_elsz)
                                       : 0;

    bool is_lstm = rd.cell_kind == dnnl_vanilla_lstm;

    bool require_copy_src_layer = [&]() {
        auto &strides = src_layer_d.strides();
        auto &pdims = src_layer_d.padded_dims();
        auto dt_size = types::data_type_size(src_layer_d.data_type());

        // The GEMM interface assumes input buffers are well aligned. We need to
        // implement a way to avoid kernels relying on this alignment to remove
        // this restriction.
        if (pdims[0] > 1 && (strides[0] * dt_size) % 8) return true;

        if (rnn.merge_gemm_layer) {
            // GEMM inputs are represented as 2d inputs. As such, the merged
            // dimension need to be dense. This restriction could be removed by
            // using batched GEMM with appropriate strides instead.
            constexpr int iter_dim = 0, mb_dim = 1;
            if (pdims[iter_dim] > 1 && pdims[mb_dim] > 1
                    && (strides[iter_dim] != strides[mb_dim] * rnn.mb))
                return true;
            if (rnn.exec_dir != rnn_utils::l2r) return true;
        }

        // Bug workaround, likely related to the undefined mb stride
        if (pdims[1] == 1) return true;

        return false;
    }();

    bool prefer_copy_src_layer = [&]() {
        auto &strides = src_layer_d.strides();
        auto &pdims = src_layer_d.padded_dims();
        auto dt_size = types::data_type_size(src_layer_d.data_type());

        // Data is already well aligned. Copying does not provide benefit
        if (pdims[1] == 1 || strides[1] == rnn.gates_ws_ld
                || (strides[1] % 64 == 0))
            return false;

        // Better to rely on GEMM to emit reorder if it is necessary if there is
        // limited data reuse
        const dim_t data_reuse = rnn.n_dir * (rnn.is_training ? 2 : 1);
        if (data_reuse < 2) return false;

        // Prefer lower memory usage
        if (src_layer_d.nelems(true) * dt_size >= 1024 * 1024 * 1024)
            return false;

        return true;
    }();

    bool copy_src_layer = dev_getenv("copy_src_layer", prefer_copy_src_layer)
            || require_copy_src_layer;

    bool require_copy_diff_dst_layer = [&]() {
        if (is_fwd) return false;

        auto &strides = diff_dst_layer_d.strides();
        auto &pdims = diff_dst_layer_d.padded_dims();
        auto dt_size = types::data_type_size(diff_dst_layer_d.data_type());

        // The GEMM interface assumes input buffers are well aligned. We need to
        // implement a way to avoid kernels relying on this alignment to remove
        // this restriction.
        if (pdims[0] > 1 && (strides[0] * dt_size) % 8) return true;

        if (rnn.merge_gemm_layer) {
            // GEMM inputs are represented as 2d inputs. As such, the merged
            // dimension need to be dense. This restriction could be removed by
            // using batched GEMM with appropriate strides instead.
            constexpr int iter_dim = 0, mb_dim = 1;
            if (pdims[iter_dim] > 1 && pdims[mb_dim] > 1
                    && (strides[iter_dim] != strides[mb_dim] * rnn.mb))
                return true;
            if (rnn.exec_dir != rnn_utils::r2l) return true;
        }

        // Bug workaround, likely related to the undefined mb stride
        if (pdims[1] == 1) return true;

        return false;
    }();
    bool copy_diff_dst_layer = dev_getenv("copy_diff_dst_layer", false)
            || require_copy_diff_dst_layer;

    bool require_copy_diff_src_layer = [&]() {
        if (is_fwd) return false;

        // Unimplemented
        if (rnn.merge_gemm_iter || rnn.merge_gemm_layer) return true;

        // Direction need to be summed together. This requires generating new
        // GEMM kernels to perform a sum accumulation for the final accumulation.
        if (rnn.n_dir > 1) return true;

        return false;
    }();
    bool copy_diff_src_layer = dev_getenv("copy_diff_src_layer", false)
            || require_copy_diff_src_layer;
    rnn.copy_src_layer = copy_src_layer;
    rnn.copy_diff_dst_layer = copy_diff_dst_layer;
    rnn.copy_diff_src_layer = copy_diff_src_layer;
    rnn.ws_states_cell_size = rnn.mb * rnn.states_ws_ld * rnn.ws_states_elsz;
    rnn.ws_states_size = (copy_src_layer ? rnn.n_layer + 1 : rnn.n_layer)
            * rnn.n_dir * (rnn.n_iter + 1) * rnn.ws_states_cell_size;

    // we do not need a good ld for iter_c as it is not involved in GEMM
    // for now reverting it back to what it was originally
    // TODO: seprate diff_c_offsets from diff-states & seprate h- and c- off
    rnn.ws_c_states_cell_size
            = is_lstm ? rnn.mb * rnn.states_ws_ld * aux_elsz : 0;
    rnn.ws_c_states_size = is_lstm ? rnn.n_layer * rnn.n_dir * (rnn.n_iter + 1)
                    * rnn.ws_c_states_cell_size
                                   : 0;

    auto scratch_diff_n_states
            = copy_diff_dst_layer || copy_diff_src_layer || rnn.n_layer != 1
            ? rnn.n_states + 1
            : rnn.n_states;
    // rnn.n_layer > 1 is currently required due to copy_{init,res}_iter
    bool have_result_layer = copy_diff_src_layer || rnn.n_layer > 1;
    auto scratch_diff_n_layer
            = rnn.n_layer - 1 + copy_diff_dst_layer + have_result_layer;

    // Due to the grid iteration used, if no full layers are required, only use
    // 2 cells, one for the previous iteration and one for the current
    // iteration.
    auto scratch_diff_n_cells = is_bwd
            ? (scratch_diff_n_layer > 0
                              ? scratch_diff_n_layer * (rnn.n_iter + 1)
                              : 2)
                    * scratch_diff_n_states * rnn.n_dir
            : 0;
    rnn.scratch_diff_states_size = scratch_diff_n_cells * rnn.mb
            * rnn.scratch_diff_states_ld * aux_elsz;

    rnn.ws_gates_cell_size = rnn.mb * rnn.gates_ws_ld * aux_elsz;
    rnn.ws_gates_size = rnn.is_training
            ? (rnn.n_layer * rnn.n_dir * rnn.n_iter * rnn.ws_gates_cell_size)
            : 0;

    // Reduce workspace memory by recomputing gates for bwd
    // TODO: Extend this optimization to other alg_kind.
    bool supports_recompute_gates
            = utils::one_of(rd.cell_kind, alg_kind::vanilla_lstm,
                      alg_kind::vanilla_rnn)
            && rnn.is_training;
    bool prefer_recompute_gates = rnn.ws_gates_size >= 512 * 1024 * 1024;
    rnn.recompute_gates = dev_getenv("recompute_gates", prefer_recompute_gates)
            && supports_recompute_gates;
    if (rnn.recompute_gates) rnn.ws_gates_size = 0;

    rnn.n_iter_scratch_gates
            = (rnn.merge_gemm_layer || rnn.merge_gemm_iter) ? rnn.n_iter : 1;

    // To reduce memory usage, use scratch_diff_gates in place of scratch_gates
    // when the layout is the same, i.e. they have the same data type size.
    bool need_scratch_gates = is_fwd
            || (rnn.recompute_gates
                    && rnn.scratch_gates_elsz != rnn.scratch_diff_gates_elsz);
    rnn.scratch_gates_size = need_scratch_gates ? rnn.n_iter_scratch_gates
                    * rnn.mb * rnn.scratch_gates_ld * rnn.scratch_gates_elsz
                                                : 0;
    rnn.scratch_diff_gates_size = is_bwd ? rnn.n_iter_scratch_gates * rnn.mb
                    * rnn.scratch_diff_gates_ld * rnn.scratch_diff_gates_elsz
                                         : 0;
    rnn.scratch_dhG1_size = (rd.cell_kind == alg_kind::vanilla_gru && is_bwd)
            ? rnn.mb * rnn.scratch_diff_states_ld * sizeof(float)
            : 0;
    rnn.ws_bias_size
            = rnn.n_layer * rnn.n_dir * rnn.n_bias * rnn.dhc * aux_elsz;

    // For intermediate step in post-gemm fwd lbr gru
    rnn.scratch_cell_size = [&]() {
        if (rnn.is_lbr && is_fwd) {
            return rnn.mb * rnn.scratch_gates_ld * rnn.scratch_gates_elsz;
        } else if (rnn.is_lbr && is_bwd) {
            return rnn.mb * rnn.scratch_diff_gates_ld
                    * rnn.scratch_diff_gates_elsz;
        } else if (rd.cell_kind == alg_kind::vanilla_gru && is_bwd) {
            return rnn.mb * rnn.states_ws_ld * rnn.ws_states_elsz;
        } else {
            return static_cast<dim_t>(0);
        }
    }();

    // Used for storing the intermediate value from fwd pass in training lbr gru
    rnn.ws_per_cell = rnn.is_lbr * rnn.mb * rnn.dhc * aux_elsz;
    rnn.ws_grid_comp_size = rnn.is_lbr * rnn.is_training * rnn.n_layer
            * rnn.n_dir * rnn.n_iter * rnn.ws_per_cell;

    set_workspace_offsets(rnn, rnn.ws_gates_offset, rnn.ws_states_offset,
            rnn.ws_c_state_offset, rnn.ws_grid_comp_offset, rnn.ws_bias_offset);
}

dim_t rnn_utils::get_good_ld(
        dim_t arch_ld, dim_t dim, dim_t sizeof_dt, bool ignore_assoc) {
    // Leading dimension for matrices has 64-byte or 128-byte alignment (PVC-A)
    dim_t ld = rnd_up(dim, arch_ld / sizeof_dt);
    // Further alignment is associated with 8-way associativity of L1-cache
    return (ld % 256 == 0) && !ignore_assoc ? ld + arch_ld / sizeof_dt : ld;
}

dim_t rnn_utils::set_workspace_offsets(const conf_t &rnn,
        dim_t &ws_gates_offset, dim_t &ws_states_offset,
        dim_t &ws_c_states_offset, dim_t &ws_grid_comp_offset,
        dim_t &ws_bias_offset) {

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
    register_space(ws_gates);
    register_space(ws_states);
    register_space(ws_c_states);
    register_space(ws_grid_comp);

    ws_bias_offset = 0;
    if (rnn.copy_bias) { register_space(ws_bias); }
    return current_offset;
}

dim_t rnn_utils::get_workspace_size(const conf_t &rnn) {
    dim_t ws_gates_offset, ws_states_offset, ws_c_states_offset,
            ws_grid_comp_offset, ws_bias_offset;
    return set_workspace_offsets(rnn, ws_gates_offset, ws_states_offset,
            ws_c_states_offset, ws_grid_comp_offset, ws_bias_offset);
}

void rnn_utils::set_offsets_fwd_gemm(const conf_t &rnn, dim_t dir, dim_t lay,
        const std::vector<dim_t> &wei_layer_offsets,
        dim_t &grid_wei_lay_offset) {
    // Function overloaded. This function is called by grid execution
    dim_t n_layer = rnn.n_layer;
    dim_t n_dir = rnn.n_dir;

    const AOC<const dim_t, 3> off_weights_lay(wei_layer_offsets.data(), n_layer,
            n_dir, rnn.n_parts_weights_layer);

    grid_wei_lay_offset = off_weights_lay(lay, dir, 0);
    UNUSED(n_layer);
}

void rnn_utils::set_offsets_fwd_gemm(const conf_t &rnn, dim_t iter, dim_t dir,
        dim_t lay, const std::vector<dim_t> &wei_iter_offsets,
        dim_t &cell_wei_iter_offset) {
    if (!wei_iter_offsets.empty()) {
        const AOC<const dim_t, 3> off_weights_iter(wei_iter_offsets.data(),
                rnn.n_layer, rnn.n_dir, rnn.n_parts_weights_iter);
        cell_wei_iter_offset = off_weights_iter(lay, dir, 0);
    }
}

void rnn_utils::set_gru_offsets_part2(const conf_t &rnn, dim_t iter, dim_t dir,
        dim_t lay, const std::vector<dim_t> &wei_iter_offsets,
        dim_t &cell_wei_iter_offset, dim_t &cell_scratch_fwd_offset) {

    AOC<const dim_t, 3> off_weights_iter(wei_iter_offsets.data(), rnn.n_layer,
            rnn.n_dir, rnn.n_parts_weights_iter);
    cell_wei_iter_offset = off_weights_iter(lay, dir, 1);
    cell_scratch_fwd_offset = 2 * rnn.dhc * rnn.scratch_gates_elsz;
}

void rnn_utils::set_offsets_bwd_gemm(const conf_t &rnn, dim_t iter, dim_t dir,
        dim_t lay, dim_t &cell_diff_wei_iter_off, dim_t &cell_diff_wei_lay_off,
        dim_t &cell_diff_wei_iter_off2) {

    set_offsets_bwd_gemm(
            rnn, iter, dir, lay, cell_diff_wei_iter_off, cell_diff_wei_lay_off);
    cell_diff_wei_iter_off2
            = cell_diff_wei_iter_off + 2 * rnn.dhc * sizeof(float);
}

void rnn_utils::set_offsets_bwd_gemm(const conf_t &rnn, dim_t iter, dim_t dir,
        dim_t lay, dim_t &cell_diff_wei_iter_off,
        dim_t &cell_diff_wei_lay_off) {
    dim_t n_layers = rnn.n_layer;
    dim_t n_dir = rnn.n_dir;

    cell_diff_wei_lay_off
            = OFF3(lay, n_layers, dir, n_dir, 0,
                      rnn.diff_weights_layer_nld * rnn.diff_weights_layer_ld)
            * sizeof(float);
    cell_diff_wei_iter_off
            = OFF3(lay, n_layers, dir, n_dir, 0,
                      rnn.diff_weights_iter_nld * rnn.diff_weights_iter_ld)
            * sizeof(float);

    UNUSED(n_layers);
}

status_t rnn_utils::set_good_strides(
        dim_t ld_, memory_desc_t &weights_md, format_tag_t tag) {
    auto &strides = weights_md.format_desc.blocking.strides;
    auto dims = weights_md.dims;
    using namespace format_tag;

    if (tag == ldigo) {
        strides[2] = rnn_utils::get_good_ld(
                ld_, strides[2], types::data_type_size(weights_md.data_type));
        strides[1] = dims[2] * strides[2];
        strides[0] = dims[1] * strides[1];
    } else if (tag == ldgoi) {
        strides[4] = rnn_utils::get_good_ld(
                ld_, strides[4], types::data_type_size(weights_md.data_type));
        strides[3] = dims[4] * strides[4];
        strides[1] = dims[3] * strides[3];
        strides[0] = dims[1] * strides[1];
    } else
        return status::unimplemented;

    return status::success;
}

status_t rnn_utils::set_expected_desc(
        conf_t &rnn, memory_desc_t &weights_md, bool is_iter) {
    using namespace format_tag;
    CHECK(memory_desc_init_by_tag(weights_md, rnn.is_fwd ? ldigo : ldgoi));

    // Adjust strides for good leading dimension in GEMM
    CHECK(set_good_strides(
            rnn.arch_ld, weights_md, rnn.is_fwd ? ldigo : ldgoi));

    // set we need extra memory
    if (rnn.is_fwd && !one_of(rnn.dt_conf, all_f32, all_f16, all_bf16)) {
        weights_md.extra.flags = memory_extra_flags::rnn_u8s8_compensation;
        weights_md.extra.compensation_mask = 27; // ldigo 11011;
    }
    return status::success;
}

memory_storage_t &rnn_utils::get_storage(
        const std::unique_ptr<memory_storage_t> &storage) {
    return storage ? *storage : memory_storage_t::empty_storage();
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

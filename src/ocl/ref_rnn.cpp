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

/*
  General architecture

  for diff states, we have n_states + 1 as we have n_states diff
  to propagate to the previous iteration and 1 states to propagate
  to the previous layer
  index 0 is dh for cell(t-1, l) to consume
  index 1 is dc for cell(t-1, l) to consume
  index 2 is dh for cell(t, l-1) to consume
  this indexing enables to have the same indexing for states in elemwise
  function
  only the cell execution function should be impacted

 */
#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_traits.hpp"
#include "type_helpers.hpp"
#include "ref_rnn.hpp"
#include "cl_executor.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
using namespace prop_kind;
using namespace alg_kind;
using namespace rnn_utils;

#define AOC array_offset_calculator

//************************* Cell execution *************************//
/// @todo shall this be templated on activation function to enable svml calls
/// particularly

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
elemwise_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::rnn_elemwise)) {
    stream_t *s = ctx.stream();
    auto &executor = *(utils::downcast<cl_stream_t *>(s)->cl_executor());
    auto nd_range = cl_nd_range_t({batch, dic});

    ocl_kernel_t kernel = (aprop == prop_kind::forward)
        ? elemwise_fwd_kernel_: elemwise_bwd_kernel_;
    kernel.set_arg(0, dir);
    kernel.set_arg(1, lay);
    kernel.set_arg(2, iter);
    kernel.set_arg(3, workspace);
    kernel.set_arg(4, bias);
    executor.parallel_for(nd_range, kernel);
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
elemwise_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::lstm_elemwise))
{
    stream_t *s = ctx.stream();
    auto &executor = *(utils::downcast<cl_stream_t *>(s)->cl_executor());
    auto nd_range = cl_nd_range_t({batch, dic});
    ocl_kernel_t kernel = (aprop == prop_kind::forward)
        ? elemwise_fwd_kernel_: elemwise_bwd_kernel_;
    kernel.set_arg(0, dir);
    kernel.set_arg(1, lay);
    kernel.set_arg(2, iter);
    kernel.set_arg(3, workspace);
    kernel.set_arg(4, bias);
    executor.parallel_for(nd_range, kernel);
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
elemwise_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::gru_lbr_elemwise))
{
    assert(!"unimplemented");
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
gemm_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::packed_gemm)) {
#if USE_MKL_PACKED_GEMM
// TBD
#else
    UNUSED(m);
    UNUSED(n);
    UNUSED(k);
    UNUSED(a);
    UNUSED(b);
    UNUSED(c);
    UNUSED(is_B_trans);
    UNUSED(beta);
    UNUSED(gemm_kind);
    assert(!"packed gemm is disabled");
#endif
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
gemm_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::gemm_primitive)) {

    /* FIXME: This should be created once per execute() instead of creating
     * memory before each gemm call. Each cell type (+prop kind) might have
     * different number of GEMMs.
     * */

    memory_desc_t scratchpad_md;
    dims_t scratchpad_dims = { static_cast<int64_t>(get_scratchpad_size(*pd())) };
    mkldnn_memory_desc_init_by_tag(&scratchpad_md, 1, scratchpad_dims, src_type,
            format_tag::x);

    void *mem_storage_scratchpad = nullptr;

    memory_t *workspace = (aprop == prop_kind::forward)
        ? ctx.output(MKLDNN_ARG_WORKSPACE)
        : ctx.input(MKLDNN_ARG_WORKSPACE);

    if (use_workspace_) {
        workspace->memory_storage()->get_data_handle(&mem_storage_scratchpad);
    } else {
        scratchpad_->get_data_handle(&mem_storage_scratchpad);
    }
    using src_t = typename prec_traits<src_type>::type;
    using wei_t = typename prec_traits<weights_type>::type;
    void *mem_storage_weights = nullptr;
    memory_t *weights = nullptr;
    exec_args_t gemm_args;

    std::shared_ptr<memory_t> gemm_mem_A;
    std::shared_ptr<memory_t> gemm_mem_B;
    std::shared_ptr<memory_t> gemm_mem_C;

    switch (gemm_kind) {
    case gemm_iter:
    case gemm_layer:
        weights = (gemm_kind == gemm_layer)
            ? ctx.input(MKLDNN_ARG_WEIGHTS_LAYER)
            : ctx.input(MKLDNN_ARG_WEIGHTS_ITER);
        weights->memory_storage()->get_data_handle(&mem_storage_weights);

        gemm_mem_A.reset(new memory_t(engine(), weights->md(),
                memory_flags_t::use_backend_ptr, nullptr));
        gemm_mem_B.reset(new memory_t(engine(), &scratchpad_md,
                memory_flags_t::use_backend_ptr, nullptr));
        gemm_mem_C.reset(new memory_t(engine(), &scratchpad_md,
                memory_flags_t::use_backend_ptr, nullptr));
        gemm_mem_A->set_data_handle(mem_storage_weights);
        gemm_mem_B->set_data_handle(mem_storage_scratchpad);
        gemm_mem_C->set_data_handle(mem_storage_scratchpad);
        gemm_mem_A->memory_storage()->set_offset(off_a * sizeof(wei_t));
        gemm_mem_B->memory_storage()->set_offset(off_b * sizeof(src_t));
        gemm_mem_C->memory_storage()->set_offset(off_c * sizeof(src_t));
        break;
    case gemm_diff_wei_iter:
    case gemm_diff_wei_layer:
        weights = (gemm_kind == gemm_diff_wei_iter)
            ? ctx.output(MKLDNN_ARG_DIFF_WEIGHTS_ITER)
            : ctx.output(MKLDNN_ARG_DIFF_WEIGHTS_LAYER);
        weights->memory_storage()->get_data_handle(&mem_storage_weights);
        gemm_mem_A.reset(new memory_t(engine(), &scratchpad_md,
                memory_flags_t::use_backend_ptr, nullptr));
        gemm_mem_B.reset(new memory_t(engine(), &scratchpad_md,
                memory_flags_t::use_backend_ptr, nullptr));
        gemm_mem_C.reset(new memory_t(engine(), weights->md(),
                memory_flags_t::use_backend_ptr, nullptr));
        gemm_mem_A->set_data_handle(mem_storage_scratchpad);
        gemm_mem_B->set_data_handle(mem_storage_scratchpad);
        gemm_mem_C->set_data_handle(mem_storage_weights);
        gemm_mem_A->memory_storage()->set_offset(off_a * sizeof(src_t));
        gemm_mem_B->memory_storage()->set_offset(off_b * sizeof(src_t));
        gemm_mem_C->memory_storage()->set_offset(off_c * sizeof(wei_t));
        break;
    default:
        assert(!"unknown gemm_kind");
    }

    gemm_args[MKLDNN_ARG_SRC_0] = {gemm_mem_A.get(), true};
    gemm_args[MKLDNN_ARG_SRC_1] = {gemm_mem_B.get(), true};
    gemm_args[MKLDNN_ARG_DST] = {gemm_mem_C.get(), false};

    auto gemm_ctx = exec_ctx_t(ctx.stream(), std::move(gemm_args));

    switch (gemm_kind) {
    case gemm_iter:
        gemm_iter_->execute(gemm_ctx);
        break;
    case gemm_layer:
        gemm_layer_->execute(gemm_ctx);
        break;
    case gemm_diff_wei_iter:
        gemm_diff_wei_iter_->execute(gemm_ctx);
        break;
    case gemm_diff_wei_layer:
        gemm_diff_wei_layer_->execute(gemm_ctx);
        break;
    default:
        assert(!"unknown gemm_kind");
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
void _ref_rnn_common_t<aprop, src_type, weights_type>::gates_reduction(
    const exec_ctx_t &ctx,
    int dir, int lay, int iter,
    int n_gates, int dic, int batch,
    const memory_storage_t &ws, const memory_storage_t &diff_bias) const {
    auto s = ctx.stream();
    auto &executor = *(utils::downcast<cl_stream_t *>(s)->cl_executor());
    gates_reduction_kernel_.set_arg(0, dir);
    gates_reduction_kernel_.set_arg(1, lay);
    gates_reduction_kernel_.set_arg(2, iter);
    gates_reduction_kernel_.set_arg(3, diff_bias);
    gates_reduction_kernel_.set_arg(4, ws);

    auto nd_range = cl_nd_range_t({ n_gates, dic });
    executor.parallel_for(nd_range, gates_reduction_kernel_);
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::cell_execution))
{

    if (aprop == prop_kind::forward) {
        AOC<size_t, 3> off_weights_i(weights_input, n_layer, n_direction,
                n_parts_wei_i);
        AOC<size_t, 3> off_weights_st(weights_states, n_layer, n_direction,
                n_parts_wei_st);

        cl_ulong offset_w_input = (cl_ulong)(off_weights_i(lay, dir, 0));
        cl_ulong offset_w_state = (cl_ulong)(off_weights_st(lay, dir, 0));

        cl_ulong offset_states = (cl_ulong)(ws_states_offset_
                + OFF4(lay + 1, n_layer + 1, dir, n_direction, iter, n_iter + 1,
                    0, batch * n_states * wic));
        cl_ulong offset_input = (cl_ulong)(ws_states_offset_
                + OFF4(lay, n_layer + 1, dir, n_direction, iter + 1, n_iter + 1,
                    0, batch * n_states * wic));
        cl_ulong offset_gates = (cl_ulong)(ws_gates_offset_
                + OFF4(lay, n_layer, dir, n_direction, iter, n_iter,
                    0, batch * n_gates * dic));

        gemm_primitive(ctx, n_gates * dic, batch, slc, n_gates * dic, slc,
                batch, wic, n_gates * dic, batch, w_input, offset_w_input,
                workspace, offset_input, workspace, offset_gates, false, 0.0f,
                gemm_layer);
        gemm_primitive(ctx, n_gates * dic, batch, sic, n_gates * dic, sic,
                batch, wic, n_gates * dic, batch, w_state, offset_w_state,
                workspace, offset_states, workspace, offset_gates, false, 1.0f,
                gemm_iter);

        (this->*elemwise_func)(ctx, dir, lay, iter, dic, wic, batch, workspace,
                bias);

    } else { // backward

        AOC<size_t, 3> off_weights_i(weights_input, n_layer, n_direction,
                n_parts_wei_i);
        AOC<size_t, 3> off_weights_st(weights_states, n_layer, n_direction,
                n_parts_wei_st);

        (this->*elemwise_func)(ctx, dir, lay, iter, dic, wic, batch,
            workspace, bias);

        cl_ulong offset_w_state = (cl_ulong)(off_weights_st(lay, dir, 0));
        cl_ulong offset_w_input = (cl_ulong)(off_weights_i(lay, dir, 0));

        gemm_primitive(ctx,
            sic, batch, n_gates * dic, sic, n_gates * dic, batch, n_gates * dic,
            wic, batch, w_state, offset_w_state, workspace, ws_gates_offset_
                + OFF4(lay, n_layer, dir, n_direction, iter, n_iter, 0,
                        n_gates * batch * dic), workspace,
                    ws_diff_states_offset_ + OFF4(lay, n_layer + 1, dir,
                n_direction, iter, n_iter + 1, 0, (n_states + 1) * batch * wic),
            false, 0.0f, gemm_iter);

        gemm_primitive(ctx,
            sic, batch, n_gates * dic, slc, n_gates * dic, batch, n_gates * dic,
            wic, batch, w_input, offset_w_input, workspace, ws_gates_offset_
                + OFF4(lay, n_layer, dir, n_direction, iter,
                n_iter, 0, n_gates * batch * dic), workspace,
            ws_diff_states_offset_ + OFF4(lay, n_layer + 1, dir, n_direction,
                iter, n_iter + 1, 0, (n_states + 1) * batch * wic)
                    + n_states * (batch * wic), false, 0.0f, gemm_layer);

        gemm_primitive(ctx,
            n_gates * dic, slc, batch, n_gates * dic, batch, wic, batch,
            n_gates * dic, slc, workspace, ws_gates_offset_ + OFF4(lay, n_layer,
                dir, n_direction, iter, n_iter, 0, n_gates * batch * dic),
            workspace, ws_states_offset_ + OFF4(lay, n_layer + 1, dir,
                n_direction, iter + 1, n_iter + 1, 0, n_states * batch * wic),
            diff_weights_layer, OFF3(lay, n_layer, dir, n_direction, 0,
                slc * n_gates * dic), true, 1.0f, gemm_diff_wei_layer);

        gemm_primitive(ctx,
            n_gates * dic, sic, batch, n_gates * dic, batch, wic, batch,
            n_gates * dic, sic, workspace, ws_gates_offset_ + OFF4(lay, n_layer,
                dir, n_direction, iter, n_iter, 0, n_gates * batch * dic),
            workspace, ws_states_offset_ + OFF4(lay + 1, n_layer + 1, dir,
                n_direction, iter, n_iter + 1, 0, n_states * batch * wic),
            diff_weights_iter, OFF3(lay, n_layer, dir, n_direction, 0,
                sic * n_gates * dic), true, 1.0f, gemm_diff_wei_iter);

        gates_reduction(ctx, dir, lay, iter, n_gates, dic, batch,
            workspace, diff_bias);
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::cell_execution_gru_lbr)) {
    assert(!"unimplemented");
}
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
cell_execution_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::cell_execution_gru)) {
    assert(!"unimplemented");
}

//*************** Grid computations strategy: linear ***************//
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
grid_execution_sig(
        (_ref_rnn_common_t<aprop, src_type, weights_type>::linear_execution)) {
    // We run the grid of computation
    for (int dir = 0; dir < n_direction; dir++) {
        for (int j = 0; j < n_layer; j++) {
            for (int i = 0; i < n_iter; i++) {
                int lay, iter;
                if (aprop == prop_kind::forward) {
                    lay = j;
                    iter = i;
                } else { // backward
                    lay = n_layer - j - 1;
                    iter = n_iter - i - 1;
                }
                (this->*cell_func)(
                    ctx, dir, lay, iter,
                    dic, slc, sic, wic, batch, n_layer, n_direction,
                    n_iter, n_gates, n_states, n_bias,
                    offset_wei_input_, n_parts_wei_i,
                    offset_wei_state_, n_parts_wei_st,
                    bias,
                    workspace,
                    w_input,
                    w_state,
                    diff_weights_layer,
                    diff_weights_iter,
                    diff_bias
                    );
            }
        }
    }
}

//********* GRID computations strategy: utility functions **********//

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
void _ref_rnn_common_t<aprop, src_type, weights_type>::copy_init_layer(
        const stream_t *s, bool lr, bool rl, int n_iter, int batch, int slc,
        const memory_storage_t &ws, const memory_storage_t &input,
        const memory_storage_t &diff_dst_layer) const {

    auto &executor = *(utils::downcast<const cl_stream_t *>(s)->cl_executor());
    if (aprop == prop_kind::forward) {
        copy_init_layer_kernel_.set_arg(0, ws);
        copy_init_layer_kernel_.set_arg(1, input);
        copy_init_layer_kernel_.set_arg(2, (cl_int)lr);
        copy_init_layer_kernel_.set_arg(3, (cl_int)rl);
        executor.parallel_for(cl_nd_range_t({slc, batch, n_iter}),
            copy_init_layer_kernel_);
    } else {
        copy_init_layer_kernel_.set_arg(0, ws);
        copy_init_layer_kernel_.set_arg(1, diff_dst_layer);
        copy_init_layer_kernel_.set_arg(2, (cl_int)0);
        copy_init_layer_kernel_.set_arg(3, (cl_int)0);
        executor.parallel_for(cl_nd_range_t({batch, n_iter}),
            copy_init_layer_kernel_);
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
void _ref_rnn_common_t<aprop, src_type, weights_type>::copy_init_iter(
        const stream_t *s, int n_layer, int n_direction,
        int batch, int sic, int dic, const memory_storage_t &ws,
        const memory_storage_t &firstit_states, const memory_storage_t &firstit_c_states,
        const memory_storage_t &diff_dst_iter, const memory_storage_t &diff_dst_iter_c) const {

    auto &executor = *(utils::downcast<const cl_stream_t *>(s)->cl_executor());

    if (aprop == prop_kind::forward) {
        copy_init_iter_kernel_.set_arg(0, ws);
        copy_init_iter_kernel_.set_arg(1, firstit_states);
        copy_init_iter_kernel_.set_arg(2, firstit_c_states);
        executor.parallel_for(cl_nd_range_t(
            {sic, batch, n_layer * n_direction}),
            copy_init_iter_kernel_);
    } else {
        copy_init_iter_kernel_.set_arg(0, ws);
        copy_init_iter_kernel_.set_arg(1, diff_dst_iter);
        copy_init_iter_kernel_.set_arg(2, diff_dst_iter_c);
        executor.parallel_for(cl_nd_range_t(
            {dic, batch, n_layer * n_direction}),
            copy_init_iter_kernel_);
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
void _ref_rnn_common_t<aprop, src_type, weights_type>::copy_res_layer(
        const stream_t *s, bool lr, bool rl, int n_iter, int batch, int slc,
        int dic, const memory_storage_t &dst_last_layer,
        const memory_storage_t &diff_src_layer, const memory_storage_t &ws) const {

    auto &executor = *(utils::downcast<const cl_stream_t *>(s)->cl_executor());
    if (aprop == prop_kind::forward) {
        copy_res_layer_kernel_.set_arg(0, ws);
        copy_res_layer_kernel_.set_arg(1, dst_last_layer);
        copy_res_layer_kernel_.set_arg(2, (cl_int)lr);
        copy_res_layer_kernel_.set_arg(3, (cl_int)rl);
        executor.parallel_for(cl_nd_range_t({dic, batch, n_iter}),
            copy_res_layer_kernel_);
    } else {
        copy_res_layer_kernel_.set_arg(0, ws);
        copy_res_layer_kernel_.set_arg(1, diff_src_layer);
        copy_res_layer_kernel_.set_arg(2, (cl_int)lr);
        copy_res_layer_kernel_.set_arg(3, (cl_int)rl);
        executor.parallel_for(cl_nd_range_t({slc, batch, n_iter}),
            copy_res_layer_kernel_);
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
void _ref_rnn_common_t<aprop, src_type, weights_type>::copy_res_iter(
        const stream_t *s, int n_layer, int n_direction,
        int batch, int sic, int dic, const memory_storage_t &dst_last_iter, const memory_storage_t &dst_last_iter_c,
        const memory_storage_t &diff_src_iter, const memory_storage_t &diff_src_iter_c, const memory_storage_t &ws) const {

    auto &executor = *(utils::downcast<const cl_stream_t *>(s)->cl_executor());
    if (aprop == prop_kind::forward) {
        copy_res_iter_kernel_.set_arg(0, ws);
        copy_res_iter_kernel_.set_arg(1, dst_last_iter);
        copy_res_iter_kernel_.set_arg(2, dst_last_iter_c);
        executor.parallel_for(cl_nd_range_t(
            {dic, batch, n_layer * n_direction}),
            copy_res_iter_kernel_);
    } else {
        copy_res_iter_kernel_.set_arg(0, ws);
        copy_res_iter_kernel_.set_arg(1, diff_src_iter);
        copy_res_iter_kernel_.set_arg(2, diff_src_iter_c);
        executor.parallel_for(cl_nd_range_t(
            {sic, batch, n_layer * n_direction}),
            copy_res_iter_kernel_);
    }
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
void _ref_rnn_common_t<aprop, src_type, weights_type>::ws_set(
    const stream_t *s, const memory_storage_t &workspace_,
    const cl_ulong ws_offset, const float val, const size_t size) const {

    ws_set_kernel_.set_arg(0, workspace_);
    ws_set_kernel_.set_arg(1, ws_offset);
    ws_set_kernel_.set_arg(2, val);
    auto &executor = *(utils::downcast<const cl_stream_t *>(s)->cl_executor());
    auto nd_range = cl_nd_range_t({size});
    executor.parallel_for(nd_range, ws_set_kernel_);
}
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
packing_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::pack_weights)) {
#if USE_MKL_PACKED_GEMM
// TBD
#else
    UNUSED(n_layer);
    UNUSED(n_direction);
    UNUSED(n_weights);
    UNUSED(n_gates);
    UNUSED(n_parts);
    UNUSED(gates_per_part);
    UNUSED(batch);
    UNUSED(OC_size);
    UNUSED(IC_size);
    UNUSED(weights_);
    UNUSED(w_);
    assert(!"packed gemm is disabled");
#endif
}

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
packing_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::no_pack_weights)) {
    AOC<size_t, 3> weights(weights_, n_layer, n_direction, n_parts);

    for (int i = 0; i < n_layer; i++) {
        for (int d = 0; d < n_direction; d++) {
            weights(i, d, 0) = OFF5(i, n_layer, d, n_direction, 0, OC_size,
                    0, n_gates, 0, IC_size);
            for (int p = 1; p < n_parts; p++) {
               weights(i, d, p) = OFF5(i, n_layer, d, n_direction, 0, OC_size,
                       gates_per_part[p-1], n_gates, 0, IC_size);
            }
        }
    }
}
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
free_packed_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::free_packed_weights)) {
#if USE_MKL_PACKED_GEMM
// TBD
#else
    UNUSED(n_layer);
    UNUSED(n_direction);
    UNUSED(n_parts);
    UNUSED(weights_);
    assert(!"packed gemm is disabled");
#endif
}
template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
free_packed_sig((_ref_rnn_common_t<aprop, src_type, weights_type>::free_no_packed_weights)) {
    UNUSED(n_layer);
    UNUSED(n_direction);
    UNUSED(n_parts);
    UNUSED(weights_);
}

//********************* Execution function *********************//

template <prop_kind_t aprop, data_type_t src_type, data_type_t weights_type>
status_t _ref_rnn_common_t<aprop, src_type, weights_type>::execute_(
    const exec_ctx_t &ctx) const {

    auto rnn = this->pd();

    int n_layer = rnn->L();
    int n_direction = rnn->D();
    int n_iter = rnn->T();
    int n_gates = rnn->G();
    int n_bias = n_gates + rnn->is_lbr();
    int n_states = rnn->jrnn_.n_states;
    int n_weights_input = rnn->SLC();
    int n_weights_state = rnn->SIC();
    int batch = rnn->MB();
    int slc = rnn->SLC();
    int sic = rnn->SIC();
    int dic = rnn->DIC();
    int dlc = rnn->DLC();
    int wic = nstl::max(slc, nstl::max(sic, dic));
    bool is_orig_gru = rnn->cell_kind()
        == alg_kind::vanilla_gru;
    int n_parts_wei_st = is_orig_gru ? 2 : 1, n_parts_wei_i = 1;
    int parts_wei_st = n_gates, parts_wei_i = n_gates,
        parts_wei_st_gru[2] = {2, 1};
    bool is_fwd = aprop == prop_kind::forward;

    stream_t *s = ctx.stream();

    auto &input_native_ = CTX_IN_STORAGE(MKLDNN_ARG_SRC_LAYER);
    auto &states_native_ = CTX_IN_STORAGE(MKLDNN_ARG_SRC_ITER);
    auto &c_states_native_ = CTX_IN_STORAGE(MKLDNN_ARG_SRC_ITER_C);
    auto &w_input_native_ = CTX_IN_STORAGE(MKLDNN_ARG_WEIGHTS_LAYER);
    auto &w_state_native_ = CTX_IN_STORAGE(MKLDNN_ARG_WEIGHTS_ITER);
    auto &bias_native_ = CTX_IN_STORAGE(MKLDNN_ARG_BIAS);

    auto &dst_last_layer_native_ = rnn->is_fwd()
        ? CTX_OUT_STORAGE(MKLDNN_ARG_DST_LAYER)
        : CTX_IN_STORAGE(MKLDNN_ARG_DST_LAYER);
    auto &dst_last_iter_native_ = rnn->is_fwd()
        ? CTX_OUT_STORAGE(MKLDNN_ARG_DST_ITER)
        : CTX_IN_STORAGE(MKLDNN_ARG_DST_ITER);
    auto &dst_last_iter_c_native_ = rnn->is_fwd()
        ? CTX_OUT_STORAGE(MKLDNN_ARG_DST_ITER_C)
        : CTX_IN_STORAGE(MKLDNN_ARG_DST_ITER_C);

    auto &diff_dst_layer_native_ = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST_LAYER);
    auto &diff_dst_iter_native_ = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST_ITER);
    auto &diff_dst_iter_c_native_ = CTX_IN_STORAGE(MKLDNN_ARG_DIFF_DST_ITER_C);

    auto &workspace_ = is_training(*pd())
        ? rnn->is_fwd()
            ? CTX_OUT_STORAGE(MKLDNN_ARG_WORKSPACE)
            : CTX_IN_STORAGE(MKLDNN_ARG_WORKSPACE)
#if !EMULATED_SCRATCHPAD
        : CTX_IN_STORAGE(MKLDNN_ARG_SCRATCHPAD);
#else
        : *scratchpad_;
#endif

    auto &diff_src_layer_native_ = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC_LAYER);
    auto &diff_src_iter_native_ = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC_ITER);
    auto &diff_src_iter_c_native_ = CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_SRC_ITER_C);

    auto &diff_weights_layer_native_ =
        CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_WEIGHTS_LAYER);
    auto &diff_weights_iter_native_ =
        CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_WEIGHTS_ITER);
    auto &diff_bias_native_ =
        CTX_OUT_STORAGE(MKLDNN_ARG_DIFF_BIAS);

    (void)is_fwd;
    (void)dlc;
    (void)parts_wei_st;
    (void)parts_wei_i;
    (void)parts_wei_st_gru;
    (void)n_weights_input;
    (void)n_weights_state;

    auto prints = [=](void){
    DPRINT("\n%s\n","+++++++++++++++");
    DPRINT(" aprop = %d\n",(int)aprop);
    DPRINT("%s\n","+++++++++++++++");
    DPRINT("  n_layer         = %d\n",n_layer        );
    DPRINT("  n_direction     = %d\n",n_direction    );
    DPRINT("  n_iter          = %d\n",n_iter         );
    DPRINT("  n_gates         = %d\n",n_gates        );
    DPRINT("  n_bias          = %d\n",n_bias         );
    DPRINT("  n_states        = %d\n",n_states       );
    DPRINT("  n_weights_input = %d\n",n_weights_input);
    DPRINT("  n_weights_state = %d\n",n_weights_state);
    DPRINT("  batch           = %d\n",batch          );
    DPRINT("  slc             = %d\n",slc            );
    DPRINT("  sic             = %d\n",sic            );
    DPRINT("  dic             = %d\n",dic            );
    DPRINT("  dlc             = %d\n",dlc            );
    DPRINT("  wic             = %d\n",wic            );
    DPRINT("  n_parts_wei_st  = %d\n",n_parts_wei_st );
    DPRINT("  parts_wei_st    = %d\n",parts_wei_st   );
    DPRINT("%s\n","+++++++++++++++");
    DPRINT("  is_fwd          = %s\n",is_fwd      ?"yes":"no");
    DPRINT("  is_orig_gru     = %s\n",is_orig_gru ?"yes":"no");
    DPRINT("  use_scratchpad_ = %s\n",use_scratchpad_ ?"yes":"no");
    DPRINT("%s\n","+++++++++++++++");
    DPRINT("  with_src_iter   = %s\n",rnn->with_src_iter() ?"yes":"no");
    DPRINT("  with_src_iter_c = %s\n",rnn->with_src_iter_c() ?"yes":"no");
    DPRINT("  with_bias       = %s\n",rnn->with_bias() ?"yes":"no");
    DPRINT("  with_dst_iter   = %s\n",rnn->with_dst_iter() ?"yes":"no");
    DPRINT("  with_dst_iter_c = %s\n",rnn->with_dst_iter_c() ?"yes":"no");
    DPRINT("%s\n","+++++++++++++++");
    };

#ifdef DEBUGPRINT
    prints();
#else
    UNUSED(prints);
#endif

    // initialize diff_state to 0
    if (aprop == prop_kind::backward) {
        ws_set(s, workspace_, ws_diff_states_offset_, 0.0f,
                ws_diff_states_size(*pd()));
    }

    // TODO: implement without copies
    bool is_lr = !one_of(exec_dir, b2t_r2l, t2b_r2l);
    bool is_rl = !one_of(exec_dir, b2t_l2r, t2b_l2r);

    // XXX: this function is used for calculating offsets for buffers and not
    // used for packing weights
    (this->*weights_state_pack_func)(n_layer, n_direction, n_weights_state,
            n_gates, batch, dic, sic, offset_wei_state_, n_parts_wei_st,
            (is_orig_gru ? parts_wei_st_gru : &parts_wei_st), w_state_native_);
    (this->*weights_input_pack_func)(n_layer, n_direction, n_weights_input,
            n_gates, batch, dic, slc, offset_wei_input_, n_parts_wei_i,
            &parts_wei_i, w_input_native_);

    // we first need to copy the initial states and input into ws
    copy_init_layer(s, is_lr, is_rl, n_iter, batch, slc,
        workspace_, input_native_, diff_dst_layer_native_);
    copy_init_iter(s, n_layer, n_direction, batch, sic, dic,
        workspace_, states_native_, c_states_native_, diff_dst_iter_native_,
        diff_dst_iter_c_native_);

    // run the execution on the grid
    (this->*grid_computation)(
            ctx,
            dic, slc, sic, wic, batch, n_layer, n_direction,
            n_iter, n_gates, n_states, n_bias,
            offset_wei_input_, n_parts_wei_i,
            offset_wei_state_, n_parts_wei_st,
            bias_native_,
            workspace_,
            w_input_native_,
            w_state_native_,
            diff_weights_layer_native_,
            diff_weights_iter_native_,
            diff_bias_native_
            );

    // Finally we copy the results to the result buffers

    copy_res_layer(s, is_lr, is_rl, n_iter, batch, slc, dic,
        dst_last_layer_native_, diff_src_layer_native_, workspace_);
    copy_res_iter(s, n_layer, n_direction, batch, sic, dic,
        dst_last_iter_native_, dst_last_iter_c_native_, diff_src_iter_native_,
        diff_src_iter_c_native_,workspace_);

    /* NOT USED YET */
#if 0
    // We free the packed weights if they were packed internally
    // currently - not
    (this->*weights_state_free_packed_func)(n_layer, n_direction,
            n_parts_wei_st, offset_wei_state_);
    (this->*weights_input_free_packed_func)(n_layer, n_direction,
            n_parts_wei_i, offset_wei_input_);
#endif

    return status::success;
};

template
struct _ref_rnn_common_t<prop_kind::forward, data_type::f16, data_type::f16>;
template
struct _ref_rnn_common_t<prop_kind::forward, data_type::f32, data_type::f32>;
template
struct _ref_rnn_common_t<prop_kind::backward, data_type::f32, data_type::f32>;
}
}
}

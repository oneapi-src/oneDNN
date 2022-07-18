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

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "shuffle/graph_shuffle.hpp"
#include "shuffle/shuffle.hpp"

namespace benchdnnext {
namespace shuffle {

static int check_known_skipped_case_graph(
        const ::shuffle::prb_t *prb, res_t *res) noexcept {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::shuffle::init_pd, prb, res), WARN);

    auto const_pd = query_pd(prim);
    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }
    return OK;
}

fill_status_t append_graph_with_block(const ::shuffle::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto data_type = convert_dt(prb->dt);
    const int64_t axis = prb->axis;
    const std::string tag = prb->tag;
    int64_t group;
    if (prb->dir & FLAG_FWD) {
        group = prb->group;
    } else {
        // example (axis = 1)
        //
        //     | (4, '8', 4, 4)
        //  reshape
        //     | (4, '2', '4', 4, 4)
        // transpose
        //     | (4, '4', '2', 4, 4)
        //  reshape
        //     | (4, '8', 4, 4)
        //
        // If we look at this pattern from up to bottom, then groups_fwd = 4,
        // from bottom to up however, groups_bwd = 2 = channel_dim / groups_fwd
        group = prb->dims[axis] / prb->group;
    }

    // reshape0
    const auto reshape0_src_dims = prb->dims;
    // - dst dims should be same as src dims except at 'channel' axis
    // - 'channel' value should be replaced with (C / g, g),
    // therefore shape attr will have one more dimension than the src dims.
    auto reshape0_dst_dims = prb->dims;
    reshape0_dst_dims[axis] /= group;
    reshape0_dst_dims.insert(reshape0_dst_dims.begin() + axis + 1, group);

    // transpose
    auto transpose_dst_dims = reshape0_dst_dims;
    std::swap(transpose_dst_dims[axis], transpose_dst_dims[axis + 1]);
    // After reshape, we have to make a transposition of g and C / g.
    // To do that we have to do following steps:
    // - fill the order with n consecutive values, where n = input size - 1
    // - swap indices of 'channel' axis with successor axis
    std::vector<int64_t> transpose_order(transpose_dst_dims.size());
    std::iota(transpose_order.begin(), transpose_order.end(), 0);
    std::swap(transpose_order[axis], transpose_order[axis + 1]);

    // reshape1
    // input and output dims of the whole pattern must equal
    const auto reshape1_dst_dims = prb->dims;

    const auto reshape0_op_id = graph.generate_id_for(entry_kind::RESHAPE);
    const auto reshape0_src_id = graph.has_blocks()
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(reshape0_op_id, lt_kind::SRC);
    const auto reshape0_dst_id
            = graph.generate_id_for(reshape0_op_id, lt_kind::DST);

    graph.create_lt(reshape0_src_id, data_type, reshape0_src_dims, tag);
    graph.create_lt(reshape0_dst_id, data_type, reshape0_dst_dims, tag);

    dnnl::graph::op reshape0_op(reshape0_op_id,
            dnnl::graph::op::kind::StaticReshape,
            graph.stringify_id(reshape0_op_id));
    reshape0_op.set_attr("shape", reshape0_dst_dims)
            .set_attr("special_zero", false);

    graph.append(
            reshape0_op_id, reshape0_op, {reshape0_src_id}, {reshape0_dst_id});

    const auto transpose_op_id = graph.generate_id_for(entry_kind::TRANSPOSE);
    const auto transpose_dst_id
            = graph.generate_id_for(transpose_op_id, lt_kind::DST);

    graph.create_lt(transpose_dst_id, data_type, transpose_dst_dims, tag);

    dnnl::graph::op transpose_op(transpose_op_id,
            dnnl::graph::op::kind::StaticTranspose,
            graph.stringify_id(transpose_op_id));
    transpose_op.set_attr("order", transpose_order);

    graph.append(transpose_op_id, transpose_op, {reshape0_dst_id},
            {transpose_dst_id});

    const auto reshape1_op_id = graph.generate_id_for(entry_kind::RESHAPE);
    const auto reshape1_dst_id
            = graph.generate_id_for(reshape1_op_id, lt_kind::DST);

    graph.create_lt(reshape1_dst_id, data_type, reshape1_dst_dims, tag);

    dnnl::graph::op reshape1_op(reshape1_op_id,
            dnnl::graph::op::kind::StaticReshape,
            graph.stringify_id(reshape1_op_id));
    reshape1_op.set_attr("shape", reshape1_dst_dims)
            .set_attr("special_zero", false);

    graph.append(
            reshape1_op_id, reshape1_op, {transpose_dst_id}, {reshape1_dst_id});

    graph.close_block();

    return fill_status::DONE;
}

// In oneDNN Graph we use a specific
// combination of Reshape -> Transpose -> Reshape chain,
// to describe a single Shuffle op.
int doit(const ::shuffle::prb_t *prb, res_t *res) {
    res->impl_name = "graph";
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const auto status = append_graph_with_block(prb);
    if (status != fill_status::DONE
            && status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto &graph = graph_t::get();

    // Filter partitions
    const auto partitions = graph.get_partitions();
    if (partitions.empty() || partitions.size() > 1) {
        cleanup();
        return res->state = FAILED, FAIL;
    }

    const auto par = partitions[0];
    if (!par.is_supported()) {
        cleanup();
        return res->state = UNIMPLEMENTED, FAIL;
    }

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::shuffle::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], prb->tag);
    auto dst_dt = make_dnn_mem(outs[0], prb->tag);

    SAFE(::shuffle::fill_src(prb, src_dt, src_fp), WARN);

    const dnnl::graph::engine &eng = get_test_engine();

    dnnl::graph::tensor src_tensor(ins[0], eng, static_cast<void *>(src_dt));
    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));

    std::vector<dnnl::graph::tensor> tensors_in {src_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    if (is_bench_mode(CORR)) {
        args_t args, ref_args;

        if (prb->dir & FLAG_FWD) {
            args.set(DNNL_ARG_DST, dst_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);

            check_correctness(
                    prb, {DST}, args, ref_args, ::shuffle::setup_cmp, res);
        } else if (prb->dir & FLAG_BWD) {
            args.set(DNNL_ARG_DIFF_SRC, dst_dt);
            ref_args.set(DNNL_ARG_DIFF_DST, src_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, dst_fp);

            check_correctness(
                    prb, {SRC}, args, ref_args, ::shuffle::setup_cmp, res);
        } else {
            SAFE(FAIL, CRIT);
        }
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    cleanup();

    return OK;
}

} // namespace shuffle
} // namespace benchdnnext

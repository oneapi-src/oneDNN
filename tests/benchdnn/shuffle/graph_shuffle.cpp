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

shuffle_graph_prb_t::spec_t::spec_t(const ::shuffle::prb_t *prb) {
    dtype = convert_dt(prb->dt);
    axis = prb->axis;
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
        group = prb->dims[prb->axis] / prb->group;
    }
    raw_tag = prb->tag;

    // reshape0
    reshape0_src_dims = prb->dims;
    // - dst dims should be same as src dims except at 'channel' axis
    // - 'channel' value should be replaced with (C / g, g),
    // therefore shape attr will have one more dimension than the src dims.
    reshape0_dst_dims = prb->dims;
    reshape0_dst_dims[axis] /= group;
    reshape0_dst_dims.insert(reshape0_dst_dims.begin() + axis + 1, group);

    // transpose
    transpose_dst_dims = reshape0_dst_dims;
    std::swap(transpose_dst_dims[axis], transpose_dst_dims[axis + 1]);
    // After reshape, we have to make a transposition of g and C / g.
    // To do that we have to do following steps:
    // - fill the order with n consecutive values, where n = input size - 1
    // - swap indices of 'channel' axis with successor axis
    transpose_order.resize(transpose_dst_dims.size());
    std::iota(transpose_order.begin(), transpose_order.end(), 0);
    std::swap(transpose_order[axis], transpose_order[axis + 1]);

    // reshape1
    // input and output dims of the whole pattern must equal
    reshape1_dst_dims = prb->dims;
}

void check_known_skipped_case_graph(const ::shuffle::prb_t *prb, res_t *res) {
    ::shuffle::check_known_skipped_case(prb, res);
}

fill_status_t shuffle_graph_prb_t::handle_reshape_(int id) {
    if (!(id == 0 || id == 1)) return fill_status::UNSUPPORTED_OP;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["reshape" + std::to_string(id)].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string DST {TENSOR_ID + "_DST"};

    if (id == 0) {
        tensor_descs_.emplace(
                SRC, spec_.dtype, spec_.reshape0_src_dims, spec_.raw_tag);
        tensor_descs_.emplace(
                DST, spec_.dtype, spec_.reshape0_dst_dims, spec_.raw_tag);
    } else {
        tensor_descs_.emplace(
                DST, spec_.dtype, spec_.reshape1_dst_dims, spec_.raw_tag);
    }
    const auto src_tensor_desc = id == 0
            ? tensor_descs_[SRC]
            : tensor_descs_[curr_out_map_ids_.back() + "_DST"];
    dnnl::graph::op reshape(new_op_id, get_main_op_kind(), {src_tensor_desc},
            {tensor_descs_[DST]}, "reshape" + std::to_string(id));

    // set shape attr to be same as previously calculated dst dims
    const auto shape
            = id == 0 ? spec_.reshape0_dst_dims : spec_.reshape1_dst_dims;
    reshape.set_attr("shape", shape);
    reshape.set_attr("special_zero", false);

    ops_.emplace_back(reshape);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t shuffle_graph_prb_t::handle_transpose_() {
    using op = dnnl::graph::op;
    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["transpose"].push_back(TENSOR_ID);
    const std::string DST {TENSOR_ID + "_DST"};

    tensor_descs_.emplace(
            DST, spec_.dtype, spec_.transpose_dst_dims, spec_.raw_tag);

    op transpose(new_op_id, op::kind::StaticTranspose,
            {tensor_descs_[curr_out_map_ids_.back() + "_DST"]},
            {tensor_descs_[DST]}, "transpose");

    transpose.set_attr("order", spec_.transpose_order);

    ops_.emplace_back(transpose);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

// In oneDNN Graph we use a specific
// combination of Reshape -> Transpose -> Reshape chain,
// to describe a single Shuffle op.
// See @example cpu_shuffle_pattern_f32.cpp for more information.
int doit(const ::shuffle::prb_t *prb, res_t *res) {
    res->impl_name = "graph";
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    shuffle_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();

    // Filter partitions
    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::shuffle::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], prb->tag);
    auto dst_dt = make_dnn_mem(outs[0], prb->tag);

    SAFE(::shuffle::fill_src(prb, src_dt, src_fp), WARN);

    dnnl::graph::engine &eng = get_test_engine();

    dnnl::graph::tensor src_tensor(ins[0], eng, static_cast<void *>(src_dt));
    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));

    std::vector<dnnl::graph::tensor> tensors_in {src_tensor};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    if (is_bench_mode(CORR)) {
        args_t ref_args;
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);

        TIME_REF(::shuffle::compute_ref(prb, ref_args));

        compare::compare_t cmp;
        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    return OK;
}

} // namespace shuffle
} // namespace benchdnnext

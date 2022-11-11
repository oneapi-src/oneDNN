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

#include "resampling/graph_resampling.hpp"
#include "binary/binary.hpp"

#include <ctime>
#include <random>
#include <tuple>

namespace benchdnnext {
namespace resampling {

static int check_known_skipped_case_graph(
        const ::resampling::prb_t *prb, res_t *res) noexcept {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::resampling::init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    auto const_pd = query_pd(prim);
    if (check_mem_size(const_pd) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    //Skip if source and destination datatypes are different.
    if (prb->sdt != prb->ddt) {
        res->state = SKIPPED, res->reason = INVALID_CASE;
    }
    return OK;
}

static std::vector<int64_t> get_spatial_dims(const std::vector<int64_t> &dims) {
    std::vector<int64_t> new_dims = dims;
    new_dims.erase(new_dims.begin(), new_dims.begin() + 2);
    return new_dims;
}

static std::vector<int64_t> get_sizes(const std::vector<int64_t> &dst_dims) {
    return get_spatial_dims(dst_dims);
}

static std::vector<float> get_scales(const std::vector<int64_t> &src_dims,
        const std::vector<int64_t> &dst_dims) {
    const auto src_spatial_dims = get_spatial_dims(src_dims);
    const auto dst_spatial_dims = get_spatial_dims(dst_dims);
    if (src_spatial_dims.size() != dst_spatial_dims.size())
        return std::vector<float>();

    std::vector<float> scales_dims(src_spatial_dims.size());
    for (size_t i = 0; i < src_spatial_dims.size(); ++i) {
        scales_dims[i] = static_cast<float>(dst_spatial_dims[i])
                / static_cast<float>(src_spatial_dims[i]);
    }

    return scales_dims;
}

fill_status_t append_graph_with_block(
        const ::resampling::prb_t *prb, int rand_testmode) {
    graph_t &graph = graph_t::get();

    const auto connect_to_previous_block = graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::RESAMPLING);
    const auto dst_lt_kind
            = prb->dir & FLAG_FWD ? lt_kind::DST : lt_kind::DIFF_DST;
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, lt_kind::SRC);
    const auto dst_id = graph.generate_id_for(op_id, dst_lt_kind);

    const auto src_dt = convert_dt(prb->sdt);
    const auto dst_dt = convert_dt(prb->ddt);

    graph.create_lt(src_id, src_dt, prb->src_dims(), prb->tag);
    graph.create_lt(dst_id, dst_dt, prb->dst_dims(), prb->tag);

    std::vector<size_t> src_ids {src_id};
    std::vector<size_t> dst_ids {};
    dnnl::graph::op::kind resampling_kind;
    if (prb->dir & FLAG_FWD) {
        dst_ids.push_back(dst_id);
        resampling_kind = dnnl::graph::op::kind::Interpolate;
    } else {
        const auto d_src_id = graph.generate_id_for(op_id, lt_kind::DIFF_SRC);
        graph.create_lt(d_src_id, src_dt, prb->src_dims(), prb->tag);
        src_ids.push_back(dst_id);
        dst_ids.push_back(d_src_id);
        resampling_kind = dnnl::graph::op::kind::InterpolateBackprop;
    }
    if (rand_testmode == test_mode_t::SIZES_INPUT_TENSOR) {
        const auto size_id = graph.generate_id_for(op_id, lt_kind::SRC1);
        graph.create_lt(size_id, dnnl::graph::logical_tensor::data_type::s32,
                {1}, tag::x);
        src_ids.push_back(size_id);
    }

    dnnl::graph::op resampling_op(
            op_id, resampling_kind, graph.stringify_id(op_id));
    resampling_op.set_attr("data_format", std::string("NCX"))
            .set_attr("mode", std::string(alg2str(prb->alg)));
    if (rand_testmode == test_mode_t::SIZES_ATTR)
        resampling_op.set_attr("sizes", get_sizes(prb->dst_dims()));
    else if (rand_testmode == test_mode_t::SCALES_ATTR)
        resampling_op.set_attr(
                "scales", get_scales(prb->src_dims(), prb->dst_dims()));

    graph.append(op_id, resampling_op, src_ids, dst_ids);

    // handle post ops
    fill_status_t status;
    for (const auto &entry : prb->attr.post_ops.entry) {
        if (entry.is_binary_kind()) {
            std::tie(status, std::ignore) = append_graph_with_binary(entry);
            BENCHDNNEXT_VERIFY(status);
        } else if (entry.is_eltwise_kind()) {
            status = append_graph_with_eltwise(entry);
            BENCHDNNEXT_VERIFY(status);
        } else if (entry.is_sum_kind()) {
            std::tie(status, std::ignore) = append_graph_with_sum(entry);
            BENCHDNNEXT_VERIFY(status);
        }
    }

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::resampling::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    // TODO: test mode should be fixed
    srand(std::time(NULL));
    const int rand_testmode = (rand() % 2);

    const auto status = append_graph_with_block(prb, rand_testmode);
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

    auto cp = compile_partition(
            ::resampling::init_pd, prb, res, par, ins, outs);
    const auto cp_dst_lt = cp.query_logical_tensor(outs[0].get_id());

    const dnnl::graph::engine &eng = get_test_engine();
    auto src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(cp_dst_lt, dt::f32, tag::abx);

    auto src_dt = make_dnn_mem(ins[0], prb->tag);
    auto dst_dt = make_dnn_mem(cp_dst_lt, prb->tag);
    if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM) >= 0)
        SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    // When post-ops occur, the relative difference can change
    // between the output from reference and the kernel. The compare
    // function usually uses to compare a relative difference.
    // Therefore, we should not lead to a situation where the
    // relative difference is very small after executing a
    // post-ops operation. Therefore, all values for binary post_ops
    // are positive when the linear algorithm is present. This is
    // important because there may be small differences in the result
    // between the expected value and the gotten value with this algorithm.
    const bool only_positive_values = prb->alg == ::resampling::linear;

    size_t idx_ins = rand_testmode == test_mode_t::SIZES_INPUT_TENSOR ? 1 : 0;
    const auto post_bin_indices
            = get_post_bin_indices(prb->attr.post_ops.entry);

    for (size_t i = 0; i < post_bin_indices.size(); ++i) {
        binary_po_fp.emplace_back(
                make_dnn_mem(ins[++idx_ins], dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins[idx_ins], prb->tag));
        const int po_idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                   static_cast<int>(post_bin_indices[i]))
                | DNNL_ARG_SRC_1;
        ::binary::fill_mem(
                po_idx, binary_po_dt[i], binary_po_fp[i], only_positive_values);
        binary_po_args.push_back(po_idx);
    }

    SAFE(::resampling::fill_src(prb, src_dt, src_fp, res), WARN);

    if (prb->dir & FLAG_FWD) {
        dnnl::graph::tensor src_tensor(
                ins[0], eng, static_cast<void *>(src_dt));
        dnnl::graph::tensor dst_tensor(
                cp_dst_lt, eng, static_cast<void *>(dst_dt));

        std::vector<dnnl::graph::tensor> tensors_in {src_tensor};
        dnnl::graph::tensor sizes_tensor;
        std::vector<int64_t> sizes_v = get_sizes(prb->dst_dims());
        idx_ins = 0;
        if (rand_testmode == test_mode_t::SIZES_INPUT_TENSOR) {
            sizes_tensor = dnnl::graph::tensor(
                    ins[++idx_ins], eng, static_cast<void *>(sizes_v.data()));
            tensors_in.emplace_back(sizes_tensor);
        }
        size_t bin_dt_idx = 0;
        for (const auto &po_entry : prb->attr.post_ops.entry) {
            if (po_entry.is_binary_kind()) {
                dnnl::graph::tensor bin_tensor(ins[++idx_ins], eng,
                        static_cast<void *>(binary_po_dt[bin_dt_idx++]));
                tensors_in.emplace_back(bin_tensor);
            } else if (po_entry.is_sum_kind()) {
                dnnl::graph::tensor sum_src1_tensor(
                        ins[++idx_ins], eng, static_cast<void *>(dst_dt));
                tensors_in.emplace_back(sum_src1_tensor);
            }
        }
        std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args_t args, ref_args;

            args.set(DNNL_ARG_DST, dst_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(binary_po_args, binary_po_fp);

            check_correctness(
                    prb, {DST}, args, ref_args, ::resampling::setup_cmp, res);
        }

        SAFE(measure_perf(
                     res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
                WARN);
    } else {
        auto d_dst_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
        auto d_dst_dt = make_dnn_mem(ins[1], prb->tag);

        dnnl::graph::tensor src_tensor(
                ins[0], eng, static_cast<void *>(src_dt));

        SAFE(::resampling::fill_dst(prb, d_dst_dt, d_dst_fp, res), WARN);
        dnnl::graph::tensor d_dst_tensor(
                ins[1], eng, static_cast<void *>(d_dst_dt));
        dnnl::graph::tensor d_src_tensor(
                outs[0], eng, static_cast<void *>(dst_dt));

        std::vector<dnnl::graph::tensor> tensors_in {src_tensor};
        dnnl::graph::tensor sizes_tensor;
        std::vector<int64_t> sizes_v = get_sizes(prb->dst_dims());
        if (rand_testmode == test_mode_t::SIZES_INPUT_TENSOR) {
            sizes_tensor = dnnl::graph::tensor(
                    ins[2], eng, static_cast<void *>(sizes_v.data()));
            tensors_in.emplace_back(sizes_tensor);
        }
        tensors_in.emplace_back(d_dst_tensor);

        std::vector<dnnl::graph::tensor> tensors_out {d_src_tensor};

        SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

        if (is_bench_mode(CORR)) {
            args_t args, ref_args;

            args.set(DNNL_ARG_DIFF_SRC, dst_dt);
            ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
            ref_args.set(DNNL_ARG_DIFF_SRC, dst_fp);

            check_correctness(
                    prb, {SRC}, args, ref_args, ::resampling::setup_cmp, res);
        }

        SAFE(measure_perf(
                     res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
                WARN);
    }

    cleanup();

    return OK;
}

} // namespace resampling
} // namespace benchdnnext

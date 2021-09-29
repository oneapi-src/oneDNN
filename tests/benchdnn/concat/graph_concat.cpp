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

#include <vector>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "concat/concat.hpp"
#include "concat/graph_concat.hpp"

namespace benchdnnext {
namespace concat {

concat_graph_prb_t::spec_t::spec_t(const ::concat::prb_t *prb) {
    src_dims.resize(prb->n_inputs());
    raw_src_tag.resize(prb->n_inputs());

    for (auto i = 0; i < prb->n_inputs(); ++i) {
        src_dims[i] = prb->sdims[i];
        raw_src_tag[i] = prb->stag[i];
    }

    dst_dims = prb->ddims;
    raw_dst_tag = prb->dtag;

    src_dt = convert_dt(prb->sdt);
    dst_dt = convert_dt(prb->ddt);

    axis = prb->axis;
}

void check_ranks_shapes_and_dtypes(const ::concat::prb_t *prb, res_t *res) {
    auto rank = prb->sdims[0].size();
    auto axis = prb->axis;

    // In oneDNN graph axis can take negative values. But in oneDNN
    // it has to have values greater or equal to 0. Therefore for consistency,
    // we create a condition, that the axis has a range of values
    // the same as the native benchdnn code.
    // Axis should also be smaller than the rank.
    const auto axis_in_range = (axis >= 0 || axis < static_cast<int64_t>(rank));
    if (!axis_in_range) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    // Rank of all tensors should match
    const auto same_rank = std::all_of(prb->sdims.cbegin(), prb->sdims.cend(),
            [rank](const dims_t &sdim) { return rank == sdim.size(); });
    if (!same_rank) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    // Shapes for all inputs should match at every position except axis position
    for (auto i = 0; i < rank; ++i) {
        if (axis == static_cast<int64_t>(i)) continue;
        const auto d = prb->ddims[i];
        const auto same_dim
                = std::all_of(prb->sdims.cbegin(), prb->sdims.cend(),
                        [d, i](const dims_t &sdim) { return d == sdim[i]; });
        if (!same_dim) {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }

    const auto dst_axis_dim = prb->ddims[axis];
    const auto src_axis_dim_sum = std::accumulate(prb->sdims.cbegin(),
            prb->sdims.cend(), 0L, [axis](int64_t acc, const dims_t &sdim) {
                return acc + sdim[axis];
            });
    if (dst_axis_dim != src_axis_dim_sum) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }

    // Types of all tensors should match
    const auto same_dtypes = prb->sdt == prb->ddt;
    if (!same_dtypes) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

void check_known_skipped_case_graph(const ::concat::prb_t *prb, res_t *res) {
    // srcs and dst should have same tags
    const auto tag = prb->dtag;
    const auto same_tags = std::all_of(prb->stag.cbegin(), prb->stag.cend(),
            [tag](const std::string &stag) { return !tag.compare(stag); });
    if (!same_tags) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

fill_status_t concat_graph_prb_t::handle_main_op_() {
    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string DST {TENSOR_ID + "_DST"};

    std::vector<dnnl::graph::logical_tensor> tensor_descs_srcs;
    tensor_descs_srcs.reserve(spec_.n_inputs());

    for (auto i = 0; i < spec_.n_inputs(); ++i) {
        const auto SRC_I = SRC + std::to_string(i);
        tensor_descs_.emplace(
                SRC_I, spec_.src_dt, spec_.src_dims[i], spec_.raw_src_tag[i]);
        tensor_descs_srcs.emplace_back(tensor_descs_[SRC_I]);
    }
    tensor_descs_.emplace(DST, spec_.dst_dt, spec_.dst_dims, spec_.raw_dst_tag);

    dnnl::graph::op concat(new_op_id, get_main_op_kind(), tensor_descs_srcs,
            {tensor_descs_[DST]}, "concat");

    concat.set_attr("axis", spec_.axis);

    ops_.emplace_back(concat);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

int doit(const ::concat::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    ::concat::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;
    check_ranks_shapes_and_dtypes(prb, res);
    if (res->state == SKIPPED) return OK;

    concat_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();
    const auto spec = graph_prb.spec();

    // // Filter partitions
    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::concat::init_pd, prb, res, par, ins, outs);

    auto dst_fp = make_dnn_mem(outs[0], spec.dst_dims, dt::f32, tag::abx);
    auto dst_dt = make_dnn_mem(outs[0], spec.dst_dims, spec.raw_dst_tag);

    std::vector<dnn_mem_t> src_fp, src_dt;
    src_fp.reserve(prb->n_inputs());
    src_dt.reserve(prb->n_inputs());

    std::vector<dnnl::graph::tensor> src_tensors {};
    src_tensors.reserve(prb->n_inputs());

    dnnl::graph::engine &eng = get_test_engine();

    for (auto i = 0; i < prb->n_inputs(); ++i) {
        src_fp.emplace_back(
                make_dnn_mem(ins[i], spec.src_dims[i], dt::f32, tag::abx));
        src_dt.emplace_back(
                make_dnn_mem(ins[i], spec.src_dims[i], spec.raw_src_tag[i]));

        SAFE(::concat::fill_src(i, prb->ddt, src_dt[i], src_fp[i]), WARN);

        src_tensors.emplace_back(dnnl::graph::tensor(
                ins[i], eng, static_cast<void *>(src_dt[i])));
    }

    dnnl::graph::tensor dst_tensor(outs[0], eng, static_cast<void *>(dst_dt));

    std::vector<dnnl::graph::tensor> tensors_in {src_tensors};
    std::vector<dnnl::graph::tensor> tensors_out {dst_tensor};

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (is_bench_mode(CORR)) {
        ::concat::compute_ref(prb, src_fp, dst_fp);
        compare::compare_t cmp;
        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }

    SAFE(measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out),
            WARN);

    return OK;
}

} // namespace concat
} // namespace benchdnnext

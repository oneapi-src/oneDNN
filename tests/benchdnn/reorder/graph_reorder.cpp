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

#include <utility>
#include "utils/compare.hpp"

#include "reorder/graph_reorder.hpp"

namespace benchdnnext {
namespace reorder {

reorder_graph_prb_t::spec_t::spec_t(const ::reorder::prb_t *prb) noexcept {
    dims = prb->reorder.dims;
    src_dt = convert_dt(prb->conf_in->dt);
    dst_dt = convert_dt(prb->conf_out->dt);
    src_tag = convert_tag(prb->reorder.tag_in);
    dst_tag = convert_tag(prb->reorder.tag_out);
}

void check_known_skipped_case_graph(
        const ::reorder::prb_t *prb, res_t *res) noexcept {
    if (prb->conf_in->dt != prb->conf_out->dt) { res->state = SKIPPED; }
}

fill_status_t reorder_graph_prb_t::handle_main_op_(
        const std::string &tag_in, const std::string &tag_out) {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    const std::string SRC {TENSOR_ID + "_SRC"};
    const std::string DST {TENSOR_ID + "_DST"};

    //TODO: how to pass layout_id??
    tensor_descs_.emplace(SRC, spec_.src_dt, spec_.dims,
            calculate_strides(spec_.dims, spec_.src_dt, tag_in));
    tensor_descs_.emplace(DST, spec_.dst_dt, spec_.dims,
            calculate_strides(spec_.dims, spec_.dst_dt, tag_out));

    op reorder_op(new_op_id, op::kind::Reorder, {tensor_descs_[SRC]},
            {tensor_descs_[DST]}, "reorder");

    ops_.emplace_back(reorder_op);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

int doit(const ::reorder::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    ::reorder::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    reorder_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();
    const auto spec = graph_prb.spec();

    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::reorder::init_pd, prb, res, par, ins, outs);

    auto src_fp
            = make_dnn_mem(ins[0], spec.src_dt, (prb->reorder.tag_in).c_str());
    // we need src_fp for proper comparison, => no in-place reference
    auto dst_fp = make_dnn_mem(
            outs[0], spec.dst_dt, (prb->reorder.tag_out).c_str());

    auto src_dt = make_dnn_mem(ins[0], (prb->reorder.tag_in).c_str());
    auto dst_dt = make_dnn_mem(outs[0], (prb->reorder.tag_out).c_str());

    //TODO: need to extend for post ops
    SAFE(fill_memory(prb, SRC, src_fp), WARN);
    SAFE(src_dt.reorder(src_fp), WARN);

    //TODO: fill for sum / zeropoints

    std::vector<dnnl::graph::tensor> tensors_in;
    std::vector<dnnl::graph::tensor> tensors_out;
    tensors_in.emplace_back(
            dnnl::graph::tensor(ins[0], static_cast<void *>(src_dt)));
    tensors_out.emplace_back(
            dnnl::graph::tensor(outs[0], static_cast<void *>(dst_dt)));

    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (is_bench_mode(CORR)) {
        //TODO: do we need runtime compensation??
        SAFE(ref_reorder(prb, dst_fp, src_fp), WARN);

        compare::compare_t cmp;
        const bool has_s32 = spec.src_dt == dt::s32 || spec.dst_dt == dt::s32;
        const bool has_s8 = spec.src_dt == dt::s8 || spec.dst_dt == dt::s8;
        const bool has_u8 = spec.src_dt == dt::u8 || spec.dst_dt == dt::u8;
        if (has_u8)
            cmp.set_zero_trust_percent(58.f); // 4/7 inputs becomes 0
        else if (has_s32 || has_s8)
            cmp.set_zero_trust_percent(43.f); // 3/7 inputs becomes 0

        // A hack to avoid false-positive result from f32->s32 conversion
        // in case of sum post-op on GPU happening when two max_dt values
        // are summed together.
        using cmp_args_t = compare::compare_t::driver_check_func_args_t;
        const auto reorder_add_check = [&](const cmp_args_t &args) {
            if (args.dt == dnnl_s32 && args.got == max_dt(args.dt)
                    && is_gpu()) {
                // 128.f = float(INT_MAX)
                //                - BENCHDNN_S32_TO_F32_SAT_CONST;
                return args.diff == 128.f;
            }
            return false;
        };
        cmp.set_driver_check_function(reorder_add_check);

        // TODO: enable additional checks for border values validity.
        SAFE(cmp.compare(dst_fp, dst_dt, prb->attr, res), WARN);
    }
    SAFE(measure_perf(res->timer, cp, tensors_in, tensors_out), WARN);

    return OK;
}

} // namespace reorder
} // namespace benchdnnext

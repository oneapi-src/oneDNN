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
#include "oneapi/dnnl/dnnl_types.h"

#include "dnn_graph_types.hpp"
#include "dnnl_graph_common.hpp"
#include "utils/compare.hpp"

#include "binary/binary.hpp"
#include "pool/graph_pool.hpp"

#include <algorithm>

namespace benchdnnext {
namespace pool {

pool_graph_prb_t::spec_t::spec_t(const ::pool::prb_t *prb) noexcept {
    using graph_op = dnnl::graph::op::kind;

    const dim_t src_1d_dims[] = {prb->mb, prb->ic, prb->iw};
    const dim_t src_2d_dims[] = {prb->mb, prb->ic, prb->ih, prb->iw};
    const dim_t src_3d_dims[] = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};

    const dim_t dst_1d_dims[] = {prb->mb, prb->ic, prb->ow};
    const dim_t dst_2d_dims[] = {prb->mb, prb->ic, prb->oh, prb->ow};
    const dim_t dst_3d_dims[] = {prb->mb, prb->ic, prb->od, prb->oh, prb->ow};

    switch (prb->ndims) {
        case 5: {
            src_dims.assign(src_3d_dims, end(src_3d_dims));
            dst_dims.assign(dst_3d_dims, end(dst_3d_dims));
        } break;

        case 4: {
            src_dims.assign(src_2d_dims, end(src_2d_dims));
            dst_dims.assign(dst_2d_dims, end(dst_2d_dims));
        } break;

        case 3: {
            src_dims.assign(src_1d_dims, end(src_1d_dims));
            dst_dims.assign(dst_1d_dims, end(dst_1d_dims));
        } break;

        default: break;
    }

    src_dt = convert_dt(prb->cfg[SRC].dt);
    dst_dt = convert_dt(prb->cfg[DST].dt);

    const dim_t strides_nd[] = {prb->sd, prb->sh, prb->sw};
    const dim_t kernel_nd[] = {prb->kd, prb->kh, prb->kw};
    const dim_t padding_l_nd[] = {prb->pd, prb->ph, prb->pw};
    const dim_t padding_r_nd[] = {prb->pd_r, prb->ph_r, prb->pw_r};

    const size_t max_ndims = 5;
    const size_t offset = max_ndims - prb->ndims;

    strides.assign(strides_nd + offset, end(strides_nd));
    kernel.assign(kernel_nd + offset, end(kernel_nd));
    pads_begin.assign(padding_l_nd + offset, end(padding_l_nd));
    pads_end.assign(padding_r_nd + offset, end(padding_r_nd));

    rounding_type = "floor";
    data_format = "NCX";
    raw_data_format = prb->tag;
    is_fwd = prb->dir & FLAG_FWD;

    if (is_fwd) {
        op_kind = (prb->alg == ::pool::max) ? graph_op::MaxPool
                                            : graph_op::AvgPool;
        op_name = (prb->alg == ::pool::max) ? "max_pool" : "avg_pool";
    } else {
        op_kind = (prb->alg == ::pool::max) ? graph_op::MaxPoolBackprop
                                            : graph_op::AvgPoolBackprop;
        op_name = (prb->alg == ::pool::max) ? "max_pool_bwd" : "avg_pool_bwd";
    }

    // attributes specific to pooling type
    const dim_t dilation_nd[] = {prb->dd, prb->dh, prb->dw};
    dilations.assign(dilation_nd + offset, end(dilation_nd));
    // "0" dilations in oneDNN is "1" in oneDNN graph
    std::for_each(dilations.begin(), dilations.end(), [](dim_t &v) { v += 1; });

    exclude_pad = prb->alg == ::pool::avg_np;
}

void check_known_skipped_case_graph(
        const ::pool::prb_t *prb, res_t *res) noexcept {
    ::pool::check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return;

    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.is_binary_kind()) {
            // currently, in the backend there are supported
            // only two policies for binary post op:
            // COMMON and PER_OC
            const auto policy = po.binary.policy;
            if (!(policy == attr_t::COMMON || policy == attr_t::PER_OC)) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }

            // currently, for int8 cases, in the backend we
            // support only int8 data type for 2nd binary input
            const dt src_dt = convert_dt(prb->cfg[SRC].dt);
            const dt dst_dt = convert_dt(prb->cfg[DST].dt);
            const dt bin_src_dt = convert_dt(po.binary.src1_dt);
            if (is_low_precision({src_dt, dst_dt})
                    && !is_low_precision({bin_src_dt})) {
                res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
                return;
            }
        }
    }
}

fill_status_t pool_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const size_t new_op_id = ops_.size();
    const std::string TENSOR_ID = std::to_string(new_op_id);
    tensor_id["main"].push_back(TENSOR_ID);
    std::vector<dnnl::graph::logical_tensor> inputs {};
    std::vector<dnnl::graph::logical_tensor> outputs {};

    dt src_dt;
    dt dst_dt;
    if (benchdnnext::is_low_precision({spec_.src_dt, spec_.dst_dt})) {
        src_dt = dt::f32;
        dst_dt = dt::f32;
    } else {
        src_dt = spec_.src_dt;
        dst_dt = spec_.dst_dt;
    }

    const std::string SRC {TENSOR_ID + "_SRC"};
    tensor_descs_.emplace(SRC, src_dt, spec_.src_dims, spec_.raw_data_format);
    if (spec_.is_fwd) {
        const std::string DST {TENSOR_ID + "_DST"};
        tensor_descs_.emplace(
                DST, dst_dt, spec_.dst_dims, spec_.raw_data_format);
        inputs = {tensor_descs_[SRC]};
        outputs = {tensor_descs_[DST]};
    } else {
        const std::string DIFF_DST {TENSOR_ID + "_DIFF_DST"};
        const std::string DIFF_SRC {TENSOR_ID + "_DIFF_SRC"};
        tensor_descs_.emplace(
                DIFF_DST, dst_dt, spec_.dst_dims, spec_.raw_data_format);
        tensor_descs_.emplace(
                DIFF_SRC, src_dt, spec_.src_dims, spec_.raw_data_format);
        outputs = {tensor_descs_[DIFF_SRC]};
        if (spec_.op_kind == op::kind::AvgPoolBackprop) {
            inputs = {tensor_descs_[DIFF_DST]};
        } else {
            if (is_bench_mode(PERF)) {
                const std::string DST_FWD_I {TENSOR_ID + "_DST_FWD_I"};
                tensor_descs_.emplace(DST_FWD_I, dt::s32, spec_.dst_dims,
                        spec_.raw_data_format);
                inputs = {tensor_descs_[SRC], tensor_descs_[DIFF_DST],
                        tensor_descs_[DST_FWD_I]};
            } else { //workaround for correctness test
                inputs = {tensor_descs_[SRC], tensor_descs_[DIFF_DST]};
            }
        }
    }

    op pool(new_op_id, spec_.op_kind, inputs, outputs, spec_.op_name);

    pool.set_attr("strides", spec_.strides)
            .set_attr("pads_begin", spec_.pads_begin)
            .set_attr("pads_end", spec_.pads_end)
            .set_attr("kernel", spec_.kernel)
            .set_attr("data_format", spec_.data_format)
            .set_attr("auto_pad", spec_.auto_pad);

    if (spec_.op_kind == op::kind::MaxPool
            || spec_.op_kind == op::kind::MaxPoolBackprop) {
        pool.set_attr("dilations", spec_.dilations);
    } else { // AvgPool
        pool.set_attr("exclude_pad", spec_.exclude_pad);
    }
    if (spec_.is_fwd) pool.set_attr("rounding_type", spec_.rounding_type);
    if (spec_.op_kind == op::kind::AvgPoolBackprop)
        pool.set_attr("input_shape", spec_.src_dims);

    ops_.emplace_back(pool);
    curr_out_map_ids_.assign({TENSOR_ID});

    return fill_status::DONE;
}

fill_status_t pool_graph_prb_t::handle_bin_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.pool.bin_handler(*this, spec_.data_format, po_entry);
}

fill_status_t pool_graph_prb_t::handle_low_precision_(
        const ::pool::prb_t *prb_) {
    low_precision_attr lp_attr = low_precision_attr::lp_attr(
            spec_.src_dt, spec_.dst_dt, spec_.raw_data_format);

    fill_status_t ctor_status;
    ctor_status
            = po_handler.pool.low_precision_handler.handle_low_precision_src(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    ctor_status
            = po_handler.pool.low_precision_handler.handle_low_precision_dst(
                    *this, lp_attr);
    if (ctor_status != fill_status::DONE) return ctor_status;

    if (has_post_bin()) {
        ctor_status = po_handler.pool.low_precision_handler
                              .handle_low_precision_post_bin(*this, lp_attr,
                                      prb_->attr.post_ops.entry);
    }

    return ctor_status;
}

void graph_bwd_check_correctness(const ::pool::prb_t *prb, const args_t &args,
        const args_t &ref_args, res_t *res) {
    compare::compare_t cmp;
    cmp.set_data_kind(SRC);
    ::pool::setup_cmp(cmp, prb, SRC, ref_args);

    const int arg = DNNL_ARG_DIFF_SRC;
    const auto &mem_dt = args.find(arg);
    const auto &mem_fp = ref_args.find(arg);

    cmp.compare(mem_fp, mem_dt, prb->attr, res);
}

int doit(const ::pool::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case_graph(prb, res);
    if (res->state == SKIPPED) return OK;

    pool_graph_prb_t graph_prb(prb);
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

    bool is_fwd = prb->dir & FLAG_FWD;
    bool is_max_pool = prb->alg == ::pool::max;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    auto cp = compile_partition(::pool::init_pd, prb, res, par, ins, outs);

    auto src_fp = make_dnn_mem(is_fwd ? ins[0] : outs[0], dt::f32, tag::abx);
    auto dst_fp = make_dnn_mem(is_fwd ? outs[0] : is_max_pool ? ins[1] : ins[0],
            dt::f32, tag::abx);
    dnn_mem_t ws_fp
            = make_dnn_mem(is_fwd ? outs[0] : is_max_pool ? ins[1] : ins[0],
                    dt::s32, tag::abx);

    auto src_dt = make_dnn_mem(is_fwd ? ins[0] : outs[0], prb->tag);
    auto dst_dt = make_dnn_mem(
            is_fwd ? outs[0] : is_max_pool ? ins[1] : ins[0], prb->tag);

    SAFE(fill_src(prb, src_dt, src_fp, res), WARN);

    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    std::vector<int> binary_po_args;
    if (graph_prb.has_post_bin()) {
        binary_po_fp.emplace_back(make_dnn_mem(ins.back(), dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins.back(), prb->tag));
        const int po_idx = DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1;
        ::binary::fill_mem(po_idx, binary_po_dt.back(), binary_po_fp.back());
        binary_po_args.push_back(po_idx);
    }
    dnnl::graph::engine &eng = get_test_engine();

    dnn_mem_t d_dst_dt, d_src_dt, ws_dt;
    args_t args, ref_args;

    if (is_fwd) {
        tensors_in.emplace_back(ins[0], eng, static_cast<void *>(src_dt));
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(dst_dt));
        if (graph_prb.has_post_bin()) {
            tensors_in.emplace_back(dnnl::graph::tensor(
                    ins.back(), eng, static_cast<void *>(binary_po_dt.back())));
        }
        if (is_bench_mode(CORR)) {
            args.set(DNNL_ARG_DST, dst_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
            ref_args.set(binary_po_args, binary_po_fp);

            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);
            check_correctness(
                    prb, {DST}, args, ref_args, ::pool::setup_cmp, res);

            return OK;
        }
    } else {
        auto d_dst_fp = make_dnn_mem(
                is_max_pool ? ins[1] : ins[0], dt::f32, tag::abx);
        auto d_src_fp = make_dnn_mem(outs[0], dt::f32, tag::abx);
        d_dst_dt = make_dnn_mem(is_max_pool ? ins[1] : ins[0], prb->tag);
        d_src_dt = make_dnn_mem(outs[0], prb->tag);
        SAFE(fill_dst(prb, d_dst_dt, d_dst_fp, res), WARN);
        tensors_out.emplace_back(outs[0], eng, static_cast<void *>(d_src_dt));

        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        ref_args.set(DNNL_ARG_SRC, src_fp);
        ref_args.set(DNNL_ARG_DST, dst_fp);
        ref_args.set(DNNL_ARG_WORKSPACE, ws_fp);
        ref_args.set(binary_po_args, binary_po_fp);
        ref_args.set(DNNL_ARG_DIFF_DST, d_dst_fp);
        ref_args.set(DNNL_ARG_DIFF_SRC, d_src_fp);

        TIME_REF(::pool::compute_ref(prb, ref_args));

        if (is_max_pool) {
            tensors_in.emplace_back(ins[0], eng, static_cast<void *>(src_dt));
            tensors_in.emplace_back(ins[1], eng, static_cast<void *>(d_dst_dt));
            if (is_bench_mode(PERF)) {
                ws_dt = make_dnn_mem(ins[2], prb->tag);
                SAFE(ws_dt.reorder(ws_fp), WARN);
                tensors_in.emplace_back(
                        ins[2], eng, static_cast<void *>(ws_dt));
            }
        } else {
            tensors_in.emplace_back(ins[0], eng, static_cast<void *>(d_dst_dt));
        }

        if (is_bench_mode(CORR)) {
            SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);
            graph_bwd_check_correctness(prb, args, ref_args, res);

            return OK;
        }
    }

    SAFE(measure_perf(
                 res->timer_map.perf_timer(), cp, tensors_in, tensors_out, res),
            WARN);

    return OK;
}

} // namespace pool
} // namespace benchdnnext

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

#include "dnnl_graph_common.hpp"

#include "binary/binary.hpp"
#include "deconv/graph_deconv.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace benchdnnext {
namespace deconv {

static int check_known_skipped_case_graph(
        const ::deconv::prb_t *prb, res_t *res) noexcept {

    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> prim;
    SAFE(init_prim(prim, ::deconv::init_pd, prb, res), WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    const bool with_groups = prb->g > 1;
    if (with_groups) {
        const std::string wei_dnnl_fmt_tag_str
                = get_ou_format(normalize_tag(prb->wtag, prb->ndims));
        const dnnl_format_tag_t wei_dnnl_fmt_tag
                = dnnl_fmt_str2tag(wei_dnnl_fmt_tag_str);
        const std::vector<dnnl_format_tag_t> acceptable_wei_fmt_tags {
                dnnl_acbd, dnnl_acbde, dnnl_acbdef};
        const bool valid_wei_fmt_tag = std::any_of(
                acceptable_wei_fmt_tags.begin(), acceptable_wei_fmt_tags.end(),
                [wei_dnnl_fmt_tag](const dnnl_format_tag_t tag) {
                    return tag == wei_dnnl_fmt_tag;
                });
        // required in order to make strides aligned with oneDNN graph expectations
        if (!valid_wei_fmt_tag)
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
    }
    if (res->state == SKIPPED) return OK;

    // TODO(xiang): remove after onednn fix this.
    // GPU only support deconv with per_tensor output scale=1
    bool oscale_support = prb->attr.oscale.is_def()
            || (prb->attr.oscale.policy == policy_t::PER_OC
                    && prb->attr.oscale.scale == 1);
    // GPU doesn't support deconv with zero points
    bool zp_support = prb->attr.zero_points.is_def();
    if (is_gpu() && (!oscale_support || !zp_support)) {
        res->state = SKIPPED;
        res->reason = CASE_NOT_SUPPORTED;
    }
    if (res->state == SKIPPED) return OK;

    check_graph_scales_and_zps_support(prb->attr, res);
    return OK;
}

static dims_t get_graph_compatible_wei_dims(const dims_t &wei_dims) {
    dims_t new_dims = wei_dims;
    const auto groups = new_dims[0];
    new_dims.erase(new_dims.begin());
    new_dims[1] *= groups;
    return new_dims;
}

static dims_t get_acbdx_strides(const dims_t &wei_dims) {
    // permute dims OIX => IOX
    const dims_t wei_dims_permuted = [&wei_dims]() {
        auto d = wei_dims;
        std::swap(d[0], d[1]);
        return d;
    }();
    // calculate the original strides
    dims_t strides(wei_dims_permuted.size());
    strides[strides.size() - 1] = 1;
    for (int i = static_cast<int>(strides.size()) - 2; i >= 0; --i) {
        strides[i] = wei_dims_permuted[i + 1] * strides[i + 1];
    }
    // permute strides IOX => OIX
    const dims_t strides_permuted = [&strides]() {
        auto s = strides;
        std::swap(s[0], s[1]);
        return s;
    }();

    return strides_permuted;
}

static quant_data_t get_qdata_for(int arg, const ::deconv::prb_t *prb) {
    const auto q_dt = convert_dt(prb->cfg[arg].dt);
    if (arg == SRC) {
        const int64_t zp_val = prb->attr.zero_points.is_def(DNNL_ARG_SRC)
                ? 0L
                : prb->src_zp[0];
        return quant_data_t(q_dt, {1.0f}, {zp_val}, prb->stag);
    } else if (arg == WEI) {
        const auto scales = get_scales(prb->attr.oscale, prb->scales, prb->oc);
        const std::vector<int64_t> zps(scales.size(), 0L);
        const std::string q_type = prb->attr.oscale.policy == policy_t::COMMON
                ? "per_tensor"
                : "per_channel";
        if (prb->has_groups && prb->g > 1)
            return quant_data_t(q_dt, scales, zps, q_type, 0,
                    get_acbdx_strides(
                            get_graph_compatible_wei_dims(prb->wei_dims())));
        return quant_data_t(q_dt, scales, zps, q_type, 0, prb->wtag);
    } else if (arg == DST) {
        const float scale_val = 1.f
                * (1.f / get_post_eltwise_scale(prb->attr.post_ops.entry));
        const int64_t zp_val = prb->attr.zero_points.is_def(DNNL_ARG_DST)
                ? 0L
                : prb->dst_zp[0];
        return quant_data_t(q_dt, {scale_val}, {zp_val}, prb->dtag);
    }

    BENCHDNN_PRINT(
            0, "warning: returning default quant_data_t for arg: %d\n", arg);
    return quant_data_t();
}

static quant_data_t get_qdata_for(
        const attr_t::post_ops_t::entry_t &entry, const ::deconv::prb_t *prb) {
    if (entry.is_binary_kind())
        return bin_po_entry2quant_data(
                entry, prb->dtag, convert_dt(prb->cfg[DST].dt));
    else if (entry.is_sum_kind())
        return sum_po_entry2quant_data(
                entry, prb->dtag, convert_dt(prb->cfg[DST].dt));

    printf("warning: returning default quant_data_t for unsupported post op\n");
    return quant_data_t();
}

static std::vector<dnnl::graph::logical_tensor::data_type> collect_data_types(
        const ::deconv::prb_t *prb) {
    return {convert_dt(prb->cfg[SRC].dt), convert_dt(prb->cfg[WEI].dt),
            convert_dt(prb->cfg[DST].dt)};
}

fill_status_t append_graph_with_block(const ::deconv::prb_t *prb) {
    graph_t &graph = graph_t::get();

    const auto orig_dts = collect_data_types(prb);
    const auto with_dq = is_low_precision(orig_dts);
    const auto connect_to_previous_block = !with_dq && graph.has_blocks();

    // handle main op
    const auto op_id = graph.generate_id_for(entry_kind::DECONV);
    const auto src_lt_kind
            = prb->dir == BWD_D ? lt_kind::DIFF_SRC : lt_kind::SRC;
    const auto wei_lt_kind
            = prb->dir == BWD_W ? lt_kind::DIFF_WEI : lt_kind::WEI;
    const auto dst_lt_kind
            = prb->dir & FLAG_FWD ? lt_kind::DST : lt_kind::DIFF_DST;
    const auto src_id = connect_to_previous_block
            ? graph.get_last_block_out_id()
            : graph.generate_id_for(op_id, src_lt_kind);
    const auto wei_id = graph.generate_id_for(op_id, wei_lt_kind);
    const auto bia_id = graph.generate_id_for(op_id, lt_kind::BIA);
    const auto dst_id = graph.generate_id_for(op_id, dst_lt_kind);

    dims_t wei_dims = prb->has_groups
            ? get_graph_compatible_wei_dims(prb->wei_dims())
            : prb->wei_dims();
    dims_t wei_permuted_strides {};
    const bool with_permuted_wei_str = prb->has_groups && prb->g > 1;
    if (with_permuted_wei_str) {
        wei_permuted_strides = get_acbdx_strides(wei_dims);
    }

    dims_t dilations = prb->dilations();
    // oneDNN graph dilation = 1 is equivalent of oneDNN
    // dilation = 0
    std::transform(dilations.begin(), dilations.end(), dilations.begin(),
            [](const dim_t d) { return d + 1; });

    const auto src_dt = dequantize_dtype(orig_dts[0]);
    const auto wei_dt = dequantize_dtype(orig_dts[1]);
    const auto dst_dt = dequantize_dtype(orig_dts[2]);
    const auto bia_dt = convert_dt(prb->cfg[BIA].dt);

    std::vector<dnnl::graph::logical_tensor::data_type> dt_constraints {
            dnnl::graph::logical_tensor::data_type::bf16,
            dnnl::graph::logical_tensor::data_type::f16,
            dnnl::graph::logical_tensor::data_type::f32};
    for (auto actual_dt : {src_dt, wei_dt, dst_dt}) {
        if (std::find(dt_constraints.begin(), dt_constraints.end(), actual_dt)
                == dt_constraints.end()) {
            return fill_status::UNSUPPORTED_CONFIG;
        }
    }

    graph.create_lt(src_id, src_dt, prb->src_dims(), prb->stag);
    graph.create_lt(dst_id, dst_dt, prb->dst_dims(), prb->dtag);
    if (with_permuted_wei_str)
        graph.create_lt(wei_id, wei_dt, wei_dims, wei_permuted_strides);
    else
        graph.create_lt(wei_id, wei_dt, wei_dims, prb->wtag);
    if (prb->dir & FLAG_BIA)
        graph.create_lt(bia_id, bia_dt, prb->bia_dims(), lt::strided,
                dnnl::graph::logical_tensor::property_type::constant);

    std::vector<size_t> src_ids {};
    std::vector<size_t> dst_ids {};
    dnnl::graph::op::kind deconv_kind;

    if (prb->dir & FLAG_FWD) {
        src_ids = {src_id, wei_id};
        dst_ids = {dst_id};
        deconv_kind = dnnl::graph::op::kind::ConvTranspose;
    } else if (prb->dir == BWD_D) {
        src_ids = {dst_id, wei_id};
        dst_ids = {src_id};
        deconv_kind = dnnl::graph::op::kind::ConvTransposeBackpropData;
    } else if (prb->dir == BWD_W) {
        src_ids = {src_id, dst_id};
        dst_ids = {wei_id};
        deconv_kind = dnnl::graph::op::kind::ConvTransposeBackpropFilters;
    } else {
        return fill_status::UNSUPPORTED_CONFIG;
    }
    if (prb->dir & FLAG_BIA) src_ids.push_back(bia_id);

    dnnl::graph::op deconv_op(op_id, deconv_kind, graph.stringify_id(op_id));
    deconv_op.set_attr("strides", prb->strides())
            .set_attr("pads_begin", prb->padding())
            .set_attr("pads_end", prb->padding_r())
            .set_attr("dilations", dilations)
            .set_attr("auto_pad", std::string("None"))
            .set_attr("groups", prb->g)
            .set_attr("data_format", std::string("NCX"))
            .set_attr("filter_format", std::string("OIX"));
    if (prb->dir == BWD_W) deconv_op.set_attr("filter_shape", wei_dims);

    graph.append(op_id, deconv_op, src_ids, dst_ids);

    fill_status_t status;
    // if required - apply dequantize to block inputs
    if (with_dq) {
        status = insert_dequant_before(src_id, get_qdata_for(SRC, prb));
        BENCHDNNEXT_VERIFY(status);
        status = insert_dequant_before(wei_id, get_qdata_for(WEI, prb), true);
        BENCHDNNEXT_VERIFY(status);
    }

    // handle post ops
    for (const auto &entry : prb->attr.post_ops.entry) {
        const auto with_src1_dq
                = entry.is_sum_kind() || is_dequantize_required_for(entry);
        size_t po_src1_id;
        if (entry.is_binary_kind()) {
            std::tie(status, po_src1_id) = append_graph_with_binary(entry);
            BENCHDNNEXT_VERIFY(status);
            if (with_src1_dq) {
                status = insert_dequant_before(
                        po_src1_id, get_qdata_for(entry, prb));
                BENCHDNNEXT_VERIFY(status);
            }
        } else if (entry.is_eltwise_kind()) {
            status = append_graph_with_eltwise(entry);
            BENCHDNNEXT_VERIFY(status);
        } else if (entry.is_sum_kind()) {
            std::tie(status, po_src1_id) = append_graph_with_sum(entry);
            BENCHDNNEXT_VERIFY(status);
            if (with_dq && with_src1_dq) {
                status = insert_dequant_before(
                        po_src1_id, get_qdata_for(entry, prb));
                BENCHDNNEXT_VERIFY(status);
            }
        }
    }

    // if required - add quantize op
    if (is_low_precision({orig_dts[2]})) {
        status = insert_quant_after(
                graph.get_cur_block_out_id(), get_qdata_for(DST, prb));
        BENCHDNNEXT_VERIFY(status);
    }

    graph.close_block();

    return fill_status::DONE;
}

int doit(const ::deconv::prb_t *prb, res_t *res) {
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

    auto mode = convert_fpmath_mode(prb->attr.fpmath_mode);
    // Filter partitions
    const auto partitions = graph.get_partitions(mode);
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

    const auto &e = benchdnnext::get_test_engine();
    auto cp = par.compile(ins, outs, e);

    auto src_lt = (prb->dir == BWD_D) ? outs[0] : ins[0];
    auto wei_lt = (prb->dir == BWD_W) ? outs[0] : ins[1];
    auto dst_lt = [&ins, &outs](dir_t dir) {
        if (dir & FLAG_FWD)
            return outs[0];
        else if (dir == BWD_D)
            return ins[0];
        else
            return ins[1]; // BWD_W
    }(prb->dir);

    auto src_fp = make_dnn_mem(src_lt, prb->src_dims(), dt::f32, tag::abx);
    auto src_dt = make_dnn_mem(src_lt, prb->src_dims(), prb->stag);

    auto wei_fp = make_dnn_mem(wei_lt, prb->wei_dims(), dt::f32, tag::abx);
    auto wei_dt = make_dnn_mem(wei_lt, prb->wei_dims(), prb->wtag);

    auto dst_fp = make_dnn_mem(dst_lt, prb->dst_dims(), dt::f32, tag::abx);
    auto dst_dt = make_dnn_mem(dst_lt, prb->dst_dims(), prb->dtag);

    dnn_mem_t bia_fp;
    dnn_mem_t bia_dt;
    size_t idx_ins = 2;
    if (prb->dir & FLAG_BIA) {
        bia_fp = make_dnn_mem(ins[idx_ins], dt::f32, tag::x);
        bia_dt = make_dnn_mem(ins[idx_ins++], tag::x);
    }

    auto wei_tr_dims = prb->wei_dims();
    std::swap(wei_tr_dims[1], wei_tr_dims[2]);
    auto wei_tr_fp = make_dnn_mem(wei_lt, wei_tr_dims, dt::f32, tag::abx);

    if (prb->dir & FLAG_FWD || prb->dir == BWD_W)
        SAFE(fill_src(prb, src_dt, src_fp, res), WARN);
    if (prb->dir & FLAG_BWD) SAFE(fill_dst(prb, dst_dt, dst_fp, res), WARN);
    if (prb->dir & FLAG_BIA) SAFE(fill_bia(prb, bia_dt, bia_fp, res), WARN);
    if (prb->dir & FLAG_FWD || prb->dir == BWD_D) {
        SAFE(fill_wei(prb, wei_dt, wei_fp, res), WARN);
        SAFE(::deconv::transpose_data_wei(prb, wei_fp, wei_tr_fp), WARN);
    }

    std::vector<dnn_mem_t> binary_po_fp, binary_po_dt;
    const auto post_bin_indices
            = get_post_bin_indices(prb->attr.post_ops.entry);
    for (size_t i = 0; i < post_bin_indices.size(); ++i) {
        binary_po_fp.emplace_back(
                make_dnn_mem(ins[idx_ins], dt::f32, tag::abx));
        binary_po_dt.emplace_back(make_dnn_mem(ins[idx_ins++], tag::abx));
        binary::fill_mem(DNNL_ARG_ATTR_MULTIPLE_POST_OP(
                                 static_cast<int>(post_bin_indices[i])),
                binary_po_dt[i], binary_po_fp[i]);
    }

    const dnnl::graph::engine &eng = get_test_engine();

    dnnl::graph::tensor src_tensor(src_lt, eng, static_cast<void *>(src_dt));
    dnnl::graph::tensor wei_tensor(wei_lt, eng, static_cast<void *>(wei_dt));
    dnnl::graph::tensor dst_tensor(dst_lt, eng, static_cast<void *>(dst_dt));
    dnnl::graph::tensor bia_tensor;

    std::vector<dnnl::graph::tensor> tensors_in {};
    std::vector<dnnl::graph::tensor> tensors_out {};
    if (prb->dir & FLAG_FWD) {
        tensors_in = {src_tensor, wei_tensor};
        tensors_out = {dst_tensor};
    } else if (prb->dir == BWD_D) {
        tensors_in = {dst_tensor, wei_tensor};
        tensors_out = {src_tensor};
    } else if (prb->dir == BWD_W) {
        tensors_in = {src_tensor, dst_tensor};
        tensors_out = {wei_tensor};
    }

    idx_ins = 2;
    if (prb->dir & FLAG_BIA) {
        bia_tensor = dnnl::graph::tensor(
                ins[idx_ins++], eng, static_cast<void *>(bia_dt));
        tensors_in.emplace_back(bia_tensor);
    }

    size_t bin_dt_idx = 0;
    for (const auto &po_entry : prb->attr.post_ops.entry) {
        if (po_entry.is_sum_kind()) {
            dnnl::graph::tensor sum_src1_tensor(
                    ins[idx_ins++], eng, static_cast<void *>(dst_dt));
            tensors_in.emplace_back(sum_src1_tensor);
        } else if (po_entry.is_binary_kind()) {
            dnnl::graph::tensor bin_src1_tensor(ins[idx_ins++], eng,
                    static_cast<void *>(binary_po_dt[bin_dt_idx]));
            tensors_in.emplace_back(bin_src1_tensor);
            ++bin_dt_idx;
        }
    }

    SAFE(execute_and_wait(cp, tensors_in, tensors_out, res), WARN);

    if (is_bench_mode(CORR)) {
        ::deconv::prb_t prb_tr((::deconv::desc_t)*prb, prb->dir, prb->cfg,
                prb->stag, prb->wtag, prb->dtag, prb->alg, prb->attr, prb->mb);
        std::swap(prb_tr.ic, prb_tr.oc);
        std::swap(prb_tr.ih, prb_tr.oh);
        std::swap(prb_tr.id, prb_tr.od);
        std::swap(prb_tr.iw, prb_tr.ow);

        std::vector<int> binary_po_args;
        for (const size_t idx_bin : post_bin_indices) {
            binary_po_args.emplace_back(
                    (DNNL_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>(idx_bin))
                            | DNNL_ARG_SRC_1));
        }

        // re-scale bias
        dnn_mem_t bia_fp_scaled;
        if (prb->dir == FWD_B) {
            bia_fp_scaled = make_dnn_mem(ins[2], dt::f32, tag::x);
            scale_bia(bia_fp_scaled, bia_fp,
                    get_scales(prb->attr.oscale, prb->scales, prb->oc));
        }

        args_t args, ref_args;
        if (prb->dir & FLAG_FWD) {
            args.set(DNNL_ARG_DST, dst_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_BIAS, bia_fp_scaled);
            ref_args.set(DNNL_ARG_DST, dst_fp);
            ref_args.set(DNNL_ARG_WEIGHTS_1, wei_tr_fp); // Hack. See ref.
            ref_args.set(binary_po_args, binary_po_fp);

            check_correctness(
                    prb, {DST}, args, ref_args, ::deconv::setup_cmp, res);
        } else if (prb->dir == BWD_D) {
            args.set(DNNL_ARG_DIFF_SRC, src_dt);
            ref_args.set(DNNL_ARG_DIFF_SRC, src_fp);
            ref_args.set(DNNL_ARG_WEIGHTS, wei_fp);
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ref_args.set(DNNL_ARG_WEIGHTS_1, wei_tr_fp); // Hack. See ref.

            check_correctness(
                    prb, {SRC}, args, ref_args, ::deconv::setup_cmp, res);
        } else if (prb->dir == BWD_W) {
            args.set(DNNL_ARG_DIFF_WEIGHTS, wei_dt);
            ref_args.set(DNNL_ARG_SRC, src_fp);
            ref_args.set(DNNL_ARG_DIFF_WEIGHTS_1, wei_tr_fp); // Hack. See ref.
            ref_args.set(DNNL_ARG_DIFF_DST, dst_fp);
            ref_args.set(DNNL_ARG_DIFF_WEIGHTS, wei_fp);

            check_correctness(
                    prb, {WEI}, args, ref_args, ::deconv::setup_cmp, res);
        }
    }

    measure_perf(res->timer_map.perf_timer(), cp, tensors_in, tensors_out);

    cleanup();

    return OK;
}

} // namespace deconv
} // namespace benchdnnext

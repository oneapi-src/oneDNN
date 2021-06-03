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

#include <oneapi/dnnl/dnnl_debug.h>

#include "dnnl_graph_common.hpp"

namespace benchdnnext {

inline bool should_stop(const benchdnn_timer_t &t) {
    const bool stop = false
            || (fix_times_per_prb && t.times() >= fix_times_per_prb)
            || (!fix_times_per_prb && t.total_ms() >= max_ms_per_prb
                    && t.times() >= min_times_per_prb);
    return stop;
}

dnnl::graph::logical_tensor::data_type convert_dt(const dnnl_data_type_t dt) {
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    switch (dt) {
        case dnnl_f16: return graph_dt::f16;
        case dnnl_bf16: return graph_dt::bf16;
        case dnnl_f32: return graph_dt::f32;
        case dnnl_s32: return graph_dt::s32;
        case dnnl_s8: return graph_dt::s8;
        case dnnl_u8: return graph_dt::u8;
        case dnnl_data_type_undef:
        default: return graph_dt::undef;
    }
}

dnnl_data_type_t convert_dt(const dnnl::graph::logical_tensor::data_type dt) {
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    switch (dt) {
        case graph_dt::f16: return dnnl_f16;
        case graph_dt::bf16: return dnnl_bf16;
        case graph_dt::f32: return dnnl_f32;
        case graph_dt::s32: return dnnl_s32;
        case graph_dt::s8: return dnnl_s8;
        case graph_dt::u8: return dnnl_u8;
        case graph_dt::undef:
        default: return dnnl_data_type_undef;
    }
}

dnnl::graph::op::kind convert_alg_kind(const dnnl_alg_kind_t kind) {
    using graph_op = dnnl::graph::op::kind;
    // all options could be easily added later
    switch (kind) {
        case dnnl_eltwise_abs: return graph_op::Abs;
        case dnnl_eltwise_clip: return graph_op::HardTanh;
        case dnnl_eltwise_elu: return graph_op::Elu;
        case dnnl_eltwise_exp: return graph_op::Exp;
        case dnnl_eltwise_gelu_erf: return graph_op::GELU;
        case dnnl_eltwise_log: return graph_op::Log;
        case dnnl_eltwise_logistic: return graph_op::Sigmoid;
        case dnnl_eltwise_pow: return graph_op::Pow;
        case dnnl_eltwise_relu: return graph_op::ReLU;
        case dnnl_eltwise_round: return graph_op::Round;
        case dnnl_eltwise_sqrt: return graph_op::Sqrt;
        case dnnl_eltwise_square: return graph_op::Square;
        case dnnl_eltwise_tanh: return graph_op::Tanh;
        case dnnl_binary_add: return graph_op::Add;
        case dnnl_binary_div: return graph_op::Divide;
        case dnnl_binary_max: return graph_op::Maximum;
        case dnnl_binary_min: return graph_op::Minimum;
        case dnnl_binary_mul: return graph_op::Multiply;
        // TODO (damianszw): find nicer way to tell about unsupported type
        case dnnl_eltwise_bounded_relu:
        case dnnl_eltwise_clip_v2:
        case dnnl_eltwise_clip_v2_use_dst_for_bwd:
        case dnnl_eltwise_elu_use_dst_for_bwd:
        case dnnl_eltwise_exp_use_dst_for_bwd:
        case dnnl_eltwise_gelu_tanh:
        case dnnl_eltwise_hardswish:
        case dnnl_eltwise_linear:
        case dnnl_eltwise_logistic_use_dst_for_bwd:
        case dnnl_eltwise_logsigmoid:
        case dnnl_eltwise_mish:
        case dnnl_eltwise_relu_use_dst_for_bwd:
        case dnnl_eltwise_sqrt_use_dst_for_bwd:
        case dnnl_eltwise_soft_relu:
        case dnnl_eltwise_swish:
        case dnnl_eltwise_tanh_use_dst_for_bwd:
        default: return graph_op::LastSymbol;
    }
}

std::string convert_tag(const std::string &tag, bool activation_tag) {
    if (tag == "abx") return (activation_tag) ? "NCX" : "OIX";
    if (tag == "axb") return "NXC";
    if (tag == "xba") return "XIO";
    // default cases
    return (activation_tag) ? "NXC" : "XIO";
}

dims_t convert_bin_policy(const dims_t &lhs_dims, const attr_t::policy_t policy,
        const std::string &data_format) {
    using bin_pol = attr_t::policy_t;

    auto rhs_dims = dims_t(lhs_dims.size(), 1);

    switch (policy) {
        case bin_pol::PER_TENSOR: rhs_dims = lhs_dims; break;
        case bin_pol::PER_OC:
            if (data_format == "NCX")
                rhs_dims[1] = lhs_dims[1];
            else
                rhs_dims.back() = lhs_dims.back(); // "NXC" case
            break;
        case bin_pol::PER_DIM_0: rhs_dims[0] = lhs_dims[0]; break;
        case bin_pol::PER_DIM_1: rhs_dims[1] = lhs_dims[1]; break;
        case bin_pol::PER_DIM_01:
            rhs_dims[0] = lhs_dims[0];
            rhs_dims[1] = lhs_dims[1];
            break;
        case bin_pol::COMMON:
        default: break;
    }

    return rhs_dims;
}

std::map<std::string, float> convert_eltw_entry(
        const dnnl::graph::op::kind kind,
        const attr_t::post_ops_t::entry_t &entry) {
    using graph_op = dnnl::graph::op::kind;

    std::map<std::string, float> attrs;
    // all options could be easily added later
    switch (kind) {
        case graph_op::Elu: attrs["alpha"] = entry.eltwise.alpha; return attrs;
        case graph_op::HardTanh:
            attrs["min"] = entry.eltwise.alpha;
            attrs["max"] = entry.eltwise.beta;
            return attrs;
        default: return attrs;
    }
}

int scale_bia(dnn_mem_t &dst, dnn_mem_t &src, const std::vector<float> scales) {
    if (scales.empty()) {
        dst = std::move(src);
        return OK;
    }
    constexpr float eps = 1.e-9;
    std::vector<float> bia_scales(scales.size(), 0.f);
    std::transform(scales.begin(), scales.end(), bia_scales.begin(),
            [eps](const float scale) { return 1.f / (scale + eps); });
    const int bia_mask = bia_scales.size() == 1 ? 0 : 1;
    dnnl_primitive_attr_t bia_attr = nullptr;
    dnnl_primitive_attr_create(&bia_attr);
    dnnl_primitive_attr_set_output_scales(
            bia_attr, bia_scales.size(), bia_mask, bia_scales.data());
    SAFE(dst.reorder(src, bia_attr), CRIT);

    return OK;
}

dnnl_format_tag_t dnnl_fmt_str2tag(const std::string &fmt_str) {
    dnnl_format_tag_t tag = dnnl_format_tag_undef;
    for (int i = 0; i < dnnl_format_tag_last; ++i) {
        tag = static_cast<dnnl_format_tag_t>(i);
        if (dnnl_fmt_tag2str(tag) == fmt_str) break;
    }
    if (tag == dnnl_format_tag_undef)
        []() {
            SAFE(FAIL, CRIT);
            return 0;
        }();
    return tag;
};

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt,
        const dnnl::graph::logical_tensor::data_type &graph_dt,
        const char *atag) {
    using graph_layout = dnnl::graph::logical_tensor::layout_type;

    const auto &dnnl_test_engine = ::get_test_engine();
    const auto dims = lt.get_dims();
    const int ndims = static_cast<int>(dims.size());
    std::string valid_tag = atag ? normalize_tag(atag, ndims) : "abx";

    // NOTE: oneDNN Graph cannot get the concrete format from any-format logical
    //   tensor. Given that some tags in benchdnn is any by default, we should
    //   consider any to be default plain format for oneDNN Graph.
    if (valid_tag == tag::any) valid_tag = normalize_tag("abx", ndims);

    dnnl_dims_t dnnl_dims = {0};
    std::copy(dims.begin(), dims.end(), dnnl_dims);

    auto ltype = lt.get_layout_type();
    if (ltype == graph_layout::any)
        return dnn_mem_t(ndims, dnnl_dims, convert_dt(graph_dt), valid_tag,
                dnnl_test_engine);
    else if (ltype == graph_layout::strided) {
        const auto strides = lt.get_strides();
        dnnl_dims_t dnnl_strides;
        std::copy(strides.begin(), strides.end(), dnnl_strides);
        return dnn_mem_t(ndims, dnnl_dims, convert_dt(graph_dt), dnnl_strides,
                dnnl_test_engine);
    } else if (ltype == graph_layout::opaque) {
        return dnn_mem_t(ndims, dnnl_dims, convert_dt(graph_dt), valid_tag,
                dnnl_test_engine);
    } else {
        []() {
            SAFE(FAIL, CRIT);
            return 0;
        }();
        return dnn_mem_t();
    }
}

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt,
        const dims_t &dims,
        const dnnl::graph::logical_tensor::data_type &graph_dt,
        const char *atag) {
    dnnl::graph::logical_tensor new_lt(
            lt.get_id(), lt.get_data_type(), dims, lt.get_layout_type());
    return make_dnn_mem(new_lt, graph_dt, atag);
}

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt,
        const dims_t &dims, const std::string &atag) {
    dnnl::graph::logical_tensor new_lt(
            lt.get_id(), lt.get_data_type(), dims, lt.get_layout_type());
    return make_dnn_mem(new_lt, atag);
}

dnn_mem_t make_dnn_mem(
        const dnnl::graph::logical_tensor &lt, const std::string &tag) {
    return make_dnn_mem(lt, tag.empty() ? nullptr : tag.c_str());
}

dnn_mem_t make_dnn_mem(const dnnl::graph::logical_tensor &lt, const char *tag) {
    return make_dnn_mem(lt, lt.get_data_type(), tag);
}

void compiled_partition_executor(dnnl::graph::compiled_partition &cp,
        dnnl::graph::stream &stream,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    cp.execute(stream, inputs, outputs);
}

int execute_and_wait(perf_function_t &exec_func, dnnl::graph::engine &engine,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    dnnl::graph::stream stream(engine);
    BENCHDNNEXT_SAFE(exec_func(stream, inputs, outputs), CRIT);
    BENCHDNNEXT_SAFE(stream.wait(), CRIT);

    return OK;
}

int execute_and_wait(dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    perf_function_t perf_func
            = std::bind(&compiled_partition_executor, cp, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3);
    dnnl::graph::engine &engine = get_test_engine();

    return execute_and_wait(perf_func, engine, inputs, outputs);
};

inline int measure_perf_individual(benchdnn_timer_t &t,
        dnnl::graph::stream &stream, perf_function_t &perf_func,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    t.reset();
    while (true) {
        BENCHDNNEXT_SAFE(perf_func(stream, inputs, outputs), WARN);
        t.stamp();
        if (should_stop(t)) break;
    }
    return OK;
}

int measure_perf(benchdnn_timer_t &t, perf_function_t &perf_func,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    int ret = OK;
    if (is_bench_mode(PERF)) {
        dnnl::graph::stream stream(::benchdnnext::get_test_engine());
        ret = measure_perf_individual(t, stream, perf_func, inputs, outputs);
    }
    return ret;
}

int measure_perf(benchdnn_timer_t &t, dnnl::graph::compiled_partition &cp,
        const std::vector<dnnl::graph::tensor> &inputs,
        const std::vector<dnnl::graph::tensor> &outputs) {
    perf_function_t perf_func
            = std::bind(&compiled_partition_executor, cp, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3);

    return measure_perf(t, perf_func, inputs, outputs);
}

fill_status_t po_handlers_t::bias_po_handler_t::operator()(graph_prb_t &p,
        const std::string &dst_dataf,
        const dnnl::graph::logical_tensor::data_type bia_dt) {
    using op = dnnl::graph::op;

    const auto dst_lt = p.tensor_descs_[p.curr_out_map_ids_.back()];
    const auto dst_dims = dst_lt.get_dims();
    const auto dst_dt = dst_lt.get_data_type();
    const dim_t channels = (dst_dataf == "NCX") ? dst_dims[1] : dst_dims.back();
    const dims_t bia_dims = {channels};

    const std::string BIA_SRC {"bias_src"};
    const std::string BIA_DST {"bias_dst"};

    p.tensor_descs_.emplace(BIA_SRC, bia_dt, bia_dims, lt::strided);
    p.tensor_descs_.emplace(BIA_DST, dst_dt, dst_dims, lt::strided);

    const size_t new_op_id = p.ops_.size();
    op bias(new_op_id, op::kind::BiasAdd,
            {p.tensor_descs_[p.curr_out_map_ids_.back()],
                    p.tensor_descs_[BIA_SRC]},
            {p.tensor_descs_[BIA_DST]}, "bias");

    bias.set_attr("data_format", dst_dataf);

    p.ops_.emplace_back(bias);
    p.curr_out_map_ids_.assign({BIA_DST});

    return fill_status::DONE;
}

fill_status_t po_handlers_t::eltwise_po_handler_t::operator()(
        graph_prb_t &p, const attr_t::post_ops_t::entry_t &po_entry) {
    using op = dnnl::graph::op;

    const auto post_op_kind = convert_alg_kind(po_entry.eltwise.alg);
    if (post_op_kind == op::kind::LastSymbol)
        return fill_status::UNSUPPORTED_OP;

    const auto dst_lt = p.tensor_descs_[p.curr_out_map_ids_.back()];
    const auto dst_dims = dst_lt.get_dims();
    const auto dst_dt = dst_lt.get_data_type();

    const std::string ELT_DST {"elt_dst"};

    p.tensor_descs_.emplace(ELT_DST, dst_dt, dst_dims, lt::strided);

    const size_t new_op_id = p.ops_.size();
    op eltwise(new_op_id, post_op_kind,
            {p.tensor_descs_[p.curr_out_map_ids_.back()]},
            {p.tensor_descs_[ELT_DST]}, "eltwise");

    const auto attrs = convert_eltw_entry(post_op_kind, po_entry);
    for (const auto &kv : attrs) {
        eltwise.set_attr(kv.first, kv.second);
    }

    p.ops_.emplace_back(eltwise);
    p.curr_out_map_ids_.assign({ELT_DST});

    return fill_status::DONE;
}

fill_status_t po_handlers_t::binary_po_handler_t::operator()(graph_prb_t &p,
        const std::string &dst_dataf,
        const attr_t::post_ops_t::entry_t &po_entry) {
    using op = dnnl::graph::op;

    const auto post_op_kind = convert_alg_kind(po_entry.binary.alg);
    if (post_op_kind == op::kind::LastSymbol)
        return fill_status::UNSUPPORTED_OP;

    const auto dst_lt = p.tensor_descs_[p.curr_out_map_ids_.back()];
    const auto dst_dims = dst_lt.get_dims();
    const auto dst_dt = dst_lt.get_data_type();
    const auto bin_src_dims
            = convert_bin_policy(dst_dims, po_entry.binary.policy, dst_dataf);
    const auto bin_src_dt = convert_dt(po_entry.binary.src1_dt);

    const std::string BIN_SRC {"bin_src1"};
    const std::string BIN_DST {"bin_dst"};

    p.tensor_descs_.emplace(BIN_SRC, bin_src_dt, bin_src_dims, lt::strided);
    p.tensor_descs_.emplace(BIN_DST, dst_dt, dst_dims, lt::strided);

    const size_t new_op_id = p.ops_.size();
    op binary(new_op_id, post_op_kind,
            {p.tensor_descs_[p.curr_out_map_ids_.back()],
                    p.tensor_descs_[BIN_SRC]},
            {p.tensor_descs_[BIN_DST]}, "binary");

    p.ops_.emplace_back(binary);
    p.curr_out_map_ids_.assign({BIN_DST});

    return fill_status::DONE;
}

fill_status_t po_handlers_t::sum_po_handler_t::operator()(graph_prb_t &p) {
    using op = dnnl::graph::op;

    const auto dst_lt = p.tensor_descs_[p.curr_out_map_ids_.back()];
    const auto dst_dims = dst_lt.get_dims();
    const auto dst_dt = dst_lt.get_data_type();

    const std::string SUM_SRC {"sum_src1"};
    const std::string SUM_DST {"sum_dst"};

    p.tensor_descs_.emplace(SUM_SRC, dst_dt, dst_dims, lt::strided);
    p.tensor_descs_.emplace(SUM_DST, dst_dt, dst_dims, lt::strided);

    const size_t new_op_id = p.ops_.size();
    op sum(new_op_id, op::kind::Add,
            {p.tensor_descs_[p.curr_out_map_ids_.front()],
                    p.tensor_descs_[SUM_SRC]},
            {p.tensor_descs_[SUM_DST]}, "sum");

    p.ops_.emplace_back(sum);
    p.curr_out_map_ids_.assign({SUM_DST});

    return fill_status::DONE;
}

} // namespace benchdnnext

/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#ifndef GRAPH_EXAMPLE_COMPAT_COMPAT_SDPA_HPP
#define GRAPH_EXAMPLE_COMPAT_COMPAT_SDPA_HPP

#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "oneapi/dnnl/dnnl_graph.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "oneapi/dnnl/dnnl_graph_ocl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#include "compat_helpers.hpp"

// Wrappers on top of oneDNN Graph API to mimic the interfaces of cuDNN frontend
// API (v1.8). Currently, it's focused on the APIs used by the example:
// https://github.com/NVIDIA/cudnn-frontend/blob/936021bfed8c91dc416af1588b2c4eca631a9e45/samples/cpp/sdpa/fp16_fwd.cpp

namespace compat {

// cudnn_frontend::detail::Context
class Context {
    using lt = dnnl::graph::logical_tensor;

    lt::data_type compute_data_type = lt::data_type::undef;
    lt::data_type intermediate_data_type = lt::data_type::undef;
    lt::data_type io_data_type = lt::data_type::undef;
    std::string name = "";

public:
    Context &set_intermediate_data_type(lt::data_type dt) {
        intermediate_data_type = dt;
        return *this;
    }

    Context &set_io_data_type(lt::data_type dt) {
        io_data_type = dt;
        return *this;
    }

    Context &set_compute_data_type(lt::data_type dt) {
        compute_data_type = dt;
        return *this;
    }

    lt::data_type get_io_data_type() const { return io_data_type; }

    lt::data_type get_intermediate_data_type() const {
        return intermediate_data_type;
    }

    lt::data_type get_compute_data_type() const { return compute_data_type; }

    Context &set_name(const std::string &name_) {
        name = name_;
        return *this;
    }

    std::string get_name() const { return name; }
};

// cudnn_fronteend::graph::Tensor_attributes
class Tensor_attributes {
public:
    using lt = dnnl::graph::logical_tensor;
    using uid_t = size_t;

    Tensor_attributes() = default;
    Tensor_attributes &set_name(const std::string &name) {
        name_ = name;
        return *this;
    }

    Tensor_attributes &set_uid(uid_t uid) {
        uid_ = uid;
        return *this;
    }

    Tensor_attributes &set_dim(const lt::dims &dim) {
        dim_ = dim;
        return *this;
    }

    const lt::dims &get_dim() const { return dim_; }

    Tensor_attributes &set_stride(const lt::dims &stride) {
        stride_ = stride;
        return *this;
    }

    Tensor_attributes &set_data_type(lt::data_type dt) {
        dt_ = dt;
        return *this;
    }

    const lt::data_type &get_data_type() const { return dt_; }

    Tensor_attributes &set_is_virtual(bool is_virtual) {
        is_virtual_ = is_virtual;
        return *this;
    }

    bool get_is_virtual() const { return is_virtual_; }

    Tensor_attributes &set_output(bool output) {
        return set_is_virtual(!output);
    }

    lt to_logical_tensor() const { return lt(uid_, dt_, dim_, stride_); }

    Tensor_attributes &fill_from_context(const Context &ctx) {
        if (get_data_type() == lt::data_type::undef) {
            if (get_is_virtual()) {
                set_data_type(ctx.get_intermediate_data_type());
            } else {
                set_data_type(ctx.get_io_data_type());
            }
        }
        return *this;
    }

private:
    std::string name_ = "n/a";
    lt::data_type dt_ = lt::data_type::undef;
    lt::dims dim_ = {};
    lt::dims stride_ = {};
    uid_t uid_ = 0;
    bool is_virtual_ = false;
};

// cudnn_fronteend::graph::SDPA_attributes
class SDPA_attributes {
public:
    friend class Graph;

    enum class input_names {
        Q,
        K,
        V,
        Attn_scale,
        Bias,
        SEQ_LEN_Q,
        SEQ_LEN_KV,
        Seed,
        Offset,
        Dropout_mask,
        Dropout_scale,
        Page_table_K,
        Page_table_V
    };
    enum class output_names { O, Stats, RNG_DUMP };
    std::unordered_map<input_names, std::shared_ptr<Tensor_attributes>> inputs;
    std::unordered_map<output_names, std::shared_ptr<Tensor_attributes>>
            outputs;

    SDPA_attributes() = default;

    SDPA_attributes &set_is_inference(bool is_inference) {
        is_inference_ = is_inference;
        return *this;
    }

    SDPA_attributes &set_attn_scale(std::shared_ptr<Tensor_attributes> scale) {
        inputs[input_names::Attn_scale] = std::move(scale);
        return *this;
    }

    SDPA_attributes &set_attn_scale(float scale) {
        attn_scale_value_ = scale;
        return *this;
    }

    float get_attn_scale() const { return attn_scale_value_; }

    SDPA_attributes &set_bias(std::shared_ptr<Tensor_attributes> bias) {
        inputs[input_names::Bias] = std::move(bias);
        return *this;
    }

    SDPA_attributes &set_alibi_mask(bool alibi) {
        alibi_mask_ = alibi;
        return *this;
    }

    SDPA_attributes &set_padding_mask(bool padding) {
        padding_mask_ = padding;
        return *this;
    }

    SDPA_attributes &set_seq_len_q(
            std::shared_ptr<Tensor_attributes> seq_len_q) {
        inputs[input_names::SEQ_LEN_Q] = std::move(seq_len_q);
        return *this;
    }

    SDPA_attributes &set_seq_len_kv(
            std::shared_ptr<Tensor_attributes> seq_len_kv) {
        inputs[input_names::SEQ_LEN_KV] = std::move(seq_len_kv);
        return *this;
    }

    SDPA_attributes &set_causal_mask(bool causal) {
        causal_mask_ = causal;
        return *this;
    }

    bool get_causal_mask() const { return causal_mask_; }

    SDPA_attributes &set_causal_mask_bottom_right(bool causal) {
        causal_mask_bottom_right_ = causal;
        return *this;
    }

    SDPA_attributes &set_name(const std::string &name) {
        name_ = name;
        return *this;
    }

    void fill_from_context(const Context &ctx) {
        for (auto &v : inputs) {
            if (v.second) { v.second->fill_from_context(ctx); }
        }

        for (auto &v : outputs) {
            if (v.second) { v.second->fill_from_context(ctx); }
        }
    }

protected:
    std::string name_ = "n/a";
    bool is_inference_ = true;
    bool alibi_mask_ = false;
    bool padding_mask_ = false;
    bool causal_mask_ = false;
    bool causal_mask_bottom_right_ = false;
    float attn_scale_value_ = 1.0f;
};

// forward declare.
class SDPANode;

// cudnn_frontend::graph::Graph
class Graph {
public:
    using lt = dnnl::graph::logical_tensor;

    Graph() = default;

    Graph &set_io_data_type(lt::data_type dt) {
        ctx_.set_io_data_type(dt);
        return *this;
    }

    Graph &set_intermediate_data_type(lt::data_type dt) {
        ctx_.set_intermediate_data_type(dt);
        return *this;
    }

    Graph &set_compute_data_type(lt::data_type dt) {
        ctx_.set_compute_data_type(dt);
        return *this;
    }

    Graph &set_name(const std::string &name) {
        ctx_.set_name(name);
        return *this;
    }

    std::shared_ptr<Tensor_attributes> tensor(
            const Tensor_attributes &tensor_attr) {
        auto tensor_ptr = std::make_shared<Tensor_attributes>(tensor_attr);
        full_graph_inputs.emplace(tensor_ptr);
        return tensor_ptr;
    }

    std::shared_ptr<Tensor_attributes> output_tensor(const std::string &name) {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_name(name).set_is_virtual(true);
        full_graph_outputs.insert(tensor);
        return tensor;
    }

    std::array<std::shared_ptr<Tensor_attributes>, 2> sdpa(
            std::shared_ptr<Tensor_attributes> q,
            std::shared_ptr<Tensor_attributes> k,
            std::shared_ptr<Tensor_attributes> v, SDPA_attributes attr) {
        // Make required output tensors
        auto O = attr.outputs[SDPA_attributes::output_names::O]
                = output_tensor(attr.name_ + "::O");

        std::shared_ptr<Tensor_attributes> Stats = nullptr;
        if (attr.is_inference_ == false) {
            Stats = attr.outputs[SDPA_attributes::output_names::Stats]
                    = output_tensor(attr.name_ + "::Stats");
        }

        // Set inputs
        attr.inputs[SDPA_attributes::input_names::Q] = q;
        attr.inputs[SDPA_attributes::input_names::K] = k;
        attr.inputs[SDPA_attributes::input_names::V] = v;

        sdpa_ = std::make_shared<SDPANode>(std::move(attr), ctx_);

        return {O, Stats};
    }

    void build(Handle &handle);

    void execute(Handle &handle,
            std::unordered_map<Tensor_attributes::uid_t, void *>
                    &tensor_to_pointer_map) const;

private:
    std::unordered_set<std::shared_ptr<Tensor_attributes>> full_graph_inputs;
    std::unordered_set<std::shared_ptr<Tensor_attributes>> full_graph_outputs;
    std::unordered_set<Tensor_attributes::uid_t> used_uids;

    Context ctx_ {};

    // we will need to extend this to store more predefined nodes.
    std::shared_ptr<SDPANode> sdpa_;
};

// define internal input IDs.
#define ATTN_SCALE_ID 100000
#define INF_ID 100001

// cudnn_frontend::graph::SDPANode
class SDPANode {
public:
    using logical_tensor = dnnl::graph::logical_tensor;
    using op = dnnl::graph::op;
    using layout_type = logical_tensor::layout_type;
    using data_type = logical_tensor::data_type;
    using cp = dnnl::graph::compiled_partition;
    using tensor = dnnl::graph::tensor;

    SDPANode(const SDPA_attributes &attr, const Context &ctx)
        : attr_(attr), ctx_(ctx) {}

    void build(Handle &handle) {
        dnnl::graph::graph sdpa(handle.get_engine_kind());

        attr_.fill_from_context(ctx_);

        const auto &Q = attr_.inputs[SDPA_attributes::input_names::Q];
        const auto &K = attr_.inputs[SDPA_attributes::input_names::K];
        const auto &V = attr_.inputs[SDPA_attributes::input_names::V];

        const std::shared_ptr<Tensor_attributes> &SCAL
                = attr_.inputs.find(SDPA_attributes::input_names::Attn_scale)
                        != attr_.inputs.end()
                ? attr_.inputs[SDPA_attributes::input_names::Attn_scale]
                : nullptr;

        const auto dt = Q->get_data_type();

        const logical_tensor::dim mb = Q->get_dim()[0];
        const logical_tensor::dim hd = Q->get_dim()[1];
        const logical_tensor::dim qs = Q->get_dim()[2];
        const logical_tensor::dim ds = Q->get_dim()[3];
        const logical_tensor::dim ks = K->get_dim()[2];

        // Prepare input and output shapes to construct the sdpa graph.
        const logical_tensor::dims qv_sz = {mb, hd, qs, ds};
        // const logical_tensor::dims k_sz = {mb, hd, ks, ds};
        const logical_tensor::dims score_sz = {mb, hd, qs, ks};
        // const logical_tensor::dims scale_sz = {1};
        const logical_tensor::dims mask_sz = {mb, 1, qs, ks};

        // TODO(xxx): We will need a better ID allocation algorithm: 1) avoid
        // collision between internal tensor IDs and external tensor IDs. 2)
        // thread-safe. Currently, 65535 is used as the base of internal IDs.
        // 0-65535 is reserved for users.
        size_t id = 65535;

        auto query = Q->to_logical_tensor();
        auto key = K->to_logical_tensor();
        auto value = V->to_logical_tensor();

        std::vector<logical_tensor> inputs = {query, key, value};

        // score = query x key.T
        auto score = logical_tensor(id++, dt, score_sz, layout_type::strided);
        auto bmm1 = op(id++, op::kind::MatMul, "bmm1");
        bmm1.set_attr<bool>(op::attr::transpose_b, true);
        bmm1.add_inputs({query, key});
        bmm1.add_outputs({score});
        sdpa.add_op(bmm1);

        // scaled_score = score / scale
        auto scaled_score = score;
        if (SCAL) {
            std::cout << "input attn_scale tensor" << std::endl;
            auto scale = SCAL->to_logical_tensor();
            scaled_score
                    = logical_tensor(id++, dt, score_sz, layout_type::strided);
            auto scale_div = op(id++, op::kind::Divide, "scale_div");
            scale_div.add_inputs({score, scale});
            scale_div.add_outputs({scaled_score});
            sdpa.add_op(scale_div);
            inputs.push_back(scale);
        } else if (attr_.get_attn_scale() != 1.f) {
            std::cout << "float attn_scale value: " << attr_.get_attn_scale()
                      << std::endl;
            auto scale = logical_tensor(
                    ATTN_SCALE_ID, dt, 0, layout_type::strided);
            scaled_score
                    = logical_tensor(id++, dt, score_sz, layout_type::strided);
            auto scale_div = op(id++, op::kind::Divide, "scale_div");
            scale_div.add_inputs({score, scale});
            scale_div.add_outputs({scaled_score});
            sdpa.add_op(scale_div);
            inputs.push_back(scale);
        }

        // masked_score = scaled_score + mask
        auto masked_score = scaled_score;
        if (0) {
            auto mask = logical_tensor(id++, dt, mask_sz, layout_type::strided);
            masked_score
                    = logical_tensor(id++, dt, score_sz, layout_type::strided);
            auto mask_add = op(id++, op::kind::Add, "mask_add");
            mask_add.add_inputs({scaled_score, mask});
            mask_add.add_outputs({masked_score});
            sdpa.add_op(mask_add);
            inputs.push_back(mask);
        }

        // implicit causal mask
        if (attr_.get_causal_mask()) {
            std::cout << "causal_mask_: true" << std::endl;
            auto row_index = logical_tensor(
                    id++, data_type::s32, score_sz, layout_type::strided);
            auto genindex_row = op(id++, op::kind::GenIndex, "genindex_row");
            genindex_row.set_attr<int64_t>(op::attr::axis, 2);
            genindex_row.add_input(scaled_score);
            genindex_row.add_output(row_index);

            auto col_index = logical_tensor(
                    id++, data_type::s32, score_sz, layout_type::strided);
            auto genindex_col = op(id++, op::kind::GenIndex, "genindex_col");
            genindex_col.set_attr<int64_t>(op::attr::axis, 3);
            genindex_col.add_input(scaled_score);
            genindex_col.add_output(col_index);

            auto mask = logical_tensor(
                    id++, data_type::boolean, score_sz, layout_type::strided);
            auto ge = op(id++, op::kind::GreaterEqual, "mask_greater_equal");
            ge.add_inputs({row_index, col_index});
            ge.add_output(mask);

            const logical_tensor::dims inf_sz = {1};
            auto inf = logical_tensor(
                    INF_ID, dt, inf_sz, layout_type::strided); // scalar -inf.
            masked_score
                    = logical_tensor(id++, dt, score_sz, layout_type::strided);
            auto select = op(id++, op::kind::Select, "select");
            select.add_inputs({mask, scaled_score, inf});
            select.add_output(masked_score);

            sdpa.add_op(genindex_row);
            sdpa.add_op(genindex_col);
            sdpa.add_op(ge);
            sdpa.add_op(select);
            inputs.push_back(inf);
        }

        // attention_probs = softmax(masked_score)
        auto probs = logical_tensor(id++, dt, score_sz, layout_type::strided);
        auto softmax = op(id++, op::kind::SoftMax, "softmax");
        softmax.set_attr<int64_t>(op::attr::axis, -1);
        softmax.add_inputs({masked_score});
        softmax.add_outputs({probs});
        sdpa.add_op(softmax);

        // attention_output = attention_probs x value
        auto output = logical_tensor(id++, dt, qv_sz, layout_type::strided);
        auto bmm2 = op(id++, op::kind::MatMul, "bmm2");
        bmm2.add_inputs({probs, value});
        bmm2.add_outputs({output});
        sdpa.add_op(bmm2);
        sdpa.finalize();

        auto parts = sdpa.get_partitions();
        if (parts.size() != 1) throw std::runtime_error("partition failed ...");

        // Compile the partition with inputs, outputs, and an engine.
        exec_ = parts[0].compile(inputs, {output}, *(handle.get_engine()));
    }

    void execute(Handle &handle,
            std::unordered_map<Tensor_attributes::uid_t, void *> &pack) {
        const auto &Q = attr_.inputs[SDPA_attributes::input_names::Q];
        const auto &K = attr_.inputs[SDPA_attributes::input_names::K];
        const auto &V = attr_.inputs[SDPA_attributes::input_names::V];
        const auto &O = attr_.outputs[SDPA_attributes::output_names::O];

        // Consider moving these stuff to build time to save time.
        auto query = Q->to_logical_tensor();
        auto key = K->to_logical_tensor();
        auto value = V->to_logical_tensor();
        auto out = O->to_logical_tensor();

        auto eng = handle.get_engine();
        auto ts_q = tensor(query, *eng, pack[query.get_id()]);
        auto ts_k = tensor(key, *eng, pack[key.get_id()]);
        auto ts_v = tensor(value, *eng, pack[value.get_id()]);
        auto ts_o = tensor(out, *eng, pack[out.get_id()]);

        // handle float attn_scale and inf for causal mask
        const auto dt = query.get_data_type();
        Surface raw_scale(dt, 1, &handle, attr_.get_attn_scale());
        const auto scale
                = logical_tensor(ATTN_SCALE_ID, dt, 0, layout_type::strided);
        tensor ts_scale = tensor(scale, *eng, raw_scale.get_ptr());

        tensor ts_inf;
        Surface raw_inf(
                dt, 1, &handle, -1 * std::numeric_limits<float>::infinity());
        if (attr_.get_causal_mask()) {
            const auto inf
                    = logical_tensor(INF_ID, dt, 0, layout_type::strided);
            ts_inf = tensor(inf, *eng, raw_inf.get_ptr());
        }

        exec_.execute(*(handle.get_stream()),
                {ts_q, ts_k, ts_v, ts_scale, ts_inf}, {ts_o});
        handle.synchronize();
    }

private:
    SDPA_attributes attr_;
    Context ctx_;
    cp exec_;
};

void Graph::build(Handle &handle) {
    // TODO: validation before build.
    sdpa_->build(handle);
}

void Graph::execute(Handle &handle,
        std::unordered_map<Tensor_attributes::uid_t, void *> &pack) const {
    sdpa_->execute(handle, pack);
}

} // namespace compat

#endif
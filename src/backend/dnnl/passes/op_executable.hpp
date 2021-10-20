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
#ifndef BACKEND_DNNL_PASSES_OP_EXECUTABLE_HPP
#define BACKEND_DNNL_PASSES_OP_EXECUTABLE_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "dnnl.hpp"

#include <utils/utils.hpp>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/passes/lower_down.hpp"
#include "backend/dnnl/passes/utils.hpp"

#define DNNL_GRAPH_ARG_POST_SRC (-1)

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

inline dnnl::convolution_forward::primitive_desc create_conv_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return static_cast<dnnl::convolution_forward::primitive_desc &>(
                pd_cache.at(op.get()));
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>("strides");
    auto dilates = op->get_attr<dims>("dilations");
    auto pads_begin = op->get_attr<dims>("pads_begin");
    auto pads_end = op->get_attr<dims>("pads_end");
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr("primitive_attr_key")) {
        int64_t key = op->get_attr<int64_t>("primitive_attr_key");
        prm_attr = prm_attr_mgr.get_attr(key);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto weight = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    dnnl::convolution_forward::primitive_desc pd;
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        pd = dnnl::convolution_forward::primitive_desc(
                {prop_kind::forward_inference, algorithm::convolution_direct,
                        src, weight, bias, dst, strides, dilates, pads_begin,
                        pads_end},
                prm_attr, p_engine);
    } else {
        pd = dnnl::convolution_forward::primitive_desc(
                {prop_kind::forward_inference, algorithm::convolution_direct,
                        src, weight, dst, strides, dilates, pads_begin,
                        pads_end},
                prm_attr, p_engine);
    }

    pd_cache.insert({op.get(), pd});

    return pd;
}

inline dnnl::deconvolution_forward::primitive_desc create_deconv_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return static_cast<dnnl::deconvolution_forward::primitive_desc &>(
                pd_cache.at(op.get()));
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>("strides");
    auto dilates = op->get_attr<dims>("dilations");
    auto pads_begin = op->get_attr<dims>("pads_begin");
    auto pads_end = op->get_attr<dims>("pads_end");
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr("primitive_attr_key")) {
        int64_t key = op->get_attr<int64_t>("primitive_attr_key");
        prm_attr = prm_attr_mgr.get_attr(key);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto weight = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    dnnl::deconvolution_forward::primitive_desc pd;
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        pd = dnnl::deconvolution_forward::primitive_desc(
                {prop_kind::forward_inference, algorithm::deconvolution_direct,
                        src, weight, bias, dst, strides, dilates, pads_begin,
                        pads_end},
                prm_attr, p_engine);
    } else {
        pd = dnnl::deconvolution_forward::primitive_desc(
                {prop_kind::forward_inference, algorithm::deconvolution_direct,
                        src, weight, dst, strides, dilates, pads_begin,
                        pads_end},
                prm_attr, p_engine);
    }

    pd_cache.insert({op.get(), pd});

    return pd;
}

inline dnnl::matmul::primitive_desc create_matmul_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return static_cast<dnnl::matmul::primitive_desc &>(
                pd_cache.at(op.get()));
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr("primitive_attr_key")) {
        int64_t key = op->get_attr<int64_t>("primitive_attr_key");
        prm_attr = prm_attr_mgr.get_attr(key);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto wei = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    dnnl::matmul::primitive_desc pd;
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        pd = dnnl::matmul::primitive_desc(
                {src, wei, bias, dst}, prm_attr, p_engine);
    } else {
        pd = dnnl::matmul::primitive_desc({src, wei, dst}, prm_attr, p_engine);
    }

    pd_cache.insert({op.get(), pd});

    return pd;
}

inline dnnl::pooling_v2_forward::primitive_desc create_pool_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return static_cast<dnnl::pooling_v2_forward::primitive_desc &>(
                pd_cache.at(op.get()));
    }

    dims strides = op->get_attr<dims>("strides");
    dims kernel = op->get_attr<dims>("kernel");
    dims pads_begin = op->get_attr<dims>("pads_begin");
    dims pads_end = op->get_attr<dims>("pads_end");
    dims dilations;
    if (op->has_attr("dilations")) {
        dilations = op->get_attr<dims>("dilations");
    }
    std::string data_format = op->get_attr<std::string>("data_format");

    dnnl::primitive_attr prm_attr;
    if (op->has_attr("primitive_attr_key")) {
        int64_t key = op->get_attr<int64_t>("primitive_attr_key");
        prm_attr = prm_attr_mgr.get_attr(key);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto src = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto dst = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    dilations = get_compatible_dilates(dilations, src.dims().size());

    // infer dnnl expilicit pad
    dims new_pads_end(pads_end);
    bool adj_pad = false;
    std::string rounding_type = "floor";
    if (op->has_attr("rounding_type")) {
        rounding_type = op->get_attr<std::string>("rounding_type");
    }
    if (rounding_type == "ceil") {
        dims src_sp = src.dims();
        src_sp.erase(src_sp.begin(), src_sp.begin() + 2);
        dims output_sp = dst.dims();
        output_sp.erase(output_sp.begin(), output_sp.begin() + 2);
        for (size_t i = 0; i < kernel.size(); ++i) {
            dim_t dilated = dilations[i] * (kernel[i] - 1) + 1;
            if (op->get_kind() == impl::op_kind::AvgPool) dilated += 1;
            dim_t cur_pads_end = (output_sp[i] - 1) * strides[i] + dilated
                    - src_sp[i] - pads_begin[i];
            new_pads_end[i] = cur_pads_end;
        }
        adj_pad = true;
    }

    algorithm algo = algorithm::undef;
    if (op->get_kind() == impl::op_kind::MaxPool
            || (op->get_kind() == op_kind::dnnl_pool
                    && op->get_attr<std::string>("kind") == "maxpool"))
        algo = algorithm::pooling_max;
    else if (op->get_kind() == impl::op_kind::AvgPool
            || (op->get_kind() == op_kind::dnnl_pool
                    && op->get_attr<std::string>("kind") == "avgpool")) {
        const bool exclude_pad = op->get_attr<bool>("exclude_pad");
        algo = (exclude_pad || adj_pad)
                ? algorithm::pooling_avg_exclude_padding
                : algorithm::pooling_avg_include_padding;
    } else {
        BACKEND_DNNL_ENFORCE(
                0, "Currently only int8 MaxPool/AvgPool is supported.");
    }

    dnnl::pooling_v2_forward::primitive_desc pd(
            {prop_kind::forward_inference, algo, src, dst, strides, kernel,
                    dilations, pads_begin, new_pads_end},
            prm_attr, p_engine);

    pd_cache.insert({op.get(), pd});

    return pd;
}

inline dnnl::convolution_backward_data::primitive_desc create_conv_bwd_data_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr, pd_cache_t &pd_cache) {
    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        return static_cast<dnnl::convolution_backward_data::primitive_desc &>(
                pd_cache.at(op.get()));
    }

    // prepare the operator attributes
    auto strides = op->get_attr<dims>("strides");
    auto dilates = op->get_attr<dims>("dilations");
    auto pads_begin = op->get_attr<dims>("pads_begin");
    auto pads_end = op->get_attr<dims>("pads_end");
    dilates = get_compatible_dilates(dilates);

    dnnl::primitive_attr prm_attr;
    if (op->has_attr("primitive_attr_key")) {
        int64_t key = op->get_attr<int64_t>("primitive_attr_key");
        prm_attr = prm_attr_mgr.get_attr(key);
    }
    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto diff_dst = make_dnnl_memory_desc(
            op->get_input_value(0)->get_logical_tensor());
    auto weight = make_dnnl_memory_desc(
            op->get_input_value(1)->get_logical_tensor());
    auto diff_src = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    auto fwd_hints = dnnl::convolution_forward::primitive_desc(
            {dnnl::prop_kind::forward_training,
                    dnnl::algorithm::convolution_direct, diff_src, weight,
                    diff_dst, strides, dilates, pads_begin, pads_end},
            dnnl::primitive_attr(), p_engine);

    dnnl::convolution_backward_data::primitive_desc pd(
            {dnnl::algorithm::convolution_direct, diff_src, weight, diff_dst,
                    strides, dilates, pads_begin, pads_end},
            p_engine, fwd_hints);

    pd_cache.insert({op.get(), pd});

    return pd;
}

struct op_executable {
    virtual ~op_executable() = default;
    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const = 0;
};

struct memory_reparser : public op_executable {
    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        UNUSED(stream);
        assertm(args.find(DNNL_ARG_FROM)->second.get_data_handle()
                        == args.find(DNNL_ARG_TO)->second.get_data_handle(),
                "memory_parser must be inplaced");
    }
};

struct conv_fwd_executable : public op_executable {
    conv_fwd_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_conv_pd(op, p_engine, prm_attr_mgr, pd_cache);
        prim_ = dnnl::convolution_forward(pd_);
        if (op->has_attr("with_sum"))
            with_sum_ = op->get_attr<bool>("with_sum");
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }

        prim_.execute(stream, args);
    }

private:
    dnnl::convolution_forward::primitive_desc pd_;
    dnnl::convolution_forward prim_;
    bool with_sum_ {false};
};

struct deconv_fwd_executable : public op_executable {
    deconv_fwd_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_deconv_pd(op, p_engine, prm_attr_mgr, pd_cache);
        prim_ = dnnl::deconvolution_forward(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        const std::unordered_map<int, memory> &cached_args = args;

        prim_.execute(stream, cached_args);
    }

private:
    dnnl::deconvolution_forward::primitive_desc pd_;
    dnnl::deconvolution_forward prim_;
};

struct matmul_executable : public op_executable {
    matmul_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_matmul_pd(op, p_engine, prm_attr_mgr, pd_cache);
        prim_ = dnnl::matmul(pd_);

        // The scratchpad size of pd created by using any format tag may be
        // different from the scratchpad size of pd created by using queried
        // optimal format tag
        dnnl::memory::desc stored = make_dnnl_memory_desc(
                op->get_output_value(1)->get_logical_tensor());
        dnnl::memory::desc real = pd_.scratchpad_desc();
        if (stored != real) {
            auto scratchpad_val = op->get_output_value(1);
            scratchpad_val->set_layout_type(impl::layout_type::any);
            fill_layout_info(scratchpad_val, real);
        }

        if (op->has_attr("with_sum"))
            with_sum_ = op->get_attr<bool>("with_sum");
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            memory &dst_mem
                    = const_cast<memory &>(args.find(DNNL_ARG_DST)->second);
            memory &psrc_mem = const_cast<memory &>(
                    args.find(DNNL_GRAPH_ARG_POST_SRC)->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, psrc_mem, dst_mem);
            }
        }
        prim_.execute(stream, args);
    }

private:
    dnnl::matmul::primitive_desc pd_;
    dnnl::matmul prim_;
    bool with_sum_ {false};
};

struct pool_executable : public op_executable {
    pool_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_pool_pd(op, p_engine, prm_attr_mgr, pd_cache);
        prim_ = dnnl::pooling_v2_forward(pd_);
    }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

private:
    dnnl::pooling_v2_forward::primitive_desc pd_;
    dnnl::pooling_v2_forward prim_;
};

struct reorder_executable : public op_executable {
    reorder_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr) {
        auto in_md = make_dnnl_memory_desc(
                op->get_input_value(0)->get_logical_tensor());
        auto out_md = make_dnnl_memory_desc(
                op->get_output_value(0)->get_logical_tensor());

        dnnl::primitive_attr prm_attr;
        if (op->has_attr("primitive_attr_key")) {
            int64_t attr_key = op->get_attr<int64_t>("primitive_attr_key");
            prm_attr = prm_attr_mgr.get_attr(attr_key);
        } else {
            int64_t attr_key = prm_attr_mgr.init_attr();
            op->set_attr<int64_t>("primitive_attr_key", attr_key);
            dnnl::primitive_attr &initial_attr
                    = prm_attr_mgr.get_attr(attr_key);

            int mask = 0;
            if (op->has_attr("axis") && op->has_attr("scales")) {
                int64_t axis = op->get_attr<int64_t>("axis");
                auto scales = op->get_attr<std::vector<float>>("scales");
                mask = scales.size() == 1 ? 0 : 1 << axis;
                initial_attr.set_output_scales(mask, scales);
            }

            if (op->has_attr("src_zps")) {
                auto zps = op->get_attr<std::vector<int64_t>>("src_zps");
                std::vector<int32_t> neg_zps = dnnl_impl::utils::fmap(zps,
                        [](int64_t zp) { return static_cast<int32_t>(-zp); });
                initial_attr.set_zero_points(DNNL_ARG_FROM, mask, neg_zps);
            } else if (op->has_attr("dst_zps")) {
                auto zps = op->get_attr<std::vector<int64_t>>("dst_zps");
                std::vector<int32_t> int32_zps = dnnl_impl::utils::fmap(zps,
                        [](int64_t zp) { return static_cast<int32_t>(zp); });
                initial_attr.set_zero_points(DNNL_ARG_TO, mask, int32_zps);
            } else if (op->get_kind() == op_kind::dnnl_u8_to_s8) {
                initial_attr.set_zero_points(DNNL_ARG_TO, mask, {-128});
            }

            prm_attr = initial_attr;
        }

        pd_ = dnnl::reorder::primitive_desc(
                p_engine, in_md, p_engine, out_md, prm_attr);
        prim_ = dnnl::reorder(pd_);
    }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

private:
    dnnl::reorder::primitive_desc pd_;
    dnnl::reorder prim_;
};

struct bn_folding : public op_executable {
    bn_folding(std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine) {
        epsilon_ = op->get_attr<float>("epsilon");
        data_format_ = op->get_attr<std::string>("data_format");
        filter_format_ = op->get_attr<std::string>("filter_format");
        with_bias_ = op->get_attr<bool>("with_bias");

        size_t in_idx = 0;
        auto weights = make_dnnl_memory_desc(
                op->get_input_value(in_idx++)->get_logical_tensor());
        auto bias = with_bias_ ? make_dnnl_memory_desc(
                            op->get_input_value(in_idx++)->get_logical_tensor())
                               : memory::desc();
        auto scale = make_dnnl_memory_desc(
                op->get_input_value(in_idx++)->get_logical_tensor());
        auto shift = make_dnnl_memory_desc(
                op->get_input_value(in_idx++)->get_logical_tensor());
        auto mean = make_dnnl_memory_desc(
                op->get_input_value(in_idx++)->get_logical_tensor());
        auto variance = make_dnnl_memory_desc(
                op->get_input_value(in_idx++)->get_logical_tensor());

        // 1. sqrt_variance = sqrt(variance + epsilon)

        // temp = variance + epsilon
        memory::dims epsilon_dims(variance.data.ndims, 1);
        epsilon_desc_ = memory::desc(
                epsilon_dims, memory::data_type::f32, memory::format_tag::a);
        dnnl::binary::desc add_d(
                algorithm::binary_add, variance, epsilon_desc_, variance);

        post_ops add_post_ops;
        // sqrt_variance = sqrt(temp)
        add_post_ops.append_eltwise(1.0f, algorithm::eltwise_sqrt, 0.0f, 0.0f);

        primitive_attr add_attr;
        add_attr.set_post_ops(add_post_ops);
        dnnl::binary::primitive_desc add_pd(add_d, add_attr, p_engine);
        add_prim_ = dnnl::binary(add_pd);

        // 2. updated_weight = weights * scale / sqrt_variance

        // expand 1D scale and variance to same ndims with weights
        new_scale_desc_ = expand(scale, weights.data.ndims);
        new_variance_desc_ = expand(variance, weights.data.ndims);

        // after expand, the c channel is on the last dimension, which
        // meet the requirement of NXC format. But for NCX format, we
        // need permute c channel to the second dimension
        if (filter_format_ == "NCX") { // matmul case
            new_scale_desc_ = permute_NXC2NCX(new_scale_desc_);
            new_variance_desc_ = permute_NXC2NCX(new_variance_desc_);
        }

        // after expand, the c channel is on the last dimension, which
        // meet the requirement of XIO format. But for OIX format, we
        // need permute c channel to the first dimension
        if (filter_format_ == "OIX") { // conv case
            new_scale_desc_ = permute_XIO2OIX(new_scale_desc_);
            new_variance_desc_ = permute_XIO2OIX(new_variance_desc_);
        }

        // temp = weights * scale
        dnnl::binary::desc mul_d(
                algorithm::binary_mul, weights, new_scale_desc_, weights);

        post_ops mul_post_ops;
        // updated_weight = temp / sqrt_variance
        mul_post_ops.append_binary(algorithm::binary_div, new_variance_desc_);

        primitive_attr mul_attr;
        mul_attr.set_post_ops(mul_post_ops);
        dnnl::binary::primitive_desc mul_pd(mul_d, mul_attr, p_engine);
        mul_prim_ = dnnl::binary(mul_pd);

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift

        // temp = bias - mean
        memory::desc valid_bias = bias.is_zero() ? mean : bias;
        dnnl::binary::desc sub_d(
                algorithm::binary_sub, valid_bias, mean, valid_bias);

        post_ops sub_post_ops;
        // temp = temp * scale
        sub_post_ops.append_binary(algorithm::binary_mul, scale);
        // temp = temp / sqrt_variance
        sub_post_ops.append_binary(algorithm::binary_div, variance);
        // temp = temp + shift
        sub_post_ops.append_binary(algorithm::binary_add, shift);

        primitive_attr sub_attr;
        sub_attr.set_post_ops(sub_post_ops);
        dnnl::binary::primitive_desc sub_pd(sub_d, sub_attr, p_engine);
        sub_prim_ = dnnl::binary(sub_pd);

        memory::dims scratchpad_dims = variance.dims();
        // sqrt_variance, zero_bias and others (like epslion),
        // or no need to alloc bias
        size_t factor = bias.is_zero() ? 3 : 2;
        scratchpad_dims[0] *= factor;
        scratchpad_desc_ = memory::desc(
                scratchpad_dims, variance.data_type(), memory::format_tag::a);
    }

    const memory::desc &scratchpad_desc() const { return scratchpad_desc_; }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        UNUSED(args);

        auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
        auto bias = with_bias_ ? args.find(DNNL_ARG_BIAS)->second : memory();
        auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
        auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
        auto mean = args.find(DNNL_ARG_MEAN)->second;
        auto variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

        auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
        auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

        // 0. split scratchpad buffer to specific intermediate memory
        // sqrt_variance
        char *buf_start = (char *)scratchpad.get_data_handle();
        memory sqrt_variance = make_dnnl_memory(variance.get_desc(),
                scratchpad.get_engine(), (void *)buf_start);
        buf_start += variance.get_desc().get_size();
        // zero_bias
        memory valid_bias = bias;
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            valid_bias = make_dnnl_memory(variance.get_desc(),
                    scratchpad.get_engine(), (void *)buf_start);
            buf_start += valid_bias.get_desc().get_size();
        }
        // epslion
        memory epsilon_mem = make_dnnl_memory(
                epsilon_desc_, scratchpad.get_engine(), (void *)buf_start);

        // 1. sqrt_variance = sqrt(variance + epsilon)
        if (variance.get_engine().get_kind() == engine::kind::cpu) {
            float *ptr = (float *)epsilon_mem.get_data_handle();
            *ptr = epsilon_;
        } else {
            engine cpu_eng(engine::kind::cpu, 0);
            memory cpu_mem = make_dnnl_memory(
                    epsilon_desc_, cpu_eng, (void *)&epsilon_);
            dnnl::reorder(cpu_mem, epsilon_mem)
                    .execute(stream, cpu_mem, epsilon_mem);
        }

        add_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}});

        // 2. updated_weight = weights * scale / sqrt_variance
        memory new_scale(
                new_scale_desc_, scale.get_engine(), scale.get_data_handle());
        memory new_sqrt_variance(new_variance_desc_, sqrt_variance.get_engine(),
                sqrt_variance.get_data_handle());
        mul_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                        {DNNL_ARG_DST, updated_weights},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                new_sqrt_variance}});

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            // initialize the bias with zero value
            std::vector<float> zero(
                    impl::utils::prod(variance.get_desc().dims()), 0.0f);
            if (mean.get_engine().get_kind() == engine::kind::cpu) {
                std::memcpy(valid_bias.get_data_handle(), zero.data(),
                        valid_bias.get_desc().get_size());
            } else {
                engine cpu_eng(engine::kind::cpu, 0);
                memory cpu_mem = make_dnnl_memory(
                        variance.get_desc(), cpu_eng, zero.data());
                dnnl::reorder(cpu_mem, valid_bias)
                        .execute(stream, cpu_mem, valid_bias);
            }
        }

        sub_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}});
    }

private:
    memory::desc scratchpad_desc_;

    dnnl::binary add_prim_;
    dnnl::binary mul_prim_;
    dnnl::binary sub_prim_;

    float epsilon_;
    std::string data_format_;
    std::string filter_format_;

    memory::desc epsilon_desc_;
    memory::desc new_scale_desc_;
    memory::desc new_variance_desc_;

    bool with_bias_ {false};
};

struct conv_bwd_data_executable : public op_executable {
    conv_bwd_data_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr,
            pd_cache_t &pd_cache) {
        pd_ = create_conv_bwd_data_pd(op, p_engine, prm_attr_mgr, pd_cache);
        prim_ = dnnl::convolution_backward_data(pd_);
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

private:
    dnnl::convolution_backward_data::primitive_desc pd_;
    dnnl::convolution_backward_data prim_;
    bool perm_dst_ {false};
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

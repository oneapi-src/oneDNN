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
#ifndef BACKEND_DNNL_SUBGRAPH_OP_EXECUTABLE_HPP
#define BACKEND_DNNL_SUBGRAPH_OP_EXECUTABLE_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "dnnl.hpp"

#include "backend/dnnl/legacy.hpp"
#include "backend/dnnl/subgraph/passes.hpp"

#define DNNL_GRAPH_ARG_POST_SRC -1

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

inline dnnl::convolution_forward::primitive_desc create_conv_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr) {
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

    if (op->has_attr("output_format")
            && op->get_attr<std::string>("output_format") == "NXC") {
        dst = permute_NXC2NCX(dst);
    }

    dnnl::convolution_forward::primitive_desc pd;
    if (op->has_attr("with_bias") && op->get_attr<bool>("with_bias")) {
        auto bias = make_dnnl_memory_desc(
                op->get_input_value(2)->get_logical_tensor());
        pd = dnnl::convolution_forward::primitive_desc(
                {prop_kind::forward, algorithm::convolution_direct, src, weight,
                        bias, dst, strides, dilates, pads_begin, pads_end},
                prm_attr, p_engine);
    } else {
        pd = dnnl::convolution_forward::primitive_desc(
                {prop_kind::forward, algorithm::convolution_direct, src, weight,
                        dst, strides, dilates, pads_begin, pads_end},
                prm_attr, p_engine);
    }

    return pd;
}

inline dnnl::matmul::primitive_desc create_matmul_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr) {
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
    return pd;
}

inline dnnl::pooling_v2_forward::primitive_desc create_pool_pd(
        std::shared_ptr<impl::op_t> &op, const dnnl::engine &p_engine,
        primitive_attr_mgr &prm_attr_mgr) {
    dims strides = op->get_attr<dims>("strides");
    dims kernel = op->get_attr<dims>("kernel");
    dims pads_begin = op->get_attr<dims>("pads_begin");
    dims pads_end = op->get_attr<dims>("pads_end");
    dims dilations = op->get_attr<dims>("dilations");
    dilations = get_compatible_dilates(dilations);
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

    if (op->has_attr("output_format")
            && op->get_attr<std::string>("output_format") == "NXC") {
        dst = permute_NXC2NCX(dst);
    }
    algorithm algo = algorithm::undef;
    if (op->get_kind() == op_kind::MaxPool
            || op->get_kind() == op_kind::dnnl_maxpool)
        algo = algorithm::pooling_max;
    else {
        BACKEND_DNNL_ENFORCE(0, "Currently only int8 MaxPool is supported.");
    }

    dnnl::pooling_v2_forward::primitive_desc pd(
            {prop_kind::forward_inference, algo, src, dst, strides, kernel,
                    dilations, pads_begin, pads_end},
            prm_attr, p_engine);
    return pd;
}

struct op_executable {
    virtual ~op_executable() {}
    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const = 0;
};

struct memory_reparser : public op_executable {
    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        UNUSED(stream);
        const memory &in_mem = args.find(DNNL_ARG_FROM)->second;
        memory &out_mem = const_cast<memory &>(args.find(DNNL_ARG_TO)->second);
        out_mem.set_data_handle(in_mem.get_data_handle());
    }
};

struct conv_fwd_executable : public op_executable {
    conv_fwd_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr) {
        pd_ = create_conv_pd(op, p_engine, prm_attr_mgr);
        if (op->has_attr("with_sum"))
            with_sum_ = op->get_attr<bool>("with_sum");
        if (op->has_attr("output_format")
                && op->get_attr<std::string>("output_format") == "NXC")
            perm_dst_ = true;
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        std::unordered_map<int, memory> cached_args = args;

        if (perm_dst_) {
            memory dst_mem = make_dnnl_memory(pd_.dst_desc(), pd_.get_engine(),
                    args.find(DNNL_ARG_DST)->second.get_data_handle());
            cached_args[DNNL_ARG_DST] = dst_mem;
        }

        if (with_sum_) {
            memory &psrc_mem
                    = cached_args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            memory &dst_mem = cached_args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, psrc_mem, dst_mem);
            }
        }

        dnnl::convolution_forward(pd_).execute(stream, cached_args);
    }

private:
    dnnl::convolution_forward::primitive_desc pd_;
    bool with_sum_ {false};
    bool perm_dst_ {false};
};

struct matmul_executable : public op_executable {
    matmul_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr) {
        pd_ = create_matmul_pd(op, p_engine, prm_attr_mgr);

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
        dnnl::matmul(pd_).execute(stream, args);
    }

private:
    dnnl::matmul::primitive_desc pd_;
    bool with_sum_ {false};
};

struct pool_executable : public op_executable {
    pool_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr) {
        pd_ = create_pool_pd(op, p_engine, prm_attr_mgr);
        if (op->has_attr("output_format")
                && op->get_attr<std::string>("output_format") == "NXC")
            perm_dst_ = true;
    }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        std::unordered_map<int, memory> cached_args_ = args;
        if (perm_dst_) {
            memory dst_mem = make_dnnl_memory(pd_.dst_desc(), pd_.get_engine(),
                    args.find(DNNL_ARG_DST)->second.get_data_handle());
            cached_args_[DNNL_ARG_DST] = dst_mem;
        }
        dnnl::pooling_v2_forward(pd_).execute(stream, cached_args_);
    }

private:
    dnnl::pooling_v2_forward::primitive_desc pd_;
    bool perm_dst_ {false};
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
            if (op->has_attr("axis") && op->has_attr("scales")) {
                int64_t axis = op->get_attr<int64_t>("axis");
                auto scales = op->get_attr<std::vector<float>>("scales");
                int mask = scales.size() == 1 ? 0 : 1 << axis;
                initial_attr.set_output_scales(mask, scales);
            }
            prm_attr = initial_attr;
        }

        pd_ = dnnl::reorder::primitive_desc(
                p_engine, in_md, p_engine, out_md, prm_attr);
    }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        dnnl::reorder(pd_).execute(stream, args);
    }

private:
    dnnl::reorder::primitive_desc pd_;
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

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

    int64_t key = op->get_attr<int64_t>("primitive_attr_key");
    dnnl::primitive_attr prm_attr = prm_attr_mgr.get_attr(key);
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
    int64_t key = op->get_attr<int64_t>("primitive_attr_key");
    dnnl::primitive_attr prm_attr = prm_attr_mgr.get_attr(key);
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

struct op_executable {
    virtual ~op_executable() {}
    virtual void execute(
            const stream &stream, const std::unordered_map<int, memory> &args)
            = 0;
};

struct memory_reparser : public op_executable {
    virtual void execute(
            const stream &stream, const std::unordered_map<int, memory> &args) {
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
        if (op->has_attr("output_format"))
            output_format_ = op->get_attr<std::string>("output_format");
    }

    memory::desc scratchpad_desc() const { return pd_.scratchpad_desc(); }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) override {
        // only first iteration
        if (first_iteration_) {
            cached_args = args;

            if (output_format_ == "NXC") {
                org_dst_mem_ = args.find(DNNL_ARG_DST)->second;
                dst_mem_ = make_dnnl_memory(pd_.dst_desc(), pd_.get_engine(),
                        org_dst_mem_.get_data_handle());
                cached_args[DNNL_ARG_DST] = dst_mem_;
                perm_dst_ = true;
            } else {
                dst_mem_ = cached_args.find(DNNL_ARG_DST)->second;
            }

            if (with_sum_) {
                psrc_mem_ = cached_args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
                if (psrc_mem_.get_data_handle() != dst_mem_.get_data_handle()) {
                    psrc_reorder_pd_ = dnnl::reorder::primitive_desc(
                            psrc_mem_, dst_mem_);
                    reorder_psrc_ = true;
                }
            }
        }

        // every iteration
        if (perm_dst_) {
            dst_mem_.set_data_handle(org_dst_mem_.get_data_handle());
        }

        if (reorder_psrc_) {
            dnnl::reorder(psrc_reorder_pd_)
                    .execute(stream, psrc_mem_, dst_mem_);
        }

        dnnl::convolution_forward(pd_).execute(stream, cached_args);
        first_iteration_ = false;
    }

private:
    dnnl::convolution_forward::primitive_desc pd_;
    bool with_sum_ {false};
    std::string output_format_ {"NCX"};

    std::unordered_map<int, memory> cached_args;
    bool first_iteration_ {true};
    bool perm_dst_ {false};
    memory psrc_mem_;
    memory dst_mem_;
    memory org_dst_mem_;
    dnnl::reorder::primitive_desc psrc_reorder_pd_;
    bool reorder_psrc_ {false};
};

struct matmul_executable : public op_executable {
    matmul_executable(std::shared_ptr<impl::op_t> &op,
            const dnnl::engine &p_engine, primitive_attr_mgr &prm_attr_mgr) {
        pd_ = create_matmul_pd(op, p_engine, prm_attr_mgr);
        if (op->has_attr("with_sum"))
            with_sum_ = op->get_attr<bool>("with_sum");
    }

    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) override {
        if (with_sum_) {
            auto post_src_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            auto dst_mem = args.find(DNNL_ARG_DST)->second;
            if (post_src_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder::primitive_desc post_src_reorder_pd(
                        post_src_mem, dst_mem);
                dnnl::reorder(post_src_reorder_pd)
                        .execute(stream, post_src_mem, dst_mem);
            }
        }

        dnnl::matmul(pd_).execute(stream, args);
    }

private:
    dnnl::matmul::primitive_desc pd_;
    bool with_sum_ {false};
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

    virtual void execute(
            const stream &stream, const std::unordered_map<int, memory> &args) {
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

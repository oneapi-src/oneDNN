/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_DNNL_OPERATORS_BINARY_HPP
#define BACKEND_DNNL_OPERATORS_BINARY_HPP

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "interface/backend.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/legacy.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace bin {
enum binary_inputs { kSrc0, kSrc1 };
enum binary_outputs { kDst };
} // namespace bin

/// Currently, we only support unidirectional broadcast. The shape of src0
/// and src1 have to meet the one of following requirement:
/// property 1: src0 and src1 both have exactly the same shape.
/// property 2: src0 and src1 all have the same number of dimensions
///             and the length of each src1 dimensions is either a common
///             length or src1's length is 1.
/// property 3: src1 has too few dimensions, and src1 can have its
///             shapes prepended with a dimension of length 1 to
///             satisfy property 2.
/// \note src0 and src1 in the above description can be swapped
struct binary : public dnnl::binary, public kernel_base {
private:
    primitive_desc pd_;

    std::string auto_broadcast_ {"numpy"};
    algorithm alg_kind_;

    size_t idx_src0_ {bin::kSrc0};
    size_t idx_src1_ {bin::kSrc1};
    size_t idx_dst_ {bin::kDst};

    bool broadcast_ {false};
    bool unidirectional_ {false};

    dnnl::memory expected_src1_;
    dnnl::memory expected_dst_;

    void *expected_dst_buf_ {nullptr};

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;
    impl::allocator_t *g_alloc_;

    bool require_broadcast(const dnnl::memory::desc &src0,
            const dnnl::memory::desc &src1, const dnnl::memory::desc &dst) {
        return !(src0.dims() == src1.dims() && src0.dims() == dst.dims());
    }

public:
    ~binary() {
        if (expected_dst_buf_)
            allocator::free(expected_dst_buf_, p_engine_, g_alloc_);
    }

    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using ltw = impl::logical_tensor_wrapper;
        using desc = dnnl::memory::desc;

        if (anode->has_attr("auto_broadcast")) {
            auto_broadcast_ = anode->get_attr<std::string>("auto_broadcast");
        }

        switch (anode->get_op_kind()) {
            case op_kind::Add: alg_kind_ = algorithm::binary_add; break;
            case op_kind::Multiply: alg_kind_ = algorithm::binary_mul; break;
            case op_kind::Maximum: alg_kind_ = algorithm::binary_max; break;
            case op_kind::Minimum: alg_kind_ = algorithm::binary_min; break;
            default: return status::compile_fail;
        }

        p_engine_ = make_dnnl_engine(*g_engine);

        // dnnl only support src1 broadcast now
        if (ltw(inputs[idx_src0_]).vdims() != ltw(outputs[idx_dst_]).vdims()) {
            std::swap(idx_src0_, idx_src1_);
        }

        desc src0 = make_dnnl_memory_desc(inputs.at(idx_src0_));
        desc src1 = make_dnnl_memory_desc(inputs.at(idx_src1_));
        desc dst = make_dnnl_memory_desc(outputs.at(idx_dst_));

        // expand for broadcast
        if (require_broadcast(src0, src1, dst)) {
            if (auto_broadcast_ != "numpy") return status::compile_fail;

            broadcast_ = true;
            int src0_ndims = src0.data.ndims;
            int src1_ndims = src1.data.ndims;
            if (src0.dims() == dst.dims() && (src0_ndims >= src1_ndims)) {
                src1 = expand(src1, src0_ndims);
                unidirectional_ = true;
            } else {
                return status::compile_fail;
            }
        }

        // to any for allowing dnnl choose optimal layout for dst
        desc dst_any(dst.dims(), dst.data_type(), format_tag::any);
        pd_ = primitive_desc({alg_kind_, src0, src1, dst_any}, p_engine_);

        impl::logical_tensor_t *orgi_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(idx_dst_));
        fill_layout_info(orgi_dst_lt, pd_.dst_desc());
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(anode);

        memory src0 = make_dnnl_memory(inputs.at(idx_src0_), p_engine_);
        memory src1 = make_dnnl_memory(inputs.at(idx_src1_), p_engine_);
        memory dst = make_dnnl_memory(outputs.at(idx_dst_), p_engine_);

        // Deal with src1; If it's:
        // broadcasted: parse its buffer with the reshaped desc
        // not broadcasted:use the origin memory object directly
        if (broadcast_) {
            if (unidirectional_) {
                expected_src1_ = memory(
                        pd_.src1_desc(), p_engine_, src1.get_data_handle());
            }
        } else {
            expected_src1_ = src1;
        }

        g_alloc_ = g_stream->get_engine()->get_allocator();

        // Deal with the dst:
        // when create the primitive, we use any format for dst, so the
        // optiminal layout may be different from the original. we need
        // to check this and alloc new memory for optiminal dst
        if (pd_.dst_desc() != dst.get_desc()) {
            if (!expected_dst_) {
                expected_dst_buf_ = allocator::malloc(
                        pd_.dst_desc().get_size(), p_engine_, g_alloc_);
                expected_dst_
                        = memory(pd_.dst_desc(), p_engine_, expected_dst_buf_);
            }
        } else {
            expected_dst_ = dst;
        }

        exec_args args {{DNNL_ARG_SRC_0, src0},
                {DNNL_ARG_SRC_1, expected_src1_},
                {DNNL_ARG_DST, expected_dst_}};

        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        dnnl::binary(pd_).execute(p_stream_, args);

        if (expected_dst_ != dst) {
            dnnl::reorder(expected_dst_, dst)
                    .execute(p_stream_, expected_dst_, dst);
        }
        return impl::status::success;
    }
};

struct bias_add : public dnnl::binary, public kernel_base {
private:
    primitive_desc pd_;
    std::string data_format_ {"NXC"};

    size_t idx_src_ {0};
    size_t idx_bias_ {1};
    size_t idx_dst_ {0};

    dnnl::memory expected_bias_;
    dnnl::memory expected_dst_;

    void *expected_dst_buf_ {nullptr};

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;
    impl::allocator_t *g_alloc_;

public:
    ~bias_add() {
        if (expected_dst_buf_)
            allocator::free(expected_dst_buf_, p_engine_, g_alloc_);
    }

    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = dnnl::memory::desc;

        data_format_ = anode->get_attr<std::string>("data_format");

        desc src = make_dnnl_memory_desc(inputs.at(idx_src_));
        desc bias = make_dnnl_memory_desc(inputs.at(idx_bias_));
        desc dst = make_dnnl_memory_desc(outputs.at(idx_dst_));

        int src_ndims = src.data.ndims;

        // do expand always, c in the last dim
        bias = expand(bias, src_ndims);

        // do permute
        // NCX data_format_ means src's channel is in the second dim. so we
        // need permute the expanded bias to NCX too
        if (data_format_ == "NCX") { bias = permute_NXC2NCX(bias); }

        p_engine_ = make_dnnl_engine(*g_engine);

        desc dst_any(dst.dims(), dst.data_type(), format_tag::any);
        pd_ = primitive_desc(
                {algorithm::binary_add, src, bias, dst_any}, p_engine_);

        impl::logical_tensor_t *orgi_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(idx_dst_));
        fill_layout_info(orgi_dst_lt, pd_.dst_desc());
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(anode);

        memory src = make_dnnl_memory(inputs.at(idx_src_), p_engine_);
        memory bias = make_dnnl_memory(inputs.at(idx_bias_), p_engine_);
        memory dst = make_dnnl_memory(outputs.at(idx_dst_), p_engine_);

        // Deal with bias:
        // bias is always broadcasted: parse its buffer with the reshaped desc
        expected_bias_
                = memory(pd_.src1_desc(), p_engine_, bias.get_data_handle());

        g_alloc_ = g_stream->get_engine()->get_allocator();

        // Deal with the dst:
        // when create the primitive, we use any format for dst, so the
        // optiminal layout may be different from the original. we need
        // to check this and alloc new memory for optiminal dst
        if (pd_.dst_desc() != dst.get_desc()) {
            if (!expected_dst_) {
                expected_dst_buf_ = allocator::malloc(
                        pd_.dst_desc().get_size(), p_engine_, g_alloc_);
                expected_dst_
                        = memory(pd_.dst_desc(), p_engine_, expected_dst_buf_);
            }
        } else {
            expected_dst_ = dst;
        }

        exec_args args {{DNNL_ARG_SRC_0, src}, {DNNL_ARG_SRC_1, expected_bias_},
                {DNNL_ARG_DST, expected_dst_}};

        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);

        dnnl::binary(pd_).execute(p_stream_, args);

        if (expected_dst_ != dst) {
            dnnl::reorder(expected_dst_, dst)
                    .execute(p_stream_, expected_dst_, dst);
        }
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

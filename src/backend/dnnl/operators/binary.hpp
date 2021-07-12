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
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "interface/backend.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/f32_kernel_resource.hpp"
#include "backend/dnnl/legacy.hpp"
#include "backend/dnnl/resource.hpp"
#include "backend/dnnl/scratchpad.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace bin {
enum binary_inputs { kSrc0, kSrc1 };
enum binary_outputs { kDst };
enum mem_keys {
    kOpt_src0,
    kOpt_src1,
    kOpt_dst,
};
} // namespace bin

// We support both multidirectional and unidirectional broadcast. And the
// broadcast semantics is consistent with PyTorch broadcast:
// Two tensors are “broadcastable” if the following rules hold:
// - Each tensor has at least one dimension.
// - When iterating over the dimension sizes, starting at the trailing
//   dimension, the dimension sizes must either be equal, one of them is 1, or
//   one of them does not exist.
struct binary : public dnnl::binary, public kernel_base {
    using super = dnnl::binary;

private:
    primitive_desc pd_;
    attr_t attr_;
    dnnl::binary prim_;

    std::string auto_broadcast_ {"numpy"};
    algorithm alg_kind_;

    size_t idx_src0_ {bin::kSrc0};
    size_t idx_src1_ {bin::kSrc1};
    size_t idx_dst_ {bin::kDst};

    bool broadcast_ {false};
    bool with_post_sum_ {false};

    dnnl::reorder::primitive_desc dst_reorder_pd_;

    dnnl::engine p_engine_;

    registry_t registry_;
    size_t res_key_;
    resource_cache_t::creator_t resource_ctor_;

    f32_kernel_resource_t::desc_t res_desc_;

    bool require_broadcast(const dnnl::memory::desc &src0,
            const dnnl::memory::desc &src1, const dnnl::memory::desc &dst) {
        return !(src0.dims() == src1.dims() && src0.dims() == dst.dims());
    }

    // (3, 4) * (3, 4) is doable
    // (1, 4) * (3, 4) is doable
    // (3, 4, 5) * (4, 5) is doable
    // (3, 4, 5) * (1, 5) is doable
    // (3, 4, 5) * (2, 4, 5) is NOT doable
    bool doable(const std::vector<dim_t> &shape_0,
            const std::vector<dim_t> &shape_1) {
        const int ndims_0 = static_cast<int>(shape_0.size());
        const int ndims_1 = static_cast<int>(shape_1.size());
        const int small = ndims_0 < ndims_1 ? ndims_0 : ndims_1;
        for (int i = 1; i <= small; ++i) {
            bool match = shape_0[ndims_0 - i] == shape_1[ndims_1 - i]
                    || shape_0[ndims_0 - i] == 1 || shape_1[ndims_1 - i] == 1;
            if (!match) return false;
        }
        return true;
    }

    /**
     * Check if binary operator fuses add
     *
     * @param kind operator kind
     * @return whether the operator fused add
     */
    static bool fuse_add(op_kind_t kind) {
        static const std::unordered_set<op_kind_t, enum_hash> with_add_set {
                op_kind::multiply_add, op_kind::maximum_add,
                op_kind::minimum_add};
        return with_add_set.find(kind) != with_add_set.end();
    }

    static algorithm get_eltwise_algo(op_kind_t kind) {
        switch (static_cast<int>(kind)) {
            case op_kind::add_relu:
            case op_kind::multiply_relu:
            case op_kind::maximum_relu:
            case op_kind::minimum_relu: return (algorithm::eltwise_relu);
            case op_kind::add_sigmoid:
            case op_kind::multiply_sigmoid:
            case op_kind::maximum_sigmoid:
            case op_kind::minimum_sigmoid: return (algorithm::eltwise_logistic);
            default: return (algorithm::undef);
        }
    }

    static algorithm get_binary_algo(op_kind_t kind) {
        switch (static_cast<int>(kind)) {
            case op_kind::Add:
            case op_kind::add_relu:
            case op_kind::add_sigmoid: return (algorithm::binary_add);
            case op_kind::Multiply:
            case op_kind::multiply_relu:
            case op_kind::multiply_sigmoid:
            case op_kind::multiply_add: return (algorithm::binary_mul);
            case op_kind::Maximum:
            case op_kind::maximum_relu:
            case op_kind::maximum_sigmoid:
            case op_kind::maximum_add: return (algorithm::binary_max);
            case op_kind::Minimum:
            case op_kind::minimum_relu:
            case op_kind::minimum_sigmoid:
            case op_kind::minimum_add: return (algorithm::binary_min);
            default: return (algorithm::undef);
        }
    }

    impl::status_t prepare_inplace_pairs_impl(const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(g_engine);
        if (with_post_sum_) {
            size_t input_idx = idx_src1_ + 1;
            constexpr size_t output_idx = 0;

            const logical_tensor_wrapper post_src_lt(inputs[input_idx]);
            const logical_tensor_wrapper dst_lt(outputs[output_idx]);
            if (post_src_lt.is_opaque() && dst_lt.is_opaque()
                    && post_src_lt.is_similar(dst_lt))
                inplace_pairs_.push_back(
                        {inputs[input_idx].id, outputs[output_idx].id});
        }
        return impl::status::success;
    }

public:
    ~binary() {}

    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using ltw = impl::logical_tensor_wrapper;
        using desc = dnnl::memory::desc;

        if (!doable(ltw(inputs[idx_src0_]).vdims(),
                    ltw(inputs[idx_src1_]).vdims())) {
            return status::invalid_shape;
        }

        if (op->has_attr("auto_broadcast")) {
            auto_broadcast_ = op->get_attr<std::string>("auto_broadcast");
        }

        alg_kind_ = get_binary_algo(op->get_kind());
        if (alg_kind_ == algorithm::undef) return status::compile_fail;

        p_engine_ = make_dnnl_engine(*g_engine);

        res_desc_.cvt_src_ = make_dnnl_memory_desc(inputs.at(idx_src0_));
        res_desc_.cvt_src1_ = make_dnnl_memory_desc(inputs.at(idx_src1_));
        res_desc_.cvt_dst_ = make_dnnl_memory_desc(outputs.at(idx_dst_));

        // expand for broadcast
        if (require_broadcast(res_desc_.cvt_src_, res_desc_.cvt_src1_,
                    res_desc_.cvt_dst_)) {
            if (auto_broadcast_ != "numpy") return status::compile_fail;

            broadcast_ = true;
            res_desc_.cvt_src_
                    = expand(res_desc_.cvt_src_, res_desc_.cvt_dst_.data.ndims);
            res_desc_.cvt_src1_ = expand(
                    res_desc_.cvt_src1_, res_desc_.cvt_dst_.data.ndims);
        }

        // to any for allowing dnnl choose optimal layout for dst
        desc dst_any(res_desc_.cvt_dst_.dims(), res_desc_.cvt_dst_.data_type(),
                format_tag::any);

        // set sum post-op
        with_post_sum_ = fuse_add(op->get_kind());
        if (with_post_sum_) {
            res_desc_.cvt_post_src_ = make_dnnl_memory_desc(inputs.back());
            attr_ = attr_t::fuse_sum();
        }

        // set eltwise post-op
        const algorithm algo = get_eltwise_algo(op->get_kind());
        if (algo != algorithm::undef) {
            attr_ = attr_t::fuse_eltwise(algo, 1.f, 0.f, 0.f);
        }

        pd_ = primitive_desc(
                {alg_kind_, res_desc_.cvt_src_, res_desc_.cvt_src1_, dst_any},
                attr_, p_engine_);

        prim_ = super(pd_);

        // Note: opt_src should be equal to cvt_src, because we don't use any
        // for input
        res_desc_.opt_src_ = pd_.src0_desc();
        res_desc_.opt_src1_ = pd_.src1_desc();
        res_desc_.opt_dst_ = pd_.dst_desc();
        if (impl::logical_tensor_wrapper(outputs.at(idx_dst_)).is_any()) {
            res_desc_.cvt_dst_ = pd_.dst_desc();
        }

        impl::logical_tensor_t *orgi_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(idx_dst_));
        fill_layout_info(orgi_dst_lt, pd_.dst_desc());

        registrar_t registrar = registry_.registrar();
        if (res_desc_.opt_dst_ != res_desc_.cvt_dst_) {
            registrar.book(bin::kOpt_dst, res_desc_.opt_dst_.get_size());
            dst_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                    res_desc_.opt_dst_, p_engine_, res_desc_.cvt_dst_);
        }

        res_key_ = impl::utils::hash_combine(0, res_desc_);
        res_key_ = impl::utils::hash_combine(res_key_, p_engine_.get());
        resource_ctor_ = [this]() {
            return std::unique_ptr<resource_t>(new f32_kernel_resource_t(
                    this->res_desc_, this->p_engine_));
        };

        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        // each thread's own local resource
        resource_cache_t res_cache;
        f32_kernel_resource_t *res = res_cache.get<f32_kernel_resource_t>(
                res_key_, resource_ctor_, true /*is f32*/);

        impl::allocator_t *g_alloc_ = g_stream->get_engine()->get_allocator();
        temporary_scratchpad_t scratchpad(
                registry_.size(), p_engine_, *g_alloc_);
        grantor_t grantor = registry_.grantor(scratchpad.get_buffer());

        res->cvt_src_.set_data_handle(inputs.at(idx_src0_).get_data_handle());
        res->cvt_src1_.set_data_handle(inputs.at(idx_src1_).get_data_handle());
        res->cvt_dst_.set_data_handle(outputs.at(idx_dst_).get_data_handle());

        res->opt_src_.set_data_handle(res->cvt_src_.get_data_handle());
        res->opt_src1_.set_data_handle(res->cvt_src1_.get_data_handle());

        // Deal with the dst:
        // when create the primitive, we use any format for dst, so the
        // optimal layout may be different from the original. we need
        // to check this and alloc new memory for optimal dst
        if (res_desc_.cvt_dst_ != res_desc_.opt_dst_) {
            res->opt_dst_.set_data_handle(grantor.get(bin::kOpt_dst));
        } else {
            res->opt_dst_.set_data_handle(res->cvt_dst_.get_data_handle());
        }

        if (with_post_sum_) {
            res->cvt_post_src_.set_data_handle(inputs.back().get_data_handle());
        }

        if (with_post_sum_
                && res->cvt_post_src_.get_data_handle()
                        != res->opt_dst_.get_data_handle()) {
            dnnl::reorder(res->cvt_post_src_, res->opt_dst_)
                    .execute(p_stream, res->cvt_post_src_, res->opt_dst_);
        }

        dnnl::binary(pd_).execute(p_stream, res->exec_args_);

        if (res_desc_.cvt_dst_ != res_desc_.opt_dst_) {
            dnnl::reorder(dst_reorder_pd_)
                    .execute(p_stream, res->opt_dst_, res->cvt_dst_);
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

    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = dnnl::memory::desc;

        data_format_ = op->get_attr<std::string>("data_format");

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

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);

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

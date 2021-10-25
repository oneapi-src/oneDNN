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

#ifndef BACKEND_DNNL_KERNELS_POOL_HPP
#define BACKEND_DNNL_KERNELS_POOL_HPP

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include "interface/c_types_map.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/f32_kernel_resource.hpp"
#include "backend/dnnl/scratchpad.hpp"
#include "backend/dnnl/tensor.hpp"
#include "backend/dnnl/thread_local_cache.hpp"
#include "backend/dnnl/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace pool {
enum pool_inputs { kSrc };
enum pool_outputs { kDst };
enum mem_keys {
    kOpt_src,
    kOpt_dst,
    kScratchpad,
    kWorkspace,
};
} // namespace pool

namespace pool_bwd {
enum pool_bwd_inputs { kSrc, kDiff_dst };
enum pool_bwd_outputs { kDiff_src };
} // namespace pool_bwd

namespace pool_bwd_with_indices {
enum maxpool_bwd_inputs { kSrc, kIndices, kDiff_dst };
} // namespace pool_bwd_with_indices

struct pooling_forward : public dnnl::pooling_v2_forward, public kernel_base {
    using super = dnnl::pooling_v2_forward;
    using dims_t = std::vector<dnnl::graph::impl::dim_t>;

private:
    attr_t attr_;
    algorithm algo_;
    primitive_desc pd_;
    dnnl::pooling_v2_forward prim_;

    bool is_training_ {false};
    bool with_post_binary_ {false};
    prop_kind prop_kind_;

    dnnl::engine p_engine_;

    bool with_workspace_;

    dnnl::reorder::primitive_desc src_reorder_pd_;
    dnnl::reorder::primitive_desc dst_reorder_pd_;

    registry_t registry_;
    std::function<std::shared_ptr<f32_kernel_resource_t>()> resource_ctor_;

    f32_kernel_resource_t::desc_t res_desc_;

    /**
     * Check if pooling operator fuses binary
     *
     * @param kind operator kind
     * @return whether the operator fused binary
     */
    static bool fuse_binary(op_kind_t kind) {
        static const std::unordered_set<op_kind_t, enum_hash> with_binary_set {
                op_kind::avgpool_add, op_kind::maxpool_add};
        return with_binary_set.find(kind) != with_binary_set.end();
    }

    static op_kind_t get_fuse_base_op(op_kind_t kind) {
        switch (static_cast<int>(kind)) {
            case op_kind::avgpool_add: return (impl::op_kind::AvgPool);
            case op_kind::maxpool_add: return (impl::op_kind::MaxPool);
            default: return (impl::op_kind::LastSymbol);
        }
    }

    static algorithm get_binary_algo(op_kind_t kind) {
        switch (static_cast<int>(kind)) {
            case op_kind::avgpool_add:
            case op_kind::maxpool_add: return (algorithm::binary_add);
            default: return (algorithm::undef);
        }
    }

public:
    virtual ~pooling_forward() {
        thread_local_cache_t<f32_kernel_resource_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
    }

public:
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        dims strides = op->get_attr<dims>("strides");
        dims kernel = op->get_attr<dims>("kernel");
        dims pads_begin = op->get_attr<dims>("pads_begin");
        dims pads_end = op->get_attr<dims>("pads_end");
        std::string data_format = op->get_attr<std::string>("data_format");

        res_desc_.cvt_src_ = make_dnnl_memory_desc(inputs.at(pool::kSrc));
        res_desc_.cvt_dst_ = make_dnnl_memory_desc(outputs.at(pool::kDst));
        if (data_format == "NXC") { // permute NXC to NCX
            res_desc_.cvt_src_ = permute_NXC2NCX(res_desc_.cvt_src_);
            res_desc_.cvt_dst_ = permute_NXC2NCX(res_desc_.cvt_dst_);
        }

        const auto kind = op->get_kind();
        dims dilations;
        if (kind == impl::op_kind::MaxPool
                || get_fuse_base_op(kind) == impl::op_kind::MaxPool) {
            algo_ = algorithm::pooling_max;
            dilations = op->get_attr<dims>("dilations");
            // default dilations are all 1s but in primitive, they're 0s.
            std::for_each(dilations.begin(), dilations.end(),
                    [](dim_t &v) { v -= 1; });
        } else if (kind == impl::op_kind::AvgPool
                || get_fuse_base_op(kind) == impl::op_kind::AvgPool) {
            dilations = dims(strides.size(), 0);
            bool exclude_pad = op->get_attr<bool>("exclude_pad");
            algo_ = exclude_pad ? algorithm::pooling_avg_exclude_padding
                                : algorithm::pooling_avg_include_padding;
        } else {
            BACKEND_DNNL_ENFORCE(0, "Unsupported pool op.");
        }

        if (fuse_binary(kind)) {
            with_post_binary_ = true;

            const int dst_ndims
                    = static_cast<int>(res_desc_.cvt_dst_.dims().size());
            res_desc_.cvt_post_src_ = make_dnnl_memory_desc(inputs.back());
            if (data_format == "NXC") {
                res_desc_.cvt_post_src_
                        = permute_NXC2NCX(res_desc_.cvt_post_src_);
            }
            res_desc_.cvt_post_src_
                    = expand(res_desc_.cvt_post_src_, dst_ndims);

            // currently, we support two scenarios that are optimized for post
            // binary, per tensor and per channel broadcast. That means
            // the expanded shape of post src should be all one or the
            // post_src_dim[c_axis] == dst_dims[c_axis]
            const int c_axis = 1;
            for (int i = dst_ndims - 1; i >= 0; i--) {
                if (res_desc_.cvt_dst_.dims()[i]
                                != res_desc_.cvt_post_src_.dims()[i]
                        && res_desc_.cvt_post_src_.dims()[i] != 1) {
                    return impl::status::compile_fail;
                }
            }
            attr_ = attr_t::fuse_binary(
                    res_desc_.cvt_post_src_, get_binary_algo(kind));
        }

        p_engine_ = make_dnnl_engine(*g_engine);

        // workaround: use src once issue intel/mkl-dnn#588 is
        // resolved
        auto expected_src = is_4c_blocked(res_desc_.cvt_src_)
                ? to_default_format(res_desc_.cvt_src_)
                : res_desc_.cvt_src_;
        auto any_dst = to_format_any(res_desc_.cvt_dst_);

        prop_kind_ = is_training_ ? prop_kind::forward
                                  : prop_kind::forward_inference;

        with_workspace_ = prop_kind_ == prop_kind::forward_training
                && algo_ == dnnl::algorithm::pooling_max;

        pd_ = primitive_desc({prop_kind_, algo_, expected_src, any_dst, strides,
                                     kernel, dilations, pads_begin, pads_end},
                attr_, p_engine_);
        prim_ = super(pd_);
        res_desc_.opt_src_ = pd_.src_desc();
        res_desc_.opt_dst_ = pd_.dst_desc();
        if (impl::logical_tensor_wrapper(outputs.at(pool::kDst)).is_any()) {
            res_desc_.cvt_dst_ = pd_.dst_desc();
        }
        res_desc_.scratchpad_ = pd_.scratchpad_desc();
        if (with_workspace_) res_desc_.workspace_ = pd_.workspace_desc();

        registrar_t registrar = registry_.registrar();

        if (res_desc_.opt_src_ != res_desc_.cvt_src_) {
            registrar.book(pool::kOpt_src, res_desc_.opt_src_.get_size());
            src_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                    res_desc_.cvt_src_, p_engine_, res_desc_.opt_src_);
        }

        if (res_desc_.opt_dst_ != res_desc_.cvt_dst_) {
            registrar.book(pool::kOpt_dst, res_desc_.opt_dst_.get_size());
            dst_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                    res_desc_.opt_dst_, p_engine_, res_desc_.cvt_dst_);
        }

        registrar.book(pool::kScratchpad, res_desc_.scratchpad_.get_size());
        if (with_workspace_)
            registrar.book(pool::kWorkspace, res_desc_.workspace_.get_size());

        impl::logical_tensor_t *ori_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(pool::kDst));
        if (data_format == "NXC") {
            memory::desc tmp = permute_NCX2NXC(pd_.dst_desc());
            fill_layout_info(ori_dst_lt, tmp);
        } else {
            fill_layout_info(ori_dst_lt, pd_.dst_desc());
        }

        resource_ctor_ = [this]() {
            return std::make_shared<f32_kernel_resource_t>(
                    this->res_desc_, this->p_engine_);
        };

        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        // each thread's own local resource
        thread_local_cache_t<f32_kernel_resource_t> res_cache;
        f32_kernel_resource_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        temporary_scratchpad_t scratchpad(registry_.size(), p_engine_, *alc);
        grantor_t grantor = registry_.grantor(scratchpad.get_buffer());

        res->cvt_src_.set_data_handle(inputs.at(pool::kSrc).get_data_handle());
        res->cvt_dst_.set_data_handle(outputs.at(pool::kDst).get_data_handle());

        if (res_desc_.cvt_src_ != res_desc_.opt_src_) {
            res->opt_src_.set_data_handle(grantor.get(pool::kOpt_src));
            dnnl::reorder(src_reorder_pd_)
                    .execute(p_stream, res->cvt_src_, res->opt_src_);
        } else {
            res->opt_src_.set_data_handle(res->cvt_src_.get_data_handle());
        }

        if (res_desc_.cvt_dst_ != res_desc_.opt_dst_) {
            res->opt_dst_.set_data_handle(grantor.get(pool::kOpt_dst));
        } else {
            res->opt_dst_.set_data_handle(res->cvt_dst_.get_data_handle());
        }

        res->scratchpad_.set_data_handle(grantor.get(pool::kScratchpad));
        if (with_workspace_)
            res->workspace_.set_data_handle(grantor.get(pool::kWorkspace));

        if (with_post_binary_) {
            res->cvt_post_src_.set_data_handle(inputs.back().get_data_handle());
        }

        prim_.execute(p_stream, res->exec_args_);

        // if output layout has been set and different from optimal layout
        // we have to do reorder
        if (res_desc_.cvt_dst_ != res_desc_.opt_dst_) {
            dnnl::reorder(dst_reorder_pd_)
                    .execute(p_stream, res->opt_dst_, res->cvt_dst_);
        }
        return impl::status::success;
    }
};

struct pooling_backward : public dnnl::pooling_v2_backward, public kernel_base {
    using super = dnnl::pooling_v2_backward;

private:
    dnnl::pooling_v2_forward::primitive_desc forward_hints_;
    primitive_desc pd_;
    op_kind_t kind_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    void compute(const tensor &diff_dst, const tensor &src, tensor &diff_src,
            const dnnl::engine &p_engine, impl::allocator_t *alc,
            const dnnl::stream &p_stream, tensor indices = tensor {}) {
        // generate indices tensor from src when it's needed
        // but can't get from function parameters
        if (kind_ == impl::op_kind::MaxPoolBackprop && indices.is_empty()) {
            auto expected_src = src.reorder_if_differ_in(
                    p_stream, forward_hints_.src_desc());
            auto expected_dst
                    = tensor {forward_hints_.dst_desc(), p_engine, alc};
            indices = tensor {forward_hints_.workspace_desc(), p_engine, alc};
            exec_args args {{DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_DST, expected_dst},
                    {DNNL_ARG_WORKSPACE, indices}};

            dnnl::pooling_v2_forward(forward_hints_).execute(p_stream, args);
        }

        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(p_stream, pd_.diff_dst_desc());
        auto expected_diff_src
                = diff_src.reorder_if_differ_in(p_stream, pd_.diff_src_desc());

        exec_args args = exec_args {
                {DNNL_ARG_DIFF_DST, expected_diff_dst},
                {DNNL_ARG_DIFF_SRC, expected_diff_src},
        };

        if (!indices.is_empty()) { args.insert({DNNL_ARG_WORKSPACE, indices}); }

        super(pd_).execute(p_stream, args);

        if (expected_diff_src != diff_src) {
            dnnl::reorder(expected_diff_src, diff_src)
                    .execute(p_stream, expected_diff_src, diff_src);
        }
    }

    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        // prepare the inputs and outputs' tensors' descs
        const desc src {inputs.at(pool_bwd::kSrc)};
        const desc diff_dst {inputs.at(pool_bwd::kDiff_dst)};
        impl::logical_tensor_t *diff_src_lt
                = const_cast<impl::logical_tensor_t *>(
                        &outputs.at(pool_bwd::kDiff_src));
        const desc diff_src {*diff_src_lt};

        dims strides = op->get_attr<dims>("strides");
        dims kernel = op->get_attr<dims>("kernel");
        dims pads_begin = op->get_attr<dims>("pads_begin");
        dims pads_end = op->get_attr<dims>("pads_end");

        kind_ = op->get_kind();
        algorithm algo = algorithm::undef;
        dims dilations {};
        if (kind_ == impl::op_kind::AvgPoolBackprop) {
            bool exclude_pad = op->get_attr<bool>("exclude_pad");
            algo = exclude_pad ? algorithm::pooling_avg_exclude_padding
                               : algorithm::pooling_avg_include_padding;
            dilations = dims(strides.size(), 0);
        } else if (kind_ == impl::op_kind::MaxPoolBackprop) {
            algo = algorithm::pooling_max;
            dilations = op->get_attr<dims>("dilations");
            // default dilations are all 1s but in primitive, they're 0s.
            std::for_each(dilations.begin(), dilations.end(),
                    [](dim_t &v) { v -= 1; });
        } else {
            return status::unsupported;
        }

        p_engine_ = make_dnnl_engine(*g_engine);
        forward_hints_ = dnnl::pooling_v2_forward::primitive_desc(
                {prop_kind::forward_training, algo, src, diff_dst, strides,
                        kernel, dilations, pads_begin, pads_end},
                p_engine_);

        pd_ = primitive_desc({algo, src, diff_dst, strides, kernel, dilations,
                                     pads_begin, pads_end},
                p_engine_, forward_hints_);

        const tensor::desc optimal_diff_src_desc {pd_.diff_src_desc()};
        fill_layout_info(diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        tensor src {inputs.at(pool_bwd::kSrc), p_engine_, alc};
        tensor diff_dst {};
        tensor indices {};
        if (op->get_kind() == impl::op_kind::MaxPoolBackprop
                && inputs.size() > pool_bwd_with_indices::kDiff_dst) {
            diff_dst = tensor {inputs.at(pool_bwd_with_indices::kDiff_dst),
                    p_engine_, alc};
            indices = tensor {
                    inputs.at(pool_bwd_with_indices::kIndices), p_engine_, alc};
        } else {
            diff_dst = tensor {inputs.at(pool_bwd::kDiff_dst), p_engine_, alc};
        }

        tensor diff_src {outputs.at(pool_bwd::kDiff_src), p_engine_, alc};
        pooling_backward::compute(
                diff_dst, src, diff_src, p_engine_, alc, p_stream_, indices);
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

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

template <bool quantized>
struct pooling_fwd_t : public kernel_base_t {
private:
    dnnl::engine p_engine_;
    impl::allocator_t *g_alloc_;

    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

public:
    ~pooling_fwd_t() override {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
    }

    impl::status_t compile_impl(const dnnl_partition_impl_t *part,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        // TODO(wuxun): since oneDNN pooling primitive only support u8u8 or
        // s8s8 on CPU device for now, we need to check whether the data types
        // between input and output are compatible. If we enable this check in
        // op schema or primitive supports u8s8/s8u8, then this check can be
        // safely removed.
        if (inputs[0].data_type != outputs[0].data_type)
            return status::unsupported;

        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = g_engine->get_allocator();

        subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_);
        BACKEND_DNNL_CHECK(
                set_given_inputs_outputs(subgraph_, inputs, outputs));

        subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
            return this->memory_planner_.get_memory_info(val);
        });
        pass_pipeline_t pipeline(vis);

        // for those primitive ops like pooling, it requires the scales and zps
        // between input tensor and output tensor are the same. So here, we
        // don't need to split Dequant and Quant ops firstly, it should be okay
        // to directly fuse Dequant and Quant into int8 pool op.
        if (quantized) { BACKEND_DNNL_ADD_PASS(pipeline, fuse_to_int8_pool); }

        BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);
        BACKEND_DNNL_ADD_PASS(pipeline, insert_permute);

        BACKEND_DNNL_ADD_PASS(pipeline, infer_shape);

        pipeline.reset_visualize_arg(true, false);
        BACKEND_DNNL_ADD_PASS(pipeline, infer_type);
        BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

        // bind the memory for each op
        auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
            return memory_planner_.run(sg);
        };
        pipeline.reset_visualize_arg(true, true);
        BACKEND_DNNL_ADD_PASS(pipeline, memory_plan);
        BACKEND_DNNL_ADD_PASS(pipeline, compile_ops);

        // Run the added passes
        BACKEND_DNNL_CHECK(pipeline.run(subgraph_));

        // fill information for inputs logical tensors
        for (size_t i = 0; i < inputs.size(); i++) {
            BACKEND_DNNL_CHECK(set_shape_and_layout(
                    const_cast<impl::logical_tensor_t &>(inputs[i]),
                    subgraph_->ins_[i]));
        }

        // fill information for outputs logical tensors
        for (size_t i = 0; i < outputs.size(); i++) {
            BACKEND_DNNL_CHECK(set_shape_and_layout(
                    const_cast<impl::logical_tensor_t &>(outputs[i]),
                    subgraph_->outs_[i]));
        }

        // generate a hash key for exec_args_mgr
        resource_ctor_ = [this]() {
            return this->memory_planner_.get_exec_args_set().clone();
        };

        return impl::status::success;
    }

    impl::status_t execute_impl(const dnnl_partition_impl_t *part,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(part);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        // each thread's own local resource
        thread_local_cache_t<execution_args_set_t> res_cache;
        execution_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
            mem_idx.first.set_data_handle(
                    inputs[mem_idx.second].get_data_handle());
        }
        for (const auto &mem_idx : res->get_mems_use_external_outputs()) {
            mem_idx.first.set_data_handle(
                    outputs[mem_idx.second].get_data_handle());
        }

        temporary_scratchpad_t scratchpad(
                memory_planner_.total_internal_temporary_size(), p_engine_,
                *g_alloc_);
        assertm(scratchpad.size()
                        >= memory_planner_.total_internal_temporary_size(),
                "no enough scratchpad memory");
        grantor_t var_grantor = memory_planner_.internal_temporary_grantor(
                scratchpad.get_buffer());

        registry_t::key_t key = 0;
        for (auto &mem_offkey : res->get_mems_use_internal_temporary()) {
            mem_offkey.first.set_data_handle(
                    var_grantor.get(mem_offkey.second));
        }

        for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
            subgraph_->execs_[i]->execute(p_stream, res->get_exec_args()[i]);
        }

        return impl::status::success;
    }
};

using float_pooling_fwd = pooling_fwd_t</* quantized */ false>;
using quantized_pooling = pooling_fwd_t</* quantized */ true>;

struct pooling_backward : public dnnl::pooling_v2_backward,
                          public kernel_base_t {
    using super = dnnl::pooling_v2_backward;

private:
    dnnl::pooling_v2_forward::primitive_desc forward_hints_;
    primitive_desc pd_;
    op_kind_t kind_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    void compute(const dnnl_tensor_t &diff_dst, const dnnl_tensor_t &src,
            dnnl_tensor_t &diff_src, const dnnl::engine &p_engine,
            impl::allocator_t *alc, const dnnl::stream &p_stream,
            dnnl_tensor_t indices = dnnl_tensor_t {}) {
        // generate indices dnnl_tensor_t from src when it's needed
        // but can't get from function parameters
        if (kind_ == impl::op_kind::MaxPoolBackprop && indices.is_empty()) {
            auto expected_src = src.reorder_if_differ_in(
                    p_stream, forward_hints_.src_desc());
            auto expected_dst
                    = dnnl_tensor_t {forward_hints_.dst_desc(), p_engine, alc};
            indices = dnnl_tensor_t {
                    forward_hints_.workspace_desc(), p_engine, alc};
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
        using desc = dnnl_tensor_t::desc_t;
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

        const dnnl_tensor_t::desc_t optimal_diff_src_desc {pd_.diff_src_desc()};
        fill_layout_info(diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        dnnl_tensor_t src {inputs.at(pool_bwd::kSrc), p_engine_, alc};
        dnnl_tensor_t diff_dst {};
        dnnl_tensor_t indices {};
        if (op->get_kind() == impl::op_kind::MaxPoolBackprop
                && inputs.size() > pool_bwd_with_indices::kDiff_dst) {
            diff_dst = dnnl_tensor_t {
                    inputs.at(pool_bwd_with_indices::kDiff_dst), p_engine_,
                    alc};
            indices = dnnl_tensor_t {
                    inputs.at(pool_bwd_with_indices::kIndices), p_engine_, alc};
        } else {
            diff_dst = dnnl_tensor_t {
                    inputs.at(pool_bwd::kDiff_dst), p_engine_, alc};
        }

        dnnl_tensor_t diff_src {
                outputs.at(pool_bwd::kDiff_src), p_engine_, alc};
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

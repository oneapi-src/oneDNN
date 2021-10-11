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
struct pooling_fwd : public kernel_base {
private:
    dnnl::engine p_engine_;
    impl::allocator_t *g_alloc_;

    primitive_attr_mgr prm_attr_mgr_;
    memory_planner_t memory_planner_;
    executable_mgr exec_mgr_;

    std::vector<op_executable *> execs_;
    std::vector<exec_args> exec_args_;

    std::vector<std::shared_ptr<impl::op_t>> opt_subgraph_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

    pd_cache_t pd_cache_;

public:
    virtual ~pooling_fwd() {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
    }

    virtual impl::status_t compile_impl(const dnnl_partition_impl_t *part,
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

        std::vector<std::shared_ptr<op_t>> subgraph = part->get_ops();

        set_all_layout_to_any(subgraph);

        // have to set the given inputs and outputs before infer shape and
        // compile
        set_given_inputs_outputs(subgraph, inputs, outputs);

        // for those primitive ops like pooling, it requires the scales and zps
        // between input tensor and output tensor are the same. So here, we
        // don't need to split Dequant and Quant ops firstly, it should be okay
        // to directly fuse Dequant and Quant into int8 pool op.
        if (quantized) { fuse_to_int8_pool(subgraph); }

        BACKEND_DNNL_CHECK(fuse_post_ops(subgraph, prm_attr_mgr_));

        insert_permute(subgraph);
        insert_reorder(subgraph);

        subgraph_visualizer_t vis(part->id());
        vis.run(subgraph, "after_lower_down", false);

        impl::graph_t agraph(subgraph);
        BACKEND_DNNL_CHECK(agraph.infer_shape());
        BACKEND_DNNL_CHECK(infer_type(agraph));

        vis.run(subgraph, "after_infer_shape_infer_type", true);

        BACKEND_DNNL_CHECK(layout_propagation(
                subgraph, p_engine_, prm_attr_mgr_, pd_cache_));

        vis.run(subgraph, "after_layout_propagation", true);

        // fill layout information for outputs logical tensors
        for (size_t i = 0; i < outputs.size(); i++) {
            for (auto out_val : impl::graph_t(subgraph).get_output_values()) {
                auto compiled_lt = out_val->get_logical_tensor();
                if (compiled_lt.id == outputs[i].id) {
                    auto lt = const_cast<impl::logical_tensor_t *>(&outputs[i]);
                    auto md = make_dnnl_memory_desc(compiled_lt);
                    lt->ndims = compiled_lt.ndims;
                    impl::utils::array_copy(
                            lt->dims, compiled_lt.dims, DNNL_GRAPH_MAX_NDIMS);
                    impl::utils::array_copy(lt->layout.strides,
                            compiled_lt.layout.strides, DNNL_GRAPH_MAX_NDIMS);
                    fill_layout_info(lt, md);
                }
            }
        }

        // bind the memory for each op
        BACKEND_DNNL_CHECK(memory_planner_.run(
                subgraph, inputs, outputs, p_engine_, prm_attr_mgr_));

        vis.run(subgraph, "after_memory_planning", true, true,
                [this](const value_t *val) {
                    return this->memory_planner_.get_memory_info(val);
                });

        BACKEND_DNNL_CHECK(compile_ops(
                subgraph, p_engine_, prm_attr_mgr_, exec_mgr_, pd_cache_));

        // topologically sort the executables
        impl::topo_order_visit(impl::graph_t(subgraph).get_output_ops(),
                [this](impl::op_t *op) {
                    auto exec_key = op->get_attr<int64_t>("executable_key");
                    auto &exec = exec_mgr_.get_executable(exec_key);
                    execs_.emplace_back(exec.get());

                    return impl::status::success;
                });

        opt_subgraph_ = subgraph;

        // generate a hash key for exec_args_mgr
        resource_ctor_ = [this]() {
            return this->memory_planner_.get_exec_args_set().clone();
        };

        return impl::status::success;
    }

    virtual impl::status_t execute_impl(const dnnl_partition_impl_t *part,
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

        for (size_t i = 0; i < execs_.size(); i++) {
            execs_[i]->execute(p_stream, res->get_exec_args()[i]);
        }

        return impl::status::success;
    }
};

using float_pooling_fwd = pooling_fwd</* quantized */ false>;
using quantized_pooling = pooling_fwd</* quantized */ true>;

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

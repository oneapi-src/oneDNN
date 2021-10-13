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

#ifndef BACKEND_DNNL_KERNELS_CONV_HPP
#define BACKEND_DNNL_KERNELS_CONV_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "interface/backend.hpp"
#include "interface/graph.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/constant_cache.hpp"
#include "backend/dnnl/dnnl_partition_impl.hpp"
#include "backend/dnnl/passes/compile_ops.hpp"
#include "backend/dnnl/passes/constant_propagation.hpp"
#include "backend/dnnl/passes/infer_type.hpp"
#include "backend/dnnl/passes/insert_ops.hpp"
#include "backend/dnnl/passes/layout_propagation.hpp"
#include "backend/dnnl/passes/lower_down.hpp"
#include "backend/dnnl/passes/memory_planning.hpp"
#include "backend/dnnl/passes/op_executable.hpp"
#include "backend/dnnl/scratchpad.hpp"
#include "backend/dnnl/thread_local_cache.hpp"
#include "backend/dnnl/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

struct conv_base : public kernel_base {
protected:
    dnnl::engine p_engine_;
    impl::allocator_t *g_alloc_;

    std::vector<std::shared_ptr<impl::op_t>> opt_subgraph_;

    primitive_attr_mgr prm_attr_mgr_;
    executable_mgr exec_mgr_;
    memory_planner_t memory_planner_;

    std::vector<op_executable *> execs_;

    std::function<std::shared_ptr<execution_args_set_t>()> resource_ctor_;

    // FIXME(qun) improve the cache key
    constant_cache_t::key_t constant_key_
            = reinterpret_cast<constant_cache_t::key_t>(this);

    bool enable_constant_cache_ = is_constant_cache_enabled();

    std::vector<bool> is_constant_;

    pd_cache_t pd_cache_;

public:
    virtual ~conv_base() {
        thread_local_cache_t<execution_args_set_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));

        if (enable_constant_cache_) {
            constant_cache_t constant_cache;
            constant_cache.remove_if_exist(constant_key_);
        }
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

        // update the data of partition in/outputs args
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

        if (enable_constant_cache_) {
            std::promise<constant_cache_t::cached_t> c_promise;
            constant_cache_t global_constant_cache;
            constant_cache_t::value_t cached_value
                    = global_constant_cache.get_or_add(
                            constant_key_, c_promise.get_future());
            bool is_from_cache = cached_value.valid();
            if (is_from_cache) {
                constant_cache_t::cached_t c_buffer = cached_value.get();
                grantor_t c_grantor
                        = memory_planner_.internal_persistent_grantor(
                                c_buffer->data<char>());
                registry_t::key_t key = 0;
                for (auto &mem_offkey :
                        res->get_mems_use_internal_persistent()) {
                    mem_offkey.first.set_data_handle(
                            c_grantor.get(mem_offkey.second));
                }
            } else {
                constant_cache_t::cached_t c_buffer
                        = std::make_shared<constant_buffer_t>(
                                memory_planner_
                                        .total_internal_persistent_size(),
                                p_engine_, g_alloc_);
                grantor_t c_grantor
                        = memory_planner_.internal_persistent_grantor(
                                c_buffer->data<char>());
                registry_t::key_t key = 0;
                for (auto &mem_offkey :
                        res->get_mems_use_internal_persistent()) {
                    mem_offkey.first.set_data_handle(
                            c_grantor.get(mem_offkey.second));
                }

                for (size_t i = 0; i < execs_.size(); i++) {
                    if (!is_constant_[i]) continue;
                    execs_[i]->execute(p_stream, res->get_exec_args()[i]);
                }

                c_promise.set_value(c_buffer);
            }
        }

        for (size_t i = 0; i < execs_.size(); i++) {
            if (is_constant_[i]) continue;
            execs_[i]->execute(p_stream, res->get_exec_args()[i]);
        }

        return impl::status::success;
    }
};

template <bool quantized>
struct conv_fwd : public conv_base {
public:
    virtual impl::status_t compile_impl(const dnnl_partition_impl_t *part,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = g_engine->get_allocator();

        // get subgraph from the deep copied partition
        std::vector<std::shared_ptr<impl::op_t>> subgraph = part->get_ops();

        set_all_layout_to_any(subgraph);

        // have to set the given inputs and outputs fusion because some fusions
        // can't be performed if shape unknown
        set_given_inputs_outputs(subgraph, inputs, outputs);

        fuse_bias_add(subgraph);

        if (!quantized) { insert_bn_folding(subgraph); }

        check_with_bias(subgraph);
        fuse_mul_sigmoid_to_swish(subgraph);

        // Because we use binary post-ops for broadcast add and sum post-ops for
        // non-broadcast add. So we have to know concret shape before fuse
        // post-ops
        BACKEND_DNNL_CHECK(impl::graph_t(subgraph).infer_shape());

        if (quantized) {
            split_quant_dequant(subgraph);
            fuse_to_int8_conv_or_deconv(subgraph);
            folding_mul_scales(subgraph);
            fuse_output_scales(subgraph, prm_attr_mgr_);
        }

        BACKEND_DNNL_CHECK(fuse_post_ops(subgraph, prm_attr_mgr_));

        if (quantized) {
            fuse_zero_points(subgraph, prm_attr_mgr_);
            // fuse neighboring mul_scales and zdd_zps op to quantize/dequantize
            fuse_mul_scales_add_zps(subgraph);
        }

        insert_permute(subgraph);
        insert_to_group_for_conv_or_deconv(subgraph);
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

        // fill layout information for inputs logical tensors
        // FIXME(qun) for not breaking conversion example and some C API.
        // We should not need to set layout id for inputs in the further
        for (size_t i = 0; i < inputs.size(); i++) {
            for (auto in_val : impl::graph_t(subgraph).get_input_values()) {
                auto compiled_lt = in_val->get_logical_tensor();
                if (compiled_lt.id == inputs[i].id) {
                    auto lt = const_cast<impl::logical_tensor_t *>(&inputs[i]);
                    auto md = make_dnnl_memory_desc(compiled_lt);
                    fill_layout_info(lt, md);
                }
            }
        }

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

        // constant propagation
        if (enable_constant_cache_) { constant_propagation(subgraph); }

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

                    is_constant_.push_back(op->has_attr("is_constant")
                            && op->get_attr<bool>("is_constant"));
                    return impl::status::success;
                });

        opt_subgraph_ = subgraph;

        resource_ctor_ = [this]() {
            return this->memory_planner_.get_exec_args_set().clone();
        };

        return impl::status::success;
    }

    virtual impl::status_t prepare_inplace_pairs_impl(
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        UNUSED(g_engine);

        op_t *conv_op = nullptr;
        for (auto &op : opt_subgraph_) {
            if (op->get_kind() == impl::op_kind::Convolution
                    || op->get_kind() == op_kind::dnnl_convolution) {
                conv_op = op.get();
                break;
            }
        }

        bool with_sum = conv_op->has_attr("with_sum")
                ? conv_op->get_attr<bool>("with_sum")
                : false;

        if (with_sum) {
            // post_src should always be the last one input of conv op
            auto val = conv_op->get_input_value(conv_op->num_inputs() - 1);
            if (val->has_producer()
                    && val->get_producer().get_kind() == op_kind::permute) {
                val = val->get_producer().get_input_value(0);
            }
            size_t post_src_id = val->get_logical_tensor().id;

            // find the given post src index
            size_t idx = 0;
            for (size_t i = 0; i < inputs.size(); i++) {
                if (inputs[i].id == post_src_id) {
                    idx = i;
                    break;
                }
            }

            const logical_tensor_wrapper post_src_lt(inputs[idx]);
            const logical_tensor_wrapper dst_lt(outputs[0]);
            // TODO(qun) we didn't report iplace pair if two lts have different
            // layout type because of frontend users didn't process this
            // situation at this moment. In the future, we need to fix this for
            // more inplace opportunities.
            if (((post_src_lt.is_opaque() && dst_lt.is_opaque())
                        || (post_src_lt.is_strided() && dst_lt.is_strided()))
                    && post_src_lt.is_similar(dst_lt))
                inplace_pairs_.push_back({post_src_id, outputs[0].id});
        }
        return impl::status::success;
    }
};

struct conv_bwd_data : public conv_base {
public:
    virtual impl::status_t compile_impl(const dnnl_partition_impl_t *part,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        p_engine_ = make_dnnl_engine(*g_engine);
        g_alloc_ = g_engine->get_allocator();

        // get subgraph from the deep copied partition
        std::vector<std::shared_ptr<impl::op_t>> subgraph = part->get_ops();

        set_all_layout_to_any(subgraph);

        // have to set the given inputs and outputs fusion because some fusions
        // can't be performed if shape unknown
        set_given_inputs_outputs(subgraph, inputs, outputs);

        conv_bwd_data_canonicalization(subgraph);

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

                    is_constant_.push_back(op->has_attr("is_constant")
                            && op->get_attr<bool>("is_constant"));

                    return impl::status::success;
                });

        opt_subgraph_ = subgraph;

        resource_ctor_ = [this]() {
            return this->memory_planner_.get_exec_args_set().clone();
        };

        return impl::status::success;
    }
};

namespace conv_bwd_filter {
enum conv_bwd_inputs { kSrc, kDiffdst };
enum conv_bwd_outputs { kDiffweight, kDiffbias };
} // namespace conv_bwd_filter

struct convolution_backward_weights : public dnnl::convolution_backward_weights,
                                      public kernel_base {
    using super = dnnl::convolution_backward_weights;
    using dims_t = std::vector<dnnl::graph::impl::dim_t>;

private:
    dims strides_;
    dims dilates_;
    dims pads_begin_;
    dims pads_end_;
    int64_t groups_;

    bool with_diff_bias_;

    primitive_desc pd_;

    dnnl::engine p_engine_;
    dnnl::stream p_stream_;

public:
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        const op_kind_t conv_kind = op->get_kind();

        // update shape
        std::string data_format = op->get_attr<std::string>("data_format");
        std::string filter_format = op->get_attr<std::string>("filter_format");

        impl::logical_tensor_t src_lt = inputs.at(conv_bwd_filter::kSrc);
        impl::logical_tensor_t diff_weight_lt
                = outputs.at(conv_bwd_filter::kDiffweight);
        impl::logical_tensor_t diff_dst_lt
                = inputs.at(conv_bwd_filter::kDiffdst);

        // "NXC"
        if (data_format == "NXC") {
            diff_dst_lt = logical_tensor_wrapper(
                    &inputs.at(conv_bwd_filter::kDiffdst))
                                  .reorder_data_dims_strides();
            src_lt = impl::logical_tensor_wrapper(
                    &inputs.at(conv_bwd_filter::kSrc))
                             .reorder_data_dims_strides();
        }
        // "XIO"
        if (filter_format == "XIO") {
            diff_weight_lt = impl::logical_tensor_wrapper(
                    &outputs.at(conv_bwd_filter::kDiffweight))
                                     .reorder_weight_dims_strides();
        }

        const desc src_desc {src_lt};
        const desc diff_dst_desc {diff_dst_lt};
        const desc diff_weights_desc {diff_weight_lt};

        with_diff_bias_ = op_kind::conv_bwd_f_biasadd_bwd == conv_kind;

        impl::logical_tensor_t *diff_bias_lt = nullptr;
        if (with_diff_bias_) {
            diff_bias_lt = const_cast<impl::logical_tensor_t *>(
                    &outputs.at(conv_bwd_filter::kDiffbias));
        }

        // cache operator attributes
        strides_ = op->get_attr<dims>("strides");
        dilates_ = op->get_attr<dims>("dilations");

        // make dilates compatible with oneDNN
        dilates_ = get_compatible_dilates(dilates_);

        pads_begin_ = op->get_attr<dims>("pads_begin");
        pads_end_ = op->get_attr<dims>("pads_end");
        groups_ = op->get_attr<int64_t>("groups");

        p_engine_ = make_dnnl_engine(*g_engine);

        const auto src_desc_any = src_desc.to_format_any();
        const auto diff_dst_desc_any = diff_dst_desc.to_format_any();
        const auto diff_weights_desc_any = groups_ <= 1
                ? diff_weights_desc.to_format_any()
                : diff_weights_desc.to_grouped(groups_).to_format_any();

        // for forward hint, weights_desc should have same data_type
        // with other input desc, expect for bias_desc
        auto weights_desc = diff_weights_desc;
        auto diff_weight_type_in = diff_weights_desc.get_data_type();
        auto diff_dst_type = diff_dst_desc.get_data_type();
        if (diff_weight_type_in != diff_dst_type) {
            weights_desc = weights_desc.to_type(diff_dst_type);
        }

        const desc diff_bias_desc = desc(
                {diff_dst_desc.get_dim(1)}, diff_weight_type_in, tag::any);

        if (with_diff_bias_) {
            auto forward_hints = get_fwd_primitive_desc<
                    /*with_diff_bias=*/true>(src_desc_any, weights_desc,
                    diff_bias_desc, diff_dst_desc_any, strides_, dilates_,
                    pads_begin_, pads_end_, p_engine_, attr_t(),
                    algorithm::convolution_direct, prop_kind::forward_training);

            pd_ = primitive_desc({algorithm::convolution_direct, src_desc_any,
                                         diff_weights_desc_any, diff_bias_desc,
                                         diff_dst_desc_any, strides_, dilates_,
                                         pads_begin_, pads_end_},
                    p_engine_, forward_hints);
        } else {
            auto forward_hints = get_fwd_primitive_desc<
                    /*with_diff_bias=*/false>(src_desc_any, weights_desc,
                    diff_bias_desc, diff_dst_desc_any, strides_, dilates_,
                    pads_begin_, pads_end_, p_engine_, attr_t(),
                    algorithm::convolution_direct, prop_kind::forward_training);

            pd_ = primitive_desc(
                    {algorithm::convolution_direct, src_desc_any,
                            diff_weights_desc_any, diff_dst_desc_any, strides_,
                            dilates_, pads_begin_, pads_end_},
                    p_engine_, forward_hints);
        }

        const desc optimal_diff_weights_desc {pd_.diff_weights_desc()};
        auto ori_diff_weights_lt = const_cast<impl::logical_tensor_t *>(
                &outputs.at(conv_bwd_filter::kDiffweight));
        fill_layout_info(ori_diff_weights_lt, optimal_diff_weights_desc);
        if (with_diff_bias_) {
            const desc optimal_diff_bias_desc {pd_.diff_bias_desc()};
            fill_layout_info(diff_bias_lt, optimal_diff_bias_desc);
        }
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::op_t *op,
            const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        p_stream_ = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        auto &src_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_filter::kSrc).get_logical_tensor());
        auto &diff_dst_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(conv_bwd_filter::kDiffdst).get_logical_tensor());
        auto &diff_weights_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(conv_bwd_filter::kDiffweight).get_logical_tensor());

        auto diff_bias = with_diff_bias_ ? const_cast<impl::tensor_t &>(
                                 outputs.at(conv_bwd_filter::kDiffbias))
                                         : impl::tensor_t {};

        // "NXC"
        if (op->get_attr<std::string>("data_format") == "NXC") {
            diff_dst_lt = impl::logical_tensor_wrapper(diff_dst_lt)
                                  .reorder_data_dims_strides();
            src_lt = impl::logical_tensor_wrapper(src_lt)
                             .reorder_data_dims_strides();
        }
        // "XIO"
        if (op->get_attr<std::string>("filter_format") == "XIO") {
            diff_weights_lt = impl::logical_tensor_wrapper(diff_weights_lt)
                                      .reorder_data_dims_strides();
        }
        const tensor src {src_lt, p_engine_, alc,
                inputs.at(conv_bwd_filter::kSrc).get_data_handle()};
        const tensor diff_dst {diff_dst_lt, p_engine_, alc,
                inputs.at(conv_bwd_filter::kDiffdst).get_data_handle()};
        tensor diff_weights {diff_weights_lt, p_engine_, alc,
                outputs.at(conv_bwd_filter::kDiffweight).get_data_handle()};
        tensor diff_b = with_diff_bias_ ? tensor {diff_bias, p_engine_, alc}
                                        : tensor {};

        auto diff_weights_dims
                = impl::logical_tensor_wrapper(diff_weights_lt).vdims();
        conv_backward_weights_impl(src, diff_dst, diff_weights, diff_b,
                diff_weights_dims, alc, p_stream_);
        return impl::status::success;
    }

private:
    void conv_backward_weights_impl(const tensor &src, const tensor &diff_dst,
            tensor &diff_weights, tensor &diff_bias,
            const std::vector<dim_t> &diff_weights_dims, impl::allocator_t *alc,
            const dnnl::stream &p_stream) {
        if (with_diff_bias_) {
            compute_impl</*with_diff_bias=*/true>(src, diff_dst,
                    diff_weights_dims, diff_weights, diff_bias, p_engine_, alc,
                    p_stream);
        } else {
            compute_impl</*with_diff_bias=*/false>(src, diff_dst,
                    diff_weights_dims, diff_weights, diff_bias, p_engine_, alc,
                    p_stream);
        }
    }

    template <bool with_diff_bias>
    void compute_impl(const tensor &src, const tensor &diff_dst,
            const dims &diff_weights_dims, tensor &diff_weights,
            tensor &diff_bias, const dnnl::engine &p_engine,
            impl::allocator_t *alc, const dnnl::stream &p_stream) {
        UNUSED(diff_weights_dims);
        auto expected_diff_dst
                = diff_dst.reorder_if_differ_in(p_stream, pd_.diff_dst_desc());
        auto expected_src = src.reorder_if_differ_in(p_stream, pd_.src_desc());
        // embed group info into diff_weights_desc
        auto expected_diff_weights_desc
                = tensor::desc(pd_.diff_weights_desc(), groups_);

        tensor expected_diff_weights = diff_weights;
        if (pd_.diff_weights_desc() != diff_weights.get_desc()) {
            expected_diff_weights
                    = tensor {expected_diff_weights_desc, p_engine, alc};
        }
        tensor expected_diff_bias = diff_bias;

        if (with_diff_bias_) {
            if (pd_.diff_bias_desc() != diff_bias.get_desc()) {
                expected_diff_bias
                        = tensor {pd_.diff_bias_desc(), p_engine, alc};
            }
            super(pd_).execute(p_stream,
                    {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                            {DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
                            {DNNL_ARG_DIFF_BIAS, expected_diff_bias}});
        } else {
            super(pd_).execute(p_stream,
                    {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                            {DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights}});
        }

        if (expected_diff_weights != diff_weights) {
            dnnl::reorder(expected_diff_weights, diff_weights)
                    .execute(p_stream, expected_diff_weights, diff_weights);
        }

        if (with_diff_bias_ && expected_diff_bias != diff_bias) {
            dnnl::reorder(expected_diff_bias, diff_bias)
                    .execute(p_stream, expected_diff_bias, diff_bias);
        }
    }

    template <bool with_bias>
    static dnnl::convolution_forward::primitive_desc get_fwd_primitive_desc(
            const tensor::desc &src_desc, const tensor::desc &weights_desc,
            const tensor::desc &bias_desc, const tensor::desc &dst_desc,
            const dims &strides, const dims &dilates, const dims &pads_begin,
            const dims &pads_end, const dnnl::engine &p_engine,
            const attr_t &attr = attr_t(),
            algorithm aalgorithm = algorithm::convolution_direct,
            prop_kind aprop_kind = prop_kind::forward) {
        auto src_desc_any = src_desc.to_format_any();
        auto weights_desc_any = weights_desc.to_format_any();
        auto bias_desc_any
                = with_bias ? bias_desc.to_format_any() : tensor::desc();
        auto dst_desc_any = dst_desc.to_format_any();

        if (with_bias) {
            return dnnl::convolution_forward::primitive_desc(
                    {aprop_kind, aalgorithm, src_desc_any, weights_desc_any,
                            bias_desc_any, dst_desc_any, strides, dilates,
                            pads_begin, pads_end},
                    attr, p_engine);
        } else {
            return dnnl::convolution_forward::primitive_desc(
                    {aprop_kind, aalgorithm, src_desc_any, weights_desc_any,
                            dst_desc_any, strides, dilates, pads_begin,
                            pads_end},
                    attr, p_engine);
        }
    }
};

using float_conv_fwd = conv_fwd</* quantized */ false>;
using quantized_conv = conv_fwd</* quantized */ true>;

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

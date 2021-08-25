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

#ifndef BACKEND_DNNL_OPERATORS_CONVTRANSPOSE_HPP
#define BACKEND_DNNL_OPERATORS_CONVTRANSPOSE_HPP

#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "utils/utils.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/constant_cache.hpp"
#include "backend/dnnl/f32_kernel_resource.hpp"
#include "backend/dnnl/legacy.hpp"
#include "backend/dnnl/scratchpad.hpp"
#include "backend/dnnl/thread_local_cache.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace convtranspose {
enum convtranspose_inputs { kSrc, kWeight, kBias };
enum convtranspose_outputs { kDst };
enum mem_keys { kOpt_src, kOpt_weights, kOpt_dst, kOpt_bias, kScratchpad };
} // namespace convtranspose

struct convtranspose_forward : public dnnl::deconvolution_forward,
                               public kernel_base {
    using super = dnnl::deconvolution_forward;
    using dims_t = std::vector<dnnl::graph::impl::dim_t>;

private:
    primitive_desc pd_;
    dnnl::deconvolution_forward prim_;

    dims strides_;
    dims dilations_;
    dims pads_begin_;
    dims pads_end_;
    int64_t groups_;
    std::string data_format_, filter_format_;

    attr_t attr_;
    bool with_bias_ {false};

    dnnl::engine p_engine_;
    impl::allocator_t *alc_ {nullptr};

    dnnl::reorder::primitive_desc src_reorder_pd_;
    dnnl::reorder::primitive_desc weight_reorder_pd_;
    dnnl::reorder::primitive_desc dst_reorder_pd_;
    dnnl::reorder::primitive_desc bias_reorder_pd_;

    registry_t registry_;
    registry_t c_registry_;
    std::function<std::shared_ptr<f32_kernel_resource_t>()> resource_ctor_;

    f32_kernel_resource_t::desc_t res_desc_;

    bool enable_constant_cache_ = utils::is_enable_constant_cache();

public:
    virtual ~convtranspose_forward() {
        thread_local_cache_t<f32_kernel_resource_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));

        if (enable_constant_cache_) {
            constant_cache_t::key_t cache_key
                    = reinterpret_cast<constant_cache_t::key_t>(this);
            constant_cache_t constant_cache;
            constant_cache.remove_if_exist(cache_key);
        }
    }
    impl::status_t compile_impl(const impl::op_t *op,
            const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        // prepare the operator attributes
        strides_ = op->get_attr<dims>("strides");
        dilations_ = op->get_attr<dims>("dilations");
        pads_begin_ = op->get_attr<dims>("pads_begin");
        pads_end_ = op->get_attr<dims>("pads_end");
        groups_ = op->get_attr<int64_t>("groups");
        data_format_ = op->get_attr<std::string>("data_format");
        filter_format_ = op->get_attr<std::string>("filter_format");
        with_bias_ = inputs.size() == convtranspose::kBias + 1;

        auto src = make_dnnl_memory_desc(inputs.at(convtranspose::kSrc));
        auto weight = make_dnnl_memory_desc(inputs.at(convtranspose::kWeight));
        auto dst = make_dnnl_memory_desc(outputs.at(convtranspose::kDst));

        if (data_format_ == "NXC") {
            src = permute_NXC2NCX(src);
            dst = permute_NXC2NCX(dst);
        }
        if (filter_format_ == "XIO") { weight = permute_XIO2OIX(weight); }

        res_desc_.cvt_src_ = src;
        res_desc_.cvt_wei_ = weight;
        res_desc_.cvt_dst_ = dst;

        memory::desc bias = with_bias_
                ? make_dnnl_memory_desc(inputs.at(convtranspose::kBias))
                : memory::desc {};

        res_desc_.cvt_bias_ = bias;

        p_engine_ = make_dnnl_engine(*g_engine);
        alc_ = g_engine->get_allocator();

        // make weights and dilations compatible with oneDNN
        memory::desc group_weight;
        if (groups_ <= 1)
            group_weight = weight;
        else {
            auto permuted_weight = transpose(weight, 0, 1);
            auto permuted_group_weight = to_grouped(permuted_weight, groups_);
            group_weight = transpose(permuted_group_weight, 1, 2);
        }
        auto com_dilations = get_compatible_dilates(dilations_);

        memory::desc src_desc = to_format_any(src);
        memory::desc weight_desc = to_format_any(group_weight);
        memory::desc dst_desc = to_format_any(dst);
        memory::desc bias_desc
                = with_bias_ ? to_format_any(bias) : memory::desc();

        attr_t op_attr = attr_;
        op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        pd_ = with_bias_
                ? primitive_desc(
                        {prop_kind::forward, algorithm::deconvolution_direct,
                                src_desc, weight_desc, bias_desc, dst_desc,
                                strides_, com_dilations, pads_begin_,
                                pads_end_},
                        attr_, p_engine_)
                : primitive_desc(
                        {prop_kind::forward, algorithm::deconvolution_direct,
                                src_desc, weight_desc, dst_desc, strides_,
                                com_dilations, pads_begin_, pads_end_},
                        attr_, p_engine_);
        prim_ = super(pd_);

        res_desc_.opt_src_ = pd_.src_desc();
        res_desc_.opt_wei_ = pd_.weights_desc();
        res_desc_.opt_dst_ = pd_.dst_desc();
        if (with_bias_) { res_desc_.opt_bias_ = pd_.bias_desc(); }
        res_desc_.scratchpad_ = pd_.scratchpad_desc();

        if (groups_ > 1) res_desc_.cvt_wei_ = group_weight;

        if (impl::logical_tensor_wrapper(inputs.at(convtranspose::kSrc))
                        .is_any())
            res_desc_.cvt_src_ = pd_.src_desc();
        if (impl::logical_tensor_wrapper(inputs.at(convtranspose::kWeight))
                        .is_any())
            res_desc_.cvt_wei_ = pd_.weights_desc();
        if (impl::logical_tensor_wrapper(outputs.at(convtranspose::kDst))
                        .is_any()) {
            res_desc_.cvt_dst_ = pd_.dst_desc();
        }

        if ((with_bias_)
                && impl::logical_tensor_wrapper(inputs.at(convtranspose::kBias))
                           .is_any()) {
            res_desc_.cvt_bias_ = pd_.bias_desc();
        }

        impl::logical_tensor_t *ori_dst_lt
                = const_cast<impl::logical_tensor_t *>(
                        &outputs.at(convtranspose::kDst));
        if (data_format_ == "NXC") {
            memory::desc tmp = permute_NCX2NXC(pd_.dst_desc());
            fill_layout_info(ori_dst_lt, tmp);
        } else {
            fill_layout_info(ori_dst_lt, pd_.dst_desc());
        }

        registrar_t registrar = registry_.registrar();
        registrar_t constant_registrar = c_registry_.registrar();
        registrar_t &wei_bias_registrar
                = enable_constant_cache_ ? constant_registrar : registrar;

        if (res_desc_.cvt_wei_ != res_desc_.opt_wei_) {
            wei_bias_registrar.book(
                    convtranspose::kOpt_weights, res_desc_.opt_wei_.get_size());
            weight_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                    res_desc_.cvt_wei_, p_engine_, res_desc_.opt_wei_);
        }

        if (with_bias_ && (res_desc_.cvt_bias_ != res_desc_.opt_bias_)) {
            wei_bias_registrar.book(
                    convtranspose::kOpt_bias, res_desc_.opt_bias_.get_size());
            bias_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                    res_desc_.cvt_bias_, p_engine_, res_desc_.opt_bias_);
        }

        if (res_desc_.opt_src_ != res_desc_.cvt_src_) {
            registrar.book(
                    convtranspose::kOpt_src, res_desc_.opt_src_.get_size());
            src_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                    res_desc_.cvt_src_, p_engine_, res_desc_.opt_src_);
        }
        if (res_desc_.opt_dst_ != res_desc_.cvt_dst_) {
            registrar.book(
                    convtranspose::kOpt_dst, res_desc_.opt_dst_.get_size());
            dst_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                    res_desc_.opt_dst_, p_engine_, res_desc_.cvt_dst_);
        }
        registrar.book(
                convtranspose::kScratchpad, res_desc_.scratchpad_.get_size());

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
        UNUSED(op);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        // each thread's own local resource
        thread_local_cache_t<f32_kernel_resource_t> res_cache;
        f32_kernel_resource_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        temporary_scratchpad_t scratchpad(registry_.size(), p_engine_, *alc_);
        grantor_t grantor = registry_.grantor(scratchpad.get_buffer());

        ///////////// weight/bias process start ///////////////////////////
        // lookup constant buffer
        constant_cache_t::key_t cache_key
                = reinterpret_cast<constant_cache_t::key_t>(this);
        constant_cache_t global_constant_cache;

        std::promise<constant_cache_t::cached_t> c_promise;

        bool is_from_cache;
        constant_cache_t::value_t cached_value;
        constant_cache_t::cached_t c_buffer;
        if (enable_constant_cache_) {
            cached_value = global_constant_cache.get_or_add(
                    cache_key, c_promise.get_future());
            is_from_cache = cached_value.valid();
            if (is_from_cache) {
                // get from cache or wait for other thread
                c_buffer = cached_value.get();
            } else {
                c_buffer = std::make_shared<constant_buffer_t>(
                        c_registry_.size(), p_engine_, alc_);
            }
        }

        grantor_t c_grantor = c_registry_.grantor(
                enable_constant_cache_ ? c_buffer->data<char>() : nullptr);
        grantor_t &wei_bias_grantor
                = enable_constant_cache_ ? c_grantor : grantor;

        res->cvt_wei_.set_data_handle(
                inputs.at(convtranspose::kWeight).get_data_handle());
        if (with_bias_) {
            res->cvt_bias_.set_data_handle(
                    inputs.at(convtranspose::kBias).get_data_handle());
        }

        if (res_desc_.cvt_wei_ != res_desc_.opt_wei_) {
            res->opt_wei_.set_data_handle(
                    wei_bias_grantor.get(convtranspose::kOpt_weights));
            if (!(enable_constant_cache_ && is_from_cache)) {
                dnnl::reorder(weight_reorder_pd_)
                        .execute(p_stream, res->cvt_wei_, res->opt_wei_);
            }
        } else {
            res->opt_wei_.set_data_handle(res->cvt_wei_.get_data_handle());
        }

        if (with_bias_) {
            if (res_desc_.cvt_bias_ != res_desc_.opt_bias_) {
                res->opt_bias_.set_data_handle(
                        wei_bias_grantor.get(convtranspose::kOpt_bias));
                if (!enable_constant_cache_ || !is_from_cache) {
                    dnnl::reorder(bias_reorder_pd_)
                            .execute(p_stream, res->cvt_bias_, res->opt_bias_);
                }
            } else {
                res->opt_bias_.set_data_handle(
                        res->cvt_bias_.get_data_handle());
            }
        }

        if (enable_constant_cache_ && !is_from_cache) {
            c_promise.set_value(c_buffer);
        }
        ///////////// weight/bias process end ///////////////////////////

        res->cvt_src_.set_data_handle(
                inputs.at(convtranspose::kSrc).get_data_handle());
        res->cvt_dst_.set_data_handle(
                outputs.at(convtranspose::kDst).get_data_handle());

        if (res_desc_.cvt_src_ != res_desc_.opt_src_) {
            res->opt_src_.set_data_handle(grantor.get(convtranspose::kOpt_src));
            dnnl::reorder(src_reorder_pd_)
                    .execute(p_stream, res->cvt_src_, res->opt_src_);
        } else {
            res->opt_src_.set_data_handle(res->cvt_src_.get_data_handle());
        }

        if (res_desc_.cvt_dst_ != res_desc_.opt_dst_) {
            res->opt_dst_.set_data_handle(grantor.get(convtranspose::kOpt_dst));
        } else {
            res->opt_dst_.set_data_handle(res->cvt_dst_.get_data_handle());
        }

        res->scratchpad_.set_data_handle(
                grantor.get(convtranspose::kScratchpad));

        prim_.execute(p_stream, res->exec_args_);

        // reorder optimal output to given output buffer
        if (res_desc_.cvt_dst_ != res_desc_.opt_dst_) {
            dnnl::reorder(dst_reorder_pd_)
                    .execute(p_stream, res->opt_dst_, res->cvt_dst_);
        }
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

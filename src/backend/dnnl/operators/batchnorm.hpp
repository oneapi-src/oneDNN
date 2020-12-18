/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_BACKEND_DNNL_OPERATORS_BATCHNORM_HPP
#define LLGA_BACKEND_DNNL_OPERATORS_BATCHNORM_HPP

#include <memory>
#include <string>
#include <vector>

#include "interface/backend.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/tensor.hpp"
#include "backend/dnnl/utils.hpp"
#include "sum.hpp"

namespace llga {
namespace impl {
namespace dnnl_impl {

namespace batch_normalization {
enum batch_normalization_inputs { kSrc, kScale, kShift, kMean, kVariance };
enum batch_normalization_outputs {
    kDst,
    kRunning_mean,
    kRunning_variance,
    kBatch_mean,
    kBatch_variance
};
} // namespace batch_normalization

namespace batch_normalization_bwd {
enum batch_normalization_bwd_inputs {
    kSrc,
    kDiff_dst,
    kScale,
    kMean,
    kVariance
};
enum batch_normalization_bwd_outputs { kDiff_src, kDiff_scale, kDiff_shift };
} // namespace batch_normalization_bwd

struct batch_normalization_forward_inference
    : public dnnl::batch_normalization_forward,
      public kernel_base {

    using super = dnnl::batch_normalization_forward;
    using dims_t = std::vector<llga::impl::dim_t>;

private:
    primitive_desc pd_;
    float epsilon_;

    tensor scale_shift_;
    tensor expected_mean_;
    tensor expected_var_;

    // FIXME(qun) NOT well designed
    /// \note Currently we don't have enough information from framework to
    /// decide cache or not. Also we think that caching data in a library
    /// is not safe at this moment.
    bool disable_cache_data_ {true};

public:
    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        std::string data_format = anode->get_attr<std::string>("data_format");
        // "NXC" format will be converted to "NCX" format
        impl::logical_tensor_t src_lt = inputs.at(batch_normalization::kSrc);

        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper(
                    &inputs.at(batch_normalization::kSrc))
                             .reorder_data_dims_strides();
        }

        using desc = tensor::desc;
        // prepare the inputs and outputs tensors' descs
        desc src {src_lt};
        const bool use_stats = inputs.size() > batch_normalization::kMean;

        epsilon_ = anode->get_attr<float>("epsilon");

        auto flags = normalization_flag::use_scale_shift;
        if (use_stats) flags |= normalization_flag::use_global_stats;
        if (anode->get_op_kind() == op_kind::bn_relu)
            flags |= normalization_flag::fuse_norm_relu;

        src = src.is_4c_blocked() ? src.to_default_format() : src;

        auto eng = engine_manager::get()->get_engine(*aengine);
        pd_ = primitive_desc(
                {prop_kind::forward_inference, src, epsilon_, flags}, *eng);
        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        impl::logical_tensor_t *ori_dst_lt
                = const_cast<impl::logical_tensor_t *>(
                        &outputs.at(batch_normalization::kDst));
        fill_layout_info(ori_dst_lt, optimal_dst_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        std::string data_format = anode->get_attr<std::string>("data_format");
        auto &src_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(batch_normalization::kSrc).get_logical_tensor());
        auto &dst_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(batch_normalization::kDst).get_logical_tensor());
        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper(src_lt)
                             .reorder_data_dims_strides();
            dst_lt = impl::logical_tensor_wrapper(dst_lt)
                             .reorder_data_dims_strides();
        }

        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));
        tensor x {src_lt,
                inputs.at(batch_normalization::kSrc).get_data_handle(), *eng};
        tensor y {dst_lt,
                outputs.at(batch_normalization::kDst).get_data_handle(), *eng};
        tensor w {inputs.at(batch_normalization::kScale), *eng};
        tensor b {inputs.at(batch_normalization::kShift), *eng};

        auto channels = x.get_dims()[1];
        if (channels != w.get_dims()[0]) {
            throw std::runtime_error("channel mismatch");
        }
        if (inputs.size() > batch_normalization::kMean) {
            tensor m {inputs.at(batch_normalization::kMean), *eng};
            tensor v {inputs.at(batch_normalization::kVariance), *eng};
            compute(x, m, v, w, b, y, epsilon_, *eng);
        } else {
            compute(x, w, b, y, epsilon_, *eng);
        }

        return impl::status::success;
    }

private:
    void compute(const tensor &src, const tensor &scale, const tensor &shift,
            tensor &dst, float epsilon, const engine &aengine) {
        static tensor dummy;
        compute_impl</*use_stats=*/false>(
                src, dummy, dummy, scale, shift, dst, epsilon, aengine);
    }

    void compute(const tensor &src, const tensor &mean, const tensor &variance,
            const tensor &scale, const tensor &shift, tensor &dst,
            float epsilon, const engine &aengine) {
        compute_impl</*use_stats=*/true>(
                src, mean, variance, scale, shift, dst, epsilon, aengine);
    }

    template <bool use_stats>
    void compute_impl(const tensor &src, const tensor &mean,
            const tensor &variance, const tensor &scale, const tensor &shift,
            tensor &dst, float epsilon, const engine &aengine) {
        // copy scale and shift to scale_shift tensor and cache it
        if (disable_cache_data_ || scale_shift_.is_empty()) {
            if (scale_shift_.is_empty())
                scale_shift_ = tensor {pd_.weights_desc(), aengine};
            auto *scale_shift_buf
                    = static_cast<char *>(scale_shift_.get_data_handle());
#if DNNL_GRAPH_WITH_SYCL
            stream s(aengine);
            cl::sycl::queue q = dnnl::sycl_interop::get_queue(s);
            q.memcpy(scale_shift_buf, scale.get_data_handle(), scale.get_size())
                    .wait();
            q.memcpy(scale_shift_buf + scale.get_size(),
                     shift.get_data_handle(), shift.get_size())
                    .wait();
#else
            std::memcpy(
                    scale_shift_buf, scale.get_data_handle(), scale.get_size());
            std::memcpy(scale_shift_buf + scale.get_size(),
                    shift.get_data_handle(), shift.get_size());
#endif
        }

        auto expected_src = src.reorder_if_differ_in(pd_.src_desc());

        tensor expected_dst = dst;
        if (pd_.dst_desc() != dst.get_desc()) {
            expected_dst = tensor {pd_.dst_desc(), aengine};
        }

        stream s(aengine);
        if (use_stats) {
            // cache reordered mean and var
            if (disable_cache_data_ || expected_mean_.is_empty()
                    || expected_var_.is_empty()) {
                expected_mean_ = mean.reorder_if_differ_in(pd_.mean_desc());
                expected_var_
                        = variance.reorder_if_differ_in(pd_.variance_desc());
            }
            super(pd_).execute(s,
                    {{DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_SCALE_SHIFT, scale_shift_},
                            {DNNL_ARG_VARIANCE, expected_var_},
                            {DNNL_ARG_MEAN, expected_mean_},
                            {DNNL_ARG_DST, expected_dst}});
        } else {
            super(pd_).execute(s,
                    {{DNNL_ARG_SRC, expected_src},
                            {DNNL_ARG_SCALE_SHIFT, scale_shift_},
                            {DNNL_ARG_DST, expected_dst}});
        }
        s.wait();

        // if output layout has been set and different from optimal layout
        // we have to do reorder
        if (expected_dst.get_desc() != dst.get_desc()) {
            expected_dst.reorder_to(dst);
        }
    }
};

struct batch_normalization_forward_training
    : public dnnl::batch_normalization_forward,
      public kernel_base {

    using super = dnnl::batch_normalization_forward;

private:
    float epsilon_;
    float mom_;
    primitive_desc pd_;

    tensor scale_shift_;
    tensor original_dst_;

public:
    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using desc = tensor::desc;
        epsilon_ = anode->get_attr<float>("epsilon");
        mom_ = anode->get_attr<float>("momentum");
        std::string data_format = anode->get_attr<std::string>("data_format");

        impl::logical_tensor_t src_lt = inputs.at(batch_normalization::kSrc);
        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper(
                    &inputs.at(batch_normalization::kSrc))
                             .reorder_data_dims_strides();
        }

        // prepare the inputs and outputs tensors' descs
        desc src {src_lt};
        const bool use_stats = inputs.size() > batch_normalization::kMean;
        impl::logical_tensor_t dst_lt = outputs.at(batch_normalization::kDst);
        impl::logical_tensor_t rm_lt
                = outputs.at(batch_normalization::kRunning_mean);
        impl::logical_tensor_t rv_lt
                = outputs.at(batch_normalization::kRunning_variance);

        auto flags = normalization_flag::use_scale_shift;
        if (use_stats) flags |= normalization_flag::use_global_stats;

        // workaround: use src once issue intel/mkl-dnn#588 is
        // resolved
        src = src.is_4c_blocked() ? src.to_default_format() : src;

        auto eng = engine_manager::get()->get_engine(*aengine);
        pd_ = primitive_desc(
                {prop_kind::forward_training, src, epsilon_, flags}, *eng);
        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        const tensor::desc optimal_rm_desc {pd_.mean_desc()};
        const tensor::desc optimal_rv_desc {pd_.variance_desc()};
        fill_layout_info(&dst_lt, optimal_dst_desc);
        fill_layout_info(&rm_lt, optimal_rm_desc);
        fill_layout_info(&rv_lt, optimal_rv_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));
        std::string data_format = anode->get_attr<std::string>("data_format");
        auto &src_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(batch_normalization::kSrc).get_logical_tensor());
        auto &dst_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(batch_normalization::kDst).get_logical_tensor());
        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper(src_lt)
                             .reorder_data_dims_strides();
            dst_lt = impl::logical_tensor_wrapper(dst_lt)
                             .reorder_data_dims_strides();
        }
        tensor x {src_lt,
                inputs.at(batch_normalization::kSrc).get_data_handle(), *eng};
        tensor w {inputs.at(batch_normalization::kScale), *eng};
        tensor b {inputs.at(batch_normalization::kShift), *eng};
        tensor y {dst_lt,
                outputs.at(batch_normalization::kDst).get_data_handle(), *eng};
        tensor m {inputs.at(batch_normalization::kMean), *eng};
        tensor v {inputs.at(batch_normalization::kVariance), *eng};
        tensor rm {outputs.at(batch_normalization::kRunning_mean), *eng};
        tensor rv {outputs.at(batch_normalization::kRunning_variance), *eng};
        tensor bm {outputs.at(batch_normalization::kBatch_mean), *eng};
        tensor bv {outputs.at(batch_normalization::kBatch_variance), *eng};
        compute(x, w, b, y, m, v, rm, rv, bm, bv, mom_, epsilon_, *eng);
        return impl::status::success;
    }

private:
    void compute_impl(tensor &src, const tensor &scale, const tensor &shift,
            tensor &dst, tensor &mean, tensor &variance, float momentum,
            float epsilon, const engine &eng) {
        UNUSED(momentum);

        original_dst_ = dst;
        scale_shift_ = tensor {pd_.weights_desc(), eng};
        auto *scale_shift_buf
                = static_cast<char *>(scale_shift_.get_data_handle());
        stream s(eng);
#if DNNL_GRAPH_WITH_SYCL
        cl::sycl::queue q = dnnl::sycl_interop::get_queue(s);
        q.memcpy(scale_shift_buf, scale.get_data_handle(), scale.get_size())
                .wait();
        q.memcpy(scale_shift_buf + scale.get_size(), shift.get_data_handle(),
                 shift.get_size())
                .wait();
#else
        std::memcpy(scale_shift_buf, scale.get_data_handle(), scale.get_size());
        std::memcpy(scale_shift_buf + scale.get_size(), shift.get_data_handle(),
                shift.get_size());
#endif

        mean.reinit_if_possible(pd_.mean_desc());
        variance.reinit_if_possible(pd_.variance_desc());
        dst.reinit_if_possible(pd_.dst_desc());
        super(pd_).execute(s,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_SCALE_SHIFT, scale_shift_},
                        {DNNL_ARG_MEAN, mean}, {DNNL_ARG_VARIANCE, variance},
                        {DNNL_ARG_DST, dst}});
        s.wait();
        if (original_dst_.get_desc() != dst.get_desc()) {
            dst.reorder_to(original_dst_);
        }
    }

    void compute(tensor &src, const tensor &scale, const tensor &shift,
            tensor &dst, tensor &mean, tensor &variance, tensor &running_mean,
            tensor &running_var, tensor &batch_mean, tensor &batch_var,
            float momentum, float epsilon, const engine &eng) {
        compute_impl(
                src, scale, shift, dst, mean, variance, momentum, epsilon, eng);
        // running_mean, running_mean's buffer can be empty
        sum::compute({momentum, 1 - momentum}, {running_mean, mean},
                running_mean, eng);
        sum::compute({momentum, 1 - momentum}, {running_var, variance},
                running_var, eng);
        // copy data
        batch_mean.reinit_if_possible(mean.get_desc());
        batch_var.reinit_if_possible(variance.get_desc());
#if DNNL_GRAPH_WITH_SYCL
        stream s(eng);
        cl::sycl::queue q = dnnl::sycl_interop::get_queue(s);
        q.memcpy(batch_mean.get_data_handle(), mean.get_data_handle(),
                 batch_mean.get_size())
                .wait();
        q.memcpy(batch_var.get_data_handle(), variance.get_data_handle(),
                 batch_var.get_size())
                .wait();
#else
        std::memcpy(batch_mean.get_data_handle(), mean.get_data_handle(),
                batch_mean.get_size());
        std::memcpy(batch_var.get_data_handle(), variance.get_data_handle(),
                batch_var.get_size());
#endif
    }
};

struct batch_normalization_backward : public dnnl::batch_normalization_backward,
                                      public kernel_base {
    using super = dnnl::batch_normalization_backward;

private:
    primitive_desc pd_;
    float epsilon_;

    tensor diff_scale_shift_;
    tensor original_diff_src_;

public:
    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) {
        using desc = tensor::desc;
        epsilon_ = anode->get_attr<float>("epsilon");
        std::string data_format = anode->get_attr<std::string>("data_format");

        impl::logical_tensor_t src_lt = inputs.at(batch_normalization::kSrc);
        impl::logical_tensor_t diff_src_lt
                = outputs.at(batch_normalization_bwd::kDiff_src);
        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper(&src_lt)
                             .reorder_data_dims_strides();
        }
        // prepare the inputs and outputs tensors' descs
        desc src {src_lt};

        const bool use_stats = inputs.size() > batch_normalization::kMean;

        auto flags = normalization_flag::use_scale_shift;
        if (use_stats) flags |= normalization_flag::use_global_stats;

        // workaround: use src once issue intel/mkl-dnn#588 is
        // resolved
        src = src.is_4c_blocked() ? src.to_default_format() : src;

        auto eng = engine_manager::get()->get_engine(*aengine);

        auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
                {prop_kind::forward_training, src, epsilon_, flags}, *eng);
        // cache diff_dst in order to compute
        pd_ = primitive_desc({prop_kind::backward, forward_hints.dst_desc(),
                                     src, epsilon_, flags},
                *eng, forward_hints);

        diff_scale_shift_ = tensor(pd_.diff_weights_desc(), *eng);
        const tensor::desc optimal_diff_src_desc {pd_.diff_src_desc()};
        fill_layout_info(&diff_src_lt, optimal_diff_src_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) {
        auto eng = engine_manager::get()->get_engine(*(astream->get_engine()));
        std::string data_format = anode->get_attr<std::string>("data_format");
        auto &src_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(batch_normalization_bwd::kSrc).get_logical_tensor());
        auto &diff_dst_lt = const_cast<impl::logical_tensor_t &>(
                inputs.at(batch_normalization_bwd::kDiff_dst)
                        .get_logical_tensor());
        auto &diff_src_lt = const_cast<impl::logical_tensor_t &>(
                outputs.at(batch_normalization_bwd::kDiff_src)
                        .get_logical_tensor());
        // "NXC"
        if (data_format == "NXC") {
            src_lt = impl::logical_tensor_wrapper(src_lt)
                             .reorder_data_dims_strides();
            diff_dst_lt = impl::logical_tensor_wrapper(diff_dst_lt)
                                  .reorder_data_dims_strides();
            diff_src_lt = impl::logical_tensor_wrapper(diff_src_lt)
                                  .reorder_data_dims_strides();
        }
        tensor x {src_lt,
                inputs.at(batch_normalization_bwd::kSrc).get_data_handle(),
                *eng};
        tensor w {inputs.at(batch_normalization_bwd::kScale), *eng};
        tensor m {inputs.at(batch_normalization_bwd::kMean), *eng};
        tensor v {inputs.at(batch_normalization_bwd::kVariance), *eng};
        tensor diff_dst {diff_dst_lt,
                inputs.at(batch_normalization_bwd::kDiff_dst).get_data_handle(),
                *eng};
        tensor diff_scale {
                outputs.at(batch_normalization_bwd::kDiff_scale), *eng};
        tensor diff_shift {
                outputs.at(batch_normalization_bwd::kDiff_shift), *eng};
        tensor diff_src {diff_src_lt,
                outputs.at(batch_normalization_bwd::kDiff_src)
                        .get_data_handle(),
                *eng};
        compute(x, m, v, diff_dst, w, diff_src, diff_scale, diff_shift, *eng);
        return impl::status::success;
    }

private:
    void compute_impl(tensor &src, tensor &mean, tensor &variance,
            tensor &diff_dst, const tensor &scale, tensor &diff_src,
            const engine &aengine) {
        // TODO(xxx): support no-affine model
        auto flags = normalization_flag::use_scale_shift;
        original_diff_src_ = diff_src;

        diff_dst.reinit_if_possible(pd_.diff_dst_desc());
        src.reinit_if_possible(pd_.src_desc());
        diff_src.reinit_if_possible(pd_.diff_src_desc());
        diff_scale_shift_.reinit_if_possible(pd_.diff_weights_desc());

        stream s(aengine);
        mean.reinit_if_possible(pd_.mean_desc());
        variance.reinit_if_possible(pd_.variance_desc());
        super(pd_).execute(s,
                {{DNNL_ARG_SRC, src}, {DNNL_ARG_DIFF_DST, diff_dst},
                        {DNNL_ARG_SCALE_SHIFT, scale}, // only need scale
                        {DNNL_ARG_MEAN, mean}, {DNNL_ARG_VARIANCE, variance},
                        {DNNL_ARG_DIFF_SRC, diff_src},
                        {DNNL_ARG_DIFF_SCALE_SHIFT, diff_scale_shift_}});
        s.wait();
        if (diff_src.get_desc() != original_diff_src_.get_desc()) {
            diff_src.reorder_to(original_diff_src_);
        }
    }

    void compute(tensor &src, tensor &mean, tensor &variance, tensor &diff_dst,
            const tensor &scale, tensor &diff_src, tensor &diff_scale,
            tensor &diff_shift, const engine &aengine) {
        compute_impl(src, mean, variance, diff_dst, scale, diff_src, aengine);
        diff_scale.reinit_if_possible(scale.get_desc());
        diff_shift.reinit_if_possible(scale.get_desc());
        auto *diff_scale_shift_buf
                = static_cast<char *>(diff_scale_shift_.get_data_handle());
#if DNNL_GRAPH_WITH_SYCL
        stream s(aengine);
        cl::sycl::queue q = dnnl::sycl_interop::get_queue(s);
        q.memcpy(diff_scale.get_data_handle(), diff_scale_shift_buf,
                 diff_scale.get_size())
                .wait();
        q.memcpy(diff_shift.get_data_handle(),
                 diff_scale_shift_buf + diff_scale.get_size(),
                 diff_shift.get_size())
                .wait();
#else
        std::memcpy(diff_scale.get_data_handle(), diff_scale_shift_buf,
                diff_scale.get_size());
        std::memcpy(diff_shift.get_data_handle(),
                diff_scale_shift_buf + diff_scale.get_size(),
                diff_shift.get_size());
#endif
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif

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

#ifndef BACKEND_DNNL_OPERATORS_REORDER_HPP
#define BACKEND_DNNL_OPERATORS_REORDER_HPP

#include <algorithm>
#include <vector>

#include "backend/dnnl/tensor.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace {
enum reorder_input { kSrc };
enum reorder_output { kDst };
} // namespace

struct reorder : public dnnl::reorder, public kernel_base {
    using super = dnnl::reorder;

private:
    primitive_desc pd_;
    dnnl::engine p_engine_;

    // handle the case in which the src and dst's dims are different
    bool need_reshape_ {false};
    // only valid if need_reshape_ is true
    bool reshape_first_ {false};
    tensor::desc reshaped_desc_;

public:
    impl::status_t compile_impl(const op_t *op, const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using ltw = logical_tensor_wrapper;
        // check data types between input and output
        // TODO(wuxun): in the future there will be fused with TypeCast op,
        // then we can make this check more general
        if (op->get_kind() == op_kind::Reorder
                && ltw(inputs[reorder_input::kSrc]).data_type()
                        != ltw(outputs[reorder_output::kDst]).data_type())
            return status::compile_fail;

        // check same shape between input and output
        const dims &src_lt_dims = ltw(inputs[reorder_input::kSrc]).vdims();
        const dims &dst_lt_dims = ltw(outputs[reorder_output::kDst]).vdims();
        if (!std::equal(src_lt_dims.begin(), src_lt_dims.end(),
                    dst_lt_dims.begin()))
            return status::compile_fail;

        tensor::desc src {inputs.at(reorder_input::kSrc)};
        tensor::desc dst {outputs.at(reorder_output::kDst)};

        bool has_group = src.is_grouped() || dst.is_grouped();
        bool ok = src.has_same_shape_as(dst)
                || (has_group
                        && std::abs(src.get_ndims() - dst.get_ndims()) == 1);
        if (!ok) return status::compile_fail;

        // has group
        if (!src.has_same_shape_as(dst)) {
            need_reshape_ = true;
            // Case 1: if src is blocked format with group while dst has plain
            // format, perhaps for backward path
            if (src.is_grouped() && src.get_ndims() > dst.get_ndims()) {
                reshape_first_ = false;
                reshaped_desc_ = src.is_plain()
                        ? src
                        : tensor::desc {src.get_dims(), src.get_data_type(),
                                src.get_format_tag()};
            } else if (dst.is_grouped() && src.get_ndims() < dst.get_ndims()) {
                // Case 2: if src has plain format while dst has blocked format
                // with group, typically for weight prepacking
                reshape_first_ = true;
                reshaped_desc_ = tensor::desc {
                        src.reshape(dst.get_dims()), dst.get_dim(0)};
            } else {
                return status::compile_fail;
            }
        }

        p_engine_ = make_dnnl_engine(*g_engine);

        if (need_reshape_) {
            if (reshape_first_)
                src = reshaped_desc_;
            else
                dst = reshaped_desc_;
        }
        pd_ = primitive_desc(
                /*src_engine=*/p_engine_, src, /*dst_engine=*/p_engine_, dst);
        return status::success;
    }

    impl::status_t execute_impl(const op_t *op, const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        tensor src_ts {inputs.at(reorder_input::kSrc), p_engine_, alc};
        tensor dst_ts {outputs.at(reorder_output::kDst), p_engine_, alc};

        if (need_reshape_) {
            if (reshape_first_) {
                src_ts = tensor {reshaped_desc_, p_engine_, alc,
                        inputs[reorder_input::kSrc].get_data_handle()};
            } else {
                dst_ts = tensor {reshaped_desc_, p_engine_, alc,
                        outputs[reorder_output::kDst].get_data_handle()};
            }
        }

        super(pd_).execute(p_stream, src_ts, dst_ts);
        return status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

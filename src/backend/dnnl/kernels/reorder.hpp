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

#ifndef BACKEND_DNNL_KERNELS_REORDER_HPP
#define BACKEND_DNNL_KERNELS_REORDER_HPP

#include <algorithm>
#include <vector>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"

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
    dnnl::reorder prim_;
    dnnl::engine p_engine_;

    // handle the case in which the src and dst's dims are different
    bool need_reshape_ {false};
    // only valid if need_reshape_ is true
    bool reshape_first_ {false};
    memory::desc reshaped_desc_;

public:
    impl::status_t compile_impl(const op_t *op, const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using ltw = logical_tensor_wrapper;

        // check same shape between input and output
        const dims &src_lt_dims = ltw(inputs[reorder_input::kSrc]).vdims();
        const dims &dst_lt_dims = ltw(outputs[reorder_output::kDst]).vdims();
        if (src_lt_dims != dst_lt_dims) return status::compile_fail;

        memory::desc src
                = make_dnnl_memory_desc(inputs.at(reorder_input::kSrc));
        memory::desc dst
                = make_dnnl_memory_desc(outputs.at(reorder_output::kDst));

        if (op->get_kind() == impl::op_kind::Reorder) {
            const bool src_has_group = src.data.ndims == dst.data.ndims + 1;
            const bool dst_has_group = src.data.ndims + 1 == dst.data.ndims;

            const bool has_group = src_has_group || dst_has_group;
            if (!(check_same_shape(src, dst) || has_group))
                return status::compile_fail;

            // check if the dims is in (g, O, I, H, W) format
            bool consistency = true;
            if (src_has_group) {
                int group = src.data.dims[0];
                consistency = group * src.data.dims[1] == dst.data.dims[0];
            } else if (dst_has_group) {
                int group = dst.data.dims[0];
                consistency = group * dst.data.dims[1] == src.data.dims[0];
            }
            if (!consistency) return status::compile_fail;

            // has group
            if (has_group) {
                need_reshape_ = true;
                // Case 1: if src is blocked format with group while dst has
                // plain format, perhaps for backward path.
                // No such case for now, so just disable
                if (src_has_group) {
                    return status::unsupported;
                } else {
                    // Case 2: if src has plain format while dst has blocked
                    // format with group, typically for weight prepacking
                    reshape_first_ = true;
                    reshaped_desc_ = src.reshape(dst.dims());
                }
            }

            if (need_reshape_) {
                if (reshape_first_)
                    src = reshaped_desc_;
                else
                    dst = reshaped_desc_;
            }
        }

        p_engine_ = make_dnnl_engine(*g_engine);

        pd_ = primitive_desc(
                /*src_engine=*/p_engine_, src, /*dst_engine=*/p_engine_, dst);
        prim_ = super(pd_);
        return status::success;
    }

    impl::status_t execute_impl(const op_t *op, const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);
        impl::allocator_t *alc = g_stream->get_engine()->get_allocator();

        memory src
                = make_dnnl_memory(inputs.at(reorder_input::kSrc), p_engine_);
        memory dst
                = make_dnnl_memory(outputs.at(reorder_output::kDst), p_engine_);

        if (op->get_kind() == impl::op_kind::Reorder && need_reshape_) {
            if (reshape_first_) {
                src = make_dnnl_memory(reshaped_desc_, p_engine_,
                        inputs[reorder_input::kSrc].get_data_handle());
            } else {
                dst = make_dnnl_memory(reshaped_desc_, p_engine_,
                        outputs[reorder_output::kDst].get_data_handle());
            }
        }

        prim_.execute(p_stream, src, dst);
        return status::success;
    }

private:
    bool check_same_shape(
            const memory::desc &in1, const memory::desc &in2) const {
        if (in1.data.ndims != in2.data.ndims) return false;
        return std::equal(
                in1.data.dims, in1.data.dims + in1.data.ndims, in2.data.dims);
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

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

#ifndef LLGA_BACKEND_DNNL_OPERATORS_BINARY_HPP
#define LLGA_BACKEND_DNNL_OPERATORS_BINARY_HPP

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "interface/backend.hpp"

#include "backend/dnnl/abstract_types.hpp"
#include "backend/dnnl/tensor.hpp"

namespace llga {
namespace impl {
namespace dnnl_impl {

namespace bin {
enum binary_inputs { kSrc0, kSrc1 };
enum binary_outputs { kDst };
} // namespace bin

/// \note the shape of src0 and src1 have to meet the one of
/// following requirement:
/// property 1: src0 and src1 both have exactly the same shape.
/// property 2: src0 and src1 all have the same number of dimensions
///             and the length of each src1 dimensions is either a common
///             length or src1's length is 1.
/// property 3: src1 has too few dimensions, and src1 can have its
///             shapes prepended with a dimension of length 1 to
///             satisfy property 2.
/// \note src0 and src1 in the above description can be swapped
struct binary : public dnnl::binary, public kernel_base {
    using super = dnnl::binary;

private:
    primitive_desc pd_;
    std::string auto_broadcast_ {"numpy"};
    bool unidirectional_broadcast_ {false};
    bool multidirectional_broadcast_ {false};

    tensor::desc before_broadcast_src0_desc_;

    tensor expected_src0_;
    tensor expected_src1_;
    tensor expected_dst_;

    dnnl::engine eng_;

    bool swapped_ {false};

    bool require_broadcast(const tensor::desc &src0_desc,
            const tensor::desc &src1_desc, const tensor::desc &dst_desc) {
        return !(src0_desc.has_same_shape_as(src1_desc)
                && src0_desc.has_same_shape_as(dst_desc));
    }

    bool prepare_bias_add_broadcast(const tensor::desc &src0_desc,
            const tensor::desc &src1_desc, tensor::desc &new_src1_dest) {
        // only support plain format broadcast
        auto default_src1_desc = src1_desc;
        if (!src1_desc.is_plain())
            default_src1_desc = src1_desc.to_default_format();

        auto src0_ndims = src0_desc.get_ndims();
        auto bias_ndims = default_src1_desc.get_ndims();

        // We only support 1D bias's broadcast add now
        if (bias_ndims != 1) { return false; }

        auto src0_dims = src0_desc.get_dims();
        auto bias_dims = default_src1_desc.get_dims();

        dims new_src1_dims(src0_ndims, 0);
        dims new_src1_strides(src0_ndims, 0);

        // in oneDNN, the channel dim will always be the second dim
        const size_t channel_dim = 1;

        if (src0_ndims == 1 && bias_dims[0] != 1) {
            // bias length must be equal to 1 when broadcast to 1 dim src
            return false;
        } else if (bias_dims[0] != src0_dims[channel_dim]) {
            // bias length must be equal to src's channel value
            // when broadcast to non-one dim src
            return false;
        }

        // we always unsequeeze 1D bias channel dim and default format
        // for example src0 {8, 96, 224, 224}, bias {96} will be
        // unsequeeze to bias {1, 96, 1, 1}, and its format_tag
        // will be abcd
        for (int i = 0; i < src0_ndims; i++) {
            new_src1_dims[i] = i == channel_dim ? bias_dims[0] : 1;
            new_src1_strides[i] = i == 0 ? bias_dims[0] : 1;
        }

        // create a new desc according to broadcasted dims and strides
        new_src1_dest = tensor::desc(new_src1_dims,
                default_src1_desc.get_data_type(), new_src1_strides);

        return true;
    }

    // for common unidirectional broadcast operation
    // we use the same rule with numpy and onnx:
    // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    bool prepare_unidirectional_broadcast(const tensor::desc &src0_desc,
            const tensor::desc &src1_desc, tensor::desc &new_src1_dest) {
        // only support plain format broadcast
        auto default_src1_desc = src1_desc;
        if (!src1_desc.is_plain())
            default_src1_desc = src1_desc.to_default_format();

        auto src0_ndims = src0_desc.get_ndims();
        auto src1_ndims = default_src1_desc.get_ndims();
        auto src0_dims = src0_desc.get_dims();
        auto src1_dims = default_src1_desc.get_dims();
        auto src1_strides = default_src1_desc.get_strides();

        // check valid shape
        for (int i = src1_ndims - 1, j = src0_ndims - 1; i >= 0; i--, j--) {
            if (src1_dims[i] != src0_dims[j] && src1_dims[i] != 1) {
                // src1 dim value need to be equal to src0 dims or 1
                return false;
            }
        }

        // unsqueeze
        dims new_src1_dims(src0_ndims, 1);
        dims new_src1_strides(src0_ndims, src1_strides[0] * src1_dims[0]);
        for (int i = src1_ndims - 1, j = src0_ndims - 1; i >= 0; i--, j--) {
            new_src1_dims[j] = src1_dims[i];
            new_src1_strides[j] = src1_strides[i];
        }

        // create a new desc according to broadcasted dims and strides
        new_src1_dest = tensor::desc(new_src1_dims,
                default_src1_desc.get_data_type(), new_src1_strides);

        return true;
    }

    // FIXME(qun) we don't support multidirectional broadcast now
    // for multidirectional broadcast operation
    // we use the same rule with numpy and onnx:
    // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    bool prepare_multidirectional_broadcast(const tensor::desc &src0_desc,
            const tensor::desc &src1_desc, tensor::desc &new_src0_dest,
            tensor::desc &new_src1_dest) {
        // only support plain format broadcast
        auto default_src0_desc = src0_desc;
        if (!src0_desc.is_plain())
            default_src0_desc = src0_desc.to_default_format();

        auto default_src1_desc = src1_desc;
        if (!src1_desc.is_plain())
            default_src1_desc = src1_desc.to_default_format();

        auto src0_ndims = default_src0_desc.get_ndims();
        auto src1_ndims = default_src1_desc.get_ndims();
        auto src0_dims = default_src0_desc.get_dims();
        auto src1_dims = default_src1_desc.get_dims();

        // check valid shape
        auto min_ndims = std::min(src0_ndims, src1_ndims);
        for (int i = src0_ndims - 1, j = src1_ndims - 1, k = 0; k < min_ndims;
                i--, j--, k++) {
            if (src0_dims[i] != src1_dims[j]
                    && (src0_dims[i] != 1 && src1_dims[j] != 1)) {
                // dim value need to be equal or 1
                return false;
            }
        }

        // unsqueeze lambda
        auto unsqueeze = [](const tensor::desc &src_desc,
                                 int target_ndims) -> tensor::desc {
            auto src_dims = src_desc.get_dims();
            auto src_strides = src_desc.get_strides();

            dims target_dims(target_ndims, 1);
            dims target_strides(target_ndims, src_strides[0] * src_dims[0]);
            for (int i = src_desc.get_ndims() - 1, j = target_ndims - 1; i >= 0;
                    i--, j--) {
                target_dims[j] = src_dims[i];
                target_strides[j] = src_strides[i];
            }

            return tensor::desc(
                    target_dims, src_desc.get_data_type(), target_strides);
        };

        // unsqueeze the short desc
        if (src0_ndims > src1_ndims) {
            default_src1_desc = unsqueeze(default_src1_desc, src0_ndims);
            src1_ndims = default_src1_desc.get_ndims();
            src1_dims = default_src1_desc.get_dims();
        } else {
            default_src0_desc = unsqueeze(default_src0_desc, src1_ndims);
            src0_ndims = default_src0_desc.get_ndims();
            src0_dims = default_src0_desc.get_dims();
        }

        before_broadcast_src0_desc_ = default_src0_desc;

        // broadcast sr0
        // we only broadcast src0, the src0 and src1 ndims have to be equal
        dims target_dims(src0_ndims, 0);
        for (int i = 0; i < src0_ndims; i++) {
            target_dims[i] = std::max(src0_dims[i], src1_dims[i]);
        }
        new_src0_dest
                = tensor::desc(target_dims, default_src0_desc.get_data_type(),
                        default_src0_desc.get_format_tag());

        new_src1_dest = default_src1_desc;
        return true;
    }

public:
    impl::status_t compile_impl(const impl::node_t *anode,
            const impl::engine_t *aengine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        size_t src0_index = bin::kSrc0, src1_index = bin::kSrc1;

        // onednn only support src1 broadcast, so the src0 shape
        // must be equal to dst
        if (impl::logical_tensor_wrapper(inputs[bin::kSrc0]).vdims()
                != impl::logical_tensor_wrapper(outputs[bin::kDst]).vdims()) {
            std::swap(src0_index, src1_index);
            swapped_ = true;
        }

        tensor::desc src0_desc {inputs.at(src0_index)};
        tensor::desc src1_desc {inputs.at(src1_index)};

        impl::logical_tensor_t dst_lt = outputs.at(bin::kDst);
        const tensor::desc dst_desc_any = tensor::desc(dst_lt).to_format_any();

        // TODO(qun) get auto_broadcast flag
        // auto_broadcast_ = anode->get_attr<std::string>("auto_broadcast");

        if (require_broadcast(src0_desc, src1_desc, dst_desc_any)) {
            assert(auto_broadcast_ == "numpy");

            if (src0_desc.has_same_shape_as(dst_desc_any)
                    && (src0_desc.get_ndims() >= src1_desc.get_ndims())) {
                // unidirectional broadcast
                tensor::desc new_src1_dest;

                auto prepare_func = [&]() {
                    return anode->get_op_kind() == op_kind::BiasAdd
                            ? prepare_bias_add_broadcast(
                                    src0_desc, src1_desc, new_src1_dest)
                            : prepare_unidirectional_broadcast(
                                    src0_desc, src1_desc, new_src1_dest);
                };

                // FIXME(qun) assert(prepare_func()) will fail in pytest
                bool ret = prepare_func();
                assert(ret);

                src1_desc = new_src1_dest;
                unidirectional_broadcast_ = true;
            } else if ((!src0_desc.has_same_shape_as(dst_desc_any))
                    && (src0_desc.get_ndims() == dst_desc_any.get_ndims()
                            || src1_desc.get_ndims()
                                    == dst_desc_any.get_ndims())) {
                // multidirectional broadcast
                tensor::desc new_src0_dest, new_src1_dest;

                bool ret = prepare_multidirectional_broadcast(
                        src0_desc, src1_desc, new_src0_dest, new_src1_dest);
                assert(ret);

                src0_desc = new_src0_dest;
                src1_desc = new_src1_dest;

                // the broadcasted src0 shape should be equal to dst shape
                if (src0_desc.get_dims() != dst_desc_any.get_dims())
                    return impl::status::invalid_argument;
                multidirectional_broadcast_ = true;
            } else {
                assert(!"can't perform broadcast for current inputs shape");
            }
        }

        algorithm alg_kind;
        switch (anode->get_op_kind()) {
            case op_kind::BiasAdd:
            case op_kind::Add: alg_kind = algorithm::binary_add; break;
            case op_kind::Multiply: alg_kind = algorithm::binary_mul; break;
            case op_kind::Maximum: alg_kind = algorithm::binary_max; break;
            case op_kind::Minimum: alg_kind = algorithm::binary_min; break;
            default:
                assert(!"can't perform broadcast for current inputs shape");
        }

        eng_ = make_dnnl_engine(*aengine);
        pd_ = primitive_desc(
                {alg_kind, src0_desc, src1_desc, dst_desc_any}, eng_);

        const tensor::desc optimal_dst_desc {pd_.dst_desc()};
        impl::logical_tensor_t *orgi_dst_lt
                = const_cast<impl::logical_tensor_t *>(&outputs.at(bin::kDst));
        fill_layout_info(orgi_dst_lt, optimal_dst_desc);
        return impl::status::success;
    }

    impl::status_t execute_impl(const impl::node_t *anode,
            const impl::stream *astream,
            const std::vector<impl::tensor> &inputs,
            const std::vector<impl::tensor> &outputs) override {
        impl::allocator_t *alc = astream->get_engine()->get_allocator();

        size_t src0_index = bin::kSrc0, src1_index = bin::kSrc1;
        if (swapped_) { std::swap(src0_index, src1_index); }

        const tensor src0 {inputs.at(src0_index), eng_, alc};
        tensor src1 {inputs.at(src1_index), eng_, alc};
        tensor dst {outputs.at(bin::kDst), eng_, alc};

        if (pd_.src0_desc() != src0.get_desc()) {
            // allocate memory for optimal layout src0 in the first iteration
            if (expected_src0_.is_empty()) {
                expected_src0_ = tensor {pd_.src0_desc(), eng_, alc};
            }
            src0.reorder_to(expected_src0_);
        } else {
            expected_src0_ = src0;
        }

        tensor plain_src1;
        if (unidirectional_broadcast_) {
            if (!src1.get_desc().is_plain()) {
                plain_src1 = tensor(
                        src1.get_desc().to_default_format(), eng_, alc);
                src1.reorder_to(plain_src1);
                expected_src1_ = tensor(pd_.src1_desc(), eng_, alc,
                        plain_src1.get_data_handle());
            } else {
                expected_src1_ = tensor(
                        pd_.src1_desc(), eng_, alc, src1.get_data_handle());
            }

        } else if (multidirectional_broadcast_) {
            assert(!"don't support multidirectional broadcas now");
        } else {
            if (pd_.src1_desc() != src1.get_desc()) {
                // allocate memory for optimal layout src1
                // in the first iteration
                if (expected_src1_.is_empty()) {
                    expected_src1_ = tensor {pd_.src1_desc(), eng_, alc};
                }
                src1.reorder_to(expected_src1_);
            } else {
                expected_src1_ = src1;
            }
        }

        if (pd_.dst_desc() != dst.get_desc()) {
            if (expected_dst_.is_empty()) {
                expected_dst_ = tensor {pd_.dst_desc(), eng_, alc};
            }
        } else {
            expected_dst_ = dst;
        }

        dnnl::stream s(eng_);
        super(pd_).execute(s,
                {{DNNL_ARG_SRC_0, expected_src0_},
                        {DNNL_ARG_SRC_1, expected_src1_},
                        {DNNL_ARG_DST, expected_dst_}});
        s.wait();

        // if output layout has been set and different from optimal layout
        // we have to do reorder
        if (expected_dst_ != dst) {
            dnnl::reorder(expected_dst_, dst).execute(s, expected_dst_, dst);
            s.wait();
        }
        return impl::status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif

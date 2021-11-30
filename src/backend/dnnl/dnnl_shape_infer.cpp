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
#include <algorithm>
#include "dnnl_shape_infer.hpp"
#include "interface/shape_infer.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

status_t infer_dnnl_conv_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = impl::logical_tensor_wrapper_t;

    auto backup = *inputs[1];
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    bool out_shape_unknown = out0.is_shape_unknown();
    if (n->get_attr<int64_t>("groups") > 1) {
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        dims[1] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < ndims; i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    infer_conv_output_shape(n, inputs, outputs);
    *inputs[1] = backup;

    return status::success;
}

status_t infer_dnnl_conv_depthwise_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    logical_tensor_t tmp_out = empty_logical_tensor_with_default_id();
    std::vector<logical_tensor_t *> tmp_outs {&tmp_out};
    const status_t ret = infer_conv_output_shape(n, inputs, tmp_outs);
    if (ret != status::success) return ret;

    // at this stage tmp_out corresponds to conv_1x1 dst
    // we now just need to adjust oh and ow in case of dw_k3s2p1 post-op
    dims output_dims(logical_tensor_wrapper_t(&tmp_out).vdims());
    if (n->get_attr<std::string>("dw_type") == "k3s2p1") {
        const std::string src_fmt = n->get_attr<std::string>("data_format");
        const size_t oh_offset
                = (src_fmt == "NCX") ? output_dims.size() - 2 : 1;
        const size_t ow_offset
                = (src_fmt == "NCX") ? output_dims.size() - 1 : 2;
        const dim_t stride = 2;
        const dim_t new_oh = static_cast<dim_t>(
                std::ceil(output_dims[oh_offset] / stride));
        const dim_t new_ow = static_cast<dim_t>(
                std::ceil(output_dims[ow_offset] / stride));
        output_dims[oh_offset] = new_oh;
        output_dims[ow_offset] = new_ow;
    }

    set_shape_and_strides(*outputs[0], output_dims);
    return status::success;
}

status_t infer_dnnl_convtranspose_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = impl::logical_tensor_wrapper_t;

    auto backup = *inputs[1];
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    bool out_shape_unknown = out0.is_shape_unknown();
    if (n->get_attr<int64_t>("groups") > 1) {
        // [g, O/g, I/g, H, W]
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        dims[2] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < ndims; i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    infer_convtranspose_output_shape(n, inputs, outputs);
    *inputs[1] = backup;

    return status::success;
}

status_t infer_dnnl_pool_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    bool out_shape_unknown
            = logical_tensor_wrapper_t(outputs[0]).is_shape_unknown();
    infer_pool_output_shape(n, inputs, outputs);
    return status::success;
}

status_t infer_permute_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;
    auto out0 = ltw(outputs[0]);

    // this permute is actually a transpose
    if (n->get_attr<std::string>("permute_kind") == "transpose") {
        auto tmp = ltw(inputs[0]).vdims();
        auto rank = tmp.size();
        // swap the right-most two elements
        std::swap(tmp[rank - 2], tmp[rank - 1]);

        // check the given shape
        if (!out0.is_shape_unknown()) {
            if (!validate(tmp, out0.vdims())) { return status::invalid_shape; }
        }

        set_shape_and_strides(*outputs[0], tmp);
        return status::success;
    }

    auto from_format = n->get_attr<std::string>("from_format");
    auto to_format = n->get_attr<std::string>("to_format");
    impl::logical_tensor_t tmp;
    std::vector<dim_t> tmp_dims;
    if (from_format == "NCX" && to_format == "NXC") {
        auto in_dims = ltw(inputs[0]).vdims();
        tmp_dims.emplace_back(in_dims[0]); // N
        for (size_t i = 2; i < in_dims.size(); i++) { // X
            tmp_dims.emplace_back(in_dims[i]);
        }
        tmp_dims.emplace_back(in_dims[1]); // C
    } else if (from_format == "NXC" && to_format == "NCX") {
        tmp = ltw(inputs[0]).reorder_data_dims_strides();
        tmp_dims = ltw(tmp).vdims();
    } else if (from_format == "XIO" && to_format == "OIX") {
        tmp = ltw(inputs[0]).reorder_weight_dims_strides();
        tmp_dims = ltw(tmp).vdims();
    } else {
        assertm(false, "should not reach here");
        return status::unsupported;
    }

    // check the given shape
    if (!out0.is_shape_unknown()) {
        if (!validate(tmp_dims, out0.vdims())) { return status::invalid_shape; }
    }
    set_shape_and_strides(*outputs[0], tmp_dims);

    return status::success;
}

status_t infer_to_group_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    if (!out0.is_shape_unknown()) return status::success;

    auto groups = n->get_attr<int64_t>("groups");
    dims in_dims = in0.vdims();

    if (n->has_attr("is_convtranspose")
            && n->get_attr<bool>("is_convtranspose")) {
        in_dims[1] /= groups;
    } else {
        in_dims[0] /= groups;
    }
    in_dims.insert(in_dims.begin(), groups);

    // We should compute output dense strides instead of
    // directly copying input strides to it
    set_shape_and_strides(*outputs[0], in_dims);
    UNUSED(n);
    return status::success;
}

status_t infer_expand_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;
    if (!ltw(outputs[0]).is_shape_unknown()) return status::success;

    auto in_dims = ltw(inputs[0]).vdims();
    if (n->has_attr("insert_1dim")) {
        auto insert_1dim = n->get_attr<std::string>("insert_1dim");
        if (insert_1dim == "before") {
            in_dims.insert(in_dims.begin(), 1);
        } else if (insert_1dim == "after") {
            in_dims.insert(in_dims.end(), 1);
        }
    }

    if (n->has_attr("expand_to")) {
        auto target_ndims = n->get_attr<int64_t>("expand_to");
        if (target_ndims != -1) {
            in_dims.insert(in_dims.begin(),
                    static_cast<size_t>(target_ndims) - in_dims.size(), 1);
        }
    }
    set_shape_and_strides(*outputs[0], in_dims);
    return status::success;
}

status_t infer_squeeze_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;
    if (!ltw(outputs[0]).is_shape_unknown()) return status::success;

    auto in_dims = ltw(inputs[0]).vdims();
    auto in_ndim = in_dims.size();

    auto axes = n->get_attr<std::vector<int64_t>>("axes");
    // convert negative axis to positive one
    std::transform(axes.begin(), axes.end(), axes.begin(),
            [&in_ndim](int64_t axis) -> int64_t {
                return axis < 0 ? axis + in_ndim : axis;
            });

    dims inferred_output_shape = {};
    for (size_t i = 0; i < in_ndim; ++i) {
        if (axes.empty() && in_dims[i] == 1) {
            continue;
        } else if (!axes.empty()
                && std::find(axes.begin(), axes.end(), i) != axes.end()) {
            if (in_dims[i] != 1) {
                // Dimension must be 1
                return status::invalid_argument;
            }
        } else {
            inferred_output_shape.push_back(in_dims[i]);
        }
    }
    set_shape_and_strides(*outputs[0], inferred_output_shape);
    return status::success;
}

status_t infer_bn_folding_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    auto out1 = logical_tensor_wrapper_t(outputs[1]);
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    auto in1 = logical_tensor_wrapper_t(inputs[1]);

    if (!out0.is_shape_unknown() && !out1.is_shape_unknown())
        return status::success;

    // check if partial set shape aligns with infered shape
    if (out0.ndims() != -1) {
        if (!impl::validate(in0.vdims(), out0.vdims())) {
            return impl::status::invalid_shape;
        }
    }

    if (out1.ndims() != -1) {
        if (!impl::validate(in1.vdims(), out1.vdims())) {
            return impl::status::invalid_shape;
        }
    }

    // We should compute output dense strides instead of
    // directly copying input strides to it
    set_shape_and_strides(*outputs[0], in0.vdims());
    set_shape_and_strides(*outputs[1], in1.vdims());
    UNUSED(n);
    return status::success;
}

status_t infer_dnnl_conv_bwd_data_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = impl::logical_tensor_wrapper_t;

    auto backup = *inputs[1];
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    bool out_shape_unknown = out0.is_shape_unknown();
    if (n->get_attr<int64_t>("groups") > 1) {
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        dims[1] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < ndims; i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    auto ret = infer_conv_bprop_data_output_shape(n, inputs, outputs);
    if (ret != status::success) return ret;
    *inputs[1] = backup;
    return status::success;
}

status_t infer_dnnl_batchnorm_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    status_t stat = status::success;
    if (n->get_attr<bool>("is_training"))
        stat = infer_bn_fwd_train_output_shape(n, inputs, outputs);
    else
        stat = infer_identity_output_shape(n, inputs, outputs);
    return stat;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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
#include <unordered_set>

#include "graph/interface/shape_infer.hpp"

#include "graph/backend/dnnl/dnnl_shape_infer.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

static status_t infer_dnnl_conv_common_bwd_weight_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs,
        const size_t axis_with_groups) {
    bool canonicalized = n->has_attr(op_attr::canonicalized)
            && n->get_attr<bool>(op_attr::canonicalized);
    const auto groups = n->get_attr<int64_t>(op_attr::groups);

    auto out = logical_tensor_wrapper_t(outputs[0]); // diff_wei
    if (canonicalized && groups > 1 && !out.is_shape_unknown()) {
        // convert the out shape to uncanonicalized form to reuse the frontend
        // shape infer function.
        auto out_dims = out.vdims();
        auto groups = out_dims[0];
        out_dims.erase(out_dims.begin());
        out_dims[axis_with_groups] *= groups;
        set_shape_and_strides(*outputs[0], out_dims);
    }

    // infer pad and filter shape (groups not included)
    const auto ret = infer_conv_bprop_filters_output_shape(n, inputs, outputs);
    if (ret != status::success) return ret;

    // add groups into weights shape
    if (canonicalized && groups > 1) {
        auto out_dims = logical_tensor_wrapper_t(outputs[0]).vdims();
        out_dims[axis_with_groups] /= groups;
        out_dims.insert(out_dims.begin(), groups);

        set_shape_and_strides(*outputs[0], out_dims);
    }

    return status::success;
}

status_t infer_dnnl_conv_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;

    auto out = logical_tensor_wrapper_t(outputs[0]);
    const bool org_out_shape_unknown = out.is_shape_unknown();

    auto backup_wei_shape = *inputs[1];
    auto backup_groups = n->get_attr<int64_t>(op_attr::groups);
    if (n->has_attr(op_attr::canonicalized)
            && n->get_attr<bool>(op_attr::canonicalized)
            && (ltw(inputs[1]).ndims() == ltw(inputs[0]).ndims() + 1)) {
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        n->set_attr<int64_t>(op_attr::groups, static_cast<int64_t>(dims[0]));

        dims[1] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    infer_conv_output_shape(n, inputs, outputs);
    *inputs[1] = backup_wei_shape;
    n->set_attr<int64_t>(op_attr::groups, backup_groups);

    // The following code will take effect only when fusing dw conv.
    // At this stage outputs[0] corresponds to conv_1x1 dst
    // we now just need to adjust oh and ow in case of dw_k3s2p1 post-op
    dims output_dims(logical_tensor_wrapper_t(outputs[0]).vdims());
    if (org_out_shape_unknown && n->has_attr(op_attr::dw_type)
            && n->get_attr<std::string>(op_attr::dw_type) == "k3s2p1") {
        const std::string src_fmt
                = n->get_attr<std::string>(op_attr::data_format);
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
        set_shape_and_strides(*outputs[0], output_dims);
    }

    return status::success;
}

status_t infer_dnnl_convtranspose_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;

    auto backup = *inputs[1];
    auto backup_groups = n->get_attr<int64_t>(op_attr::groups);
    bool is_canonicalized = n->has_attr(op_attr::canonicalized)
            && n->get_attr<bool>(op_attr::canonicalized);
    if (is_canonicalized
            && (ltw(inputs[1]).ndims() == ltw(inputs[0]).ndims() + 1)) {
        // [g, O/g, I/g, H, W]
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        n->set_attr<int64_t>(op_attr::groups, static_cast<int64_t>(dims[0]));

        dims[2] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    infer_convtranspose_output_shape(n, inputs, outputs);
    *inputs[1] = backup;
    n->set_attr<int64_t>(op_attr::groups, backup_groups);
    return status::success;
}

status_t infer_dnnl_convtranspose_bwd_data_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;

    auto backup_wei_shape = *inputs[1];
    auto backup_groups = n->get_attr<int64_t>(op_attr::groups);
    if (n->has_attr(op_attr::canonicalized)
            && n->get_attr<bool>(op_attr::canonicalized)
            && (ltw(inputs[1]).ndims() == ltw(inputs[0]).ndims() + 1)) {
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        n->set_attr<int64_t>(op_attr::groups, static_cast<int64_t>(dims[0]));

        dims[2] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    infer_convtranspose_bprop_data_output_shape(n, inputs, outputs);
    *inputs[1] = backup_wei_shape;
    n->set_attr<int64_t>(op_attr::groups, backup_groups);
    return status::success;
}

status_t infer_dnnl_convtranspose_bwd_weight_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    const size_t axis_with_groups = 1;
    return infer_dnnl_conv_common_bwd_weight_output_shape(
            n, inputs, outputs, axis_with_groups);
}

status_t infer_dnnl_pool_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    infer_pool_output_shape(n, inputs, outputs);
    return status::success;
}

status_t infer_permute_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;
    auto out0 = ltw(outputs[0]);

    auto in_dims = ltw(inputs[0]).vdims();
    auto perm = n->get_attr<std::vector<int64_t>>(op_attr::permutation);
    std::vector<dim_t> inferred_out_dims(perm.size(), DNNL_GRAPH_UNKNOWN_DIM);
    for (size_t i = 0; i < perm.size(); i++) {
        inferred_out_dims[perm[i]] = in_dims[i];
    }

    // check the given shape
    if (!out0.is_shape_unknown()) {
        if (!validate(inferred_out_dims, out0.vdims())) {
            return status::invalid_shape;
        }
    }
    set_shape_and_strides(*outputs[0], inferred_out_dims);

    return status::success;
}

status_t infer_to_group_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper_t(outputs[0]);
    auto in0 = logical_tensor_wrapper_t(inputs[0]);
    if (!out0.is_shape_unknown()) return status::success;

    auto groups = n->get_attr<int64_t>(op_attr::groups);
    dims in_dims = in0.vdims();

    if (n->has_attr(op_attr::is_convtranspose)
            && n->get_attr<bool>(op_attr::is_convtranspose)) {
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

status_t infer_from_group_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out = logical_tensor_wrapper_t(outputs[0]);
    if (!out.is_shape_unknown()) return status::success;

    const auto groups = n->get_attr<int64_t>(op_attr::groups);
    dims inferred_out_dims = logical_tensor_wrapper_t(inputs[0]).vdims();
    inferred_out_dims.erase(inferred_out_dims.begin());
    if (n->has_attr(op_attr::is_convtranspose)
            && n->get_attr<bool>(op_attr::is_convtranspose)) {
        inferred_out_dims[1] *= groups;
    } else {
        inferred_out_dims[0] *= groups;
    }

    set_shape_and_strides(*outputs[0], inferred_out_dims);

    return status::success;
}

status_t infer_unsqueeze_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;
    if (!ltw(outputs[0]).is_shape_unknown()) return status::success;

    auto axes = (n->has_attr(op_attr::axes))
            ? n->get_attr<std::vector<int64_t>>(op_attr::axes)
            : std::vector<int64_t>();
    const auto in_dims = ltw(inputs[0]).vdims();

    const auto out_ndim = static_cast<int64_t>(in_dims.size() + axes.size());
    if (std::any_of(axes.begin(), axes.end(), [&out_ndim](int64_t axis) {
            return axis < -out_ndim || axis >= out_ndim;
        }))
        return status::unimplemented;

    // convert negative axis to positive one
    std::transform(axes.begin(), axes.end(), axes.begin(),
            [&out_ndim](int64_t axis) -> int64_t {
                return axis < 0 ? out_ndim + axis : axis;
            });

    if (std::unordered_set<int64_t>(axes.begin(), axes.end()).size()
            < axes.size())
        return status::unimplemented;

    std::vector<size_t> indices(out_ndim);
    std::iota(indices.begin(), indices.end(), 0);
    dims inferred_output_shape(out_ndim, 1);
    size_t in_dims_idx = 0;
    for (const auto i : indices) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end())
            inferred_output_shape[i] = in_dims[in_dims_idx++];
    }

    set_shape_and_strides(*outputs[0], inferred_output_shape);

    return status::success;
}

status_t infer_squeeze_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;
    if (!ltw(outputs[0]).is_shape_unknown()) return status::success;

    auto in_dims = ltw(inputs[0]).vdims();
    auto in_ndim = in_dims.size();

    auto axes = n->get_attr<std::vector<int64_t>>(op_attr::axes);
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
                return status::invalid_arguments;
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

    // check if partial set shape aligns with inferred shape
    if (out0.ndims() != -1) {
        if (!validate(in0.vdims(), out0.vdims())) {
            return status::invalid_shape;
        }
    }

    if (out1.ndims() != -1) {
        if (!validate(in1.vdims(), out1.vdims())) {
            return status::invalid_shape;
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
    using ltw = logical_tensor_wrapper_t;

    auto backup = *inputs[1];
    if (n->get_attr<int64_t>(op_attr::groups) > 1) {
        auto ndims = ltw(inputs[1]).ndims() - 1;
        auto dims = ltw(inputs[1]).vdims();
        dims[1] *= dims[0];
        dims.erase(dims.begin());

        inputs[1]->ndims = ndims;
        for (size_t i = 0; i < static_cast<size_t>(ndims); i++) {
            inputs[1]->dims[i] = dims[i];
        }
    }

    auto ret = infer_conv_bprop_data_output_shape(n, inputs, outputs);
    if (ret != status::success) return ret;
    *inputs[1] = backup;
    return status::success;
}

status_t infer_dnnl_conv_bwd_weight_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    const size_t axis_with_groups = 0;
    return infer_dnnl_conv_common_bwd_weight_output_shape(
            n, inputs, outputs, axis_with_groups);
}

status_t infer_dnnl_batchnorm_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    status_t stat = status::success;
    if (n->get_attr<bool>(op_attr::is_training))
        stat = infer_bn_fwd_train_output_shape(n, inputs, outputs);
    else
        stat = infer_identity_output_shape(n, inputs, outputs);
    return stat;
}

status_t infer_dnnl_batchnorm_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    // skip shape inference for scratchpad output
    // FIXME(wuxun): may remove this temporary solution after we refine op
    // definition to handle one or more optional input/outputs
    auto new_outputs = outputs;
    new_outputs.pop_back();
    infer_bn_bwd_output_shape(n, inputs, new_outputs);
    return status::success;
}

status_t infer_dnnl_constant_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    // `dnnl_constant_[scales|zps]` op doesn't have any inputs
    auto out_shape = n->get_attr<std::vector<int64_t>>(op_attr::shape);
    set_shape_and_strides(*outputs[0], out_shape);

    return status::success;
}

status_t infer_dnnl_pool_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto diff_src_shape = n->get_attr<std::vector<int64_t>>(op_attr::src_shape);
    set_shape_and_strides(*outputs[0], diff_src_shape);

    // get attr value
    const dims &strides = n->get_attr<dims>(op_attr::strides);
    const dims &kernel = n->get_attr<dims>(op_attr::kernel);
    const dims &pads_begin = n->get_attr<dims>(op_attr::pads_begin);
    const dims &pads_end = n->get_attr<dims>(op_attr::pads_end);
    std::string src_format = n->get_attr<std::string>(op_attr::data_format);

    dims dilations(kernel.size(), 1);
    if (n->has_attr(op_attr::dilations)) {
        auto dilations_tmp = n->get_attr<dims>(op_attr::dilations);
        if (dilations_tmp.size() != dilations.size()) {
            return status::invalid_arguments;
        } else {
            dilations = dilations_tmp;
        }
    }

    logical_tensor_wrapper_t diff_src_ltw(outputs[0]);
    dims src_sp = diff_src_ltw.get_src_spatial_dims(src_format);

    // if paddings are empty vectors?
    dims new_pads_begin(pads_begin);
    if (new_pads_begin.empty()) { new_pads_begin.assign(src_sp.size(), 0); }
    dims new_pads_end(pads_end);
    if (new_pads_end.empty()) { new_pads_end.assign(src_sp.size(), 0); }
    if (n->has_attr(op_attr::auto_pad)
            && n->get_attr<std::string>(op_attr::auto_pad) != "None") {
        std::string auto_pad = n->get_attr<std::string>(op_attr::auto_pad);
        // infer auto_pad
        for (size_t i = 0; i < src_sp.size(); ++i) {
            auto ret = infer_auto_pad(src_sp[i], strides[i], kernel[i],
                    dilations[i], auto_pad, new_pads_begin[i], new_pads_end[i]);

            if (ret != status::success) return ret;
        }
        n->set_attr(op_attr::pads_begin, new_pads_begin);
        n->set_attr(op_attr::pads_end, new_pads_end);
    }

    return status::success;
}

status_t infer_dnnl_binary_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    const bool is_bias_add = n->has_attr(op_attr::is_bias_add)
            && n->get_attr<bool>(op_attr::is_bias_add);

    auto ret = is_bias_add
            ? infer_bias_add_output_shape(n, inputs, outputs)
            : infer_elemwise_arithmetic_output_shape(n, inputs, outputs);

    return ret;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

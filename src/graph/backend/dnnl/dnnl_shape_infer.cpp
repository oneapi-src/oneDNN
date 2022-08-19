/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
    if (n->get_attr<std::string>(op_attr::dw_type) == "k3s2p1") {
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
    }

    set_shape_and_strides(*outputs[0], output_dims);
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

    // this permute is actually a transpose
    if (n->get_attr<std::string>(op_attr::permute_kind) == "transpose") {
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

    auto from_format = n->get_attr<std::string>(op_attr::from_format);
    auto to_format = n->get_attr<std::string>(op_attr::to_format);
    logical_tensor_t tmp;
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
    } else if (from_format == "OIX" && to_format == "XIO") {
        auto in_dims = ltw(inputs[0]).vdims();
        for (size_t i = 2; i < in_dims.size(); i++) { // X
            tmp_dims.emplace_back(in_dims[i]);
        }
        tmp_dims.emplace_back(in_dims[1]); // I
        tmp_dims.emplace_back(in_dims[0]); // O
    } else {
        assertm(false, "should not reach here");
        return status::unimplemented;
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

status_t infer_expand_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper_t;
    if (!ltw(outputs[0]).is_shape_unknown()) return status::success;

    auto axes = (n->has_attr(op_attr::axes))
            ? n->get_attr<std::vector<int64_t>>(op_attr::axes)
            : std::vector<int64_t>();
    const auto in_dims = ltw(inputs[0]).vdims();
    // if axes are present, we ignore other attributes,
    // otherwise, we calculate axes based on them
    if (axes.empty()) {
        const int64_t first = 0;
        const int64_t last = -1;
        if (n->has_attr(op_attr::insert_1dim)) {
            const auto insert_1dim
                    = n->get_attr<std::string>(op_attr::insert_1dim);
            if (insert_1dim == "before") {
                axes.push_back(first);
            } else if (insert_1dim == "after") {
                axes.push_back(last);
            }
        }
        if (n->has_attr(op_attr::expand_to)) {
            const auto target_ndims = n->get_attr<int64_t>(op_attr::expand_to);
            const size_t to_insert = static_cast<size_t>(target_ndims)
                    - in_dims.size() - axes.size();
            const size_t offset
                    = (!axes.empty() && axes.front() == first) ? 1 : 0;
            for (size_t i = 0; i < to_insert; ++i) {
                axes.push_back(i + offset);
            }
        }
    }

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
    auto diff_src_shape
            = n->get_attr<std::vector<int64_t>>(op_attr::input_shape);
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
            infer_auto_pad(src_sp[i], strides[i], kernel[i], dilations[i],
                    auto_pad, new_pads_begin[i], new_pads_end[i]);
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

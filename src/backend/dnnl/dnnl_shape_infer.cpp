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
#include "dnnl_shape_infer.hpp"
#include "interface/shape_infer.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

status_t infer_dnnl_conv_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = impl::logical_tensor_wrapper;

    auto backup = *inputs[1];
    auto out0 = logical_tensor_wrapper(outputs[0]);
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

    // permute output from NCX to NXC
    if (out_shape_unknown && n->has_attr("output_format")
            && n->get_attr<std::string>("output_format") == "NXC") {
        auto ndims = outputs[0]->ndims;
        auto channel = outputs[0]->dims[1];
        for (size_t i = 1; i < ndims - 1; i++) {
            outputs[0]->dims[i] = outputs[0]->dims[i + 1];
        }
        outputs[0]->dims[ndims - 1] = channel;
    }

    // set strides
    set_shape_and_strides(
            *outputs[0], logical_tensor_wrapper(outputs[0]).vdims());

    return status::success;
}

status_t infer_dnnl_pool_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    bool out_shape_unknown
            = logical_tensor_wrapper(outputs[0]).is_shape_unknown();
    infer_pool_output_shape(n, inputs, outputs);

    // permute output from NCX to NXC
    if (out_shape_unknown && n->has_attr("output_format")
            && n->get_attr<std::string>("output_format") == "NXC") {
        auto ndims = outputs[0]->ndims;
        auto channel = outputs[0]->dims[1];
        for (size_t i = 1; i < ndims - 1; i++) {
            outputs[0]->dims[i] = outputs[0]->dims[i + 1];
        }
        outputs[0]->dims[ndims - 1] = channel;
    }

    // set strides
    set_shape_and_strides(
            *outputs[0], logical_tensor_wrapper(outputs[0]).vdims());
    return status::success;
}

status_t infer_permute_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    using ltw = logical_tensor_wrapper;
    auto out0 = ltw(outputs[0]);
    if (!out0.is_shape_unknown()) return status::success;

    // this permute is actually a transpose
    if (n->get_attr<std::string>("permute_kind") == "transpose") {
        auto tmp = ltw(inputs[0]).vdims();
        auto rank = tmp.size();
        // swap the right-most two elements
        std::swap(tmp[rank - 2], tmp[rank - 1]);
        set_shape_and_strides(*outputs[0], tmp);
        return status::success;
    }

    auto from_format = n->get_attr<std::string>("from_format");
    auto to_format = n->get_attr<std::string>("to_format");
    if (from_format == "NCX" && to_format == "NXC") {
        // TODO(xxx)
        return status::unsupported;
    } else if (from_format == "NXC" && to_format == "NCX") {
        auto tmp = ltw(inputs[0]).reorder_data_dims_strides();
        set_shape_and_strides(*outputs[0], ltw(tmp).vdims());
    } else if (from_format == "XIO" && to_format == "OIX") {
        auto tmp = ltw(inputs[0]).reorder_weight_dims_strides();
        set_shape_and_strides(*outputs[0], ltw(tmp).vdims());
    } else {
        assertm(false, "should not reach here");
        return status::unsupported;
    }

    return status::success;
}

status_t infer_to_group_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper(outputs[0]);
    auto in0 = logical_tensor_wrapper(inputs[0]);
    if (!out0.is_shape_unknown()) return status::success;

    auto groups = n->get_attr<int64_t>("groups");
    dims in_dims = in0.vdims();

    in_dims[0] /= groups;
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
    using ltw = logical_tensor_wrapper;
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

status_t infer_bn_folding_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) {
    auto out0 = logical_tensor_wrapper(outputs[0]);
    auto out1 = logical_tensor_wrapper(outputs[1]);
    auto in0 = logical_tensor_wrapper(inputs[0]);
    auto in1 = logical_tensor_wrapper(inputs[1]);

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
    using ltw = impl::logical_tensor_wrapper;

    auto backup = *inputs[1];
    auto out0 = logical_tensor_wrapper(outputs[0]);
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

    // permute output from NCX to NXC
    if (out_shape_unknown && n->has_attr("output_format")
            && n->get_attr<std::string>("output_format") == "NXC") {
        auto ndims = outputs[0]->ndims;
        auto channel = outputs[0]->dims[1];
        for (size_t i = 1; i < ndims - 1; i++) {
            outputs[0]->dims[i] = outputs[0]->dims[i + 1];
        }
        outputs[0]->dims[ndims - 1] = channel;
    }

    // set strides
    set_shape_and_strides(
            *outputs[0], logical_tensor_wrapper(outputs[0]).vdims());

    return status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

/*******************************************************************************
* Copyright 2020-2023 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_convolution_utils.hpp"
#include "common/utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_convolution_utils {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::alg_kind;
using namespace prop_kind;
using namespace data_type;
using uint = unsigned int;

status_t acl_init_conf(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr) {

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper wei_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bia_d(&bias_md);

    // Compute Library currently supports forward propagation only
    const prop_kind_t prop_kind = cd.prop_kind;
    const bool is_fwd = (prop_kind == dnnl_forward_training)
            || (prop_kind == dnnl_forward_inference);
    if (!is_fwd) return status::unimplemented;

    const int ndims = src_d.ndims();
    const bool is_depthwise = wei_d.ndims() == 5 && wei_d.dims()[1] == 1
            && wei_d.dims()[2] == 1;

    ACL_CHECK_SUPPORT(
            ndims != 4 && !is_depthwise, " only supports 2 spatial dimensions");

    const int with_groups = wei_d.ndims() == src_d.ndims() + 1;
    ACL_CHECK_SUPPORT(with_groups && !is_depthwise, " does not support groups");

    ACL_CHECK_SUPPORT(!one_of(true,
                              everyone_is(data_type::f32, src_d.data_type(),
                                      wei_d.data_type(), dst_d.data_type()),
                              everyone_is(data_type::f16, src_d.data_type(),
                                      wei_d.data_type(), dst_d.data_type())),
            " src, dst and wei must be fp16 or fp32");

    // batch size
    const int mb = src_d.dims()[0];

    // src/input  channels, height, width
    const int ic = src_d.dims()[1];
    const int ih = src_d.dims()[ndims - 2];
    const int iw = src_d.dims()[ndims - 1];

    // dst/output channels, height, width
    const int oc = dst_d.dims()[1];
    const int oh = dst_d.dims()[ndims - 2];
    const int ow = dst_d.dims()[ndims - 1];

    // weights height and width
    const int kh = wei_d.dims()[with_groups + ndims - 2];
    const int kw = wei_d.dims()[with_groups + ndims - 1];

    // height and width strides
    const int stride_h = cd.strides[ndims - 4];
    const int stride_w = cd.strides[ndims - 3];

    // height and width dilations
    int dilate_h = cd.dilates[ndims - 4];
    int dilate_w = cd.dilates[ndims - 3];
    // oneDNN dilations:          dk = 1 + (k_size - 1) * (dilate_size + 1)
    // Compute Library dilations: dk = dilate_size * (k_size - 1) + 1
    // thus acl_dilation = oneDNN_dilation + 1
    dilate_h += 1;
    dilate_w += 1;

    acp.dilation_info = arm_compute::Size2D(dilate_w, dilate_h);

    // left, right, top, bottom padding
    const int l_pad = cd.padding[0][1];
    const int t_pad = cd.padding[0][0];
    // Compute Library assumes the padding to be \geq 0, and r(b)_pad may be
    // equal to -1 in oneDNN for some cases, when the very right (bottom)
    // spatial elements of the input tensor are not used in the convolution.
    // On the other hand l(t)_pad are guaranteed to be non-negative.
    const int r_pad = std::max(static_cast<int>(cd.padding[1][1]), 0);
    const int b_pad = std::max(static_cast<int>(cd.padding[1][0]), 0);

    if (is_depthwise
            && (t_pad >= kh || b_pad >= kh || l_pad >= kw || r_pad >= kw))
        return status::unimplemented;

    acp.padstride_info = arm_compute::PadStrideInfo(stride_w, stride_h,
            static_cast<unsigned int>(l_pad), static_cast<unsigned int>(r_pad),
            static_cast<unsigned int>(t_pad), static_cast<unsigned int>(b_pad),
            arm_compute::DimensionRoundingType::FLOOR);

    acp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (wei_d.format_kind() != format_kind::any && !is_depthwise)
        return status::unimplemented;

    auto src_tag = memory_desc_matches_one_of_tag(
            src_md, format_tag::nhwc, format_tag::nchw);
    auto dst_tag = memory_desc_matches_one_of_tag(
            dst_md, format_tag::nhwc, format_tag::nchw);

    // We want src and dst to match, preferrably both to be NHWC
    if (src_d.format_kind() == format_kind::any
            && dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, format_tag::nhwc));
        CHECK(memory_desc_init_by_tag(dst_md, format_tag::nhwc));
    } else if (src_d.format_kind() == format_kind::any
            && dst_tag != format_tag::undef) {
        CHECK(memory_desc_init_by_tag(src_md, dst_tag));
    } else if (dst_d.format_kind() == format_kind::any
            && src_tag != format_tag::undef) {
        CHECK(memory_desc_init_by_tag(dst_md, src_tag));
    }

    // Recompute tags after potentially running memory desc init
    src_tag = memory_desc_matches_one_of_tag(
            src_md, format_tag::nhwc, format_tag::nchw);
    dst_tag = memory_desc_matches_one_of_tag(
            dst_md, format_tag::nhwc, format_tag::nchw);

    if (src_tag == format_tag::undef || dst_tag == format_tag::undef
            || src_tag != dst_tag)
        return status::unimplemented;

    if (is_depthwise) {
        CHECK(memory_desc_init_by_tag(weights_md, format_tag::hwigo));
    } else {
        // Set weights to initially be the same as src
        CHECK(memory_desc_init_by_tag(weights_md, src_tag));
    }

    // Bias is just 1D, set to be the obvious format
    if (acp.with_bias && bias_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));

    bool is_nhwc = src_tag == format_tag::nhwc;
    if (!is_nhwc && is_depthwise) { return status::unimplemented; }
    // The layouts have to match (although we may later modify the weights)
    const auto acl_layout = is_nhwc ? arm_compute::DataLayout::NHWC
                                    : arm_compute::DataLayout::NCHW;

    // all have the same datatype
    auto acl_data_type = acl_utils::get_acl_data_t(src_d.data_type());

    // clang-format off
    acp.src_tensor_info = arm_compute::TensorInfo(
            is_nhwc ? arm_compute::TensorShape(ic, iw, ih, mb) :
            arm_compute::TensorShape(iw, ih, ic, mb),
            1,
            acl_data_type,
            acl_layout);

    acp.wei_tensor_info = arm_compute::TensorInfo(
            is_nhwc ? arm_compute::TensorShape(ic, kw, kh, oc) :
            arm_compute::TensorShape(kw, kh, ic, oc),
            1,
            acl_data_type,
            acl_layout);
    if(is_depthwise) {
       // We need to set that values are not constant so that we
       // we can update them in-place in ACL
      acp.wei_tensor_info.set_are_values_constant(false);
    }

    acp.dst_tensor_info = arm_compute::TensorInfo(
            is_nhwc ? arm_compute::TensorShape(oc, ow, oh, mb) :
            arm_compute::TensorShape(ow, oh, oc, mb),
            1,
            acl_data_type,
            acl_layout);

    acp.bia_tensor_info = arm_compute::TensorInfo(
            acp.with_bias ? arm_compute::TensorShape(oc)
                          : arm_compute::TensorShape(),
            1,
            acl_data_type,
            acl_layout);
    // clang-format on

    // Are we allowed to cast down to bf16 or not?
    acp.fast_math
            = one_of(attr.fpmath_mode_, fpmath_mode::bf16, fpmath_mode::any);
    if (is_depthwise) {
        // There is no support for fixed format kernels for depthwise convolution
        // in ACL so we are going to use weight format that we set up earlier
        return status::success;
    }

    // WeightFormat::ANY tells ACL we can handle any format
    acp.weights_info = arm_compute::WeightsInfo(
            false, kw, kh, oc, false, arm_compute::WeightFormat::ANY);

    // Get the format that the ACL kernel will expect the weights to be
    // in (if a kernel exists). Note that these are referred to as fixed format
    // kernels, because they require one specific weights format
    arm_compute::WeightFormat expected_weight_format;
    ACL_CHECK_VALID(arm_compute::NEGEMMConvolutionLayer::has_opt_impl(
            expected_weight_format, &acp.src_tensor_info, &acp.wei_tensor_info,
            acp.with_bias ? &acp.bia_tensor_info : nullptr,
            &acp.dst_tensor_info, acp.padstride_info, acp.weights_info,
            acp.dilation_info, acp.act_info, acp.fast_math));

    // Set weights info to the one returned by has_opt_impl
    acp.weights_info.set_weight_format(expected_weight_format);

    // has_opt_impl may return a non fast math kernel, even if we requested one
    acp.fast_math
            = arm_compute::is_fixed_format_fast_math(expected_weight_format);

    // Map OIHW used in ACL WeightFormat to the logical dimensions of the memory descriptor
    dim_t O_dim = 0;
    dim_t I_dim = 1;
    dim_t H_dim = 2;
    dim_t W_dim = 3;

    if (!is_nhwc) {
        // We can try to support NCHW by swapping IHW around, note that this
        // requires weights_md.dims[I_dim] % block_by != 0 (see next block)
        O_dim = 0;
        I_dim = 3;
        H_dim = 1;
        W_dim = 2;
    }

    // We can't currently support nchw and block_by != 1. If this is the case,
    // try a non fast math kernel, which currently have no blocking
    int block_by = arm_compute::block_by(acp.weights_info.weight_format());
    if (!is_nhwc && weights_md.dims[I_dim] % block_by != 0 && acp.fast_math) {
        acp.fast_math = false;
        acp.weights_info.set_weight_format(arm_compute::WeightFormat::ANY);
        ACL_CHECK_VALID(arm_compute::NEGEMMConvolutionLayer::has_opt_impl(
                expected_weight_format, &acp.src_tensor_info,
                &acp.wei_tensor_info,
                acp.with_bias ? &acp.bia_tensor_info : nullptr,
                &acp.dst_tensor_info, acp.padstride_info, acp.weights_info,
                acp.dilation_info, acp.act_info, acp.fast_math));
        acp.weights_info.set_weight_format(expected_weight_format);
        block_by = arm_compute::block_by(expected_weight_format);
        // This shouldn't happen, because non-fastmath have no blocking, but
        // guard against it because it would silently return incorrect results
        if (weights_md.dims[I_dim] % block_by != 0)
            return status::unimplemented;
    }

    acl_utils::reorder_to_weight_format(acp.wei_tensor_info, weights_md,
            expected_weight_format, I_dim, O_dim, {W_dim, H_dim}, {});

    return status::success;
}

status_t init_conf_gemm(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr) {
    if (weights_md.ndims != 4) return status::unimplemented;

    // General Compute Library checks, memory tags are also set there
    CHECK(acl_init_conf(acp, src_md, weights_md, dst_md, bias_md, cd, attr));

    // clang-format off
    // Validate convolution manually to check for return status
    ACL_CHECK_VALID(arm_compute::NEGEMMConvolutionLayer::validate(
        &acp.src_tensor_info,
        &acp.wei_tensor_info,
        acp.with_bias ? &acp.bia_tensor_info : nullptr,
        &acp.dst_tensor_info,
        acp.padstride_info,
        acp.weights_info,
        acp.dilation_info,
        acp.act_info,
        acp.fast_math));
    // clang-format on

    return status::success;
}

status_t init_conf_indirect_gemm(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr) {
    if (weights_md.ndims != 4) return status::unimplemented;

    // Indirect is slower for small convolution kernels
    if (weights_md.dims[2] == 1 && weights_md.dims[3] == 1)
        return status::unimplemented;

    CHECK(acl_init_conf(acp, src_md, weights_md, dst_md, bias_md, cd, attr));

    // Indirect is slower than gemm for low thread counts, except for fast math
    if (dnnl_get_max_threads() < 28 && !acp.fast_math)
        return status::unimplemented;

    // If we do not need to pad input channels for fast math mode then it would
    // be faster to run convolution with im2row instead of using indirect kernel
    int block_by = arm_compute::block_by(acp.weights_info.weight_format());
    int ic = src_md.dims[1];
    if (acp.fast_math && ic % block_by == 0) return status::unimplemented;

    // clang-format off
    // NOTE: indirect convolution method supports only nhwc layout.
    ACL_CHECK_VALID(arm_compute::NEGEMMConv2d::validate(
        &acp.src_tensor_info,
        &acp.wei_tensor_info,
        acp.with_bias ? &acp.bia_tensor_info : nullptr,
        &acp.dst_tensor_info,
        arm_compute::Conv2dInfo(acp.padstride_info,
                                acp.dilation_info,
                                acp.act_info,
                                acp.fast_math,
                                1, {}, acp.weights_info)));
    // clang-format on

    return status::success;
}

status_t init_conf_depthwise(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr) {
    if (weights_md.ndims != 5) return status::unimplemented;

    CHECK(acl_init_conf(acp, src_md, weights_md, dst_md, bias_md, cd, attr));

    ACL_CHECK_VALID(arm_compute::NEDepthwiseConvolutionLayer::validate(
            &acp.src_tensor_info, &acp.wei_tensor_info,
            acp.with_bias ? &acp.bia_tensor_info : nullptr,
            &acp.dst_tensor_info, acp.padstride_info,
            1, // depth multiplier default value
            acp.act_info, acp.dilation_info));

    return status::success;
}

} // namespace acl_convolution_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

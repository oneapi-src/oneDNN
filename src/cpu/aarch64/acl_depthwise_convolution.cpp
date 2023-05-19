/*******************************************************************************
* Copyright 2023 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_depthwise_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

status_t acl_depthwise_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    std::lock_guard<std::mutex> _lock {this->mtx};

    auto *acl_resource
            = ctx.get_resource_mapper()
                      ->get<acl_depthwise_convolution_resource_t>(this);
    acl_obj_t<arm_compute::NEDepthwiseConvolutionLayer> &acl_depthwise_obj
            = acl_resource->get_acl_obj();

    return execute_forward_conv_acl<
            acl_obj_t<arm_compute::NEDepthwiseConvolutionLayer>, pd_t, data_t>(
            ctx, acl_depthwise_obj, pd());
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

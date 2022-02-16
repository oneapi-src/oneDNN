/*******************************************************************************
* Copyright 2021-2022 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_common_utils {

using namespace dnnl::impl::alg_kind;
using namespace data_type;

arm_compute::DataType get_acl_data_t(const dnnl_data_type_t dt) {
    switch (dt) {
        case bf16: return arm_compute::DataType::BFLOAT16; break;
        case f32: return arm_compute::DataType::F32; break;
        case s32: return arm_compute::DataType::S32; break;
        case f16: return arm_compute::DataType::F16; break;
        case s8: return arm_compute::DataType::QASYMM8_SIGNED; break;
        case u8: return arm_compute::DataType::QASYMM8; break;
        default: return arm_compute::DataType::UNKNOWN;
    }
}

arm_compute::ActivationLayerInfo convert_to_acl_act(
        const alg_kind_t eltwise_alg, const float alpha, const float beta) {
    using acl_act_t = arm_compute::ActivationLayerInfo::ActivationFunction;
    acl_act_t acl_act_alg;

    switch (eltwise_alg) {
        case eltwise_relu:
            // oneDNN defines RELU: f(x) = (x > 0) ? x : a*x
            // Compute Library defines LEAKY_RELU: f(x) = (x > 0) ? x : a*x
            // whilst Compute Library RELU is defined as: f(x) = max(0,x)
            if (alpha == 0) {
                acl_act_alg = acl_act_t::RELU;
            } else {
                acl_act_alg = acl_act_t::LEAKY_RELU;
            }
            break;
        case eltwise_tanh:
            // oneDNN defines TANH activation as:          f(x) = tanh(x)
            // Compute Library defines TANH activation as: f(x) = a*tanh(b*x)
            // Setting a=b=1 makes the two equivalent
            return arm_compute::ActivationLayerInfo(acl_act_t::TANH, 1.f, 1.f);
            break;
        case eltwise_elu: acl_act_alg = acl_act_t::ELU; break;
        case eltwise_square: acl_act_alg = acl_act_t::SQUARE; break;
        case eltwise_abs: acl_act_alg = acl_act_t::ABS; break;
        case eltwise_sqrt: acl_act_alg = acl_act_t::SQRT; break;
        case eltwise_linear: acl_act_alg = acl_act_t::LINEAR; break;
        case eltwise_bounded_relu: acl_act_alg = acl_act_t::BOUNDED_RELU; break;
        case eltwise_soft_relu: acl_act_alg = acl_act_t::SOFT_RELU; break;
        case eltwise_logistic: acl_act_alg = acl_act_t::LOGISTIC; break;
        default: return arm_compute::ActivationLayerInfo();
    }

    return arm_compute::ActivationLayerInfo(acl_act_alg, alpha, beta);
}

arm_compute::ActivationLayerInfo get_acl_act(const primitive_attr_t &attr) {
    const auto &post_ops = attr.post_ops_;
    const int entry_idx = post_ops.find(primitive_kind::eltwise);
    if (entry_idx == -1) { return arm_compute::ActivationLayerInfo(); }

    const auto eltwise_alg = post_ops.entry_[entry_idx].eltwise.alg;
    float alpha = post_ops.entry_[entry_idx].eltwise.alpha;
    float beta = post_ops.entry_[entry_idx].eltwise.beta;

    return convert_to_acl_act(eltwise_alg, alpha, beta);
}

arm_compute::ActivationLayerInfo get_acl_act(const eltwise_desc_t &ed) {
    const alg_kind_t eltwise_alg = ed.alg_kind;
    float alpha = ed.alpha;
    float beta = ed.beta;

    return convert_to_acl_act(eltwise_alg, alpha, beta);
}

bool acl_act_ok(alg_kind_t eltwise_activation) {
    return utils::one_of(eltwise_activation, eltwise_relu, eltwise_tanh,
            eltwise_elu, eltwise_square, eltwise_abs, eltwise_sqrt,
            eltwise_linear, eltwise_bounded_relu, eltwise_soft_relu,
            eltwise_logistic);
}

void acl_thread_bind() {
    static std::once_flag flag_once;
    // The threads in Compute Library are bound for the cores 0..max_threads-1
    // dnnl_get_max_threads() returns OMP_NUM_THREADS
    const int max_threads = dnnl_get_max_threads();
    // arm_compute::Scheduler does not support concurrent access thus a
    // workaround here restricts it to only one call
    std::call_once(flag_once, [&]() {
        arm_compute::Scheduler::get().set_num_threads(max_threads);
    });
}

status_t tensor_info(arm_compute::TensorInfo &info, const memory_desc_t &md) {
    const memory_desc_wrapper md_wrap(&md);
    return tensor_info(info, md_wrap);
}

status_t tensor_info(
        arm_compute::TensorInfo &info, const memory_desc_wrapper &md) {

    // All the cases we don't support
    if (!md.is_blocking_desc() || !md.is_dense() || !md.is_plain()
            || md.has_zero_dim())
        return status::unimplemented;

    // Set each of the dimensions in the TensorShape from the memory desc
    // ACL indexes dimensions the opposite way to oneDNN
    arm_compute::TensorShape shape;
    size_t acl_dim_i = 0;
    for (int i = md.ndims() - 1; i >= 0; --i) {
        shape.set(acl_dim_i, md.dims()[i]);
        acl_dim_i++;
    }

    // Set each of the ACL Strides from the memory blocking desc
    // ACL indexes strides the opposite way to oneDNN
    arm_compute::Strides strides_in_bytes;
    const blocking_desc_t &blocking_desc = md.blocking_desc();
    size_t acl_stride_i = 0;
    for (int i = md.ndims() - 1; i >= 0; --i) {
        // ACL strides are in bytes, oneDNN strides are in numbers of elements,
        // multiply by data type size to convert
        strides_in_bytes.set(
                acl_stride_i, blocking_desc.strides[i] * md.data_type_size());
        ++acl_stride_i;
    }

    arm_compute::DataType data_type = get_acl_data_t(md.data_type());
    size_t num_channels = 1;
    size_t offset_first_element_in_bytes = 0;
    size_t total_size_in_bytes = md.size();

    info.init(shape, num_channels, data_type, strides_in_bytes,
            offset_first_element_in_bytes, total_size_in_bytes);

    return status::success;
}

status_t insert_singleton_dimension(arm_compute::TensorInfo &ti, size_t dim_i) {

    // Max 6 dims in ACL, so we can't insert another
    if (ti.num_dimensions() >= 6) return status::unimplemented;

    // Copy dimensions from old to new shape, inserting a dimension of size 1
    arm_compute::TensorShape shape = ti.tensor_shape();
    for (size_t old_i = 0, new_i = 0; old_i < ti.num_dimensions(); ++old_i) {
        if (old_i == dim_i) {
            shape.set(new_i, 1, false);
            ++new_i;
        }
        shape.set(new_i, ti.tensor_shape()[old_i], false);
        ++new_i;
    }

    // Copy strides from old to new tensor, inserting a duplicate stride
    arm_compute::Strides strides;
    for (size_t old_i = 0, new_i = 0; old_i < ti.num_dimensions(); ++old_i) {
        if (old_i == dim_i) {
            strides.set(new_i, ti.strides_in_bytes()[old_i], false);
            ++new_i;
        }
        strides.set(new_i, ti.strides_in_bytes()[old_i], false);
        ++new_i;
    }

    // Reinit TensorInfo with modified shape and strides
    ti.init(shape, ti.num_channels(), ti.data_type(), strides,
            ti.offset_first_element_in_bytes(), ti.total_size());

    return status::success;
}

status_t permute_common_dense_dimension_to_last(memory_desc_t *d0_permed,
        memory_desc_t *d1_permed, memory_desc_t *d2_permed,
        const memory_desc_t *d0, const memory_desc_t *d1,
        const memory_desc_t *d2) {

    // Number of dimensions must match
    int ndims = d0->ndims;
    if (ndims != d1->ndims || ndims != d2->ndims) return status::unimplemented;

    if (d0->format_kind != format_kind::blocked
            || d1->format_kind != format_kind::blocked
            || d2->format_kind != format_kind::blocked)
        return status::unimplemented;

    const dnnl_dims_t &d0_strides = d0->format_desc.blocking.strides;
    const dnnl_dims_t &d1_strides = d1->format_desc.blocking.strides;
    const dnnl_dims_t &d2_strides = d2->format_desc.blocking.strides;

    int inner_dim = ndims - 1;

    // descs already share a common dense axis, no need to permute, just copy
    // By dense we mean that it has a stride of 1
    if (d0_strides[inner_dim] == 1 && d1_strides[inner_dim] == 1
            && d2_strides[inner_dim] == 1) {
        *d0_permed = *d0;
        *d1_permed = *d1;
        *d2_permed = *d2;
        return status::success;
    }

    // Create permutation which swaps nothing
    std::vector<int> perm(ndims);
    for (int i = inner_dim; i >= 0; --i) {
        perm[i] = i;
    }

    // Look for the innermost common dense axis
    for (int i = inner_dim; i >= 0; --i) {
        if (d0_strides[i] == 1 && d1_strides[i] == 1 && d2_strides[i] == 1) {
            // We have found it! Swap this dimension with inner one
            perm[i] = inner_dim;
            perm[inner_dim] = i;
            break;
        }
        // Got to the outermost dimension without finding a common dense axis
        if (i == 0) return status::unimplemented;
    }

    dnnl_memory_desc_permute_axes(d0_permed, d0, perm.data());
    dnnl_memory_desc_permute_axes(d1_permed, d1, perm.data());
    dnnl_memory_desc_permute_axes(d2_permed, d2, perm.data());
    return status::success;
}

} // namespace acl_common_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

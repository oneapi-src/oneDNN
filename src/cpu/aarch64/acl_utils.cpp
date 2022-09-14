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

namespace acl_utils {

using namespace dnnl::impl::alg_kind;
using namespace data_type;

arm_compute::DataType get_acl_data_t(
        const dnnl_data_type_t dt, const bool is_quantized) {
    switch (dt) {
        case bf16: return arm_compute::DataType::BFLOAT16;
        case f32: return arm_compute::DataType::F32;
        case s32: return arm_compute::DataType::S32;
        case f16: return arm_compute::DataType::F16;
        case s8:
            if (is_quantized)
                return arm_compute::DataType::QASYMM8_SIGNED;
            else
                return arm_compute::DataType::S8;
        case u8:
            if (is_quantized)
                return arm_compute::DataType::QASYMM8;
            else
                return arm_compute::DataType::U8;
        default: return arm_compute::DataType::UNKNOWN;
    }
}

status_t convert_to_acl_act(alg_kind_t eltwise_alg, float alpha, float beta,
        arm_compute::ActivationLayerInfo &act_info) {

    using namespace arm_compute;
    using act_func = ActivationLayerInfo::ActivationFunction;

    switch (eltwise_alg) {
        case eltwise_relu:
            // oneDNN defines RELU: f(x) = (x > 0) ? x : a*x
            // Compute Library defines LEAKY_RELU: f(x) = (x > 0) ? x : a*x
            // whilst Compute Library RELU is defined as: f(x) = max(0,x)
            if (alpha == 0) {
                act_info = ActivationLayerInfo(act_func::RELU, alpha, beta);
            } else {
                act_info = ActivationLayerInfo(
                        act_func::LEAKY_RELU, alpha, beta);
            }
            break;
        case eltwise_tanh:
            // oneDNN defines TANH activation as:          f(x) = tanh(x)
            // Compute Library defines TANH activation as: f(x) = a*tanh(b*x)
            // Setting a=b=1 makes the two equivalent
            act_info = ActivationLayerInfo(act_func::TANH, 1.f, 1.f);
            break;
        case eltwise_elu:
            act_info = ActivationLayerInfo(act_func::ELU, alpha, beta);
            break;
        case eltwise_square:
            act_info = ActivationLayerInfo(act_func::SQUARE, alpha, beta);
            break;
        case eltwise_abs:
            act_info = ActivationLayerInfo(act_func::ABS, alpha, beta);
            break;
        case eltwise_sqrt:
            act_info = ActivationLayerInfo(act_func::SQRT, alpha, beta);
            break;
        case eltwise_linear:
            act_info = ActivationLayerInfo(act_func::LINEAR, alpha, beta);
            break;
        case eltwise_soft_relu:
            act_info = ActivationLayerInfo(act_func::SOFT_RELU, alpha, beta);
            break;
        case eltwise_logistic:
            act_info = ActivationLayerInfo(act_func::LOGISTIC, alpha, beta);
            break;
        default: act_info = ActivationLayerInfo(); return status::unimplemented;
    }

    return status::success;
}

status_t convert_to_acl_act(
        const eltwise_desc_t &ed, arm_compute::ActivationLayerInfo &act_info) {
    return convert_to_acl_act(ed.alg_kind, ed.alpha, ed.beta, act_info);
}

status_t convert_to_acl_act(const post_ops_t::entry_t::eltwise_t &elt,
        arm_compute::ActivationLayerInfo &act_info) {
    return convert_to_acl_act(elt.alg, elt.alpha, elt.beta, act_info);
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

int reorder_dimensions_by_stride(std::vector<memory_desc_t *> permuted_mds,
        std::vector<const memory_desc_t *> mds) {

    // Vectors must be the same length and not empty
    if (permuted_mds.size() != mds.size() || mds.empty()) return 0;

    const dim_t ndims = mds[0]->ndims;

    for (const auto &md : mds) {
        // Number of dimensions must match and must be blocked
        if (md->ndims != ndims || md->format_kind != format_kind::blocked)
            return 0;
    }

    int reordered_dims = 0;

    // Create initial permutation which swaps nothing
    std::vector<int> perm(ndims);
    std::iota(perm.begin(), perm.end(), 0);

    // For each dimension d1, find a dimension (d2) in which every md has the
    // next smallest stride, then swap d2 into d1. Stride is initially 1 (i.e.
    // dense) but will increase each time we find a dimension. The target
    // strides may be different across dimensions if they are broadcasted.
    std::vector<dim_t> next_smallest_stride(mds.size(), 1);
    for (dim_t d1 = ndims - 1; d1 >= 0; --d1) {
        bool found_swap = false;
        for (dim_t d2 = d1; d2 >= 0; --d2) {
            // Check that all mds have the right stride
            found_swap = true;
            for (size_t i = 0; i < mds.size(); i++) {
                auto &md_strides = mds[i]->format_desc.blocking.strides;
                // Either it is the next smallest stride, or the dimensions is 1
                // so we can ignore it
                bool can_swap = md_strides[perm[d2]] == next_smallest_stride[i]
                        || mds[i]->dims[perm[d2]] == 1;
                if (!can_swap) {
                    found_swap = false;
                    break;
                }
            }
            if (found_swap) {
                // Multiply next smallest strides by dimension we just found
                for (size_t i = 0; i < mds.size(); i++)
                    next_smallest_stride[i] *= mds[i]->dims[perm[d2]];

                // Swap the found dimension (perm[d2]) into d1
                nstl::swap(perm[d2], perm[d1]);
                ++reordered_dims;
                break;
            }
        }
        // We didn't find a swap for this dimension, we can't continue
        if (!found_swap) break;
    }

    // dnnl_memory_desc_permute_axes applies the inverse of the permutation
    // so we need to invert our permutation to get what we want
    std::vector<int> invperm(ndims);
    for (dim_t d = 0; d < ndims; ++d)
        invperm[perm[d]] = d;

    // Apply the inverse permutation to each dimension axis
    for (size_t i = 0; i < mds.size(); i++) {
        dnnl_memory_desc_permute_axes(permuted_mds[i], mds[i], invperm.data());
    }

    return reordered_dims;
}

} // namespace acl_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

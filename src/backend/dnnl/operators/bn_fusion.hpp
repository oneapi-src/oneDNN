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

#ifndef BACKEND_DNNL_OPERATORS_BN_FUSION_HPP
#define BACKEND_DNNL_OPERATORS_BN_FUSION_HPP

#include <functional>

#include "backend/dnnl/utils.hpp"
#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#include <dnnl_sycl.hpp>
#endif

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

/// Convolution/MatMul + BatchNormalization fusion
/// BatchNormalization can be decribed as the following formula:
///   dst(n,c,h,w) = \gamma(c) \cdot
///       \frac{src(n,c,h,w) - \miu(c)}{\sqrt{\sigma^2(c) + \epsilon}}
///       + \beta(c)
/// Substituting src(n,c,h,w) by Convolution/MatMul operator $W(c) \ cdot
/// x + b(c)$
/// derives:
///   dst(n,c,h,w) = \alpha(c) \cdot W(c) \cdot x
///       + \alpha(c) \cdot (b - \miu(c)) + \bata(c)
/// where $\alpha(c) = \frac{src(n,c,h,w) - \miu(c)}
/// {\sqrt{\sigma^2(c) + \epsilon}}$. Hence, BatchNormalization can be fused
/// into a new Convolution/MatMul operator, where params can be derived from
/// origin:   W'_c = \alpha(c) \cdot W(c)
/// and
///   b'_c = \alpha(c) \cdot (b(c) - \miu(c)) + \beta(c)
///
/// The computing kernel is inheritated from convolution_forward/matmul_forward
/// operator. See convolution_forward/matmul_forward for more info.
struct bn_fusion {
    /// Fold BatchNormalization states into Convolution weights
    ///
    /// @param updated_weights The tensor pointer of folded weights to be
    ///     returned.
    /// @param updated_bias The tensor pointer of folded bias to be returned.
    /// @param weights The original convolution weights.
    /// @param bias The original convolution bias, can be an empty tensor.
    /// @param mean The mean state of BatchNormalization.
    /// @param variance The variance state of BatchNormalizaiton.
    /// @param scale Scale value of BatchNormalization.
    /// @param shift Shift value of BatchNormalization.
    /// @param epsilon A constant to improve numerical stability.
    static void folding(tensor *updated_weights, tensor *updated_bias,
            const tensor &weights, const tensor &bias, const tensor &mean,
            const tensor &variance, const tensor &scale, const tensor &shift,
            float epsilon, const impl::stream_t &g_stream) {
        const data_type weights_dtype = weights.get_data_type();
#ifdef DNNL_GRAPH_WITH_SYCL
        BACKEND_DNNL_TYPE_DISPATCH(weights_dtype, dtype, {
            folding_sycl_impl<dtype>(updated_weights, updated_bias, weights,
                    bias, mean, variance, scale, shift, epsilon, g_stream);
        });
#else
        UNUSED(g_stream);
        BACKEND_DNNL_TYPE_DISPATCH(weights_dtype, dtype, {
            folding_impl<dtype>(updated_weights, updated_bias, weights, bias,
                    mean, variance, scale, shift, epsilon);
        });
#endif
    }

private:
    template <typename dtype>
    static void folding_impl(tensor *updated_weights, tensor *updated_bias,
            const tensor &weights, const tensor &bias, const tensor &mean,
            const tensor &variance, const tensor &scale, const tensor &shift,
            float epsilon) {
        const size_t num_channel = static_cast<size_t>(mean.get_dim(0));
        dtype *weights_ptr = static_cast<dtype *>(weights.get_data_handle());
        dtype *bias_ptr = !bias.is_empty()
                ? static_cast<dtype *>(bias.get_data_handle())
                : nullptr;
        dtype *mean_ptr = static_cast<dtype *>(mean.get_data_handle());
        dtype *variance_ptr = static_cast<dtype *>(variance.get_data_handle());
        dtype *scale_ptr = static_cast<dtype *>(scale.get_data_handle());
        dtype *shift_ptr = static_cast<dtype *>(shift.get_data_handle());

        const dims weights_dims = weights.get_dims();
        // The first dimension of conv weights is related to output channel.
        const size_t volume_per_channel
                = static_cast<size_t>(std::accumulate(weights_dims.begin() + 1,
                        weights_dims.end(), 1, std::multiplies<dim>()));
        dtype *updated_weights_ptr
                = static_cast<dtype *>(updated_weights->get_data_handle());
        dtype *updated_bias_ptr
                = static_cast<dtype *>(updated_bias->get_data_handle());
        // todo(zixuanwe): Implement OpenMP in the future
        for (size_t c = 0; c < num_channel; ++c) {
            dtype alpha = scale_ptr[c]
                    / static_cast<dtype>(sqrt(variance_ptr[c] + epsilon));

            const dtype *nth_src = weights_ptr + c * volume_per_channel;
            dtype *nth_dst = updated_weights_ptr + c * volume_per_channel;
            for (size_t k = 0; k < volume_per_channel; ++k) {
                nth_dst[k] = alpha * nth_src[k];
            }

            if (bias_ptr) {
                updated_bias_ptr[c]
                        = alpha * (bias_ptr[c] - mean_ptr[c]) + shift_ptr[c];
            } else {
                updated_bias_ptr[c] = shift_ptr[c] - alpha * mean_ptr[c];
            }
        }
    }

#ifdef DNNL_GRAPH_WITH_SYCL
    template <typename dtype>
    static void folding_sycl_impl(tensor *updated_weights, tensor *updated_bias,
            const tensor &weights, const tensor &bias, const tensor &mean,
            const tensor &variance, const tensor &scale, const tensor &shift,
            float epsilon, const impl::stream_t &g_stream) {
        sycl::queue q = g_stream.get_queue();
        const size_t num_channel = static_cast<size_t>(mean.get_dim(0));
        dtype *weights_ptr = static_cast<dtype *>(weights.get_data_handle());
        dtype *bias_ptr = !bias.is_empty()
                ? static_cast<dtype *>(bias.get_data_handle())
                : nullptr;
        dtype *mean_ptr = static_cast<dtype *>(mean.get_data_handle());
        dtype *variance_ptr = static_cast<dtype *>(variance.get_data_handle());
        dtype *scale_ptr = static_cast<dtype *>(scale.get_data_handle());
        dtype *shift_ptr = static_cast<dtype *>(shift.get_data_handle());

        const dims weights_dims = weights.get_dims();
        // The first dimension of conv weights is related to output channel.
        const size_t volume_per_channel
                = static_cast<size_t>(std::accumulate(weights_dims.begin() + 1,
                        weights_dims.end(), 1, std::multiplies<dim>()));
        dtype *updated_weights_ptr
                = static_cast<dtype *>(updated_weights->get_data_handle());
        dtype *updated_bias_ptr
                = static_cast<dtype *>(updated_bias->get_data_handle());
        q.submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1> {num_channel}, [=](sycl::id<1> it) {
                const int i = it[0];
                dtype alpha
                        = scale_ptr[i] / sycl::sqrt(variance_ptr[i] + epsilon);

                auto nth_src = weights_ptr + i * volume_per_channel;
                auto nth_dst = updated_weights_ptr + i * volume_per_channel;

                for (size_t k = 0; k < volume_per_channel; ++k) {
                    nth_dst[k] = alpha * nth_src[k];
                }

                if (bias_ptr) {
                    updated_bias_ptr[i] = alpha * (bias_ptr[i] - mean_ptr[i])
                            + shift_ptr[i];
                } else {
                    updated_bias_ptr[i] = shift_ptr[i] - alpha * mean_ptr[i];
                }
            });
        });
    }
#endif
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif

/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

/**
 * This file is useful in static build to make sure the initializers are
 * referenced and prevent them from being removed by the linker.
 * */

#define RUN_ON_PRODUCTION(F) \
    F(matmul_core); \
    F(softmax); \
    F(static_reshape); \
    F(dynamic_reshape); \
    F(shape_of_tensor); \
    F(matmul); \
    F(quantize); \
    F(exp); \
    F(dynamic_transpose); \
    F(add); \
    F(padding); \
    F(reorder); \
    F(reduce); \
    F(reduce_sum); \
    F(reduce_prod); \
    F(reduce_max); \
    F(reduce_mean); \
    F(reduce_min); \
    F(relu); \
    F(relu_backprop); \
    F(sigmoid_backprop); \
    F(conv_fwd); \
    F(conv_fwd_core); \
    F(conv_bwd_data); \
    F(conv_bwd_weight); \
    F(batchnorm_inference); \
    F(batchnorm_forward_training); \
    F(batchnorm_training_backprop); \
    F(managed_matmul_core); \
    F(select); \
    F(layernorm); \
    F(gelu); \
    F(abs); \
    F(elu); \
    F(hardswish); \
    F(log); \
    F(pow); \
    F(mish); \
    F(soft_plus); \
    F(square); \
    F(swish); \
    F(prelu); \
    F(hardsigmoid); \
    F(clamp); \
    F(leaky_relu); \
    F(duplicate); \
    F(pooling_avg); \
    F(pooling_max); \
    F(broadcast);

#define DECL_INIT(NAME) extern volatile bool __help_dummy_##NAME;

#define RUN_ON_REFLECTION_CLASS(F) F(shared_ptr_static_data)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
RUN_ON_PRODUCTION(DECL_INIT)

#define DECL_REFLECTION_INIT(NAME) extern void *__reflection_init_##NAME;
namespace reflection {
RUN_ON_REFLECTION_CLASS(DECL_REFLECTION_INIT)
}

void __dummy_init() {
#define REF_INIT(NAME) (void)__help_dummy_##NAME;
    RUN_ON_PRODUCTION(REF_INIT)

#define REF_REFLECTION_INIT(NAME) (void)reflection::__reflection_init_##NAME;
    RUN_ON_REFLECTION_CLASS(REF_REFLECTION_INIT)
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

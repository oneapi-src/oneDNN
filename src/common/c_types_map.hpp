/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef TYPE_MAPPING_HPP
#define TYPE_MAPPING_HPP

#include "mkldnn_types.h"

namespace mkldnn { namespace impl {

// TODO: autogenerate this

using nd_offset_t = mkldnn_nd_offset_t;
using dims_t = mkldnn_dims_t;

using status_t = mkldnn_status_t;
namespace status {
    const status_t success = mkldnn_success;
    const status_t out_of_memory = mkldnn_out_of_memory;
    const status_t try_again = mkldnn_try_again;
    const status_t invalid_arguments = mkldnn_invalid_arguments;
    const status_t not_ready = mkldnn_not_ready;
    const status_t unimplemented = mkldnn_unimplemented;
}

using prop_kind_t = mkldnn_prop_kind_t;
namespace prop_kind {
    const prop_kind_t forward_training = mkldnn_forward_training;
    const prop_kind_t forward_scoring = mkldnn_forward_scoring;
    const prop_kind_t forward = mkldnn_forward;
    const prop_kind_t backward_data = mkldnn_backward_data;
    const prop_kind_t backward_weights = mkldnn_backward_weights;
    const prop_kind_t backward_bias = mkldnn_backward_bias;
}

using alg_kind_t = mkldnn_alg_kind_t;
namespace alg_kind {
    const alg_kind_t convolution_direct = mkldnn_convolution_direct;
    const alg_kind_t pooling_max = mkldnn_pooling_max;
    const alg_kind_t lrn_across_channels = mkldnn_lrn_across_channels;
    const alg_kind_t lrn_within_channel = mkldnn_lrn_within_channel;
}

using tensor_desc_t = mkldnn_tensor_desc_t;

using precision_t = mkldnn_precision_t;
namespace precision {
    const precision_t undef = mkldnn_precision_undef;
    const precision_t f32 = mkldnn_f32;
    const precision_t u32 = mkldnn_u32;
}

using memory_format_t = mkldnn_memory_format_t;
namespace memory_format {
    const memory_format_t undef = mkldnn_format_undef;
    const memory_format_t any = mkldnn_any;
    const memory_format_t blocked = mkldnn_blocked;
    const memory_format_t x = mkldnn_x;
    const memory_format_t nc = mkldnn_nc;
    const memory_format_t nchw = mkldnn_nchw;
    const memory_format_t nhwc = mkldnn_nhwc;
    const memory_format_t nChw8c = mkldnn_nChw8c;
    const memory_format_t oi = mkldnn_oi;
    const memory_format_t oihw = mkldnn_oihw;
    const memory_format_t oIhw8i = mkldnn_oIhw8i;
    const memory_format_t OIhw8i8o = mkldnn_OIhw8i8o;
    const memory_format_t Ohwi8o = mkldnn_Ohwi8o;
    const memory_format_t goihw = mkldnn_goihw;
    const memory_format_t gOIhw8i8o = mkldnn_gOIhw8i8o;
}

using padding_kind_t = mkldnn_padding_kind_t;
namespace padding_kind {
    const padding_kind_t padding_zero = mkldnn_padding_zero;
}

using engine_kind_t = mkldnn_engine_kind_t;
namespace engine_kind {
    const engine_kind_t any_engine = mkldnn_any_engine;
    const engine_kind_t cpu = mkldnn_cpu;
    const engine_kind_t cpu_lazy = mkldnn_cpu_lazy;
}

using engine = mkldnn_engine;

using primitive_kind_t = mkldnn_primitive_kind_t;
namespace primitive_kind {
    const primitive_kind_t undefined = mkldnn_undefined_primitive;
    const primitive_kind_t memory = mkldnn_memory;
    const primitive_kind_t reorder = mkldnn_reorder;
    const primitive_kind_t convolution = mkldnn_convolution;
    const primitive_kind_t pooling = mkldnn_pooling;
    const primitive_kind_t relu = mkldnn_relu;
    const primitive_kind_t lrn = mkldnn_lrn;
    const primitive_kind_t inner_product = mkldnn_inner_product;
    const primitive_kind_t convolution_relu = mkldnn_convolution_relu;
}

using blocking_desc_t = mkldnn_blocking_desc_t;
using memory_desc_t = mkldnn_memory_desc_t;
using convolution_desc_t = mkldnn_convolution_desc_t;
using pooling_desc_t = mkldnn_pooling_desc_t;
using relu_desc_t = mkldnn_relu_desc_t;
using lrn_desc_t = mkldnn_lrn_desc_t;
using inner_product_desc_t = mkldnn_inner_product_desc_t;
using convolution_relu_desc_t = mkldnn_convolution_relu_desc_t;

struct op_desc_t {
    primitive_kind_t _kind;
    union {
        memory_desc_t memory;
        convolution_desc_t convolution;
        pooling_desc_t pooling;
        relu_desc_t relu;
        lrn_desc_t lrn;
        inner_product_desc_t inner_product;
        convolution_relu_desc_t convolution_relu;
    };

#   define DECL_CTOR_AND_CONVERTERS(c_type, name) \
    op_desc_t(const c_type &_): _kind(primitive_kind::name), name(_) {} \
    static op_desc_t *convert_from_c(c_type *_) \
    { return reinterpret_cast<op_desc_t*>(_); } \
    static const op_desc_t *convert_from_c(const c_type *_) \
    { return reinterpret_cast<const op_desc_t*>(_); }

    DECL_CTOR_AND_CONVERTERS(memory_desc_t, memory);
    DECL_CTOR_AND_CONVERTERS(convolution_desc_t, convolution);
    DECL_CTOR_AND_CONVERTERS(pooling_desc_t, pooling);
    DECL_CTOR_AND_CONVERTERS(relu_desc_t, relu);
    DECL_CTOR_AND_CONVERTERS(lrn_desc_t, lrn);
    DECL_CTOR_AND_CONVERTERS(inner_product_desc_t, inner_product);
    DECL_CTOR_AND_CONVERTERS(convolution_relu_desc_t, convolution_relu);

#   undef DECL_CTOR_AND_CONVERTERS
};

using primitive_base_desc_t = mkldnn_primitive_base_desc_t;
using memory_primitive_desc_t = mkldnn_memory_primitive_desc_t;
using reorder_primitive_desc_t = mkldnn_reorder_primitive_desc_t;
using convolution_primitive_desc_t = mkldnn_convolution_primitive_desc_t;
using pooling_primitive_desc_t = mkldnn_pooling_primitive_desc_t;
using relu_primitive_desc_t = mkldnn_relu_primitive_desc_t;
using lrn_primitive_desc_t = mkldnn_lrn_primitive_desc_t;
using inner_product_primitive_desc_t = mkldnn_inner_product_primitive_desc_t;
using convolution_relu_primitive_desc_t =
        mkldnn_convolution_relu_primitive_desc_t;

union primitive_desc_t {
    primitive_base_desc_t base;
    memory_primitive_desc_t memory;
    reorder_primitive_desc_t reorder;
    convolution_primitive_desc_t convolution;
    pooling_primitive_desc_t pooling;
    relu_primitive_desc_t relu;
    lrn_primitive_desc_t lrn;
    inner_product_primitive_desc_t inner_product;
    convolution_relu_primitive_desc_t convolution_relu;
#   define DECL_CTOR_AND_CONVERTERS(c_type, name) \
    primitive_desc_t(const c_type &_): name(_) {} \
    static primitive_desc_t *convert_from_c(c_type *_) \
    { return reinterpret_cast<primitive_desc_t*>(_); } \
    static const primitive_desc_t *convert_from_c(const c_type *_) \
    { return reinterpret_cast<const primitive_desc_t*>(_); } \

    DECL_CTOR_AND_CONVERTERS(primitive_base_desc_t, base);
    DECL_CTOR_AND_CONVERTERS(memory_primitive_desc_t, memory);
    DECL_CTOR_AND_CONVERTERS(reorder_primitive_desc_t, reorder);
    DECL_CTOR_AND_CONVERTERS(convolution_primitive_desc_t, convolution);
    DECL_CTOR_AND_CONVERTERS(pooling_primitive_desc_t, pooling);
    DECL_CTOR_AND_CONVERTERS(relu_primitive_desc_t, relu);
    DECL_CTOR_AND_CONVERTERS(lrn_primitive_desc_t, lrn);
    DECL_CTOR_AND_CONVERTERS(inner_product_primitive_desc_t, inner_product);
    DECL_CTOR_AND_CONVERTERS(convolution_relu_primitive_desc_t,
            convolution_relu);

#   undef DECL_CTOR_AND_CONVERTERS
};

using primitive = mkldnn_primitive;

using primitive_at_t = mkldnn_primitive_at_t;

using stream = mkldnn_stream;

}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

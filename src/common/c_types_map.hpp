#ifndef TYPE_MAPPING_HPP
#define TYPE_MAPPING_HPP

#include "mkl_dnn_types.h"

namespace mkl_dnn { namespace impl {

// TODO: autogenerate this

using nd_offset_t = mkl_dnn_nd_offset_t;
using dims_t = mkl_dnn_dims_t;

using status_t = mkl_dnn_status_t;
namespace status {
    const status_t success = mkl_dnn_success;
    const status_t invalid = mkl_dnn_invalid;
    const status_t out_of_memory = mkl_dnn_out_of_memory;
    const status_t try_again = mkl_dnn_try_again;
    const status_t invalid_arguments = mkl_dnn_invalid_arguments;
    const status_t not_ready = mkl_dnn_not_ready;
    const status_t unimplemented = mkl_dnn_unimplemented;
}

using prop_kind_t = mkl_dnn_prop_kind_t;
namespace prop_kind {
    const prop_kind_t forward = mkl_dnn_forward;
    const prop_kind_t backward_data = mkl_dnn_backward_data;
    const prop_kind_t backward_weights = mkl_dnn_backward_weights;
    const prop_kind_t backward_bias = mkl_dnn_backward_bias;
}

using alg_kind_t = mkl_dnn_alg_kind_t;
namespace alg_kind {
    const alg_kind_t convolution_direct = mkl_dnn_convolution_direct;
    const alg_kind_t pooling_max = mkl_dnn_pooling_max;
    const alg_kind_t lrn_across_channels = mkl_dnn_lrn_across_channels;
    const alg_kind_t lrn_within_channel = mkl_dnn_lrn_within_channel;
}

using tensor_desc_t = mkl_dnn_tensor_desc_t;

using precision_t = mkl_dnn_precision_t;
namespace precision {
    const precision_t f32 = mkl_dnn_f32;
}

using memory_format_t = mkl_dnn_memory_format_t;
namespace memory_format {
    const memory_format_t any = mkl_dnn_any;
    const memory_format_t n = mkl_dnn_n;
    const memory_format_t nc = mkl_dnn_nc;
    const memory_format_t nchw = mkl_dnn_nchw;
    const memory_format_t nhwc = mkl_dnn_nhwc;
    const memory_format_t nChw8c = mkl_dnn_nChw8c;
    const memory_format_t oi = mkl_dnn_oi;
    const memory_format_t oihw = mkl_dnn_oihw;
    const memory_format_t OIhw8i8o = mkl_dnn_OIhw8i8o;
    const memory_format_t goihw = mkl_dnn_goihw;
    const memory_format_t gOIhw8i8o = mkl_dnn_gOIhw8i8o;
    const memory_format_t blocked = mkl_dnn_blocked;
}

using padding_kind_t = mkl_dnn_padding_kind_t;
namespace padding_kind {
    const padding_kind_t padding_zero = mkl_dnn_padding_zero;
}

using engine_kind_t = mkl_dnn_engine_kind_t;
namespace engine_kind {
    const engine_kind_t any_engine = mkl_dnn_any_engine;
    const engine_kind_t cpu = mkl_dnn_cpu;
    const engine_kind_t cpu_lazy = mkl_dnn_cpu_lazy;
}

using engine = mkl_dnn_engine;

using primitive_kind_t = mkl_dnn_primitive_kind_t;
namespace primitive_kind {
    const primitive_kind_t undefined = mkl_dnn_undefined_primitive;
    const primitive_kind_t memory = mkl_dnn_memory;
    const primitive_kind_t reorder = mkl_dnn_reorder;
    const primitive_kind_t convolution = mkl_dnn_convolution;
    const primitive_kind_t pooling = mkl_dnn_pooling;
    const primitive_kind_t relu = mkl_dnn_relu;
    const primitive_kind_t lrn = mkl_dnn_lrn;
}

using blocking_desc_t = mkl_dnn_blocking_desc_t;
using memory_desc_t = mkl_dnn_memory_desc_t;
using convolution_desc_t = mkl_dnn_convolution_desc_t;
using pooling_desc_t = mkl_dnn_pooling_desc_t;
using relu_desc_t = mkl_dnn_relu_desc_t;
using lrn_desc_t = mkl_dnn_lrn_desc_t;

struct op_desc_t {
    primitive_kind_t _kind;
    union {
        memory_desc_t memory;
        convolution_desc_t convolution;
        pooling_desc_t pooling;
        relu_desc_t relu;
        lrn_desc_t lrn;
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

#   undef DECL_CTOR_AND_CONVERTERS
};

using primitive_base_desc_t = mkl_dnn_primitive_base_desc_t;
using memory_primitive_desc_t = mkl_dnn_memory_primitive_desc_t;
using reorder_primitive_desc_t = mkl_dnn_reorder_primitive_desc_t;
using convolution_primitive_desc_t = mkl_dnn_convolution_primitive_desc_t;
using pooling_primitive_desc_t = mkl_dnn_pooling_primitive_desc_t;
using relu_primitive_desc_t = mkl_dnn_relu_primitive_desc_t;
using lrn_primitive_desc_t = mkl_dnn_lrn_primitive_desc_t;
#if 0
using primitive_desc_t = mkl_dnn_primitive_desc_t;
using const_primitive_desc_t = const_mkl_dnn_primitive_desc_t;
#else
union primitive_desc_t {
    primitive_base_desc_t base;
    memory_primitive_desc_t memory;
    reorder_primitive_desc_t reorder;
    convolution_primitive_desc_t convolution;
    pooling_primitive_desc_t pooling;
    relu_primitive_desc_t relu;
    lrn_primitive_desc_t lrn;
#   define DECL_CTOR_AND_CONVERTERS(c_type, name) \
    primitive_desc_t(const c_type &_): name(_) {} \
    static primitive_desc_t *convert_from_c(c_type *_) \
    { return reinterpret_cast<primitive_desc_t*>(_); } \
    static const primitive_desc_t *convert_from_c(const c_type *_) \
    { return reinterpret_cast<const primitive_desc_t*>(_); }

    DECL_CTOR_AND_CONVERTERS(primitive_base_desc_t, base);
    DECL_CTOR_AND_CONVERTERS(memory_primitive_desc_t, memory);
    DECL_CTOR_AND_CONVERTERS(reorder_primitive_desc_t, reorder);
    DECL_CTOR_AND_CONVERTERS(convolution_primitive_desc_t, convolution);
    DECL_CTOR_AND_CONVERTERS(pooling_primitive_desc_t, pooling);
    DECL_CTOR_AND_CONVERTERS(relu_primitive_desc_t, relu);
    DECL_CTOR_AND_CONVERTERS(lrn_primitive_desc_t, lrn);

#   undef DECL_CTOR_AND_CONVERTERS
};
#endif

using primitive = mkl_dnn_primitive;

using primitive_at_t = mkl_dnn_primitive_at_t;

using stream = mkl_dnn_stream;

}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

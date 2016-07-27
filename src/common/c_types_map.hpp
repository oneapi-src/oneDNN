#ifndef TYPE_MAPPING_HPP
#define TYPE_MAPPING_HPP

#include "mkl_dnn_types.h"

namespace mkl_dnn { namespace impl {

using nd_offset_t = mkl_dnn_nd_offset_t;
using nd_pos_t = mkl_dnn_nd_pos_t;
using dims_t = mkl_dnn_dims_t;

using status_t = mkl_dnn_status_t;

using prop_kind_t = mkl_dnn_prop_kind_t;
using alg_kind_t = mkl_dnn_alg_kind_t;
using tensor_desc_t = mkl_dnn_tensor_desc_t;
using memory_format_t = mkl_dnn_memory_format_t;
using padding_kind_t = mkl_dnn_padding_kind_t;
using blocking_desc_t = mkl_dnn_blocking_desc_t;
using memory_desc_t = mkl_dnn_memory_desc_t;
using convolution_desc_t = mkl_dnn_convolution_desc_t;
using engine_kind_t = mkl_dnn_engine_kind_t;

using engine = mkl_dnn_engine;

using primitive_kind_t = mkl_dnn_primitive_kind_t;
using primitive_base_desc_t = mkl_dnn_primitive_base_desc_t;
using memory_primitive_desc_t = mkl_dnn_memory_primitive_desc_t;
using reorder_primitive_desc_t = mkl_dnn_reorder_primitive_desc_t;
using convolution_primitive_desc_t = mkl_dnn_convolution_primitive_desc_t;
using primitive_desc_t = mkl_dnn_primitive_desc_t;
using const_primitive_desc_t = const_mkl_dnn_primitive_desc_t;

using primitive = mkl_dnn_primitive;

using primitive_at_t = mkl_dnn_primitive_at_t;

using stream = mkl_dnn_stream;

}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0

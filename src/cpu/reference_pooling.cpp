#include "mkl_dnn_types.h"

#include "c_types_map.hpp"
#include "reference_pooling.hpp"
#include "type_helpers.hpp"

#include <limits>

#define CHECK(f) do { \
    status_t status = f; \
    if (status != success) return status; \
} while(0)

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::alg_kind;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::memory_format;
using namespace mkl_dnn::impl::primitive_kind;

template <impl::precision_t prec>
status_t reference_pooling<prec>::execute_forward() {
    const data_t *src = reinterpret_cast<const data_t *>(this->input()[0].primitive->output()[this->input()[0].output_index]->memory_const());
    uint32_t *indices = reinterpret_cast<uint32_t*>(this->input()[1].primitive->output()[this->input()[1].output_index]->memory());
    data_t *dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_ppd.src_primitive_desc.memory_desc),
        indices_d(this->_ppd.indices_primitive_desc.memory_desc),
        dst_d(this->_ppd.dst_primitive_desc.memory_desc);

    const uint32_t IH = src_d.dims()[2];
    const uint32_t IW = src_d.dims()[3];
    const uint32_t KH = this->_ppd.pooling_desc.kernel[0];
    const uint32_t KW = this->_ppd.pooling_desc.kernel[1];
    const uint32_t SH = this->_ppd.pooling_desc.strides[0];
    const uint32_t SW = this->_ppd.pooling_desc.strides[1];
    const uint32_t PH = this->_ppd.pooling_desc.padding[0];
    const uint32_t PW = this->_ppd.pooling_desc.padding[1];

    auto ker = [=](data_t *d, uint32_t mb, uint32_t oc, uint32_t oh,
            uint32_t ow)
    {
        for (uint32_t kh = 0; kh < KH; ++kh) {
            for (uint32_t kw = 0; kw < KW; ++kw) {
                if (oh*SH + kh < PH) continue;
                if (ow*SW + kw < PW) continue;

                if (oh*SH + kh >= IH + PH) continue;
                if (ow*SW + kw >= IW + PW) continue;

                const uint32_t ih = oh * SH - PH + kh;
                const uint32_t iw = ow * SW - PW + kw;

                if (src[src_d.off(mb, oc, ih, iw)] > d[0]) {
                    d[0] = src[src_d.off(mb, oc, ih, iw)];
                    indices[indices_d.off(mb, oc, oh, ow)] = kh*KW + kw;
                }
            }
        }
    };

    const uint32_t MB = src_d.dims()[0];
    const uint32_t OC = dst_d.dims()[1];
    const uint32_t OH = dst_d.dims()[2];
    const uint32_t OW = dst_d.dims()[3];

#   pragma omp parallel for collapse(4)
    for (uint32_t mb = 0; mb < MB; ++mb) {
        for (uint32_t oc = 0; oc < OC; ++oc) {
            for (uint32_t oh = 0; oh < OH; ++oh) {
                for (uint32_t ow = 0; ow < OW; ++ow) {
                    data_t *d = &dst[dst_d.off(mb, oc, oh, ow)];
                    d[0] = -std::numeric_limits<data_t>::infinity();
                    ker(d, mb, oc, oh, ow);
                }
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t reference_pooling<prec>::execute_backward_data() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t reference_pooling<prec>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkl_dnn::impl::engine &engine) {
    if (op_desc._kind != primitive_kind::pooling)
        return invalid;
    auto pool_d = op_desc.pooling;

    // TODO: f32 ?
    if (pool_d.prop_kind != forward)
        return unimplemented;
    if (pool_d.alg_kind != pooling_max)
        return unimplemented;

    /* memory descriptors check and fill-in */
    if (pool_d.src_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&pool_d.src_desc,
        &pool_d.src_desc.tensor_desc, f32, nchw));
    if (pool_d.indices_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&pool_d.indices_desc,
        &pool_d.indices_desc.tensor_desc, u32, nchw));
    if (pool_d.dst_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&pool_d.dst_desc,
        &pool_d.dst_desc.tensor_desc, f32, nchw));

    /* memory primitive descriptors check */
    memory_primitive_desc_t src_pd, indices_pd, dst_pd;
    CHECK(mkl_dnn_memory_primitive_desc_init(&src_pd,
        &pool_d.src_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&indices_pd,
        &pool_d.indices_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&dst_pd,
        &pool_d.dst_desc, &engine));

    /* final stage */
    pooling_primitive_desc_t ppd = {
        .base = {
            .primitive_kind = pooling,
            .engine = &engine,
            .implementation = reinterpret_cast<const void*>(&implementation),
        },
        .pooling_desc = pool_d,
        .src_primitive_desc   = src_pd,
        .indices_primitive_desc = indices_pd,
        .dst_primitive_desc  = dst_pd,
    };

    // if (!pooling_primitive_desc_is_ok(ppd)) return invalid; // ???

    primitive_desc->pooling = ppd;

    return success;
}

namespace {
template <impl::precision_t prec>
status_t create(primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == pooling);

    auto& ppd = primitive_desc->pooling;
    // TODO: some checks here.

    *aprimitive = new reference_pooling<prec>(ppd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}
}

template <impl::precision_t prec>
const primitive_impl reference_pooling<prec>::implementation = {
    .primitive_create = create<prec>,
};

template class reference_pooling<f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

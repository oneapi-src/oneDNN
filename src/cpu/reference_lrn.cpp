#include "mkl_dnn_types.h"

#include "c_types_map.hpp"
#include "reference_lrn.hpp"
#include "type_helpers.hpp"
#include <cmath>

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
status_t reference_lrn<prec>::execute_forward() {
    const data_t *src =
        reinterpret_cast<const data_t *>(this->input()[0].primitive->output()[this->input()[0].output_index]->memory_const());
    data_t *scratch =
        reinterpret_cast<data_t *>(this->input()[1].primitive->output()[this->input()[1].output_index]->memory());
    data_t *dst =
        reinterpret_cast<data_t *>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_ppd.src_primitive_desc.memory_desc),
        scratch_d(this->_ppd.scratch_primitive_desc.memory_desc),
        dst_d(this->_ppd.dst_primitive_desc.memory_desc);

    const uint32_t C = src_d.dims()[1];
    const uint32_t H = src_d.dims()[2];
    const uint32_t W = src_d.dims()[3];

    auto ker = [=](data_t *d, uint32_t n, uint32_t oc, uint32_t oh, uint32_t ow)
    {
        const double alpha = this->_ppd.lrn_desc.alpha;
        const double beta = this->_ppd.lrn_desc.beta;

        const uint32_t size = this->_ppd.lrn_desc.local_size;
        const uint32_t CSIZE = this->_ppd.lrn_desc.alg_kind == lrn_across_channels ? size : 1;
        const uint32_t HWSIZE = size + 1 - CSIZE;

        data_t sum = 0.0;
        uint32_t summands = 0;
        for (uint32_t c = oc; c < oc + CSIZE; ++c) {
            if (c < (CSIZE - 1) / 2) continue;
            if (c >= C + (CSIZE - 1) / 2) continue;
            for (uint32_t h = oh; h < oh + HWSIZE; ++h) {
                if (h < (HWSIZE - 1) / 2) continue;
                if (h >= H + (HWSIZE - 1) / 2) continue;
                for (uint32_t w = ow; w < ow + HWSIZE; ++w) {
                    if (w < (HWSIZE - 1) / 2) continue;
                    if (w >= W + (HWSIZE - 1) / 2) continue;
                    data_t s = src[src_d.off(n, c - (CSIZE - 1) / 2, h - (HWSIZE - 1) / 2, w - (HWSIZE - 1) / 2)];
                    sum += s * s;
                    summands += 1;
                }
            }
        }
        d[0] = pow(1 + alpha * sum / summands, beta);
        scratch[scratch_d.off(n, oc, oh, ow)] = d[0] / (1 + alpha * sum / summands); // for back prop
    };

    const uint32_t N = src_d.dims()[0];

#   pragma omp parallel for collapse(4)
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t c = 0; c < C; ++c) {
            for (uint32_t h = 0; h < H; ++h) {
                for (uint32_t w = 0; w < W; ++w) {
                    ker(&dst[dst_d.off(n, c, h, w)], n, c, h, w);
                }
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t reference_lrn<prec>::execute_backward_data() {
    return unimplemented;
}

template <impl::precision_t prec>
status_t reference_lrn<prec>::primitive_desc_init(
        primitive_desc_t *primitive_desc, const op_desc_t &op_desc,
        const mkl_dnn::impl::engine &engine) {
    if (op_desc._kind != primitive_kind::lrn)
        return invalid_arguments;
    auto lrn_d = op_desc.lrn;

    // TODO: f32 ?
    if (lrn_d.prop_kind != forward)
        return unimplemented;

    /* memory descriptors check and fill-in */
    if (lrn_d.src_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&lrn_d.src_desc,
        &lrn_d.src_desc.tensor_desc, f32, nchw));
    if (lrn_d.dst_desc.format == any)
        CHECK(mkl_dnn_memory_desc_init(&lrn_d.dst_desc,
        &lrn_d.dst_desc.tensor_desc, f32, nchw));

    memory_desc_t scratch_desc;
    CHECK(mkl_dnn_memory_desc_init(&scratch_desc,
        &lrn_d.dst_desc.tensor_desc, f32, lrn_d.dst_desc.format));

    /* memory primitive descriptors check */
    memory_primitive_desc_t src_pd, scratch_pd, dst_pd;
    CHECK(mkl_dnn_memory_primitive_desc_init(&src_pd,
        &lrn_d.src_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&dst_pd,
        &lrn_d.dst_desc, &engine));
    CHECK(mkl_dnn_memory_primitive_desc_init(&scratch_pd,
        &scratch_desc, &engine));

    /* final stage */
    lrn_primitive_desc_t ppd = {
        .base = {
            .primitive_kind = lrn,
            .engine = &engine,
            .implementation = reinterpret_cast<const void*>(&implementation),
        },
        .lrn_desc = lrn_d,
        .src_primitive_desc   = src_pd,
        .scratch_primitive_desc = scratch_pd,
        .dst_primitive_desc  = dst_pd,
    };

    // if (!lrn_primitive_desc_is_ok(ppd)) return invalid_arguments; // ???

    primitive_desc->lrn = ppd;

    return success;
}

namespace {
template <impl::precision_t prec>
status_t create(primitive **aprimitive, const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], const primitive *outputs[]) {
    assert(primitive_desc->base.primitive_kind == lrn);

    auto& ppd = primitive_desc->lrn;
    // TODO: some checks here.

    *aprimitive = new reference_lrn<prec>(ppd, inputs, outputs);
    return aprimitive ? success : out_of_memory;
}
}

template <impl::precision_t prec>
const primitive_impl reference_lrn<prec>::implementation = {
    .primitive_create = create<prec>,
};

template class reference_lrn<f32>;

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

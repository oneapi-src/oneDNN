#include <assert.h>

#include "cpu_engine.hpp"

#include "cpu/cpu_memory.hpp"
#include "cpu/reference_convolution.hpp"
#include "cpu/jit_avx2_convolution.hpp"
#include "cpu/reference_pooling.hpp"
#include "cpu/reference_relu.hpp"
#include "cpu/reference_lrn.hpp"
#include "cpu/jit_avx2_lrn.hpp"
#include "cpu/reference_inner_product.hpp"

#include "cpu/reference_reorder.hpp"

namespace mkl_dnn {
namespace impl {
namespace cpu {

cpu_engine_factory engine_factory(false);
cpu_engine_factory engine_factory_lazy(true);

namespace {
using namespace mkl_dnn::impl::precision;

primitive_desc_init_f primitive_inits[] = {
    cpu_memory::memory_desc_init,
    jit_avx2_convolution<f32>::primitive_desc_init,
    reference_convolution<f32>::primitive_desc_init,
    reference_pooling<f32>::primitive_desc_init,
    reference_relu<f32>::primitive_desc_init,
    jit_avx2_lrn<f32>::primitive_desc_init,
    reference_lrn<f32>::primitive_desc_init,
    reference_inner_product<f32>::primitive_desc_init,
    NULL,
};

reorder_primitive_desc_init_f reorder_inits[] = {
    reference_reorder<f32, f32>::reorder_primitive_desc_init,
    NULL,
};
}

primitive_desc_init_f *cpu_engine::get_primitive_inits() const {
    return primitive_inits;
}

reorder_primitive_desc_init_f *cpu_engine::get_reorder_inits() const {
    return reorder_inits;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

#if 0
#include <assert.h>

#include "cpu_engine.hpp"
#include "cpu/reference_convolution.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

namespace {
primitive_desc_init_f convolution_inits[] = {
    reference_convolution<impl::precision::f32>::primitive_desc_init,
    NULL,
};
}

primitive_desc_init_f *cpu_engine::get_convolution_inits() const {
    return convolution_inits;
}

}}}
#endif
// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

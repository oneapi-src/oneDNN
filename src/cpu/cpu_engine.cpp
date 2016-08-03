#include <assert.h>

#include "cpu_engine.hpp"

#include "cpu/cpu_memory.hpp"
#include "cpu/reference_convolution.hpp"
#include "cpu/reference_pooling.hpp"

namespace mkl_dnn {
namespace impl {
namespace cpu {

cpu_engine_factory engine_factory(false);
cpu_engine_factory engine_factory_lazy(true);

namespace {
primitive_desc_init_f memory_inits[] = {
    cpu_memory::memory_desc_init,
    reference_convolution<impl::precision::f32>::primitive_desc_init,
    reference_pooling<impl::precision::f32>::primitive_desc_init,
    NULL,
};
}


primitive_desc_init_f *cpu_engine::get_primitive_inits() const {
    return memory_inits;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

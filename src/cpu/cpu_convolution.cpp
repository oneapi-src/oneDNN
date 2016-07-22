#include <assert.h>

#include "cpu_engine.hpp"
#include "cpu/reference_convolution.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

namespace {
primitive_desc_init_f convolution_inits[] = {
    reference_convolution::primitive_desc_init,
    NULL,
};
}

primitive_desc_init_f *cpu_engine::get_convolution_inits() const {
    return convolution_inits;
}

}}}

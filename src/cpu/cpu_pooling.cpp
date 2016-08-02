#include <assert.h>

#include "cpu_engine.hpp"
#include "cpu/reference_pooling.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

namespace {
primitive_desc_init_f pooling_inits[] = {
    reference_pooling<impl::precision::f32>::primitive_desc_init,
    NULL,
};

}

primitive_desc_init_f *cpu_engine::get_pooling_inits() const {
    return pooling_inits;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

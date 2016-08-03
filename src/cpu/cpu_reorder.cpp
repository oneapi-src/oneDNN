#include <assert.h>

#include "cpu_engine.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

namespace {
reorder_primitive_desc_init_f reorder_inits[] = {
    NULL,
};
}

reorder_primitive_desc_init_f *cpu_engine::get_reorder_inits() const {
    return reorder_inits;
}

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

#ifndef REORDER_HPP
#define REORDER_HPP

#include "mkl_dnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"

namespace mkl_dnn {
namespace impl {

typedef status_t (*reorder_primitive_desc_init_f)(
        primitive_desc_t *primitive_desc,
        const memory_primitive_desc_t *input,
        const memory_primitive_desc_t *output);

}
}

#endif

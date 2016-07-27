#include "cpu_engine.hpp"

namespace mkl_dnn {
namespace impl {
namespace cpu {

cpu_engine_factory engine_factory(false);
cpu_engine_factory engine_factory_lazy(true);

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
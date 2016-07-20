#include "cpu_engine.hpp"

namespace mkl_dnn {
namespace impl {
namespace cpu {

cpu_engine_factory engine_factory(false);
cpu_engine_factory engine_factory_lazy(true);

}
}
}

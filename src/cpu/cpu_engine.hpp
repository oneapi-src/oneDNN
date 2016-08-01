#ifndef CPU_ENGINE_HPP
#define CPU_ENGINE_HPP

#include <assert.h>

#include "mkl_dnn.h"

#include "c_types_map.hpp"
#include "../common/engine.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;

class cpu_engine: public engine {
private:
    bool _lazy;
public:
    cpu_engine(bool lazy): _lazy(lazy) {}
    virtual bool is_lazy() const { return _lazy; }
    virtual bool is_ok() const { return true; }
    virtual status_t submit(size_t n, primitive *primitives[],
            primitive **error_primitive) {
        assert(error_primitive);
        *error_primitive = 0;
        for (size_t i = 0; i < n; i++) {
            status_t rc = primitives[i]->execute();
            if (rc != success) {
                *error_primitive = primitives[i];
                return rc;
            }
        }
        return success;
    }

    virtual primitive_desc_init_f *get_memory_inits() const;
    virtual primitive_desc_init_f *get_convolution_inits() const;
};

class cpu_engine_factory: public engine_factory {
private:
    bool _lazy;
public:
    cpu_engine_factory(bool lazy): _lazy(lazy) {}
    virtual size_t count() { return 1; }
    virtual engine_kind_t kind()
    { return _lazy ? engine_kind::cpu_lazy : engine_kind::cpu; }
    virtual status_t engine_create(engine **aengine, size_t index) {
        assert(index == 0);
        *aengine = new cpu_engine(_lazy);
        return success;
    };
};

extern cpu_engine_factory engine_factory;
extern cpu_engine_factory engine_factory_lazy;

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

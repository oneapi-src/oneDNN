#ifndef CPU_ENGINE_HPP
#define CPU_ENGINE_HPP

#include <assert.h>

#include "mkl_dnn.h"

#include "../common/engine.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

class cpu_engine: public dnn_engine {
private:
    bool _lazy;
public:
    cpu_engine(bool lazy): _lazy(lazy) {}
    virtual bool is_lazy() const { return _lazy; }
    virtual bool is_ok() const { return true; }
    virtual status_t submit(size_t n, dnn_primitive *primitives[],
            dnn_primitive **error_primitive) {
        dnn_primitive *p;
        if (!error_primitive) error_primitive = &p;
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
};

class cpu_engine_factory: public engine_factory {
private:
    bool _lazy;
public:
    cpu_engine_factory(bool lazy): _lazy(lazy) {}
    virtual size_t count() { return 1; }
    virtual engine_kind_t kind()
    { return _lazy ? engine_kind_cpu_lazy : engine_kind_cpu; }
    virtual status_t engine_create(dnn_engine **engine, size_t index) {
        assert(index == 0);
        *engine = new cpu_engine(_lazy);
        return success;
    };
};

extern cpu_engine_factory engine_factory;
extern cpu_engine_factory engine_factory_lazy;

}}}

#endif

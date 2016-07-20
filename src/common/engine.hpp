#ifndef ENGINE_HPP
#define ENGINE_HPP

#include "mkl_dnn.h"

#include "utils.hpp"
#include "primitive.hpp"
#include "convolution.hpp"

namespace mkl_dnn { namespace impl {

class engine: public c_compatible {
public:
    virtual bool is_lazy() const = 0;
    virtual bool is_ok() const = 0;
    virtual status_t submit(size_t n,
            primitive *primitives[], primitive **error_primitive) = 0;

    /* primitives' descriptor initializators
     * the default one guarantees to return at least an empty list,
     * so no need to check the return value on NULL */
    virtual primitive_desc_init_f *get_memory_inits() const;
    virtual primitive_desc_init_f *get_reorder_inits() const;
    virtual primitive_desc_init_f *get_convolution_inits() const;
};

class engine_factory: public c_compatible {
public:
    virtual size_t count() = 0;
    virtual engine_kind_t kind() = 0;
    virtual status_t engine_create(engine **engine, size_t index) = 0;
};

}}

#endif

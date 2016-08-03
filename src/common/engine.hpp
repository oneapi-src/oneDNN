#ifndef ENGINE_HPP
#define ENGINE_HPP

#include "mkl_dnn.h"

#include "c_types_map.hpp"
#include "utils.hpp"
#include "primitive.hpp"
#include "reorder.hpp"

class mkl_dnn_engine: public mkl_dnn::impl::c_compatible {
protected:
    mkl_dnn::impl::engine_kind_t _kind;
public:
    mkl_dnn_engine(): _kind(mkl_dnn::impl::engine_kind::any_engine) {}
    mkl_dnn_engine(mkl_dnn::impl::engine_kind_t kind): _kind(kind) {}
    virtual ~mkl_dnn_engine() {}

    virtual bool is_lazy() const = 0;
    virtual bool is_ok() const = 0;
    mkl_dnn::impl::engine_kind_t kind() const { return _kind; }

    virtual mkl_dnn::impl::status_t submit(size_t n,
            mkl_dnn::impl::primitive *primitives[],
            mkl_dnn::impl::primitive **error_primitive) = 0;

    /* primitives' descriptor initializators
     * the default one guarantees to return at least an empty list,
     * so no need to check the return value on NULL */
    virtual mkl_dnn::impl::primitive_desc_init_f *get_primitive_inits() const;
    virtual mkl_dnn::impl::reorder_primitive_desc_init_f *get_reorder_inits() const;
};

namespace mkl_dnn { namespace impl {

class engine_factory: public c_compatible {
public:
    virtual size_t count() = 0;
    virtual engine_kind_t kind() = 0;
    virtual status_t engine_create(engine **engine, size_t index) = 0;
};

}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

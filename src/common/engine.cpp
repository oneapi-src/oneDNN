#include "mkl_dnn.h"
#include "engine.hpp"
#include "nstl.hpp"

#include "../cpu/cpu_engine.hpp"

namespace mkl_dnn { namespace impl {

// TODO: we need some caching+refcounting mechanism so that an engine could not
// be created twice and is only destroyed when the refcount is 0

// With STL we would've used vector. Alas, we cannot use STL..
engine_factory *engine_factories[] = {
    &cpu::engine_factory,
    &cpu::engine_factory_lazy,
    NULL,
};

static inline engine_factory *get_engine_factory(engine_kind_t kind)
{
    if (kind < 0 || kind >= engine_kind_last)
            return NULL;
    for (engine_factory **ef = engine_factories; *ef; ef++)
        if ((*ef)->kind() == kind)
            return *ef;
    return NULL;
}

namespace {
    primitive_desc_init_f empty_list[] = { nullptr };
}

primitive_desc_init_f *engine::get_memory_inits() const { return empty_list; }
primitive_desc_init_f *engine::get_reorder_inits() const { return empty_list; }
primitive_desc_init_f *engine::get_convolution_inits() const {
    return empty_list;
}

}}

size_t engine_get_count(engine_kind_t kind)
{
    mkl_dnn::impl::engine_factory *ef = mkl_dnn::impl::get_engine_factory(kind);
    return ef != NULL ? ef->count() : 0;
}

status_t engine_create(engine_t *engine, engine_kind_t kind, size_t index)
{
    if (engine == NULL)
        return invalid_arguments;

    mkl_dnn::impl::engine_factory *ef = mkl_dnn::impl::get_engine_factory(kind);
    if (ef == NULL || index >= ef->count())
        return invalid_arguments;

    return ef->engine_create(
            reinterpret_cast<mkl_dnn::impl::engine**>(engine), index);
}

status_t engine_destroy(engine_t engine)
{
    mkl_dnn::impl::engine *e = static_cast<mkl_dnn::impl::engine*>(engine);
    delete e;
}

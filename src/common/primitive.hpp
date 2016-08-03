#ifndef PRIMITIVE_HPP
#define PRIMITIVE_HPP

#include "mkl_dnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"

// TODO: consider using smart pointers for storing primitives. External handles
// then would have to be cast to smart pointers. This would ensure that
// accedentally deleting a primitive that is a dependency for another one does
// not cause a segfault.

struct mkl_dnn_primitive: public mkl_dnn::impl::c_compatible {
public:
    enum exec_state { done, busy, not_ready, error };

private:
    // TODO: copy, equality and assignment -- all must be banned...

protected:
    const mkl_dnn::impl::primitive_desc_t _primitive_desc;
    mkl_dnn::impl::engine *_engine;
    exec_state _exec_state;
    mkl_dnn::impl::nstl::vector<mkl_dnn::impl::primitive_at_t> _input;
    mkl_dnn::impl::nstl::vector<mkl_dnn::impl::primitive*> _output;

    virtual mkl_dnn::impl::status_t execute_impl() = 0;

    mkl_dnn_primitive(const mkl_dnn::impl::primitive_desc_t& primitive_desc,
            mkl_dnn::impl::engine *engine, exec_state state = not_ready)
        : _primitive_desc(primitive_desc)
        , _engine(engine)
        , _exec_state(state) {}

public:
    virtual ~mkl_dnn_primitive() {}

    mkl_dnn::impl::primitive_desc_t primitive_desc() const
    { return _primitive_desc; }
    mkl_dnn::impl::engine *engine() const { return _engine; }
    mkl_dnn::impl::primitive_kind_t kind() const
    { return _primitive_desc.base.primitive_kind; }

    virtual bool own_memory() const { return false; }

    virtual exec_state get_exec_state() const { return _exec_state; }
    virtual mkl_dnn::impl::status_t reset_exec_state(exec_state state) {
        // TODO: some checks here?
        _exec_state = state;
        return mkl_dnn::impl::status::success;
    }

    bool inputs_ready() const {
        for (auto i = 0UL; i < _input.size(); i++)
            if (_input[i].primitive->get_exec_state() != done)
                return false;
        return true;
    }
    mkl_dnn::impl::status_t execute() {
        if (!inputs_ready())
            return mkl_dnn::impl::status::not_ready;
        return execute_impl();
    }

    size_t input_count() const { return _input.size(); }
    mkl_dnn::impl::nstl::vector<mkl_dnn::impl::primitive_at_t> &input() {
        return _input;
    }

    size_t output_count() const { return _output.size(); }
    const mkl_dnn::impl::nstl::vector<mkl_dnn::impl::primitive*> &output() const
    { return _output; }

    virtual char* memory() { return output()[0]->memory(); }
    virtual const char* memory_const() { return output()[0]->memory_const(); }
};

namespace mkl_dnn { namespace impl {

typedef status_t (*primitive_desc_init_f)(primitive_desc_t *primitive_desc,
        const op_desc_t &op_desc, const engine &aengine);
typedef status_t (*primitive_create_f)(primitive **aprimitive,
        const primitive_desc_t *primitive_desc, const primitive_at_t inputs[],
        primitive *outputs[]);

struct primitive_impl /* : public c_compatible */ {
    const primitive_create_f primitive_create;
};

status_t primitive_desc_init(primitive_desc_t *primitive_desc,
        const op_desc_t &op_desc, const engine &aengine);

status_t inline check_inputs_array(size_t n, const primitive_at_t inputs[]) {
    for (size_t i = 0; i < n; i++)
        if (inputs[i].primitive->output_count() <= inputs[i].output_index)
            return status::invalid_arguments;
    return status::success;
}

}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

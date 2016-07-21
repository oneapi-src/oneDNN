#ifndef PRIMITIVE_HPP
#define PRIMITIVE_HPP

#include "mkl_dnn.h"

#include "nstl.hpp"

namespace mkl_dnn { namespace impl {

class engine;
class memory;

// TODO: consider using smart pointers for storing primitives. External handles
// then would have to be cast to smart pointers. This would ensure that
// accedentally deleting a primitive that is a dependency for another one does
// not cause a segfault.

struct primitive: public c_compatible {
private:
    // TODO: copy, equality and assignment -- all must be banned...
protected:
    dnn_engine *_engine;
    ::primitive_kind_t _kind;
    nstl::vector<primitive*> _input;
    nstl::vector<primitive*> _output;

    virtual status_t execute_impl() = 0;

    primitive(dnn_engine *engine, ::primitive_kind_t kind)
        : _engine(engine)
        , _kind(kind) {}
public:
    virtual ~primitive() {}

    ::primitive_kind_t kind() const { return _kind; }
    dnn_engine *engine() const { return _engine; }

    virtual bool own_memory() const { return false; }

    enum exec_state { done, busy, not_ready, error };
    virtual exec_state get_exec_state() const = 0;
    bool inputs_ready() {
        for (auto i = 0; i < _input.size(); i++)
            if (_input[i]->get_exec_state() != done)
                return false;
        return true;
    }
    status_t execute() {
        if (!inputs_ready())
            return ::not_ready;
        return execute_impl();
    }

    size_t input_count() const { return _input.size(); }
    nstl::vector<primitive*> &input() { return _input; };

    size_t output_count() const { return _output.size(); }
    nstl::vector<primitive*> &output() { return _output; };

    // XXX: memory -> primitive?
    virtual const impl::memory *output_memory_const(size_t at = 0) const {
        return _output[at]->memory_const();
    }
    virtual impl::memory *output_memory(size_t at = 0) const {
        return _output[at]->memory();
    }

    virtual impl::memory *memory() { return 0; }
    virtual impl::memory *memory_const() { return 0; }
};

typedef const void* const_op_desc_t;
typedef status_t (*primitive_desc_init_f)(primitive_desc_t *primitive_desc,
        const_op_desc_t op_desc, const dnn_engine& engine);
typedef status_t (*primitive_create_f)(impl::primitive **primitive,
        ::const_primitive_desc_t primitive_desc,
        const impl::primitive *inputs[], const impl::primitive *outputs[]);

struct primitive_impl /* : public c_compatible */ {
    const primitive_desc_init_f primitive_desc_init;
    const primitive_create_f primitive_create;
};

struct memory: public primitive { };

}}

#endif

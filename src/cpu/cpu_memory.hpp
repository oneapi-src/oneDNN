#ifndef CPU_MEMORY_HPP
#define CPU_MEMORY_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl;
using namespace mkl_dnn::impl::status;

class cpu_memory: public primitive {
private:
    char *_memory_buffer;
    const bool _owns_memory;

protected:
    status_t execute_impl() { return success; }

public:
    cpu_memory(const memory_primitive_desc_t &mpd, char* ptr)
        : primitive(mpd, const_cast<impl::engine*>(mpd.base.engine), done)
        , _memory_buffer(ptr), _owns_memory(ptr == nullptr) {
        primitive_at_t input_at = { this, 0 };
        _input.push_back(input_at);
        _output.push_back(this);
        if (_memory_buffer == nullptr) {
            const size_t size = types::get_size(mpd);
            _memory_buffer = new char[size];
        }
    }
    ~cpu_memory() { if (_owns_memory) delete [] _memory_buffer; }

    bool owns_memory() const { return _memory_buffer != NULL; }

    virtual char* memory() { return _memory_buffer; }
    virtual const char* memory_const() { return _memory_buffer; }

    /* static magic */
    static status_t memory_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkl_dnn::impl::engine &aengine);

    static status_t memory_create(primitive **primitive,
            const primitive_desc_t *primitive_desc,
            const primitive_at_t inputs[],
            mkl_dnn::impl::primitive *outputs[]);

    static const primitive_impl memory_implementation;
};

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

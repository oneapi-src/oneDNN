#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl;

class cpu_memory: public primitive {
private:
    char *_memory_buffer;
    const bool _owns_memory;

protected:
    status_t execute_impl() { return mkl_dnn_success; }

public:
    cpu_memory(const memory_primitive_desc_t &mpd, char* ptr):
        primitive(const_cast<mkl_dnn_engine*>(mpd.base.engine),
                mkl_dnn_memory),
        _memory_buffer(ptr), _owns_memory(ptr == nullptr) {
        _input.push_back(this);
        _output.push_back(this);
        if (_memory_buffer == nullptr) {
            const size_t size = types::get_size(mpd);
            _memory_buffer = new char[size];
        }
    }
    ~cpu_memory() { if (_owns_memory) delete [] _memory_buffer; }

    bool owns_memory() const { return _memory_buffer != NULL; }
    exec_state get_exec_state() const { return done; }

    /* static magic */
    static status_t memory_desc_init(primitive_desc_t *primitive_desc,
            const_op_desc_t op_desc, const mkl_dnn_engine &engine) {
        auto memory_primitive_desc =
            reinterpret_cast<memory_primitive_desc_t*>(primitive_desc);
        auto memory_desc = static_cast<const memory_desc_t*>(op_desc);

        memory_primitive_desc_t mpd = {
            .base = {
                .primitive_kind = mkl_dnn_memory,
                .engine = &engine,
                .implementation =
                    reinterpret_cast<const void*>(&memory_implementation),
            },
            .memory_desc = *memory_desc
        };
        // if (!memory_primitive_desc_is_ok(mpd)) return invalid; // ???
        *memory_primitive_desc = mpd;

        return mkl_dnn_success;
    }

    static status_t memory_create(mkl_dnn_primitive **primitive,
            const_primitive_desc_t primitive_desc,
            const primitive_at_t inputs[], const mkl_dnn_primitive *outputs[]) {
        auto& mpd = *static_cast<const memory_primitive_desc_t*>(primitive_desc);
        assert(mpd.base.primitive_kind == mkl_dnn_memory);
        assert(inputs[0].primitive == outputs[0]);
        char* ptr = const_cast<char*>(reinterpret_cast<const char*>(outputs[0]));

        *primitive = new cpu_memory(mpd, ptr);
        if (primitive)
			return mkl_dnn_success;
        return mkl_dnn_out_of_memory;
    }

    static const primitive_impl memory_implementation;
};

const primitive_impl cpu_memory::memory_implementation = {
    .primitive_desc_init = cpu_memory::memory_desc_init,
    .primitive_create = cpu_memory::memory_create
};

namespace {
primitive_desc_init_f memory_inits[] = {
    cpu_memory::memory_desc_init,
    NULL,
};
}

primitive_desc_init_f *cpu_engine::get_memory_inits() const {
    return memory_inits;
}

}}}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0
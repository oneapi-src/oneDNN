#ifndef CPU_REFERENCE_CONVOLUTION_HPP
#define CPU_REFERENCE_CONVOLUTION_HPP

#include <assert.h>

#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

class reference_convolution: public dnn_primitive {
private:
    const convolution_primitive_desc_t &_cpd;
    exec_state _exec_state;

    // TODO: implement in cpp.
    status_t execute_forward();
    status_t execute_backward_data();
    status_t execute_backward_weights();
    status_t execute_backward_bias();

protected:
    status_t execute_impl() {
        status_t status = success;
        _exec_state = busy;
        switch (_cpd.convolution_desc.prop_kind) {
        case forward: status = execute_forward(); break;
        case backward_data: status = execute_backward_data(); break;
        case backward_weights: status = execute_backward_weights(); break;
        case backward_bias: status = execute_backward_bias(); break;
        default: _exec_state = error; return unimplemented;
        }
        _exec_state = done;
        return status;
    }

public:
    reference_convolution(const convolution_primitive_desc_t &cpd):
            dnn_primitive(const_cast<dnn_engine*>(cpd.base.engine),
                primitive_kind_convolution),
            _exec_state(not_ready),
            _cpd(cpd) {
        // TODO: implement
    }
    ~reference_convolution() {}

    exec_state get_exec_state() const { return _exec_state; } // TODO: put this in common?

    /* static magic */
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const_op_desc_t op_desc, const dnn_engine &engine);
    static status_t create(dnn_primitive **primitive,
        const_primitive_desc_t primitive_desc,
        const dnn_primitive_at_t inputs[], const dnn_primitive *outputs[]);
    static const primitive_impl implementation;
};

}}}

#endif

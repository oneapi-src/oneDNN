#ifndef CPU_REFERENCE_CONVOLUTION_HPP
#define CPU_REFERENCE_CONVOLUTION_HPP

#include <assert.h>

#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

class reference_convolution: public primitive {
private:
    exec_state _exec_state;
    const convolution_primitive_desc_t &_cpd;

    // TODO: implement in cpp.
    status_t execute_forward();
    status_t execute_backward_data();
    status_t execute_backward_weights();
    status_t execute_backward_bias();

protected:
    status_t execute_impl() {
        status_t status = mkl_dnn_success;
        _exec_state = busy;
        switch (_cpd.convolution_desc.prop_kind) {
        case mkl_dnn_forward: status = execute_forward(); break;
        case mkl_dnn_backward_data: status = execute_backward_data(); break;
        case mkl_dnn_backward_weights: status = execute_backward_weights(); break;
        case mkl_dnn_backward_bias: status = execute_backward_bias(); break;
        default: _exec_state = error; return mkl_dnn_unimplemented;
        }
        _exec_state = done;
        return status;
    }

public:
    reference_convolution(const convolution_primitive_desc_t &cpd):
            primitive(const_cast<mkl_dnn_engine*>(cpd.base.engine),
                mkl_dnn_convolution),
            _exec_state(not_ready),
            _cpd(cpd) {
        // TODO: implement
    }
    ~reference_convolution() {}

    exec_state get_exec_state() const { return _exec_state; } // TODO: put this in common?

    /* static magic */
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const_op_desc_t op_desc, const mkl_dnn_engine &engine);
    static status_t create(primitive **primitive,
        const_primitive_desc_t primitive_desc,
        const primitive_at_t inputs[], const mkl_dnn_primitive *outputs[]);
    static const primitive_impl implementation;
};

}}}

#endif

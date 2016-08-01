#ifndef CPU_REFERENCE_CONVOLUTION_HPP
#define CPU_REFERENCE_CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "primitive.hpp"
#include "cpu_engine.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::primitive_kind;

template <impl::precision_t prec>
class reference_convolution: public primitive {
private:
    const impl::convolution_primitive_desc_t &_cpd;
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
        default: assert(0 && "invalid prop_kind"); // should never happen
        }
        _exec_state = done;
        return status;
    }

public:
    typedef typename precision2type<prec>::type data_t;

    reference_convolution(const convolution_primitive_desc_t &cpd,
            const primitive_at_t *inputs, primitive *outputs[])
        : primitive(cpd, const_cast<impl::engine*>(cpd.base.engine))
        , _cpd(_primitive_desc.convolution)
        , _exec_state(not_ready)
    {
        for (int i = 0; i < 3; ++i)
            _input.push_back(inputs[i]);
        _output.push_back(outputs[0]);
    }
    ~reference_convolution() {}

    exec_state get_exec_state() const { return _exec_state; } // TODO: put this in common?

    /* static magic */
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const_op_desc_t op_desc, const mkl_dnn::impl::engine &aengine);
    static status_t create(primitive **aprimitive,
        const primitive_desc_t *primitive_desc,
        const primitive_at_t inputs[], primitive *outputs[]);
    static const primitive_impl implementation;
};

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

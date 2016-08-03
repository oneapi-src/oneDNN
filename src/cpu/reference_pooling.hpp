#ifndef CPU_REFERENCE_POOLING_HPP
#define CPU_REFERENCE_POOLING_HPP

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
class reference_pooling: public primitive {
private:
    const impl::pooling_primitive_desc_t &_ppd;

    // TODO: implement in cpp.
    status_t execute_forward();
    status_t execute_backward_data();

protected:
    status_t execute_impl() {
        status_t status = success;
        _exec_state = busy;
        switch (_ppd.pooling_desc.prop_kind) {
        case forward: status = execute_forward(); break;
        case backward_data: status = execute_backward_data(); break;
        default:  assert(0 && "invalid prop_kind"); // should never happen
        }
        _exec_state = done;
        return status;
    }

public:
    typedef typename precision2type<prec>::type data_t;

    reference_pooling(const pooling_primitive_desc_t &ppd,
            const primitive_at_t *inputs, primitive *outputs[])
        : primitive(ppd, const_cast<impl::engine*>(ppd.base.engine), not_ready)
        , _ppd(_primitive_desc.pooling)
    {
        _input.push_back(inputs[0]);
        _input.push_back(inputs[1]);
        _output.push_back(outputs[0]);
    }
    ~reference_pooling() {}

    /* static magic */
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkl_dnn::impl::engine &aengine);
    static status_t create(primitive **aprimitive,
            const primitive_desc_t *primitive_desc,
            const primitive_at_t inputs[], primitive *outputs[]);
    static const primitive_impl implementation;
};

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

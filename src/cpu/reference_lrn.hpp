#ifndef CPU_REFERENCE_LRN_HPP
#define CPU_REFERENCE_LRN_HPP

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
class reference_lrn: public primitive {
private:
    const impl::lrn_primitive_desc_t &_ppd;

    // TODO: implement in cpp.
    status_t execute_forward();
    status_t execute_backward_data();

protected:
    status_t execute_impl() {
        status_t status = success;
        _exec_state = busy;
        switch (_ppd.lrn_desc.prop_kind) {
        case forward: status = execute_forward(); break;
        case backward_data: status = execute_backward_data(); break;
        default:  assert(0 && "invalid prop_kind"); // should never happen
        }
        _exec_state = done;
        return status;
    }

public:
    typedef typename precision2type<prec>::type data_t;

    reference_lrn(const lrn_primitive_desc_t &ppd,
            const primitive_at_t *inputs, const primitive *outputs[])
        : primitive(ppd, const_cast<impl::engine*>(ppd.base.engine), not_ready)
        , _ppd(_primitive_desc.lrn)
    {
        _input.push_back(inputs[0]);
        _input.push_back(inputs[1]);
        _output.push_back(outputs[0]);
    }
    ~reference_lrn() {}

    /* static magic */
    static status_t primitive_desc_init(primitive_desc_t *primitive_desc,
            const op_desc_t &op_desc, const mkl_dnn::impl::engine &aengine);
    static const primitive_impl implementation;
};

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

#ifndef MKL_DNN_HPP
#define MKL_DNN_HPP

#include <memory>
#include <stdexcept>
#include <vector>

// TODO: do we need a separate namespace to avoid pollution?

#include "mkl_dnn.h"

namespace mkl_dnn {

// TODO: do we need a special exception class to encapsulate a status as well?

struct primitive {
    typedef std::remove_pointer<primitive_t>::type primitive_val_t;
    std::shared_ptr<primitive_val_t> data;
    primitive(primitive_t aprimitive = nullptr): data(aprimitive, primitive_destroy) {}
    // TODO: other manupulation functions and operators
};

struct engine {
    typedef std::remove_pointer<dnn_engine_t>::type engine_val_t;
    std::shared_ptr<engine_val_t> data;
    enum kind {
        any = engine_kind_any,
        automatic = engine_kind_automatic,
        cpu = engine_kind_cpu,
        cpu_lazy = engine_kind_cpu_lazy,
        last = engine_kind_last,
    };
    static engine_kind_t convert_to_c(kind akind) {
        return static_cast<engine_kind_t>(akind);
    }
    static size_t get_count(kind akind) {
        return engine_get_count(convert_to_c(akind));
    }
    engine(const engine &other): data(other.data) {}
    explicit engine(kind akind, size_t index) {
        dnn_engine_t anengine;
        if (engine_create(&anengine, convert_to_c(akind), index) != success)
            throw std::runtime_error("Could not create engine");
        data = std::shared_ptr<engine_val_t>(anengine, engine_destroy);
    }
};

struct memory: public primitive  {
    enum format {
        any = memory_format_any,
        automatic = memory_format_automatic,
        n_f32 = memory_format_n_f32,
        nchw_f32 = memory_format_nchw_f32,
        oihw_f32 = memory_format_oihw_f32,
        nhwc_f32 = memory_format_nhwc_f32,
        nChw8_f32 = memory_format_nChw8_f32,
        IOhw88_f32 = memory_format_IOhw88_f32,
        nChw16_f32 = memory_format_nChw16_f32,
        IOhw1616_f32 = memory_format_IOhw1616_f32,
        blocked_f32 = memory_format_blocked_f32,
    };
    static memory_format_t convert_to_c(format aformat) {
        return static_cast<memory_format_t>(aformat);
    }

    struct tensor_desc {
        tensor_desc_t data;
        // TODO: convenience overloads
        tensor_desc(uint32_t batch, uint32_t channels,
                uint32_t spatial, std::vector<uint32_t> dims) {
            if (tensor_desc_init(&data, batch, channels, spatial,
                        &dims[0]) != success)
                throw std::runtime_error("Could not initialize a tensor descriptor");
        }
    };

    struct desc {
        memory_desc_t data;
        desc(const tensor_desc &atensor_desc, format aformat) {
            if (memory_desc_init(&data, &atensor_desc.data,
                        convert_to_c(aformat)) != success)
                throw std::runtime_error("Could not initialize a memory descriptor");
        }
    };

    struct primitive_desc {
        memory_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            if (memory_primitive_desc_init(&data,
                        &adesc.data, aengine.data.get()) != success)
                throw std::runtime_error("Could not inittialize a memory primitive descriptor");
        }
    };

    memory(const primitive_desc &adesc, void *input = nullptr) {
        primitive_t result;
        if (memory_create(&result, &adesc.data, input) != success)
            throw std::runtime_error("Could not create a memory primitive");
        data.reset(result, primitive_destroy);
    }
};

}
#endif

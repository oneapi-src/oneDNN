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
    typedef std::remove_pointer<mkl_dnn_primitive_t>::type primitive_val_t;
    std::shared_ptr<primitive_val_t> data;
    primitive(mkl_dnn_primitive_t aprimitive = nullptr):
        data(aprimitive, mkl_dnn_primitive_destroy) {}
    // TODO: other manupulation functions and operators
};

struct engine {
    typedef std::remove_pointer<mkl_dnn_engine_t>::type engine_val_t;
    std::shared_ptr<engine_val_t> data;
    enum kind {
        any = mkl_dnn_any_engine,
        cpu = mkl_dnn_cpu,
        cpu_lazy = mkl_dnn_cpu_lazy,
    };
    static mkl_dnn_engine_kind_t convert_to_c(kind akind) {
        return static_cast<mkl_dnn_engine_kind_t>(akind);
    }
    static size_t get_count(kind akind) {
        return mkl_dnn_engine_get_count(convert_to_c(akind));
    }
    engine(const engine &other): data(other.data) {}
    explicit engine(kind akind, size_t index) {
        mkl_dnn_engine_t anengine;
        if (mkl_dnn_engine_create(&anengine, convert_to_c(akind), index)
				!= mkl_dnn_success)
            throw std::runtime_error("Could not create engine");
        data = std::shared_ptr<engine_val_t>(anengine, mkl_dnn_engine_destroy);
    }
};

struct memory: public primitive  {
    enum format {
        any = mkl_dnn_any_f32,
        n_f32 = mkl_dnn_n_f32,
        nchw_f32 = mkl_dnn_nchw_f32,
        oihw_f32 = mkl_dnn_oihw_f32,
        nhwc_f32 = mkl_dnn_nhwc_f32,
        nChw8_f32 = mkl_dnn_nChw8_f32,
        IOhw88_f32 = mkl_dnn_IOhw88_f32,
        nChw16_f32 = mkl_dnn_nChw16_f32,
        IOhw1616_f32 = mkl_dnn_IOhw1616_f32,
        blocked_f32 = mkl_dnn_blocked_f32,
    };
    static mkl_dnn_memory_format_t convert_to_c(format aformat) {
        return static_cast<mkl_dnn_memory_format_t>(aformat);
    }

    struct tensor_desc {
        mkl_dnn_tensor_desc_t data;
        // TODO: convenience overloads
        tensor_desc(uint32_t batch, uint32_t channels,
                uint32_t spatial, std::vector<uint32_t> dims) {
            if (mkl_dnn_tensor_desc_init(&data, batch, channels, spatial,
                        &dims[0]) != mkl_dnn_success)
                throw std::runtime_error("Could not initialize a tensor descriptor");
        }
    };

    struct desc {
        mkl_dnn_memory_desc_t data;
        desc(const tensor_desc &atensor_desc, format aformat) {
            if (mkl_dnn_memory_desc_init(&data, &atensor_desc.data,
                        convert_to_c(aformat)) != mkl_dnn_success)
                throw std::runtime_error("Could not initialize a memory descriptor");
        }
    };

    struct primitive_desc {
        mkl_dnn_memory_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            if (mkl_dnn_memory_primitive_desc_init(&data,
                        &adesc.data, aengine.data.get()) != mkl_dnn_success)
                throw std::runtime_error("Could not inittialize a memory primitive descriptor");
        }
    };

    memory(const primitive_desc &adesc, void *input = nullptr) {
        mkl_dnn_primitive_t result;
        if (mkl_dnn_memory_create(&result, &adesc.data, input) != mkl_dnn_success)
            throw std::runtime_error("Could not create a memory primitive");
        data.reset(result, mkl_dnn_primitive_destroy); // XXX: must be a method
    }
};

}
#endif

#ifndef MKL_DNN_HPP
#define MKL_DNN_HPP

#include <memory>
#include <vector>

namespace mkl_dnn {

namespace c_api {
#include "mkl_dnn.h"
}

struct error {
    c_api::mkl_dnn_status_t status;
    std::string message;
    error(c_api::mkl_dnn_status_t astatus, std::string amessage)
        : status(astatus)
        , message(amessage) {}
    static void wrap_c_api(c_api::mkl_dnn_status_t status,
            std::string message) {
        if (status != c_api::mkl_dnn_success)
            throw error(status, message);
    }
};


struct primitive {
    typedef std::remove_pointer<c_api::mkl_dnn_primitive_t>::type
        primitive_val_t;
    std::shared_ptr<primitive_val_t> data;
    void set_data(c_api::mkl_dnn_primitive_t aprimitive)
    { data.reset(aprimitive, c_api::mkl_dnn_primitive_destroy); }
    primitive(c_api::mkl_dnn_primitive_t aprimitive = nullptr)
    { set_data(aprimitive); }
    // TODO: other manupulation functions and operators
};

struct engine {
    typedef std::remove_pointer<c_api::mkl_dnn_engine_t>::type engine_val_t;
    std::shared_ptr<engine_val_t> data;
    enum kind {
        any = c_api::mkl_dnn_any_engine,
        cpu = c_api::mkl_dnn_cpu,
        cpu_lazy = c_api::mkl_dnn_cpu_lazy,
    };
    static c_api::mkl_dnn_engine_kind_t convert_to_c(kind akind) {
        return static_cast<c_api::mkl_dnn_engine_kind_t>(akind);
    }
    static size_t get_count(kind akind) {
        return c_api::mkl_dnn_engine_get_count(convert_to_c(akind));
    }
    engine(const engine &other): data(other.data) {}
    explicit engine(kind akind, size_t index) {
        c_api::mkl_dnn_engine_t aengine;
        error::wrap_c_api(
                c_api::mkl_dnn_engine_create(&aengine,
                    convert_to_c(akind), index),
                "could not create engine");
        data.reset(aengine, c_api::mkl_dnn_engine_destroy);
    }
};

struct tensor {
    typedef std::vector<std::remove_extent<c_api::mkl_dnn_dims_t>::type> dims;
    typedef std::vector<std::remove_extent<c_api::mkl_dnn_nd_pos_t>::type> nd_pos;
    typedef std::vector<std::remove_extent<c_api::mkl_dnn_nd_offset_t>::type> nd_offset;
    template <typename T> static bool validate_len(std::vector<T> v)
    { return v.size() <= TENSOR_MAX_DIMS; } // TODO: throw from here?
    struct desc {
        c_api::mkl_dnn_tensor_desc_t data;
        // TODO: convenience overloads
        desc(uint32_t batch, uint32_t channels,
                uint32_t spatial, dims adims) {
            // TODO: check dims.size() via validate_len()
            error::wrap_c_api(
                    c_api::mkl_dnn_tensor_desc_init(&data,
                        batch, channels, spatial, &adims[0]),
                    "could not initialize a tensor descriptor");
        }
    };

};

struct memory: public primitive  {
    enum format {
        any = c_api::mkl_dnn_any_f32,
        n_f32 = c_api::mkl_dnn_n_f32,
        nchw_f32 = c_api::mkl_dnn_nchw_f32,
        oihw_f32 = c_api::mkl_dnn_oihw_f32,
        nhwc_f32 = c_api::mkl_dnn_nhwc_f32,
        nChw8_f32 = c_api::mkl_dnn_nChw8_f32,
        IOhw88_f32 = c_api::mkl_dnn_IOhw88_f32,
        nChw16_f32 = c_api::mkl_dnn_nChw16_f32,
        IOhw1616_f32 = c_api::mkl_dnn_IOhw1616_f32,
        blocked_f32 = c_api::mkl_dnn_blocked_f32,
    };
    static c_api::mkl_dnn_memory_format_t convert_to_c(format aformat) {
        return static_cast<c_api::mkl_dnn_memory_format_t>(aformat);
    }

    struct desc {
        c_api::mkl_dnn_memory_desc_t data;
        desc(const tensor::desc &atensor_desc, format aformat) {
            error::wrap_c_api(
                    c_api::mkl_dnn_memory_desc_init(&data,
                        &atensor_desc.data, convert_to_c(aformat)),
                    "could not initialize a memory descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkl_dnn_memory_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(
                    c_api::mkl_dnn_memory_primitive_desc_init(&data,
                        &adesc.data, aengine.data.get()),
                    "could not inittialize a memory primitive descriptor");
        }
    };

    memory(const primitive_desc &adesc, void *input = nullptr) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(
                c_api::mkl_dnn_memory_create(&result, &adesc.data, input),
                "could not create a memory primitive");
        set_data(result);
    }
};

}
#endif

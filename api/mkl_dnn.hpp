#ifndef MKL_DNN_HPP
#define MKL_DNN_HPP

#include <memory>
#include <vector>
#include <algorithm>

namespace mkl_dnn {

namespace c_api {
#include "mkl_dnn.h"
}


struct primitive {
    typedef std::remove_pointer<c_api::mkl_dnn_primitive_t>::type
        primitive_val_t;
    std::shared_ptr<primitive_val_t> data;
    void set_data(c_api::mkl_dnn_primitive_t aprimitive)
    { data.reset(aprimitive, c_api::mkl_dnn_primitive_destroy); }
    primitive(c_api::mkl_dnn_primitive_t aprimitive = nullptr)
    { set_data(aprimitive); }
    // TODO: other manupulation functions and operators
    struct at {
        c_api::mkl_dnn_primitive_at_t data;
        at(const primitive &aprimitive, size_t at = 0)
            : data(c_api::mkl_dnn_primitive_at(aprimitive.data.get(), at)) { }
    };
};

struct error {
    c_api::mkl_dnn_status_t status;
    std::string message;
    primitive error_primitive;
    error(c_api::mkl_dnn_status_t astatus, std::string amessage,
            primitive aerror_primitive = nullptr)
        : status(astatus)
        , message(amessage)
        , error_primitive(aerror_primitive) {}
    static void wrap_c_api(c_api::mkl_dnn_status_t status,
            std::string message, primitive aerror_primitive = nullptr) {
        if (status != c_api::mkl_dnn_success)
            throw error(status, message, aerror_primitive);
    }
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
        primitive_desc() {} // XXX: should be private? should take C type?
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(
                    c_api::mkl_dnn_memory_primitive_desc_init(&data,
                        &adesc.data, aengine.data.get()),
                    "could not inittialize a memory primitive descriptor");
        }
        bool operator==(const primitive_desc &other) {
            return mkl_dnn_memory_primitive_desc_equal(&data, &other.data);
        }
    };

    memory(const primitive_desc &adesc, void *input = nullptr) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(
                c_api::mkl_dnn_memory_create(&result, &adesc.data, input),
                "could not create a memory primitive");
        set_data(result);
    }
    primitive_desc get_primitive_desc() {
        primitive_desc desc;
        error::wrap_c_api(c_api::mkl_dnn_memory_get_primitive_desc(data.get(),
                    &desc.data),
                "could not get primitive descriptor from a memory primitive");
        return desc;
    }
};

enum padding_kind {
    zero = c_api::mkl_dnn_padding_zero,
};
inline c_api::mkl_dnn_padding_kind_t convert_to_c(padding_kind kind) {
    return static_cast<c_api::mkl_dnn_padding_kind_t>(kind);
}

enum prop_kind {
    forward = c_api::mkl_dnn_forward,
    backward_data = c_api::mkl_dnn_backward_data,
    backward_weights,g = c_api::mkl_dnn_backward_bias,
    backward_bias = c_api::mkl_dnn_backward_bias,
};
inline c_api::mkl_dnn_prop_kind_t convert_to_c(prop_kind kind) {
    return static_cast<c_api::mkl_dnn_prop_kind_t>(kind);
}

struct convolution: public primitive {
    enum algorithm { direct = c_api::mkl_dnn_convolution_direct };
    static c_api::mkl_dnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
        return static_cast<c_api::mkl_dnn_alg_kind_t>(aalgorithm);
    }
    struct desc {
        c_api::mkl_dnn_convolution_desc_t data;
        // XXX: convolution w/o bias
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &input_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &output_desc,
                const tensor::dims strides,
                const tensor::nd_pos padding,
                const padding_kind apadding_kind)
        {
            // TODO: check vector lengths
            error::wrap_c_api(c_api::mkl_dnn_convolution_desc_init(&data,
                        mkl_dnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &input_desc.data, &weights_desc.data, &bias_desc.data,
                        &output_desc.data, &strides[0], &padding[0],
                        mkl_dnn::convert_to_c(apadding_kind)),
                    "could not create a convolution descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkl_dnn_convolution_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkl_dnn_convolution_primitive_desc_init(
                        &data, &adesc.data, aengine.data.get()),
                    "could not create a convolution primitive descriptor");
        }
    };

    // XXX: convolution w/o bias
    convolution(const primitive_desc &aprimitive_desc,
            const primitive::at &input, const primitive::at &weights,
            const primitive::at &bias, const memory &output) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_convolution_create(&result,
                    &aprimitive_desc.data, input.data, weights.data,
                    bias.data, output.data.get()),
                "could not create a convolution primitive");
        set_data(result);
    }
};

struct stream {
    typedef std::remove_pointer<c_api::mkl_dnn_stream_t>::type stream_val_t;
    std::shared_ptr<stream_val_t> data;
    stream() {
        c_api::mkl_dnn_stream_t astream;
        error::wrap_c_api(c_api::mkl_dnn_stream_create(&astream),
                "could not create a stream");
        data.reset(astream, c_api::mkl_dnn_stream_destroy);
    }
    stream &submit(std::vector<primitive> primitives) {
        c_api::mkl_dnn_primitive_t error_primitive;
        // TODO: find a proper way to convert vector<primitive> to
        // vector<c_api::mkl_dnn_primitive_t>
        std::vector<c_api::mkl_dnn_primitive_t> c_api_primitives;
        c_api_primitives.reserve(primitives.size());
        std::transform(primitives.begin(), primitives.end(),
                std::back_inserter(c_api_primitives),
                [](primitive p) { return p.data.get(); });
        error::wrap_c_api(c_api::mkl_dnn_stream_submit(data.get(),
                    c_api_primitives.size(), &c_api_primitives[0],
                    &error_primitive),
                "could not submit primitives to a stream",
                error_primitive);
        return *this;
    }
    stream &wait(bool block = true) {
        c_api::mkl_dnn_primitive_t error_primitive;
        error::wrap_c_api(c_api::mkl_dnn_stream_wait(data.get(),
                    block, &error_primitive),
                "could not wait on a stream");
        return *this;
    }
};

}

#endif

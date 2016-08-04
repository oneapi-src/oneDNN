#ifndef MKL_DNN_HPP
#define MKL_DNN_HPP

#include <assert.h>
#include <memory>
#include <vector>
#include <algorithm>

namespace mkl_dnn {

template <typename T> class handle_traits {};
template <typename T, typename traits=handle_traits<T>> class handle {
private:
    std::shared_ptr<typename std::remove_pointer<T>::type> _data;
    handle(const handle &&) {}
    handle &operator=(const handle &&other) {}
protected:
    void reset(T t, bool own = true) {
        auto dummy_destructor = [](T) { return decltype(traits::destructor(0))(0); };
        _data.reset(t, own ? traits::destructor : dummy_destructor);
    }
    handle(T t = 0, bool own = true): _data(0) { reset(t, own); }

    bool operator==(const T other) const { return other == _data.get(); }
    bool operator!=(const T other) const { return !(*this == other); }
public:
    handle(const handle &other): _data(other._data) {}
    handle &operator=(const handle &other) {
        _data = other._data;
        return *this;
    }

    T get() const { return _data.get(); }

    bool operator==(const handle &other) const { return other._data.get() == _data.get(); }
    bool operator!=(const handle &other) const { return !(*this == other); }
};

namespace c_api {
#include "mkl_dnn.h"
}

template <> struct handle_traits<c_api::mkl_dnn_primitive_t> {
    static constexpr auto destructor = &c_api::mkl_dnn_primitive_destroy;
};

class primitive: public handle<c_api::mkl_dnn_primitive_t> {
    friend struct error;
    friend struct stream;
    friend class primitive_at;
    using handle::handle;
public:
    struct at {
        c_api::mkl_dnn_primitive_at_t data;
        at(const primitive &aprimitive, size_t at = 0)
            : data(c_api::mkl_dnn_primitive_at(aprimitive.get(), at)) {}
        inline operator primitive() const;
    };

    inline c_api::mkl_dnn_primitive_desc_t get_primitive_desc() const;
};

struct error {
    c_api::mkl_dnn_status_t status;
    std::string message;
    primitive error_primitive;

    error(c_api::mkl_dnn_status_t astatus, std::string amessage,
            c_api::mkl_dnn_primitive_t aerror_primitive = 0)
        : status(astatus)
        , message(amessage)
        , error_primitive(aerror_primitive, false)
    {}

    static void wrap_c_api(c_api::mkl_dnn_status_t status,
            std::string message,
            c_api::mkl_dnn_primitive_t *error_primitive = 0)
    {
        // XXX: do we really need this ^ pointer ?
        if (status != c_api::mkl_dnn_success)
            throw error(status, message, *error_primitive);
    }
};

// XXX: reorder me with struct error :(
inline primitive::at::operator primitive() const {
    c_api::mkl_dnn_primitive_t output;
    error::wrap_c_api(
            c_api::mkl_dnn_primitive_get_output(data.primitive,
                data.output_index, &output),
            "could not get an output primitive");
    return primitive(output, false);
}

inline c_api::mkl_dnn_primitive_desc_t primitive::get_primitive_desc() const {
    c_api::mkl_dnn_primitive_desc_t pd;
    error::wrap_c_api(mkl_dnn_primitive_get_primitive_desc(get(), &pd),
            "could not get primive descriptor by primitive");
    return pd;
}

template <> struct handle_traits<c_api::mkl_dnn_engine_t> {
    static constexpr auto destructor = &c_api::mkl_dnn_engine_destroy;
};

struct engine: public handle<c_api::mkl_dnn_engine_t> {
    friend class primitive;
    using handle::handle;

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

    explicit engine(const c_api::mkl_dnn_engine_t& aengine)
        : handle(aengine, false) {}
    explicit engine(const c_api::const_mkl_dnn_engine_t& aengine)
        : handle(const_cast<const c_api::mkl_dnn_engine_t>(aengine), false) {} // XXX: cast Roma
    explicit engine(kind akind, size_t index) {
        c_api::mkl_dnn_engine_t aengine;
        error::wrap_c_api(
                c_api::mkl_dnn_engine_create(&aengine,
                    convert_to_c(akind), index),
                "could not create an engine");
        reset(aengine);
    }
};

struct tensor {
    typedef std::vector<std::remove_extent<c_api::mkl_dnn_dims_t>::type> dims;
    typedef std::vector<std::remove_extent<c_api::mkl_dnn_nd_offset_t>::type> nd_offset;

    template <typename T> static void validate_dims(std::vector<T> v) {
        if (v.size() > TENSOR_MAX_DIMS)
            throw error(c_api::mkl_dnn_invalid_arguments,
                    "invalid dimensions");
    }

    struct desc {
        c_api::mkl_dnn_tensor_desc_t data;
        desc(uint32_t batch, uint32_t channels,
                uint32_t spatial, dims adims) {
            validate_dims(adims);
            error::wrap_c_api(
                    c_api::mkl_dnn_tensor_desc_init(&data,
                        batch, channels, spatial, &adims[0]),
                    "could not initialize a tensor descriptor");
        }
        // TODO: convenience overloads
    };

};

struct memory: public primitive  {
    enum precision {
        f32 = c_api::mkl_dnn_f32
    };
    static c_api::mkl_dnn_precision_t convert_to_c(precision aprecision) {
        return static_cast<c_api::mkl_dnn_precision_t>(aprecision);
    }
    enum format {
        any = c_api::mkl_dnn_any,
        n = c_api::mkl_dnn_n,
        nchw = c_api::mkl_dnn_nchw,
        nhwc = c_api::mkl_dnn_nhwc,
        nChw8c = c_api::mkl_dnn_nChw8c,
        oihw = c_api::mkl_dnn_oihw,
        OIhw8i8o = c_api::mkl_dnn_OIhw8i8o,
        goihw = c_api::mkl_dnn_goihw,
        gOIhw8i8o = c_api::mkl_dnn_gOIhw8i8o,
        blocked = c_api::mkl_dnn_blocked,
    };
    static c_api::mkl_dnn_memory_format_t convert_to_c(format aformat) {
        return static_cast<c_api::mkl_dnn_memory_format_t>(aformat);
    }

    struct desc {
        c_api::mkl_dnn_memory_desc_t data;
        desc(c_api::mkl_dnn_memory_desc_t adata): data(adata) {} // XXX: cast Roma
        desc(const tensor::desc &atensor_desc, precision aprecision,
                format aformat) {
            error::wrap_c_api(
                    c_api::mkl_dnn_memory_desc_init(&data,
                        &atensor_desc.data, convert_to_c(aprecision),
                        convert_to_c(aformat)),
                    "could not initialize a memory descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkl_dnn_memory_primitive_desc_t data;
        primitive_desc() {} // XXX: should be private? should take C type?
        primitive_desc(c_api::mkl_dnn_memory_primitive_desc_t adata)
            : data(adata) {} // XXX: cast Roma
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(
                    c_api::mkl_dnn_memory_primitive_desc_init(&data,
                        &adesc.data, aengine.get()),
                    "could not inittialize a memory primitive descriptor");
        }
        memory::desc desc() const { return memory::desc(data.memory_desc); } // XXX: cast Roma
        bool operator==(const primitive_desc &other) {
            return mkl_dnn_memory_primitive_desc_equal(&data, &other.data);
        }
        bool operator!=(const primitive_desc &other) {
            return (mkl_dnn_memory_primitive_desc_equal(&data, &other.data) == 0);
        }
    };

    memory(const primitive &aprimitive): primitive(aprimitive) {} // XXX: cast Roma
    memory(const primitive_desc &adesc, void *input = nullptr) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(
                c_api::mkl_dnn_memory_create(&result, &adesc.data, input),
                "could not create a memory primitive");
        reset(result);
    }
    primitive_desc get_primitive_desc() const {
        primitive_desc adesc;
        error::wrap_c_api(c_api::mkl_dnn_memory_get_primitive_desc(get(),
                    &adesc.data),
                "could not get primitive descriptor from a memory primitive");
        return adesc;
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
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const tensor::dims strides,
                const tensor::nd_offset padding,
                const padding_kind apadding_kind)
        {
            tensor::validate_dims(strides);
            tensor::validate_dims(padding);
            error::wrap_c_api(c_api::mkl_dnn_convolution_desc_init(&data,
                        mkl_dnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &padding[0],
                        mkl_dnn::convert_to_c(apadding_kind)),
                    "could not create a convolution descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkl_dnn_convolution_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkl_dnn_convolution_primitive_desc_init(
                        &data, &adesc.data, aengine.get()),
                    "could not create a convolution primitive descriptor");
        }
    };

    // XXX: convolution w/o bias
    convolution(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_convolution_create(&result,
                    &aprimitive_desc.data, src.data, weights.data,
                    bias.data, dst.get()),
                "could not create a convolution primitive");
        reset(result);
    }

    convolution(prop_kind aprop_kind, algorithm aalgorithm,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst,
            const tensor::dims strides, const tensor::nd_offset padding,
            const padding_kind apadding_kind) {
        auto src_md = memory(src).get_primitive_desc();
        auto weights_md = memory(weights).get_primitive_desc();
        auto bias_md = memory(bias).get_primitive_desc();
        auto dst_md = dst.get_primitive_desc();

        auto conv_d = desc(aprop_kind, aalgorithm, src_md.desc(),
                weights_md.desc(), bias_md.desc(), dst_md.desc(),
                strides, padding, apadding_kind);
        auto conv_pd = primitive_desc(conv_d,
                engine(src_md.data.base.engine));
#if 0
        // WHY I CANNOT CALL THIS??????!!!!
        convolution(conv_pd, src, weights, bias, dst);
#else
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_convolution_create(&result,
                    &conv_pd.data, src.data, weights.data,
                    bias.data, dst.get()),
                "could not create a convolution primitive");
        reset(result);
#endif
    }
};

struct pooling : public primitive {
    enum algorithm { max = c_api::mkl_dnn_pooling_max };
    static c_api::mkl_dnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
        return static_cast<c_api::mkl_dnn_alg_kind_t>(aalgorithm);
    }
    struct desc {
        c_api::mkl_dnn_pooling_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            const memory::desc &indices_desc,
            const memory::desc &dst_desc,
            const tensor::dims strides,
            const tensor::dims kernel,
            const tensor::nd_offset padding,
            const padding_kind apadding_kind)
        {
            tensor::validate_dims(strides);
            tensor::validate_dims(kernel);
            tensor::validate_dims(padding);
            error::wrap_c_api(c_api::mkl_dnn_pooling_desc_init(&data,
                mkl_dnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data, &indices_desc.data,
                &dst_desc.data, &strides[0], &kernel[0], &padding[0],
                mkl_dnn::convert_to_c(apadding_kind)),
                "could not create a pooling descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkl_dnn_pooling_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkl_dnn_pooling_primitive_desc_init(
                &data, &adesc.data, aengine.get()),
                "could not create a pooling primitive descriptor");
        }
    };

    pooling(const primitive_desc &aprimitive_desc,
        const primitive::at &src, const primitive::at &indices,
        const memory &dst) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_pooling_create(&result,
            &aprimitive_desc.data, src.data, indices.data,
            dst.get()),
            "could not create a pooling primitive");
        reset(result);
    }

    pooling(prop_kind aprop_kind, algorithm aalgorithm,
        const primitive::at &src, const primitive::at &indices,
        const memory &dst,
        const tensor::dims strides, const tensor::dims kernel, const tensor::nd_offset padding,
        const padding_kind apadding_kind) {
        auto src_md = memory(src).get_primitive_desc();
        auto indices_md = memory(indices).get_primitive_desc();
        auto dst_md = dst.get_primitive_desc();

        auto pool_d = desc(aprop_kind, aalgorithm, src_md.desc(),
            indices_md.desc(), dst_md.desc(),
            strides, kernel, padding, apadding_kind);
        auto pool_pd = primitive_desc(pool_d,
            engine(src_md.data.base.engine));

        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_pooling_create(&result,
            &pool_pd.data, src.data, indices.data,
            dst.get()),
            "could not create a convolution primitive");
        reset(result);
    }
};

struct reorder: public primitive {
    struct primitive_desc {
        c_api::mkl_dnn_reorder_primitive_desc_t data;
        primitive_desc(const memory::primitive_desc &ainput,
                const memory::primitive_desc &aoutput) {
            error::wrap_c_api(c_api::mkl_dnn_reorder_primitive_desc_init(
                        &data, &ainput.data, &aoutput.data),
                    "could not create a reorder primitive descriptor");
        }
    };

    reorder(const primitive_desc &aprimitive_desc,
            const primitive::at &input, const memory &output) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_reorder_create(&result,
                    &aprimitive_desc.data, input.data, output.get()),
                "could not create a reorder primitive");
        reset(result);
    }
};

template <> struct handle_traits<c_api::mkl_dnn_stream_t> {
    static constexpr auto destructor = &c_api::mkl_dnn_stream_destroy;
};

struct stream: public handle<c_api::mkl_dnn_stream_t> {
    using handle::handle;

    explicit stream() {
        c_api::mkl_dnn_stream_t astream;
        error::wrap_c_api(c_api::mkl_dnn_stream_create(&astream),
                "could not create a stream");
        reset(astream);
    }

    stream &submit(std::vector<primitive> primitives) {
        // TODO: find a proper way to convert vector<primitive> to
        // vector<c_api::mkl_dnn_primitive_t>
        std::vector<c_api::mkl_dnn_primitive_t> c_api_primitives;
        c_api_primitives.reserve(primitives.size());
        auto convert_to_c = [](primitive p) { return p.get(); };
        std::transform(primitives.begin(), primitives.end(),
                std::back_inserter(c_api_primitives), convert_to_c);

        c_api::mkl_dnn_primitive_t c_api_error_primitive;
        error::wrap_c_api(
                c_api::mkl_dnn_stream_submit(get(),
                    c_api_primitives.size(), &c_api_primitives[0],
                    &c_api_error_primitive),
                "could not submit primitives to a stream",
                &c_api_error_primitive);

        return *this;
    }

    stream &wait(bool block = true) {
        c_api::mkl_dnn_primitive_t c_api_error_primitive;
        error::wrap_c_api(c_api::mkl_dnn_stream_wait(get(), block,
                    &c_api_error_primitive), "could not wait on a stream",
                &c_api_error_primitive);
        return *this;
    }
};

}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s,g0

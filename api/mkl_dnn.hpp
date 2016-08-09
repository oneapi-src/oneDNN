#ifndef MKL_DNN_HPP
#define MKL_DNN_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <assert.h>
#include <memory>
#include <vector>
#include <algorithm>
#endif

namespace mkl_dnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_api_utils Utils
/// @{

/// A traits class that provides the destructor for an MKL-DNN C handle
template <typename T> class handle_traits {};

/// A class for wrapping an MKL-DNN handle. It is used as the base for
/// primitive (#mkl_dnn_primitive_t), engine (#mkl_dnn_engine_t) and stream
/// (#mkl_dnn_stream_t) handles. An object of the #mkl_dnn::handle class can be
/// passed by value. This class allows wrapping both newly constructed and
/// pre-existing handles returned by the MKL-DNN C API (for example, via
/// #mkl_dnn_primitive_get_output()). In the first case, the constructed handle
/// uses reference counting provided by @p std::shared_ptr with a proper
/// deleter function specified via the @p handle_traits class. In the second
/// case, an MKL-DNN C API handle is wrapped without a deleter because it is
/// assume that the handle wrapper for the original object will delete the
/// handle (this model is similar to @p std::weak_ptr)
template <typename T, typename traits=handle_traits<T>> class handle {
private:
    std::shared_ptr<typename std::remove_pointer<T>::type> _data;
    handle(const handle &&) {}
    handle &operator=(const handle &&other) {}
protected:
    /// Resets the C handle value.
    /// @param t the new C handle value
    /// @param weak whether to destroy @p t when reference count reaches zero
    void reset(T t, bool weak = false) {
        auto dummy_destructor = [](T) { return decltype(traits::destructor(0))(0); };
        _data.reset(t, weak ? dummy_destructor: traits::destructor);
    }
    /// Constructs a C handle wrapper.
    /// @param t the C handle to wrap
    /// @param set_destructor whether to destroy @p t when reference count
    ///                       reaches zero
    handle(T t = 0, bool weak = false): _data(0) {
        reset(t, weak);
    }

    bool operator==(const T other) const { return other == _data.get(); }
    bool operator!=(const T other) const { return !(*this == other); }
public:
    handle(const handle &other): _data(other._data) {}
    handle &operator=(const handle &other) {
        _data = other._data;
        return *this;
    }

    /// Returns the underlayng C handle value
    T get() const { return _data.get(); }

    bool operator==(const handle &other) const { return other._data.get() == _data.get(); }
    bool operator!=(const handle &other) const { return !(*this == other); }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace c_api {
#include "mkl_dnn.h"
}
#endif

template <> struct handle_traits<c_api::mkl_dnn_primitive_t> {
    static constexpr auto destructor = &c_api::mkl_dnn_primitive_destroy;
};

/// Base class for all computational primitives
class primitive: public handle<c_api::mkl_dnn_primitive_t> {
    friend struct error;
    friend struct stream;
    friend class primitive_at;
    using handle::handle;
public:
    /// A wrapper structure to addess a specific output of a primitive.
    struct at {
        /// The underlying C API data structure
        c_api::mkl_dnn_primitive_at_t data;
        /// Constructs a wrapper specifying @p aprimitive output with index @p
        /// at.
        /// @param aprimitive the target primitive
        /// @param at the output index
        at(const primitive &aprimitive, size_t at = 0)
            : data(c_api::mkl_dnn_primitive_at(aprimitive.get(), at)) {}
        /// Returns the primitive addressed by this instance.
        inline operator primitive() const;
    };

    /// Returns the C API primitive descriptor of this primitive
    inline c_api::mkl_dnn_primitive_desc_t get_primitive_desc() const;
    // TODO: change to return a C++ object
};

/// MKL-DNN exception class.
///
/// This class captures a status returned by the failed C API function, an
/// error message, and, in some cases, the handle of the primitive that caused
/// the error.
struct error: public std::exception {
    c_api::mkl_dnn_status_t status;
    std::string message;
    primitive error_primitive;

    /// Constructs an error instance.
    ///
    /// @param astatus the error status returned by the C API
    /// @param amessage the error message
    /// @param aerror_primitive (optional) the C handle of primitive a that
    ///                         caused the error
    error(c_api::mkl_dnn_status_t astatus, std::string amessage,
            c_api::mkl_dnn_primitive_t aerror_primitive = 0)
        : status(astatus)
        , message(amessage)
        , error_primitive(aerror_primitive, true)
    {}

    /// A convenience function to wrapping calls to the C API. Checks for
    /// return status and throws an #error in case of failure.
    ///
    /// @param status the error status returned by the C API
    /// @param message the error message
    /// @param error_primitive (optional) the C handle of primitive a that
    ///                         caused the error
    static void wrap_c_api(c_api::mkl_dnn_status_t status,
            std::string message,
            c_api::mkl_dnn_primitive_t *error_primitive = 0)
    {
        if (status != c_api::mkl_dnn_success)
            throw error(status, message, *error_primitive);
    }
};

inline primitive::at::operator primitive() const {
    c_api::const_mkl_dnn_primitive_t output;
    error::wrap_c_api(
            c_api::mkl_dnn_primitive_get_output(data.primitive,
                data.output_index, &output),
            "could not get an output primitive");
    return primitive(const_cast<c_api::mkl_dnn_primitive_t>(output), true);
}

inline c_api::mkl_dnn_primitive_desc_t primitive::get_primitive_desc() const {
    c_api::mkl_dnn_primitive_desc_t pd;
    error::wrap_c_api(mkl_dnn_primitive_get_primitive_desc(get(), &pd),
            "could not get primitive descriptor by primitive");
    return pd;
}

/// @}

/// @addtogroup cpp_api_engine Engine
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<c_api::mkl_dnn_engine_t> {
    static constexpr auto destructor = &c_api::mkl_dnn_engine_destroy;
};
#endif

/// An execution engine.
struct engine: public handle<c_api::mkl_dnn_engine_t> {
    friend class primitive;
    using handle::handle;

    /// Engine kind
    enum kind {
        /// An unspecified engine
        any = c_api::mkl_dnn_any_engine,
        /// CPU engine
        cpu = c_api::mkl_dnn_cpu,
        /// CPU engine in a lazy execution mode
        cpu_lazy = c_api::mkl_dnn_cpu_lazy,
    };

    /// Returns the number of engines of a certain kind.
    ///
    /// @param akind kind of engines to count
    static size_t get_count(kind akind) {
        return c_api::mkl_dnn_engine_get_count(convert_to_c(akind));
    }

    /// Constructs an engine.
    ///
    /// @param akind the kind of engine to construct
    /// @param index the index of the engine; must be less than the value
    ///              returned by #get_count() for this particular engine kind
    engine(kind akind, size_t index) {
        c_api::mkl_dnn_engine_t aengine;
        error::wrap_c_api(
                c_api::mkl_dnn_engine_create(&aengine,
                    convert_to_c(akind), index),
                "could not create an engine");
        reset(aengine);
    }

    explicit engine(const c_api::mkl_dnn_engine_t& aengine)
        : handle(aengine, true) {}

    explicit engine(const c_api::const_mkl_dnn_engine_t& aengine)
        : handle(const_cast<const c_api::mkl_dnn_engine_t>(aengine), true) {}

private:
    static c_api::mkl_dnn_engine_kind_t convert_to_c(kind akind) {
        return static_cast<c_api::mkl_dnn_engine_kind_t>(akind);
    }
};

/// @}

/// @addtogroup cpp_api_memory Memory
/// @{

/// Tensor. Incapsulates a tensor description. The description is not tied to
/// any memory format, but allows describing tensor dimensions as belonging to
/// minibatch, channel/feature map, and spatial kind. MKL-DNN uses this type
/// when a mathematical description of data is required.
struct tensor {
    typedef std::vector<std::remove_extent<c_api::mkl_dnn_dims_t>::type> dims;
    typedef std::vector<std::remove_extent<c_api::mkl_dnn_nd_offset_t>::type> nd_offset;

    /// Checks that a vector specifying tensor dimensions is valid.
    ///
    /// @param v the vector to check
    /// @returns nothing; throws an #mkl_dnn::error exception if v is not valid
    template <typename T> static void validate_dims(std::vector<T> v) {
        if (v.size() > TENSOR_MAX_DIMS)
            throw error(c_api::mkl_dnn_invalid_arguments,
                    "invalid dimensions");
    }

    /// A tensor descriptor.
    struct desc {
        /// The underlying C API data structure
        c_api::mkl_dnn_tensor_desc_t data;

        /// Constructs a tensor descriptor.
        ///
        /// @param adims the tensor dimensions
        desc(dims adims) {
            validate_dims(adims);
            error::wrap_c_api(
                    c_api::mkl_dnn_tensor_desc_init(&data, adims.size(),
                        &adims[0]),
                    "could not initialize a tensor descriptor");
        }

        // TODO: add convenience overloads
    };

};

/// Memory primitive that describes data
struct memory: public primitive  {

    /// Data type specification; see #mkl_dnn_precision_t for a detailed
    /// description
    enum precision {
        precision_undef = c_api::mkl_dnn_precision_undef,
        f32 = c_api::mkl_dnn_f32,
        u32 = c_api::mkl_dnn_u32,
    };

    /// Memory format specification; see #mkl_dnn_memory_format_t for a
    /// detailed description.
    enum format {
        format_undef = c_api::mkl_dnn_format_undef,
        any = c_api::mkl_dnn_any,
        blocked = c_api::mkl_dnn_blocked,
        x = c_api::mkl_dnn_x,
        nc = c_api::mkl_dnn_nc,
        nchw = c_api::mkl_dnn_nchw,
        nhwc = c_api::mkl_dnn_nhwc,
        nChw8c = c_api::mkl_dnn_nChw8c,
        oi = c_api::mkl_dnn_oi,
        oihw = c_api::mkl_dnn_oihw,
        OIhw8i8o = c_api::mkl_dnn_OIhw8i8o,
        goihw = c_api::mkl_dnn_goihw,
        gOIhw8i8o = c_api::mkl_dnn_gOIhw8i8o,
    };

    /// A memory descriptor.
    struct desc {
        friend class memory;

        /// The underlying C API data structure
        c_api::mkl_dnn_memory_desc_t data;

        /// Constructs a memory descriptor.
        ///
        /// @param atensor_desc a tensor descriptor
        /// @param aprecision data precision/type
        /// @param aformat data layout format
        desc(const tensor::desc &atensor_desc, precision aprecision,
                format aformat) {
            error::wrap_c_api(
                    c_api::mkl_dnn_memory_desc_init(&data,
                        &atensor_desc.data, convert_to_c(aprecision),
                        convert_to_c(aformat)),
                    "could not initialize a memory descriptor");
        }

        /// Constructs a memory descriptor from an C API data structure
        ///
        /// @param adata a C API #mkl_dnn_memory_desc_t structure
        desc(c_api::mkl_dnn_memory_desc_t adata): data(adata) {}
        // TODO: make private

        /// @returns the number of data elements in the memory described.
        ///
        /// @param with_padding whether to consider the padding area in the
        ///                     computations
        size_t get_number_of_elements(bool with_padding = false) const {
            return c_api::mkl_dnn_memory_desc_get_number_of_elements(&data,
                    static_cast<int>(with_padding));
        }

        /// @returns the number of bytes required to allocate memory described
        /// including the padding area.
        size_t get_size() const {
            return c_api::mkl_dnn_memory_desc_get_size(&data);
        }
    };

    /// A memory primitive descriptor.
    struct primitive_desc {
        friend class memory;
        /// The underlying C API data structure
        c_api::mkl_dnn_memory_primitive_desc_t data;

        // TODO: make private
        primitive_desc() {}
        primitive_desc(c_api::mkl_dnn_memory_primitive_desc_t adata)
                : data(adata) {}

        /// Constructs a memory primitive descriptor.
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(
                    c_api::mkl_dnn_memory_primitive_desc_init(&data,
                        &adesc.data, aengine.get()),
                    "could not inittialize a memory primitive descriptor");
        }

        /// @returns the corresponding memory descriptor
        memory::desc desc() const { return memory::desc(data.memory_desc); }

        /// @returns the number of data elements in the memory described.
        ///
        /// @param with_padding whether to consider the padding area in the
        ///                     computations
        size_t get_number_of_elements(bool with_padding = false) const {
            return desc().get_number_of_elements(with_padding);
        }

        /// @returns the number of bytes required to allocate memory described
        /// including the padding area.
        size_t get_size() const { return desc().get_size(); }

        bool operator==(const primitive_desc &other) const {
            return mkl_dnn_memory_primitive_desc_equal(&data, &other.data);
        }

        bool operator!=(const primitive_desc &other) const {
            return !operator==(other);
        }
    };

    /// Constructs a memory primitive from a gneric primitive.
    ///
    /// @param aprimitive the primitive to treat as memory
    memory(const primitive &aprimitive): primitive(aprimitive) {}
    // TODO: remove as not type-safe

    /// Constructs a memory primitive.
    ///
    /// @param adesc memory primitive descriptor
    /// @param input pointer to previously allocated data; if null, then memory
    ///              gets allocated by the library
    memory(const primitive_desc &adesc, void *input = nullptr) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(
                c_api::mkl_dnn_memory_create(&result, &adesc.data, input),
                "could not create a memory primitive");
        reset(result);
    }

    /// @returns memory primitive descriptor for this memory primitive.
    primitive_desc get_primitive_desc() const {
        primitive_desc adesc;
        error::wrap_c_api(c_api::mkl_dnn_memory_get_primitive_desc(get(),
                    &adesc.data),
                "could not get primitive descriptor from a memory primitive");
        return adesc;
    }

    /// @returns a handle to the data contained in this memory primitive. On
    /// the CPU engine, this is a pointer to allocated memory.
    inline void *get_data_handle() const {
        void *handle;
        error::wrap_c_api(mkl_dnn_memory_get_data_handle(get(), &handle),
                "could not get native handle");
        return handle;
    }

    // Must go away or be private:
    static c_api::mkl_dnn_precision_t convert_to_c(precision aprecision) {
        return static_cast<c_api::mkl_dnn_precision_t>(aprecision);
    }
    static c_api::mkl_dnn_memory_format_t convert_to_c(format aformat) {
        return static_cast<c_api::mkl_dnn_memory_format_t>(aformat);
    }

};

/// @}

/// @addtogroup cpp_api_primitives Primitives
/// @{

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

        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const tensor::dims strides,
                const tensor::nd_offset padding,
                const padding_kind apadding_kind)
        {
            tensor::validate_dims(strides);
            tensor::validate_dims(padding);
            error::wrap_c_api(c_api::mkl_dnn_convolution_desc_init(&data,
                        mkl_dnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
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

    convolution(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_convolution_create(&result,
                    &aprimitive_desc.data, src.data, weights.data,
                    {nullptr, 0}, dst.get()),
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
                weights_md.desc(), bias_md.desc(), dst_md.desc(), strides,
                padding, apadding_kind);
        auto conv_pd = primitive_desc(conv_d, engine(src_md.data.base.engine));

        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_convolution_create(&result,
                    &conv_pd.data, src.data, weights.data, bias.data,
                    dst.get()),
                "could not create a convolution primitive");
        reset(result);
    }

    convolution(prop_kind aprop_kind, algorithm aalgorithm,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst, const tensor::dims strides,
            const tensor::nd_offset padding, const padding_kind apadding_kind) {
        auto src_md = memory(src).get_primitive_desc();
        auto weights_md = memory(weights).get_primitive_desc();
        auto dst_md = dst.get_primitive_desc();

        auto conv_d = desc(aprop_kind, aalgorithm, src_md.desc(),
                weights_md.desc(), dst_md.desc(), strides, padding,
                apadding_kind);
        auto conv_pd = primitive_desc(conv_d, engine(src_md.data.base.engine));

        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_convolution_create(&result,
                    &conv_pd.data, src.data, weights.data, {nullptr, 0},
                    dst.get()),
                "could not create a convolution primitive");
        reset(result);
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
                &src_desc.data,
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
        auto dst_md = dst.get_primitive_desc();

        auto pool_d = desc(aprop_kind, aalgorithm, src_md.desc(),
            dst_md.desc(), strides, kernel, padding, apadding_kind);
        auto pool_pd = primitive_desc(pool_d,
            engine(src_md.data.base.engine));

        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_pooling_create(&result,
            &pool_pd.data, src.data, indices.data,
            dst.get()),
            "could not create a pooling primitive");
        reset(result);
    }
};

struct lrn : public primitive {
    enum algorithm {
        across_channels = c_api::mkl_dnn_lrn_across_channels,
        within_channel  = c_api::mkl_dnn_lrn_within_channel,
    };
    static c_api::mkl_dnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
        return static_cast<c_api::mkl_dnn_alg_kind_t>(aalgorithm);
    }
    struct desc {
        c_api::mkl_dnn_lrn_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            const memory::desc &dst_desc,
            double alpha, double beta, uint32_t local_size)
        {
            error::wrap_c_api(c_api::mkl_dnn_lrn_desc_init(&data,
                mkl_dnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data, &dst_desc.data, alpha, beta, local_size),
                "could not create a lrn descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkl_dnn_lrn_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkl_dnn_lrn_primitive_desc_init(
                &data, &adesc.data, aengine.get()),
                "could not create a lrn primitive descriptor");
        }
    };

    lrn(const primitive_desc &aprimitive_desc,
        const primitive::at &src, const primitive::at &scratch,
        const memory &dst) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_lrn_create(&result,
            &aprimitive_desc.data, src.data, scratch.data,
            dst.get()),
            "could not create a lrn primitive");
        reset(result);
    }

    lrn(prop_kind aprop_kind, algorithm aalgorithm,
        const primitive::at &src, const primitive::at &scratch,
        const memory &dst,
        double alpha, double beta, uint32_t local_size) {
        auto src_md = memory(src).get_primitive_desc();
        auto dst_md = dst.get_primitive_desc();

        auto lrn_d = desc(aprop_kind, aalgorithm,
            src_md.desc(), dst_md.desc(),
            alpha, beta, local_size);
        auto lrn_pd = primitive_desc(lrn_d,
            engine(src_md.data.base.engine));

        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_lrn_create(&result,
            &lrn_pd.data, src.data, scratch.data,
            dst.get()),
            "could not create a lrn primitive");
        reset(result);
    }
};

struct reorder : public primitive {
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

    reorder(const primitive::at &input, const memory &output) {
        auto input_md = memory(input).get_primitive_desc();
        auto output_md = output.get_primitive_desc();

        auto reorder_d = primitive_desc(input_md, output_md);

        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_reorder_create(&result,
                    &reorder_d.data, input.data, output.get()),
                "could not create a reorder primitive");
        reset(result);
    }
};

struct relu: public primitive {
    struct desc {
        c_api::mkl_dnn_relu_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, T negative_slope,
                const memory::desc &src_desc,
                const memory::desc &dst_desc)
        {
            error::wrap_c_api(c_api::mkl_dnn_relu_desc_init(&data,
                        mkl_dnn::convert_to_c(aprop_kind),
                        static_cast<double>(negative_slope),
                        &src_desc.data, &dst_desc.data),
                    "could not create a relu descriptor");
        }
    };
    struct primitive_desc {
        c_api::mkl_dnn_relu_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkl_dnn_relu_primitive_desc_init(&data,
                        &adesc.data, aengine.get()),
                    "could not create a relu primitive descriptor");
        }
    };
    relu(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_relu_create(&result,
                    &aprimitive_desc.data, src.data, dst.get()),
                "could not create a relu primitive");
        reset(result);
    }
};

struct inner_product: public primitive {
    struct desc {
        c_api::mkl_dnn_inner_product_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
             const memory::desc &weights_desc, const memory::desc &bias_desc,
             const memory::desc &dst_desc)
        {
            error::wrap_c_api(c_api::mkl_dnn_inner_product_desc_init(&data,
                        mkl_dnn::convert_to_c(aprop_kind),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data),
                    "could not create a inner product descriptor");
        }

        desc(prop_kind aprop_kind, const memory::desc &src_desc,
             const memory::desc &weights_desc, const memory::desc &dst_desc)
        {
            error::wrap_c_api(c_api::mkl_dnn_inner_product_desc_init(&data,
                        mkl_dnn::convert_to_c(aprop_kind),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data),
                    "could not create a inner product descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkl_dnn_inner_product_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkl_dnn_inner_product_primitive_desc_init(
                        &data, &adesc.data, aengine.get()),
                    "could not create a inner product primitive descriptor");
        }
    };

    inner_product(prop_kind aprop_kind,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst) {
        auto src_md = memory(src).get_primitive_desc();
        auto weights_md = memory(weights).get_primitive_desc();
        auto bias_md = memory(bias).get_primitive_desc();
        auto dst_md = dst.get_primitive_desc();

        auto ip_d = desc(aprop_kind, src_md.desc(), weights_md.desc(),
                bias_md.desc(), dst_md.desc());
        auto ip_pd = primitive_desc(ip_d, engine(src_md.data.base.engine));
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_inner_product_create(&result,
                    &ip_pd.data, src.data, weights.data, bias.data, dst.get()),
                "could not create a inner product primitive");
        reset(result);
    }

    inner_product(prop_kind aprop_kind,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst) {
        auto src_md = memory(src).get_primitive_desc();
        auto weights_md = memory(weights).get_primitive_desc();
        auto dst_md = dst.get_primitive_desc();

        auto ip_d = desc(aprop_kind, src_md.desc(), weights_md.desc(),
                dst_md.desc());
        auto ip_pd = primitive_desc(ip_d, engine(src_md.data.base.engine));
        c_api::mkl_dnn_primitive_t result;
        error::wrap_c_api(c_api::mkl_dnn_inner_product_create(&result,
                    &ip_pd.data, src.data, weights.data, {nullptr, 0},
                    dst.get()),
                "could not create a inner product primitive");
        reset(result);
    }
};

/// @}

/// @addtogroup cpp_api_stream Stream
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<c_api::mkl_dnn_stream_t> {
    static constexpr auto destructor = &c_api::mkl_dnn_stream_destroy;
};
#endif

struct stream: public handle<c_api::mkl_dnn_stream_t> {
    using handle::handle;

    /// Constructs a stream
    stream() {
        c_api::mkl_dnn_stream_t astream;
        error::wrap_c_api(c_api::mkl_dnn_stream_create(&astream),
                "could not create a stream");
        reset(astream);
    }

    /// Submits a vector of primitives to a stream for computations.
    ///
    /// @param primitives the vector of primitives to submit
    /// @returns the stream itself
    stream &submit(std::vector<primitive> primitives) {
        // TODO: find a proper way to convert a vector<primitive> to
        //       a vector<c_api::mkl_dnn_primitive_t>
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

    /// Waits for all computations submited to the stream to complete.
    ///
    /// @param block whether the operation should wait indefinetly or return
    ///              immediately.
    /// @returns true if all compuptaions have completed, false otherwise
    bool wait(bool block = true) {
        c_api::mkl_dnn_primitive_t c_api_error_primitive;
        c_api::mkl_dnn_status_t status = c_api::mkl_dnn_stream_wait(get(),
                block, &c_api_error_primitive);
        if (status != c_api::mkl_dnn_success
                && status != c_api::mkl_dnn_try_again)
            error::wrap_c_api(status, "could not wait on a stream",
                    &c_api_error_primitive);
        return (status == c_api::mkl_dnn_success);
    }
};

/// @}

/// @}

}


#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s,g0

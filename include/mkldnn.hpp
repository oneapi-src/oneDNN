//==============================================================================
// Copyright 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================

#ifndef MKLDNN_HPP
#define MKLDNN_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <assert.h>
#include <memory>
#include <vector>
#include <algorithm>
#endif

namespace mkldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_api_utils Utils
/// @{

/// A class that provides the destructor for an Intel(R) MKL-DNN C handle
template <typename T> class handle_traits {};

/// A class for wrapping an Intel(R) MKL-DNN handle. It is used as the base
/// class for primitive (#mkldnn_primitive_t), engine (#mkldnn_engine_t), and
/// stream (#mkldnn_stream_t) handles. An object of the #mkldnn::handle class
/// can be passed by value. This class enables wrapping:
///  - Newly constructed handles.
///    @n In this case, the constructed handle uses reference counting provided
///    by @p std::shared_ptr with a proper deleter function specified through
///    the @p handle_traits class.
///  - Pre-existing handles returned by the Intel(R) MKL-DNN C API (for
///    example, through #mkldnn_primitive_get_output()).
///    @n In this case, an Intel(R) MKL-DNN C API handle is wrapped without a
///    deleter because it is assumed that the handle wrapper for the original
///    object deletes the handle (this model is similar to @p std::weak_ptr).
template <typename T, typename traits=handle_traits<T>> class handle {
private:
    std::shared_ptr<typename std::remove_pointer<T>::type> _data;
    handle(const handle &&) {}
    handle &operator=(const handle &&other) {}
protected:
    /// Resets the value of a C handle.
    /// @param t The new value of the C handle.
    /// @param weak A flag to specify whether the wrapper should be weak.
    void reset(T t, bool weak = false) {
        auto dummy_destructor = [](T) { return decltype(traits::destructor(0))(0); };
        _data.reset(t, weak ? dummy_destructor: traits::destructor);
    }
    /// Constructs a C handle wrapper.
    /// @param t The C handle to wrap.
    /// @param weak A flag to specify whether to construct a weak wrapper.
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

    /// Returns the value of the underlying C handle.
    T get() const { return _data.get(); }

    bool operator==(const handle &other) const { return other._data.get() == _data.get(); }
    bool operator!=(const handle &other) const { return !(*this == other); }
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace c_api {
#include "mkldnn.h"
}
#endif

template <> struct handle_traits<c_api::mkldnn_primitive_t> {
    static constexpr auto destructor = &c_api::mkldnn_primitive_destroy;
};

/// Base class for all computational primitives.
class primitive: public handle<c_api::mkldnn_primitive_t> {
    friend struct error;
    friend struct stream;
    friend class primitive_at;
    using handle::handle;
public:
    /// A wrapper structure to specify a particular output of a primitive.
    struct at {
        /// The underlying C API structure.
        c_api::mkldnn_primitive_at_t data;
        /// Constructs a wrapper specifying @p aprimitive output with index @p
        /// at.
        ///
        /// @param aprimitive The target primitive.
        /// @param at The output index.

        at(const primitive &aprimitive, size_t at = 0)
            : data(c_api::mkldnn_primitive_at(aprimitive.get(), at)) {}
        /// Returns the specified output.
        inline operator primitive() const;
    };



    /// Returns the descriptor of the underlying C API primitive
    inline c_api::mkldnn_primitive_desc_t get_primitive_desc() const;
    // TODO: use the C++ API wrapper structure.
};

/// Intel(R) MKL-DNN exception class.
///
/// This class captures the status returned by the failed C API function, error
/// message, and, optionally, handle of the primitive that caused the error.
struct error: public std::exception {
    c_api::mkldnn_status_t status;
    std::string message;
    primitive error_primitive;

    /// Constructs an error instance.
    ///
    /// @param astatus The error status returned by the C API.
    /// @param amessage The error message.
    /// @param aerror_primitive (optional) A C handle of the primitive that
    ///                         caused the error.

    error(c_api::mkldnn_status_t astatus, std::string amessage,
            c_api::mkldnn_primitive_t aerror_primitive = 0)
        : status(astatus)
        , message(amessage)
        , error_primitive(aerror_primitive, true)
    {}

    /// A convenience function for wrapping calls to the C API. Checks the
    /// return status and throws an #error in case of failure.
    ///
    /// @param status The error status returned by the C API.
    /// @param message The error message.
    /// @param error_primitive (optional) A C handle of the primitive that
    ///                        caused the error.

    static void wrap_c_api(c_api::mkldnn_status_t status,
            std::string message,
            c_api::mkldnn_primitive_t *error_primitive = 0)
    {
        if (status != c_api::mkldnn_success) {
            if (nullptr != error_primitive)
                throw error(status, message, *error_primitive);
            else
                throw error(status, message, nullptr);
        }
    }
};

inline primitive::at::operator primitive() const {
    c_api::const_mkldnn_primitive_t output;
    error::wrap_c_api(
            c_api::mkldnn_primitive_get_output(data.primitive,
                data.output_index, &output),
            "could not get an output primitive");
    return primitive(const_cast<c_api::mkldnn_primitive_t>(output), true);
}

inline c_api::mkldnn_primitive_desc_t primitive::get_primitive_desc() const {
    c_api::mkldnn_primitive_desc_t pd;
    error::wrap_c_api(mkldnn_primitive_get_primitive_desc(get(), &pd),
            "could not get primitive descriptor by primitive");
    return pd;
}

/// @}

/// @addtogroup cpp_api_engine Engine
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<c_api::mkldnn_engine_t> {
    static constexpr auto destructor = &c_api::mkldnn_engine_destroy;
};
#endif

/// An execution engine.
struct engine: public handle<c_api::mkldnn_engine_t> {
    friend class primitive;
    using handle::handle;

    /// Kinds of engines
    enum kind {
        /// An unspecified engine
        any = c_api::mkldnn_any_engine,
        /// CPU engine
        cpu = c_api::mkldnn_cpu,
        /// CPU engine in a lazy execution mode
        cpu_lazy = c_api::mkldnn_cpu_lazy,
    };

    /// Returns the number of engines of a certain kind.
    ///
    /// @param akind The kind of engines to count.

    static size_t get_count(kind akind) {
        return c_api::mkldnn_engine_get_count(convert_to_c(akind));
    }

    /// Constructs an engine.
    ///
    /// @param akind The kind of engine to construct.
    /// @param index The index of the engine. Must be less than the value
    ///              returned by #get_count() for this particular kind of engine.

    engine(kind akind, size_t index) {
        c_api::mkldnn_engine_t aengine;
        error::wrap_c_api(
                c_api::mkldnn_engine_create(&aengine,
                    convert_to_c(akind), index),
                "could not create an engine");
        reset(aengine);
    }

    explicit engine(const c_api::mkldnn_engine_t& aengine)
        : handle(aengine, true) {}

    explicit engine(const c_api::const_mkldnn_engine_t& aengine)
        : handle(const_cast<const c_api::mkldnn_engine_t>(aengine), true) {}

private:
    static c_api::mkldnn_engine_kind_t convert_to_c(kind akind) {
        return static_cast<c_api::mkldnn_engine_kind_t>(akind);
    }
};

/// @}

/// @addtogroup cpp_api_memory Memory
/// @{

/// Tensor. Incapsulates a tensor description. The description is not tied to
/// any memory format, but enables describing tensor dimensions as having the
/// mini-batch, channel/feature map, and spatial kind. Intel(R) MKL-DNN uses
/// this type when a mathematical description of data is required.
struct tensor {
    typedef std::vector<std::remove_extent<c_api::mkldnn_dims_t>::type> dims;
    typedef std::vector<std::remove_extent<c_api::mkldnn_nd_offset_t>::type> nd_offset;

    /// Checks that a vector specifying tensor dimensions is valid.
    ///
    /// @param v The vector to check.
    /// @returns Nothing, throws an #mkldnn::error exception if @p v is not valid.

    template <typename T> static void validate_dims(std::vector<T> v) {
        if (v.size() > TENSOR_MAX_DIMS)
            throw error(c_api::mkldnn_invalid_arguments,
                    "invalid dimensions");
    }

    /// A tensor descriptor.
    struct desc {
        /// The underlying C API data structure.
        c_api::mkldnn_tensor_desc_t data;

        /// Constructs a tensor descriptor.
        ///
        /// @param adims Tensor dimensions.
        desc(dims adims) {
            validate_dims(adims);
            error::wrap_c_api(
                    c_api::mkldnn_tensor_desc_init(&data, adims.size(),
                        &adims[0]),
                    "could not initialize a tensor descriptor");
        }

        // TODO: convenience overloads
    };

};

/// Memory primitive that describes the data.
struct memory: public primitive  {

    /// Data type specification. See #mkldnn_precision_t for a detailed
    /// description.
    enum precision {
        precision_undef = c_api::mkldnn_precision_undef,
        f32 = c_api::mkldnn_f32,
        u32 = c_api::mkldnn_u32,
    };

    /// Memory format specification. See #mkldnn_memory_format_t
    /// for a detailed description.
    enum format {
        format_undef = c_api::mkldnn_format_undef,
        any = c_api::mkldnn_any,
        blocked = c_api::mkldnn_blocked,
        x = c_api::mkldnn_x,
        nc = c_api::mkldnn_nc,
        nchw = c_api::mkldnn_nchw,
        nhwc = c_api::mkldnn_nhwc,
        nChw8c = c_api::mkldnn_nChw8c,
        oi = c_api::mkldnn_oi,
        oihw = c_api::mkldnn_oihw,
        oIhw8i = c_api::mkldnn_oIhw8i,
        OIhw8i8o = c_api::mkldnn_OIhw8i8o,
        Ohwi8o = c_api::mkldnn_Ohwi8o,
        goihw = c_api::mkldnn_goihw,
        gOIhw8i8o = c_api::mkldnn_gOIhw8i8o,
    };

    /// A memory descriptor.
    struct desc {
        friend struct memory;

        /// The underlying C API data structure.
        c_api::mkldnn_memory_desc_t data;

        /// Constructs a memory descriptor.
        ///
        /// @param atensor_desc A tensor descripto.r
        /// @param aprecision Data precision/type.
        /// @param aformat Data layout format.
        desc(const tensor::desc &atensor_desc, precision aprecision,
                format aformat) {
            error::wrap_c_api(
                    c_api::mkldnn_memory_desc_init(&data,
                        &atensor_desc.data, convert_to_c(aprecision),
                        convert_to_c(aformat)),
                    "could not initialize a memory descriptor");
        }

        /// Constructs a memory descriptor from a C API data structure.
        ///
        /// @param adata A C API #mkldnn_memory_desc_t structure.
        desc(const c_api::mkldnn_memory_desc_t &adata): data(adata) {}
        // TODO: make private

        /// Returns the number of data elements in the memory described.
        ///
        /// @param with_padding A flag to specify whether to consider the padding area
        ///                     in the computations.
        size_t get_number_of_elements(bool with_padding = false) const {
            return c_api::mkldnn_memory_desc_get_number_of_elements(&data,
                    static_cast<int>(with_padding));
        }

        /// Returns the number of bytes required to allocate the memory described
        /// including the padding area.
        size_t get_size() const {
            return c_api::mkldnn_memory_desc_get_size(&data);
        }
    };

    /// A memory primitive descriptor.
    struct primitive_desc {
        friend struct memory;
        /// The underlying C API data structure.
        c_api::mkldnn_memory_primitive_desc_t data;

        // TODO: make private
        primitive_desc() {}
        primitive_desc(const c_api::mkldnn_memory_primitive_desc_t &adata)
                : data(adata) {}

        /// Constructs a memory primitive descriptor.
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(
                    c_api::mkldnn_memory_primitive_desc_init(&data,
                        &adesc.data, aengine.get()),
                    "could not inittialize a memory primitive descriptor");
        }

        /// Returns the memory primitive descriptor.
        memory::desc desc() const { return memory::desc(data.memory_desc); }

        /// Returns the number of data elements in the memory described.
        ///
        /// @param with_padding A flag to specify whether to consider the padding area
        ///                     in the computations.
        size_t get_number_of_elements(bool with_padding = false) const {
            return desc().get_number_of_elements(with_padding);
        }

        /// Returns the number of bytes required to allocate the memory described
        /// including the padding area.
        size_t get_size() const { return desc().get_size(); }

        bool operator==(const primitive_desc &other) const {
            return mkldnn_memory_primitive_desc_equal(&data, &other.data);
        }

        bool operator!=(const primitive_desc &other) const {
            return !operator==(other);
        }
    };

    /// Constructs a memory primitive from a generic primitive.
    ///
    /// @param aprimitive The primitive to treat as memory.
    memory(const primitive &aprimitive): primitive(aprimitive) {}
    // TODO: remove as not type-safe

    /// Constructs a memory primitive.
    ///
    /// @param adesc Memory primitive descriptor.
    /// @param input Pointer to previously allocated data. If @c NULL, the library
    ///              allocates the memory.
    memory(const primitive_desc &adesc, void *input = nullptr) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(
                c_api::mkldnn_memory_create(&result, &adesc.data, input),
                "could not create a memory primitive");
        reset(result);
    }

    /// Returns the descriptor of the memory primitive.
    primitive_desc get_primitive_desc() const {
        primitive_desc adesc;
        error::wrap_c_api(c_api::mkldnn_memory_get_primitive_desc(get(),
                    &adesc.data),
                "could not get primitive descriptor from a memory primitive");
        return adesc;
    }

    /// Returns a handle of the data contained in the memory primitive. On
    /// the CPU engine, this is a pointer to the allocated memory.
    inline void *get_data_handle() const {
        void *handle;
        error::wrap_c_api(mkldnn_memory_get_data_handle(get(), &handle),
                "could not get native handle");
        return handle;
    }

    // Must go away or be private:
    static c_api::mkldnn_precision_t convert_to_c(precision aprecision) {
        return static_cast<c_api::mkldnn_precision_t>(aprecision);
    }
    static c_api::mkldnn_memory_format_t convert_to_c(format aformat) {
        return static_cast<c_api::mkldnn_memory_format_t>(aformat);
    }

};

/// @}

/// @addtogroup cpp_api_primitives Primitives
/// @{

enum padding_kind {
    zero = c_api::mkldnn_padding_zero,
};
inline c_api::mkldnn_padding_kind_t convert_to_c(padding_kind kind) {
    return static_cast<c_api::mkldnn_padding_kind_t>(kind);
}

enum prop_kind {
    forward_training = c_api::mkldnn_forward_training,
    forward_scoring = c_api::mkldnn_forward_scoring,
    forward = c_api::mkldnn_forward,
    backward_data = c_api::mkldnn_backward_data,
    backward_weights,g = c_api::mkldnn_backward_bias,
    backward_bias = c_api::mkldnn_backward_bias,
};
inline c_api::mkldnn_prop_kind_t convert_to_c(prop_kind kind) {
    return static_cast<c_api::mkldnn_prop_kind_t>(kind);
}

struct convolution: public primitive {
    enum algorithm { direct = c_api::mkldnn_convolution_direct };
    static c_api::mkldnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
        return static_cast<c_api::mkldnn_alg_kind_t>(aalgorithm);
    }
    struct desc {
        c_api::mkldnn_convolution_desc_t data;
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
            error::wrap_c_api(c_api::mkldnn_convolution_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &padding[0],
                        mkldnn::convert_to_c(apadding_kind)),
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
            error::wrap_c_api(c_api::mkldnn_convolution_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &padding[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkldnn_convolution_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkldnn_convolution_primitive_desc_init(
                        &data, &adesc.data, aengine.get()),
                    "could not create a convolution primitive descriptor");
        }
    };

    convolution(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_convolution_create(&result,
                    &aprimitive_desc.data, src.data, weights.data,
                    bias.data, dst.get()),
                "could not create a convolution primitive");
        reset(result);
    }

    convolution(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_convolution_create(&result,
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

        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_convolution_create(&result,
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

        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_convolution_create(&result,
                    &conv_pd.data, src.data, weights.data, {nullptr, 0},
                    dst.get()),
                "could not create a convolution primitive");
        reset(result);
    }
};

struct pooling : public primitive {
    enum algorithm { max = c_api::mkldnn_pooling_max };
    static c_api::mkldnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
        return static_cast<c_api::mkldnn_alg_kind_t>(aalgorithm);
    }
    struct desc {
        c_api::mkldnn_pooling_desc_t data;
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
            error::wrap_c_api(c_api::mkldnn_pooling_desc_init(&data,
                mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data,
                &dst_desc.data, &strides[0], &kernel[0], &padding[0],
                mkldnn::convert_to_c(apadding_kind)),
                "could not create a pooling descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkldnn_pooling_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkldnn_pooling_primitive_desc_init(
                &data, &adesc.data, aengine.get()),
                "could not create a pooling primitive descriptor");
        }
    };

    pooling(const primitive_desc &aprimitive_desc,
        const primitive::at &src, const primitive::at &indices,
        const memory &dst) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_pooling_create(&result,
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

        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_pooling_create(&result,
            &pool_pd.data, src.data, indices.data,
            dst.get()),
            "could not create a pooling primitive");
        reset(result);
    }
};

struct lrn : public primitive {
    enum algorithm {
        across_channels = c_api::mkldnn_lrn_across_channels,
        within_channel  = c_api::mkldnn_lrn_within_channel,
    };
    static c_api::mkldnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
        return static_cast<c_api::mkldnn_alg_kind_t>(aalgorithm);
    }
    struct desc {
        c_api::mkldnn_lrn_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            const memory::desc &dst_desc,
            double alpha, double beta, uint32_t local_size)
        {
            error::wrap_c_api(c_api::mkldnn_lrn_desc_init(&data,
                mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data, &dst_desc.data, alpha, beta, local_size),
                "could not create a lrn descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkldnn_lrn_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkldnn_lrn_primitive_desc_init(
                &data, &adesc.data, aengine.get()),
                "could not create a lrn primitive descriptor");
        }
    };

    lrn(const primitive_desc &aprimitive_desc,
        const primitive::at &src, const primitive::at &scratch,
        const memory &dst) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_lrn_create(&result,
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

        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_lrn_create(&result,
            &lrn_pd.data, src.data, scratch.data,
            dst.get()),
            "could not create a lrn primitive");
        reset(result);
    }
};

struct reorder : public primitive {
    struct primitive_desc {
        c_api::mkldnn_reorder_primitive_desc_t data;
        primitive_desc(const memory::primitive_desc &ainput,
                const memory::primitive_desc &aoutput) {
            error::wrap_c_api(c_api::mkldnn_reorder_primitive_desc_init(
                        &data, &ainput.data, &aoutput.data),
                    "could not create a reorder primitive descriptor");
        }
    };

    reorder(const primitive_desc &aprimitive_desc,
            const primitive::at &input, const memory &output) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_reorder_create(&result,
                    &aprimitive_desc.data, input.data, output.get()),
                "could not create a reorder primitive");
        reset(result);
    }

    reorder(const primitive::at &input, const memory &output) {
        auto input_md = memory(input).get_primitive_desc();
        auto output_md = output.get_primitive_desc();

        auto reorder_d = primitive_desc(input_md, output_md);

        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_reorder_create(&result,
                    &reorder_d.data, input.data, output.get()),
                "could not create a reorder primitive");
        reset(result);
    }
};

struct relu: public primitive {
    struct desc {
        c_api::mkldnn_relu_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, T negative_slope,
                const memory::desc &src_desc,
                const memory::desc &dst_desc)
        {
            error::wrap_c_api(c_api::mkldnn_relu_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        static_cast<double>(negative_slope),
                        &src_desc.data, &dst_desc.data),
                    "could not create a relu descriptor");
        }
    };
    struct primitive_desc {
        c_api::mkldnn_relu_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkldnn_relu_primitive_desc_init(&data,
                        &adesc.data, aengine.get()),
                    "could not create a relu primitive descriptor");
        }
    };
    relu(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_relu_create(&result,
                    &aprimitive_desc.data, src.data, dst.get()),
                "could not create a relu primitive");
        reset(result);
    }
    relu(prop_kind aprop_kind, double negative_slope,
            const primitive::at &src, const memory &dst) {
        auto src_md = memory(src).get_primitive_desc();
        auto dst_md = memory(dst).get_primitive_desc();

        auto relu_d = desc(aprop_kind, negative_slope, src_md.desc(),
                dst_md.desc());
        auto relu_pd = primitive_desc(relu_d, engine(src_md.data.base.engine));
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_relu_create(&result,
                    &relu_pd.data, src.data, dst.get()),
                "could not create a relu primitive");
        reset(result);
    }
};

struct inner_product: public primitive {
    struct desc {
        c_api::mkldnn_inner_product_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
             const memory::desc &weights_desc, const memory::desc &bias_desc,
             const memory::desc &dst_desc)
        {
            error::wrap_c_api(c_api::mkldnn_inner_product_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data),
                    "could not create a inner product descriptor");
        }

        desc(prop_kind aprop_kind, const memory::desc &src_desc,
             const memory::desc &weights_desc, const memory::desc &dst_desc)
        {
            error::wrap_c_api(c_api::mkldnn_inner_product_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data),
                    "could not create a inner product descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkldnn_inner_product_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(c_api::mkldnn_inner_product_primitive_desc_init(
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
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_inner_product_create(&result,
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
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_inner_product_create(&result,
                    &ip_pd.data, src.data, weights.data, {nullptr, 0},
                    dst.get()),
                "could not create a inner product primitive");
        reset(result);
    }
};

struct convolution_relu: public primitive {
    using algorithm = convolution::algorithm;

    struct desc {
        c_api::mkldnn_convolution_relu_desc_t data;
        desc(const convolution::desc conv_desc,
                const double negative_slope)
        {
            error::wrap_c_api(c_api::mkldnn_convolution_relu_desc_init(&data,
                        &conv_desc.data, negative_slope),
                    "could not create a convolution_relu descriptor");
        }
    };

    struct primitive_desc {
        c_api::mkldnn_convolution_relu_primitive_desc_t data;
        primitive_desc(const desc &adesc, const engine &aengine) {
            error::wrap_c_api(
                    c_api::mkldnn_convolution_relu_primitive_desc_init(
                        &data, &adesc.data, aengine.get()),
                    "could not create a convolution_relu primitive descriptor");
        }
    };

    convolution_relu(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_convolution_relu_create(&result,
                    &aprimitive_desc.data, src.data, weights.data,
                    bias.data, dst.get()),
                "could not create a convolution_relu primitive");
        reset(result);
    }

    convolution_relu(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_convolution_relu_create(&result,
                    &aprimitive_desc.data, src.data, weights.data,
                    {nullptr, 0}, dst.get()),
                "could not create a convolution_relu primitive");
        reset(result);
    }

    convolution_relu(prop_kind aprop_kind, algorithm aalgorithm,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst,
            const tensor::dims strides, const tensor::nd_offset padding,
            const padding_kind apadding_kind, const double negative_slope) {
        auto src_md = memory(src).get_primitive_desc();
        auto weights_md = memory(weights).get_primitive_desc();
        auto bias_md = memory(bias).get_primitive_desc();
        auto dst_md = dst.get_primitive_desc();

        auto conv_relu_d = desc({aprop_kind, aalgorithm, src_md.desc(),
                weights_md.desc(), bias_md.desc(), dst_md.desc(), strides,
                padding, apadding_kind}, negative_slope);
        auto conv_relu_pd = primitive_desc(conv_relu_d,
                engine(src_md.data.base.engine));

        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_convolution_relu_create(&result,
                    &conv_relu_pd.data, src.data, weights.data, bias.data,
                    dst.get()),
                "could not create a convolution_relu primitive");
        reset(result);
    }

    convolution_relu(prop_kind aprop_kind, algorithm aalgorithm,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst, const tensor::dims strides,
            const tensor::nd_offset padding, const padding_kind apadding_kind,
            const double negative_slope) {
        auto src_md = memory(src).get_primitive_desc();
        auto weights_md = memory(weights).get_primitive_desc();
        auto dst_md = dst.get_primitive_desc();

        auto conv_relu_d = desc({aprop_kind, aalgorithm, src_md.desc(),
                weights_md.desc(), dst_md.desc(), strides, padding,
                apadding_kind}, negative_slope);
        auto conv_relu_pd = primitive_desc(conv_relu_d,
                engine(src_md.data.base.engine));

        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(c_api::mkldnn_convolution_relu_create(&result,
                    &conv_relu_pd.data, src.data, weights.data, {nullptr, 0},
                    dst.get()),
                "could not create a convolution_relu primitive");
        reset(result);
    }
};

/// @}

/// @addtogroup cpp_api_stream Stream
/// @{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<c_api::mkldnn_stream_t> {
    static constexpr auto destructor = &c_api::mkldnn_stream_destroy;
};
#endif

struct stream: public handle<c_api::mkldnn_stream_t> {
    using handle::handle;

    /// Constructs a stream.
    stream() {
        c_api::mkldnn_stream_t astream;
        error::wrap_c_api(c_api::mkldnn_stream_create(&astream),
                "could not create a stream");
        reset(astream);
    }

    /// Submits a vector of primitives to a stream for computations.
    ///
    /// @param primitives The vector of primitives to submit.
    /// @returns The stream.
    stream &submit(std::vector<primitive> primitives) {
        // TODO: find a proper way to convert vector<primitive> to
        // vector<c_api::mkldnn_primitive_t>
        std::vector<c_api::mkldnn_primitive_t> c_api_primitives;
        c_api_primitives.reserve(primitives.size());
        auto convert_to_c = [](primitive p) { return p.get(); };
        std::transform(primitives.begin(), primitives.end(),
                std::back_inserter(c_api_primitives), convert_to_c);

        c_api::mkldnn_primitive_t c_api_error_primitive;
        error::wrap_c_api(
                c_api::mkldnn_stream_submit(get(),
                    c_api_primitives.size(), &c_api_primitives[0],
                    &c_api_error_primitive),
                "could not submit primitives to a stream",
                &c_api_error_primitive);

        return *this;
    }

    /// Waits for all computations submitted to the stream to complete.
    ///
    /// @param block Specifies whether the operation should wait indefinitely or return
    ///              immediately.
    /// @returns @c true if all computations completed.
    /// @returns @c false if not all computations completed.
    bool wait(bool block = true) {
        c_api::mkldnn_primitive_t c_api_error_primitive;
        c_api::mkldnn_status_t status = c_api::mkldnn_stream_wait(get(),
                block, &c_api_error_primitive);
        if (status != c_api::mkldnn_success
                && status != c_api::mkldnn_try_again)
            error::wrap_c_api(status, "could not wait on a stream",
                    &c_api_error_primitive);
        return (status == c_api::mkldnn_success);
    }
};

/// @}

/// @}

}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s,g0

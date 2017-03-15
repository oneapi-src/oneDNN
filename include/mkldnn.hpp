/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef MKLDNN_HPP
#define MKLDNN_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <assert.h>
#include <stdlib.h>
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
    handle &operator=(const handle &&other) = delete;
protected:
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
    /// Resets the value of a C handle.
    /// @param t The new value of the C handle.
    /// @param weak A flag to specify whether the wrapper should be weak.
    void reset(T t, bool weak = false) {
        auto dummy_destructor = [](T) { return decltype(traits::destructor(0))(0); };
        _data.reset(t, weak ? dummy_destructor : traits::destructor);
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
    inline c_api::const_mkldnn_primitive_desc_t get_primitive_desc() const;
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

c_api::const_mkldnn_primitive_desc_t primitive::get_primitive_desc() const {
    c_api::const_mkldnn_primitive_desc_t pd;
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

private:
    static c_api::mkldnn_engine_kind_t convert_to_c(kind akind) {
        return static_cast<c_api::mkldnn_engine_kind_t>(akind);
    }
};

/// @}

/// @addtogroup cpp_api_memory Memory
/// @{

template <> struct handle_traits<c_api::mkldnn_primitive_desc_t> {
    static constexpr auto destructor = &c_api::mkldnn_primitive_desc_destroy;
};

/// Memory primitive that describes the data.
struct memory: public primitive  {
    private:
    std::shared_ptr<char> _handle;

    public:
    typedef std::vector<std::remove_extent<c_api::mkldnn_dims_t>::type> dims;

    template <typename T> static void validate_dims(std::vector<T> v) {
        if (v.size() > TENSOR_MAX_DIMS)
            throw error(c_api::mkldnn_invalid_arguments,
                    "invalid dimensions");
    }

    /// Data type specification. See #mkldnn_data_type_t for a detailed
    /// description.
    enum data_type {
        data_undef = c_api::mkldnn_data_type_undef,
        f32 = c_api::mkldnn_f32,
        s32 = c_api::mkldnn_s32,
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
        chwn = c_api::mkldnn_chwn,
        nChw8c = c_api::mkldnn_nChw8c,
        nChw16c = c_api::mkldnn_nChw16c,
        oi = c_api::mkldnn_oi,
        io = c_api::mkldnn_io,
        oihw = c_api::mkldnn_oihw,
        ihwo = c_api::mkldnn_ihwo,
        oIhw8i = c_api::mkldnn_oIhw8i,
        oIhw16i = c_api::mkldnn_oIhw16i,
        OIhw8i8o = c_api::mkldnn_OIhw8i8o,
        OIhw16i16o = c_api::mkldnn_OIhw16i16o,
        OIhw8o8i = c_api::mkldnn_OIhw8o8i,
        OIhw16o16i = c_api::mkldnn_OIhw16o16i,
        Ohwi8o = c_api::mkldnn_Ohwi8o,
        Ohwi16o = c_api::mkldnn_Ohwi16o,
        goihw = c_api::mkldnn_goihw,
        gOIhw8i8o = c_api::mkldnn_gOIhw8i8o,
        gOIhw16i16o = c_api::mkldnn_gOIhw16i16o,
        gOIhw8o8i = c_api::mkldnn_gOIhw8o8i,
        gOIhw16o16i = c_api::mkldnn_gOIhw16o16i,
    };

    /// A memory descriptor.
    struct desc {
        friend struct memory;
        /// The underlying C API data structure.
        c_api::mkldnn_memory_desc_t data;

        /// Constructs a memory descriptor.
        ///
        /// @param adims Data dimensions
        /// @param adata_type Data precision/type.
        /// @param aformat Data layout format.
        desc(dims adims, data_type adata_type,
                format aformat) {
            validate_dims(adims);
            error::wrap_c_api(
                    c_api::mkldnn_memory_desc_init(&data, adims.size(),
                        adims.size() == 0 ? nullptr : &adims[0],
                         convert_to_c(adata_type), convert_to_c(aformat)),
                    "could not initialize a memory descriptor");
        }

        /// Constructs a memory descriptor from a C API data structure.
        ///
        /// @param adata A C API #mkldnn_memory_desc_t structure.
        desc(const c_api::mkldnn_memory_desc_t &adata): data(adata) {}
    };

    /// A memory primitive descriptor.
    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        friend struct memory;

        // TODO: make private
        primitive_desc() {}

        /// Constructs a memory primitive descriptor.
        primitive_desc(const desc &adesc, const engine &aengine) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(
                    c_api::mkldnn_memory_primitive_desc_create(&result,
                        &adesc.data, aengine.get()),
                    "could not initialize a memory primitive descriptor");
            reset(result);
        }

        /// Returns the memory primitive descriptor.
        memory::desc desc() {
            auto memory_d = mkldnn_primitive_desc_query_memory_d(get());
            return memory::desc(*memory_d); }

        /// Returns the number of data elements in the memory described.
        ///
        /// Returns the number of bytes required to allocate the memory described
        /// including the padding area.
        size_t get_size() const {
             return c_api::mkldnn_memory_primitive_desc_get_size(get());
        }

        bool operator==(const primitive_desc &other) const {
            return mkldnn_memory_primitive_desc_equal(get(), other.get());
        }

        bool operator!=(const primitive_desc &other) const {
            return !operator==(other);
        }
    };

    /// Constructs a memory primitive from a generic primitive.
    ///
    /// @param aprimitive The primitive to treat as memory.
    memory(const primitive &aprimitive): primitive(aprimitive) {}
    /// Constructs a memory primitive.
    ///
    /// @param adesc Memory primitive descriptor.
    memory(const primitive_desc &adesc) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(
                c_api::mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
                "could not create a memory primitive");
        reset(result);
        auto _malloc = [](size_t size, int alignment) {
            void *ptr;
            int rc = ::posix_memalign(&ptr, alignment, size);
            return (rc == 0) ? (char*)ptr : nullptr;
        };
        auto _free = [](char* p) { ::free((void*)p); };
        _handle.reset(_malloc(adesc.get_size(), 64), _free);
        set_data_handle(_handle.get());
    }

    memory(const primitive_desc &adesc, void *ahandle) {
        c_api::mkldnn_primitive_t result;
        error::wrap_c_api(
                c_api::mkldnn_primitive_create(&result, adesc.get(), nullptr, nullptr),
                "could not create a memory primitive");
        reset(result);
        set_data_handle(ahandle);
    }

    /// Returns the descriptor of the memory primitive.
    primitive_desc get_primitive_desc() const {
        primitive_desc adesc;
        c_api::const_mkldnn_primitive_desc_t cdesc;
        error::wrap_c_api(c_api::mkldnn_primitive_get_primitive_desc(get(),
                    &cdesc),
                "could not get primitive descriptor from a memory primitive");
        /* FIXME: no const_cast should be here */
        adesc.reset(const_cast<c_api::mkldnn_primitive_desc_t>(cdesc), true);
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

    inline void set_data_handle(void *handle) const {
        error::wrap_c_api(mkldnn_memory_set_data_handle(get(), handle),
                "could not set native handle");
    }

    // Must go away or be private:
    static c_api::mkldnn_data_type_t convert_to_c(data_type adata_type) {
        return static_cast<c_api::mkldnn_data_type_t>(adata_type);
    }
    static c_api::mkldnn_memory_format_t convert_to_c(format aformat) {
        return static_cast<c_api::mkldnn_memory_format_t>(aformat);
    }

};

enum query {
    undef = c_api::mkldnn_query_undef,

    eengine = c_api::mkldnn_query_engine,
    primitive_kind = c_api::mkldnn_query_primitive_kind,

    num_of_inputs_s32 = c_api::mkldnn_query_num_of_inputs_s32,
    num_of_outputs_s32 = c_api::mkldnn_query_num_of_outputs_s32,

    time_estimate_f64 = c_api::mkldnn_query_time_estimate_f64,
    memory_consumption_s64 = c_api::mkldnn_query_memory_consumption_s64,

    memory_d = c_api::mkldnn_query_memory_d,
    convolution_d = c_api::mkldnn_query_convolution_d,
    relu_d = c_api::mkldnn_query_relu_d,
    softmax_d = c_api::mkldnn_query_softmax_d,
    pooling_d = c_api::mkldnn_query_pooling_d,
    lrn_d = c_api::mkldnn_query_lrn_d,
    batch_normalization_d = c_api::mkldnn_query_batch_normalization_d,
    inner_product_d = c_api::mkldnn_query_inner_product_d,
    convolution_relu_d = c_api::mkldnn_query_convolution_relu_d,

    input_pd = c_api::mkldnn_query_input_pd,
    output_pd = c_api::mkldnn_query_output_pd,
    src_pd = c_api::mkldnn_query_src_pd,
    diff_src_pd = c_api::mkldnn_query_diff_src_pd,
    weights_pd = c_api::mkldnn_query_weights_pd,
    diff_weights_pd = c_api::mkldnn_query_diff_weights_pd,
    dst_pd = c_api::mkldnn_query_dst_pd,
    diff_dst_pd = c_api::mkldnn_query_diff_dst_pd,
    workspace_pd = c_api::mkldnn_query_workspace_pd,
};
inline c_api::mkldnn_query_t convert_to_c(query aquery) {
    return static_cast<c_api::mkldnn_query_t>(aquery);
}

enum padding_kind {
    zero = c_api::mkldnn_padding_zero
};
inline c_api::mkldnn_padding_kind_t convert_to_c(padding_kind kind) {
    return static_cast<c_api::mkldnn_padding_kind_t>(kind);
}

enum prop_kind {
    forward_training = c_api::mkldnn_forward_training,
    forward_scoring = c_api::mkldnn_forward_scoring,
    forward_inference = c_api::mkldnn_forward_inference,
    forward = c_api::mkldnn_forward,
    backward = c_api::mkldnn_backward,
    backward_data = c_api::mkldnn_backward_data,
    backward_weights = c_api::mkldnn_backward_weights,
    backward_bias = c_api::mkldnn_backward_bias
};
inline c_api::mkldnn_prop_kind_t convert_to_c(prop_kind kind) {
    return static_cast<c_api::mkldnn_prop_kind_t>(kind);
}

enum algorithm {
    convolution_direct = c_api::mkldnn_convolution_direct,
    lrn_across_channels = c_api::mkldnn_lrn_across_channels,
    lrn_within_channel  = c_api::mkldnn_lrn_within_channel,
    pooling_max = c_api::mkldnn_pooling_max,
    pooling_avg = c_api::mkldnn_pooling_avg
};

enum batch_normalization_flag {
    use_global_stats = c_api::mkldnn_use_global_stats,
    use_scale_shift = c_api::mkldnn_use_scaleshift,
    omit_stats = c_api::mkldnn_omit_stats
};

static c_api::mkldnn_alg_kind_t convert_to_c(algorithm aalgorithm) {
    return static_cast<c_api::mkldnn_alg_kind_t>(aalgorithm);
}

struct reorder : public primitive {
    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const memory::primitive_desc &input,
                       const memory::primitive_desc &output) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_reorder_primitive_desc_create(
                        &result, input.get(), output.get()),
                    "could not create a reorder primitive descriptor");
            reset(result);
        }
    };

    reorder(const primitive_desc &aprimitive_desc,
            const primitive::at &input, const memory &output) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { input.data };
        c_api::const_mkldnn_primitive_t outputs[] = { output.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a reorder primitive");
        reset(result);
    }

    reorder(const primitive::at &input, const memory &output) {
        auto input_mpd = memory(input).get_primitive_desc();
        auto output_mpd = output.get_primitive_desc();

        auto reorder_d = primitive_desc(input_mpd, output_mpd);

        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { input.data };
        c_api::const_mkldnn_primitive_t outputs[] = { output.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    reorder_d.get(), inputs, outputs),
                "could not create a reorder primitive");
        reset(result);
    }
};

struct view : public primitive {
    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const memory::primitive_desc &input, memory::dims dims,
                memory::dims offsets) {
            c_api::mkldnn_primitive_desc_t result;

            error::wrap_c_api(c_api::mkldnn_view_primitive_desc_create(
                    &result, input.get(), &dims[0], &offsets[0]),
                "could not create a view primitive descriptor");
            reset(result);
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc,
                        const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    view(const primitive_desc &view_pd, primitive::at input) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { input.data };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    view_pd.get(), inputs, nullptr),
                "could not create a view primitive");
        reset(result);
    }

    view(memory input, memory::dims dims, memory::dims offsets) {
        c_api::mkldnn_primitive_t result;
        primitive_desc view_pd(input.get_primitive_desc(), dims,
                offsets);
        c_api::mkldnn_primitive_at_t inputs[] = { {input.get(), 0} };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    view_pd.get(), inputs, nullptr),
                "could not create a view primitive");
        reset(result);
    }
};

struct concat : public primitive {
    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        std::vector<c_api::const_mkldnn_primitive_desc_t> cpp_to_c(
                std::vector<memory::primitive_desc> inputs) {
            std::vector<c_api::const_mkldnn_primitive_desc_t> c_api_inputs;
            c_api_inputs.reserve(inputs.size());
            auto convert_to_c = [](memory::primitive_desc d) { return d.get(); };
            std::transform(inputs.begin(), inputs.end(),
                    std::back_inserter(c_api_inputs), convert_to_c);
            return c_api_inputs;
        }

        primitive_desc(const memory::desc &output, int concat_dimension,
                std::vector<memory::primitive_desc> inputs) {
            c_api::mkldnn_primitive_desc_t result;

            auto c_api_inputs = cpp_to_c(inputs);

            error::wrap_c_api(c_api::mkldnn_concat_primitive_desc_create(
                    &result, &output.data, c_api_inputs.size(),
                    concat_dimension, &c_api_inputs[0]),
                "could not create a concat primitive descriptor");
            reset(result);
        }

        primitive_desc(int concat_dimension,
                std::vector<memory::primitive_desc> inputs) {
            c_api::mkldnn_primitive_desc_t result;

            auto c_api_inputs = cpp_to_c(inputs);

            error::wrap_c_api(c_api::mkldnn_concat_primitive_desc_create(
                    &result, nullptr, c_api_inputs.size(), concat_dimension,
                    &c_api_inputs[0]),
                "could not create a concat primitive descriptor");
            reset(result);
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

    };

    concat(const primitive_desc &concat_pd,
            std::vector<primitive::at> &inputs, const memory &output) {
        c_api::mkldnn_primitive_t result;

        std::vector<c_api::mkldnn_primitive_at_t> p_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
            p_inputs.push_back(inputs[i].data);
        c_api::const_mkldnn_primitive_t outputs[] = { output.get() };

        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    concat_pd.get(), &p_inputs[0], outputs),
                "could not create a concat primitive");
        reset(result);
    }
};

struct sum : public primitive {
    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        std::vector<c_api::const_mkldnn_primitive_desc_t> cpp_to_c(
                std::vector<memory::primitive_desc> inputs) {
            std::vector<c_api::const_mkldnn_primitive_desc_t> c_api_inputs;
            c_api_inputs.reserve(inputs.size());
            auto convert_to_c = [](memory::primitive_desc d) { return d.get();};
            std::transform(inputs.begin(), inputs.end(),
                    std::back_inserter(c_api_inputs), convert_to_c);
            return c_api_inputs;
        }

        primitive_desc(const memory::desc &output, std::vector<double> scale,
                std::vector<memory::primitive_desc> inputs) {
            c_api::mkldnn_primitive_desc_t result;

            auto c_api_inputs = cpp_to_c(inputs);

            error::wrap_c_api(c_api::mkldnn_sum_primitive_desc_create(
                    &result, &output.data, c_api_inputs.size(),
                    &scale[0], &c_api_inputs[0]),
                "could not create a sum primitive descriptor");
            reset(result);
        }

        primitive_desc(std::vector<double> scale,
                std::vector<memory::primitive_desc> inputs) {
            c_api::mkldnn_primitive_desc_t result;

            auto c_api_inputs = cpp_to_c(inputs);

            error::wrap_c_api(c_api::mkldnn_sum_primitive_desc_create(
                    &result, nullptr, c_api_inputs.size(), &scale[0],
                    &c_api_inputs[0]),
                "could not create a sum primitive descriptor");
            reset(result);
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc,
                    const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

    };

    sum(const primitive_desc &sum_pd,
            std::vector<primitive::at> &inputs, const memory &output) {
        c_api::mkldnn_primitive_t result;

        std::vector<c_api::mkldnn_primitive_at_t> p_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
            p_inputs.push_back(inputs[i].data);
        c_api::const_mkldnn_primitive_t outputs[] = { output.get() };

        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    sum_pd.get(), &p_inputs[0], outputs),
                "could not create a sum primitive");
        reset(result);
    }
};
#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <> struct handle_traits<c_api::mkldnn_stream_t> {
    static constexpr auto destructor = &c_api::mkldnn_stream_destroy;
};
#endif

struct stream: public handle<c_api::mkldnn_stream_t> {
    using handle::handle;

    enum kind { any = c_api::mkldnn_stream_kind_t::mkldnn_any_stream,
        eager = c_api::mkldnn_stream_kind_t::mkldnn_eager,
        lazy = c_api::mkldnn_stream_kind_t::mkldnn_lazy };

    static c_api::mkldnn_stream_kind_t convert_to_c(kind akind) {
        return static_cast<c_api::mkldnn_stream_kind_t>(akind);
    }
    /// Constructs a stream.
    stream(kind akind) {
        c_api::mkldnn_stream_t astream;
        error::wrap_c_api(c_api::mkldnn_stream_create(&astream,
                    convert_to_c(akind)),
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
        if (primitives.size() == 0) return *this;
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

    stream &rerun() {
        c_api::mkldnn_primitive_t c_api_error_primitive;
        error::wrap_c_api(
                c_api::mkldnn_stream_rerun(get(), &c_api_error_primitive),
                "could not rerun a stream", &c_api_error_primitive);
        return *this;
    }
};

struct convolution_forward: public primitive {
    struct desc {
        c_api::mkldnn_convolution_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(c_api::mkldnn_convolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, &bias_desc.data,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution forward descriptor");
        }
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(c_api::mkldnn_convolution_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                        &src_desc.data, &weights_desc.data, nullptr,
                        &dst_desc.data, &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution forward descriptor");
        }
    };
    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(), nullptr),
                    "could not create a convolution forward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc bias_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 1);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a bias primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    convolution_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, weights.data,
                    bias.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a convolution forward bias primitive");
        reset(result);
    }

    convolution_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, weights.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a convolution forward primitive");
        reset(result);
    }
};

struct convolution_backward_data : public primitive {
    struct desc {
        c_api::mkldnn_convolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(c_api::mkldnn_convolution_backward_data_desc_init(
                        &data, convert_to_c(aalgorithm), &diff_src_desc.data,
                        &weights_desc.data, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward data descriptor");
        }
    };
    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const convolution_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a convolution backward data primitive descriptor");
            reset(result);
        }
        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_src primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    convolution_backward_data(const primitive_desc &aprimitive_desc,
            const primitive::at &diff_dst, const primitive::at &weights,
            const memory &diff_src) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { diff_dst.data, weights.data  };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a convolution backward data primitive");
        reset(result);
    }
};

struct convolution_backward_weights : public primitive {
    struct desc {
        c_api::mkldnn_convolution_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(c_api::mkldnn_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, &diff_bias_desc.data,
                        &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }
        desc(algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims strides,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(c_api::mkldnn_convolution_backward_weights_desc_init(
                        &data, convert_to_c(aalgorithm), &src_desc.data,
                        &diff_weights_desc.data, nullptr, &diff_dst_desc.data,
                        &strides[0], &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not create a convolution backward weights descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const convolution_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a convolution backward weights primitive descriptor");
            reset(result);
        }
        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_weights_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_bias_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 1);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_bias primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    convolution_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_weights, const memory &diff_bias) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_weights.get(),
                    diff_bias.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a convolution backward weights primitive");
        reset(result);
    }
    convolution_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_weights) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_weights.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a convolution backward weights primitive");
        reset(result);
    }
};

struct convolution_relu_forward : public primitive {
    struct desc {
        c_api::mkldnn_convolution_relu_desc_t data;
        desc(const convolution_forward::desc conv_desc,
                const double negative_slope)
        {
            error::wrap_c_api(c_api::mkldnn_convolution_relu_desc_init(&data,
                        &conv_desc.data, negative_slope),
                    "could not create a convolution_relu_forward descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                    &result, &adesc.data, aengine.get(), nullptr),
                "could not create a convolution relu forward descriptor");
            reset(result);
        }
    };

    convolution_relu_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const primitive::at &bias, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, weights.data,
                bias.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a convolution relu forward primitive");
        reset(result);
    }

    convolution_relu_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, weights.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a convolution relu forward primitive");
        reset(result);
    }
};
struct lrn_forward : public primitive {
    struct desc {
        c_api::mkldnn_lrn_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            int local_size, double alpha, double beta, double k)
        {
            error::wrap_c_api(c_api::mkldnn_lrn_forward_desc_init(&data,
                mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data, local_size, alpha, beta, k),
                "could not create a lrn forward descriptor");
        }
        desc(prop_kind aprop_kind, algorithm aalgorithm,
            const memory::desc &src_desc,
            int local_size, double alpha, double beta)
        {
            error::wrap_c_api(c_api::mkldnn_lrn_forward_desc_init(&data,
                mkldnn::convert_to_c(aprop_kind), convert_to_c(aalgorithm),
                &src_desc.data, local_size, alpha, beta, double(1.0)),
                "could not create a lrn forward descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                    &result, &adesc.data, aengine.get(), nullptr),
                "could not create a lrn forward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc workspace_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t ldesc;
            c_api::const_mkldnn_primitive_desc_t const_ldesc =
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(workspace_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&ldesc, const_ldesc),
                    "could not clone a workspace primitive descriptor");
            adesc.reset(ldesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    lrn_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &workspace,
            const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get(),
                workspace.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a lrn forward primitive");
        reset(result);
    }

    lrn_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a lrn forward primitive");
        reset(result);
    }
};

struct lrn_backward : public primitive {
    struct desc {
        c_api::mkldnn_lrn_desc_t data;
        desc(algorithm aalgorithm,
            const memory::desc &data_desc,
            const memory::desc &diff_data_desc,
            int local_size, double alpha, double beta, double k)
        {
            error::wrap_c_api(c_api::mkldnn_lrn_backward_desc_init(&data,
                convert_to_c(aalgorithm), &diff_data_desc.data,
                &data_desc.data, local_size, alpha, beta, k),
                "could not create a lrn backward descriptor");
        }
        desc(algorithm aalgorithm,
            const memory::desc &data_desc,
            const memory::desc &diff_data_desc,
            int local_size, double alpha, double beta)
        {
            error::wrap_c_api(c_api::mkldnn_lrn_backward_desc_init(&data,
                convert_to_c(aalgorithm), &diff_data_desc.data,
                &data_desc.data, local_size, alpha, beta, double(1.0)),
                "could not create a lrn backward descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const lrn_forward::primitive_desc &hint_fwd_primitive_desc) {
        c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a backward lrn primitive descriptor");
            reset(result);
        }

        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc workspace_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t ldesc;
            c_api::const_mkldnn_primitive_desc_t const_ldesc =
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(workspace_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&ldesc, const_ldesc),
                    "could not clone a workspace primitive descriptor");
            adesc.reset(ldesc);
            return adesc;
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff_dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    lrn_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const primitive::at &workspace, const memory &diff_src) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data,
                workspace.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a lrn backward primitive");
        reset(result);
    }

    lrn_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_src) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a lrn backward primitive");
        reset(result);
    }
};

struct pooling_forward : public primitive {
    struct desc {
        c_api::mkldnn_pooling_desc_t data;
        desc(prop_kind aprop_kind, algorithm aalgorithm,
                const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims kernel,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(c_api::mkldnn_pooling_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        convert_to_c(aalgorithm),
                        &src_desc.data, &dst_desc.data,
                        &strides[0], &kernel[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not init a forward pooling descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
        c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(), nullptr),
                    "could not create a forward pooling primitive descriptor");
            reset(result);
        }

        memory::primitive_desc workspace_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(workspace_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a workspace primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    pooling_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get(), nullptr };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a pooling forward primitive");
        reset(result);
    }

    pooling_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst, const memory &workspace) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get(), workspace.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a pooling forward primitive");
        reset(result);
    }
};

struct pooling_backward : public primitive {
    struct desc {
        c_api::mkldnn_pooling_desc_t data;
        desc(algorithm aalgorithm,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc,
                const memory::dims &strides,
                const memory::dims &kernel,
                const memory::dims &padding_l,
                const memory::dims &padding_r,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(c_api::mkldnn_pooling_backward_desc_init(&data,
                        convert_to_c(aalgorithm),
                        &diff_src_desc.data, &diff_dst_desc.data,
                        &strides[0], &kernel[0],
                        &padding_l[0], &padding_r[0],
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not init a backward pooling descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const pooling_forward::primitive_desc &hint_fwd_primitive_desc) {
        c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a backward pooling primitive descriptor");
            reset(result);
        }
        
        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    pooling_backward(const primitive_desc &aprimitive_desc, const primitive::at &diff_dst,
            const memory &diff_src) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { diff_dst.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a pooling backward primitive");
        reset(result);
    }

    pooling_backward(const primitive_desc &aprimitive_desc, const primitive::at &diff_dst,
            const primitive::at &workspace, const memory &diff_src) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { diff_dst.data, workspace.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a pooling backward primitive");
        reset(result);
    }
};

struct relu_forward : public primitive {
    struct desc {
        c_api::mkldnn_relu_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                T negative_slope) {
            error::wrap_c_api(c_api::mkldnn_relu_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        static_cast<double>(negative_slope)),
                    "could not create a relu forward descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(), nullptr),
                    "could not create a relu forward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    relu_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a relu forward primitive");
        reset(result);
    }

    /*
    relu_forward(prop_kind aprop_kind, double negative_slope,
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
    */
};

struct relu_backward : public primitive {
    struct desc {
        c_api::mkldnn_relu_desc_t data;
        template <typename T>
        desc(const memory::desc &diff_data_desc, const memory::desc &data_desc,
            T negative_slope) {
            error::wrap_c_api(c_api::mkldnn_relu_backward_desc_init(&data,
                        &diff_data_desc.data, &data_desc.data,
                        static_cast<double>(negative_slope)),
                    "could not create a relu backward descriptor");
        }
    };
    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
        const relu_forward::primitive_desc &hint_fwd_primitive_desc) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(),
                        hint_fwd_primitive_desc.get()),
                    "could not create a relu backward primitive descriptor");
            reset(result);
        }
        
        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };
    relu_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &diff_dst,
            const memory &diff_src) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a relu backward primitive");
        reset(result);
    }
};

struct softmax_forward : public primitive {
    struct desc {
        c_api::mkldnn_softmax_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &data_desc,
             int softmax_axis) {
            error::wrap_c_api(c_api::mkldnn_softmax_forward_desc_init(&data,
                    mkldnn::convert_to_c(aprop_kind), &data_desc.data,
                    softmax_axis),
                "could not create a softmax forward descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                    &result, &adesc.data, aengine.get(), nullptr),
                "could not create a softmax forward primitive descriptor");
            reset(result);
        }
    };

    softmax_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a softmax forward primitive");
        reset(result);
    }
};

struct batch_normalization_forward : public primitive {
    struct desc {
        c_api::mkldnn_batch_normalization_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &src_desc, T epsilon,
                unsigned flags) {
            error::wrap_c_api(
                    c_api::mkldnn_batch_normalization_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        static_cast<double>(epsilon), flags),
                "could not create a batch normalization forward descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                &result, &adesc.data, aengine.get(), nullptr),
        "could not create a batch normalization forward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t bndesc;
            c_api::const_mkldnn_primitive_desc_t const_bndesc =
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc mean_primitive_desc() const {
            memory::primitive_desc aprimitive_desc;
            c_api::mkldnn_primitive_desc_t bndesc;
            c_api::mkldnn_batch_normalization_desc_t *p;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_query(
                    get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
                    "could not get a batch-normalization descriptor");
            c_api::const_mkldnn_primitive_desc_t const_bndesc =
                (p->flags & use_global_stats) ?
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                        mkldnn::convert_to_c(src_pd), 1) :
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                        mkldnn::convert_to_c(dst_pd), 1);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a mean primitive descriptor");
            aprimitive_desc.reset(bndesc);
            return aprimitive_desc;
        }

        memory::primitive_desc variance_primitive_desc() const {
            memory::primitive_desc aprimitive_desc;
            c_api::mkldnn_primitive_desc_t bndesc;
            c_api::mkldnn_batch_normalization_desc_t *p;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_query(
                    get(), mkldnn::convert_to_c(batch_normalization_d), 0, &p),
                    "could not get a batch-normalization descriptor");
            c_api::const_mkldnn_primitive_desc_t const_bndesc =
                (p->flags & use_global_stats) ?
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                        mkldnn::convert_to_c(src_pd), 2) :
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                        mkldnn::convert_to_c(dst_pd), 2);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a variance primitive descriptor");
            aprimitive_desc.reset(bndesc);
            return aprimitive_desc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc,
                        const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &weights,
            const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data, weights.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst, const memory &mean, const memory &variance) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, weights.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get(),
            mean.get(), variance.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst, const memory &mean,
            const memory &variance) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get(),
            mean.get(), variance.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &weights,
            const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, weights.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }

    batch_normalization_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization forward primitive");
        reset(result);
    }
};

struct batch_normalization_backward : public primitive {
    struct desc {
        c_api::mkldnn_batch_normalization_desc_t data;
        template <typename T>
        desc(prop_kind aprop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, T epsilon, unsigned flags) {
            error::wrap_c_api(
                    c_api::mkldnn_batch_normalization_backward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        &diff_data_desc.data, &data_desc.data,
                        static_cast<double>(epsilon), flags),
                "could not create a batch normalization backward descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const batch_normalization_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                &result, &adesc.data, aengine.get(),
                hint_fwd_primitive_desc.get()),
        "could not create a batch normalization backward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t bndesc;
            c_api::const_mkldnn_primitive_desc_t const_bndesc =
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc diff_weights_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t bndesc;
            c_api::const_mkldnn_primitive_desc_t const_bndesc =
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a diff_weights primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc mean_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t bndesc;
            c_api::const_mkldnn_primitive_desc_t const_bndesc =
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 1);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a mean primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc variance_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t bndesc;
            c_api::const_mkldnn_primitive_desc_t const_bndesc =
                    c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 2);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&bndesc,
                        const_bndesc),
                    "could not clone a variance primitive descriptor");
            adesc.reset(bndesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc,
                        const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    // Prop_kind == backward
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &diff_dst,
            const primitive::at &weights, const memory &diff_src,
            const memory &diff_weights) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data, diff_dst.data, weights.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get(),
                diff_weights.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization backward primitive");
        reset(result);
    }

    // Prop_kind == backward_data
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance,const primitive::at &diff_dst,
            const primitive::at &weights,  const memory &diff_src) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data, diff_dst.data, weights.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization backward primitive");
        reset(result);
    }

    // Prop_kind == backward_data
    batch_normalization_backward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at &mean,
            const primitive::at &variance, const primitive::at &diff_dst,
            const memory &diff_src) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data,
            mean.data, variance.data, diff_dst.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a batch normalization backward primitive");
        reset(result);
    }
};

struct inner_product_forward: public primitive {
    struct desc {
        c_api::mkldnn_inner_product_desc_t data;
        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    c_api::mkldnn_inner_product_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        &weights_desc.data, &bias_desc.data, &dst_desc.data),
                    "could not create a inner product forward descriptor");
        }

        desc(prop_kind aprop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    c_api::mkldnn_inner_product_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind), &src_desc.data,
                        &weights_desc.data, nullptr, &dst_desc.data),
                    "could not create a inner product forward descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(
                &result, &adesc.data, aengine.get(), nullptr),
        "could not create a inner product forward primitive descriptor");
            reset(result);
        }

        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc bias_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 1);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a bias primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    inner_product_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at weights,
            const primitive::at &bias, const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, weights.data,
                bias.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product forward primitive");
        reset(result);
    }

    inner_product_forward(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at weights,
            const memory &dst) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, weights.data };
        c_api::const_mkldnn_primitive_t outputs[] = { dst.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product forward primitive");
        reset(result);
    }
};

struct inner_product_backward_data: public primitive {
    struct desc {
        c_api::mkldnn_inner_product_desc_t data;
        desc(const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    c_api::mkldnn_inner_product_backward_data_desc_init(&data,
                        &diff_src_desc.data, &weights_desc.data,
                        &diff_dst_desc.data),
                "could not create a inner product backward data descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(&result,
                    &adesc.data, aengine.get(), hint_fwd_primitive_desc.get()),
        "could not create a inner product backward data primitive descriptor");
            reset(result);
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff dst primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc weights_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(weights_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    inner_product_backward_data(const primitive_desc &aprimitive_desc,
            const primitive::at &diff_dst, const primitive::at weights,
            const memory &diff_src) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { diff_dst.data, weights.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_src.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product backward data primitive");
        reset(result);
    }
};

struct inner_product_backward_weights: public primitive {
    struct desc {
        c_api::mkldnn_inner_product_desc_t data;
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    c_api::mkldnn_inner_product_backward_weights_desc_init(
                        &data, &src_desc.data, &diff_weights_desc.data,
                        &diff_bias_desc.data, &diff_dst_desc.data),
                "could not create a inner product backward weights descriptor");
        }
    };

    struct primitive_desc : public handle<c_api::mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine,
                const inner_product_forward::primitive_desc
                    &hint_fwd_primitive_desc) {
            c_api::mkldnn_primitive_desc_t result;
            error::wrap_c_api(c_api::mkldnn_primitive_desc_create(&result,
                    &adesc.data, aengine.get(), hint_fwd_primitive_desc.get()),
        "could not create a inner product backward weights primitive descriptor");
            reset(result);
        }

        memory::primitive_desc diff_dst_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_dst_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff dst primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_weights_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff weights primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc diff_bias_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(diff_weights_pd), 1);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a diff bias primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            c_api::mkldnn_primitive_desc_t cdesc;
            c_api::const_mkldnn_primitive_desc_t const_cdesc =
                c_api::mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(c_api::mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }
    };

    inner_product_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at diff_dst,
            const memory &diff_weights) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        c_api::const_mkldnn_primitive_t outputs[] = { diff_weights.get() };
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product backward weights primitive");
        reset(result);
    }

    inner_product_backward_weights(const primitive_desc &aprimitive_desc,
            const primitive::at &src, const primitive::at diff_dst,
            const memory &diff_weights, const memory &diff_bias) {
        c_api::mkldnn_primitive_t result;
        c_api::mkldnn_primitive_at_t inputs[] = { src.data, diff_dst.data };
        c_api::const_mkldnn_primitive_t outputs[] =
                { diff_weights.get(), diff_bias.get()};
        error::wrap_c_api(c_api::mkldnn_primitive_create(&result,
                aprimitive_desc.get(), inputs, outputs),
            "could not create a inner product backward weights primitive");
        reset(result);
    }
};
} // namespace mkldnn

#endif

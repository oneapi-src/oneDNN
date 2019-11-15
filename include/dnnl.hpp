/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

/// @file
/// C++ API

#ifndef DNNL_HPP
#define DNNL_HPP

#include "dnnl_config.h"

/// @cond DO_NOT_DOCUMENT_THIS
#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "dnnl.h"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include <CL/cl.h>
#endif
/// @endcond

// __cpp_exceptions is referred from
// https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_exceptions.html
// gcc < 5 does not define __cpp_exceptions but __EXCEPTIONS,
// Microsoft C++ Compiler does not provide an option to disable exceptions
#ifndef DNNL_ENABLE_EXCEPTIONS
#if __cpp_exceptions || __EXCEPTIONS \
        || (defined(_MSC_VER) && !defined(__clang__))
#define DNNL_ENABLE_EXCEPTIONS 1
#else
#define DNNL_ENABLE_EXCEPTIONS 0
#endif
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DNNL_TRAP() __builtin_trap()
#elif defined(__INTEL_COMPILER) || defined(_MSC_VER)
#define DNNL_TRAP() __debugbreak()
#else
#error "unknown compiler"
#endif

#if DNNL_ENABLE_EXCEPTIONS
#define DNNL_THROW_ERROR(status, msg) throw error(status, msg)
#else
#include <cstdio>
#define DNNL_THROW_ERROR(status, msg) \
    do { \
        fputs(msg, stderr); \
        DNNL_TRAP(); \
    } while (0)
#endif

namespace dnnl {

/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_api_utils Utils
/// @{

/// DNNL exception class.
///
/// This class captures the status returned by a failed C API function and
/// the error message from the call site.
struct error : public std::exception {
    dnnl_status_t status;
    const char *message;

    /// Constructs an instance of an exception class.
    ///
    /// @param status The error status returned by a C API function.
    /// @param message The error message.
    error(dnnl_status_t status, const char *message)
        : status(status), message(message) {}

    /// Returns the explanatory string.
    const char *what() const noexcept override { return message; }

    /// A convenience function for wrapping calls to C API functions. Checks
    /// the return status and throws an dnnl::error in case of failure.
    ///
    /// @param status The error status returned by a C API function.
    /// @param message The error message.
    static void wrap_c_api(dnnl_status_t status, const char *message) {
        if (status != dnnl_success) DNNL_THROW_ERROR(status, message);
    }
};

/// A class that provides the destructor for a DNNL C API handle.
template <typename T>
struct handle_traits {};

/// DNNL C API handle wrapper class.
///
/// This class is used as the base class for primitive (dnnl::primitive),
/// engine (dnnl::engine), and stream (dnnl::stream) classes, as well as
/// others. An object of the dnnl::handle class can be passed by value.
///
/// A handle can be weak, in which case it follows std::weak_ptr semantics.
/// Otherwise, it follows `std::shared_ptr` semantics.
///
/// @note
///     The implementation stores DNNL C API handles in a `std::shared_ptr`
///     with deleter set to a dummy function in the weak mode.
///
template <typename T, typename traits = handle_traits<T>>
struct handle {
private:
    static dnnl_status_t dummy_destructor(T) { return dnnl_success; }
    std::shared_ptr<typename std::remove_pointer<T>::type> data_ {0};

protected:
    bool operator==(const T other) const { return other == data_.get(); }
    bool operator!=(const T other) const { return !(*this == other); }

public:
    /// Constructs an empty handle object.
    ///
    /// @warning
    ///     Uninitialized object cannot be used in most library calls and is
    ///     equivalent to a null pointer. Any attempt to use its methods, or
    ///     passing it to the other library function, will cause an exception
    ///     to be thrown.
    handle() = default;

    /// Copy constructor.
    handle(const handle<T, traits> &) = default;
    /// Assignment operator.
    handle<T, traits> &operator=(const handle<T, traits> &) = default;
    /// Move constructor.
    handle(handle<T, traits> &&) = default;
    /// Move assignment operator.
    handle<T, traits> &operator=(handle<T, traits> &&) = default;

    /// Constructs a handle wrapper object from a C API handle.
    ///
    /// @param t The C API handle to wrap.
    /// @param weak A flag specifying whether to construct a weak wrapper;
    ///             defaults to `false`.
    explicit handle(T t, bool weak = false) { reset(t, weak); }

    /// Resets the handle wrapper objects to wrap a new C API handle.
    ///
    /// @param t The new value of the C API handle.
    /// @param weak A flag specifying whether the wrapper should be weak;
    ///             defaults to `false`.
    void reset(T t, bool weak = false) {
        data_.reset(t, weak ? &dummy_destructor : traits::destructor);
    }

    /// Returns the underlying C API handle.
    ///
    /// @param allow_empty A flag signifying whether the method is
    ///                    allowed to return an empty (null) object without
    ///                    throwing an exception.
    ///
    /// @returns The underlying C API handle.
    T get(bool allow_empty = false) const {
        T result = data_.get();
        if (allow_empty == false && result == nullptr)
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "attempt to use uninitialized object");
        return result;
    }

    /// Converts a handle to the underlying C API handle type. Does not throw
    /// and returns `nullptr` if the object is empty.
    ///
    /// @returns The underlying C API handle.
    explicit operator T() const { return get(true); }

    /// Checks whether the object is empty.
    ///
    /// @returns Whether the object is empty.
    explicit operator bool() const { return get(true) != nullptr; }

    /// Equality operator.
    ///
    /// @param other Another handle wrapper.
    ///
    /// @returns `true` if this and the other handle wrapper manage the same
    /// underlying C API handle, and `false` otherwise. Empty handle objects are
    /// considered to be equal.
    bool operator==(const handle<T, traits> &other) const {
        return other.data_.get() == data_.get();
    }

    /// Inequality operator.
    ///
    /// @param other Another handle wrapper.
    ///
    /// @returns `true` if this and the other handle wrapper manage different
    /// underlying C API handles, and `false` otherwise. Empty handle objects
    /// are considered to be equal.
    bool operator!=(const handle &other) const { return !(*this == other); }
};

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_memory_t> {
    static constexpr auto destructor = &dnnl_memory_destroy;
};

template <>
struct handle_traits<dnnl_primitive_desc_t> {
    static constexpr auto destructor = &dnnl_primitive_desc_destroy;
};

template <>
struct handle_traits<dnnl_primitive_t> {
    static constexpr auto destructor = &dnnl_primitive_destroy;
};

template <>
struct handle_traits<dnnl_primitive_desc_iterator_t> {
    static constexpr auto destructor = &dnnl_primitive_desc_iterator_destroy;
};
/// @endcond

/// @}

struct stream;
struct error;
struct memory;
struct primitive_desc;

/// @addtogroup cpp_api_primitives Primitives
/// Primitive operations.
/// @{

/// @addtogroup cpp_api_primitive Primitive
/// A base class for all C++ primitives.
/// @{

/// Base class for all computational primitives.
struct primitive : public handle<dnnl_primitive_t> {
    friend struct error;
    friend struct stream;

public:
    /// Kinds of primitives. Used to implement a way to extend the library with
    /// new primitives without changing the ABI.
    enum class kind {
        /// Undefined primitive
        undef = dnnl_undefined_primitive,
        /// A reorder primitive.
        reorder = dnnl_reorder,
        /// A shuffle primitive.
        shuffle = dnnl_shuffle,
        /// A (out-of-place) tensor concatenation primitive.
        concat = dnnl_concat,
        /// A summation primitive.
        sum = dnnl_sum,
        /// A convolution primitive.
        convolution = dnnl_convolution,
        /// A deconvolution primitive.
        deconvolution = dnnl_deconvolution,
        /// An element-wise primitive.
        eltwise = dnnl_eltwise,
        /// A softmax primitive.
        softmax = dnnl_softmax,
        /// A pooling primitive.
        pooling = dnnl_pooling,
        /// An LRN primitive.
        lrn = dnnl_lrn,
        /// A batch normalization primitive.
        batch_normalization = dnnl_batch_normalization,
        /// A layer normalization primitive.
        layer_normalization = dnnl_layer_normalization,
        /// An inner product primitive.
        inner_product = dnnl_inner_product,
        /// A rnn primitive.
        rnn = dnnl_rnn,
        /// A binary primitive.
        binary = dnnl_binary,
        /// A logsoftmax primitive.
        logsoftmax = dnnl_logsoftmax,
        /// A matmul (matrix multiplication) primitive.
        matmul = dnnl_matmul,
    };

    using handle::handle;

    /// Default constructor. Constructs an empty object.
    primitive() = default;

    /// Constructs a primitive from a C API primitive descriptor.
    ///
    /// @param c_pd C API primitive descriptor.
    primitive(const_dnnl_primitive_desc_t c_pd);

    /// Constructs a primitive from a primitive descriptor.
    ///
    /// @param pd Primitive descriptor.
    primitive(const primitive_desc &pd);

    /// Returns the C API primitive descriptor of the underlying C API
    /// primitive.
    ///
    /// @returns The underlying C API primitive descriptor.
    inline const_dnnl_primitive_desc_t get_primitive_desc() const;

    /// Returns the kind of the primitive.
    ///
    /// @returns The primitive kind.
    inline kind get_kind() const;

    /// Executes computations specified by the primitive in a specified stream.
    ///
    /// Arguments are passed via an arguments map containing <index,
    /// memory object> pairs. The index must be one of the `DNNL_ARG_*` values
    /// such as `DNNL_ARG_SRC`, and the memory must have a memory descriptor
    /// matching the one returned by
    /// primitive_desc::query_md(#query::exec_arg_md, index) unless using
    /// dynamic shapes (see DNNL_RUNTIME_DIM_VAL).
    ///
    /// @param stream Stream object. The stream must belong to the same engine
    ///               as the primitive.
    /// @param args Arguments map.
    ///
    void execute(
            stream &stream, const std::unordered_map<int, memory> &args) const;
};

/// Converts primitive kind enum value from C++ API to C API type.
/// @param kind C++ API primitive kind enum value.
/// @returns Corresponding C API primitive kind enum value.
inline dnnl_primitive_kind_t convert_to_c(primitive::kind kind) {
    return static_cast<dnnl_primitive_kind_t>(kind);
}

const_dnnl_primitive_desc_t primitive::get_primitive_desc() const {
    const_dnnl_primitive_desc_t pd;
    error::wrap_c_api(dnnl_primitive_get_primitive_desc(get(), &pd),
            "could not get primitive descriptor by primitive");
    return pd;
}

dnnl::primitive::kind primitive::get_kind() const {
    const_dnnl_primitive_desc_t pd = get_primitive_desc();
    // TODO (Roma): the code below is only needed because get_primitive_desc
    // returns a C type.
    dnnl_primitive_kind_t kind;
    error::wrap_c_api(dnnl_primitive_desc_query(
                              pd, dnnl_query_primitive_kind, 0, (void *)&kind),
            "could not get primitive kind from the primitive descriptor");
    return static_cast<dnnl::primitive::kind>(kind);
}

/// @}

/// @}

/// @addtogroup cpp_api_enums Common data types and enumerations
/// A proxy to @ref c_api_types in @ref c_api.
///
/// @{

/// Scratchpad mode
enum class scratchpad_mode {
    /// The library manages the scratchpad allocation according to the policy
    /// specified by the `DNNL_ENABLE_CONCURRENT_EXEC`
    /// [build option](@ref dev_guide_build_options) (default).
    ///
    /// When `DNNL_ENABLE_CONCURRENT_EXEC=OFF` (default), the library
    /// scratchpad is common to all primitives to reduce the memory footprint.
    /// This configuration comes with limited thread-safety properties, namely
    /// primitives can be created and executed in parallel but cannot migrate
    /// between threads (in other words, each primitive should be executed in
    /// the same thread it was created in).
    ///
    /// When `DNNL_ENABLE_CONCURRENT_EXEC=ON`, the library scratchpad is
    /// private to each primitive. The memory footprint is larger than when
    /// using `DNNL_ENABLE_CONCURRENT_EXEC=OFF` but different primitives can be
    /// created and run concurrently (the same primitive cannot be run
    /// concurrently from two different threads though).
    library = dnnl_scratchpad_mode_library,
    /// A user shall query and provide the scratchpad memory to primitives
    /// This mode is thread-safe as long as the scratchpad buffers
    /// are not used concurrently by two primitive executions.
    user = dnnl_scratchpad_mode_user,
};

/// Converts a scratchpad mode enum value from C++ API to C API type.
/// @param mode C++ API scratchpad mode enum value.
/// @returns Corresponding C API scratchpad mode enum value.
inline dnnl_scratchpad_mode_t convert_to_c(scratchpad_mode mode) {
    return static_cast<dnnl_scratchpad_mode_t>(mode);
}

/// Propagation kind
enum class prop_kind {
    /// Undefined propagation kind
    undef = dnnl_prop_kind_undef,
    /// Forward data propagation (training mode). In this mode primitives
    /// perform computations necessary for subsequent backward propagation.
    forward_training = dnnl_forward_training,
    /// Forward data propagation (inference mode). In this mode primitives
    /// perform only computations that are necessary for inference and omit
    /// computations that are necessary only for backward propagation.
    forward_inference = dnnl_forward_inference,
    /// Forward data propagation,
    /// alias for #dnnl::prop_kind::forward_inference
    forward_scoring = dnnl_forward_scoring,
    /// Forward data propagation,
    /// alias for #dnnl::prop_kind::forward_training
    forward = dnnl_forward,
    /// Backward propagation (with respect to all parameters).
    backward = dnnl_backward,
    /// Backward data propagation.
    backward_data = dnnl_backward_data,
    /// Backward weights propagation.
    backward_weights = dnnl_backward_weights,
    /// Backward bias propagation.
    backward_bias = dnnl_backward_bias
};

/// Converts propagation kind enum value from C++ API to C API type.
/// @param kind C++ API propagation kind enum value.
/// @returns Corresponding C API propagation kind enum value.
inline dnnl_prop_kind_t convert_to_c(prop_kind kind) {
    return static_cast<dnnl_prop_kind_t>(kind);
}

/// Kinds of algorithms.
enum class algorithm {
    undef = dnnl_alg_kind_undef,
    /// Convolution algorithm (either direct or Winograd) is chosen just in time
    convolution_auto = dnnl_convolution_auto,
    /// Direct convolution
    convolution_direct = dnnl_convolution_direct,
    /// Winograd convolution
    convolution_winograd = dnnl_convolution_winograd,
    /// Direct deconvolution
    deconvolution_direct = dnnl_deconvolution_direct,
    /// Winograd deconvolution
    deconvolution_winograd = dnnl_deconvolution_winograd,
    /// Eltwise: ReLU
    eltwise_relu = dnnl_eltwise_relu,
    /// Eltwise: hyperbolic tangent non-linearity (tanh)
    eltwise_tanh = dnnl_eltwise_tanh,
    /// Eltwise: parametric exponential linear unit (elu)
    eltwise_elu = dnnl_eltwise_elu,
    /// Eltwise: square
    eltwise_square = dnnl_eltwise_square,
    /// Eltwise: abs
    eltwise_abs = dnnl_eltwise_abs,
    /// Eltwise: square root
    eltwise_sqrt = dnnl_eltwise_sqrt,
    /// Eltwise: x*sigmoid(a*x)
    eltwise_swish = dnnl_eltwise_swish,
    /// Eltwise: linear
    eltwise_linear = dnnl_eltwise_linear,
    /// Eltwise: bounded_relu
    eltwise_bounded_relu = dnnl_eltwise_bounded_relu,
    /// Eltwise: soft_relu
    eltwise_soft_relu = dnnl_eltwise_soft_relu,
    /// Eltwise: logistic
    eltwise_logistic = dnnl_eltwise_logistic,
    /// Eltwise: exponent
    eltwise_exp = dnnl_eltwise_exp,
    /// Eltwise: gelu
    eltwise_gelu = dnnl_eltwise_gelu,
    /// Eltwise: natural logarithm
    eltwise_log = dnnl_eltwise_log,
    /// Local response normalization (LRN) across multiple channels
    lrn_across_channels = dnnl_lrn_across_channels,
    /// LRN within a single channel
    lrn_within_channel = dnnl_lrn_within_channel,
    /// Max pooling
    pooling_max = dnnl_pooling_max,
    /// Average pooling exclude padding,
    /// alias for #dnnl::algorithm::pooling_avg_include_padding
    pooling_avg = dnnl_pooling_avg,
    /// Average pooling include padding
    pooling_avg_include_padding = dnnl_pooling_avg_include_padding,
    /// Average pooling exclude padding
    pooling_avg_exclude_padding = dnnl_pooling_avg_exclude_padding,
    /// RNN cell
    vanilla_rnn = dnnl_vanilla_rnn,
    /// LSTM cell
    vanilla_lstm = dnnl_vanilla_lstm,
    /// GRU cell
    vanilla_gru = dnnl_vanilla_gru,
    /// GRU cell with linear before reset
    ///
    /// Modification of original GRU cell. Differs from
    /// #dnnl::algorithm::vanilla_gru in how the new memory gate is
    /// calculated:
    /// \f[ c_t = tanh(W_c*x_t + b_{c_x} + r_t*(U_c*h_{t-1}+b_{c_h})) \f]
    /// Primitive expects 4 biases on input:
    /// \f$[b_{u}, b_{r}, b_{c_x}, b_{c_h}]\f$
    lbr_gru = dnnl_lbr_gru,
    /// Binary add
    binary_add = dnnl_binary_add,
    /// Binary mul
    binary_mul = dnnl_binary_mul,
};

/// Converts algorithm kind enum value from C++ API to C API type.
/// @param algorithm C++ API algorithm kind enum value.
/// @returns Corresponding C API algorithm kind enum value.
inline dnnl_alg_kind_t convert_to_c(algorithm algorithm) {
    return static_cast<dnnl_alg_kind_t>(algorithm);
}

/// Flags for batch normalization primitive.
enum class normalization_flags : unsigned {
    /// Use global statistics
    ///
    /// If specified
    ///  - on forward propagation use mean and variance provided by user (input)
    ///  - on backward propagation reduces the amount of computations, since
    ///    mean and variance are considered as constants
    ///
    ///  If not specified:
    ///   - on forward propagation mean and variance are computed and stored in
    ///     output
    ///   - on backward propagation compute full derivative with respect to
    ///     data
    use_global_stats = dnnl_use_global_stats,

    /// Use scale and shift parameters
    ///
    /// If specified:
    ///  - on forward propagation use scale and shift (also named scale and
    ///    bias) for the batch normalization results
    ///  - on backward propagation
    ///    (for `prop_kind` == #dnnl::prop_kind::backward) compute
    ///    gradient (diff) with respect to scale and shift and use an extra
    ///    output
    ///
    /// If not specified:
    ///  - on backward propagation
    ///    `prop_kind` == #dnnl::prop_kind::backward_data has the
    ///    same behavior as `prop_kind` == #dnnl::prop_kind::backward
    use_scale_shift = dnnl_use_scaleshift,

    /// Fuse with ReLU
    ///
    /// If specified:
    ///  - on inference this option behaves the same as if the primitive were
    ///    fused with ReLU via post ops API
    ///  - on training primitive requires workspace (required to be able to
    ///    perform backward propagation)
    fuse_norm_relu = dnnl_fuse_norm_relu
};

/// Converts normalization flags enum value from C++ API to C API type.
/// @param flags C++ API normalization flags enum value.
/// @returns Corresponding C API normalization flags enum value.
inline dnnl_normalization_flags_t convert_to_c(normalization_flags flags) {
    return static_cast<dnnl_normalization_flags_t>(flags);
}

/// RNN cell flags.
enum class rnn_flags : unsigned {
    /// Undefined RNN flags
    undef = dnnl_rnn_flags_undef
};

/// Converts RNN cell flags enum value from C++ API to C API type.
/// @param flags C++ API RNN cell flags enum value.
/// @returns Corresponding C API RNN cell flags enum value.
inline dnnl_rnn_flags_t convert_to_c(rnn_flags flags) {
    return static_cast<dnnl_rnn_flags_t>(flags);
}

#define DNNL_DEFINE_BITMASK_OPS(enum_name) \
    inline enum_name operator|(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name operator&(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name operator^(enum_name lhs, enum_name rhs) { \
        return static_cast<enum_name>( \
                static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs)); \
    } \
\
    inline enum_name &operator|=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name &operator&=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name &operator^=(enum_name &lhs, enum_name rhs) { \
        lhs = static_cast<enum_name>( \
                static_cast<unsigned>(lhs) ^ static_cast<unsigned>(rhs)); \
        return lhs; \
    } \
\
    inline enum_name operator~(enum_name rhs) { \
        return static_cast<enum_name>(~static_cast<unsigned>(rhs)); \
    }

DNNL_DEFINE_BITMASK_OPS(normalization_flags)
DNNL_DEFINE_BITMASK_OPS(rnn_flags)

/// A direction of RNN primitive execution
enum class rnn_direction {
    /// Unidirectional execution of RNN primitive from left to right.
    unidirectional_left2right = dnnl_unidirectional_left2right,
    /// Unidirectional execution of RNN primitive from right to left.
    unidirectional_right2left = dnnl_unidirectional_right2left,
    /// Bidirectional execution of RNN primitive with concatenation of the
    /// results.
    bidirectional_concat = dnnl_bidirectional_concat,
    /// Bidirectional execution of RNN primitive with summation of the
    /// results.
    bidirectional_sum = dnnl_bidirectional_sum,
    /// Alias for #dnnl::rnn_direction::unidirectional_left2right
    unidirectional = dnnl_unidirectional,
};

/// Converts RNN direction enum value from C++ API to C API type.
/// @param dir C++ API RNN direction enum value.
/// @returns Corresponding C API RNN direction enum value.
inline dnnl_rnn_direction_t convert_to_c(rnn_direction dir) {
    return static_cast<dnnl_rnn_direction_t>(dir);
}

/// Primitive descriptor query specification.
///
/// In general, queries are not used with the C++ API because most queries are
/// implemented as class members.
///
/// See @ref dnnl_query_t for more information.
enum class query {
    /// no query
    undef = dnnl_query_undef,

    /// execution engine
    engine = dnnl_query_engine,
    /// primitive kind
    primitive_kind = dnnl_query_primitive_kind,

    /// number of inputs expected
    num_of_inputs_s32 = dnnl_query_num_of_inputs_s32,
    /// number of outputs expected
    num_of_outputs_s32 = dnnl_query_num_of_outputs_s32,

    /// runtime estimation (seconds), unimplemented
    time_estimate_f64 = dnnl_query_time_estimate_f64,
    /// memory consumption (bytes)
    ///
    /// extra (scratch) memory, additional to all inputs and outputs memory
    ///
    /// @sa @ref dev_guide_attributes_scratchpad
    memory_consumption_s64 = dnnl_query_memory_consumption_s64,

    /// scratchpad engine
    ///
    /// engine to be used for creating scratchpad memory
    scratchpad_engine = dnnl_query_scratchpad_engine,

    /// reorder source engine
    reorder_src_engine = dnnl_query_reorder_src_engine,
    /// reorder destination engine
    reorder_dst_engine = dnnl_query_reorder_dst_engine,

    /// implementation name
    impl_info_str = dnnl_query_impl_info_str,

    /// propagation kind
    prop_kind = dnnl_query_prop_kind,

    /// op descriptor
    op_d = dnnl_query_op_d,
    /// convolution descriptor
    convolution_d = dnnl_query_convolution_d,
    /// deconvolution descriptor
    deconvolution_d = dnnl_query_deconvolution_d,
    /// shuffle descriptor
    shuffle_d = dnnl_query_shuffle_d,
    /// eltwise descriptor
    eltwise_d = dnnl_query_eltwise_d,
    /// softmax descriptor
    softmax_d = dnnl_query_softmax_d,
    /// pooling descriptor
    pooling_d = dnnl_query_pooling_d,
    /// lrn descriptor
    lrn_d = dnnl_query_lrn_d,
    /// batch normalization descriptor
    batch_normalization_d = dnnl_query_batch_normalization_d,
    /// layer normalization descriptor
    layer_normalization_d = dnnl_query_layer_normalization_d,
    /// inner product descriptor
    inner_product_d = dnnl_query_inner_product_d,
    /// rnn descriptor
    rnn_d = dnnl_query_rnn_d,
    /// binary descriptor
    binary_d = dnnl_query_binary_d,
    /// logsoftmax descriptor
    logsoftmax_d = dnnl_query_logsoftmax_d,
    /// matmul descriptor
    matmul_d = dnnl_query_matmul_d,

    /// source memory desc
    src_md = dnnl_query_src_md,
    /// source gradient memory desc
    diff_src_md = dnnl_query_diff_src_md,
    /// weights memory descriptor desc
    weights_md = dnnl_query_weights_md,
    /// weights grad. memory desc
    diff_weights_md = dnnl_query_diff_weights_md,
    /// destination memory desc
    dst_md = dnnl_query_dst_md,
    /// destination grad. memory desc
    diff_dst_md = dnnl_query_diff_dst_md,
    /// workspace memory desc
    workspace_md = dnnl_query_workspace_md,
    /// scratchpad memory desc
    scratchpad_md = dnnl_query_scratchpad_md,
    /// memory desc of an execute argument
    exec_arg_md = dnnl_query_exec_arg_md,
};

/// Converts query enum value from C++ API to C API type.
/// @param query C++ API query enum value.
/// @returns Corresponding C API query enum value.
inline dnnl_query_t convert_to_c(query query) {
    return static_cast<dnnl_query_t>(query);
}

/// @}

/// @addtogroup cpp_api_attr Attributes
/// An extension for controlling primitive behavior.
///
/// @sa @ref c_api_attributes in @ref c_api
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_post_ops_t> {
    static constexpr auto destructor = &dnnl_post_ops_destroy;
};
/// @endcond

/// Post operations
///
/// @sa @ref dev_guide_attributes_post_ops
struct post_ops : public handle<dnnl_post_ops_t> {
    using handle<dnnl_post_ops_t>::handle;

    /// Constructs an empty sequence of post operations.
    post_ops() {
        dnnl_post_ops_t result;
        error::wrap_c_api(dnnl_post_ops_create(&result),
                "could not create post operation sequence");
        reset(result);
    }

    /// Returns the number of post-ops entries.
    int len() const { return dnnl_post_ops_len(get()); }

    /// Returns the primitive kind of post-op at entry with a certain index.
    ///
    /// @param index Index of the post-op to return the kind for.
    ///
    /// @returns Primitive kind of the post-op at the specified index.
    primitive::kind kind(int index) const {
        error::wrap_c_api(index < len() ? dnnl_success : dnnl_invalid_arguments,
                "post_ops index is out of range");
        return static_cast<primitive::kind>(
                dnnl_post_ops_get_kind(get(), index));
    }

    /// Appends an accumulation (sum) post operation. Prior to accumulating the
    /// result, the previous value would be multiplied by a scaling factor
    /// @p scale.
    ///
    /// The kind of this post operation is #dnnl::primitive::kind::sum.
    ///
    /// This feature might improve performance for cases like residual learning
    /// blocks, where the result of convolution is accumulated to the previously
    /// computed activations. The parameter @p scale might be extreme for the
    /// integer-based computations when the result and previous activations have
    /// different logical scaling factors.
    ///
    /// In the simplest case when the accumulation is the only post operation,
    /// the computations would be:
    ///
    /// dst[] <- scale * dst[] + op(...) instead of dst[] <- op(...)
    ///
    /// @note
    ///     This post operation (as well as all the others) disregards the
    ///     original layout of the destination; that is, the layout of the
    ///     original destination is expected to be the same as the layout of the
    ///     stored destination.
    ///
    /// @param scale Scaling factor.
    void append_sum(float scale = 1.) {
        error::wrap_c_api(
                dnnl_post_ops_append_sum(get(), scale), "could not append sum");
    }

    /// Returns the parameters of a sum post-op.
    ///
    /// @param index Index of the sum post-op.
    /// @param scale Scaling factor of the sum post-op.
    void get_params_sum(int index, float &scale) const {
        error::wrap_c_api(dnnl_post_ops_get_params_sum(get(), index, &scale),
                "could not get sum params");
    }

    /// Appends an eltwise post operation.
    ///
    /// The kind of this post operation is #dnnl::primitive::kind::eltwise.
    ///
    /// In the simplest case when the eltwise is the only post operation, the
    /// computations would be:
    /// dst[] <- scale * eltwise_op ( op(...) ) instead of dst[] <- op(...)
    /// where eltwise_op is configured with the given parameters.
    ///
    /// @param scale Scaling factor.
    /// @param algorithm Eltwise algorithm.
    /// @param alpha Alpha parameter for the eltwise algorithm.
    /// @param beta Beta parameter for the eltwise algorithm.
    void append_eltwise(
            float scale, algorithm algorithm, float alpha, float beta) {
        error::wrap_c_api(dnnl_post_ops_append_eltwise(get(), scale,
                                  convert_to_c(algorithm), alpha, beta),
                "could not append eltwise");
    }

    /// Returns the eltwise parameters of a post-up.
    ///
    /// @param index Index of the post-op.
    /// @param scale Scaling factor.
    /// @param algorithm Eltwise algorithm.
    /// @param alpha Alpha parameter for the eltwise algorithm.
    /// @param beta Beta parameter for the eltwise algorithm.
    void get_params_eltwise(int index, float &scale, algorithm &algorithm,
            float &alpha, float &beta) const {
        dnnl_alg_kind_t c_alg;
        error::wrap_c_api(dnnl_post_ops_get_params_eltwise(
                                  get(), index, &scale, &c_alg, &alpha, &beta),
                "could not get eltwise params");
        algorithm = static_cast<dnnl::algorithm>(c_alg);
    }
};

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_primitive_attr_t> {
    static constexpr auto destructor = &dnnl_primitive_attr_destroy;
};
/// @endcond

/// Primitive attributes
///
/// @sa @ref dev_guide_attributes
struct primitive_attr : public handle<dnnl_primitive_attr_t> {
    using handle<dnnl_primitive_attr_t>::handle;

    /// Constructs default (empty) primitive attributes.
    primitive_attr() {
        dnnl_primitive_attr_t result;
        error::wrap_c_api(dnnl_primitive_attr_create(&result),
                "could not create a primitive attr");
        reset(result);
    }

    /// Creates primitive attributes from a C API ::dnnl_primitive_attr_t
    /// handle. The resulting handle is not weak and the C handle will be
    /// destroyed during the destruction of the C++ object.
    ///
    /// @param attr The C API primitive attributes.
    primitive_attr(dnnl_primitive_attr_t attr)
        : handle<dnnl_primitive_attr_t>(attr) {}

    /// Returns the scratchpad mode.
    scratchpad_mode get_scratchpad_mode() const {
        dnnl_scratchpad_mode_t result;
        error::wrap_c_api(
                dnnl_primitive_attr_get_scratchpad_mode(get(), &result),
                "could not get scratchpad mode");
        return scratchpad_mode(result);
    }

    /// Sets scratchpad mode.
    ///
    /// @param mode Specified scratchpad mode.
    void set_scratchpad_mode(scratchpad_mode mode) {
        error::wrap_c_api(dnnl_primitive_attr_set_scratchpad_mode(
                                  get(), dnnl::convert_to_c(mode)),
                "could not set scratchpad mode");
    }

    /// Returns output scaling factors correspondence mask and values.
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///             correspondence between the output tensor dimensions and
    ///             the @p scales vector. The set i-th bit indicates that a
    ///             dedicated output scaling factor is used for each index
    ///             along that dimension. The mask value of 0 implies a common
    ///             output scaling factor for the whole output tensor.
    /// @param scales Vector of output scaling factors.
    void get_output_scales(int &mask, std::vector<float> &scales) const {
        dnnl_dim_t count;
        int c_mask;
        const float *c_scales;
        error::wrap_c_api(dnnl_primitive_attr_get_output_scales(
                                  get(), &count, &c_mask, &c_scales),
                "could not get int output scales");
        scales.resize(count);

        mask = c_mask;
        for (dnnl_dim_t c = 0; c < count; ++c)
            scales[c] = c_scales[c];
    }

    /// Sets output scale correspondence mask and values.
    ///
    /// @param mask   Defines the correspondence between the output tensor
    ///               dimensions and the @p scales vector. The set i-th bit
    ///               indicates that a dedicated scaling factor is used for
    ///               each index along that dimension. Set the mask to 0 to
    ///               use a common output scaling factor for the whole output
    ///               tensor.
    /// @param scales Constant vector of output scaling factors.
    ///
    /// @note
    ///      The order of dimensions does not depend on how elements are laid
    ///      out in memory. For example:
    ///       - for a 2D CNN activations tensor the order is always (n, c)
    ///       - for a 4D CNN activations tensor the order is always (n, c, h, w)
    ///       - for a 5D CNN weights tensor the order is always
    ///         (g, oc, ic, kh, kw)
    void set_output_scales(int mask, const std::vector<float> &scales) {
        error::wrap_c_api(dnnl_primitive_attr_set_output_scales(get(),
                                  (dnnl_dim_t)scales.size(), mask, &scales[0]),
                "could not set int output scales");
    }

    /// Returns zero points correspondence mask and values.
    ///
    /// @param arg  Parameter argument index as passed to the
    ///             primitive::execute() call.
    /// @param mask Zero points correspondence mask that defines the
    ///             correspondence between the output tensor dimensions and
    ///             the @p zero_points vector. The set i-th bit indicates that
    ///             a dedicated zero point is used for each index along that
    ///             dimension. Set the mask to 0 to use a common zero point
    ///             for the whole output tensor.
    /// @param zero_points Output vector of zero points.
    void get_zero_points(
            int arg, int &mask, std::vector<int32_t> &zero_points) const {
        dnnl_dim_t count;
        int c_mask;
        const int32_t *c_zero_points;
        error::wrap_c_api(dnnl_primitive_attr_get_zero_points(
                                  get(), arg, &count, &c_mask, &c_zero_points),
                "could not get zero points");
        zero_points.resize(count);

        mask = c_mask;
        for (dnnl_dim_t c = 0; c < count; ++c)
            zero_points[c] = c_zero_points[c];
    }

    /// Sets zero points for primitive operations for given memory argument.
    ///
    /// @param arg  Parameter argument index as passed to the
    ///             primitive::execute() call.
    /// @param mask Zero point correspondence mask that defines the
    ///             correspondence between the tensor dimensions and the @p
    ///             zero_points vector. The set i-th bit indicates that a
    ///             dedicated zero point is used for each index along that
    ///             dimension. Set the mask to 0 to use a common zero point
    ///             for the whole output tensor.
    /// @param zero_points Constant vector of zero points.
    ///
    /// @sa dnnl_primitive_attr_set_zero_points
    /// @sa dnnl::primitive_attr::set_output_scales
    void set_zero_points(
            int arg, int mask, const std::vector<int32_t> &zero_points) {
        error::wrap_c_api(
                dnnl_primitive_attr_set_zero_points(get(), arg,
                        (dnnl_dim_t)zero_points.size(), mask, &zero_points[0]),
                "could not set zero points");
    }

    /// @returns Post-ops previously set via set_post_ops().
    const post_ops get_post_ops() const {
        post_ops result;
        const_dnnl_post_ops_t c_result;
        error::wrap_c_api(dnnl_primitive_attr_get_post_ops(get(), &c_result),
                "could not get post operation sequence");
        result.reset(const_cast<dnnl_post_ops_t>(c_result), true);
        return result;
    }

    /// Sets post-ops.
    ///
    /// @param ops Post-ops object to copy post-ops from.
    void set_post_ops(const post_ops ops) {
        error::wrap_c_api(dnnl_primitive_attr_set_post_ops(get(), ops.get()),
                "could not set post-ops");
    }

    /// Sets quantization scale and shift parameters for RNN data tensors.
    ///
    /// For performance reasons, the low-precision configuration of the RNN
    /// primitives expect input activations to have the unsigned 8-bit integer
    /// data type. The scale and shift parameters are used to quantize
    /// floating-point data to unsigned integer and must be passed to the RNN
    /// primitive using attributes.
    /// The quantization formula is `scale * (data + shift)`.
    ///
    /// @note Quantization scale and shift are common for src_layer, src_iter,
    ///     dst_iter, and dst_layer.
    ///
    /// @param scale The value to scale the data by.
    /// @param shift The value to shift the data by.
    void set_rnn_data_qparams(float scale, float shift) {
        error::wrap_c_api(
                dnnl_primitive_attr_set_rnn_data_qparams(get(), scale, shift),
                "could not set rnn data int scale/shift");
    }

    /// Sets quantization scaling factors for RNN weights tensors. The
    /// low-precision configuration of the RNN primitives expect input weights
    /// to use the signed 8-bit integer data type. The scaling factors are
    /// used to quantize floating-point data to signed integer and must be
    /// passed to RNN primitives using attributes.
    ///
    /// @param mask Scaling factors correspondence mask that defines the
    ///             correspondence between the output tensor dimensions and
    ///             the @p scales vector. The set i-th bit indicates that a
    ///             dedicated scaling factor should be used for any dimension
    ///             j with j >= i. Set the mask to 0 to use a common scaling
    ///             factor for the whole output tensor.
    /// @param scales Vector of output scaling factors.
    ///
    /// @note
    ///      The dimension order is always native and does not depend on the
    ///      actual layout used. For example, five-dimensional weights always
    ///      have (l, d, i, g, o) logical dimension ordering.
    ///
    /// @note
    ///     Quantization scales are common for weights_layer and
    ///     weights_iteration
    ///
    /// @note
    ///     There is no way to check whether @p count corresponds to @p mask
    ///     until an actual primitive descriptor is created, so it is the user's
    ///     responsibility to set proper values. The following formula must
    ///     hold:
    ///
    ///      \f[count = \prod\limits_{d \in mask} output.dims[d]\f]
    void set_rnn_weights_qparams(int mask, const std::vector<float> &scales) {
        error::wrap_c_api(dnnl_primitive_attr_set_rnn_weights_qparams(
                                  get(), (int)scales.size(), mask, &scales[0]),
                "could not set rnn weights int scales");
    }
};

/// @}

/// @addtogroup cpp_api_engine Engine
/// Engine operations.
///
/// @sa @ref c_api_engine in @ref c_api
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_engine_t> {
    static constexpr auto destructor = &dnnl_engine_destroy;
};
/// @endcond

/// An execution engine.
struct engine : public handle<dnnl_engine_t> {
    friend struct primitive;
    friend struct reorder;

    /// Kinds of engines.
    enum class kind {
        /// An unspecified engine
        any = dnnl_any_engine,
        /// CPU engine
        cpu = dnnl_cpu,
        /// GPU engine
        gpu = dnnl_gpu,
    };

    using handle::handle;

    /// Constructs an empty engine. An empty engine cannot be used in any
    /// operations.
    engine() = default;

    /// Returns the number of engines of a certain kind.
    ///
    /// @param kind The kind of engines to count.
    /// @returns The number of engines of the specified kind.
    static size_t get_count(kind kind) {
        return dnnl_engine_get_count(convert_to_c(kind));
    }

    /// Constructs an engine.
    ///
    /// @param kind The kind of engine to construct.
    /// @param index The index of the engine. Must be less than the value
    ///              returned by #get_count() for this particular kind
    ///              of engine.
    engine(kind kind, size_t index) {
        dnnl_engine_t engine;
        error::wrap_c_api(
                dnnl_engine_create(&engine, convert_to_c(kind), index),
                "could not create an engine");
        reset(engine);
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    /// Constructs an engine from OpenCL device and context objects.
    ///
    /// @param kind The kind of engine to construct.
    /// @param device The OpenCL device that this engine will encapsulate.
    /// @param context The OpenCL context (containing the device) that this
    ///                engine will use for all operations.
    engine(kind kind, cl_device_id device, cl_context context) {
        dnnl_engine_t engine;
        error::wrap_c_api(dnnl_engine_create_ocl(
                                  &engine, convert_to_c(kind), device, context),
                "could not create an engine");
        reset(engine);
    }
#endif

    /// Constructs an engine based on a primitive from the primitive
    /// descriptor @p pd by querying its engine.
    ///
    /// @param pd The primitive descriptor based on which the engine is
    ///           constructed.
    engine(const handle<dnnl_primitive_desc_t> &pd) {
        dnnl_engine_t c_engine;
        error::wrap_c_api(
                dnnl_primitive_desc_query(pd.get(),
                        dnnl::convert_to_c(dnnl::query::engine), 0, &c_engine),
                "could not get engine from primitive_desc");
        reset(c_engine, true);
    }

    /// @returns The kind of the engine.
    kind get_kind() const {
        dnnl_engine_kind_t kind;
        error::wrap_c_api(dnnl_engine_get_kind(get(), &kind),
                "could not get the engine kind");
        return static_cast<engine::kind>(kind);
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    /// Returns the OpenCL context associated with the engine.
    cl_context get_ocl_context() const {
        cl_context context = nullptr;
        error::wrap_c_api(dnnl_engine_get_ocl_context(get(), &context),
                "could not get a context handle");
        return context;
    }

    /// Returns the OpenCL device associated with the engine.
    cl_device_id get_ocl_device() const {
        cl_device_id device = nullptr;
        error::wrap_c_api(dnnl_engine_get_ocl_device(get(), &device),
                "could not get a device handle");
        return device;
    }
#endif

    /// Returns the engine of a primitive descriptor.
    ///
    /// @param pd The primitive descriptor based on which the engine is
    ///           constructed.
    /// @returns A weak handle to the engine that the primitive descriptor was
    ///          created with.
    template <class primitive_desc>
    static engine query(const primitive_desc &pd) {
        return query(pd, dnnl::query::engine);
    }

private:
    static dnnl_engine_kind_t convert_to_c(kind kind) {
        return static_cast<dnnl_engine_kind_t>(kind);
    }

    template <class primitive_desc>
    static engine query(const primitive_desc &pd, dnnl::query what) {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_primitive_desc_query(pd.get(),
                                  dnnl::convert_to_c(what), 0, &c_engine),
                "could not get engine from primitive_desc");
        return engine(c_engine, true);
    }
};

/// @}

/// @addtogroup cpp_api_stream Stream
/// Execution stream operations
///
/// @sa @ref c_api_stream in @ref c_api
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
template <>
struct handle_traits<dnnl_stream_t> {
    static constexpr auto destructor = &dnnl_stream_destroy;
};
/// @endcond

/// An execution stream.
struct stream : public handle<dnnl_stream_t> {
    using handle::handle;

    /// Stream flags. Can be combined using the bitwise OR operator.
    enum class flags : unsigned {
        /// Default order execution. Either in-order or out-of-order depending
        /// on the engine runtime
        default_order = dnnl_stream_default_order,
        /// In-order execution.
        in_order = dnnl_stream_default_order,
        /// Out-of-order execution.
        out_of_order = dnnl_stream_out_of_order,
        /// Default stream configuration.
        default_flags = dnnl_stream_default_flags,
    };

    /// Constructs an empty stream. An empty stream cannot be used in any
    /// operations.
    stream() = default;

    /// Constructs a stream for the specified engine and with behavior
    /// controlled by the specified flags.
    ///
    /// @param engine Engine to create the stream on.
    /// @param flags Flags controlling stream behavior.
    stream(const engine &engine, flags flags = flags::default_flags) {
        dnnl_stream_t stream;
        error::wrap_c_api(dnnl_stream_create(&stream, engine.get(),
                                  static_cast<dnnl_stream_flags_t>(flags)),
                "could not create a stream");
        reset(stream);
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    /// Constructs a stream for the specified engine and the OpenCL queue.
    ///
    /// @param engine Engine to create the stream on.
    /// @param queue OpenCL queue to use for the stream.
    stream(const engine &engine, cl_command_queue queue) {
        dnnl_stream_t stream;
        error::wrap_c_api(dnnl_stream_create_ocl(&stream, engine.get(), queue),
                "could not create a stream");
        reset(stream);
    }

    /// @returns The underlying OpenCL queue object.
    cl_command_queue get_ocl_command_queue() const {
        cl_command_queue queue = nullptr;
        error::wrap_c_api(dnnl_stream_get_ocl_command_queue(get(), &queue),
                "could not get OpenCL command queue");
        return queue;
    }
#endif

    /// Waits for all primitives executing in the stream to finish.
    stream &wait() {
        error::wrap_c_api(dnnl_stream_wait(get()), "could not wait a stream");
        return *this;
    }
};

DNNL_DEFINE_BITMASK_OPS(stream::flags)

/// @}

/// @addtogroup cpp_api_memory_related Memory and memory related operations
/// @{

/// @addtogroup cpp_api_memory Memory
/// An object to describe and store data.
///
/// For more information, refer to @ref c_api_memory in @ref c_api.
/// @{

/// Memory object.
///
/// A memory object encapsulates a handle to a memory buffer allocated on a
/// specific engine, tensor dimensions, data type, and memory format, which is
/// the way tensor indices map to offsets in linear memory space. Memory
/// objects are passed to primitives during execution.
struct memory : public handle<dnnl_memory_t> {
    /// Integer type for representing dimension sizes and indices.
    typedef dnnl_dim_t dim;
    /// Vector of dimensions. Implementations are free to force a limit on the
    /// vector's length.
    typedef std::vector<dim> dims;

    /// Helper function that validates that an `std::vector` of dimensions can
    /// be safely converted to the C API array ::dnnl_dims_t. Throws if
    /// validation fails.
    ///
    /// @param v Vector of dimensions.
    template <typename T>
    static void validate_dims(const std::vector<T> &v) {
        if (v.size() > DNNL_MAX_NDIMS)
            DNNL_THROW_ERROR(dnnl_invalid_arguments, "invalid dimensions");
    }

    /// Data type specification
    enum class data_type {
        /// Undefined data type (used for empty memory descriptors).
        undef = dnnl_data_type_undef,
        /// 16-bit/half-precision floating point.
        f16 = dnnl_f16,
        /// non-standard 16-bit floating point with 7-bit mantissa.
        bf16 = dnnl_bf16,
        /// 32-bit/single-precision floating point.
        f32 = dnnl_f32,
        /// 32-bit signed integer.
        s32 = dnnl_s32,
        /// 8-bit signed integer.
        s8 = dnnl_s8,
        /// 8-bit unsigned integer.
        u8 = dnnl_u8,
    };

    /// Memory format kind
    enum class format_kind {
        /// Undefined memory format kind, used for empty memory descriptors.
        undef = dnnl_format_kind_undef,
        /// Unspecified format kind.
        /// The primitive selects a format automatically.
        any = dnnl_format_kind_any,
        /// A tensor in a generic format described by the stride and blocking
        /// values in each dimension. See @ref dnnl_blocking_desc_t for more
        /// information.
        blocked = dnnl_blocked,
        /// Weights format used in 8bit Winograd convolution.
        wino = dnnl_format_kind_wino,
        /// Packed weights format used in RNN.
        packed = dnnl_format_kind_rnn_packed,
    };

    /// Memory format tag specification.
    ///
    /// Memory format tags can be further divided into two categories:
    ///
    ///  - Domain-agnostic names, i.e. names that do not depend on the tensor
    ///    usage in the specific primitive. These names use letters from `a`
    ///    to `l` to denote logical dimensions and form the order in which the
    ///    dimensions are laid in memory. For example,
    ///    #dnnl::memory::format_tag::ab is used to denote a 2D tensor where the
    ///    second logical dimension (denoted as `b`) is the innermost, i.e.
    ///    has stride = 1, and the first logical dimension (`a`) is laid out in
    ///    memory with stride equal to the size of the second dimension. On the
    ///    other hand, #dnnl::memory::format_tag::ba is the transposed version
    ///    of the same tensor: the outermost dimension (`a`) becomes the
    ///    innermost one.
    ///
    ///  - Domain-specific names, i.e. names that make sense only in the
    ///    context of a certain domain, such as CNN. These names are
    ///    aliases to the corresponding domain-agnostic tags and used mostly
    ///    for convenience. For example, #dnnl::memory::format_tag::nc
    ///    is used to denote 2D CNN activations tensor memory format, where
    ///    the channels dimension is the innermost one and the batch dimension
    ///    is the outermost one. Moreover, #dnnl::memory::format_tag::nc is
    ///    an alias for #dnnl::memory::format_tag::ab, because for DNNL
    ///    CNN primitives the logical dimensions of activations tensors come
    ///    in order: batch, channels, spatial.  In other words, batch
    ///    corresponds to the first logical dimension (`a`), and channels
    ///    correspond to the second one (`b`).
    ///
    /// The following domain-specific notation applies to memory format tags:
    ///  - @c 'n' denotes the mini-batch dimension
    ///  - @c 'c' denotes a channels dimension
    ///  - When there are multiple channel dimensions (for example,
    ///    in convolution weights tensor), @c 'i' and @c 'o' denote dimensions
    ///    of input and output channels
    ///  - @c 'c' denotes a groups dimension for convolution weights
    ///  - @c 'd', @c 'h', and @c 'w' denote spatial depth, height, and width
    ///    respectively
    ///
    /// See @ref dnnl_format_tag_t for a detailed description.
    enum class format_tag {
        /// Undefined memory format tag
        undef = dnnl_format_tag_undef,
        /// Placeholder memory format tag. Used to instruct the primitive to
        /// select a format automatically.
        any = dnnl_format_tag_any,

        a = dnnl_a, ///< plain 1D tensor
        ab = dnnl_ab, ///< plain 2D tensor
        abc = dnnl_abc, ///< plain 3D tensor
        abcd = dnnl_abcd, ///< plain 4D tensor
        abcde = dnnl_abcde, ///< plain 5D tensor
        abcdef = dnnl_abcdef, ///< plain 6D tensor

        // Permuted plain formats

        abdec = dnnl_abdec, ///< permuted 5D tensor
        acb = dnnl_acb, ///< permuted 3D tensor
        acbde = dnnl_acbde, ///< permuted 5D tensor
        acdb = dnnl_acdb, ///< permuted 4D tensor
        acdeb = dnnl_acdeb, ///< permuted 5D tensor
        ba = dnnl_ba, ///< permuted 2D tensor
        bac = dnnl_bac, ///< permuted 3D tensor
        bacd = dnnl_bacd, ///< permuted 4D tensor
        bcda = dnnl_bcda, ///< permuted 4D tensor
        cba = dnnl_cba, ///< permuted 3D tensor
        cdba = dnnl_cdba, ///< permuted 4D tensor
        cdeba = dnnl_cdeba, ///< permuted 5D tensor
        decab = dnnl_decab, ///< permuted 5D tensor

        // Opaque blocked formats

        Abc16a = dnnl_Abc16a,
        ABc16a16b = dnnl_ABc16a16b,
        ABc4a4b = dnnl_ABc4a4b,
        aBc16b = dnnl_aBc16b,
        ABc16b16a = dnnl_ABc16b16a,
        Abc4a = dnnl_Abc4a,
        aBc4b = dnnl_aBc4b,
        ABc4b16a4b = dnnl_ABc4b16a4b,
        ABc2b8a4b = dnnl_ABc2b8a4b,
        ABc4b4a = dnnl_ABc4b4a,
        ABc8a16b2a = dnnl_ABc8a16b2a,
        ABc8a8b = dnnl_ABc8a8b,
        aBc8b = dnnl_aBc8b,
        ABc8b16a2b = dnnl_ABc8b16a2b,
        ABc8b8a = dnnl_ABc8b8a,
        Abcd16a = dnnl_Abcd16a,
        ABcd16a16b = dnnl_ABcd16a16b,
        aBcd16b = dnnl_aBcd16b,
        ABcd16b16a = dnnl_ABcd16b16a,
        aBCd16b16c = dnnl_aBCd16b16c,
        aBCd16c16b = dnnl_aBCd16c16b,
        Abcd4a = dnnl_Abcd4a,
        aBcd4b = dnnl_aBcd4b,
        ABcd4b16a4b = dnnl_ABcd4b16a4b,
        ABcd2b8a4b = dnnl_ABcd2b8a4b,
        ABcd4b4a = dnnl_ABcd4b4a,
        ABcd4a4b = dnnl_ABcd4a4b,
        aBCd4c16b4c = dnnl_aBCd4c16b4c,
        aBCd2c8b4c = dnnl_aBCd2c8b4c,
        aBCd4c4b = dnnl_aBCd4c4b,
        aBCd4b4c = dnnl_aBCd4b4c,
        ABcd8a16b2a = dnnl_ABcd8a16b2a,
        ABcd8a8b = dnnl_ABcd8a8b,
        /// 4D tensor blocked by 2nd dimension with block size 8
        aBcd8b = dnnl_aBcd8b,
        ABcd8b16a2b = dnnl_ABcd8b16a2b,
        aBCd8b16c2b = dnnl_aBCd8b16c2b,
        /// 4D tensor blocked by 1st and 2nd dimension with block size 8
        ABcd8b8a = dnnl_ABcd8b8a,
        aBCd8b8c = dnnl_aBCd8b8c,
        aBCd8c16b2c = dnnl_aBCd8c16b2c,
        aBCd8c8b = dnnl_aBCd8c8b,
        Abcde16a = dnnl_Abcde16a,
        ABcde16a16b = dnnl_ABcde16a16b,
        aBcde16b = dnnl_aBcde16b,
        ABcde16b16a = dnnl_ABcde16b16a,
        aBCde16b16c = dnnl_aBCde16b16c,
        aBCde16c16b = dnnl_aBCde16c16b,
        aBCde2c8b4c = dnnl_aBCde2c8b4c,
        Abcde4a = dnnl_Abcde4a,
        aBcde4b = dnnl_aBcde4b,
        ABcde4b4a = dnnl_ABcde4b4a,
        ABcde4a4b = dnnl_ABcde4a4b,
        aBCde4b4c = dnnl_aBCde4b4c,
        aBCde4c16b4c = dnnl_aBCde4c16b4c,
        aBCde4c4b = dnnl_aBCde4c4b,
        Abcde8a = dnnl_Abcde8a,
        ABcde8a8b = dnnl_ABcde8a8b,
        aBcde8b = dnnl_aBcde8b,
        ABcde8b16a2b = dnnl_ABcde8b16a2b,
        ABcde4b16a4b = dnnl_ABcde4b16a4b,
        aBCde8b16c2b = dnnl_aBCde8b16c2b,
        ABcde8b8a = dnnl_ABcde8b8a,
        aBCde8b8c = dnnl_aBCde8b8c,
        ABcd4a8b8a4b = dnnl_ABcd4a8b8a4b,
        ABcd2a8b8a2b = dnnl_ABcd2a8b8a2b,
        aBCde4b8c8b4c = dnnl_aBCde4b8c8b4c,
        aBCde2b8c8b2c = dnnl_aBCde2b8c8b2c,
        aBCde8c16b2c = dnnl_aBCde8c16b2c,
        aBCde8c8b = dnnl_aBCde8c8b,
        aBcdef16b = dnnl_aBcdef16b,
        aBCdef16b16c = dnnl_aBCdef16b16c,
        aBCdef16c16b = dnnl_aBCdef16c16b,
        aBcdef4b = dnnl_aBcdef4b,
        aBCdef4c4b = dnnl_aBCdef4c4b,
        aBCdef4b4c = dnnl_aBCdef4b4c,
        aBCdef8b8c = dnnl_aBCdef8b8c,
        aBCdef8c16b2c = dnnl_aBCdef8c16b2c,
        aBCdef4c16b4c = dnnl_aBCdef4c16b4c,
        aBCdef8c8b = dnnl_aBCdef8c8b,
        aBdc16b = dnnl_aBdc16b,
        aBdc4b = dnnl_aBdc4b,
        aBdc8b = dnnl_aBdc8b,
        aBdec16b = dnnl_aBdec16b,
        aBdec4b = dnnl_aBdec4b,
        aBdec8b = dnnl_aBdec8b,
        aBdefc16b = dnnl_aBdefc16b,
        aCBdef16c16b = dnnl_aCBdef16c16b,
        aBdefc4b = dnnl_aBdefc4b,
        aBdefc8b = dnnl_aBdefc8b,
        Acb16a = dnnl_Acb16a,
        Acb4a = dnnl_Acb4a,
        Acb8a = dnnl_Acb8a,
        aCBd16b16c = dnnl_aCBd16b16c,
        aCBd16c16b = dnnl_aCBd16c16b,
        aCBde16b16c = dnnl_aCBde16b16c,
        aCBde16c16b = dnnl_aCBde16c16b,
        Acdb16a = dnnl_Acdb16a,
        Acdb4a = dnnl_Acdb4a,
        Acdb8a = dnnl_Acdb8a,
        Acdeb16a = dnnl_Acdeb16a,
        Acdeb4a = dnnl_Acdeb4a,
        Acdeb8a = dnnl_Acdeb8a,
        BAc16a16b = dnnl_BAc16a16b,
        BAc16b16a = dnnl_BAc16b16a,
        BAcd16a16b = dnnl_BAcd16a16b,
        BAcd16b16a = dnnl_BAcd16b16a,
        ABcd32a32b = dnnl_ABcd32a32b,
        BAcde16b16 = dnnl_BAcde16b16a,
        aBdec32b = dnnl_aBdec32b,
        Abcdef16a = dnnl_Abcdef16a,
        Acdb32a = dnnl_Acdb32a,
        format_tag_last = dnnl_format_tag_last,

        x = dnnl_x,
        /// 2D CNN activations tensor,
        /// an alias for #dnnl::memory::format_tag::ab
        nc = dnnl_nc,
        cn = dnnl_cn,
        tn = dnnl_tn,
        nt = dnnl_nt,
        ncw = dnnl_ncw,
        nwc = dnnl_nwc,
        /// 4D CNN activations tensor,
        /// an alias for #dnnl::memory::format_tag::abcd
        nchw = dnnl_nchw,
        /// 4D CNN activations tensor,
        /// an alias for #dnnl::memory::format_tag::acdb
        nhwc = dnnl_nhwc,
        /// 4D CNN activations tensor,
        /// an alias for #dnnl::memory::format_tag::bcda
        chwn = dnnl_chwn,
        ncdhw = dnnl_ncdhw,
        ndhwc = dnnl_ndhwc,
        oi = dnnl_oi,
        io = dnnl_io,
        oiw = dnnl_oiw,
        wio = dnnl_wio,
        oihw = dnnl_oihw,
        hwio = dnnl_hwio,
        ihwo = dnnl_ihwo,
        iohw = dnnl_iohw,
        oidhw = dnnl_oidhw,
        dhwio = dnnl_dhwio,
        goiw = dnnl_goiw,
        goihw = dnnl_goihw,
        hwigo = dnnl_hwigo,
        giohw = dnnl_giohw,
        goidhw = dnnl_goidhw,
        tnc = dnnl_tnc,
        ntc = dnnl_ntc,
        ldnc = dnnl_ldnc,
        ldigo = dnnl_ldigo,
        ldgoi = dnnl_ldgoi,
        ldgo = dnnl_ldgo,
        nCdhw16c = dnnl_nCdhw16c,
        nCdhw4c = dnnl_nCdhw4c,
        nCdhw8c = dnnl_nCdhw8c,
        nChw16c = dnnl_nChw16c,
        nChw4c = dnnl_nChw4c,
        nChw8c = dnnl_nChw8c,
        nCw16c = dnnl_nCw16c,
        nCw4c = dnnl_nCw4c,
        nCw8c = dnnl_nCw8c,
        NCw16n16c = dnnl_NCw16n16c,
        NChw16n16c = dnnl_NChw16n16c,
        NCdhw16n16c = dnnl_NCdhw16n16c,
        NChw32n32c = dnnl_NChw32n32c,
        IOhw16i16o = dnnl_IOhw16i16o,
        Ohwi32o = dnnl_Ohwi32o,
        IOdhw16i16o = dnnl_IOdhw16i16o,
        gIOhw16i16o = dnnl_gIOhw16i16o,
        gOhwi32o = dnnl_gOhwi32o,
        Goidhw16g = dnnl_Goidhw16g,
        IOw16o16i = dnnl_IOw16o16i,
        OIw16i16o = dnnl_OIw16i16o,
        IOw16i16o = dnnl_IOw16i16o,
        gIOw16i16o = dnnl_gIOw16i16o,
        OIw16o16i = dnnl_OIw16o16i,
        Oiw16o = dnnl_Oiw16o,
        OIw4i16o4i = dnnl_OIw4i16o4i,
        OIw2i8o4i = dnnl_OIw2i8o4i,
        OIw4i4o = dnnl_OIw4i4o,
        OIw4o4i = dnnl_OIw4o4i,
        Oiw4o = dnnl_Oiw4o,
        OIw8i16o2i = dnnl_OIw8i16o2i,
        OIw8i8o = dnnl_OIw8i8o,
        OIw8o16i2o = dnnl_OIw8o16i2o,
        OIw8o8i = dnnl_OIw8o8i,
        Owi16o = dnnl_Owi16o,
        Owi4o = dnnl_Owi4o,
        Owi8o = dnnl_Owi8o,
        IOhw16o16i = dnnl_IOhw16o16i,
        Ohwi16o = dnnl_Ohwi16o,
        Ohwi4o = dnnl_Ohwi4o,
        Ohwi8o = dnnl_Ohwi8o,
        OIhw16i16o = dnnl_OIhw16i16o,
        OIhw16o16i = dnnl_OIhw16o16i,
        Oihw16o = dnnl_Oihw16o,
        OIhw4i16o4i = dnnl_OIhw4i16o4i,
        OIhw4i4o = dnnl_OIhw4i4o,
        OIhw4o4i = dnnl_OIhw4o4i,
        Oihw4o = dnnl_Oihw4o,
        OIhw8i16o2i = dnnl_OIhw8i16o2i,
        OIhw8i8o = dnnl_OIhw8i8o,
        OIhw8o16i2o = dnnl_OIhw8o16i2o,
        OIhw8o8i = dnnl_OIhw8o8i,
        OIhw2i8o4i = dnnl_OIhw2i8o4i,
        Odhwi16o = dnnl_Odhwi16o,
        Odhwi4o = dnnl_Odhwi4o,
        Odhwi8o = dnnl_Odhwi8o,
        OIdhw16i16o = dnnl_OIdhw16i16o,
        OIdhw16o16i = dnnl_OIdhw16o16i,
        Oidhw16o = dnnl_Oidhw16o,
        OIdhw4i4o = dnnl_OIdhw4i4o,
        OIdhw4o4i = dnnl_OIdhw4o4i,
        Oidhw4o = dnnl_Oidhw4o,
        OIdhw8i16o2i = dnnl_OIdhw8i16o2i,
        OIdhw4i16o4i = dnnl_OIdhw4i16o4i,
        OIdhw8i8o = dnnl_OIdhw8i8o,
        OIdhw8o8i = dnnl_OIdhw8o8i,
        gIOw16o16i = dnnl_gIOw16o16i,
        gOIw16i16o = dnnl_gOIw16i16o,
        gOIw16o16i = dnnl_gOIw16o16i,
        gOiw16o = dnnl_gOiw16o,
        gOIw4i16o4i = dnnl_gOIw4i16o4i,
        gOIw2i8o4i = dnnl_gOIw2i8o4i,
        gOIw4i4o = dnnl_gOIw4i4o,
        gOIw4o4i = dnnl_gOIw4o4i,
        gOiw4o = dnnl_gOiw4o,
        gOIw8i16o2i = dnnl_gOIw8i16o2i,
        gOIw8i8o = dnnl_gOIw8i8o,
        gOIw8o16i2o = dnnl_gOIw8o16i2o,
        gOIw8o8i = dnnl_gOIw8o8i,
        gOwi16o = dnnl_gOwi16o,
        gOwi4o = dnnl_gOwi4o,
        gOwi8o = dnnl_gOwi8o,
        Goiw8g = dnnl_Goiw8g,
        Goiw16g = dnnl_Goiw16g,
        gIOhw16o16i = dnnl_gIOhw16o16i,
        gOhwi16o = dnnl_gOhwi16o,
        gOhwi4o = dnnl_gOhwi4o,
        gOhwi8o = dnnl_gOhwi8o,
        Goihw16g = dnnl_Goihw16g,
        gOIhw16i16o = dnnl_gOIhw16i16o,
        gOIhw16o16i = dnnl_gOIhw16o16i,
        gOihw16o = dnnl_gOihw16o,
        gOIhw4i16o4i = dnnl_gOIhw4i16o4i,
        gOIhw2i8o4i = dnnl_gOIhw2i8o4i,
        gOIhw4i4o = dnnl_gOIhw4i4o,
        gOIhw4o4i = dnnl_gOIhw4o4i,
        gOihw4o = dnnl_gOihw4o,
        Goihw8g = dnnl_Goihw8g,
        gOIhw8i16o2i = dnnl_gOIhw8i16o2i,
        gOIhw8i8o = dnnl_gOIhw8i8o,
        gOIhw8o16i2o = dnnl_gOIhw8o16i2o,
        OIhw4o8i8o4i = dnnl_OIhw4o8i8o4i,
        OIhw2o8i8o2i = dnnl_OIhw2o8i8o2i,
        gOIhw4o8i8o4i = dnnl_gOIhw4o8i8o4i,
        gOIhw2o8i8o2i = dnnl_gOIhw2o8i8o2i,
        gOIhw8o8i = dnnl_gOIhw8o8i,
        gIOdhw16i16o = dnnl_gIOdhw16i16o,
        gOdhwi16o = dnnl_gOdhwi16o,
        gOdhwi4o = dnnl_gOdhwi4o,
        gOdhwi8o = dnnl_gOdhwi8o,
        gOIdhw16i16o = dnnl_gOIdhw16i16o,
        gOIdhw16o16i = dnnl_gOIdhw16o16i,
        gOidhw16o = dnnl_gOidhw16o,
        gOIdhw4i4o = dnnl_gOIdhw4i4o,
        gOIdhw4o4i = dnnl_gOIdhw4o4i,
        gOidhw4o = dnnl_gOidhw4o,
        gOIdhw8i16o2i = dnnl_gOIdhw8i16o2i,
        gOIdhw4i16o4i = dnnl_gOIdhw4i16o4i,
        gOIdhw8i8o = dnnl_gOIdhw8i8o,
        gOIdhw8o8i = dnnl_gOIdhw8o8i,
    };

    /// A memory descriptor.
    struct desc {
        friend struct memory;
        /// The underlying C API data structure.
        dnnl_memory_desc_t data;

        /// Constructs a zero (empty) memory descriptor. Such a memory
        /// descriptor can be used to indicate absence of an argument.
        desc() : data() {}

        /// Constructs a memory descriptor.
        ///
        /// @param dims Tensor dimensions.
        /// @param data_type Data precision/type.
        /// @param format_tag Memory format tag.
        desc(const memory::dims &dims, data_type data_type,
                format_tag format_tag) {
            validate_dims(dims);
            error::wrap_c_api(
                    dnnl_memory_desc_init_by_tag(&data, (int)dims.size(),
                            dims.size() == 0 ? nullptr : &dims[0],
                            convert_to_c(data_type), convert_to_c(format_tag)),
                    "could not initialize a memory descriptor by tag");
        }

        /// Constructs a memory descriptor by strides.
        ///
        /// @param dims Tensor dimensions.
        /// @param data_type Data precision/type.
        /// @param strides The strides for each dimension.
        desc(const memory::dims &dims, data_type data_type,
                const memory::dims &strides) {
            validate_dims(dims);
            error::wrap_c_api(
                    dnnl_memory_desc_init_by_strides(&data, (int)dims.size(),
                            dims.size() == 0 ? nullptr : &dims[0],
                            convert_to_c(data_type),
                            strides.size() == 0 ? nullptr : &strides[0]),
                    "could not initialize a memory descriptor by strides");
        }

        /// Constructs a memory descriptor from a C API data structure.
        ///
        /// @param data A C API ::dnnl_memory_desc_t structure.
        desc(const dnnl_memory_desc_t &data) : data(data) {}

        /// Constructs a sub-memory descriptor.
        //
        /// @param dims Sizes of a sub-memory.
        /// @param offsets Offsets of a sub-memory from the encompassing
        ///                memory object in each dimension
        desc submemory_desc(
                const memory::dims &dims, const memory::dims &offsets) {
            dnnl_memory_desc_t sub_md;
            error::wrap_c_api(dnnl_memory_desc_init_submemory(
                                      &sub_md, &data, &dims[0], &offsets[0]),
                    "could not initialize a sub-memory");
            return desc(sub_md);
        }

        /// Constructs a memory descriptor by reshaping existing one.
        //
        /// @param dims New dimensions. The product of dimensions must
        /// remain constant.
        desc reshape(const memory::dims &dims) {
            dnnl_memory_desc_t out_md;
            error::wrap_c_api(dnnl_memory_desc_reshape(&out_md, &data,
                                      (int)dims.size(), &dims[0]),
                    "could not reshape a memory descriptor");
            return desc(out_md);
        }

        /// Returns a copy of the dimensions array.
        ///
        /// Potentially expensive due to the data copy involved.
        memory::dims dims() const {
            return memory::dims(data.dims, data.dims + data.ndims);
        }

        /// Returns the data type.
        memory::data_type data_type() const {
            return static_cast<memory::data_type>(data.data_type);
        }

        /// Returns the number of bytes required to allocate a memory buffer for
        /// the memory object described by this memory descriptor including
        /// the padding area.
        size_t get_size() const { return dnnl_memory_desc_get_size(&data); }

        /// Returns true if the memory descriptor describes an empty memory.
        bool is_zero() const { return data.ndims == 0; }

        /// An equality operator.
        /// @param other Another memory descriptor.
        /// @returns Whether this and the other memory descriptors have
        ///          the same format tag, dimensions, strides, blocking, etc.
        bool operator==(const desc &other) const {
            return dnnl_memory_desc_equal(&data, &other.data) != 0;
        }

        /// An inequality operator.
        /// @param other Another memory descriptor.
        /// @returns Whether this and the other memory descriptors describe
        ///          different memory.
        bool operator!=(const desc &other) const { return !operator==(other); }
    };

    // Default constructor.
    //
    // Constructs an empty memory object, which can be used to indicate absence
    // of a parameter.
    memory() = default;

    /// Constructs a memory object.
    ///
    /// @param md Memory descriptor.
    /// @param engine Engine to store the data on.
    /// @param handle Handle of the memory buffer to use as an underlying
    ///               storage. On CPU this is a pointer.
    memory(const desc &md, const engine &engine, void *handle) {
        dnnl_memory_t result;
        error::wrap_c_api(
                dnnl_memory_create(&result, &md.data, engine.get(), handle),
                "could not create a memory");
        reset(result);
    }

    /// Constructs a memory object.
    ///
    /// The underlying storage for the memory will be allocated by the library.
    ///
    /// @param md Memory descriptor.
    /// @param engine Engine to store the data on.
    memory(const desc &md, const engine &engine)
        : memory(md, engine, DNNL_MEMORY_ALLOCATE) {}

    /// Returns the associated memory descriptor.
    desc get_desc() const {
        const dnnl_memory_desc_t *cdesc;
        error::wrap_c_api(dnnl_memory_get_memory_desc(get(), &cdesc),
                "could not get memory descriptor from a memory");
        return desc(*cdesc);
    }

    /// Returns the associated engine.
    engine get_engine() const {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_memory_get_engine(get(), &c_engine),
                "could not get engine from a memory");
        return engine(c_engine, true);
    }

    /// Returns the underlying memory buffer.
    ///
    /// On the CPU engine this is a pointer to the allocated memory.
    void *get_data_handle() const {
        void *handle;
        error::wrap_c_api(dnnl_memory_get_data_handle(get(), &handle),
                "could not get native handle");
        return handle;
    }

    /// Sets memory buffer.
    ///
    /// @param handle Memory buffer to use as the underlying storage. It must
    ///               have at least get_desc().get_size() bytes allocated.
    void set_data_handle(void *handle) const {
        error::wrap_c_api(dnnl_memory_set_data_handle(get(), handle),
                "could not set native handle");
    }

    /// Maps the data of the memory.
    ///
    /// Mapping allows to read/write directly from/to the memory contents for
    /// engines that do not support direct memory access.
    ///
    /// Mapping is an exclusive operation - a memory object cannot be used in
    /// other operations until this memory object is unmapped.
    /// @tparam T Type of the pointer to be mapped.
    ///
    /// @note Any primitives working with the memory should be completed before
    ///       mapping. Use stream::wait() to synchronize the corresponding
    ///       execution stream.
    ///
    /// @note Map/unmap API is provided mainly for debug/testing purposes and
    ///       its performance may be suboptimal.
    ///
    /// @returns Pointer to the mapped memory.
    template <typename T = void>
    T *map_data() const {
        void *mapped_ptr;
        error::wrap_c_api(dnnl_memory_map_data(get(), &mapped_ptr),
                "could not map the data");
        return static_cast<T *>(mapped_ptr);
    }

    /// Unmaps the previously mapped data for the memory.
    ///
    /// Any changes of the mapped data are synchronized back to the memory
    /// after the call is complete. The mapped pointer must be
    /// obtained through a map_data() call.
    ///
    /// @note Map/unmap API is provided mainly for debug/testing purposes and
    ///       its performance may be suboptimal.
    ///
    /// @param mapped_ptr A pointer previously returned by map_data().
    void unmap_data(void *mapped_ptr) const {
        error::wrap_c_api(dnnl_memory_unmap_data(get(), mapped_ptr),
                "could not unmap the data");
    }

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    /// Returns the OpenCL memory object associated with the memory.
    cl_mem get_ocl_mem_object() const {
        cl_mem mem_object;
        error::wrap_c_api(dnnl_memory_get_ocl_mem_object(get(), &mem_object),
                "could not get OpenCL memory object");
        return mem_object;
    }

    /// Sets the OpenCL memory object @p mem_object associated with the memory.
    ///
    /// @param mem_object OpenCL cl_mem object to use as the underlying
    ///                   storage. It must have at least get_desc().get_size()
    ///                   bytes allocated.
    void set_ocl_mem_object(cl_mem mem_object) {
        error::wrap_c_api(dnnl_memory_set_ocl_mem_object(get(), mem_object),
                "could not set OpenCL memory object");
    }
#endif

    static dnnl_data_type_t convert_to_c(data_type data_type) {
        return static_cast<dnnl_data_type_t>(data_type);
    }
    static dnnl_format_tag_t convert_to_c(format_tag format) {
        return static_cast<dnnl_format_tag_t>(format);
    }
};

inline bool operator==(dnnl_data_type_t a, memory::data_type b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(dnnl_data_type_t a, memory::data_type b) {
    return !(a == b);
}
inline bool operator==(memory::data_type a, dnnl_data_type_t b) {
    return b == a;
}
inline bool operator!=(memory::data_type a, dnnl_data_type_t b) {
    return !(a == b);
}

inline bool operator==(dnnl_format_tag_t a, memory::format_tag b) {
    return a == memory::convert_to_c(b);
}
inline bool operator!=(dnnl_format_tag_t a, memory::format_tag b) {
    return !(a == b);
}
inline bool operator==(memory::format_tag a, dnnl_format_tag_t b) {
    return b == a;
}
inline bool operator!=(memory::format_tag a, dnnl_format_tag_t b) {
    return !(a == b);
}

/// @}

/// @}

/// @addtogroup cpp_api_primitives Primitives
/// @{

/// @addtogroup cpp_api_primitive_descriptors Primitive descriptors
/// @{

/// Base class for all primitive descriptors.
struct primitive_desc_base : public handle<dnnl_primitive_desc_t> {
    using handle<dnnl_primitive_desc_t>::handle;

    /// Default constructor. Produces an empty object.
    primitive_desc_base() = default;

    /// Returns the engine of the primitive descriptor.
    engine get_engine() const { return engine::query(*this); }

    /// Returns implementation name.
    const char *impl_info_str() const {
        const char *res;
        error::wrap_c_api(dnnl_primitive_desc_query(
                                  get(), dnnl_query_impl_info_str, 0, &res),
                "could not query implementation info string");
        return res;
    }

    /// Returns a memory::dim value (same as int64_t).
    memory::dim query_s64(query what) const {
        memory::dim res;
        dnnl_status_t status = dnnl_primitive_desc_query(
                get(), dnnl::convert_to_c(what), 0, &res);
        return status == dnnl_success ? res : 0;
    }

    /// Returns a memory descriptor.
    ///
    /// @param what The memory descriptor to query; can be #dnnl::query::src_md,
    ///             #dnnl::query::dst_md, etc.
    /// @param idx Index of the descriptor. For example, convolution bias can
    ///            be queried with what = #query::weights_md and idx=1.
    /// @returns The requested memory descriptor.
    ///
    /// @note There are convenience methods
    /// #dnnl::primitive_desc_base::src_desc(),
    /// #dnnl::primitive_desc_base::dst_desc(), and others.
    memory::desc query_md(query what, int idx = 0) const {
        std::vector<query> valid_q {query::src_md, query::diff_src_md,
                query::weights_md, query::diff_weights_md, query::dst_md,
                query::diff_dst_md, query::workspace_md, query::scratchpad_md,
                query::exec_arg_md};
        if (!std::any_of(valid_q.cbegin(), valid_q.cend(),
                    [=](query q) { return what == q; }))
            DNNL_THROW_ERROR(dnnl_invalid_arguments, "invalid memory query");

        const dnnl_memory_desc_t *cdesc = dnnl_primitive_desc_query_md(
                get(), dnnl::convert_to_c(what), idx);
        return memory::desc(*cdesc);
    }

    /// @returns Source memory descriptor with index @p idx.
    /// @returns A zero_md if such source memory descriptor is not defined.
    memory::desc src_desc(int idx) const {
        return query_md(query::src_md, idx);
    }

    /// @returns Destination memory descriptor with index @p idx.
    /// @returns A zero_md if such destination memory descriptor is not defined.
    memory::desc dst_desc(int idx) const {
        return query_md(query::dst_md, idx);
    }

    /// @returns Weights memory descriptor with index @p idx.
    /// @returns A zero_md if such weights memory descriptor is not defined.
    memory::desc weights_desc(int idx) const {
        return query_md(query::weights_md, idx);
    }

    /// @returns Diff source memory descriptor with index @p idx.
    /// @returns A zero_md if such diff source memory descriptor is not defined.
    memory::desc diff_src_desc(int idx) const {
        return query_md(query::diff_src_md, idx);
    }

    /// @returns Diff destination memory descriptor with index @p idx.
    /// @returns A zero_md if such diff destination memory descriptor is not
    ///          defined.
    memory::desc diff_dst_desc(int idx) const {
        return query_md(query::diff_dst_md, idx);
    }

    /// @returns Diff weights memory descriptor with index @p idx.
    /// @returns a zero_md if such diff weights memory descriptor is not
    ///          defined.
    memory::desc diff_weights_desc(int idx) const {
        return query_md(query::diff_weights_md, idx);
    }

    // Separate versions without the index argument for documentation
    // purposes.

    /// Queries source memory descriptor.
    /// Returns a zero_md if it is not defined.
    memory::desc src_desc() const { return src_desc(0); }

    /// Queries destination memory descriptor.
    /// Returns a zero_md if it is not defined.
    memory::desc dst_desc() const { return dst_desc(0); }

    /// Queries weights memory descriptor.
    /// Returns a zero_md if it is not defined.
    memory::desc weights_desc() const { return weights_desc(0); }

    /// Queries diff source memory descriptor.
    /// Returns a zero_md if it is not defined.
    memory::desc diff_src_desc() const { return diff_src_desc(0); }

    /// Queries diff destination memory descriptor.
    /// Returns a zero_md if it is not defined.
    memory::desc diff_dst_desc() const { return diff_dst_desc(0); }

    /// Queries diff weights memory descriptor.
    /// Returns a zero_md if it is not defined.
    memory::desc diff_weights_desc() const { return diff_weights_desc(0); }

    /// Queries works workspace descriptor. Returns a zero_md if workspace is
    /// not required.
    memory::desc workspace_desc() const {
        return query_md(query::workspace_md, 0);
    }

    /// Returns scratchpad memory descriptor.
    ///
    /// @returns A zero_md if no scratchpad is required by the primitive
    /// descriptor.
    ///
    /// @sa @ref dev_guide_attributes_scratchpad
    memory::desc scratchpad_desc() const {
        return query_md(query::scratchpad_md, 0);
    }

    /// @returns The engine that owns the scratchpad memory.
    engine scratchpad_engine() const {
        dnnl_engine_t c_engine;
        error::wrap_c_api(dnnl_primitive_desc_query(get(),
                                  dnnl::convert_to_c(query::scratchpad_engine),
                                  0, &c_engine),
                "could not get scratchpad engine from a primitive_desc");
        return engine(c_engine, true);
    }

    /// Returns the primitive attributes.
    primitive_attr get_primitive_attr() const {
        const_dnnl_primitive_attr_t const_c_attr;
        error::wrap_c_api(dnnl_primitive_desc_get_attr(get(), &const_c_attr),
                "could not get attributes");
        dnnl_primitive_attr_t c_attr;
        error::wrap_c_api(dnnl_primitive_attr_clone(&c_attr, const_c_attr),
                "could not clone attributes");
        return primitive_attr(c_attr);
    }

    /// Returns the kind of the primitive descriptor.
    dnnl::primitive::kind get_kind() const {
        dnnl_primitive_kind_t kind;
        error::wrap_c_api(dnnl_primitive_desc_query(get(),
                                  dnnl_query_primitive_kind, 0, (void *)&kind),
                "could not get primitive kind from the primitive descriptor");
        return static_cast<dnnl::primitive::kind>(kind);
    }

protected:
    void reset_with_clone(const_dnnl_primitive_desc_t pd) {
        dnnl_primitive_desc_t new_pd;
        error::wrap_c_api(dnnl_primitive_desc_clone(&new_pd, pd),
                "could not clone primitive descriptor");
        reset(new_pd);
    }

    primitive_desc_base(
            dnnl_primitive_desc_t pd, dnnl::primitive::kind prim_kind)
        : primitive_desc_base(pd, prim_kind, dnnl::prop_kind::undef) {}

    primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::primitive::kind prim_kind, dnnl::prop_kind prop_kind)
        : primitive_desc_base(pd, prim_kind, prop_kind, prop_kind) {}

    /// Constructs a primitive_desc from a C counterpart. Performs certain
    /// checks to make sure that the C counterpart refers to a primitive
    /// descriptor of a particular primitive kind and propagation kind.
    ///
    /// Note: primitive_desc constructed this way does not support
    /// next_impl().
    primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::primitive::kind prim_kind, dnnl::prop_kind prop_kind1,
            dnnl::prop_kind prop_kind2) {
        // It is OK to pass an empty primitive descriptor
        if (pd == nullptr) return;

        dnnl_status_t rc;

        dnnl_primitive_kind_t c_prim_kind = convert_to_c(prim_kind);
        dnnl_prop_kind_t c_prop_kind1 = convert_to_c(prop_kind1);
        dnnl_prop_kind_t c_prop_kind2 = convert_to_c(prop_kind2);

        // Check that primitive kind matches
        dnnl_primitive_kind_t pd_kind;
        rc = dnnl_primitive_desc_query(
                pd, dnnl_query_primitive_kind, 0, (void *)&pd_kind);
        error::wrap_c_api(rc,
                "could not get primitive kind from the primitive descriptor");
        if (pd_kind != c_prim_kind)
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "primitive descriptor operation kind mismatch");

        // Check that propagation kind matches
        dnnl_prop_kind_t pd_prop_kind;
        rc = dnnl_primitive_desc_query(
                pd, dnnl_query_prop_kind, 0, (void *)&pd_prop_kind);

        // Something went wrong
        if (rc != dnnl_success && rc != dnnl_unimplemented)
            DNNL_THROW_ERROR(dnnl_invalid_arguments,
                    "could not get propagation kind "
                    "from the primitive descriptor");

        // Everything is fine
        if ((rc == dnnl_unimplemented && c_prop_kind1 == dnnl_prop_kind_undef)
                || (rc == dnnl_success
                        && (pd_prop_kind == c_prop_kind1
                                || pd_prop_kind == c_prop_kind2))) {
            reset_with_clone(pd);
            return;
        }

        // We could get the propagation kind but there is a mismatch
        DNNL_THROW_ERROR(dnnl_invalid_arguments,
                "primitive descriptor propagation kind mismatch");
    }

protected:
    using base = primitive_desc_base;
};

/// @}

/// @}

/// @addtogroup cpp_api_memory_related Memory and memory related operations
/// @{

/// @addtogroup cpp_api_reorder Reorder
/// A primitive to copy data between memory formats.
///
/// @sa @ref dev_guide_reorder in developer guide
/// @sa @ref c_api_reorder in @ref c_api
/// @{

/// Reorder primitive.
struct reorder : public primitive {
    /// Primitive descriptor for a reorder primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a reorder primitive using the description of the source
        /// (@p src_engine and @p src_md) and destination (@p dst_engine and
        /// @p dst_md) memory, and an @p attr attributes.
        primitive_desc(const engine &src_engine, const memory::desc &src_md,
                const engine &dst_engine, const memory::desc &dst_md,
                const primitive_attr &attr = primitive_attr()) {
            dnnl_primitive_desc_t result;
            error::wrap_c_api(
                    dnnl_reorder_primitive_desc_create(&result, &src_md.data,
                            src_engine.get(), &dst_md.data, dst_engine.get(),
                            attr.get()),
                    "could not create a reorder primitive descriptor");
            reset(result);
        }

        /// Constructs a reorder primitive moving data between memory objects
        /// @p src and @p dat with @p attr attributes.
        ///
        /// Note that the memory objects are used only to obtain information
        /// about memory descriptors and engines.
        primitive_desc(const memory &src, const memory &dst,
                const primitive_attr &attr = primitive_attr()) {
            dnnl_primitive_desc_t result;
            auto src_md = src.get_desc();
            auto dst_md = dst.get_desc();
            error::wrap_c_api(
                    dnnl_reorder_primitive_desc_create(&result, &src_md.data,
                            src.get_engine().get(), &dst_md.data,
                            dst.get_engine().get(), attr.get()),
                    "could not create a reorder primitive descriptor");
            reset(result);
        }

        /// Constructs a reorder primitive descriptor for reorder from a C
        /// primitive descriptor @p pd which must have a matching kind.
        primitive_desc(dnnl_primitive_desc_t pd)
            : primitive_desc_base(pd, dnnl::primitive::kind::reorder) {}

        /// Returns the engine on which the source memory is allocated.
        engine get_src_engine() const {
            return engine::query(*this, dnnl::query::reorder_src_engine);
        }

        /// Returns the engine on which the destination memory is allocated.
        engine get_dst_engine() const {
            return engine::query(*this, dnnl::query::reorder_dst_engine);
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    reorder() = default;

    /// Constructs a reorder primitive from a primitive descriptor @p pd
    /// of a corresponding type.
    reorder(const primitive_desc &pd) : primitive(pd.get()) {}

    /// Constructs a reorder primitive that would reorder data between memory
    /// objects having the same memory descriptors as memory objects @p src and
    /// @p dst.
    ///
    /// @param src Source memory object.
    /// @param dst Destination memory object.
    /// @param attr Primitive attributes to use (optional).
    reorder(const memory &src, const memory &dst,
            const primitive_attr &attr = primitive_attr())
        : primitive(primitive_desc(src, dst, attr).get()) {}

    using primitive::execute;

    /// Executes the reorder primitive.
    ///
    /// @param stream Stream object. The stream must belong to the same engine
    ///               as the primitive.
    /// @param src Source memory object.
    /// @param dst Destination memory object.
    void execute(stream stream, memory &src, memory &dst) {
        primitive::execute(stream, {{DNNL_ARG_FROM, src}, {DNNL_ARG_TO, dst}});
    }
};

/// @}

/// @addtogroup cpp_api_concat Concat
/// A primitive to concatenate data by arbitrary dimension.
///
/// @sa @ref dev_guide_concat in developer guide
/// @sa @ref c_api_concat in @ref c_api
/// @{

/// @cond DO_NOT_DOCUMENT_THIS
inline std::vector<dnnl_memory_desc_t> convert_to_c(
        const std::vector<memory::desc> &mems) {
    std::vector<dnnl_memory_desc_t> c_mems;
    c_mems.reserve(mems.size());
    for (const auto &s : mems)
        c_mems.push_back(s.data);
    return c_mems;
}
/// @endcond

/// Tensor concatenation (concat) primitive.
struct concat : public primitive {
    /// Primitive descriptor for a concat primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an out-of-place concatenation
        /// primitive.
        ///
        /// @param dst Destination memory descriptor
        /// @param concat_dimension Source tensors will be concatenated over
        ///                         dimension with this index. Note that order
        ///                         of dimensions does not depend on memory
        ///                         format.
        /// @param srcs Vector of source memory descriptors.
        /// @param engine Engine to perform the operation on.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(const memory::desc &dst, int concat_dimension,
                const std::vector<memory::desc> &srcs, const engine &engine,
                const primitive_attr &attr = primitive_attr()) {
            auto c_srcs = convert_to_c(srcs);

            dnnl_primitive_desc_t result;
            error::wrap_c_api(
                    dnnl_concat_primitive_desc_create(&result, &dst.data,
                            (int)c_srcs.size(), concat_dimension, &c_srcs[0],
                            attr.get(), engine.get()),
                    "could not create a concat primitive descriptor");
            reset(result);
        }

        /// Constructs a primitive descriptor for an out-of-place concatenation
        /// primitive.
        ///
        /// This version derives the destination memory descriptor
        /// automatically.
        ///
        /// @param concat_dimension Source tensors will be concatenated over
        ///                         dimension with this index. Note that order
        ///                         of dimensions does not depend on memory
        ///                         format.
        /// @param srcs Vector of source memory descriptors.
        /// @param engine Engine to perform the operation on.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(int concat_dimension,
                const std::vector<memory::desc> &srcs, const engine &engine,
                const primitive_attr &attr = primitive_attr()) {
            auto c_api_srcs = convert_to_c(srcs);

            dnnl_primitive_desc_t result;
            error::wrap_c_api(
                    dnnl_concat_primitive_desc_create(&result, nullptr,
                            (int)c_api_srcs.size(), concat_dimension,
                            &c_api_srcs[0], attr.get(), engine.get()),
                    "could not create a concat primitive descriptor");
            reset(result);
        }

        /// Initializes a primitive descriptor for concat from a C primitive
        /// descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : primitive_desc_base(pd, dnnl::primitive::kind::concat) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc(int)
        memory::desc src_desc(int idx = 0) const { return base::src_desc(idx); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    concat() = default;

    /// Constructs a concatenation primitive from a primitive descriptor @p pd
    /// of a corresponding type.
    concat(const primitive_desc &pd) : primitive(pd.get()) {}
};

/// @}

/// @addtogroup cpp_api_sum Sum
/// A primitive to sum multiple tensors.
///
/// @sa @ref dev_guide_sum in developer guide
/// @sa @ref c_api_sum in @ref c_api
/// @{

/// Summation (sum) primitive.
struct sum : public primitive {
    /// Primitive descriptor for a sum primitive.
    struct primitive_desc : public primitive_desc_base {
        using primitive_desc_base::primitive_desc_base;

        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs an out-of-place primitive descriptor for a sum primitive.
        ///
        /// @param dst Destination memory descriptor.
        /// @param scales Vector of scales to multiply data in each source
        ///               memory by.
        /// @param srcs Vector of source memory descriptors.
        /// @param engine Engine to perform the operation on.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(const memory::desc &dst,
                const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs, const engine &engine,
                const primitive_attr &attr = primitive_attr()) {
            error::wrap_c_api(scales.size() == srcs.size()
                            ? dnnl_success
                            : dnnl_invalid_arguments,
                    "number of scales not equal to number of srcs");

            auto c_api_srcs = convert_to_c(srcs);

            dnnl_primitive_desc_t result;
            error::wrap_c_api(dnnl_sum_primitive_desc_create(&result, &dst.data,
                                      (int)c_api_srcs.size(), &scales[0],
                                      &c_api_srcs[0], attr.get(), engine.get()),
                    "could not create a sum primitive descriptor");
            reset(result);
        }

        /// Constructs an out-of-place primitive descriptor for a sum primitive.
        ///
        /// This version derives the destination memory descriptor
        /// automatically.
        ///
        /// @param scales Vector of scales by which to multiply data in each
        ///               source memory.
        /// @param srcs Vector of source memory descriptors.
        /// @param engine Engine on which to perform the operation.
        /// @param attr Primitive attributes to use (optional).
        primitive_desc(const std::vector<float> &scales,
                const std::vector<memory::desc> &srcs, const engine &engine,
                const primitive_attr &attr = primitive_attr()) {
            error::wrap_c_api(scales.size() == srcs.size()
                            ? dnnl_success
                            : dnnl_invalid_arguments,
                    "number of scales not equal to number of srcs");

            auto c_api_srcs = convert_to_c(srcs);
            dnnl_primitive_desc_t result;
            error::wrap_c_api(dnnl_sum_primitive_desc_create(&result, nullptr,
                                      (int)c_api_srcs.size(), &scales[0],
                                      &c_api_srcs[0], attr.get(), engine.get()),
                    "could not create a sum primitive descriptor");
            reset(result);
        }

        /// Constructs a primitive descriptor for a sum primitive from a C API
        /// primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : primitive_desc_base(pd, dnnl::primitive::kind::sum) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc(int)
        memory::desc src_desc(int idx = 0) const { return base::src_desc(idx); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    sum() = default;

    /// Constructs a sum primitive from a primitive descriptor @p pd of a
    /// corresponding type.
    sum(const primitive_desc &pd) : primitive(pd.get()) {}
};

/// @}

/// @}

/// @addtogroup cpp_api_primitives Primitives
/// @{

/// @addtogroup cpp_api_primitive_descriptors Primitive descriptors
/// @{

/// A base class for descriptors of all primitives that have an operation
/// descriptor and that support iteration over multiple implementations.
struct primitive_desc : public primitive_desc_base {
    using primitive_desc_base::primitive_desc_base;

    primitive_desc() = default;

    /// Creates a primitive descriptor from an @p op_desc, @p attr, @p engine,
    /// and, optionally, a primitive descriptor hint from forward propagation.
    /// If @p allow_empty is true, the constructor does not throw if a
    /// primitive descriptor cannot be created. But calling next_impl() in
    /// this case *will* throw.
    primitive_desc(const_dnnl_op_desc_t desc, const primitive_attr *attr,
            const engine &engine, const_dnnl_primitive_desc_t hint_fwd_pd,
            bool allow_empty = false)
        : allow_empty_(allow_empty) {
        dnnl_primitive_desc_iterator_t iterator = nullptr;
        dnnl_status_t status = dnnl_primitive_desc_iterator_create(&iterator,
                desc, attr ? attr->get() : nullptr, engine.get(), hint_fwd_pd);
        if (!allow_empty)
            error::wrap_c_api(
                    status, "could not create a primitive descriptor iterator");
        pd_iterator.reset(iterator);
        fetch_impl();
    }

    /// Advances the next implementation for the given op descriptor.
    ///
    /// Returns:
    /// - @c true on success
    /// - @c false if the last implementation reached, and
    ///   the primitive descriptor itself is kept unchanged
    bool next_impl() {
        dnnl_status_t status
                = dnnl_primitive_desc_iterator_next(pd_iterator.get());
        if (status == dnnl_iterator_ends) return false;
        error::wrap_c_api(status, "primitive descriptor iterator next failed");
        fetch_impl();
        return true;
    }

private:
    bool allow_empty_ = false;
    handle<dnnl_primitive_desc_iterator_t> pd_iterator;
    void fetch_impl() {
        dnnl_primitive_desc_t pd = dnnl_primitive_desc_iterator_fetch(
                pd_iterator.get(allow_empty_));
        error::wrap_c_api(pd != nullptr || allow_empty_ ? dnnl_success
                                                        : dnnl_runtime_error,
                "could not fetch a primitive descriptor from the iterator");
        reset(pd);
    }
};

/// @}

/// @addtogroup cpp_api_convolution Convolution
/// Computes a forward propagation, backward propagation, or weight update
/// for convolution operation with bias on a batch of multi-dimensional tensors.
///
/// @sa @ref dev_guide_convolution in developer guide
/// @sa @ref c_api_convolution in @ref c_api
/// @{

/// Convolution forward propagation primitive.
struct convolution_forward : public primitive {
    /// Descriptor for a convolution forward propagation primitive.
    struct desc {
        dnnl_convolution_desc_t data;

        /// Constructs a descriptor for a convolution forward propagation
        /// primitive with bias using @p prop_kind (acceptable values are
        /// #dnnl::forward_training and #dnnl::forward_inference), @p
        /// algorithm, memory descriptors, @p strides, @p padding_l, and @p
        /// padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_convolution_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            convert_to_c(algorithm), &src_desc.data,
                            &weights_desc.data, &bias_desc.data, &dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a convolution forward descriptor");
        }

        /// Constructs a descriptor for a convolution forward propagation
        /// primitive without bias using @p prop_kind (acceptable values are
        /// #dnnl::forward_training and #dnnl::forward_inference), @p
        /// algorithm, memory descriptors, @p strides, @p padding_l, and @p
        /// padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_convolution_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            convert_to_c(algorithm), &src_desc.data,
                            &weights_desc.data, nullptr, &dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a convolution forward descriptor");
        }

        /// Constructs a descriptor for a dilated convolution forward
        /// propagation primitive with bias using @p prop_kind (possible
        /// values are #dnnl::forward_training and #dnnl::forward_inference),
        /// @p algorithm, memory descriptors, @p strides, @p dilates, @p
        /// padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &dilates,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(dnnl_dilated_convolution_forward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      convert_to_c(algorithm), &src_desc.data,
                                      &weights_desc.data, &bias_desc.data,
                                      &dst_desc.data, &strides[0], &dilates[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a dilated convolution forward "
                    "descriptor");
        }

        /// Constructs a descriptor for a dilated convolution forward
        /// propagation primitive without bias using @p prop_kind (possible
        /// values are #dnnl::forward_training and #dnnl::forward_inference),
        /// @p algorithm, memory descriptors, @p strides, @p dilates, @p
        /// padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(dnnl_dilated_convolution_forward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      convert_to_c(algorithm), &src_desc.data,
                                      &weights_desc.data, nullptr,
                                      &dst_desc.data, &strides[0], &dilates[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a dilated convolution forward "
                    "descriptor");
        }
    };

    /// Primitive descriptor for a convolution forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a convolution forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a convolution forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a convolution forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Initializes a primitive descriptor for convolution forward
        /// propagation from a C primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::convolution,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// Returns the bias memory descriptor. Returns a zero_md if no bias
        /// was specified at op_desc creation time.
        memory::desc bias_desc() const { return base::weights_desc(1); }
    };

    /// Default constructor. Produces an empty object.
    convolution_forward() = default;

    /// Constructs a convolution forward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    convolution_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Convolution backward propagation primitive.
struct convolution_backward_data : public primitive {

    /// Descriptor for a convolution backward propagation primitive.
    struct desc {
        dnnl_convolution_desc_t data;

        /// Constructs a descriptor for a convolution backward propagation
        /// primitive using @p algorithm, memory descriptors, @p strides, @p
        /// padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_convolution_backward_data_desc_init(&data,
                            convert_to_c(algorithm), &diff_src_desc.data,
                            &weights_desc.data, &diff_dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a convolution backward data descriptor");
        }

        /// Constructs a descriptor for dilated convolution backward
        /// propagation using @p algorithm, memory descriptors, @p strides, @p
        /// dilates, @p padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_dilated_convolution_backward_data_desc_init(&data,
                            convert_to_c(algorithm), &diff_src_desc.data,
                            &weights_desc.data, &diff_dst_desc.data,
                            &strides[0], &dilates[0], &padding_l[0],
                            &padding_r[0]),
                    "could not create a convolution backward data descriptor");
        }
    };

    /// Primitive descriptor for a convolution backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a convolution backward propagation
        ///             primitive.
        /// @param engine Engine to perform the operation on.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a convolution backward propagation
        ///             primitive.
        /// @param engine Engine to perform the operation on.
        /// @param attr Primitive attributes to use.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Constructs a primitive descriptor for a convolution backward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::convolution,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    convolution_backward_data() = default;

    /// Constructs a convolution backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    convolution_backward_data(const primitive_desc &pd) : primitive(pd) {}
};

/// Convolution weights update primitive.
struct convolution_backward_weights : public primitive {
    /// Descriptor for a convolution weights update primitive.
    struct desc {
        dnnl_convolution_desc_t data;

        /// Constructs a descriptor for a convolution weights update primitive
        /// with bias using @p algorithm, memory descriptors, @p strides, @p
        /// padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_convolution_backward_weights_desc_init(&data,
                            convert_to_c(algorithm), &src_desc.data,
                            &diff_weights_desc.data, &diff_bias_desc.data,
                            &diff_dst_desc.data, &strides[0], &padding_l[0],
                            &padding_r[0]),
                    "could not create a convolution backward weights "
                    "descriptor");
        }

        /// Constructs a descriptor for a convolution weights update primitive
        /// without bias using @p algorithm, memory descriptors, @p strides,
        /// @p padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(dnnl_convolution_backward_weights_desc_init(&data,
                                      convert_to_c(algorithm), &src_desc.data,
                                      &diff_weights_desc.data, nullptr,
                                      &diff_dst_desc.data, &strides[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a convolution backward weights "
                    "descriptor");
        }

        /// Constructs a descriptor for a dilated convolution weights update
        /// primitive with bias using @p algorithm, memory descriptors, @p
        /// strides, @p dilates @p padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_dilated_convolution_backward_weights_desc_init(&data,
                            convert_to_c(algorithm), &src_desc.data,
                            &diff_weights_desc.data, &diff_bias_desc.data,
                            &diff_dst_desc.data, &strides[0], &dilates[0],
                            &padding_l[0], &padding_r[0]),
                    "could not create a convolution backward weights "
                    "descriptor");
        }

        /// Constructs a descriptor for a dilated convolution weights update
        /// primitive without bias using @p algorithm, memory descriptors, @p
        /// strides, @p dilates @p padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_dilated_convolution_backward_weights_desc_init(&data,
                            convert_to_c(algorithm), &src_desc.data,
                            &diff_weights_desc.data, nullptr,
                            &diff_dst_desc.data, &strides[0], &dilates[0],
                            &padding_l[0], &padding_r[0]),
                    "could not create a convolution backward weights "
                    "descriptor");
        }
    };

    /// Primitive descriptor for a convolution weights update primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a convolution weights update
        /// primitive.
        ///
        /// @param desc Descriptor for a convolution weights update primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights update
        /// primitive.
        ///
        /// @param desc Descriptor for a convolution weights update primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a convolution forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const convolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Constructs a primitive descriptor for a convolution weights
        /// update primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::convolution,
                    dnnl::prop_kind::backward_weights) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// Returns diff bias memory descriptor. Returns a zero_md if no bias
        /// was specified at op_desc creation time.
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    convolution_backward_weights() = default;

    /// Constructs a convolution weights update primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    convolution_backward_weights(const primitive_desc &pd) : primitive(pd) {}
};

/// @}
//
/// @addtogroup cpp_api_deconvolution Deconvolution
/// A primitive to compute deconvolution using different algorithms.
///
/// @sa @ref c_api_deconvolution in @ref c_api
/// @{

/// Deconvolution forward propagation primitive.
struct deconvolution_forward : public primitive {
    /// Descriptor for a deconvolution forward propagation primitive.
    struct desc {
        dnnl_deconvolution_desc_t data;

        /// Constructs a descriptor for a deconvolution forward propagation
        /// primitive with bias using @p prop_kind (acceptable values are
        /// #dnnl::forward_training and #dnnl::forward_inference), @p
        /// algorithm, memory descriptors, @p strides, @p padding_l, and @p
        /// padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_deconvolution_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            convert_to_c(algorithm), &src_desc.data,
                            &weights_desc.data, &bias_desc.data, &dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a deconvolution forward descriptor");
        }

        /// Constructs a descriptor for a deconvolution forward propagation
        /// primitive without bias using @p prop_kind (acceptable values are
        /// #dnnl::forward_training and #dnnl::forward_inference), @p
        /// algorithm, memory descriptors, @p strides, @p padding_l, and @p
        /// padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_deconvolution_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            convert_to_c(algorithm), &src_desc.data,
                            &weights_desc.data, nullptr, &dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a deconvolution forward descriptor");
        }

        /// Constructs a descriptor for a dilated deconvolution forward
        /// propagation primitive with bias using @p prop_kind (possible
        /// values are #dnnl::forward_training and #dnnl::forward_inference),
        /// @p algorithm memory descriptors, @p strides, @p dilates, @p
        /// padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &dilates,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(dnnl_dilated_deconvolution_forward_desc_init(
                                      &data, dnnl::convert_to_c(prop_kind),
                                      convert_to_c(algorithm), &src_desc.data,
                                      &weights_desc.data, &bias_desc.data,
                                      &dst_desc.data, &strides[0], &dilates[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a dilated deconvolution forward "
                    "descriptor");
        }

        /// Constructs a descriptor for a dilated deconvolution forward
        /// propagation primitive without bias using @p prop_kind (possible
        /// values are #dnnl::forward_training and #dnnl::forward_inference),
        /// @p algorithm, memory descriptors, @p strides, @p dilates, @p
        /// padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(dnnl_dilated_deconvolution_forward_desc_init(
                                      &data, dnnl::convert_to_c(prop_kind),
                                      convert_to_c(algorithm), &src_desc.data,
                                      &weights_desc.data, nullptr,
                                      &dst_desc.data, &strides[0], &dilates[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a dilated deconvolution forward "
                    "descriptor");
        }
    };

    /// Primitive descriptor for a deconvolution forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a deconvolution forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a deconvolution forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution forward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::deconvolution,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()
        memory::desc bias_desc() const { return base::weights_desc(1); }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_forward() = default;

    /// Constructs a deconvolution forward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    deconvolution_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Deconvolution backward propagation primitive.
struct deconvolution_backward_data : public primitive {
    /// Descriptor for a deconvolution backward propagation primitive.
    struct desc {
        dnnl_deconvolution_desc_t data;

        /// Constructs a descriptor for a deconvolution backward propagation
        /// primitive using @p algorithm, memory descriptors, @p strides, @p
        /// padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_deconvolution_backward_data_desc_init(&data,
                            convert_to_c(algorithm), &diff_src_desc.data,
                            &weights_desc.data, &diff_dst_desc.data,
                            &strides[0], &padding_l[0], &padding_r[0]),
                    "could not create a deconvolution backward data "
                    "descriptor");
        }

        /// Constructs a descriptor for a dilated deconvolution backward
        /// propagation primitive using @p algorithm, memory descriptors, @p
        /// strides, @p dilates, @p padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_dilated_deconvolution_backward_data_desc_init(&data,
                            convert_to_c(algorithm), &diff_src_desc.data,
                            &weights_desc.data, &diff_dst_desc.data,
                            &strides[0], &dilates[0], &padding_l[0],
                            &padding_r[0]),
                    "could not create a dilated deconvolution backward data "
                    "descriptor");
        }
    };

    /// Primitive descriptor for a deconvolution backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive.
        ///
        /// @param desc descriptor for a deconvolution backward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive.
        ///
        /// @param desc descriptor for a deconvolution backward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution backward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::deconvolution,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_backward_data() = default;

    /// Constructs a deconvolution backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    deconvolution_backward_data(const primitive_desc &pd) : primitive(pd) {}
};

/// Deconvolution weights update primitive.
struct deconvolution_backward_weights : public primitive {
    /// Descriptor for a deconvolution weights update primitive.
    struct desc {
        dnnl_deconvolution_desc_t data;

        /// Constructs a descriptor for a deconvolution weights update primitive
        /// with bias using @p algorithm, memory descriptors, @p strides, @p
        /// padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_deconvolution_backward_weights_desc_init(&data,
                            convert_to_c(algorithm), &src_desc.data,
                            &diff_weights_desc.data, &diff_bias_desc.data,
                            &diff_dst_desc.data, &strides[0], &padding_l[0],
                            &padding_r[0]),
                    "could not create a deconvolution weights update primitive "
                    "descriptor");
        }

        /// Constructs a descriptor for a deconvolution weights update primitive
        /// without bias using @p algorithm, memory descriptors, @p strides,
        /// @p padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(dnnl_deconvolution_backward_weights_desc_init(
                                      &data, convert_to_c(algorithm),
                                      &src_desc.data, &diff_weights_desc.data,
                                      nullptr, &diff_dst_desc.data, &strides[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not create a deconvolution weights update primitive "
                    "descriptor");
        }

        /// Constructs a descriptor for a dilated deconvolution weights update
        /// primitive with bias using @p algorithm, memory descriptors, @p
        /// strides, @p dilates @p padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_dilated_deconvolution_backward_weights_desc_init(&data,
                            convert_to_c(algorithm), &src_desc.data,
                            &diff_weights_desc.data, &diff_bias_desc.data,
                            &diff_dst_desc.data, &strides[0], &dilates[0],
                            &padding_l[0], &padding_r[0]),
                    "could not create a dilated  weights update primitive "
                    "descriptor");
        }

        /// Constructs a descriptor for a dilated deconvolution weights update
        /// primitive without bias using @p algorithm, memory descriptors, @p
        /// strides, @p dilates @p padding_l, and @p padding_r.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(algorithm algorithm, const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &dilates, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(dilates);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_dilated_deconvolution_backward_weights_desc_init(&data,
                            convert_to_c(algorithm), &src_desc.data,
                            &diff_weights_desc.data, nullptr,
                            &diff_dst_desc.data, &strides[0], &dilates[0],
                            &padding_l[0], &padding_r[0]),
                    "could not create a dilated deconvolution weights update "
                    "primitive descriptor");
        }
    };

    /// Primitive descriptor for a deconvolution weights update primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a deconvolution weights
        /// update primitive.
        ///
        /// @param desc descriptor for a deconvolution weights update primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        /// update primitive.
        ///
        /// @param desc descriptor for a deconvolution weights update primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a deconvolution forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const deconvolution_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Constructs a primitive descriptor for a deconvolution weights
        /// update primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::deconvolution,
                    dnnl::prop_kind::backward_weights) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::convolution_backward_weights::primitive_desc::diff_bias_desc()
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    deconvolution_backward_weights() = default;

    /// Constructs a deconvolution weights update primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    deconvolution_backward_weights(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_lrn LRN
/// A primitive to perform local response normalization (LRN) across or within
/// channels.
///
/// @sa @ref dev_guide_lrn in developer guide
/// @sa @ref c_api_lrn in @ref c_api
/// @{

/// Local response normalization (LRN) forward propagation primitive.
struct lrn_forward : public primitive {
    /// Descriptor for an LRN forward propagation primitive.
    struct desc {
        dnnl_lrn_desc_t data;

        /// Constructs a descriptor for a LRN forward propagation primitive
        /// using @p prop_kind (acceptable values are #dnnl::forward_training
        /// and #dnnl::forward_inference), @p algorithm, memory descriptor @p
        /// data_desc, and regularization parameters @p local_size, @p alpha,
        /// @p beta, and @p k.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, memory::dim local_size,
                float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(dnnl_lrn_forward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      convert_to_c(algorithm), &src_desc.data,
                                      local_size, alpha, beta, k),
                    "could not create a lrn forward descriptor");
        }
    };

    /// Primitive descriptor for an LRN forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LRN forward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for an LRN forward propagation primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LRN forward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for an LRN forward propagation primitive.
        /// @param engine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Construct a primitive descriptor for an LRN forward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::lrn,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    lrn_forward() = default;

    /// Constructs an LRN forward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    lrn_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Local response normalization (LRN) backward propagation primitive.
struct lrn_backward : public primitive {
    /// Descriptor for an LRN backward propagation primitive.
    struct desc {
        dnnl_lrn_desc_t data;

        /// Constructs a descriptor for an LRN backward propagation primitive
        /// using @p algorithm, memory descriptors @p data_desc and @p
        /// diff_data_desc, and regularization parameters @p local_size, @p
        /// alpha, @p beta, and @p k.
        desc(algorithm algorithm, const memory::desc &data_desc,
                const memory::desc &diff_data_desc, memory::dim local_size,
                float alpha, float beta, float k = 1.f) {
            error::wrap_c_api(
                    dnnl_lrn_backward_desc_init(&data, convert_to_c(algorithm),
                            &diff_data_desc.data, &data_desc.data, local_size,
                            alpha, beta, k),
                    "could not create a lrn backward descriptor");
        }
    };

    /// Primitive descriptor for an LRN backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LRN backward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for an LRN backward propagation primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LRN forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const lrn_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LRN backward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for an LRN backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LRN forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const lrn_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Constructs a primitive descriptor for a local response normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::lrn,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    lrn_backward() = default;

    /// Constructs an LRN backward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    lrn_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_pooling Pooling
/// A primitive to perform max or average pooling.
///
/// @sa @ref dev_guide_pooling in developer guide
/// @sa @ref c_api_pooling in @ref c_api
/// @{

/// Pooling forward propagation primitive.
struct pooling_forward : public primitive {
    /// Descriptor for a pooling forward propagation primitive.
    struct desc {
        dnnl_pooling_desc_t data;

        /// Constructs a descriptor for a pooling forward propagation primitive
        /// using @p prop_kind (acceptable values are #dnnl::forward_training
        /// and #dnnl::forward_inference), @p algorithm, memory descriptors,
        /// and pooling parameters in the spatial domain: @p strides, @p
        /// kernel sizes, @p padding_l, and @p padding_r.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                const memory::dims &strides, const memory::dims &kernel,
                const memory::dims &padding_l, const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(dnnl_pooling_forward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      convert_to_c(algorithm), &src_desc.data,
                                      &dst_desc.data, &strides[0], &kernel[0],
                                      &padding_l[0], &padding_r[0]),
                    "could not init a forward pooling descriptor");
        }
    };

    /// Primitive descriptor for a pooling forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a pooling forward propagation primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a pooling forward propagation primitive.
        /// @param engine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a pooling forward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::pooling,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    pooling_forward() = default;

    /// Constructs an pooling forward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    pooling_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Pooling backward propagation primitive.
struct pooling_backward : public primitive {
    /// Descriptor for a pooling backward propagation primitive.
    struct desc {
        dnnl_pooling_desc_t data;

        /// Constructs a pooling descriptor for a pooling backward propagation
        /// primitive using @p algorithm, memory descriptors, and pooling
        /// parameters in the spatial domain: @p strides, @p kernel sizes, @p
        /// padding_l, and @p padding_r.
        desc(algorithm algorithm, const memory::desc &diff_src_desc,
                const memory::desc &diff_dst_desc, const memory::dims &strides,
                const memory::dims &kernel, const memory::dims &padding_l,
                const memory::dims &padding_r) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(
                    dnnl_pooling_backward_desc_init(&data,
                            convert_to_c(algorithm), &diff_src_desc.data,
                            &diff_dst_desc.data, &strides[0], &kernel[0],
                            &padding_l[0], &padding_r[0]),
                    "could not init a backward pooling descriptor");
        }
    };

    /// Primitive descriptor for a pooling backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a pooling backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a pooling backward propagation primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const pooling_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a pooling backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a pooling backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a pooling forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const pooling_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Creates a primitive descriptor for a pooling backward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::pooling,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    pooling_backward() = default;

    /// Constructs an pooling backward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    pooling_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_eltwise Eltwise
/// A primitive to compute element-wise operations such as rectified linear
/// unit (ReLU).
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can point to the same memory for forward
/// propagation, and diff_dst and diff_src can point to the same memory for
/// backward propagation.
///
/// @warning
///     Because the original source data is required for traning backward
///     propagation, in-place training forward propagation is not generally
///     supported.  However, for some kinds of element-wise operations (namely
///     ReLU with alpha parameter equals 0), dst and src can be
///     interchangeable for the backward propagation, which enables
///     performance of in-place forward even for training.
///
/// @sa @ref dev_guide_eltwise in developer guide
/// @sa @ref c_api_eltwise in @ref c_api
/// @{

/// Elementwise unary operation forward propagation primitive.
///
/// This primitive computes unary elementwise operations such as rectified
/// linear unit (ReLU), Sigmoid, tanh or others for each element of the
/// source.
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst can point to the same memory for forward
/// propagation, and diff_dst and diff_src can point to the same memory for
/// backward propagation.
struct eltwise_forward : public primitive {
    /// Descriptor for an elementwise forward propagation primitive.
    struct desc {
        dnnl_eltwise_desc_t data;

        /// Constructs a descriptor for an elementwise forward propagation
        /// primitive using @p prop_kind (acceptable values are
        /// #dnnl::forward_training and #dnnl::forward_inference), @p
        /// algorithm algorithm, memory descriptor @p data_desc, @p alpha, and
        /// @p beta parameters.
        desc(prop_kind prop_kind, algorithm algorithm,
                const memory::desc &src_desc, float alpha = 0, float beta = 0) {
            error::wrap_c_api(dnnl_eltwise_forward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      dnnl::convert_to_c(algorithm),
                                      &src_desc.data, alpha, beta),
                    "could not create an eltwise forward descriptor");
        }
    };

    /// Primitive descriptor for an elementwise forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for an elementwise forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for an elementwise forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Creates a primitive descriptor for an elementwise forward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::eltwise,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    eltwise_forward() = default;

    /// Constructs an elementwise forward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    eltwise_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Elementwise unary operation backward propagation primitive.
///
/// @sa eltwise_forward
struct eltwise_backward : public primitive {
    /// Descriptor for an elementwise backward propagation primitive.
    struct desc {
        dnnl_eltwise_desc_t data;

        /// Constructs an eltwise descriptor for backward propagation using @p
        /// algorithm algorithm memory descriptors @p diff_data_desc and @p
        /// data_desc, and the @p alpha and @p beta parameters.
        desc(algorithm algorithm, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, float alpha = 0,
                float beta = 0) {
            error::wrap_c_api(
                    dnnl_eltwise_backward_desc_init(&data,
                            dnnl::convert_to_c(algorithm), &diff_data_desc.data,
                            &data_desc.data, alpha, beta),
                    "could not create an eltwise backward descriptor");
        }
    };

    /// Primitive descriptor for eltwise backward propagation.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for an elementwise backward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an elementwise forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const eltwise_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for an elementwise backward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an elementwise forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const eltwise_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Initializes a primitive descriptor for element-wise operations for
        /// backward propagation from a C primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::eltwise,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    eltwise_backward() = default;

    /// Constructs an elementwise backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    eltwise_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_softmax Softmax
/// A primitive to perform softmax.
///
/// @sa @ref dev_guide_softmax in developer guide
/// @sa @ref c_api_softmax in @ref c_api
/// @{

/// Softmax forward propagation primitive.
struct softmax_forward : public primitive {
    /// Descriptor for a softmax forward propagation primitive.
    struct desc {
        dnnl_softmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a softmax descriptor for forward propagation using @p
        /// prop_kind (acceptable values are #dnnl::forward_training and
        /// #dnnl::forward_inference) and memory descriptor @p data_desc.
        desc(prop_kind prop_kind, const memory::desc &data_desc,
                int softmax_axis) {
            error::wrap_c_api(dnnl_softmax_forward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      &data_desc.data, softmax_axis),
                    "could not create a softmax forward descriptor");
        }

        /// Constructs a softmax descriptor for a softmax forward propagation
        /// primitive using a C API softmax descriptor @p data.
        desc(dnnl_softmax_desc_t data) {
            error::wrap_c_api(
                    dnnl_softmax_forward_desc_init(&this->data, data.prop_kind,
                            &data.data_desc, data.softmax_axis),
                    "could not create a softmax forward descriptor");
        }
    };

    /// Primitive descriptor for a softmax forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive.
        ///
        /// @param desc descriptor for a softmax forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a softmax forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a softmax forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Creates a primitive descriptor for a softmax forward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @returns The underlying operation descriptor.
        desc op_desc() const {
            dnnl_softmax_desc_t *data;
            error::wrap_c_api(
                    dnnl_primitive_desc_query(
                            get(), dnnl::convert_to_c(query::op_d), 0, &data),
                    "could not retrive a softmax op desc");
            return desc(*data);
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    softmax_forward() = default;

    /// Constructs an softmax forward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    softmax_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Softmax backward propagation primitive.
struct softmax_backward : public primitive {
    /// Descriptor for a softmax backward propagation primitive.
    struct desc {
        dnnl_softmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a softmax descriptor for backward propagation using
        /// memory descriptors @p diff_desc and @p data_desc.
        desc(const memory::desc &diff_desc, const memory::desc &data_desc,
                int softmax_axis) {
            error::wrap_c_api(
                    dnnl_softmax_backward_desc_init(&data, &diff_desc.data,
                            &data_desc.data, softmax_axis),
                    "could not init a backward softmax descriptor");
        }

        /// Constructs a softmax descriptor for a softmax backward propagation
        /// primitive using a C API softmax descriptor @p data.
        desc(dnnl_softmax_desc_t data) {
            error::wrap_c_api(dnnl_softmax_backward_desc_init(&this->data,
                                      &data.diff_desc, &data.data_desc,
                                      data.softmax_axis),
                    "could not create a softmax forward descriptor");
        }
    };

    /// Primitive descriptor for a softmax backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a softmax backward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a softmax forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const softmax_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a softmax backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a softmax backward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a softmax forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const softmax_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Creates a primitive descriptor for a softmax backward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::softmax_forward::primitive_desc::op_desc()
        desc op_desc() const {
            dnnl_softmax_desc_t *data;
            error::wrap_c_api(
                    dnnl_primitive_desc_query(
                            get(), dnnl::convert_to_c(query::op_d), 0, &data),
                    "could not retrive a softmax op desc");
            return desc(*data);
        }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    softmax_backward() = default;

    /// Constructs an softmax backward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    softmax_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_logsoftmax LogSoftmax
/// A primitive to perform logsoftmax.
///
/// @sa @ref dev_guide_logsoftmax in developer guide
/// @sa @ref c_api_logsoftmax in @ref c_api
/// @{

/// Logsoftmax forward propagation primitive.
struct logsoftmax_forward : public primitive {
    /// Descriptor for a logsoftmax forward propagation primitive.
    struct desc {
        dnnl_logsoftmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Creates a logsoftmax descriptor for forward propagation using @p
        /// prop_kind (possible values are #dnnl::forward_training and
        /// #dnnl::forward_inference) and memory descriptor @p data_desc.
        desc(prop_kind prop_kind, const memory::desc &data_desc,
                int logsoftmax_axis) {
            error::wrap_c_api(dnnl_logsoftmax_forward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      &data_desc.data, logsoftmax_axis),
                    "could not create a logsoftmax forward descriptor");
        }

        /// Constructs a logsoftmax descriptor for a logsoftmax forward
        /// propagation primitive using a C API logsoftmax descriptor @p data.
        desc(dnnl_logsoftmax_desc_t data) {
            error::wrap_c_api(
                    dnnl_logsoftmax_forward_desc_init(&this->data,
                            data.prop_kind, &data.data_desc, data.softmax_axis),
                    "could not create a logsoftmax forward descriptor");
        }
    };

    /// Primitive descriptor for a logsoftmax forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a logsoftmax forward
        /// propagation primitive.
        ///
        /// @param desc descriptor for a logsoftmax forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a logsoftmax forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Creates a primitive descriptor for a logsoftmax forward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    // Logsoftmax and softmax share the implementation and
                    // currently report the same primitive kind. Hence this
                    // must be softmax and not logsoftmax.
                    dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::softmax_forward::primitive_desc::op_desc()
        desc op_desc() const {
            dnnl_logsoftmax_desc_t *data;
            error::wrap_c_api(
                    dnnl_primitive_desc_query(
                            get(), dnnl::convert_to_c(query::op_d), 0, &data),
                    "could not retrive a logsoftmax op desc");
            return desc(*data);
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    logsoftmax_forward() = default;

    /// Constructs a logsoftmax forward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    logsoftmax_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Logsoftmax backward propagation primitive.
struct logsoftmax_backward : public primitive {
    /// Descriptor for a logsoftmax backward propagation primitive.
    struct desc {
        dnnl_logsoftmax_desc_t data;

        /// Default constructor. Produces an empty object.
        desc() = default;

        /// Constructs a logsoftmax descriptor for backward propagation using
        /// memory descriptors @p diff_desc and @p data_desc.
        desc(const memory::desc &diff_desc, const memory::desc &data_desc,
                int logsoftmax_axis) {
            error::wrap_c_api(
                    dnnl_logsoftmax_backward_desc_init(&data, &diff_desc.data,
                            &data_desc.data, logsoftmax_axis),
                    "could not init a backward logsoftmax descriptor");
        }

        /// Create a logsoftmax descriptor for backward propagation using
        /// primitive a C API logsoftmax descriptor @p data.
        desc(dnnl_logsoftmax_desc_t data) {
            error::wrap_c_api(dnnl_logsoftmax_backward_desc_init(&this->data,
                                      &data.diff_desc, &data.data_desc,
                                      data.softmax_axis),
                    "could not create a logsoftmax forward descriptor");
        }
    };

    /// Primitive descriptor for a logsoftmax backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a logsoftmax backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a logsoftmax backward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a logsoftmax forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const logsoftmax_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a logsoftmax backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a logsoftmax backward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a logsoftmax forward
        ///                    propagation primitive. It is used as a hint
        ///                    for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const logsoftmax_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Creates a primitive descriptor for a logsoftmax backward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    // Logsoftmax and softmax share the implementation and
                    // currently report the same primitive kind. Hence this
                    // must be softmax and not logsoftmax.
                    dnnl::primitive::kind::softmax,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::softmax_forward::primitive_desc::op_desc()
        desc op_desc() const {
            dnnl_logsoftmax_desc_t *data;
            error::wrap_c_api(
                    dnnl_primitive_desc_query(
                            get(), dnnl::convert_to_c(query::op_d), 0, &data),
                    "could not retrive a logsoftmax op desc");
            return desc(*data);
        }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    logsoftmax_backward() = default;

    /// Constructs a logsoftmax backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    logsoftmax_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_batch_normalization Batch normalization
/// A primitive to perform batch normalization.
///
/// Both forward and backward passes support in-place operation; that is, src
/// and dst can point to the same memory for forward propagation, and diff_dst
/// and diff_src can point to the same memory for backward propagation.
///
/// The layer normalization primitives computations can be controlled by
/// specifying different dnnl::normalization_flags values. For example, batch
/// normalization can compute the mean and variance on its own or take them as
/// inputs.  It can either perform scaling and shifting using gamma and beta
/// parameters or not. Optionally, it can also perform a fused ReLU, which in
/// case of training would also require a workspace.
///
/// @sa @ref dev_guide_batch_normalization in developer guide
/// @sa @ref c_api_batch_normalization in @ref c_api
/// @{

/// Batch normalization forward propagation primitive.
struct batch_normalization_forward : public primitive {
    /// Descriptor for a batch normalization forward propagation primitive.
    struct desc {
        dnnl_batch_normalization_desc_t data;

        /// Constructs a batch normalization descriptor for forward
        /// propagation using @p prop_kind (acceptable values are
        /// #dnnl::forward_training and #dnnl::forward_inference), memory
        /// descriptor @p data_desc, normalization parameter @p epsilon, and
        /// flags set using bit @p flags.
        desc(prop_kind prop_kind, const memory::desc &src_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(
                    dnnl_batch_normalization_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind), &src_desc.data,
                            epsilon, convert_to_c(flags)),
                    "could not create a batch normalization forward "
                    "descriptor");
        }
    };

    /// Primitive descriptor for a batch normalization forward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a batch normalization forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a batch normalization forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a batch normalization forward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Initializes a primitive descriptor for a batch normalization forward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::batch_normalization,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// Queries mean memory descriptor.
        memory::desc mean_desc() const { return stat_desc(mean); }

        /// Queries variance memory descriptor.
        memory::desc variance_desc() const { return stat_desc(var); }

    private:
        enum {
            mean = 1,
            var = 2,
        };
        memory::desc stat_desc(int kind) const {
            dnnl_batch_normalization_desc_t *p;
            error::wrap_c_api(
                    dnnl_primitive_desc_query(get(),
                            dnnl::convert_to_c(query::batch_normalization_d), 0,
                            &p),
                    "could not get a batch normalization descriptor");
            return query_md(p->flags & dnnl_use_global_stats ? query::src_md
                                                             : query::dst_md,
                    kind);
        }
    };

    /// Default constructor. Produces an empty object.
    batch_normalization_forward() = default;

    /// Constructs a batch normalization forward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    batch_normalization_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Batch normalization backward propagation primitive.
struct batch_normalization_backward : public primitive {
    /// Descriptor for a batch normalization backward propagation primitive.
    struct desc {
        dnnl_batch_normalization_desc_t data;

        /// Constructs a batch normalization descriptor for backward
        /// propagation with respect to data and scale-shift parameters using
        /// memory descriptors @p data_desc and @p diff_data_desc,
        /// normalization parameter @p epsilon, and flags set using bit @p
        /// flags.
        desc(prop_kind prop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(
                    dnnl_batch_normalization_backward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind), &diff_data_desc.data,
                            &data_desc.data, epsilon, convert_to_c(flags)),
                    "could not create a batch normalization backward "
                    "descriptor");
        }
    };

    /// Primitive descriptor for a batch normalization backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a batch normalization backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a batch normalization backward
        ///             propagation primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a batch normalization
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const batch_normalization_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a batch normalization backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a batch normalization backward
        ///             propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a batch normalization
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const batch_normalization_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Initializes a primitive descriptor for a batch normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::batch_normalization,
                    dnnl::prop_kind::backward, dnnl::prop_kind::backward_data) {
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::mean_desc()
        memory::desc mean_desc() const { return query_md(query::src_md, 1); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::variance_desc()
        memory::desc variance_desc() const {
            return query_md(query::src_md, 2);
        }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    batch_normalization_backward() = default;

    /// Constructs a batch normalization backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    batch_normalization_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_layer_normalization Layer normalization
///
/// Layer normalization is a normalization over the last logical axis of a
/// data tensor.
///
/// Both forward and backward propagation primitives support in-place
/// operation; that is, src and dst memory objects may be the same or point to
/// the same underlying memory for forward propagation, and diff_dst and
/// diff_src point to the same memory for backward propagation.
///
/// The layer normalization primitives computations can be controlled by
/// specifying different dnnl::normalization_flags values. For example,
/// layer normalization forward propagation can be configured to either
/// compute the mean and variance or take them as arguments. It can either
/// perform scaling and shifting using gamma and beta parameters or not.
/// Optionally, it can also perform a fused ReLU, which in case of training
/// would also require a workspace.
///
/// @sa @ref dev_guide_layer_normalization in developer guide @sa @ref
/// c_api_layer_normalization in @ref c_api @{

/// Layer normalization forward propagation primitive.
struct layer_normalization_forward : public primitive {
    /// Descriptor for a layer normalization forward propagation primitive.
    struct desc {
        dnnl_layer_normalization_desc_t data;

        /// Constructs a layer normalization descriptor for forward
        /// propagation using @p prop_kind (acceptable values are
        /// #dnnl::forward_training and #dnnl::forward_inference), data memory
        /// descriptor @p data_desc, statistics memory descriptor @p
        /// stat_desc, normalization parameter @p epsilon, and flags set
        /// using bit @p flags.
        desc(prop_kind prop_kind, const memory::desc &data_desc,
                const memory::desc &stat_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(
                    dnnl_layer_normalization_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind), &data_desc.data,
                            &stat_desc.data, epsilon, convert_to_c(flags)),
                    "could not create a layer normalization forward "
                    "descriptor");
        }

        /// Constructs a layer normalization descriptor for forward
        /// propagation using @p prop_kind (acceptable values are
        /// #dnnl::forward_training and #dnnl::forward_inference), data memory
        /// descriptor @p data_desc, normalization parameter @p epsilon, and
        /// flags set using bit @p flags.
        desc(prop_kind prop_kind, const memory::desc &data_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(
                    dnnl_layer_normalization_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind), &data_desc.data,
                            nullptr, epsilon, convert_to_c(flags)),
                    "could not create a layer normalization forward "
                    "descriptor");
        }
    };

    /// Primitive descriptor for a layer normalization forward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a layer normalization forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a layer normalization forward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Initializes a primitive descriptor for layer normalization forward
        /// propagation from a C primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::layer_normalization,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const { return base::workspace_desc(); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::mean_desc()
        memory::desc mean_desc() const { return stat_desc(mean); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::variance_desc()
        memory::desc variance_desc() const { return stat_desc(var); }

    private:
        enum {
            mean = 1,
            var = 2,
        };
        memory::desc stat_desc(int kind) const {
            dnnl_layer_normalization_desc_t *p;
            error::wrap_c_api(
                    dnnl_primitive_desc_query(get(),
                            dnnl::convert_to_c(query::layer_normalization_d), 0,
                            &p),
                    "could not get a layer-normalization descriptor");
            return query_md(p->flags & dnnl_use_global_stats ? query::src_md
                                                             : query::dst_md,
                    kind);
        }
    };

    /// Default constructor. Produces an empty object.
    layer_normalization_forward() = default;

    /// Constructs a layer normalization forward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    layer_normalization_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Layer normalization backward propagation primitive.
struct layer_normalization_backward : public primitive {
    /// Descriptor for a layer normalization backward propagation primitive.
    struct desc {
        dnnl_layer_normalization_desc_t data;

        /// Constructs a layer normalization descriptor for backward
        /// propagation with respect to data and scale-shift parameters using
        /// memory descriptors @p diff_data_desc, @p data_desc and @p
        /// stat_desc, normalization parameter @p epsilon, and p flags set
        /// using bit @p flags.
        desc(prop_kind prop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, const memory::desc &stat_desc,
                float epsilon, normalization_flags flags) {
            error::wrap_c_api(
                    dnnl_layer_normalization_backward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind), &diff_data_desc.data,
                            &data_desc.data, &stat_desc.data, epsilon,
                            convert_to_c(flags)),
                    "could not create a layer normalization backward "
                    "descriptor");
        }

        /// Constructs a layer normalization descriptor for backward
        /// propagation with respect to data and scale-shift parameters using
        /// memory descriptors @p diff_data_desc and @p data_desc,
        /// normalization parameter @p epsilon, and flags set using bit
        /// @p flags.
        desc(prop_kind prop_kind, const memory::desc &diff_data_desc,
                const memory::desc &data_desc, float epsilon,
                normalization_flags flags) {
            error::wrap_c_api(dnnl_layer_normalization_backward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      &diff_data_desc.data, &data_desc.data,
                                      nullptr, epsilon, convert_to_c(flags)),
                    "could not create a layer normalization backward "
                    "descriptor");
        }
    };

    /// Primitive descriptor for a layer normalization backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a layer normalization backward
        ///             propagation primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a layer normalization backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a layer normalization backward
        ///             propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a layer normalization
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const layer_normalization_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Initializes a primitive descriptor for a layer normalization
        /// backward propagation primitive from a C API primitive descriptor
        /// @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd,
                    dnnl::primitive::kind::layer_normalization,
                    dnnl::prop_kind::backward, dnnl::prop_kind::backward_data) {
        }

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::mean_desc()
        memory::desc mean_desc() const { return query_md(query::src_md, 1); }

        /// @copydoc dnnl::batch_normalization_forward::primitive_desc::variance_desc()
        memory::desc variance_desc() const {
            return query_md(query::src_md, 2);
        }

        /// @copydoc dnnl::primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const { return base::workspace_desc(); }
    };

    /// Default constructor. Produces an empty object.
    layer_normalization_backward() = default;

    /// Constructs a layer normalization backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    layer_normalization_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_inner_product Inner Product
/// A primitive to compute an inner product.
///
/// @sa @ref dev_guide_inner_product in developer guide
/// @sa @ref c_api_inner_product in @ref c_api
/// @{

/// Inner product forward propagation primitive.
struct inner_product_forward : public primitive {
    /// Descriptor for an inner product forward propagation primitive.
    struct desc {
        dnnl_inner_product_desc_t data;

        /// Constructs a descriptor for an inner product forward propagation
        /// primitive with bias.
        ///
        /// @param prop_kind Propagation kind (either
        ///                  #dnnl::prop_kind::forward_training or
        ///                  #dnnl::prop_kind::forward_inference).
        /// @param src_desc Memory descriptor for src.
        /// @param weights_desc Memory descriptor for diff weights.
        /// @param bias_desc Memory descriptor for diff bias.
        /// @param dst_desc Memory descriptor for diff dst.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &bias_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(dnnl_inner_product_forward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      &src_desc.data, &weights_desc.data,
                                      &bias_desc.data, &dst_desc.data),
                    "could not create an inner product forward descriptor");
        }

        /// Constructs a descriptor for an inner product forward propagation
        /// primitive without bias.
        ///
        /// @param prop_kind Propagation kind (either
        ///                  #dnnl::prop_kind::forward_training or
        ///                  #dnnl::prop_kind::forward_inference).
        /// @param src_desc Memory descriptor for src.
        /// @param weights_desc Memory descriptor for diff weights.
        /// @param dst_desc Memory descriptor for diff dst.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, const memory::desc &src_desc,
                const memory::desc &weights_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    dnnl_inner_product_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind), &src_desc.data,
                            &weights_desc.data, nullptr, &dst_desc.data),
                    "could not create an inner product forward descriptor");
        }
    };

    /// Primitive descriptor for an inner product forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for an inner product forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an inner product forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for an inner product forward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Initializes a primitive descriptor for an inner product forward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::inner_product,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()
        memory::desc bias_desc() const { return base::weights_desc(1); }
    };

    /// Default constructor. Produces an empty object.
    inner_product_forward() = default;

    /// Constructs an inner product forward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    inner_product_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Inner product backward propagation primitive.
struct inner_product_backward_data : public primitive {
    /// Descriptor for an inner product backward propagation primitive.
    struct desc {
        dnnl_inner_product_desc_t data;

        /// Constructs a descriptor for an inner product backward propagation
        /// primitive.
        ///
        /// @param diff_src_desc Memory descriptor for diff src.
        /// @param weights_desc Memory descriptor for weights.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(const memory::desc &diff_src_desc,
                const memory::desc &weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(dnnl_inner_product_backward_data_desc_init(&data,
                                      &diff_src_desc.data, &weights_desc.data,
                                      &diff_dst_desc.data),
                    "could not create an inner product backward data "
                    "descriptor");
        }
    };

    /// Primitive descriptor for an inner product backward propagation
    /// primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for an inner product backward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for an inner product backward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Initializes a primitive descriptor for an inner product backward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::inner_product,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const { return base::weights_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    inner_product_backward_data() = default;

    /// Constructs an inner product backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    inner_product_backward_data(const primitive_desc &pd) : primitive(pd) {}
};

/// Inner product weights update primitive.
struct inner_product_backward_weights : public primitive {
    /// Descriptor for an inner product weights update primitive.
    struct desc {
        dnnl_inner_product_desc_t data;

        /// Constructs a descriptor for an inner product descriptor weights
        /// update primitive with bias.
        ///
        /// @param src_desc Memory descriptor for src.
        /// @param diff_weights_desc Memory descriptor for diff weights.
        /// @param diff_bias_desc Memory descriptor for diff bias.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    dnnl_inner_product_backward_weights_desc_init(&data,
                            &src_desc.data, &diff_weights_desc.data,
                            &diff_bias_desc.data, &diff_dst_desc.data),
                    "could not create an inner product backward weights "
                    "descriptor");
        }

        /// Constructs a descriptor for an inner product descriptor weights
        /// update primitive without bias.
        ///
        /// @param src_desc Memory descriptor for src.
        /// @param diff_weights_desc Memory descriptor for diff weights.
        /// @param diff_dst_desc Memory descriptor for diff dst.
        ///
        /// @note
        ///     Memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(const memory::desc &src_desc,
                const memory::desc &diff_weights_desc,
                const memory::desc &diff_dst_desc) {
            error::wrap_c_api(
                    dnnl_inner_product_backward_weights_desc_init(&data,
                            &src_desc.data, &diff_weights_desc.data, nullptr,
                            &diff_dst_desc.data),
                    "could not create an inner product backward weights "
                    "descriptor");
        }
    };

    /// Primitive descriptor for an inner product weights update primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive.
        ///
        /// @param desc Descriptor for an inner product weights update
        ///             primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an inner product weights
        /// update primitive.
        ///
        /// @param desc Descriptor for an inner product weights update
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an inner product
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const inner_product_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Constructs a primitive descriptor for inner product weights
        /// update from a C API primitive descriptor @p cpd.
        primitive_desc(dnnl_primitive_desc_t cpd)
            : dnnl::primitive_desc(cpd, dnnl::primitive::kind::inner_product,
                    dnnl::prop_kind::backward_weights) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_weights_desc()
        memory::desc diff_weights_desc() const {
            return base::diff_weights_desc(0);
        }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }

        /// @copydoc dnnl::convolution_backward_weights::primitive_desc::diff_bias_desc()
        memory::desc diff_bias_desc() const {
            return base::diff_weights_desc(1);
        }
    };

    /// Default constructor. Produces an empty object.
    inner_product_backward_weights() = default;

    /// Constructs an inner product weights update primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    inner_product_backward_weights(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_rnn RNN
/// A primitive to compute common recurrent layer.
///
/// @sa @ref dev_guide_rnn in developer guide
/// @sa @ref c_api_rnn in @ref c_api
/// @{

/// Base class for primitive descriptors for RNN primitives.
struct rnn_primitive_desc_base : public primitive_desc {
    using primitive_desc::primitive_desc;

    /// Default constructor. Produces an empty object.
    rnn_primitive_desc_base() = default;

    /// Constructs an RNN primitive descriptor base from a C API primitive
    /// descriptor while checking that it actually describes the expected
    /// primitive by comparing propagation and primitive kinds.
    ///
    /// @param pd C API primitive descriptor.
    /// @param prop_kind Expected propagation kind.
    /// @param cell_kind Expected cell kind.
    rnn_primitive_desc_base(dnnl_primitive_desc_t pd, dnnl::prop_kind prop_kind,
            dnnl::algorithm cell_kind)
        : rnn_primitive_desc_base(pd, prop_kind, prop_kind, cell_kind) {}

    /// Queries source layer memory descriptor.
    memory::desc src_layer_desc() const { return base::src_desc(0); }

    /// Queries source iteration memory descriptor. Returns a zero_md if no
    /// src_iter was specified at op_desc creation time.
    memory::desc src_iter_desc() const { return base::src_desc(1); }

    /// Queries source recurrent cell state memory descriptor.
    memory::desc src_iter_c_desc() const { return base::src_desc(2); }

    /// Queries weights layer memory descriptor.
    memory::desc weights_layer_desc() const { return base::weights_desc(0); }

    /// Queries weights iteration memory descriptor.
    memory::desc weights_iter_desc() const { return base::weights_desc(1); }

    /// Queries bias memory descriptor. Returns a zero_md if no bias was
    /// specified at op_desc creation time.
    memory::desc bias_desc() const { return base::weights_desc(2); }

    /// Queries destination layer memory descriptor.
    memory::desc dst_layer_desc() const { return base::dst_desc(0); }

    /// Queries destination iteration memory descriptor. Returns a zero_md if
    /// no dst_iter was specified at op_desc creation time.
    memory::desc dst_iter_desc() const { return base::dst_desc(1); }

    /// Queries destination recurrent cell state memory descriptor.
    memory::desc dst_iter_c_desc() const { return base::dst_desc(2); }

    /// Queries diff source layer memory descriptor.
    memory::desc diff_src_layer_desc() const { return base::diff_src_desc(0); }

    /// Queries diff source iteration memory descriptor. Returns a zero_md if
    /// no diff_src_iter was specified at op_desc creation time.
    memory::desc diff_src_iter_desc() const { return base::diff_src_desc(1); }

    /// Queries diff source recurrent cell state memory descriptor.
    memory::desc diff_src_iter_c_desc() const { return base::diff_src_desc(2); }

    /// Queries diff weights layer memory descriptor.
    memory::desc diff_weights_layer_desc() const {
        return base::diff_weights_desc(0);
    }

    /// Queries diff weights iteration memory descriptor.
    memory::desc diff_weights_iter_desc() const {
        return base::diff_weights_desc(1);
    }

    /// Queries diff bias memory descriptor.
    memory::desc diff_bias_desc() const { return base::diff_weights_desc(2); }

    /// Queries diff destination layer memory descriptor.
    memory::desc diff_dst_layer_desc() const { return base::diff_dst_desc(0); }

    /// Queries diff destination iteration memory descriptor. Returns a
    /// zero_md if no diff_dst_iter was specified at op_desc creation time.
    memory::desc diff_dst_iter_desc() const { return base::diff_dst_desc(1); }

    /// Queries diff destination recurrent cell state memory descriptor.
    memory::desc diff_dst_iter_c_desc() const { return base::diff_dst_desc(2); }

protected:
    using rnn_base = rnn_primitive_desc_base;

    // (Deliberately not using doxygen comments)
    //
    // Constructs an RNN primitive descriptor base from a C API primitive
    // descriptor while checking that it actually describes the expected
    // primitive by comparing propagation and primitive kinds. Caller can
    // pass two options propagation kinds. This is typically used to check
    // that propagation kind is inference or training forward propagation.
    //
    // @param pd C API primitive descriptor.
    // @param prop_kind1 Expected propagation kind.
    // @param prop_kind2 Expected propagation kind.
    // @param cell_kind Expected cell kind.
    rnn_primitive_desc_base(dnnl_primitive_desc_t pd,
            dnnl::prop_kind prop_kind1, dnnl::prop_kind prop_kind2,
            dnnl::algorithm cell_kind) {
        dnnl_rnn_desc_t *rnn_d;
        dnnl_status_t rc;
        rc = dnnl_primitive_desc_query(pd, dnnl_query_rnn_d, 0, &rnn_d);
        error::wrap_c_api(
                rc, "could not retrieve rnn_desc from a primitive descriptor");

        dnnl_prop_kind_t c_prop_kind1 = convert_to_c(prop_kind1);
        dnnl_prop_kind_t c_prop_kind2 = convert_to_c(prop_kind2);
        dnnl_alg_kind_t c_cell_kind = convert_to_c(cell_kind);

        bool ok = rnn_d->primitive_kind == dnnl_rnn
                && (rnn_d->prop_kind == c_prop_kind1
                        || rnn_d->prop_kind == c_prop_kind2)
                && rnn_d->cell_kind == c_cell_kind;

        if (!ok)
            DNNL_THROW_ERROR(dnnl_invalid_arguments, "rnn descriptor mismatch");

        reset_with_clone(pd);
    }
};

/// Vanilla RNN forward propagation primitive.
struct vanilla_rnn_forward : public primitive {
    /// Descriptor for a vanilla RNN forward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for a vanilla RNN forward propagation
        /// primitive using @p prop_kind, @p activation, @p direction, and
        /// memory descriptors.
        ///
        /// If @p activation is #eltwise_relu, @p alpha represents the
        /// negative slope. The @p beta and @p flags are currently ignored.
        ///
        /// The @p src_iter_desc, @p bias_desc, and @p dst_iter_desc are
        /// allowed to point to a zero memory descriptor, which would indicate
        /// that the primitive should not use them.
        ///
        /// @note
        ///     If @p prop_kind equals #dnnl::forward_training, you must
        ///     query for workspace memory descriptor before creating the
        ///     primitive.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc can be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        desc(prop_kind prop_kind, algorithm activation, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                rnn_flags flags = rnn_flags::undef, float alpha = 0.0f,
                float beta = 0.0f) {
            error::wrap_c_api(
                    dnnl_vanilla_rnn_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            dnnl::convert_to_c(activation),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            dnnl::convert_to_c(flags), alpha, beta),
                    "could not create an RNN forward descriptor");
        }
    };

    /// Primitive descriptor for a vanilla RNN forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a vanilla RNN forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a vanilla RNN forward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN forward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a vanilla RNN forward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a RNN forward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_rnn) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    vanilla_rnn_forward() = default;

    /// Constructs a vanilla RNN forward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    vanilla_rnn_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Vanilla RNN backward propagation primitive.
struct vanilla_rnn_backward : public primitive {
    /// Vanilla RNN descriptor backward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for a vanilla RNN backward propagation
        /// primitive using @p prop_kind, @p activation, @p direction, and
        /// memory descriptors.
        ///
        /// If @p activation is #eltwise_relu, @p alpha represents the
        /// negative slope. The @p beta and @p flags are currently ignored.
        ///
        /// The @p src_iter_desc (simultaneously with @p diff_src_iter_desc),
        /// @p bias_desc (simultaneously with @p diff_bias_desc), and @p
        /// dst_iter_desc (simultaneously with @p diff_src_iter_desc) are
        /// allowed point to a zero memory descriptor, which would indicate
        /// that the primitive should not use them and consider them to be
        /// zero values.
        ///
        /// @note
        ///     All memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, algorithm activation, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                rnn_flags flags = rnn_flags::undef, float alpha = 0.0f,
                float beta = 0.0f) {
            error::wrap_c_api(
                    dnnl_vanilla_rnn_backward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            dnnl::convert_to_c(activation),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data, &diff_bias_desc.data,
                            &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                            dnnl::convert_to_c(flags), alpha, beta),
                    "could not create an RNN backward descriptor");
        }
    };

    /// Primitive descriptor for a RNN backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a vanilla RNN backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a vanilla RNN backward propagation
        ///             primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a vanilla RNN
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const vanilla_rnn_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a vanilla RNN backward
        /// propagation primitive.
        ///
        /// @param desc Descriptor for a vanilla RNN backward propagation
        ///             primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a vanilla RNN
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const vanilla_rnn_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&desc.data, &attr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a RNN backward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_rnn) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    vanilla_rnn_backward() = default;

    /// Constructs a vanilla RNN backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    vanilla_rnn_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// LSTM forward propagation primitive.
struct lstm_forward : public primitive {
    /// Descriptor for an LSTM forward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for an LSTM forward propagation primitive
        /// using @p prop_kind, @p direction, and memory descriptors.
        ///
        /// The @p flags parameter is currently ignored.
        ///
        /// The @p src_iter_desc, @p src_iter_c_desc, @p bias_desc, @p
        /// dst_iter_desc and @p dst_iter_c_desc are allowed to point
        /// to a zero memory descriptor, which would indicate that the
        /// LSTM forward propagation primitive should not use them.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc can be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @note
        ///     If @p prop_kind equals #dnnl::forward_training, you must query
        ///     for workspace memory descriptor before creating the primitive.
        ///
        desc(prop_kind prop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lstm_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &src_iter_c_desc.data,
                            &weights_layer_desc.data, &weights_iter_desc.data,
                            &bias_desc.data, &dst_layer_desc.data,
                            &dst_iter_desc.data, &dst_iter_c_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create an LSTM forward descriptor");
        }
    };

    /// Primitive descriptor for an LSTM forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for an LSTM forward propagation primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for an LSTM forward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM forward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_lstm) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_c_desc() const {
            return rnn_base::src_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc dst_iter_c_desc() const {
            return rnn_base::dst_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lstm_forward() = default;

    /// Constructs an LSTM forward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    lstm_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// LSTM backward propagation primitive.
struct lstm_backward : public primitive {
    /// Descriptor for an LSTM backward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs an LSTM descriptor for backward propagation using @p
        /// prop_kind, @p direction, and memory descriptors.
        ///
        /// The @p flags parameter is currently ignored.
        ///
        /// The @p src_iter_desc (simultaneously with @p diff_src_iter_desc),
        /// @p src_iter_c_desc (simultaneously with @p diff_src_iter_c_desc),
        /// @p bias_desc (simultaneously with @p diff_bias_desc), @p
        /// dst_iter_desc (simultaneously with @p diff_src_iter_desc) and @p
        /// dst_iter_c_desc (simultaneously with @p diff_src_iter_c_desc) are
        /// allowed point to a zero memory descriptor, which would indicate
        /// that the LSTM primitive should not use them and consider them to
        /// be zero values.
        ///
        /// @note
        ///     All memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        desc(prop_kind prop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &src_iter_c_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &dst_iter_c_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_src_iter_c_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                const memory::desc &diff_dst_iter_c_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lstm_backward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &src_iter_c_desc.data,
                            &weights_layer_desc.data, &weights_iter_desc.data,
                            &bias_desc.data, &dst_layer_desc.data,
                            &dst_iter_desc.data, &dst_iter_c_desc.data,
                            &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                            &diff_src_iter_c_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data, &diff_bias_desc.data,
                            &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                            &diff_dst_iter_c_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create an LSTM backward descriptor");
        }
    };

    /// Primitive descriptor for LSTM backward propagation.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LSTM backward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for LSTM backward propagation primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LSTM
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const lstm_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM backward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for an LSTM backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for an LSTM
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const lstm_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&desc.data, &attr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LSTM backward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_lstm) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_c_desc() const {
            return rnn_base::src_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc dst_iter_c_desc() const {
            return rnn_base::dst_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_c_desc()
        memory::desc diff_src_iter_c_desc() const {
            return rnn_base::diff_src_iter_c_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_c_desc()
        memory::desc diff_dst_iter_c_desc() const {
            return rnn_base::diff_dst_iter_c_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lstm_backward() = default;

    /// Constructs an LSTM backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    lstm_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// GRU forward propagation primitive.
struct gru_forward : public primitive {
    /// Descriptor for a GRU forward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for a GRU forward propagation primitive
        /// using @p prop_kind, @p direction, and memory descriptors.
        ///
        /// The @p flags parameter is currently ignored.
        ///
        /// The @p src_iter_desc, @p bias_desc, and @p dst_iter_desc are allowed
        /// to point to a zero memory descriptor, which would indicate that
        /// the GRU primitive should not use them and will default to zero
        /// values.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc can be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @note
        ///     If @p prop_kind equals #dnnl::forward_training, you must query
        ///     a workspace memory descriptor before creating the primitive.
        ///
        desc(prop_kind prop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_gru_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create a GRU forward descriptor");
        }
    };

    /// Primitive descriptor GRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for a GRU forward propagation primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for a GRU forward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a GRU forward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::vanilla_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    gru_forward() = default;

    /// Constructs a GRU forward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    gru_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// GRU backward propagation primitive.
struct gru_backward : public primitive {
    /// Descriptor for a GRU backward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for a GRU backward propagation primitive
        /// using @p prop_kind, @p direction, and memory descriptors.
        ///
        /// The @p flags parameter is ignored.
        ///
        /// The @p src_iter_desc (simultaneously with @p diff_src_iter_desc),
        /// @p bias_desc (simultaneously with @p diff_bias_desc), and @p
        /// dst_iter_desc (simultaneously with @p diff_src_iter_desc) are
        /// allowed point to a zero memory descriptor, which would indicate
        /// that the GRU primitive should not use them and consider them to be
        /// zero values.
        ///
        /// @note
        ///     All memory descriptors are allowed to be initialized with
        ///     #dnnl::memory::format_tag::any value of @p format_tag.
        desc(prop_kind prop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_gru_backward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data, &diff_bias_desc.data,
                            &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create an GRU backward descriptor");
        }
    };

    /// Primitive descriptor for a GRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a GRU backward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for a GRU backward propagation primitive.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a GRU
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                const gru_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a GRU backward propagation
        /// primitive.
        ///
        /// @param desc Descriptor for a GRU backward propagation primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param hint_fwd_pd Primitive descriptor for a GRU
        ///                    forward propagation primitive. It is used as a
        ///                    hint for deciding which memory format to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const gru_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&desc.data, &attr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for a GRU backward
        /// propagation primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::backward,
                    dnnl::algorithm::vanilla_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    gru_backward() = default;

    /// Constructs a GRU backward propagation primitive from a
    /// primitive descriptor @p pd of a corresponding type.
    gru_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// LBR GRU forward propagation primitive.
struct lbr_gru_forward : public primitive {
    /// Descriptor for an LBR GRU forward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs an LBR GRU descriptor for forward propagation using @p
        /// prop_kind, @p direction, and memory descriptors.
        ///
        /// The @p flags parameter is currently ignored.
        ///
        /// The @p src_iter_desc, @p bias_desc, and @p dst_iter_desc are allowed
        /// to point to a zero memory descriptor, which would indicate that
        /// the LBR GRU primitive should not use them and will default to zero
        /// values.
        ///
        /// @note
        ///     All memory descriptors except @p src_iter_desc can be
        ///     initialized with an #dnnl::memory::format_tag::any value of @p
        ///     format_tag.
        ///
        /// @note
        ///     If @p prop_kind equals #dnnl::forward_training, you must query
        ///     a workspace memory descriptor before creating the primitive.
        ///
        desc(prop_kind prop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lbr_gru_forward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create an descriptor for LBR GRU forward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an LBR GRU forward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LBR GRU forward propagation
        /// primitive from a corresponding operation descriptor @p desc using
        /// engine @p engine. The @p allow_empty flag signifies whether
        /// construction is allowed to fail, in which case an empty primitive
        /// would be produced.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LBR GRU forward propagation
        /// primitive from a corresponding operation descriptor @p desc using
        /// engine @p engine, and primitive attributes @p attr. The @p
        /// allow_empty flag signifies whether construction is allowed to
        /// fail, in which case an empty primitive would be produced.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : rnn_primitive_desc_base(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an LBR GRU forward propagation
        /// primitive from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(pd, dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference,
                    dnnl::algorithm::lbr_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lbr_gru_forward() = default;

    /// Constructs a LBR GRU forward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    lbr_gru_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// LBR GRU backward propagation primitive.
struct lbr_gru_backward : public primitive {
    /// Descriptor for a LBR GRU backward propagation primitive.
    struct desc {
        dnnl_rnn_desc_t data;

        /// Constructs a descriptor for LBR GRU backward propagation primitive
        /// using @p prop_kind, @p direction, and memory descriptors.
        ///
        /// The @p flags parameter is currently ignored.
        ///
        /// The @p src_iter_desc (simultaneously with @p diff_src_iter_desc),
        /// @p bias_desc (simultaneously with @p diff_bias_desc), and @p
        /// dst_iter_desc (simultaneously with @p diff_src_iter_desc) are
        /// allowed point to a zero memory descriptor, which would indicate
        /// that the LBR GRU primitive should not use them and consider them
        /// to be zero values.
        ///
        /// @note All memory descriptors are allowed to be initialized with
        ///       #dnnl::memory::format_tag::any value of @p format_tag.
        ///
        desc(prop_kind prop_kind, rnn_direction direction,
                const memory::desc &src_layer_desc,
                const memory::desc &src_iter_desc,
                const memory::desc &weights_layer_desc,
                const memory::desc &weights_iter_desc,
                const memory::desc &bias_desc,
                const memory::desc &dst_layer_desc,
                const memory::desc &dst_iter_desc,
                const memory::desc &diff_src_layer_desc,
                const memory::desc &diff_src_iter_desc,
                const memory::desc &diff_weights_layer_desc,
                const memory::desc &diff_weights_iter_desc,
                const memory::desc &diff_bias_desc,
                const memory::desc &diff_dst_layer_desc,
                const memory::desc &diff_dst_iter_desc,
                rnn_flags flags = rnn_flags::undef) {
            error::wrap_c_api(
                    dnnl_lbr_gru_backward_desc_init(&data,
                            dnnl::convert_to_c(prop_kind),
                            dnnl::convert_to_c(direction), &src_layer_desc.data,
                            &src_iter_desc.data, &weights_layer_desc.data,
                            &weights_iter_desc.data, &bias_desc.data,
                            &dst_layer_desc.data, &dst_iter_desc.data,
                            &diff_src_layer_desc.data, &diff_src_iter_desc.data,
                            &diff_weights_layer_desc.data,
                            &diff_weights_iter_desc.data, &diff_bias_desc.data,
                            &diff_dst_layer_desc.data, &diff_dst_iter_desc.data,
                            dnnl::convert_to_c(flags)),
                    "could not create an descriptor for LBR GRU backward "
                    "propagation primitive");
        }
    };

    /// Primitive descriptor for an LBR GRU backward propagation primitive.
    struct primitive_desc : public rnn_primitive_desc_base {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an LBR GRU backward
        /// propagation primitive from a corresponding operation descriptor
        /// @p desc using engine @p engine, and an LBR GRU forward propagation
        /// primitive descriptor hint @p hint_fwd_pd. The @p allow_empty flag
        /// signifies whether construction is allowed to fail, in which case an
        /// empty primitive would be produced.
        primitive_desc(const desc &desc, const engine &engine,
                const lbr_gru_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&desc.data, nullptr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for an LBR GRU backward
        /// propagation primitive from a corresponding operation descriptor
        /// @p desc using engine @p engine, an LBR GRU forward propagation
        /// primitive descriptor hint @p hint_fwd_pd, and primitive attributes
        /// @p attr. The @p allow_empty flag signifies whether construction is
        /// allowed to fail, in which case an empty primitive would be produced.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine,
                const lbr_gru_forward::primitive_desc &hint_fwd_pd,
                bool allow_empty = false)
            : rnn_primitive_desc_base(&desc.data, &attr, engine,
                    hint_fwd_pd.get(), allow_empty) {}

        /// Constructs a primitive descriptor for LBR GRU backward
        /// propagation from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : rnn_primitive_desc_base(
                    pd, dnnl::prop_kind::backward, dnnl::algorithm::lbr_gru) {}

        /// @copydoc dnnl::rnn_primitive_desc_base::src_layer_desc()
        memory::desc src_layer_desc() const {
            return rnn_base::src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::src_iter_desc()
        memory::desc src_iter_desc() const { return rnn_base::src_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_layer_desc()
        memory::desc weights_layer_desc() const {
            return rnn_base::weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::weights_iter_desc()
        memory::desc weights_iter_desc() const {
            return rnn_base::weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::bias_desc()
        memory::desc bias_desc() const { return rnn_base::bias_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_layer_desc()
        memory::desc dst_layer_desc() const {
            return rnn_base::dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::dst_iter_desc()
        memory::desc dst_iter_desc() const { return rnn_base::dst_iter_desc(); }

        /// @copydoc dnnl::rnn_primitive_desc_base::workspace_desc()
        memory::desc workspace_desc() const {
            return rnn_base::workspace_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_layer_desc()
        memory::desc diff_src_layer_desc() const {
            return rnn_base::diff_src_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_src_iter_desc()
        memory::desc diff_src_iter_desc() const {
            return rnn_base::diff_src_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_layer_desc()
        memory::desc diff_weights_layer_desc() const {
            return rnn_base::diff_weights_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_weights_iter_desc()
        memory::desc diff_weights_iter_desc() const {
            return rnn_base::diff_weights_iter_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_bias_desc()
        memory::desc diff_bias_desc() const {
            return rnn_base::diff_bias_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_layer_desc()
        memory::desc diff_dst_layer_desc() const {
            return rnn_base::diff_dst_layer_desc();
        }

        /// @copydoc dnnl::rnn_primitive_desc_base::diff_dst_iter_desc()
        memory::desc diff_dst_iter_desc() const {
            return rnn_base::diff_dst_iter_desc();
        }
    };

    /// Default constructor. Produces an empty object.
    lbr_gru_backward() = default;

    /// Constructs a LBR GRU backward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    lbr_gru_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_shuffle Shuffle
/// A primitive to shuffle data along the axis.
///
/// @sa @ref dev_guide_shuffle in developer guide
/// @sa @ref c_api_shuffle in @ref c_api
/// @{

/// Shuffle forward propagation primitive.
struct shuffle_forward : public primitive {
    /// Descriptor for a shuffle forward propagation primitive.
    struct desc {
        dnnl_shuffle_desc_t data;

        /// Constructs a descriptor shuffle forward propagation primitive
        /// using @p prop_kind, memory descriptor @p data_desc, @p axis, and
        /// @p group_size.
        desc(prop_kind prop_kind, const memory::desc &data_desc, int axis,
                int group_size) {
            error::wrap_c_api(dnnl_shuffle_forward_desc_init(&data,
                                      dnnl::convert_to_c(prop_kind),
                                      &data_desc.data, axis, group_size),
                    "could not create a shuffle forward descriptor");
        }
    };

    /// Primitive descriptor for a shuffle forward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a shuffle forward propagation
        /// primitive from a corresponding operation descriptor @p desc using
        /// engine @p engine, and primitive attributes @p attr. The @p
        /// allow_empty flag signifies whether construction is allowed to
        /// fail, in which case an empty primitive would be produced.
        primitive_desc(const desc &desc, const engine &engine,
                const primitive_attr &attr = primitive_attr(),
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for shuffle forward propagation
        /// from a C API primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::shuffle,
                    dnnl::prop_kind::forward_training,
                    dnnl::prop_kind::forward_inference) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return base::src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    shuffle_forward() = default;

    /// Constructs a shuffle forward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    shuffle_forward(const primitive_desc &pd) : primitive(pd) {}
};

/// Shuffle backward propagation primitive.
struct shuffle_backward : public primitive {
    /// Descriptor for a shuffle primitive backward propagation
    /// primitive.
    struct desc {
        dnnl_shuffle_desc_t data;

        /// Constructs a primitive descriptor for a shuffle backward
        /// propagation primitive using a memory descriptor @p diff_data_desc,
        /// @p axis, and @p group_size.
        desc(const memory::desc &diff_data_desc, int axis, int group_size) {
            error::wrap_c_api(dnnl_shuffle_backward_desc_init(&data,
                                      &diff_data_desc.data, axis, group_size),
                    "could not create a shuffle backward descriptor");
        }
    };

    /// Primitive descriptor for a shuffle backward propagation primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a shuffle backward propagation
        /// primitive from a corresponding operation descriptor @p desc using
        /// engine @p engine, shuffle forward propagation primitive descriptor
        /// hint @p hint_fwd_pd, and primitive attributes @p attr. The @p
        /// allow_empty flag signifies whether construction is allowed to
        /// fail, in which case an empty primitive would be produced.
        primitive_desc(const desc &desc, const engine &engine,
                const shuffle_forward::primitive_desc &hint_fwd_pd,
                const primitive_attr &attr = primitive_attr(),
                bool allow_empty = false)
            : dnnl::primitive_desc(&desc.data, &attr, engine, hint_fwd_pd.get(),
                    allow_empty) {}

        /// Constructs a primitive descriptor for a shuffle backward propagation
        /// primitive from a C API primitive descriptor.
        ///
        /// @param pd C API primitive descriptor.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::shuffle,
                    dnnl::prop_kind::backward_data) {}

        /// @copydoc dnnl::primitive_desc_base::diff_src_desc()
        memory::desc diff_src_desc() const { return base::diff_src_desc(0); }

        /// @copydoc dnnl::primitive_desc_base::diff_dst_desc()
        memory::desc diff_dst_desc() const { return base::diff_dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    shuffle_backward() = default;

    /// Constructs a shuffle backward propagation primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    shuffle_backward(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_binary Binary
/// A primitive to perform tensor operations over two tensors.
///
/// @sa @ref dev_guide_binary in developer guide
/// @sa @ref c_api_binary in @ref c_api
/// @{

/// Elementwise binary operator primitive.
struct binary : public primitive {
    /// Descriptor for an elementwise binary operator primitive.
    struct desc {
        /// Underlying C operation descriptor.
        dnnl_binary_desc_t data;

        /// Constructs a descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param algorithm Elementwise algorithm.
        /// @param src0 Memory descriptor for source tensor #0.
        /// @param src1 Memory descriptor for source tensor #1.
        /// @param dst Memory descriptor for destination tensor.
        desc(algorithm algorithm, const memory::desc &src0,
                const memory::desc &src1, const memory::desc &dst) {
            error::wrap_c_api(
                    dnnl_binary_desc_init(&data, dnnl::convert_to_c(algorithm),
                            &src0.data, &src1.data, &dst.data),
                    "could not create a binary descriptor");
        }
    };

    /// Primitive descriptor for an elementwise binary operator primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param desc Descriptor for an elementwise binary operator primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise binary operator
        /// primitive.
        ///
        /// @param desc Descriptor for an elementwise binary operator primitive.
        /// @param engine Engine to use.
        /// @param attr Primitive attributes to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for an elementwise binary operator
        /// primitive from a C API primitive descriptor.
        ///
        /// @param pd C API primitive descriptor.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::binary) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc(int idx = 0) const { return base::src_desc(idx); }

        /// Returns the memory descriptor for source #0.
        memory::desc src0_desc() const { return base::src_desc(0); }

        /// Returns the memory descriptor for source #1.
        memory::desc src1_desc() const { return base::src_desc(1); }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return base::dst_desc(0); }
    };

    /// Default constructor. Produces an empty object.
    binary() = default;

    /// Constructs an elementwise binary operator primitive from a primitive
    /// descriptor @p pd of a corresponding type.
    binary(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @addtogroup cpp_api_matmul Matrix Multiplication
/// A primitive to perform matrix multiplication of two tensors and possibly
/// bias addition to the result. Batched mode supported with 3D tensors.
///
/// @sa @ref dev_guide_matmul in developer guide
/// @sa @ref c_api_matmul in @ref c_api
/// @{

/// Matrix multiplication (matmul) primitive.
struct matmul : public primitive {
    /// Descriptor for a matmul primitive.
    struct desc {
        dnnl_matmul_desc_t data;

        /// Constructs a descriptor for a matmul primitive.
        ///
        /// @param src_desc Memory descriptor for source (matrix A).
        /// @param weights_desc Memory descriptor for weights (matrix B).
        /// @param dst_desc Memory descriptor for destination (matrix C).
        desc(const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &dst_desc) {
            error::wrap_c_api(
                    dnnl_matmul_desc_init(&data, &src_desc.data,
                            &weights_desc.data, nullptr, &dst_desc.data),
                    "could not create a matmul descriptor");
        }

        /// Constructs a descriptor for a matmul primitive.
        ///
        /// @param src_desc Memory descriptor for source (matrix A).
        /// @param weights_desc Memory descriptor for weights (matrix B).
        /// @param dst_desc Memory descriptor for destination (matrix C).
        /// @param bias_desc Memory descriptor for bias.
        desc(const memory::desc &src_desc, const memory::desc &weights_desc,
                const memory::desc &bias_desc, const memory::desc &dst_desc) {
            error::wrap_c_api(dnnl_matmul_desc_init(&data, &src_desc.data,
                                      &weights_desc.data, &bias_desc.data,
                                      &dst_desc.data),
                    "could not create a matmul descriptor");
        }
    };

    /// Primitive descriptor for a matmul primitive.
    struct primitive_desc : public dnnl::primitive_desc {
        /// Default constructor. Produces an empty object.
        primitive_desc() = default;

        /// Constructs a primitive descriptor for a matmul primitive.
        ///
        /// @param desc Descriptor for a matmul primitive.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const engine &engine,
                bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, nullptr, engine, nullptr, allow_empty) {}

        /// Constructs a primitive descriptor for a matmul primitive.
        ///
        /// @param desc Descriptor for a matmul primitive.
        /// @param attr Primitive attributes to use.
        /// @param engine Engine to use.
        /// @param allow_empty A flag signifying whether construction is
        ///                    allowed to fail without throwing an exception.
        ///                    In this case an empty object will be produced.
        ///                    This flag is optional and defaults to false.
        primitive_desc(const desc &desc, const primitive_attr &attr,
                const engine &engine, bool allow_empty = false)
            : dnnl::primitive_desc(
                    &desc.data, &attr, engine, nullptr, allow_empty) {}

        /// Creates a primitive descriptor for a matmul primitive from a C API
        /// primitive descriptor @p pd.
        primitive_desc(dnnl_primitive_desc_t pd)
            : dnnl::primitive_desc(pd, dnnl::primitive::kind::matmul) {}

        /// @copydoc dnnl::primitive_desc_base::src_desc()
        memory::desc src_desc() const { return query_md(query::src_md, 0); }

        /// @copydoc dnnl::primitive_desc_base::weights_desc()
        memory::desc weights_desc() const {
            return query_md(query::weights_md, 0);
        }

        /// @copydoc dnnl::convolution_forward::primitive_desc::bias_desc()
        memory::desc bias_desc() const {
            return query_md(query::weights_md, 1);
        }

        /// @copydoc dnnl::primitive_desc_base::dst_desc()
        memory::desc dst_desc() const { return query_md(query::dst_md, 0); }
    };

    /// Default constructor. Produces an empty object.
    matmul() = default;

    /// Constructs a matmul primitive from a primitive descriptor @p pd of a
    /// corresponding type.
    matmul(const primitive_desc &pd) : primitive(pd) {}
};

/// @}

/// @} Primitives

/// @addtogroup cpp_api_service Service functions
/// Wrappers around @ref c_api_service in @ref c_api.
///
/// @{

/// @copydoc dnnl_version_t
using version_t = dnnl_version_t;

/// Status values returned by the library functions.
enum class status {
    /// @copydoc dnnl_success
    success = dnnl_success,
    /// @copydoc dnnl_out_of_memory
    out_of_memory = dnnl_out_of_memory,
    /// @copydoc dnnl_invalid_arguments
    invalid_arguments = dnnl_invalid_arguments,
    /// @copydoc dnnl_unimplemented
    unimplemented = dnnl_unimplemented,
    /// @copydoc dnnl_iterator_ends
    iterator_ends = dnnl_iterator_ends,
    /// @copydoc dnnl_runtime_error
    runtime_error = dnnl_runtime_error,
    /// @copydoc dnnl_not_required
    not_required = dnnl_not_required,
};

/// @copydoc dnnl_set_verbose()
inline status set_verbose(int level) {
    return static_cast<status>(dnnl_set_verbose(level));
}

/// @copydoc dnnl_version()
inline const version_t *version() {
    return dnnl_version();
}

/// @copydoc dnnl_set_jit_dump()
inline status set_jit_dump(int enable) {
    return static_cast<status>(dnnl_set_jit_dump(enable));
}

/// @copydoc dnnl_set_jit_profiling_flags()
inline status set_jit_profiling_flags(unsigned flags) {
    return static_cast<status>(dnnl_set_jit_profiling_flags(flags));
}

/// @copydoc dnnl_set_jit_profiling_jitdumpdir()
inline status set_jit_profiling_jitdumpdir(const std::string &dir) {
    return static_cast<status>(dnnl_set_jit_profiling_jitdumpdir(dir.c_str()));
}

/// @} Service functions

/// @addtogroup cpp_api_blas BLAS functions
/// Wrappers around @ref c_api_blas in @ref c_api.
///
/// @{

/// @copydoc dnnl_sgemm()
inline status sgemm(char transa, char transb, dnnl_dim_t M, dnnl_dim_t N,
        dnnl_dim_t K, float alpha, const float *A, dnnl_dim_t lda,
        const float *B, dnnl_dim_t ldb, float beta, float *C, dnnl_dim_t ldc) {
    return static_cast<status>(dnnl_sgemm(
            transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
}

/// @copydoc dnnl_gemm_u8s8s32()
inline status gemm_u8s8s32(char transa, char transb, char offsetc, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const uint8_t *A,
        dnnl_dim_t lda, uint8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co) {
    return static_cast<status>(dnnl_gemm_u8s8s32(transa, transb, offsetc, M, N,
            K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co));
}

/// @copydoc dnnl_gemm_s8s8s32()
inline status gemm_s8s8s32(char transa, char transb, char offsetc, dnnl_dim_t M,
        dnnl_dim_t N, dnnl_dim_t K, float alpha, const int8_t *A,
        dnnl_dim_t lda, int8_t ao, const int8_t *B, dnnl_dim_t ldb, int8_t bo,
        float beta, int32_t *C, dnnl_dim_t ldc, const int32_t *co) {
    return static_cast<status>(dnnl_gemm_s8s8s32(transa, transb, offsetc, M, N,
            K, alpha, A, lda, ao, B, ldb, bo, beta, C, ldc, co));
}

/// @} BLAS functions

/// @} C++ API

// implementation section

/// @cond DO_NOT_DOCUMENT_THIS
inline primitive::primitive(const_dnnl_primitive_desc_t c_pd) {
    dnnl_primitive_t result;
    error::wrap_c_api(dnnl_primitive_create(&result, c_pd),
            "could not create a primitive");
    reset(result);
}

inline primitive::primitive(const primitive_desc &pd) : primitive(pd.get()) {}

inline void primitive::execute(
        stream &stream, const std::unordered_map<int, memory> &args) const {
    std::vector<dnnl_exec_arg_t> c_args;
    c_args.reserve(args.size());
    for (const auto &a : args)
        c_args.push_back({a.first, a.second.get(true)});

    error::wrap_c_api(dnnl_primitive_execute(get(), stream.get(),
                              (int)c_args.size(), c_args.data()),
            "could not execute a primitive");
}
/// @endcond

#undef DNNL_DEFINE_BITMASK_OPS

} // namespace dnnl

#endif

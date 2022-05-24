/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef ONEAPI_DNNL_DNNL_GRAPH_HPP
#define ONEAPI_DNNL_DNNL_GRAPH_HPP

#include "oneapi/dnnl/dnnl_graph.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/// @addtogroup dnnl_graph_api oneDNN graph API
/// @{

namespace dnnl {
namespace graph {

/// @addtogroup dnnl_graph_api_utils Utilities
/// Utility types and definitions
/// @{

namespace detail {

template <typename T, dnnl_graph_status_t (*del)(T)>
class handle {
public:
    static constexpr auto default_del = del;

    /// Creates an empty wrapper for underlying C API handle
    handle() = default;
    virtual ~handle() = default;

    /// Custom constructor
    ///
    /// @param t Raw pointer to the C API handle
    /// @param weak A flag which indicates whether this wrapper
    ///     is a weak pointer
    handle(T t, bool weak = false) { reset(t, weak); }

    /// Copy constructor
    handle(const handle &) = default;
    /// Copy assign constructor
    handle &operator=(const handle &) = default;
    /// Move constructor
    handle(handle &&) = default;
    /// Move assign constructor
    handle &operator=(handle &&) = default;

    /// Resets the handle wrapper object to wrap a new C API handle
    ///
    /// @param t The raw pointer of C API handle
    /// @param weak A flag which indicates whether this wrapper is a
    ///     weak pointer
    void reset(T t, bool weak = false) {
        data_.reset(t, weak ? dummy_del : default_del);
    }

    /// Returns the underlying C API handle
    ///
    /// @returns The underlying C API handle
    T get() const { return data_.get(); }

private:
    std::shared_ptr<typename std::remove_pointer<T>::type> data_ {0};
    /// Dummy destructor
    static dnnl_graph_status_t dummy_del(T) { return dnnl_graph_success; }
};

#define DNNL_GRAPH_HANDLE_ALIAS(type) \
    using type##_handle = detail::handle<dnnl_graph_##type##_t, \
            dnnl_graph_##type##_destroy>

DNNL_GRAPH_HANDLE_ALIAS(allocator);
DNNL_GRAPH_HANDLE_ALIAS(engine);
DNNL_GRAPH_HANDLE_ALIAS(graph);
DNNL_GRAPH_HANDLE_ALIAS(op);
DNNL_GRAPH_HANDLE_ALIAS(stream);
DNNL_GRAPH_HANDLE_ALIAS(tensor);
DNNL_GRAPH_HANDLE_ALIAS(compiled_partition);
DNNL_GRAPH_HANDLE_ALIAS(partition);

#undef DNNL_GRAPH_HANDLE_ALIAS

} // namespace detail

/// dnnl graph exception class.
///
/// This class captures the status returned by a failed C API function and
/// the error message from the call site.
struct error : public std::exception {
    dnnl_graph_status_t result;
    std::string detailed_message;

    /// Constructs an instance of an exception class.
    ///
    /// @param result The error status returned by a C API function.
    /// @param message The error message.
    error(dnnl_graph_status_t result, const std::string &message)
        : result(result)
        , detailed_message(message + ": " + result2str(result)) {}

    /// Convert dnnl_graph_status_t to string.
    ///
    /// @param result The error status returned by a C API function.
    /// @return A string that describes the error status
    std::string result2str(dnnl_graph_status_t result) {
        switch (result) {
            case dnnl_graph_success: return "success";
            case dnnl_graph_out_of_memory: return "out of memory";
            case dnnl_graph_invalid_arguments: return "invalid arguments";
            case dnnl_graph_unimplemented: return "unimplemented";
            case dnnl_graph_iterator_ends: return "iterator ends";
            case dnnl_graph_runtime_error: return "runtime error";
            case dnnl_graph_not_required: return "not required";
            case dnnl_graph_invalid_graph: return "invalid graph";
            case dnnl_graph_invalid_graph_op: return "invalid op";
            case dnnl_graph_invalid_shape: return "invalid shape";
            case dnnl_graph_invalid_data_type: return "invalid data type";
            default: return "unknown error";
        }
    }

    /// Returns the explanatory string.
    ///
    /// @return A const char * that describes the error status
    const char *what() const noexcept override {
        return detailed_message.c_str();
    }

    /// Checks the return status and throws an error in case of failure.
    ///
    /// @param result The error status returned by a C API function.
    /// @param message The error message.
    static void check_succeed(
            dnnl_graph_status_t result, const std::string &message) {
        if (result != dnnl_graph_success) throw error(result, message);
    }
};

template <bool B>
using requires = typename std::enable_if<B, bool>::type;

template <typename T, requires<std::is_same<T, float>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_f32;
}

template <typename T, requires<std::is_same<T, int8_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_s8;
}

template <typename T, requires<std::is_same<T, uint8_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_u8;
}

// TODO(wuxun): now use int16 to simulate float16, need fix in the future
template <typename T, requires<std::is_same<T, int16_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_f16;
}

// TODO(wuxun): now use uint16 to simulate Bfloat16, need fix in the future
template <typename T, requires<std::is_same<T, uint16_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_bf16;
}

template <typename T,
        requires<!std::is_same<T, float>::value
                && !std::is_same<T, int16_t>::value
                && !std::is_same<T, uint16_t>::value
                && !std::is_same<T, int8_t>::value
                && !std::is_same<T, uint8_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_data_type_undef;
}

/// @} dnnl_graph_api_utils

/// @addtogroup dnnl_graph_api_status
/// Definitions of status values returned by the library functions.
///
/// @{

/// Status values returned by the library functions.
enum class status {
    /// The operation was successful
    success = dnnl_graph_success,
    /// The operation failed due to an out-of-memory condition
    out_of_memory = dnnl_graph_out_of_memory,
    /// The operation failed because of incorrect function arguments
    invalid_arguments = dnnl_graph_invalid_arguments,
    /// The operation failed because requested functionality is not implemented
    unimplemented = dnnl_graph_unimplemented,
    /// Primitive iterator passed over last primitive descriptor
    interator_ends = dnnl_graph_iterator_ends,
    /// Primitive or engine failed on execution
    runtime_error = dnnl_graph_runtime_error,
    /// Queried element is not required for given primitive
    not_required = dnnl_graph_not_required,
    /// The graph is not legitimate
    invalid_graph = dnnl_graph_invalid_graph,
    /// The operation is not legitimate according to op schema
    invalid_graph_op = dnnl_graph_invalid_graph_op,
    /// The shape cannot be inferred or compiled
    invalid_shape = dnnl_graph_invalid_shape,
    /// The data type cannot be inferred or compiled
    invalid_data_type = dnnl_graph_invalid_data_type,
};

/// @} dnnl_api_status

/// @addtogroup dnnl_graph_api_allocator Allocator
/// Definitions of allocator
///
/// Regarding the allocator call-back function, we also provide a hint/attribute
/// to integration, which indicates that which type of memory is being requested
/// (persistent/output/temp) and the expected alignment of memory allocation.
///
/// @{

/// Allocator
class allocator : public detail::allocator_handle {
public:
    using detail::allocator_handle::handle;

    /// An allocator attribute
    using attribute = dnnl_graph_allocator_attr_t;

    /// allocation lifetime enumeration
    enum class lifetime {
        persistent = dnnl_graph_allocator_persistent,
        output = dnnl_graph_allocator_output,
        temp = dnnl_graph_allocator_temp,
    };

    /// Constructs an allocator according to given function pointers
    ///
    /// @param cpu_malloc A pointer to malloc function for CPU
    /// @param cpu_free A pointer to free function for CPU
    allocator(dnnl_graph_cpu_allocate_f cpu_malloc,
            dnnl_graph_cpu_deallocate_f cpu_free) {
        dnnl_graph_allocator_t a = nullptr;
        error::check_succeed(
                dnnl_graph_allocator_create(&a, cpu_malloc, cpu_free),
                "could not create allocator for cpu");
        reset(a);
    }

    /// Default constructor
    allocator() {
        dnnl_graph_allocator_t a = nullptr;
        error::check_succeed(dnnl_graph_allocator_create(&a, nullptr, nullptr),
                "could not create allocator");
        reset(a);
    }

    static dnnl_graph_allocator_lifetime_t convert_to_c(lifetime type) {
        return static_cast<dnnl_graph_allocator_lifetime_t>(type);
    }
};

/// @} dnnl_graph_api_allocator

/// @addtogroup dnnl_graph_api_engine Engine
///
/// Engine represents a device and its context. Compiled partitions are
/// associated with engines. A compiled partition should only access the tensor
/// which is associated with the same device and context, no matter the tensor
/// is produced by a compiled partition or created directly by the user.

/// @{

/// An engine contains device #kind and a device_id or device_handle.
class engine : public detail::engine_handle {
public:
    using detail::engine_handle::handle;

    /// engine kind
    enum class kind {
        /// An unspecified engine
        any = dnnl_graph_any_engine,
        /// CPU engine
        cpu = dnnl_graph_cpu,
        /// GPU engine
        gpu = dnnl_graph_gpu,
    };

    /// Constructs an engine with specified kind and device_id
    ///
    /// @param akind The kind of engine to construct
    /// @param index Specify which device to be used
    engine(kind akind, size_t index) {
        dnnl_graph_engine_t e = nullptr;
        error::check_succeed(
                dnnl_graph_engine_create(&e, convert_to_c(akind), index),
                "could not create engine with engine kind and device index");
        reset(e);
    }

    /// Constructs an engine with specified kind and device_id
    ///
    /// @param akind Engine kind
    /// @param index Specify which device to be used
    /// @param alloc The memory allocator bound with engine
    engine(kind akind, size_t index, const allocator &alloc) {
        dnnl_graph_engine_t e = nullptr;
        error::check_succeed(dnnl_graph_engine_create_with_allocator(&e,
                                     convert_to_c(akind), index, alloc.get()),
                "could not create engine with engine kind, device index, and "
                "allocator");
        reset(e);
    }

    /// Returns concrete kind of the current engine
    ///
    ///@returns Kind of engine
    kind get_kind() const {
        dnnl_graph_engine_kind_t akind;
        error::check_succeed(dnnl_graph_engine_get_kind(get(), &akind),
                "could not get engine kind from the engine");
        return static_cast<kind>(akind);
    }

    static dnnl_graph_engine_kind_t convert_to_c(kind akind) {
        return static_cast<dnnl_graph_engine_kind_t>(akind);
    }
};

/// @} dnnl_graph_api_engine

/// @addtogroup dnnl_graph_api_stream Stream
///
/// Stream is the logical abstraction for execution units.
///
/// @{

/// A stream is created on top of oneDNN graph engine. For SYCL device, it
/// contains an opencl queue. oneDNN Graph engine may have multiple streams.
/// A compiled partition is submitted to a stream for execution.
class stream : public detail::stream_handle {
public:
    using detail::stream_handle::handle;

    /// Constructs a stream for the specified engine
    ///
    /// @param engine Engine to create stream on
    stream(const engine &engine) {
        dnnl_graph_stream_t s = nullptr;
        error::check_succeed(dnnl_graph_stream_create(&s, engine.get()),
                "could not create stream");
        reset(s);
    }

    /// Waits for all compiled partitions executing in the stream to finish.
    /// @returns The stream itself.
    stream &wait() {
        error::check_succeed(
                dnnl_graph_stream_wait(get()), "could not wait on a stream");
        return *this;
    }
};

/// @} dnnl_graph_api_stream

/// @addtogroup dnnl_graph_api_logical_tensor Logical tensor
///
/// Logical tensor describes the meta data of the input or output tensor, like
/// element data type, number of dimensions, size for each dimension (shape),
/// layout, and the total size of data.
///
/// Each logical tensor has an ID. The tensor metadata may be enriched in the
/// framework graph as it progresses toward final execution. Logical tensor is
/// not mutable. Users must create a new logical tensor with the same ID to pass
/// any new additional information to oneDNN Graph implementation.
///
/// @{

/// Logical tensor object
///
/// A logical tensor not only helps oneDNN graph implementation to build the
/// graph, but plays a critical role to exchange metadata between users and
/// oneDNN graph implementation.
class logical_tensor {
    friend class op;
    friend class tensor;
    friend class partition;
    friend class compiled_partition;

    dnnl_graph_logical_tensor_t data;

public:
    using dims_t = std::vector<dnnl_graph_dim_t>;

    /// Data Type
    enum class data_type {
        undef = dnnl_graph_data_type_undef,
        /// 16-bit/half-precision floating point.
        f16 = dnnl_graph_f16,
        /// non-standard 16-bit (bfloat16 w/ 7 bit mantissa) floating point.
        bf16 = dnnl_graph_bf16,
        /// 32-bit/single-precision floating point.
        f32 = dnnl_graph_f32,
        /// 32-bit signed integer.
        s32 = dnnl_graph_s32,
        /// 8-bit signed integer.
        s8 = dnnl_graph_s8,
        /// 8-bit unsigned integer.
        u8 = dnnl_graph_u8,
        /// boolean data type. The tensor element will be interpreted with C++
        /// bool type. Note that the size of C++ bool type is language
        /// implementation defined.
        boolean = dnnl_graph_boolean,
    };

    /// Layout type
    enum class layout_type {
        /// undefined layout type
        undef = dnnl_graph_layout_type_undef,
        /// any means that oneDNN graph implementation needs to decide the
        /// layout for the compiled partition.
        any = dnnl_graph_layout_type_any,
        /// strided means that the layout is determined by the strides field.
        strided = dnnl_graph_layout_type_strided,
        /// opaque means that the layout is a target-specific layout decided by
        /// oneDNN graph implementation.
        opaque = dnnl_graph_layout_type_opaque,
    };

    /// Tensor property
    enum class property_type {
        /// undefined tensor property
        undef = dnnl_graph_tensor_property_undef,
        /// variable means the tensor will be changed during iterations
        variable = dnnl_graph_tensor_property_variable,
        /// constant means the tensor will keep unchanged during iterations
        constant = dnnl_graph_tensor_property_constant,
    };

    /// default constructor
    /// construct an empty object
    logical_tensor() = default;

    /// Constructs a logical tensor object
    explicit logical_tensor(const dnnl_graph_logical_tensor_t &c_data)
        : data(c_data) {}

    /// Copy
    logical_tensor(const logical_tensor &other) = default;

    /// Assign
    logical_tensor &operator=(const logical_tensor &other) = default;

    /// Constructs a logical tensor object
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param ndims Number of dimension, -1 means it's unknown, 0 means scalar
    /// @param ltype Layout type
    /// @param ptype Property type
    logical_tensor(size_t tid, data_type dtype, int32_t ndims,
            layout_type ltype, property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;
        error::check_succeed(
                dnnl_graph_logical_tensor_init(&val, tid, convert_to_c(dtype),
                        ndims, convert_to_c(ltype), convert_to_c(ptype)),
                "could not create logical_tensor with property");
        data = val;
    }

    /// Delegated constructor
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param ltype Layout type
    logical_tensor(
            size_t tid, data_type dtype, layout_type ltype = layout_type::undef)
        : logical_tensor(tid, dtype, DNNL_GRAPH_UNKNOWN_NDIMS, ltype) {}

    /// Constructs a logical tensor object
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param adims Tensor dimensions, -1 means a particular axis of dims is
    ///        unknown, or the axis can be deduced by its size and other axis.
    /// @param ltype Layout type
    /// @param ptype Tensor property type
    logical_tensor(size_t tid, data_type dtype, const dims_t &adims,
            layout_type ltype, property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;
        // if dimension size equals to 0, it's a scalar
        if (adims.size() == 0)
            error::check_succeed(
                    dnnl_graph_logical_tensor_init(&val, tid,
                            convert_to_c(dtype), 0, convert_to_c(ltype),
                            convert_to_c(ptype)),
                    "could not create logical_tensor with property");
        else
            error::check_succeed(
                    dnnl_graph_logical_tensor_init_with_dims(&val, tid,
                            convert_to_c(dtype),
                            static_cast<int32_t>(adims.size()), adims.data(),
                            convert_to_c(ltype), convert_to_c(ptype)),
                    "could not create logical_tensor with dims and property");
        data = val;
    }

    /// Constructs a logical tensor object
    ///
    /// @note The layout_type for this constructor will always be strided
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param adims Tensor dimensions, -1 means a particular axis of dims is
    /// @param strides Tensor strides
    /// @param ptype Tensor property type
    logical_tensor(size_t tid, data_type dtype, const dims_t &adims,
            const dims_t &strides, property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;
        // TODO(lvtao): check the size of adims and strides.
        // They should be same.
        error::check_succeed(
                dnnl_graph_logical_tensor_init_with_strides(&val, tid,
                        convert_to_c(dtype), static_cast<int32_t>(adims.size()),
                        adims.data(), strides.data(), convert_to_c(ptype)),
                "could not create logical_tensor with strides and property");
        data = val;
    }

    /// Constructs a logical tensor object
    ///
    /// @note The layout_type for this constructor will always be opaque
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param adims Tensor dimensions, -1 means a particular axis of dims is
    /// @param lid Layout id
    /// @param ptype Tensor property type
    logical_tensor(size_t tid, data_type dtype, const dims_t &adims, size_t lid,
            property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;

        if (adims.size() == 0) {
            error::check_succeed(dnnl_graph_logical_tensor_init(&val, tid,
                                         convert_to_c(dtype), 0,
                                         convert_to_c(layout_type::opaque),
                                         convert_to_c(ptype)),
                    "could not create logical_tensor");
        } else {
            error::check_succeed(
                    dnnl_graph_logical_tensor_init_with_dims(&val, tid,
                            convert_to_c(dtype),
                            static_cast<int32_t>(adims.size()), adims.data(),
                            convert_to_c(layout_type::opaque),
                            convert_to_c(ptype)),
                    "could not create logical_tensor with dims");
        }

        val.layout.layout_id = lid;
        data = val;
    }

    /// Returns dimensions of the logical tensor
    ///
    /// @returns A the dimensions vector
    dims_t get_dims() const {
        if (data.ndims < 0) {
            error::check_succeed(dnnl_graph_invalid_arguments,
                    "cannot return dims when ndims < 0");
        }

        return {data.dims, data.dims + data.ndims};
    }

    /// Returns unique id of the logical tensor
    ///
    /// @returns Id number
    size_t get_id() const { return data.id; }

    /// Returns data type of the logical tensor
    ///
    /// @returns The data type
    data_type get_data_type() const {
        return static_cast<data_type>(data.data_type);
    }

    /// Returns property type of the logical tensor
    ///
    /// @returns The property type
    property_type get_property_type() const {
        return static_cast<property_type>(data.property);
    }

    /// Returns layout type of the logical tensor
    ///
    /// @returns The layout type
    layout_type get_layout_type() const {
        return static_cast<layout_type>(data.layout_type);
    }

    /// Returns the layout of the tensor
    ///
    /// @returns Layout id
    size_t get_layout_id() const {
        if (get_layout_type() != layout_type::opaque) {
            error::check_succeed(dnnl_graph_invalid_arguments,
                    "layout type should be opaque");
        }

        return data.layout.layout_id;
    }

    /// Returns strides of this logical tensor
    ///
    /// @returns A copy of strides vector
    dims_t get_strides() const {
        if (get_layout_type() != layout_type::strided) {
            error::check_succeed(dnnl_graph_invalid_arguments,
                    "layout type should be strided");
        }

        if (data.ndims < 0) {
            error::check_succeed(dnnl_graph_invalid_arguments,
                    "cannot return strides when ndims < 0");
        }

        return {data.layout.strides, data.layout.strides + data.ndims};
    }

    /// Get memory size required by this logical tensor
    ///
    /// @returns The memory size in bytes
    size_t get_mem_size() const {
        size_t size = 0;
        error::check_succeed(
                dnnl_graph_logical_tensor_get_mem_size(&data, &size),
                "could not get memory size from the logical_tensor");
        return size;
    }

    /// Compares if this and input logical tensor has the same layout
    ///
    /// @param lt The input logical tensor to be compared
    /// @returns @c true if they have the same layout
    ///        @c false if they have different layout
    bool has_same_layout(const logical_tensor &lt) const {
        uint8_t is_same {0};
        error::check_succeed(dnnl_graph_logical_tensor_has_same_layout(
                                     &data, &lt.data, &is_same),
                "could not compare the layout between these logical tensors");
        return is_same != 0;
    }

private:
    static dnnl_graph_data_type_t convert_to_c(data_type dtype) {
        return static_cast<dnnl_graph_data_type_t>(dtype);
    }

    static dnnl_graph_layout_type_t convert_to_c(layout_type ltype) {
        return static_cast<dnnl_graph_layout_type_t>(ltype);
    }

    static dnnl_graph_tensor_property_t convert_to_c(property_type ptype) {
        return static_cast<dnnl_graph_tensor_property_t>(ptype);
    }
};

/// @} dnnl_graph_api_logical_tensor

/// @addtogroup dnnl_graph_api_tensor Tensor
///
/// Tensor is an abstraction for multidimensional input and output data needed
/// in the execution of a compiled partition.
///
/// @{

/// Framework integration code is responsible for managing the tensor's,
/// lifecycle.
class tensor : public detail::tensor_handle {
public:
    using dims_t = std::vector<dnnl_graph_dim_t>;

    /// Default constructor. Constructs an empty object.
    tensor() = default;

    /// Constructs a tensor object according to the given logical tensor.
    ///
    /// @param lt The given logical tensor
    /// @param aengine Engine to store the data on.
    /// @param handle Handle of memory buffer to use as an underlying storage,
    ///     if the ndims in the logical tensor is 0, data handle holds a scalar
    tensor(const logical_tensor &lt, const engine &aengine, void *handle) {
        dnnl_graph_tensor_t t = nullptr;
        error::check_succeed(
                dnnl_graph_tensor_create(&t, &(lt.data), aengine.get(), handle),
                "could not create tensor object with the logical_tensor, "
                "engine, and handle");
        reset(t);
    }

    /// Returns the underlying memory buffer with the specific type
    ///
    /// @tparam T Type of the request buffer
    /// @returns The underlying memory buffer
    template <typename T>
    typename std::add_pointer<T>::type get_data_handle() const {
        void *handle {};
        error::check_succeed(dnnl_graph_tensor_get_if_type(
                                     get(), get_data_type<T>(), &handle),
                "could not get data handle from the tensor");
        return reinterpret_cast<typename std::add_pointer<T>::type>(handle);
    }

    /// Sets the underlying memory buffer
    ///
    /// @param handle Data handle. For the CPU engine, the data handle
    ///     is a pointer to the actual data.
    void set_data_handle(void *handle) {
        error::check_succeed(dnnl_graph_tensor_set_data_handle(get(), handle),
                "setting data handle to the tensor failed");
    }

    /// Returns the associated engine.
    ///
    /// @returns An engine object
    engine get_engine() const {
        dnnl_graph_engine_t c_engine = nullptr;
        error::check_succeed(dnnl_graph_tensor_get_engine(get(), &c_engine),
                "could not get an engine from a tensor object");
        return engine(c_engine, true);
    }
};

/// @} dnnl_graph_api_tensor

/// @addtogroup dnnl_graph_api_compiled_partition Compiled_partition
///
/// A compiled partition represents the generated code specialized for target
/// hardware and meta data described by parameter logical tensors.
///
/// @{

/// A compiled partition contains a partition and a handle representing the
/// target specific compiled object.
class compiled_partition : public detail::compiled_partition_handle {
public:
    /// Default constructor. Constructs an empty object.
    compiled_partition() = default;

    /// Constructs a compiled partition object
    compiled_partition(dnnl_graph_compiled_partition_t compiled_partition) {
        reset(compiled_partition, false);
    }

    /// Returns the logical tensor according to tensor id
    ///
    /// @param tid The unique id of required tensor
    /// @returns The logical tensor
    logical_tensor query_logical_tensor(size_t tid) const {
        dnnl_graph_logical_tensor_t lt;
        error::check_succeed(dnnl_graph_compiled_partition_query_logical_tensor(
                                     get(), tid, &lt),
                "query logical tensor from compiled_partition failed");
        return logical_tensor {lt};
    }

    /// Returns the in-place port pairs
    ///
    /// @note
    ///     Each entry of the returned vector is a pair of IDs of input and
    ///     output ports. For in-place ports, users can assign same memory
    ///     buffer when passing tensor along with execution API. The in-place
    ///     optimization is optional, users can always use different memory
    ///     buffers for the execution.
    ///
    /// @returns List of pairs of that indicates input and output use same
    ///     memory buffer.
    std::vector<std::pair<size_t, size_t>> get_inplace_ports() const {
        size_t num = 0;
        const dnnl_graph_inplace_pair_t *inplace_pairs;

        error::check_succeed(dnnl_graph_compiled_partition_get_inplace_ports(
                                     get(), &num, &inplace_pairs),
                "could not get the in-place pairs from a compiled partition");
        if (num == 0) return {};

        std::vector<std::pair<size_t, size_t>> inplace_options;
        inplace_options.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            const dnnl_graph_inplace_pair_t *inplace_pair = inplace_pairs + i;
            inplace_options.emplace_back(
                    inplace_pair->input_id, inplace_pair->output_id);
        }
        return inplace_options;
    }

    /// Execute a compiled partition
    ///
    /// @param astream Stream object to run over
    /// @param inputs A list of input tensors in the partition
    /// @param outputs A list of output tensors in the partition
    void execute(stream &astream, const std::vector<tensor> &inputs,
            const std::vector<tensor> &outputs) const {
        std::vector<const_dnnl_graph_tensor_t> c_inputs;
        c_inputs.reserve(inputs.size());
        for (auto &in : inputs) {
            c_inputs.push_back(in.get());
        }
        std::vector<const_dnnl_graph_tensor_t> c_outputs;
        c_outputs.reserve(outputs.size());
        for (auto &out : outputs) {
            c_outputs.push_back(out.get());
        }

        error::check_succeed(
                dnnl_graph_compiled_partition_execute(get(), astream.get(),
                        c_inputs.size(), c_inputs.data(), c_outputs.size(),
                        c_outputs.data()),
                "could not execute the compiled_partition");
    }
};

/// @} dnnl_graph_api_compiled_partition

/// @addtogroup dnnl_graph_api_op Op
///
/// OP is an abstraction of compute logic for deep neural network operation.
///
/// @{

/// A op contains kind, attribute, and the input and output logical tensor(s).
class op : public detail::op_handle {
public:
    /// Kinds of operations
    enum class kind {
        Abs = dnnl_graph_op_abs,
        AbsBackprop = dnnl_graph_op_abs_backprop,
        Add = dnnl_graph_op_add,
        AvgPool = dnnl_graph_op_avg_pool,
        AvgPoolBackprop = dnnl_graph_op_avg_pool_backprop,
        BatchNormForwardTraining = dnnl_graph_op_batch_norm_forward_training,
        BatchNormInference = dnnl_graph_op_batch_norm_inference,
        BatchNormTrainingBackprop = dnnl_graph_op_batch_norm_backprop,
        BiasAdd = dnnl_graph_op_bias_add,
        BiasAddBackprop = dnnl_graph_op_bias_add_backprop,
        Clamp = dnnl_graph_op_clamp,
        ClampBackprop = dnnl_graph_op_clamp_backprop,
        Concat = dnnl_graph_op_concat,
        Convolution = dnnl_graph_op_convolution,
        ConvolutionBackpropData = dnnl_graph_op_convolution_backprop_data,
        ConvolutionBackpropFilters = dnnl_graph_op_convolution_backprop_filters,
        ConvTranspose = dnnl_graph_op_conv_transpose,
        ConvTransposeBackpropData = dnnl_graph_op_conv_transpose_backprop_data,
        ConvTransposeBackpropFilters
        = dnnl_graph_op_conv_transpose_backprop_filters,
        Dequantize = dnnl_graph_op_dequantize,
        Divide = dnnl_graph_op_divide,
        DynamicDequantize = dnnl_graph_op_dynamic_dequantize,
        DynamicQuantize = dnnl_graph_op_dynamic_quantize,
        DynamicReshape = dnnl_graph_op_dynamic_reshape,
        DynamicTranspose = dnnl_graph_op_dynamic_transpose,
        Elu = dnnl_graph_op_elu,
        EluBackprop = dnnl_graph_op_elu_backprop,
        End = dnnl_graph_op_end,
        Erf = dnnl_graph_op_erf,
        Exp = dnnl_graph_op_exp,
        GELU = dnnl_graph_op_gelu,
        GELUBackprop = dnnl_graph_op_gelu_backprop,
        HardSwish = dnnl_graph_op_hard_swish,
        HardSwishBackprop = dnnl_graph_op_hard_swish_backprop,
        HardTanh = dnnl_graph_op_hard_tanh,
        HardTanhBackprop = dnnl_graph_op_hard_tanh_backprop,
        Index = dnnl_graph_op_index,
        Interpolate = dnnl_graph_op_interpolate,
        InterpolateBackprop = dnnl_graph_op_interpolate_backprop,
        LayerNorm = dnnl_graph_op_layer_norm,
        LayerNormBackprop = dnnl_graph_op_layer_norm_backprop,
        Log = dnnl_graph_op_log,
        LogSoftmax = dnnl_graph_op_log_softmax,
        LogSoftmaxBackprop = dnnl_graph_op_log_softmax_backprop,
        MatMul = dnnl_graph_op_matmul,
        Maximum = dnnl_graph_op_maximum,
        MaxPool = dnnl_graph_op_max_pool,
        MaxPoolBackprop = dnnl_graph_op_max_pool_backprop,
        Minimum = dnnl_graph_op_minimum,
        Mish = dnnl_graph_op_mish,
        MishBackprop = dnnl_graph_op_mish_backprop,
        Multiply = dnnl_graph_op_multiply,
        Negative = dnnl_graph_op_negative,
        Pow = dnnl_graph_op_pow,
        PowBackprop = dnnl_graph_op_pow_backprop,
        PowBackpropExponent = dnnl_graph_op_pow_backprop_exponent,
        PReLU = dnnl_graph_op_prelu,
        PReLUBackprop = dnnl_graph_op_prelu_backprop,
        Quantize = dnnl_graph_op_quantize,
        Reciprocal = dnnl_graph_op_reciprocal,
        ReduceL1 = dnnl_graph_op_reduce_l1,
        ReduceL2 = dnnl_graph_op_reduce_l2,
        ReduceMax = dnnl_graph_op_reduce_max,
        ReduceMean = dnnl_graph_op_reduce_mean,
        ReduceMin = dnnl_graph_op_reduce_min,
        ReduceProd = dnnl_graph_op_reduce_prod,
        ReduceSum = dnnl_graph_op_reduce_sum,
        ReLU = dnnl_graph_op_relu,
        ReLUBackprop = dnnl_graph_op_relu_backprop,
        Reorder = dnnl_graph_op_reorder,
        Round = dnnl_graph_op_round,
        Sigmoid = dnnl_graph_op_sigmoid,
        SigmoidBackprop = dnnl_graph_op_sigmoid_backprop,
        Sign = dnnl_graph_op_sign,
        SoftMax = dnnl_graph_op_softmax,
        SoftMaxBackprop = dnnl_graph_op_softmax_backprop,
        SoftPlus = dnnl_graph_op_softplus,
        SoftPlusBackprop = dnnl_graph_op_softplus_backprop,
        Sqrt = dnnl_graph_op_sqrt,
        SqrtBackprop = dnnl_graph_op_sqrt_backprop,
        Square = dnnl_graph_op_square,
        SquaredDifference = dnnl_graph_op_squared_difference,
        StaticReshape = dnnl_graph_op_static_reshape,
        StaticTranspose = dnnl_graph_op_static_transpose,
        Subtract = dnnl_graph_op_subtract,
        Tanh = dnnl_graph_op_tanh,
        TanhBackprop = dnnl_graph_op_tanh_backprop,
        TypeCast = dnnl_graph_op_type_cast,
        Wildcard = dnnl_graph_op_wildcard,
        // Sentinel
        LastSymbol = dnnl_graph_op_last_symbol,
    };

    /// Attributes of operations
    enum class attr {
        undef = dnnl_graph_op_attr_undef,

        // float32 attributes
        alpha = dnnl_graph_op_attr_alpha,
        beta = dnnl_graph_op_attr_beta,
        epsilon = dnnl_graph_op_attr_epsilon,
        max = dnnl_graph_op_attr_max,
        min = dnnl_graph_op_attr_min,
        momentum = dnnl_graph_op_attr_momentum,

        // float32 vector attributes
        scales = dnnl_graph_op_attr_scales,

        // int64_t attributes
        axis = dnnl_graph_op_attr_axis,
        begin_norm_axis = dnnl_graph_op_attr_begin_norm_axis,
        groups = dnnl_graph_op_attr_groups,

        // int64_t vector attributes
        axes = dnnl_graph_op_attr_axes,
        dilations = dnnl_graph_op_attr_dilations,
        filter_shape = dnnl_graph_op_attr_filter_shape,
        input_shape = dnnl_graph_op_attr_input_shape,
        kernel = dnnl_graph_op_attr_kernel,
        order = dnnl_graph_op_attr_order,
        output_padding = dnnl_graph_op_attr_output_padding,
        output_shape = dnnl_graph_op_attr_output_shape,
        pads_begin = dnnl_graph_op_attr_pads_begin,
        pads_end = dnnl_graph_op_attr_pads_end,
        shape = dnnl_graph_op_attr_shape,
        sizes = dnnl_graph_op_attr_sizes,
        strides = dnnl_graph_op_attr_strides,
        zps = dnnl_graph_op_attr_zps,

        // bool attributes
        exclude_pad = dnnl_graph_op_attr_exclude_pad,
        keep_dims = dnnl_graph_op_attr_keep_dims,
        keep_stats = dnnl_graph_op_attr_keep_stats,
        per_channel_broadcast = dnnl_graph_op_attr_per_channel_broadcast,
        special_zero = dnnl_graph_op_attr_special_zero,
        transpose_a = dnnl_graph_op_attr_transpose_a,
        transpose_b = dnnl_graph_op_attr_transpose_b,
        use_affine = dnnl_graph_op_attr_use_affine,
        use_dst = dnnl_graph_op_attr_use_dst,

        // string attributes
        auto_broadcast = dnnl_graph_op_attr_auto_broadcast,
        auto_pad = dnnl_graph_op_attr_auto_pad,
        coordinate_transformation_mode
        = dnnl_graph_op_attr_coordinate_transformation_mode,
        data_format = dnnl_graph_op_attr_data_format,
        filter_format = dnnl_graph_op_attr_filter_format,
        mode = dnnl_graph_op_attr_mode,
        qtype = dnnl_graph_op_attr_qtype,
        rounding_type = dnnl_graph_op_attr_rounding_type,
    };

    /// Constructs an OP object
    ///
    /// @param id The unique id of this op
    /// @param akind The op kind specifies which computation is represented by
    ///     the op, such as Convolution and ReLU.
    /// @param verbose_name The string added for debug
    op(size_t id, kind akind, const std::string &verbose_name) {
        dnnl_graph_op_t op = nullptr;
        error::check_succeed(dnnl_graph_op_create(&op, id, convert_to_c(akind),
                                     verbose_name.c_str()),
                "could not create op with id and op kind");
        reset(op);
    }

    /// Contructs an Op object based on input/output tensors and attributes
    ///
    /// @param id The unique id of this op.
    /// @param akind The op kind specifies which computation is represented by
    ///     this op, such as Convolution and ReLU.
    /// @param inputs Input logical tensor to be bound to this op.
    /// @param outputs Output logical tensor to be bound to this op
    /// @param verbose_name The string added as the op name.
    op(size_t id, kind akind, const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs,
            const std::string &verbose_name)
        : op(id, akind, verbose_name) {
        for (const auto &input : inputs) {
            error::check_succeed(dnnl_graph_op_add_input(get(), &(input.data)),
                    "adding input to the op failed");
        }
        for (const auto &output : outputs) {
            error::check_succeed(
                    dnnl_graph_op_add_output(get(), &(output.data)),
                    "adding output to the op failed");
        }
    }

    /// Adds input logical tensor to the op
    ///
    /// @param t Input logical tensor
    void add_input(const logical_tensor &t) {
        error::check_succeed(dnnl_graph_op_add_input(get(), &(t.data)),
                "adding input to the op failed");
    }

    /// Adds input logical tensors to the op
    ///
    /// @param ts The list of input logical tensors
    void add_inputs(const std::vector<logical_tensor> &ts) {
        for (const auto &t : ts) {
            error::check_succeed(dnnl_graph_op_add_input(get(), &(t.data)),
                    "adding input to the op failed");
        }
    }

    /// Adds output logical tensor to the op
    ///
    /// @param t Output logical tensor
    void add_output(const logical_tensor &t) {
        error::check_succeed(dnnl_graph_op_add_output(get(), &(t.data)),
                "adding output to the op failed");
    }

    /// Adds output logical tensors to the op
    ///
    /// @param ts The list of output logical tensors
    void add_outputs(const std::vector<logical_tensor> &ts) {
        for (const auto &t : ts) {
            error::check_succeed(dnnl_graph_op_add_output(get(), &(t.data)),
                    "adding output to the op failed");
        }
    }

    /// Sets the attribute according to the name and type (int64_t)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute
    /// @param value The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, int64_t>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        error::check_succeed(dnnl_graph_op_set_attr_s64(get(), attr, &value, 0),
                "could not set attribute to the op");
        return *this;
    }

    template <typename Type,
            requires<std::is_same<Type, int64_t>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::check_succeed(dnnl_graph_op_set_attr_s64(get(), attr, &value, 0),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (float)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute
    /// @param value The attribute's value
    /// @returns The Op self
    template <typename Type, requires<std::is_same<Type, float>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        error::check_succeed(dnnl_graph_op_set_attr_f32(get(), attr, &value, 0),
                "could not set attribute to the op");
        return *this;
    }

    template <typename Type, requires<std::is_same<Type, float>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::check_succeed(dnnl_graph_op_set_attr_f32(get(), attr, &value, 0),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (bool)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute
    /// @param value The attribute's value
    /// @returns The Op self
    template <typename Type, requires<std::is_same<Type, bool>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        const uint8_t val = value;
        error::check_succeed(dnnl_graph_op_set_attr_bool(get(), attr, &val, 0),
                "could not set attribute to the op");
        return *this;
    }

    template <typename Type, requires<std::is_same<Type, bool>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        const uint8_t val = value;
        error::check_succeed(dnnl_graph_op_set_attr_bool(get(), attr, &val, 0),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (string)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute
    /// @param value The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::string>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        error::check_succeed(dnnl_graph_op_set_attr_str(
                                     get(), attr, value.c_str(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    template <typename Type,
            requires<std::is_same<Type, std::string>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::check_succeed(dnnl_graph_op_set_attr_str(
                                     get(), attr, value.c_str(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type
    /// (std::vector<int64_t>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute
    /// @param value The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        error::check_succeed(dnnl_graph_op_set_attr_s64(
                                     get(), attr, value.data(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    template <typename Type,
            requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::check_succeed(dnnl_graph_op_set_attr_s64(
                                     get(), attr, value.data(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type
    /// (std::vector<float>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute
    /// @param value The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::vector<float>>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        error::check_succeed(dnnl_graph_op_set_attr_f32(
                                     get(), attr, value.data(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    template <typename Type,
            requires<std::is_same<Type, std::vector<float>>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::check_succeed(dnnl_graph_op_set_attr_f32(
                                     get(), attr, value.data(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

private:
    dnnl_graph_op_kind_t convert_to_c(kind akind) {
        return static_cast<dnnl_graph_op_kind_t>(akind);
    }

    dnnl_graph_op_attr_t convert_to_c(attr aattr) {
        return static_cast<dnnl_graph_op_attr_t>(aattr);
    }

    template <typename Type,
            requires<std::is_same<Type, int64_t>::value> = true>
    constexpr static dnnl_graph_attribute_kind_t attr_kind() noexcept {
        return dnnl_graph_attribute_kind_i;
    }

    template <typename Type,
            requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
    constexpr static dnnl_graph_attribute_kind_t attr_kind() noexcept {
        return dnnl_graph_attribute_kind_is;
    }

    template <typename Type, requires<std::is_same<Type, float>::value> = true>
    constexpr static dnnl_graph_attribute_kind_t attr_kind() noexcept {
        return dnnl_graph_attribute_kind_f;
    }

    template <typename Type,
            requires<std::is_same<Type, std::vector<float>>::value> = true>
    constexpr static dnnl_graph_attribute_kind_t attr_kind() noexcept {
        return dnnl_graph_attribute_kind_fs;
    }

    template <typename Type,
            requires<std::is_same<Type, std::string>::value> = true>
    constexpr static dnnl_graph_attribute_kind_t attr_kind() noexcept {
        return dnnl_graph_attribute_kind_s;
    }

    template <typename Attr, requires<std::is_same<Attr, bool>::value> = true>
    constexpr static dnnl_graph_attribute_kind_t attr_kind() noexcept {
        return dnnl_graph_attribute_kind_b;
    }

    attr str2attr(const std::string &name) {
#define IF_HANDLE(x) \
    if (name == #x) return attr::x

        IF_HANDLE(alpha);
        IF_HANDLE(beta);
        IF_HANDLE(epsilon);
        IF_HANDLE(max);
        IF_HANDLE(min);
        IF_HANDLE(momentum);
        IF_HANDLE(scales);
        IF_HANDLE(axis);
        IF_HANDLE(begin_norm_axis);
        IF_HANDLE(groups);
        IF_HANDLE(axes);
        IF_HANDLE(dilations);
        IF_HANDLE(filter_shape);
        IF_HANDLE(input_shape);
        IF_HANDLE(kernel);
        IF_HANDLE(order);
        IF_HANDLE(output_padding);
        IF_HANDLE(output_shape);
        IF_HANDLE(pads_begin);
        IF_HANDLE(pads_end);
        IF_HANDLE(shape);
        IF_HANDLE(sizes);
        IF_HANDLE(strides);
        IF_HANDLE(zps);
        IF_HANDLE(exclude_pad);
        IF_HANDLE(keep_dims);
        IF_HANDLE(keep_stats);
        IF_HANDLE(per_channel_broadcast);
        IF_HANDLE(special_zero);
        IF_HANDLE(transpose_a);
        IF_HANDLE(transpose_b);
        IF_HANDLE(use_affine);
        IF_HANDLE(use_dst);
        IF_HANDLE(auto_broadcast);
        IF_HANDLE(auto_pad);
        IF_HANDLE(coordinate_transformation_mode);
        IF_HANDLE(data_format);
        IF_HANDLE(filter_format);
        IF_HANDLE(mode);
        IF_HANDLE(qtype);
        IF_HANDLE(rounding_type);

#undef IF_HANDLE

        return attr::undef;
    }
};

/// @} dnnl_graph_api_op

/// @addtogroup dnnl_graph_api_partition Partition
///
/// Partition represents a collection of OPs identified by oneDNN graph
/// implementation as the basic unit for compilation and execution.
///
/// @{

/// A partition contains a list of OP ids.
class partition : public detail::partition_handle {
public:
    /// Policy specifications for partitioning
    enum class policy {
        /// Best optimization
        /// for now, the `max` mode is just the same as `fusion` mode.
        max = dnnl_graph_partition_policy_max,
        /// Have fusion
        fusion = dnnl_graph_partition_policy_fusion,
        /// No optimization
        debug = dnnl_graph_partition_policy_debug,
    };

    partition() = default;

    /// Constructs a partition object
    ///
    /// @param p A raw pointer to the C API handle
    partition(dnnl_graph_partition_t p) { reset(p, false); }

    /// Constructs a partition with a given op and engine kind
    ///
    /// @param aop An operator used to create the partition
    /// @param ekind Engine kind
    partition(const op &aop, engine::kind ekind) {
        dnnl_graph_partition_t p = nullptr;
        error::check_succeed(
                dnnl_graph_partition_create_with_op(&p, aop.get(),
                        static_cast<dnnl_graph_engine_kind_t>(ekind)),
                "could not create a partition with the op and engine kind");
        reset(p);
    }

    /// Returns the number of dnnl graph ops in the partition
    ///
    /// @returns Number of ops
    size_t get_ops_num() const {
        size_t num {0};
        error::check_succeed(dnnl_graph_partition_get_op_num(get(), &num),
                "could not get number of ops from the partition");
        return num;
    }

    /// Returns all op’s id of the partition
    ///
    /// @returns An unordered set of op ids
    std::vector<size_t> get_ops() const {
        auto num = get_ops_num();
        std::vector<size_t> ops(num);

        error::check_succeed(
                dnnl_graph_partition_get_ops(get(), num, ops.data()),
                "could not get op ids from the partition");
        return ops;
    }

    /// Returns the unique id of the partition
    ///
    /// @returns Unique id
    size_t get_id() const {
        size_t id {};
        error::check_succeed(dnnl_graph_partition_get_id(get(), &id),
                "could not get id of the partition");
        return id;
    }

    /// Compile the partition to generate compiled partition based
    /// on the input/output logical tensors. The order of these two lists
    /// may have already been changed according to the fwk fused op.
    ///
    /// @param inputs A list of input logical tensors
    /// @param outputs A list of output logical tensors
    /// @param e The engine used to compile the partition
    /// @returns A compiled partition
    compiled_partition compile(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine &e) const {
        if (!is_supported()) {
            error::check_succeed(dnnl_graph_invalid_arguments,
                    "could not compile an unsupported partition");
        }

        return compile_(inputs, outputs, &e);
    }

    /// Returns the supporting status of the partition
    ///
    /// @returns @c true if this partition is supported by oneDNN Graph backend
    ///     @c false if this partition isn't supported by oneDNN Graph backend
    bool is_supported() const {
        uint8_t supported {0};
        error::check_succeed(
                dnnl_graph_partition_is_supported(get(), &supported),
                "could not get supporting status of the partition");
        return supported != 0;
    }

    /// Returns a list of input logical tensors from the partition
    ///
    /// @returns A list of input logical tensors
    std::vector<logical_tensor> get_in_ports() const {
        size_t num = 0;
        error::check_succeed(dnnl_graph_partition_get_in_ports_num(get(), &num),
                "could not get number of inputs of the partition");
        if (num == 0) return {};

        std::vector<dnnl_graph_logical_tensor_t> c_inputs(num);
        error::check_succeed(
                dnnl_graph_partition_get_in_ports(get(), num, c_inputs.data()),
                "could not get input logical tensors of the partition");

        std::vector<logical_tensor> inputs;
        inputs.reserve(num);
        for (auto &c_lt : c_inputs)
            inputs.emplace_back(c_lt);
        return inputs;
    }

    /// Returns a list of output logical tensors from the partition
    ///
    /// @returns A list of output logical tensor
    std::vector<logical_tensor> get_out_ports() const {
        size_t num = 0;
        error::check_succeed(
                dnnl_graph_partition_get_out_ports_num(get(), &num),
                "cannot get number of outputs of the partition");
        if (num == 0) return {};

        std::vector<dnnl_graph_logical_tensor_t> c_outputs(num);
        error::check_succeed(dnnl_graph_partition_get_out_ports(
                                     get(), num, c_outputs.data()),
                "could not get output logical tensors of the partition");

        std::vector<logical_tensor> outputs;
        outputs.reserve(num);
        for (auto &c_lt : c_outputs)
            outputs.emplace_back(c_lt);
        return outputs;
    }

    /// Returns the engine kind of the partition
    ///
    /// @returns The engine kind
    engine::kind get_engine_kind() const {
        dnnl_graph_engine_kind_t akind;
        error::check_succeed(
                dnnl_graph_partition_get_engine_kind(get(), &akind),
                "cannot get the engine kind from the partition");

        return static_cast<engine::kind>(akind);
    }

private:
    compiled_partition compile_(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine *e) const {
        std::vector<const dnnl_graph_logical_tensor_t *> c_inputs;
        std::vector<const dnnl_graph_logical_tensor_t *> c_outputs;

        c_inputs.reserve(inputs.size());
        for (const auto &in : inputs) {
            c_inputs.push_back(&(in.data));
        }

        c_outputs.reserve(outputs.size());
        for (const auto &out : outputs) {
            c_outputs.push_back(&(out.data));
        }

        dnnl_graph_compiled_partition_t cpartitions = nullptr;
        error::check_succeed(
                dnnl_graph_compiled_partition_create(&cpartitions, get()),
                "could not create compiled_partition");
        error::check_succeed(
                dnnl_graph_partition_compile(get(), cpartitions,
                        c_inputs.size(), c_inputs.data(), c_outputs.size(),
                        c_outputs.data(), e->get()),
                "partition compile failed");

        return compiled_partition(cpartitions);
    }
};

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_graph Graph
///
/// A Graph contains a set of OPs. #dnnl::graph::graph::add_op() adds an OP and
/// its logical tensors to a graph. oneDNN Graph implementation accumulates the
/// OPs and logical tensors and constructs and validates the graph as internal
/// state.
///
/// @{

/// A graph session to start analysis of computational DAG.
class graph : public detail::graph_handle {
public:
    // floating-point math mode
    enum class fpmath_mode {
        /// Default behavior, no downconversions allowed
        strict = dnnl_graph_fpmath_mode_strict,
        /// Implicit f32->bf16 or f32->f19 conversions allowed
        bf16 = dnnl_graph_fpmath_mode_bf16,
        /// Implicit f32->f16 or f32->f19 conversions allowed
        f16 = dnnl_graph_fpmath_mode_f16,
        /// Implicit f32->f16 or f32->bf16 or f32->f19 conversions allowed
        any = dnnl_graph_fpmath_mode_any,
        /// Implicit f32->f19 conversions allowed
        f19 = dnnl_graph_fpmath_mode_f19,
    };

    /// Constructs a graph session using device information
    ///
    /// @param engine_kind Can be cpu, gpu or any supported engine.
    graph(engine::kind engine_kind) {
        dnnl_graph_graph_t g = nullptr;
        error::check_succeed(
                dnnl_graph_graph_create(&g, engine::convert_to_c(engine_kind)),
                "could not create graph with engine kind");
        reset(g);
    }

    /// Constructs a graph session using device information and math mode
    ///
    /// @param engine_kind Can be cpu, gpu or any supported engine.
    /// @param mode Specified floating-point math mode.
    graph(engine::kind engine_kind, fpmath_mode mode) {
        dnnl_graph_graph_t g = nullptr;
        error::check_succeed(
                dnnl_graph_graph_create_with_fpmath_mode(&g,
                        engine::convert_to_c(engine_kind), convert_to_c(mode)),
                "could not create graph with engine kind and math mode");
        reset(g);
    }

    /// Add an op to the graph session to construct DAG for analysis
    ///
    /// @param op An operator that represents the entry of frameworks'
    ///    graph
    /// @param allow_exception A flag indicating whether the method is allowed
    ///    to throw an exception if it fails to add the op to the graph.
    /// @returns #success or a status describing the error otherwise.
    status add_op(const op &op, bool allow_exception = true) {
        dnnl_graph_status_t ret = dnnl_graph_add_op(get(), op.get());

        if (allow_exception) {
            error::check_succeed(ret, "could not add op to the graph");
        }

        return static_cast<status>(ret);
    }

    /// Get filtered partitions
    ///
    /// @param policy Partition policy, defaults to
    ///     #dnnl::graph::partition::policy::fusion
    /// @return A vector storing the partitions
    std::vector<partition> get_partitions(
            partition::policy policy = partition::policy::fusion) {
        error::check_succeed(
                dnnl_graph_graph_filter(get(),
                        static_cast<dnnl_graph_partition_policy_t>(policy)),
                "filter graph failed");

        size_t num = 0;
        error::check_succeed(dnnl_graph_graph_get_partition_num(get(), &num),
                "could not get number of partitions from the graph");

        // return early if there is no partitions in the graph.
        if (num == 0) return {};

        std::vector<partition> out_list;
        out_list.reserve(num);

        std::vector<dnnl_graph_partition_t> partitions(num);
        for (auto &p : partitions) {
            error::check_succeed(dnnl_graph_partition_create(&p),
                    "could not create partition");
        }
        dnnl_graph_graph_get_partitions(get(), num, partitions.data());

        for (auto p : partitions) {
            out_list.emplace_back(p);
        }

        error::check_succeed(dnnl_graph_graph_visualize(get(), 0),
                "cannot visualize the graph");

        return out_list;
    }

    /// Visualize the graph
    void visualize() const {
        error::check_succeed(dnnl_graph_graph_visualize(get(), 1),
                "cannot visualize the graph");
    }

private:
    static dnnl_graph_fpmath_mode_t convert_to_c(fpmath_mode mode) {
        return static_cast<dnnl_graph_fpmath_mode_t>(mode);
    }
};

/// @} dnnl_graph_api_graph

/// @addtogroup dnnl_graph_api_compiled_partition_cache Compiled Partition Cache
///
/// A set of functions that provide compiled partition cache control.
///
/// @{

/// Returns the number of compiled partition that can be held in the compiled
/// partition cache at the same time.
inline int get_compiled_partition_cache_capacity() {
    int result = 0;
    error::check_succeed(
            dnnl_graph_get_compiled_partition_cache_capacity(&result),
            "could not get compiled partition cache capacity");
    return result;
}

/// @copydoc dnnl_graph_set_compiled_partition_cache_capacity(int capacity)
inline void set_compiled_partition_cache_capacity(int capacity) {
    error::check_succeed(
            dnnl_graph_set_compiled_partition_cache_capacity(capacity),
            "could not set compiled partition cache capacity");
}

/// @} dnnl_graph_api_compiled_partition_cache

/// @addtogroup dnnl_graph_api_service Service
///
/// A set of functions that aid in oneDNN Graph debugging and profiling.
///
/// @{

/// @copydoc dnnl_graph_version_t
using version_t = dnnl_graph_version_t;

/// @copydoc dnnl_graph_version()
inline const version_t *version() {
    return dnnl_graph_version();
}

/// @} dnnl_api_service

/// @addtogroup dnnl_graph_constant_tensor_cache Constant Tensor Cache
///
/// A set of functions that provide constant tensor cache control
///
/// @{

/// Control the enabling or disabling of constant tensor cache. This API must
/// be called once before compilation stage.
///
/// @param flag Set to positive value to enable the cache and set to 0 to
/// disable the cache. Negative values are invalid.
inline void set_constant_tensor_cache(int flag) {
    error::check_succeed(dnnl_graph_set_constant_tensor_cache(flag),
            "fail to set constant tensor cache");
}

/// Return the enabling status of constant tensor cache.
inline int get_constant_tensor_cache() {
    int result = 0;
    error::check_succeed(dnnl_graph_get_constant_tensor_cache(&result),
            "fail to get constant tensor cache");
    return result;
}

/// @} dnnl_graph_constant_tensor_cache

} // namespace graph
} // namespace dnnl

/// oneAPI namespace
// Contains the oneapi::dnnl namespace as an alias to the ::dnnl namespace.
namespace oneapi {
// Note: without this guard, doxygen warns of potentially recursive namespace
#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// oneDNN alias namespace
namespace dnnl = ::dnnl;
#endif
} // namespace oneapi

/// @} dnnl_graph_api

#endif

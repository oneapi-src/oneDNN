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

/// @cond DO_NOT_DOCUMENT_THIS
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/// @endcond

/// @addtogroup dnnl_graph_api oneDNN Graph API
/// @{

/// oneDNN namespace
namespace dnnl {

/// oneDNN Graph API namespace
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

/// @cond DO_NOT_DOCUMENT_THIS
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
DNNL_GRAPH_HANDLE_ALIAS(compilation_context);

#undef DNNL_GRAPH_HANDLE_ALIAS

/// @endcond
} // namespace detail

/// oneDNN Graph exception class.
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

/// @cond DO_NOT_DOCUMENT_THIS

template <bool B>
using requires = typename std::enable_if<B, bool>::type;

/// @endcond

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

/// @} dnnl_graph_api_status

/// @addtogroup dnnl_graph_api_allocator Allocator
///
/// Definitions of allocator which is used to acquire memory resources in
/// partition compilation and execution.
///
/// @{

/// An allocator.
class allocator : public detail::allocator_handle {
public:
    using detail::allocator_handle::handle;

    /// Constructs an allocator according to given function pointers
    ///
    /// @param host_malloc A pointer to malloc function for host.
    /// @param host_free A pointer to free function for host.
    allocator(dnnl_graph_host_allocate_f host_malloc,
            dnnl_graph_host_deallocate_f host_free) {
        dnnl_graph_allocator_t a = nullptr;
        error::check_succeed(
                dnnl_graph_allocator_create(&a, host_malloc, host_free),
                "could not create allocator for host");
        reset(a);
    }

    /// Default constructor
    allocator() {
        dnnl_graph_allocator_t a = nullptr;
        error::check_succeed(dnnl_graph_allocator_create(&a, nullptr, nullptr),
                "could not create allocator");
        reset(a);
    }
};

/// @} dnnl_graph_api_allocator

/// @addtogroup dnnl_graph_api_engine Engine
///
/// Engine represents a device (a CPU or a GPU card) and its context. A graph
/// should be created with a specific engine kind and all partitions returned
/// from the graph will inherit the engine kind. An engine object with the same
/// kind will be used to compile a partition and generate corresponding kernels.
/// The compiled partition will executed on the specific engine.
///
/// @{

/// An execution engine.
class engine : public detail::engine_handle {
public:
    using detail::engine_handle::handle;

    /// Kinds of engine.
    enum class kind {
        /// An unspecified engine
        any = dnnl_graph_any_engine,
        /// CPU engine
        cpu = dnnl_graph_cpu,
        /// GPU engine
        gpu = dnnl_graph_gpu,
    };

    /// Constructs an engine with specified kind and index.
    ///
    /// @param akind The kind of engine to construct.
    /// @param index Specify which device to be used.
    engine(kind akind, size_t index) {
        dnnl_graph_engine_t e = nullptr;
        error::check_succeed(
                dnnl_graph_engine_create(&e, convert_to_c(akind), index),
                "could not create engine with engine kind and device index");
        reset(e);
    }

    /// Constructs an engine with specified kind, index, and allocator.
    ///
    /// @param akind The kind of engine to construct.
    /// @param index Specify which device to be used.
    /// @param alloc The memory allocator for the engine.
    engine(kind akind, size_t index, const allocator &alloc) {
        dnnl_graph_engine_t e = nullptr;
        error::check_succeed(dnnl_graph_engine_create_with_allocator(&e,
                                     convert_to_c(akind), index, alloc.get()),
                "could not create engine with engine kind, device index, and "
                "allocator");
        reset(e);
    }

    /// Returns the kind of an engine.
    ///
    ///@returns Kind of the engine.
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
/// An encapsulation of execution context tied to a particular engine.
///
/// @{

/// An execution stream.
class stream : public detail::stream_handle {
public:
    using detail::stream_handle::handle;

    /// Constructs a stream for the specified engine.
    ///
    /// @param engine Engine to create stream on.
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
/// Logical tensor describes the meta-data of the input or output tensor, like
/// elements data type, number of dimensions, size for each dimension (shape),
/// layout, and the property of the tensor.
///
/// Each logical tensor has an unique ID. The library uses logical tensor IDs to
/// build up the connections between operations if the output of one operation
/// has the same ID as the input of another operation. The meta-data in a
/// logical tensor may be enriched in the framework graph as it progresses
/// toward final execution. For example, the library doesn't require detailed
/// shape information at the operation and graph creation stage. But shape
/// information of input logical tensor will be required at partition
/// compilation stage. Logical tensor is not mutable. Users must create a new
/// logical tensor with the same ID to pass any new additional information to
/// oneDNN Graph API. Please note that the library also has unique IDs for
/// operations. The ID should be unique among different logical tensors, but it
/// can have the same value between a logical tensor and an operation.
///
/// @{

/// Logical tensor object
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
        /// Undefined layout type.
        undef = dnnl_graph_layout_type_undef,
        /// Any means to let the library to decide the layout for a tensor
        /// during partition compilation.
        any = dnnl_graph_layout_type_any,
        /// Strided means that the layout of a tensor is determined by the
        /// strides field in the logical tensor.
        strided = dnnl_graph_layout_type_strided,
        /// Opaque means that the layout of a tensor is the library specific.
        /// Usually, an opaque layout is generated by a partition which is
        /// compiled with layout type any.
        opaque = dnnl_graph_layout_type_opaque,
    };

    /// Tensor property
    enum class property_type {
        /// Undefined tensor property.
        undef = dnnl_graph_tensor_property_undef,
        /// Variable means the tensor may be changed during computation or
        /// between different iterations.
        variable = dnnl_graph_tensor_property_variable,
        /// Constant means the tensor will keep unchanged during computation and
        /// between different iterations. It's useful for the library to apply
        /// optimizations for constant tensors or cache constant tensors inside
        /// the library. For example, constant weight tensors in inference
        /// scenarios.
        constant = dnnl_graph_tensor_property_constant,
    };

    /// Default constructor for an empty logical tensor object.
    logical_tensor() = default;

    /// Constructs a logical tensor object from C data.
    explicit logical_tensor(const dnnl_graph_logical_tensor_t &c_data)
        : data(c_data) {}

    /// Copy constructor.
    logical_tensor(const logical_tensor &other) = default;

    /// Assignment
    logical_tensor &operator=(const logical_tensor &other) = default;

    /// Constructs a logical tensor object with ID, data type, ndims, layout
    /// type, and property type.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param ndims Number of dimensions. -1 means unknown and 0 means a scalar
    ///     tensor.
    /// @param ltype Layout type.
    /// @param ptype Property type.
    logical_tensor(size_t tid, data_type dtype, int32_t ndims,
            layout_type ltype, property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;
        error::check_succeed(
                dnnl_graph_logical_tensor_init(&val, tid, convert_to_c(dtype),
                        ndims, convert_to_c(ltype), convert_to_c(ptype)),
                "could not create logical_tensor with property");
        data = val;
    }

    /// Delegated constructor.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param ltype Layout type.
    logical_tensor(
            size_t tid, data_type dtype, layout_type ltype = layout_type::undef)
        : logical_tensor(tid, dtype, DNNL_GRAPH_UNKNOWN_NDIMS, ltype) {}

    /// Constructs a logical tensor object with basic information and detailed
    /// dims.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param adims Logical tensor dimensions. -1 means the size of that
    ///     dimension is unknown. 0 is used to define zero-dimension tensor.
    /// @param ltype Layout type. If it's strided, the strides field in the
    ///     output logical tensor will be deduced accordingly.
    /// @param ptype Property type.
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

    /// Constructs a logical tensor object with detailed dims and strides. The
    /// layout_type of the output logical tensor object will always be strided.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param adims Logical tensor dimensions. -1 means the size of that
    ///     dimension is unknown. 0 is used to define zero-dimension tensor.
    /// @param strides Logical tensor strides. -1 means the stride of the
    ///     dimension is unknown. The library currently doesn't support other
    ///     negative stride values.
    /// @param ptype Property type.
    logical_tensor(size_t tid, data_type dtype, const dims_t &adims,
            const dims_t &strides, property_type ptype = property_type::undef) {
        dnnl_graph_logical_tensor_t val;
        // TODO(xxx): check the size of adims and strides. They should be same.
        error::check_succeed(
                dnnl_graph_logical_tensor_init_with_strides(&val, tid,
                        convert_to_c(dtype), static_cast<int32_t>(adims.size()),
                        adims.data(), strides.data(), convert_to_c(ptype)),
                "could not create logical_tensor with strides and property");
        data = val;
    }

    /// Constructs a logical tensor object with detailed dims and an opaque
    /// layout ID. layout_type of the output logical tensor object will always
    /// be opaque.
    ///
    /// @param tid Logical tensor ID.
    /// @param dtype Elements data type.
    /// @param adims Logical tensor dimensions. -1 means the size of that
    ///     dimension is unknown. 0 is used to define zero-dimension tensor.
    /// @param lid Opaque layout id.
    /// @param ptype Property type
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

    /// Returns dimensions of a logical tensor.
    ///
    /// @returns A vector describing the size of each dimension.
    dims_t get_dims() const {
        if (data.ndims < 0) {
            error::check_succeed(dnnl_graph_invalid_arguments,
                    "cannot return dims when ndims < 0");
        }

        return {data.dims, data.dims + data.ndims};
    }

    /// Returns the unique id of a logical tensor.
    ///
    /// @returns An integer value describing the ID.
    size_t get_id() const { return data.id; }

    /// Returns the data type of a logical tensor.
    ///
    /// @returns The data type.
    data_type get_data_type() const {
        return static_cast<data_type>(data.data_type);
    }

    /// Returns the property type of a logical tensor.
    ///
    /// @returns The property type.
    property_type get_property_type() const {
        return static_cast<property_type>(data.property);
    }

    /// Returns the layout type of a logical tensor.
    ///
    /// @returns The layout type.
    layout_type get_layout_type() const {
        return static_cast<layout_type>(data.layout_type);
    }

    /// Returns the layout ID of a logical tensor. The API should be called on a
    /// logical tensor with opaque layout type. Otherwise, an exception will be
    /// raised.
    ///
    /// @returns Layout ID.
    size_t get_layout_id() const {
        if (get_layout_type() != layout_type::opaque) {
            error::check_succeed(dnnl_graph_invalid_arguments,
                    "layout type should be opaque");
        }

        return data.layout.layout_id;
    }

    /// Returns the strides of a logical tensor. The API should be called on a
    /// logical tensor with strided layout type. Otherwise, an exception will be
    /// raised.
    ///
    /// @returns A vector describing the stride size of each dimension.
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

    /// Returns memory size in bytes required by this logical tensor.
    ///
    /// @returns The memory size in bytes.
    size_t get_mem_size() const {
        size_t size = 0;
        error::check_succeed(
                dnnl_graph_logical_tensor_get_mem_size(&data, &size),
                "could not get memory size from the logical_tensor");
        return size;
    }

    /// Compares if two logical tenors have the same layout.
    ///
    /// @param lt The input logical tensor to be compared.
    /// @returns @c true if they have the same layout. @c false if they have
    ///     different layout.
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
/// Tensor is an abstraction for multi-dimensional input and output data needed
/// in the execution of a compiled partition. A tensor object encapsulates a
/// handle to a memory buffer allocated on a specific engine and a logical
/// tensor which describes the dimensions, elements data type, and memory
/// layout.
///
/// @{

/// A tensor object
class tensor : public detail::tensor_handle {
public:
    using dims_t = std::vector<dnnl_graph_dim_t>;

    /// Default constructor. Constructs an empty tensor object.
    tensor() = default;

    /// Constructs a tensor object according to a given logical tensor, an
    /// engine, and a memory handle.
    ///
    /// @param lt The given logical tensor
    /// @param aengine Engine to store the data on.
    /// @param handle Handle of memory buffer to use as an underlying storage.
    tensor(const logical_tensor &lt, const engine &aengine, void *handle) {
        dnnl_graph_tensor_t t = nullptr;
        error::check_succeed(
                dnnl_graph_tensor_create(&t, &(lt.data), aengine.get(), handle),
                "could not create tensor object with the logical_tensor, "
                "engine, and handle");
        reset(t);
    }

    /// Returns the underlying memory handle from a tensor.
    ///
    /// @returns The underlying memory handle.
    void *get_data_handle() const {
        void *handle = nullptr;
        error::check_succeed(dnnl_graph_tensor_get_data_handle(get(), &handle),
                "could not get data handle from the tensor");
        return handle;
    }

    /// Sets the underlying memory handle.
    ///
    /// @param handle Memory handle.
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

/// @addtogroup dnnl_graph_api_context Context
///
/// @{

/// Context. It contains useful context information for compilation.
/// Currently, it supports passing tensor content. For example, DynamicReshape
/// has a required shape input, but this shape is stored in a tensor and can be
/// accessed until execution stage. Users can pass the shape tensor content to
/// through context.
///
/// Considering the content in context is critical to compilation, it will
/// be part of compiled partition cache key. That means, if users compile
/// a partition with different context twice time, the compiled partition
/// cache will be missed and recompilation will happen. Users should
/// guarantee the content in context is kept unchanged during each
/// `compilation-execution` period.
///
/// Users can specify the content for any input tensors, but not all of them are
/// critical to compilation. For example, the content of input shape is
/// important for the library to generate kernels for DynamicReshape operation
/// while the content of input data is not useful. Another example is the
/// constant weight tensor for Convolution operation in inference mode,
/// typically it's not critical for kernel generation but it still exposes an
/// optimization opportunity to the library by reordering or freezing the
/// constant weight tensor at compilation stage.
///
/// The context can also be used to pass tensor content in API
/// @ref dnnl::graph::compiled_partition::query_dynamic_outputs. It allows user
/// to query output shapes according to given tensor content provided at
/// execution stage.
///
class context : public detail::compilation_context_handle {
public:
    using dims_t = std::vector<dnnl_graph_dim_t>;

    /// Constructs an empty context.
    context() {
        dnnl_graph_compilation_context_t ctx = nullptr;
        error::check_succeed(dnnl_graph_compilation_context_create(&ctx),
                "could not create a context.");
        reset(ctx);
    }

    /// Constructs a context object for C data.
    ///
    /// @param ctx A raw pointer to the C API handle
    context(dnnl_graph_compilation_context_t ctx) { reset(ctx, false); }

    /// Sets tensor data handle to a context. This handle
    /// will be interpreted according to the logical tensor specified by the
    /// given id.
    ///
    /// @param id Logical tensor id
    /// @param handle A raw pointer to tensor buffer
    void set_tensor_data_handle(size_t id, void *handle) {
        error::check_succeed(
                dnnl_graph_compilation_context_set_tensor_data_handle(
                        get(), id, handle),
                "could not set tensor data handle to a context");
    }
};

/// @} dnnl_graph_api_context

/// @addtogroup dnnl_graph_api_compiled_partition Compiled partition
///
/// A compiled partition represents the generated kernels specialized for a
/// partition on a target hardware (engine) with input and output information
/// specified by the logical tensors.
///
/// @{

/// A compiled partition object.
class compiled_partition : public detail::compiled_partition_handle {
public:
    /// Default constructor. Constructs an empty compiled partition object.
    compiled_partition() = default;

    /// Constructs a compiled partition object.
    compiled_partition(dnnl_graph_compiled_partition_t compiled_partition) {
        reset(compiled_partition, false);
    }

    /// Queries an input or output logical tensor according to tensor ID. If the
    /// tensor ID doesn't belong to any input or output of the compiled
    /// partition, an empty logical tensor will be returned.
    ///
    /// @param tid The unique id of required tensor.
    /// @returns The logical tensor.
    logical_tensor query_logical_tensor(size_t tid) const {
        dnnl_graph_logical_tensor_t lt;
        error::check_succeed(dnnl_graph_compiled_partition_query_logical_tensor(
                                     get(), tid, &lt),
                "query logical tensor from compiled_partition failed");
        return logical_tensor {lt};
    }

    /// Queries a list of dynamic output logical tensors. This API is dedicated
    /// for dynamic shape case, users need provide input logical tensors with
    /// concrete input shapes and then library can help infer output shape.
    /// Other than dynamic dimensions, those determinable dimensions (denoted as
    /// -1 or any positive value) should be the same as the values which are
    /// passed to compilation API before, otherwise an exception will be raised.
    ///
    /// @param inputs The input logical tensors with concrete shapes.
    /// @param context Context information used to assist shape inference.
    /// @returns A list of output logical tensors.
    std::vector<logical_tensor> query_dynamic_outputs(
            const std::vector<logical_tensor> &inputs,
            const context &context = {}) const {
        size_t num = 0;
        error::check_succeed(
                dnnl_graph_compiled_partition_get_outputs_num(get(), &num),
                "could not get outputs num from compiled partition");
        if (num == 0) return {};

        std::vector<dnnl_graph_logical_tensor_t> c_outputs(num);
        std::vector<const dnnl_graph_logical_tensor_t *> c_inputs;
        c_inputs.reserve(inputs.size());
        for (const auto &in : inputs) {
            c_inputs.push_back(&(in.data));
        }

        error::check_succeed(
                dnnl_graph_compiled_partition_query_dynamic_outputs(get(), num,
                        c_outputs.data(), c_inputs.size(), c_inputs.data(),
                        context.get()),
                "could not query dynamic outputs from compiled partition");

        std::vector<logical_tensor> outputs;
        outputs.reserve(num);
        for (auto &c_out : c_outputs)
            outputs.emplace_back(c_out);

        return outputs;
    }

    /// Returns the hint of in-place pairs from a compiled partition. It
    /// indicates that an input and an output of the partition can share the
    /// same memory buffer for computation. In-place computation helps to reduce
    /// the memory footprint and improves cache locality. But since the library
    /// may not have a global view of user's application, it's possible that the
    /// input tensor is used at other places in user's computation graph. In
    /// this case, the user should take the in-place pair as a hint and pass a
    /// different memory buffer for output tensor to avoid overwriting the input
    /// memory buffer which will probably cause unexpected incorrect results.
    ///
    /// @returns A list of pairs of input and output IDs.
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

    /// Execute a compiled partition.
    ///
    /// @param astream Stream object to run over.
    /// @param inputs A list of input tensors.
    /// @param outputs A list of output tensors.
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
/// OP is an abstraction of computation logic for deep neural network
/// operations. An op object encapsulates an operation kind which describes the
/// computation logic, an unique ID which differentiates operations with the
/// same kind, and logical tensors which describes the input and output of the
/// operation and its connections to other operations in the graph.
///
/// @{

/// An op object.
class op : public detail::op_handle {
public:
    /// Kinds of operations.
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
        Equal = dnnl_graph_op_equal,
        Erf = dnnl_graph_op_erf,
        Exp = dnnl_graph_op_exp,
        GELU = dnnl_graph_op_gelu,
        GELUBackprop = dnnl_graph_op_gelu_backprop,
        Greater = dnnl_graph_op_greater,
        GreaterEqual = dnnl_graph_op_greater_equal,
        HardSwish = dnnl_graph_op_hard_swish,
        HardSwishBackprop = dnnl_graph_op_hard_swish_backprop,
        Index = dnnl_graph_op_index,
        Interpolate = dnnl_graph_op_interpolate,
        InterpolateBackprop = dnnl_graph_op_interpolate_backprop,
        LayerNorm = dnnl_graph_op_layer_norm,
        LayerNormBackprop = dnnl_graph_op_layer_norm_backprop,
        LeakyReLU = dnnl_graph_op_leaky_relu,
        Less = dnnl_graph_op_less,
        LessEqual = dnnl_graph_op_less_equal,
        Log = dnnl_graph_op_log,
        LogSoftmax = dnnl_graph_op_log_softmax,
        LogSoftmaxBackprop = dnnl_graph_op_log_softmax_backprop,
        LogicalAnd = dnnl_graph_op_logical_and,
        LogicalNot = dnnl_graph_op_logical_not,
        LogicalOr = dnnl_graph_op_logical_or,
        LogicalXor = dnnl_graph_op_logical_xor,
        MatMul = dnnl_graph_op_matmul,
        Maximum = dnnl_graph_op_maximum,
        MaxPool = dnnl_graph_op_max_pool,
        MaxPoolBackprop = dnnl_graph_op_max_pool_backprop,
        Minimum = dnnl_graph_op_minimum,
        Mish = dnnl_graph_op_mish,
        MishBackprop = dnnl_graph_op_mish_backprop,
        Multiply = dnnl_graph_op_multiply,
        Negative = dnnl_graph_op_negative,
        NotEqual = dnnl_graph_op_not_equal,
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
        Rsqrt = dnnl_graph_op_rsqrt,
        Select = dnnl_graph_op_select,
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

    /// Attributes of operations. Different operations support different
    /// attributes. Check the document of each operation for what attributes are
    /// supported and what are the potential values for them. Missing required
    /// attribute or illegal attribute value may lead to failure when adding the
    /// operation to a graph.
    enum class attr {
        /// Undefined op attribute.
        undef = dnnl_graph_op_attr_undef,

        // float32 attributes. The value of these attributes can be any single
        // float32 number.

        /// Specifies an alpha attribute to an op.
        alpha = dnnl_graph_op_attr_alpha,
        /// Specifies an beta attribute to an op.
        beta = dnnl_graph_op_attr_beta,
        /// Specifies an epsilon attribute to an op.
        epsilon = dnnl_graph_op_attr_epsilon,
        /// Specifies a max attribute to an op.
        max = dnnl_graph_op_attr_max,
        /// Specifies a min attribute to an op.
        min = dnnl_graph_op_attr_min,
        /// Specifies a momentum attribute to an op.
        momentum = dnnl_graph_op_attr_momentum,

        // float32 vector attributes. The value of these attributes can be a
        // vector of float32 numbers.

        /// Specifies a scales attribute to an op.
        scales = dnnl_graph_op_attr_scales,

        // int64_t attributes. The value of these attributes can be any single
        // int64 number.

        /// Specifies an axis attribute to an op.
        axis = dnnl_graph_op_attr_axis,
        /// Specifies a begin_norm_axis attribute to an op.
        begin_norm_axis = dnnl_graph_op_attr_begin_norm_axis,
        /// Specifies a groups attribute to an op.
        groups = dnnl_graph_op_attr_groups,

        // int64_t vector attributes. The value of these attributes can be a
        // vector of int64 numbers.

        /// Specifies an axes attribute to an op.
        axes = dnnl_graph_op_attr_axes,
        /// Specifies a dilations attribute to an op.
        dilations = dnnl_graph_op_attr_dilations,
        /// Specifies a filter_shape attribute to an op.
        filter_shape = dnnl_graph_op_attr_filter_shape,
        /// Specifies an input_shape attribute to an op.
        input_shape = dnnl_graph_op_attr_input_shape,
        /// Specifies a kernel attribute to an op.
        kernel = dnnl_graph_op_attr_kernel,
        /// Specifies an order attribute to an op.
        order = dnnl_graph_op_attr_order,
        /// Specifies an output_padding attribute to an op.
        output_padding = dnnl_graph_op_attr_output_padding,
        /// Specifies an output_shape attribute to an op.
        output_shape = dnnl_graph_op_attr_output_shape,
        /// Specifies a pads_begin attribute to an op.
        pads_begin = dnnl_graph_op_attr_pads_begin,
        /// Specifies a pads_end attribute to an op.
        pads_end = dnnl_graph_op_attr_pads_end,
        /// Specifies a shape attribute to an op.
        shape = dnnl_graph_op_attr_shape,
        /// Specifies a sizes attribute to an op.
        sizes = dnnl_graph_op_attr_sizes,
        /// Specifies a strides attribute to an op.
        strides = dnnl_graph_op_attr_strides,
        /// Specifies a zps attribute to an op.
        zps = dnnl_graph_op_attr_zps,

        // bool attributes. The value of these attributes can be any single bool
        // value.

        /// Specifies an exclude_pad attribute to an op.
        exclude_pad = dnnl_graph_op_attr_exclude_pad,
        /// Specifies a keep_dims attribute to an op.
        keep_dims = dnnl_graph_op_attr_keep_dims,
        /// Specifies a keep_stats attribute to an op.
        keep_stats = dnnl_graph_op_attr_keep_stats,
        /// Specifies a per_channel_broadcast attribute to an op.
        per_channel_broadcast = dnnl_graph_op_attr_per_channel_broadcast,
        /// Specifies a special_zero attribute to an op.
        special_zero = dnnl_graph_op_attr_special_zero,
        /// Specifies a transpose_a attribute to an op.
        transpose_a = dnnl_graph_op_attr_transpose_a,
        /// Specifies a transpose_b attribute to an op.
        transpose_b = dnnl_graph_op_attr_transpose_b,
        /// Specifies an use_affine attribute to an op.
        use_affine = dnnl_graph_op_attr_use_affine,
        /// Specifies an use_dst attribute to an op.
        use_dst = dnnl_graph_op_attr_use_dst,

        // string attributes. The value of these attributes can be a string.

        /// Specifies an auto_broadcast attribute to an op. The value can be
        /// "none" or "numpy".
        auto_broadcast = dnnl_graph_op_attr_auto_broadcast,
        /// Specifies an auto_pad attribute to an op. The value can be "none",
        /// "same_upper", "same_lower", or "valid".
        auto_pad = dnnl_graph_op_attr_auto_pad,
        /// Specifies an coordinate_transformation_mode attribute to an op. The
        /// value can be "half_pixel" or "align_corners". The attribute is
        /// defined for Interpolate operations.
        coordinate_transformation_mode
        = dnnl_graph_op_attr_coordinate_transformation_mode,
        /// Specifies a data_format of an op. The value can be "NCX" or "NXC".
        data_format = dnnl_graph_op_attr_data_format,
        /// Specifies a filter_format of an op. The value can be "OIX" or "XIO".
        filter_format = dnnl_graph_op_attr_filter_format,
        /// Specifies a mode attribute of an op. The value can be "nearest",
        /// "linear", "bilinear", or "trilinear". The attribute is defined for
        /// Interpolate operations.
        mode = dnnl_graph_op_attr_mode,
        /// Specifies a qtype attribute to an op. The value can be "per_channel"
        /// or "per_tensor". The attribute is defined for quantization
        /// operations.
        qtype = dnnl_graph_op_attr_qtype,
        /// Specifies a rounding_type attribute to an op. The value can be
        /// "ceil" or "floor".
        rounding_type = dnnl_graph_op_attr_rounding_type,
    };

    /// Constructs an op object with an unique ID, an operation kind, and a name
    /// string.
    ///
    /// @param id The unique ID of the op.
    /// @param akind The op kind specifies which computation is represented by
    ///     the op, such as Convolution or ReLU.
    /// @param verbose_name The string added as the op name.
    op(size_t id, kind akind, const std::string &verbose_name) {
        dnnl_graph_op_t op = nullptr;
        error::check_succeed(dnnl_graph_op_create(&op, id, convert_to_c(akind),
                                     verbose_name.c_str()),
                "could not create op with id and op kind");
        reset(op);
    }

    /// Constructs an op object with an unique ID, an operation kind, and
    /// input/output logical tensors.
    ///
    /// @param id The unique ID of this op.
    /// @param akind The op kind specifies which computation is represented by
    ///     this op, such as Convolution or ReLU.
    /// @param inputs Input logical tensor to be bound to this op.
    /// @param outputs Output logical tensor to be bound to this op.
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

    /// Adds an input logical tensor to the op.
    ///
    /// @param t Input logical tensor.
    void add_input(const logical_tensor &t) {
        error::check_succeed(dnnl_graph_op_add_input(get(), &(t.data)),
                "adding input to the op failed");
    }

    /// Adds a vector of input logical tensors to the op.
    ///
    /// @param ts The list of input logical tensors.
    void add_inputs(const std::vector<logical_tensor> &ts) {
        for (const auto &t : ts) {
            error::check_succeed(dnnl_graph_op_add_input(get(), &(t.data)),
                    "adding input to the op failed");
        }
    }

    /// Adds an output logical tensor to the op.
    ///
    /// @param t Output logical tensor.
    void add_output(const logical_tensor &t) {
        error::check_succeed(dnnl_graph_op_add_output(get(), &(t.data)),
                "adding output to the op failed");
    }

    /// Adds a vector of output logical tensors to the op.
    ///
    /// @param ts The list of output logical tensors.
    void add_outputs(const std::vector<logical_tensor> &ts) {
        for (const auto &t : ts) {
            error::check_succeed(dnnl_graph_op_add_output(get(), &(t.data)),
                    "adding output to the op failed");
        }
    }

    /// Sets the attribute according to the name and type (int64_t). This API is
    /// deprecated. Please use the version accepting `op::attr` as the attribute
    /// name.
    ///
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
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

    /// Sets the attribute according to the name and type (int64_t).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type,
            requires<std::is_same<Type, int64_t>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::check_succeed(dnnl_graph_op_set_attr_s64(get(), attr, &value, 0),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (float). This API is
    /// deprecated. Please use the version accepting `op::attr` as the attribute
    /// name.
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type, requires<std::is_same<Type, float>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        error::check_succeed(dnnl_graph_op_set_attr_f32(get(), attr, &value, 0),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (float).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type, requires<std::is_same<Type, float>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::check_succeed(dnnl_graph_op_set_attr_f32(get(), attr, &value, 0),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (bool). This API is
    /// deprecated. Please use the version accepting `op::attr` as the attribute
    /// name.
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type, requires<std::is_same<Type, bool>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        const uint8_t val = value;
        error::check_succeed(dnnl_graph_op_set_attr_bool(get(), attr, &val, 0),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (bool).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type, requires<std::is_same<Type, bool>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        const uint8_t val = value;
        error::check_succeed(dnnl_graph_op_set_attr_bool(get(), attr, &val, 0),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (string). This API is
    /// deprecated. Please use the version accepting `op::attr` as the attribute
    /// name.
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type,
            requires<std::is_same<Type, std::string>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        error::check_succeed(dnnl_graph_op_set_attr_str(
                                     get(), attr, value.c_str(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (string).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
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
    /// (std::vector<int64_t>). This API is deprecated. Please use the version
    /// accepting `op::attr` as the attribute name.
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type,
            requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        error::check_succeed(dnnl_graph_op_set_attr_s64(
                                     get(), attr, value.data(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type
    /// (std::vector<int64_t>).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type,
            requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
    op &set_attr(attr name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(name);
        error::check_succeed(dnnl_graph_op_set_attr_s64(
                                     get(), attr, value.data(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (std::vector<float>).
    /// This API is deprecated. Please use the version accepting `op::attr` as
    /// the attribute name.
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
    template <typename Type,
            requires<std::is_same<Type, std::vector<float>>::value> = true>
    op &set_attr(const std::string &name, const Type &value) {
        dnnl_graph_op_attr_t attr = convert_to_c(str2attr(name));
        error::check_succeed(dnnl_graph_op_set_attr_f32(
                                     get(), attr, value.data(), value.size()),
                "could not set attribute to the op");
        return *this;
    }

    /// Sets the attribute according to the name and type (std::vector<float>).
    ///
    /// @tparam Type Attribute's type.
    /// @param name Attribute's name.
    /// @param value The attribute's value.
    /// @returns The Op self.
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
/// Partition represents a collection of operations and their input and output
/// logical tensors identified by library as the basic unit for compilation and
/// execution.
///
/// @{

/// A partition object.
class partition : public detail::partition_handle {
public:
    /// An alias to context
    using compilation_context = context;

    /// Policy specifications for partitioning.
    enum class policy {
        /// Max policy is to be defined. The library intends to deliver best
        /// optimization and larger partition with max policy. It also means
        /// users may lose fine-grained control the operations in the partition.
        /// Currently, max policy has the same effect as fusion policy.
        max = dnnl_graph_partition_policy_max,
        /// Fusion policy returns partitions with typical post-op fusions, eg.
        /// Convolution + ReLU or other element-wise operations or a chian of
        /// post-ops.
        fusion = dnnl_graph_partition_policy_fusion,
        /// Debug policy doesn't not apply any fusions. It returns partitions
        /// with single operations in each partition. The policy is useful when
        /// users notice any bug or correctness issue in max policy or fusion
        /// policy.
        debug = dnnl_graph_partition_policy_debug,
    };

    /// Partition kind. It defines the basic structure of the subgraph contained
    /// in a partition. For example, kind
    /// #dnnl::graph::partition::kind::convolution_post_ops indicates the
    /// partition contains one Convolution and its post-ops. But the operation
    /// kind of the post-ops are not specified. Partition's kind is decided by
    /// the library internally and can be queried from a partition.
    enum class kind {
        /// The partition's kind is not defined.
        undef = dnnl_graph_partition_kind_undef,
        /// The partition contains a Convolution and its post-ops.
        convolution_post_ops = dnnl_graph_partition_kind_convolution_post_ops,
        /// The partition contains a ConvTranspose and its post-ops.
        convtranspose_post_ops
        = dnnl_graph_partition_kind_convtranspose_post_ops,
        /// The partition contains an Interpolate and its post-ops.
        interpolate_post_ops = dnnl_graph_partition_kind_interpolate_post_ops,
        /// The partition contains a MatMul and its post-ops.
        matmul_post_ops = dnnl_graph_partition_kind_matmul_post_ops,
        /// The partition contains a Reduction and its post-ops.
        reduction_post_ops = dnnl_graph_partition_kind_reduction_post_ops,
        /// The partition contains an Unary op and its post-ops.
        unary_post_ops = dnnl_graph_partition_kind_unary_post_ops,
        /// The partition contains a Binary op and its post-ops.
        binary_post_ops = dnnl_graph_partition_kind_binary_post_ops,
        /// The partition contains a Pooling op (AvgPool or MaxPool) and its
        /// post-ops.
        pooling_post_ops = dnnl_graph_partition_kind_pooling_post_ops,
        /// The partition contains a BatchNorm op and its post-ops.
        batch_norm_post_ops = dnnl_graph_partition_kind_batch_norm_post_ops,
        /// Other partitions based on post-ops but not specified by above kinds.
        misc_post_ops = dnnl_graph_partition_kind_misc_post_ops,
        /// The partition contains a quantized version of Convolution and its
        /// post-ops.
        quantized_convolution_post_ops
        = dnnl_graph_partition_kind_quantized_convolution_post_ops,
        /// The partition contains a quantized version of ConvTranspose and its
        /// post-ops.
        quantized_convtranspose_post_ops
        = dnnl_graph_partition_kind_quantized_convtranspose_post_ops,
        /// The partition contains a quantized version of MatMul and its
        /// post-ops.
        quantized_matmul_post_ops
        = dnnl_graph_partition_kind_quantized_matmul_post_ops,
        /// The partition contains a quantized version of Unary op and its
        /// post-ops.
        quantized_unary_post_ops
        = dnnl_graph_partition_kind_quantized_unary_post_ops,
        /// The partition contains a quantized version of Pooling op and its
        /// post-ops.
        quantized_pooling_post_ops
        = dnnl_graph_partition_kind_quantized_pooling_post_ops,
        /// Other partitions based quantization and post-ops but not specified
        /// by above kinds.
        misc_quantized_post_ops
        = dnnl_graph_partition_kind_misc_quantized_post_ops,
        /// The partition contains a Convolution backward op and its post-ops.
        convolution_backprop_post_ops
        = dnnl_graph_partition_kind_convolution_backprop_post_ops,
        /// The partition contains a variant of Multi-head Attention.
        mha = dnnl_graph_partition_kind_mha,
        /// The partition contains a variant of Multi-layer Perceptron.
        mlp = dnnl_graph_partition_kind_mlp,
        /// The partition contains a variant of quantized MHA.
        quantized_mha = dnnl_graph_partition_kind_quantized_mha,
        /// The partition contains a variant of quantized MLP.
        quantized_mlp = dnnl_graph_partition_kind_quantized_mlp,
        /// The partition contains a variant of residual convolutional block.
        residual_conv_blocks = dnnl_graph_partition_kind_residual_conv_blocks,
        /// The partition contains a variant of quantized version of residual
        /// convolutional block.
        quantized_residual_conv_blocks
        = dnnl_graph_partition_kind_quantized_residual_conv_blocks,
    };

    /// Default constructor. Constructs an empty partition object.
    partition() = default;

    /// Constructs a partition object for C data.
    ///
    /// @param p A raw pointer to the C API handle
    partition(dnnl_graph_partition_t p) { reset(p, false); }

    /// Creates a new partition with a given operator and engine kind. The API
    /// is used to create a partition from an operation directly without
    /// creating the graph and calling `get_partitions()`. The output partition
    /// contains only one operation.
    ///
    /// @param aop An operation used to create the partition.
    /// @param ekind Engine kind.
    partition(const op &aop, engine::kind ekind) {
        dnnl_graph_partition_t p = nullptr;
        error::check_succeed(
                dnnl_graph_partition_create_with_op(&p, aop.get(),
                        static_cast<dnnl_graph_engine_kind_t>(ekind)),
                "could not create a partition with the op and engine kind");
        reset(p);
    }

    /// Returns the number of operations contained in the partition.
    ///
    /// @returns Number of operations.
    size_t get_ops_num() const {
        size_t num {0};
        error::check_succeed(dnnl_graph_partition_get_op_num(get(), &num),
                "could not get number of ops from the partition");
        return num;
    }

    /// Returns all operation IDs contained in the partition.
    ///
    /// @returns An unordered set of operation IDs.
    std::vector<size_t> get_ops() const {
        auto num = get_ops_num();
        std::vector<size_t> ops(num);

        error::check_succeed(
                dnnl_graph_partition_get_ops(get(), num, ops.data()),
                "could not get op ids from the partition");
        return ops;
    }

    /// Returns the unique ID of the partition. Partition ID is generated by the
    /// library internally. The ID can be used for debugging purpose or verbose.
    ///
    /// @returns ID of the partition.
    size_t get_id() const {
        size_t id {};
        error::check_succeed(dnnl_graph_partition_get_id(get(), &id),
                "could not get id of the partition");
        return id;
    }

    /// Compiles a partition with given input and output logical tensors. The
    /// output logical tensors can contain unknown dimensions. For this case,
    /// the compilation will deduce the output shapes according to input shapes.
    /// The output logical tensors can also have layout type `any`. The
    /// compilation will choose the optimal layout for output tensors. The
    /// optimal layout will be represented as an opaque layout ID saved in the
    /// output logical tensor. Context may contain more information
    /// useful for compiling a partition. The library can still succeed to
    /// compile if a context is empty.
    ///
    /// @param inputs A list of input logical tensors.
    /// @param outputs A list of output logical tensors.
    /// @param e The engine used to compile the partition.
    /// @param ctx Context information used to compile the partition
    /// @returns A compiled partition.
    compiled_partition compile(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine &e,
            const compilation_context &ctx = {}) const {
        if (!is_supported()) {
            error::check_succeed(dnnl_graph_invalid_arguments,
                    "could not compile an unsupported partition");
        }

        return compile_(inputs, outputs, &e, &ctx);
    }

    /// Returns the supporting status of a partition. Some operations may not be
    /// supported by the library under certain circumstances. During
    /// partitioning stage, unsupported partitions will be returned to users
    /// with each containing an unsupported operation. Users should check the
    /// supporting status of a partition before transforming the computation
    /// graph or compiling the partition.
    ///
    /// @returns @c true if this partition is supported or @c false if this
    ///     partition isn't supported by the library
    bool is_supported() const {
        uint8_t supported {0};
        error::check_succeed(
                dnnl_graph_partition_is_supported(get(), &supported),
                "could not get supporting status of the partition");
        return supported != 0;
    }

    /// Returns a list of input logical tensors from the partition.
    ///
    /// @returns A list of input logical tensors.
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

    /// Returns a list of output logical tensors from the partition.
    ///
    /// @returns A list of output logical tensor.
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

    /// Returns the engine kind of the partition.
    ///
    /// @returns The engine kind.
    engine::kind get_engine_kind() const {
        dnnl_graph_engine_kind_t akind;
        error::check_succeed(
                dnnl_graph_partition_get_engine_kind(get(), &akind),
                "could not get the engine kind from the partition");

        return static_cast<engine::kind>(akind);
    }

    /// Returns the kind of the partition.
    ///
    /// @returns The partition kind.
    kind get_kind() const {
        dnnl_graph_partition_kind_t pkind = dnnl_graph_partition_kind_undef;
        error::check_succeed(dnnl_graph_partition_get_kind(get(), &pkind),
                "could not get the kind of the partition");

        return static_cast<kind>(pkind);
    }

private:
    compiled_partition compile_(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine *e,
            const compilation_context *ctx) const {
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
                dnnl_graph_partition_compile_v2(get(), cpartitions,
                        c_inputs.size(), c_inputs.data(), c_outputs.size(),
                        c_outputs.data(), e->get(), ctx->get()),
                "could not compile the partition");

        return compiled_partition(cpartitions);
    }
};

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_graph Graph
///
/// Graph represents a computational DAG with a set of operations.
/// #dnnl::graph::graph::add_op() adds an operation and its input and output
/// logical tensors into a graph. The library accumulates the operations and
/// logical tensors and constructs and validates the graph as an internal state.
/// A graph object is associated to a specific engine kind. The partitions
/// returned from the graph will inherit the engine kind of the graph.
///
/// @{

/// A graph object.
class graph : public detail::graph_handle {
public:
    /// floating-point math mode.
    enum class fpmath_mode {
        /// Default behavior, no downconversions allowed
        strict = dnnl_graph_fpmath_mode_strict,
        /// Implicit f32->bf16 or f32->tf32 conversions allowed
        bf16 = dnnl_graph_fpmath_mode_bf16,
        /// Implicit f32->f16 or f32->tf32 conversions allowed
        f16 = dnnl_graph_fpmath_mode_f16,
        /// Implicit f32->f16 or f32->bf16 or f32->tf32 conversions allowed
        any = dnnl_graph_fpmath_mode_any,
        /// Implicit f32->tf32 conversions allowed
        tf32 = dnnl_graph_fpmath_mode_tf32,
    };

    /// Constructs a graph with an engine kind.
    ///
    /// @param engine_kind Engine kind.
    graph(engine::kind engine_kind) {
        dnnl_graph_graph_t g = nullptr;
        error::check_succeed(
                dnnl_graph_graph_create(&g, engine::convert_to_c(engine_kind)),
                "could not create graph with engine kind");
        reset(g);
    }

    /// Creates a new empty graph with an engine kind and a floating-point math
    /// mode. All partitions returned from the graph will inherit the engine
    /// kind and floating-point math mode.
    ///
    /// @param engine_kind Engine kind.
    /// @param mode Floating-point math mode.
    graph(engine::kind engine_kind, fpmath_mode mode) {
        dnnl_graph_graph_t g = nullptr;
        error::check_succeed(
                dnnl_graph_graph_create_with_fpmath_mode(&g,
                        engine::convert_to_c(engine_kind), convert_to_c(mode)),
                "could not create graph with engine kind and math mode");
        reset(g);
    }

    /// Adds an op into the graph to construct a computational DAG. The API will
    /// return failure if the operator has already been added to the graph or
    /// the operation cannot pass the schema check in the library (eg. input and
    /// output numbers and data types, the attributes of the operation, etc.).
    ///
    /// @param op An operation to be added.
    /// @param allow_exception A flag indicating whether the method is allowed
    ///     to throw an exception if it fails to add the op to the graph.
    /// @returns #success or a status describing the error otherwise.
    status add_op(const op &op, bool allow_exception = true) {
        dnnl_graph_status_t ret = dnnl_graph_add_op(get(), op.get());

        if (allow_exception) {
            error::check_succeed(ret, "could not add op to the graph");
        }

        return static_cast<status>(ret);
    }

    /// Gets filtered partitions from a graph. Partitions will be claimed
    /// internally according to the capability of the library, the engine kind
    /// of the graph, and the policy.
    ///
    /// @param policy Partition policy, defaults to policy
    ///     #dnnl::graph::partition::policy::fusion.
    /// @return A vector storing the partitions.
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

        return out_list;
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

/// @addtogroup dnnl_graph_api_constant_tensor_cache Constant Tensor Cache
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

/// @} dnnl_graph_api_constant_tensor_cache

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

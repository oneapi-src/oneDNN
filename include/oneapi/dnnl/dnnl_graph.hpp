/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "oneapi/dnnl/dnnl_graph_base.hpp"
#include "oneapi/dnnl/dnnl_graph_detail.hpp"

#include <limits>
#include <string>
#include <utility>
#include <vector>

/// @addtogroup dnnl_graph_api oneDNN graph API
/// @{

namespace dnnl {
namespace graph {

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
        dnnl_graph_allocator_t *a {};
        error::check_succeed(
                dnnl_graph_allocator_create(&a, cpu_malloc, cpu_free),
                "could not create allocator for cpu");
        reset(a);
    }

    /// Default constructor
    allocator() {
        dnnl_graph_allocator_t *a {};
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
    /// @param device_id Specify which device to be used
    engine(kind akind, int device_id) {
        dnnl_graph_engine_t *e {};
        error::check_succeed(
                dnnl_graph_engine_create(&e, convert_to_c(akind), device_id),
                "could not create engine with engine kind and device id");
        reset(e);
    }

    /// Constructs an engine with specified kind and device_id
    ///
    /// @param akind Engine kind
    /// @param device_id Specify which device to be used
    /// @param alloc The memory allocator bound with engine
    engine(kind akind, int device_id, allocator &alloc) {
        dnnl_graph_engine_t *e {};
        error::check_succeed(
                dnnl_graph_engine_create(&e, convert_to_c(akind), device_id),
                "could not create engine with engine kind and device id");
        reset(e);

        set_allocator(alloc);
    }

    /// Set allocator to an engine
    ///
    /// @param alloc The memory allocator bound with engine
    void set_allocator(allocator &alloc) {
        error::check_succeed(
                dnnl_graph_engine_set_allocator(get(), alloc.get()),
                "could not set allocator to the engine");
    }

    /// Returns device handle of the current engine
    ///
    /// @returns Device handle
    void *get_device_handle() const {
        void *handle {};
        error::check_succeed(
                dnnl_graph_engine_get_device_handle(get(), &handle),
                "could not get device handle from the engine");
        return handle;
    }

    /// Returns device id of the current engine
    ///
    /// @returns Device id
    int get_device_id() const {
        int32_t device_id {};
        error::check_succeed(dnnl_graph_engine_get_device_id(get(), &device_id),
                "could not get device id from the engine");
        return device_id;
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

/// @addtogroup dnnl_graph_api_threadpool Thread pool
/// Definitions of thread pool
/// @{

/// Abstract threadpool interface
class thread_pool : public detail::thread_pool_handle {
public:
    /// Constructs a threadpool object
    ///
    /// @param num_threads Number of threads in the thread pool
    thread_pool(int32_t num_threads) {
        dnnl_graph_thread_pool_t *tp {};
        error::check_succeed(dnnl_graph_thread_pool_create(&tp, num_threads),
                "could not create thread pool");
        reset(tp);
    };
};

/// @} dnnl_graph_api_threadpool

/// @addtogroup dnnl_graph_api_stream_attr Stream attr
/// Definitions of stream attributes
/// @note Will be deprecated soon
/// @{

class stream_attr : public detail::stream_attr_handle {
public:
    /// Constructs stream attributes
    ///
    /// @param pool A thread pool bound to this stream attribute
    stream_attr(thread_pool &pool) {
        dnnl_graph_stream_attr_t *sa {};
        error::check_succeed(dnnl_graph_stream_attr_create(&sa, pool.get()),
                "could not create stream attributes");
        reset(sa);
    };

    /// Returns the threadpool attribute.
    ///
    /// @returns A threadpool object bound to the stream
    thread_pool get_thread_pool();
};

/// @} dnnl_graph_api_stream_attr

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
    /// @param attr A stream attribute, defaults to nullptr
    stream(engine &engine, const stream_attr *attr = nullptr) {
        dnnl_graph_stream_t *s {};
        error::check_succeed(
                [&] {
                    if (attr) {
                        return dnnl_graph_stream_create_with_attr(
                                &s, engine.get(), attr->get());
                    };
                    return dnnl_graph_stream_create(&s, engine.get());
                }(),
                "could not create stream");
        reset(s);
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
    friend class conversion;
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
    logical_tensor(
            size_t tid, data_type dtype, int32_t ndims, layout_type ltype) {
        dnnl_graph_logical_tensor_t val;
        error::check_succeed(
                dnnl_graph_logical_tensor_init(&val, tid, convert_to_c(dtype),
                        ndims, convert_to_c(ltype)),
                "could not create logical_tensor");
        data = val;
    }

    /// Delegated construtor
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
    logical_tensor(size_t tid, data_type dtype, const dims_t &adims,
            layout_type ltype) {
        dnnl_graph_logical_tensor_t val;
        // if dimension size equals to 0, it's a scalar
        if (adims.size() == 0)
            error::check_succeed(
                    dnnl_graph_logical_tensor_init(&val, tid,
                            convert_to_c(dtype), 0, convert_to_c(ltype)),
                    "could not create logical_tensor");
        else
            error::check_succeed(dnnl_graph_logical_tensor_init_with_dims(&val,
                                         tid, convert_to_c(dtype),
                                         static_cast<int32_t>(adims.size()),
                                         adims.data(), convert_to_c(ltype)),
                    "could not create logical_tensor with dims");
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
    logical_tensor(size_t tid, data_type dtype, const dims_t &adims,
            const dims_t &strides) {
        dnnl_graph_logical_tensor_t val;
        // TODO(lvtao): check the size of adims and strides.
        // They should be same.
        error::check_succeed(
                dnnl_graph_logical_tensor_init_with_strides(&val, tid,
                        convert_to_c(dtype), static_cast<int32_t>(adims.size()),
                        adims.data(), strides.data()),
                "could not create logical_tensor with strides");
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
    logical_tensor(
            size_t tid, data_type dtype, const dims_t &adims, int64_t lid) {
        dnnl_graph_logical_tensor_t val;

        if (adims.size() == 0) {
            error::check_succeed(dnnl_graph_logical_tensor_init(&val, tid,
                                         convert_to_c(dtype), 0,
                                         convert_to_c(layout_type::opaque)),
                    "could not create logical_tensor");
        } else {
            error::check_succeed(
                    dnnl_graph_logical_tensor_init_with_dims(&val, tid,
                            convert_to_c(dtype),
                            static_cast<int32_t>(adims.size()), adims.data(),
                            convert_to_c(layout_type::opaque)),
                    "could not create logical_tensor with dims");
        }

        val.layout.layout_id = lid;
        data = val;
    }

    /// Returns dimensions of the logical tensor
    ///
    /// @returns A the dimensions vector
    dims_t get_dims() const { return {data.dims, data.dims + data.ndims}; }

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

    /// Returns layout type of the logical tensor
    ///
    /// @returns The layout type
    layout_type get_layout_type() const {
        return static_cast<layout_type>(data.layout_type);
    }

    /// Returns the layout of the tensor
    ///
    /// @returns Layout id
    int64_t get_layout_id() const {
        // TODO(lvtao): exceptions on API level should be more unified
        if (get_layout_type() != layout_type::opaque) {
            throw std::runtime_error {"Error: layout type is not opaque."};
        }

        return data.layout.layout_id;
    }

    /// Returns strides of this logical tensor
    ///
    /// @returns A copy of strides vector
    dims_t get_strides() const {
        // TODO(lvtao): exceptions on API level should be more unified
        if (get_layout_type() != layout_type::strided) {
            throw std::runtime_error {"Error: layout type is not strided."};
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
    bool has_same_layout_and_dtype(const logical_tensor &lt) {
        uint8_t is_same {0};
        error::check_succeed(
                dnnl_graph_logical_tenosr_has_same_layout_and_dtype(
                        &data, &lt.data, &is_same),
                "could not compare the layout between these logical tensors");
        return static_cast<bool>(is_same);
    }

private:
    static dnnl_graph_data_type_t convert_to_c(data_type dtype) {
        return static_cast<dnnl_graph_data_type_t>(dtype);
    }

    static dnnl_graph_layout_type_t convert_to_c(layout_type ltype) {
        return static_cast<dnnl_graph_layout_type_t>(ltype);
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

    /// Constructs a tensor object according to the given logical tensor
    ///
    /// @param lt The given logical tensor
    /// @param handle Handle of memory buffer to use as an underlying storage,
    ///     if the ndims in the logical tensor is 0, data handle holds a scalar
    tensor(const logical_tensor &lt, void *handle) {
        dnnl_graph_tensor_t *t {};
        error::check_succeed(dnnl_graph_tensor_create_with_logical_tensor(
                                     &t, &(lt.data), handle),
                "could not create tensor with the logical_tensor");
        reset(t);
    }

    /// Returns the underlying memory buffer with the specific type
    ///
    /// @tparam T Type of the request buffer
    /// @returns The underlying memory buffer
    template <typename T>
    typename std::add_pointer<T>::type get_data_handle() const {
        void *data_handle {};
        error::check_succeed(dnnl_graph_tensor_get_if_type(
                                     get(), get_data_type<T>(), &data_handle),
                "could not get data handle from the tensor");
        return reinterpret_cast<typename std::add_pointer<T>::type>(
                data_handle);
    }

    /// Sets the underlying memory buffer
    ///
    /// @param handle Data handle. For the CPU engine, the data handle
    ///     is a pointer to the actual data.
    void set_data_handle(void *handle) {
        error::check_succeed(dnnl_graph_tensor_set_data_handle(get(), handle),
                "setting data handlt to the tensor failed");
    }

    /// Returns the number of elements in the tensor
    ///
    /// @returns Number of element
    int64_t get_element_num() const {
        int64_t num {};
        error::check_succeed(dnnl_graph_tensor_get_element_num(get(), &num),
                "could not get number of elements in the tensor");
        return num;
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
    compiled_partition(dnnl_graph_compiled_partition_t *compiled_partition) {
        reset(compiled_partition, false);
    }

    /// Returns the logical tensor according to tensor id
    ///
    /// @param tid The unique id of required tensor
    /// @returns The logical tensor
    logical_tensor query_logical_tensor(uint64_t tid) {
        dnnl_graph_logical_tensor_t lt;
        error::check_succeed(dnnl_graph_compiled_partition_query_logical_tensor(
                                     get(), tid, &lt),
                "query logical tensor from compiled_partition failed");
        return logical_tensor {lt};
    }

    /// Returns the in-place options
    ///
    /// @note
    ///     Each entry of the returned in-place options is a pair that maps from
    ///     input to output, indicating possible reused memory buffers in an
    ///     operator.
    ///
    /// @returns List of pairs of that indicates input and output use same
    ///     memory buffer.
    std::vector<std::pair<size_t, size_t>> get_inplace_options() const {
        size_t num_inplace_pairs = std::numeric_limits<size_t>::max();
        const dnnl_graph_inplace_pair_t *inplace_pairs;

        error::check_succeed(dnnl_graph_compiled_partition_get_inplace_pairs(
                                     get(), &num_inplace_pairs, &inplace_pairs),
                "could not get the in-place pairs from a compiled partition");
        if (num_inplace_pairs == 0) return {};
        std::vector<std::pair<size_t, size_t>> inplace_options;
        inplace_options.reserve(num_inplace_pairs);
        for (size_t i = 0; i < num_inplace_pairs; ++i) {
            const dnnl_graph_inplace_pair_t *inplace_pair = inplace_pairs + i;
            inplace_options.emplace_back(
                    inplace_pair->input, inplace_pair->output);
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
        std::vector<const dnnl_graph_tensor_t *> c_inputs;
        c_inputs.reserve(inputs.size());
        for (auto &in : inputs) {
            c_inputs.push_back(in.get());
        }
        std::vector<const dnnl_graph_tensor_t *> c_outputs;
        c_outputs.reserve(outputs.size());
        for (auto &out : outputs) {
            c_outputs.push_back(out.get());
        }

        error::check_succeed(
                dnnl_graph_compiled_partition_execute(get(), astream.get(),
                        static_cast<uint64_t>(c_inputs.size()), c_inputs.data(),
                        static_cast<uint64_t>(c_outputs.size()),
                        c_outputs.data()),
                "could not execute the compiled_partition");
    }
};

/// @} dnnl_graph_api_compiled_partition

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
    partition(dnnl_graph_partition_t *p) { reset(p, false); }

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
    std::vector<size_t> get_ops() {
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
    /// may have already been changed according to the fwk fused node.
    ///
    /// @param inputs A list of input logical tensors
    /// @param outputs A list of output logical tensors
    /// @param e The engine used to compile the partition
    /// @returns A compiled partition
    compiled_partition compile(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine &e) const {
        return compile_(inputs, outputs, &e);
    }

    /// Infer the partition's output shape
    ///
    /// @param inputs A list of input logical tensors
    /// @param outputs A list of output logical tensors
    void infer_shape(const std::vector<logical_tensor> &inputs,
            std::vector<logical_tensor> &outputs) const {
        std::vector<const dnnl_graph_logical_tensor_t *> c_inputs;
        std::vector<dnnl_graph_logical_tensor_t *> c_outputs;

        c_inputs.reserve(inputs.size());
        for (auto &&in : inputs) {
            c_inputs.push_back(&(in.data));
        }

        c_outputs.reserve(outputs.size());
        for (auto &&out : outputs) {
            c_outputs.push_back(&(out.data));
        }

        error::check_succeed(
                dnnl_graph_partition_infer_shape(get(),
                        static_cast<uint64_t>(c_inputs.size()), c_inputs.data(),
                        static_cast<uint64_t>(c_outputs.size()),
                        c_outputs.data()),
                "inferring output shape of the partition failed");
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

        dnnl_graph_compiled_partition_t *cpartitions;
        error::check_succeed(
                dnnl_graph_compiled_partition_create(&cpartitions, get()),
                "could not create compiled_partition");
        error::check_succeed(
                dnnl_graph_partition_compile(get(), cpartitions,
                        static_cast<uint64_t>(c_inputs.size()), c_inputs.data(),
                        static_cast<uint64_t>(c_outputs.size()),
                        c_outputs.data(), e->get()),
                "partition compile failed");

        return compiled_partition(cpartitions);
    }
};

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_conversion Data conversion API
///
/// `conversion` is considered as a special partition which only accepts two
/// input arguments.
///
/// @{

/// Data conversion API
class conversion : public detail::partition_handle {
public:
    /// Default constructor
    conversion() {
        dnnl_graph_partition_t *c_partition {nullptr};
        error::check_succeed(dnnl_graph_partition_create(&c_partition),
                "could not create a converison partition");
        reset(c_partition);
    }

    /// Compile the conversion partition to generate compiled partition based
    /// on the input/output logical tensor.
    ///
    /// @param input Input logical tensors
    /// @param output Output logical tensors
    /// @param aengine The engine used to compile the partition
    /// @returns A compiled partition
    compiled_partition compile(const logical_tensor &input,
            const logical_tensor &output, const engine &aengine) {
        std::vector<const dnnl_graph_logical_tensor_t *> c_inputs {&input.data};
        std::vector<const dnnl_graph_logical_tensor_t *> c_outputs {
                &output.data};

        error::check_succeed(
                dnnl_graph_conversion_init(get(), &input.data, &output.data,
                        static_cast<dnnl_graph_engine_kind_t>(
                                aengine.get_kind())),
                "could not initialize a converison");

        dnnl_graph_compiled_partition_t *cpartition {nullptr};
        error::check_succeed(
                dnnl_graph_compiled_partition_create(&cpartition, get()),
                "could not create a compiled partition");

        error::check_succeed(
                dnnl_graph_partition_compile(get(), cpartition,
                        static_cast<uint64_t>(c_inputs.size()), c_inputs.data(),
                        static_cast<uint64_t>(c_outputs.size()),
                        c_outputs.data(), aengine.get()),
                "could not compile a conversion");

        return compiled_partition(cpartition);
    }
};

/// @} dnnl_graph_api_conversion

/// @addtogroup dnnl_graph_api_op Op
///
/// OP is an abstraction of compute logic for deep neural network operation.
///
/// @{

/// A op contains kind, attribute, and the input and output logical tensor(s).
class op : public detail::op_handle {
public:
    enum class kind {
        Abs = kAbs,
        Add = kAdd,
        AvgPool = kAvgPool,
        AvgPoolBackprop = kAvgPoolBackprop,
        BatchNormInference = kBatchNormInference,
        BatchNormForwardTraining = kBatchNormForwardTraining,
        BatchNormTrainingBackprop = kBatchNormTrainingBackprop,
        BiasAddBackprop = kBiasAddBackprop,
        Clamp = kClamp,
        ClampBackprop = kClampBackprop,
        Concat = kConcat,
        Convolution = kConvolution,
        ConvolutionBackpropData = kConvolutionBackpropData,
        ConvolutionBackpropFilters = kConvolutionBackpropFilters,
        Divide = kDivide,
        Elu = kElu,
        EluBackprop = kEluBackprop,
        Erf = kErf,
        Exp = kExp,
        GELU = kGELU,
        GELUBackprop = kGELUBackprop,
        HardTanh = kHardTanh,
        HardTanhBackprop = kHardTanhBackprop,
        LayerNorm = kLayerNorm,
        LayerNormBackprop = kLayerNormBackprop,
        Log = kLog,
        LogSoftmax = kLogSoftmax,
        LogSoftmaxBackprop = kLogSoftmaxBackprop,
        MatMul = kMatMul,
        Maximum = kMaximum,
        MaxPool = kMaxPool,
        MaxPoolBackprop = kMaxPoolBackprop,
        Minimum = kMinimum,
        Multiply = kMultiply,
        Pow = kPow,
        PowBackprop = kPowBackprop,
        ReduceSum = kReduceSum,
        ReLU = kReLU,
        ReLUBackprop = kReLUBackprop,
        Reshape = kReshape,
        Round = kRound,
        Sigmoid = kSigmoid,
        SigmoidBackprop = kSigmoidBackprop,
        SoftMax = kSoftMax,
        SoftMaxBackprop = kSoftMaxBackprop,
        SoftPlus = kSoftPlus,
        SoftPlusBackprop = kSoftPlusBackprop,
        Sqrt = kSqrt,
        SqrtBackprop = kSqrtBackprop,
        Square = kSquare,
        Tanh = kTanh,
        TanhBackprop = kTanhBackprop,
        Wildcard = kWildcard,
        BiasAdd = kBiasAdd,
        Interpolate = kInterpolate,
        Transpose = kTranspose,
        Index = kIndex,
        InterpolateBackprop = kInterpolateBackprop,
        PowBackpropExponent = kPowBackpropExponent,
        LastSymbol = kLastSymbol,
    };

    /// Constructs an OP object
    ///
    /// @param id The unique id of this op
    /// @param akind The op kind specifies which computation is represented by
    ///     the op, such as Convolution and ReLU.
    /// @param debug_string The string added for debug
    op(size_t id, kind akind, const std::string &debug_string) {
        dnnl_graph_op_t *op {};
        error::check_succeed(dnnl_graph_op_create(&op, id, convert_to_c(akind),
                                     debug_string.c_str()),
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
    /// @param debug_string The string added for debug
    op(size_t id, kind akind, const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs,
            const std::string &debug_string)
        : op(id, akind, debug_string) {
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
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, int64_t>::value> = true>
    op &set_attr(const std::string &name, const Type &a) {
        constexpr auto kind = attr_kind<Type>();
        error::check_succeed(
                dnnl_graph_op_add_attr(get(), name.c_str(), kind, &a, 0),
                "setting attribute to the op failed");
        return *this;
    }

    /// Sets the attribute according to the name and type (float)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type, requires<std::is_same<Type, float>::value> = true>
    op &set_attr(const std::string &name, const Type &a) {
        constexpr auto kind = attr_kind<Type>();
        error::check_succeed(
                dnnl_graph_op_add_attr(get(), name.c_str(), kind, &a, 0),
                "setting attribute to the op failed");
        return *this;
    }

    /// Sets the attribute according to the name and type (bool)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type, requires<std::is_same<Type, bool>::value> = true>
    op &set_attr(const std::string &name, const Type &a) {
        constexpr auto kind = attr_kind<Type>();
        error::check_succeed(
                dnnl_graph_op_add_attr(get(), name.c_str(), kind, &a, 0),
                "setting attribute to the op failed");
        return *this;
    }

    /// Sets the attribute according to the name and type (string)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::string>::value> = true>
    op &set_attr(const std::string &name, const Type &a) {
        constexpr auto kind = attr_kind<Type>();
        error::check_succeed(
                dnnl_graph_op_add_attr(get(), name.c_str(), kind, a.c_str(), 0),
                "setting attribute to the op failed");
        return *this;
    }

    /// Sets the attribute according to the name and type
    /// (std::vector<int64_t>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
    op &set_attr(const std::string &name, const Type &a) {
        constexpr auto kind = attr_kind<Type>();
        error::check_succeed(dnnl_graph_op_add_attr(get(), name.c_str(), kind,
                                     a.data(), static_cast<int64_t>(a.size())),
                "setting attribute to the op failed");
        return *this;
    }

    /// Sets the attribute according to the name and type
    /// (std::vector<float>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::vector<float>>::value> = true>
    op &set_attr(const std::string &name, const Type &a) {
        constexpr auto kind = attr_kind<Type>();
        error::check_succeed(dnnl_graph_op_add_attr(get(), name.c_str(), kind,
                                     a.data(), static_cast<int64_t>(a.size())),
                "setting attribute to the op failed");
        return *this;
    }

    // TODO(lvtao): Consider if this method is needed and how to support

    /// Returns the string format of the Op id and kind
    ///
    /// @returns Op kind in string format
    std::string to_string() const {
        size_t id;
        error::check_succeed(
                dnnl_graph_op_get_id(get(), &id), "cannot get the operator id");

        dnnl_graph_op_kind_t akind;
        error::check_succeed(dnnl_graph_op_get_kind(get(), &akind),
                "cannot get the operator kind");

        return std::to_string(id) + " "
                + detail::op_kind_strings.at(
                        static_cast<std::vector<int64_t>::size_type>(akind));
    }

private:
    dnnl_graph_op_kind_t convert_to_c(kind akind) {
        return static_cast<dnnl_graph_op_kind_t>(akind);
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
};

/// @} dnnl_graph_api_op

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
    /// Constructs a graph session using device information
    ///
    /// @param engine_kind Can be cpu, gpu or any supported engine.
    graph(engine::kind engine_kind) {
        dnnl_graph_graph_t *g {};
        error::check_succeed(
                dnnl_graph_graph_create(&g, engine::convert_to_c(engine_kind)),
                "could not create graph with engine kind and device id");
        reset(g);
    };

    /// Add an op to the graph session to construct DAG for analysis
    ///
    /// @param op An operator/node that represents the entry of frameworks'
    ///    graph
    void add_op(const op &op) {
        error::check_succeed(dnnl_graph_add_op(get(), op.get()),
                "adding op to the graph failed");
    }

    /// Vector to store the partitions
    using partition_vec = std::vector<partition>;
    /// Get filtered partitions
    ///
    /// @param policy Partition policy, defaults to
    ///     #dnnl::graph::partition::policy::fusion
    /// @return partition_vec A vector storing the partitions
    partition_vec get_partitions(
            partition::policy policy = partition::policy::fusion) {
        error::check_succeed(
                dnnl_graph_graph_filter(get(),
                        static_cast<dnnl_graph_partition_policy_t>(policy)),
                "filter graph failed");

        uint64_t partitions_no {0};
        error::check_succeed(
                dnnl_graph_graph_get_partition_num(get(), &partitions_no),
                "could not get number of partitions from the graph");

        partition_vec out_list;
        out_list.reserve(partitions_no);

        std::vector<dnnl_graph_partition_t *> partitions(partitions_no);
        for (auto &p : partitions) {
            error::check_succeed(dnnl_graph_partition_create(&p),
                    "could not create partition");
        }
        dnnl_graph_graph_get_partitions(
                get(), partitions_no, partitions.data());

        for (auto p : partitions) {
            out_list.emplace_back(p);
        }

        return out_list;
    }
};

/// @} dnnl_graph_api_graph

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

/// For llga compatibility
namespace llga {
namespace api = ::dnnl::graph;
} // namespace llga

/// @} dnnl_graph_api

#endif

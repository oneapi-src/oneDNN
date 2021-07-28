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

/// @addtogroup dnnl_graph_api_engine Engine
///
/// Engine represents a device and its context. Compiled partitions are
/// associated with engines. A compiled partition should only access the tensor
/// which is associated with the same device and context, no matter the tensor
/// is produced by a compiled partition or created directly by the user.

/// @{

/// An engine contains device #kind and a device_id or device_handle.
class engine {
public:
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
    engine(kind akind, int device_id);

    /// Returns device handle of the current engine
    ///
    /// @returns Device handle
    void *get_device_handle() const;

    /// Returns device id of the current engine
    ///
    /// @returns Device id
    int get_device_id() const;

    /// Returns concrete kind of the current engine
    ///
    ///@returns Kind of engine
    kind get_kind() const;
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
class stream {
public:
    /// Constructs a stream for the specified engine
    ///
    /// @param engine Engine to create stream on
    /// @param attr A stream attribute, defaults to nullptr
    stream(engine &engine, const stream_attr *attr = nullptr);

    /// Waits for all compiled partitions executing in the stream to finish.
    /// @returns The stream itself.
    stream &wait();
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
    dnnl_graph_logical_tensor_t data;

public:
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

    /// default constructor
    /// construct an empty object
    logical_tensor() = default;

    /// Constructs a logical tensor object
    explicit logical_tensor(const dnnl_graph_logical_tensor_t &c_data);

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
            size_t tid, data_type dtype, int32_t ndims, layout_type ltype);

    /// Delegated construtor
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param ltype Layout type
    logical_tensor(
            size_t tid, data_type dtype, layout_type ltype = layout_type::undef);

    /// Constructs a logical tensor object
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param adims Tensor dimensions, -1 means a particular axis of dims is
    ///        unknown, or the axis can be deduced by its size and other axis.
    /// @param ltype Layout type
    logical_tensor(size_t tid, data_type dtype, const dims_t &adims,
            layout_type ltype);

    /// Constructs a logical tensor object
    ///
    /// @note The layout_type for this constructor will always be strided
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param adims Tensor dimensions, -1 means a particular axis of dims is
    /// @param strides Tensor strides
    logical_tensor(size_t tid, data_type dtype, const dims_t &adims,
            const dims_t &strides);

    /// Constructs a logical tensor object
    ///
    /// @note The layout_type for this constructor will always be opaque
    ///
    /// @param tid Tensor id
    /// @param dtype Data type
    /// @param adims Tensor dimensions, -1 means a particular axis of dims is
    /// @param lid Layout id
    logical_tensor(
            size_t tid, data_type dtype, const dims_t &adims, int64_t lid);

    /// Returns dimensions of the logical tensor
    ///
    /// @returns A the dimensions vector
    dims_t get_dims() const;

    /// Returns unique id of the logical tensor
    ///
    /// @returns Id number
    size_t get_id() const;

    /// Returns data type of the logical tensor
    ///
    /// @returns The data type
    data_type get_data_type() const;

    /// Returns layout type of the logical tensor
    ///
    /// @returns The layout type
    layout_type get_layout_type() const;

    /// Returns the layout of the tensor
    ///
    /// @returns Layout id
    int64_t get_layout_id() const;

    /// Returns strides of this logical tensor
    ///
    /// @returns A copy of strides vector
    dims_t get_strides() const;

    /// Get memory size required by this logical tensor
    ///
    /// @returns The memory size in bytes
    size_t get_mem_size() const;

    /// Compares if this and input logical tensor has the same layout
    ///
    /// @param lt The input logical tensor to be compared
    /// @returns @c true if they have the same layout
    ///        @c false if they have different layout
    bool has_same_layout(const logical_tensor &lt) const;
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
class tensor {
public:
    /// Default constructor. Constructs an empty object.
    tensor() = default;

    /// Constructs a tensor object according to the given logical tensor
    ///
    /// @param lt The given logical tensor
    /// @param handle Handle of memory buffer to use as an underlying storage,
    ///     if the ndims in the logical tensor is 0, data handle holds a scalar
    tensor(const logical_tensor &lt, void *handle);

    /// Returns the underlying memory buffer with the specific type
    ///
    /// @tparam T Type of the request buffer
    /// @returns The underlying memory buffer
    template <typename T>
    typename std::add_pointer<T>::type get_data_handle() const;

    /// Sets the underlying memory buffer
    ///
    /// @param handle Data handle. For the CPU engine, the data handle
    ///     is a pointer to the actual data.
    void set_data_handle(void *handle);

    /// Returns the number of elements in the tensor
    ///
    /// @returns Number of element
    int64_t get_element_num() const;
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
class compiled_partition {
public:
    /// Default constructor. Constructs an empty object.
    compiled_partition() = default;

    /// Constructs a compiled partition object
    compiled_partition(dnnl_graph_compiled_partition_t *compiled_partition);

    /// Returns the logical tensor according to tensor id
    ///
    /// @param tid The unique id of required tensor
    /// @returns The logical tensor
    logical_tensor query_logical_tensor(size_t tid) const;

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
    std::vector<std::pair<size_t, size_t>> get_inplace_ports() const;

    /// Execute a compiled partition
    ///
    /// @param astream Stream object to run over
    /// @param inputs A list of input tensors in the partition
    /// @param outputs A list of output tensors in the partition
    void execute(stream &astream, const std::vector<tensor> &inputs,
            const std::vector<tensor> &outputs) const;
};

/// @} dnnl_graph_api_compiled_partition

/// @addtogroup dnnl_graph_api_partition Partition
///
/// Partition represents a collection of OPs identified by oneDNN graph
/// implementation as the basic unit for compilation and execution.
///
/// @{

/// A partition contains a list of OP ids.
class partition {
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
    partition(dnnl_graph_partition_t *p);

    /// Constructs a partition with a given op and engine kind
    ///
    /// @param aop An operator used to create the partition
    /// @param ekind Engine kind
    partition(const op &aop, engine::kind ekind);

    /// Returns the number of dnnl graph ops in the partition
    ///
    /// @returns Number of ops
    size_t get_ops_num() const;

    /// Returns all op’s id of the partition
    ///
    /// @returns An unordered set of op ids
    std::vector<size_t> get_ops();

    /// Returns the unique id of the partition
    ///
    /// @returns Unique id
    size_t get_id() const;

    /// Compile the partition to generate compiled partition based
    /// on the input/output logical tensors. The order of these two lists
    /// may have already been changed according to the fwk fused node.
    ///
    /// @param inputs A list of input logical tensors
    /// @param outputs A list of output logical tensors
    /// @param e The engine used to compile the partition
    /// @returns A compiled partition
    compiled_partition compile(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine &e) const;

    /// Infer the partition's output shape
    ///
    /// @param inputs A list of input logical tensors
    /// @param outputs A list of output logical tensors
    void infer_shape(const std::vector<logical_tensor> &inputs,
            std::vector<logical_tensor> &outputs) const;

    /// Returns the supporting status of the partition
    ///
    /// @returns @c true if this partition is supported by oneDNN Graph backend
    ///     @c false if this partition isn't supported by oneDNN Graph backend
    bool is_supported() const;

    /// Returns a list of input logical tensors from the partition
    ///
    /// @returns A list of input logical tensors
    std::vector<logical_tensor> get_in_ports() const;

    /// Returns a list of output logical tensors from the partition
    ///
    /// @returns A list of output logical tensor
    std::vector<logical_tensor> get_out_ports() const;
};

/// @} dnnl_graph_api_partition

/// @addtogroup dnnl_graph_api_op Op
///
/// OP is an abstraction of compute logic for deep neural network operation.
///
/// @{

/// A op contains kind, attribute, and the input and output logical tensor(s).
class op {
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
        End = kEnd,
        Quantize = kQuantize,
        Dequantize = kDequantize,
        Reorder = kReorder,
        LastSymbol = kLastSymbol,
    };

    /// Constructs an OP object
    ///
    /// @param id The unique id of this op
    /// @param akind The op kind specifies which computation is represented by
    ///     the op, such as Convolution and ReLU.
    /// @param debug_string The string added for debug
    op(size_t id, kind akind, const std::string &debug_string);

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
            const std::string &debug_string);

    /// Adds input logical tensor to the op
    ///
    /// @param t Input logical tensor
    void add_input(const logical_tensor &t);

    /// Adds input logical tensors to the op
    ///
    /// @param ts The list of input logical tensors
    void add_inputs(const std::vector<logical_tensor> &ts);

    /// Adds output logical tensor to the op
    ///
    /// @param t Output logical tensor
    void add_output(const logical_tensor &t);

    /// Adds output logical tensors to the op
    ///
    /// @param ts The list of output logical tensors
    void add_outputs(const std::vector<logical_tensor> &ts);

    /// Sets the attribute according to the name and type (int64_t)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, int64_t>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type (float)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type, requires<std::is_same<Type, float>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type (bool)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type, requires<std::is_same<Type, bool>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type (string)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::string>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type
    /// (std::vector<int64_t>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    /// Sets the attribute according to the name and type
    /// (std::vector<float>)
    ///
    /// @tparam Type Attribute's type
    /// @param name Attribute's name
    /// @param a The attribute's value
    /// @returns The Op self
    template <typename Type,
            requires<std::is_same<Type, std::vector<float>>::value> = true>
    op &set_attr(const std::string &name, const Type &a);

    // TODO(lvtao): Consider if this method is needed and how to support

    /// Returns the string format of the Op id and kind
    ///
    /// @returns Op kind in string format
    std::string to_string() const;
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
class graph {
public:
    /// Constructs a graph session using device information
    ///
    /// @param engine_kind Can be cpu, gpu or any supported engine.
    graph(engine::kind engine_kind);

    /// Add an op to the graph session to construct DAG for analysis
    ///
    /// @param op An operator/node that represents the entry of frameworks'
    ///    graph
    void add_op(const op &op);

    /// Vector to store the partitions
    using partition_vec = std::vector<partition>;
    /// Get filtered partitions
    ///
    /// @param policy Partition policy, defaults to
    ///     #dnnl::graph::partition::policy::fusion
    /// @return partition_vec A vector storing the partitions
    partition_vec get_partitions(
            partition::policy policy = partition::policy::fusion);
};

/// @} dnnl_graph_api_graph

} // namespace graph
} // namespace dnnl

/// @} dnnl_graph_api

#endif

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

class logical_tensor {
dnnl_graph_logical_tensor_t data;

/// Constructs a logical tensor object 
logical_tensor(const dnnl_graph_logical_tensor_t &c_data);

/// Constructs a logical tensor object
///
/// @param tid Tensor id
/// @param dtype Data type
/// @param ndims Number of dimension, -1 means it's unknown, 0 means scalar
/// @param ltype Layout type
logical_tensor(size_t tid, data_type dtype, int32_t ndims, layout_type ltype);
 
/// Constructs a logical tensor object
///
/// @param tid Tensor id
/// @param dtype Data type
/// @param adims Tensor dimensions, -1 means a particular axis of dims is
///        unknown, or the axis can be deduced by its size and other axis.
/// @param ltype Layout type
logical_tensor(size_t tid, data_type dtype, const dims_t &adims,
                layout_type ltype);
 
/// Constructs a strided logical tensor object which accepts strides
///
/// @param tid Tensor id
/// @param dtype Data type
/// @param adims Tensor dimensions, -1 means a particular axis of dims is
///        unknown, or the axis can be deduced by its size and other axis.
/// @param strides Tensor strides
logical_tensor(size_t tid, data_type dtype, const dims_t &adims,
                const dims_t &strides);
 
/// Constructs an opaque logical tensor object which accepts layout id
///
/// @param tid Tensor id
/// @param dtype Data type
/// @param adims Tensor dimensions, -1 means a particular axis of dims is
///        unknown, or the axis can be deduced by its size and other axis.
/// @param lid Layout id
logical_tensor(size_t tid, data_type dtype, const dims_t &adims, size_t lid);

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

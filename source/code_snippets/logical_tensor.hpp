/// layout type for a logical tensor
///
/// “any” layout means that oneDNN Graph implementation to decide 
/// the layout for the compiled partition. 
/// “strided” layout_type means that the layout is determined 
/// by the strided field. 
/// “OPAQUE” means that the layout is a target-specific layout
/// decided by oneDNN Graph implementation. 

enum class layout_type {
        undef = llga_layout_type_undef,
        any = llga_layout_type_any,
        strided = llga_layout_type_strided,
        opaque = llga_layout_type_opaque,
};

class logical_tensor {
/// Constructs a logical tensor object
///
/// @param tid Tensor id
/// @param dtype Data type
/// @param ndims Number of dimension, -1 means it's unknown, 0 means scalar
/// @param ltype Layout type
logical_tensor(size_t tid, data_type dtype, int32_t ndims, layout_type ltype)
 
/// Constructs a logical tensor object
///
/// @param tid Tensor id
/// @param adims Tensor dimensions, -1 means a particular axis of dims is
///        unknown, or the axis can be deduced by its size and other axis.
/// @param dtype Data type
/// @param ltype Layout type
logical_tensor(size_t tid, const dims_t &adims, data_type dtype,
        layout_type ltype)
 
/// Constructs a strided logical tensor object which accepts strides
///
/// @param tid Tensor id
/// @param adims Tensor dimensions, -1 means a particular axis of dims is
///        unknown, or the axis can be deduced by its size and other axis.
/// @param strides Tensor strides
/// @param dtype Data type
logical_tensor(size_t tid, const dims_t &adims, const dims_t &strides,
        data_type dtype)
 
/// Constructs an opaque logical tensor object which accepts layout id
///
/// @param tid Tensor id
/// @param adims Tensor dimensions, -1 means a particular axis of dims is
///        unknown, or the axis can be deduced by its size and other axis.
/// @param lid Layout id
/// @param dtype Data type
logical_tensor(size_t tid, const dims_t &adims, size_t lid,
        data_type dtype)
 
/// Returns dimensions of the logical tensor
///
/// @returns A the dimensions vector
dims_t get_dims() const
 
/// Returns unique id of the logical tensor
///
/// @returns Id number
size_t get_id() const
 
/// Returns data type of the logical tensor
///
/// @returns The data type
data_type get_data_type() const
 
/// Returns layout type of the logical tensor
///
/// @returns The layout type
layout_type get_layout_type() const
 
/// Returns the layout of the tensor
///
/// @returns Layout id
size_t get_layout_id() const
 
/// Returns strides of this logical tensor
///
/// @returns A copy of strides vector
dims_t get_strides() const
 
/// Get memory size required by this logical tensor
///
/// @returns The memory size in bytes
size_t get_mem_size() const
};

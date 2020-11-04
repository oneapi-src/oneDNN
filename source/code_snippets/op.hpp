class op {
/// Constructs an OP object
///
/// @param id The unique id of this op
/// @param akind The op kind specifies which computation is represented by
///     the op, such as Convolution and ReLU.
/// @param debug_string The string added for debug
op(size_t id, kind akind, const std::string &debug_string)
 
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
 
/// Adds input logical tensor to the op
///
/// @param t Input logical tensor
void add_input(const logical_tensor &t)
 
/// Adds input logical tensors to the op
///
/// @param ts The list of input logical tensors
void add_inputs(const std::vector<logical_tensor> &ts)
 
/// Adds output logical tensor to the op
///
/// @param t Output logical tensor
void add_output(const logical_tensor &t)
 
/// Adds output logical tensors to the op
///
/// @param t The list of output logical tensors
void add_outputs(const std::vector<logical_tensor> &ts)
 
/// Returns the number of input logical tensor in the Op
///
/// @returns Number of inputs
uint64_t get_inputs_num() const
 
/// Returns the number of output logical tensor in the Op
///
/// @returns Number of outputs
uint64_t get_outputs_num() const
 
/// Sets the attribute according to the name and type (int64_t)
///
/// @tparam Type Attribute's type
/// @param name Attribute's name
/// @param a The attribute's value
/// @returns The Op self
template <typename Type,
        requires<std::is_same<Type, int64_t>::value> = true>
op &set_attr(const std::string &name, const Type &a)
 
/// Sets the attribute according to the name and type (float)
///
/// @tparam Type Attribute's type
/// @param name Attribute's name
/// @param a The attribute's value
/// @returns The Op self
template <typename Type, requires<std::is_same<Type, float>::value> = true>
op &set_attr(const std::string &name, const Type &a)
 
/// Sets the attribute according to the name and type (bool)
///
/// @tparam Type Attribute's type
/// @param name Attribute's name
/// @param a The attribute's value
/// @returns The Op self
template <typename Type, requires<std::is_same<Type, bool>::value> = true>
op &set_attr(const std::string &name, const Type &a)
 
/// Sets the attribute according to the name and type (string)
///
/// @tparam Type Attribute's type
/// @param name Attribute's name
/// @param a The attribute's value
/// @returns The Op self
template <typename Type,
        requires<std::is_same<Type, std::string>::value> = true>
op &set_attr(const std::string &name, const Type &a)
 
/// Sets the attribute according to the name and type
/// (std::vector<int64_t>)
///
/// @tparam Type Attribute's type
/// @param name Attribute's name
/// @param a The attribute's value
/// @returns The Op self
template <typename Type,
        requires<std::is_same<Type, std::vector<int64_t>>::value> = true>
op &set_attr(const std::string &name, const Type &a)
 
/// Sets the attribute according to the name and type
/// (std::vector<float>)
///
/// @tparam Type Attribute's type
/// @param name Attribute's name
/// @param a The attribute's value
/// @returns The Op self
template <typename Type,
        requires<std::is_same<Type, std::vector<float>>::value> = true>
op &set_attr(const std::string &name, const Type &a)
 
/// Returns the unique id of the Op
///
/// @returns Unique id
uint64_t get_id() const
 
/// Returns the concrete kind of this op
///
/// @returns kind The op kind
kind get_kind() const
 
/// Returns the string format of the Op id and kind
///
/// @returns Op id and kind in string format
std::string to_string() const
};

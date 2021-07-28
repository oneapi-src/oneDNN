class partition {
public:

/// Constructs a partition with a given op and engine kind
///
/// @param aop An operator used to create the partition
/// @param ekind Engine kind
partition(const op &aop, engine::kind ekind);

/// Returns the number of ops in the partition
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

/// Returns the kind of engine used to create the partition
///
/// @returns The engine kind
engine::kind get_engine_kind() const;

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
 
/// Infer the shape of outputs
///
/// @param inputs A list of input logical tensors
/// @param outputs A list of output logical tensors
void infer_shape(const std::vector<logical_tensor> &inputs, 
                std::vector<logical_tensor> &outputs);

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

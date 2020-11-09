class partition {
public:
/// Returns the number of ops in the partition
///
/// @returns Number of ops
uint64_t get_ops_num() const
 
/// Returns all op’s id of the partition
///
/// @returns An unordered set of op ids
std::vector<uint64_t> get_ops()
 
/// Returns the unique id of the partition
///
/// @returns Unique id
uint64_t get_id() const
 
/// Compile the partition to generate compiled partition based
/// on the input/output logical tensors. The order of these two lists
/// may have already been changed according to the fwk fused node.
///
/// @param inputs A list of input logical tensors
/// @param outputs A list of output logical tensors
/// @param e The engine used to compile the partition
/// @returns A compiled partition
compiled_partition compile(const std::vector<logical_tensor> &inputs,
        const std::vector<logical_tensor> &outputs, const engine &e) const
 
/// Infer the shape of outputs
///
/// @param inputs A list of input logical tensors
/// @param outputs A list of output logical tensors
void infer_shape(const std::vector<logical_tensor> &inputs, 
                std::vector<logical_tensor> &outputs)
 
};

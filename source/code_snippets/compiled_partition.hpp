class compiled_partition {
public:
    /// Returns the logical tensor according to tensor id
    ///
    /// @param tid The unique id of required tensor
    /// @returns The logical tensor
    logical_tensor query_logical_tensor(uint64_t tid)
 
    /// Execute a compiled partition.
    ///
    /// @param astream The stream used for execution
    /// @param inputs a vector of input tensors
    /// @param outputs a non-empty vector of output tensors 
    void execute(stream &astream, const std::vector<tensor> &inputs, 
        const std::vector<tensor> &outputs); 
};

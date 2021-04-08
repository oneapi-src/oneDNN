class compiled_partition {
public:
    /// Returns the logical tensor according to tensor id
    ///
    /// @param tid The unique id of required tensor
    /// @returns The logical tensor
    logical_tensor query_logical_tensor(size_t tid) const;

    /// Execute a compiled partition
    ///
    /// @param astream Stream object to run over
    /// @param inputs A list of input tensors in the partition
    /// @param outputs A list of output tensors in the partition
    void execute(stream &astream, const std::vector<tensor> &inputs,
            const std::vector<tensor> &outputs) const;

    /// Returns the in-place ports
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

    /// Returns the source partition used to create the compiled partition
    ///
    /// @returns The source partition
    partition get_source_partition() const;
};

/// Executes a compiled partition in a specified stream and returns a SYCL
/// event.
///
/// @param c_partition Compiled partition to execute.
/// @param astream Stream object to run over
/// @param inputs Arguments map.
/// @param outputs Arguments map.
/// @param deps Optional vector with `cl::sycl::event` dependencies.
/// @returns Output event.
inline cl::sycl::event dnnl::graph::sycl_interop::execute(
        compiled_partition &c_partition, stream &astream,
        const std::vector<tensor> &inputs, std::vector<tensor> &outputs,
        const std::vector<cl::sycl::event> &deps = {});

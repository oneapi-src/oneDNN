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
/// @param allow_exception A flag indicating whether the method is allowed
///    to throw an exception if it fails to add the op to the graph.
/// @returns #success or a status describing the error otherwise.
status add_op(const op &op, bool allow_exception = true);

using partition_vec = std::vector<partition>;
/// Get filtered partitions
///
/// @param policy Partition policy, defaults to
///     #dnnl::graph::partition::policy::fusion
/// @return partition_vec A vector storing the partitions
partition_vec get_partitions(partition_policy policy = partition::policy::fusion);
};

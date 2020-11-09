class graph {
public:
/// Constructs a graph session using device information
///
/// @param engine_kind Can be cpu, gpu or any supported engine.
graph(engine::kind engine_kind)

/// Add an op to the graph session to construct DAG for analysis
///
/// @param op An operator/node that represents the entry of frameworks'
///    graph
void add_op(const op &op)

using partition_vec = std::vector<partition>;
/// Get filtered partitions
///
/// @param policy Partition policy
/// @return partition_vec A vector storing the partitions
partition_vec get_partitions(partition_policy policy)
};

/*Add the OP to graph
  Parameter anop - an op constructed with edge and attribute   
  information   
*/
bool select(op& anop, graph& agraph);

/*  Generates list of partitions from llga graph base on policy 
  source_partition - the llga OP list 
  Parameter policy - control the granularity of partition
  Return the list of partitions, or nullptr if no partition is available */
std::vector<Partition*> filter_partitions(PartitionPolicy policy, graph& agraph);

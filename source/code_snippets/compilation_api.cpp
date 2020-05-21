/* Compiles LLGA partition to backend executable. If it’s called at optimization pass, the tensor shape information can be got from OP inputs if available. 
Parameters
  partition - the subgraph includes nodes for optimization
  inputs - a list of input logical tensors of the partition
  engine - the engine where the compiled partition is associated
  c_partition - the output compiled partition
Return whether the compilation succeeds (LLGA_RESULT_SUCCESS) or not. */
 
llga_result compile(Partition* partition, 
                    std::vector<logical_tensor*>* inputs,
                    Engine* engine,
                    CompiledPartition** c_partition);


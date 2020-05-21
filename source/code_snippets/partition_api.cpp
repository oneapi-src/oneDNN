/*Select the first OP which LLGA backend choose to grow the graph
  Parameter seed - llga OP converted from FW OP   
  Return true if selected, otherwise false  */

bool select(OP* seed, Engine* engine);

/* Grow the predecessor OP in the FW dataflow graph
Parameters
  pred_op - the predecessor OP feed input tensor to n
  cur_op  - the current selected OP
  output_idx - the output tensor index in pred_op’s output list
  input_idx - the input tensor index in to_slot's input list
Return true if the pred_op is selected, otherwise false */ 

bool select_input(OP* pred_op, OP* cur_op, size_t output_idx, size_t input_idx, Engine* engine);

/* Grow the successor OP in the FW dataflow graph
Parameters
  succ_op - the successor OP feed input tensor to n
  cur_op  - the current selected OP
  output_idx - the output tensor index in cur_op’s output list
  input_idx - the input tensor index in succ_op's input list
Return true if the pred_op is selected, otherwise false */ 

bool select_output(OP* succ_op, OP* cur_op, size_t output_idx, size_t input_index, Engine* engine);

/*  Generates list of partitions from llga graph base on policy 
  source_partition - the llga OP list 
  Parameter policy - control the granularity of partition
  Return the list of partitions, or nullptr if no partition is available */
std::vector<Partition*> filter_partitions(PartitionPolicy policy, Engine* engine);

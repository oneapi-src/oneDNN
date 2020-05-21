Typedef pair<op_id, size_t> op_id_slot_map; 
class Partition { 
  std::vector<size_t> op_list_;
  std::vector<logical_tensor*> inputs_;
  std::vector<logical_tensor*> outputs_;   
  std::map<op_input_slot, partition_input_slot> ;
  std::map<op_output_slot, partition_output_slot> ; 
  size_t id_;      
};

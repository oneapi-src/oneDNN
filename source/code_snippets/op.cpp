struct op {
  // the computation kind, conv, relu, etc.
  Op_kind kind;
  // input and output tensors
  // if the op is quantized op, the inputs/outputs
  // include scale and zero-point tensors also
  std::vector<logical_tensor*> inputs;
  std::vector<logical_tensor*> outputs;
  // unique identifier
  size_t id;
  // attributes
  std::vector<attribute_value*> attributes;
};

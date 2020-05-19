struct logical_tensor {
  // data type of tensor elements, int8, bf16, fp16, fp32
  data_type type;
  // number of dimensions
  // -1 means unknown number of dimension
  // 0 means a scalar
  int ndims;
  // -1 means a particular axis of dims is unknown, or the
  // axis can be deduced by its size and other axis.
  std::vector<int64_t> dims;
};

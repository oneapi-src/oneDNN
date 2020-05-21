// TODO: save the conversion cost from FW tensor to LLGA tensor
struct tensor {
  // a logical tensor that describes the tensor
  logical_tensor tensor_desc;
  // data type of tensor elements, bf16, int8, fp16, fp32
  data_type type;
  bool is_opaque;
  // strided format, valid !is_opaque
  std::vector<int64_t> strides;
  // if ndims in tensor_desc is 0, data_handle holds a scalar
  void *data_handle;
};

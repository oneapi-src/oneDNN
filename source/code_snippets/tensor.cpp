// TODO: save the conversion cost from FW tensor to LLGA tensor
struct tensor {
	// a logical tensor that describes the tensor
	logical_tensor tensor_desc;
	
  // if ndims in tensor_desc is 0, data_handle holds a scalar
	void *data_handle;
};


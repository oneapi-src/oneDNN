// Plain layout is numpy-compatible strided layout
#define LLGA_PLAIN_LAYOUT 0
// Layout id other than LLGA_PLAIN_LAYOUT is opaque to the DL
// frameworks. The framework integration code calls the layout
// query API to get the preferred ids, calls the layout
// conversion API to convert tensor layout when layout ids of
// producer and consumer do not match, and sets the layout ids to
// the input and output tensors of the partition during
// compilation.
typedef llga_layout_id int;

/* Query supported tensor layouts of a partition.
Parameters
  partition - the LLGA partition to query the layouts from
  engine - the engine where the compiled partition is associated
Return a list of layout lists, where the size of the layout list equals the number of input/output tensors of the partition and the list is arranged in the same order of the input/output tensors. The returned list is ordered from the most preferred layouts to least. Empty layout list or empty list of layouts corresponds to plain layout for all tensors.
NOTE: LLGA partition should always support plain format.
*/
std::vector<std::vector<llga_layout_id>>
query_layout(Partition* partition, Engine* engine);

/* Query supported tensor layouts of a partition with corresponding computation cost.
Parameters
  partition - the LLGA partition to query the layouts from
  engine - the engine where the compiled partition is associated
Return a list of pairs: (layout list, computation cost) where the size of the layout list equals the number of input/output tensors of the partition and the list is arranged in the same order of the input/output tensors. Empty layout list corresponds to plain layout for all tensors. The LLGA backend does not support this API when the pair list is empty. Usually it means the backend does not have a cost model.
NOTE: LLGA partition should always support plain format.
*/
std::vector<std::pair<std::vector<llga_layout_id>, float> >
query_layout_with_cost(Partition* partition, Engine* engine);

/* Query the conversion cost from a source tensor descriptor to a specified destination layout.
Parameters
  src_tensor_desc - the descriptor for the tensor to convert from
  dst_id - the layout id of the destination tensor
Return the conversion cost or NAN if not supported or unknown.
*/
float query_layout_conversion_cost(const logical_tensor& src_tensor_desc, llga_layout_id dst_id);

/* Convert the layout of a tensor to the specified layout id.
TODO: what’s the lifecycle of the returned tensor?
*/
tensor convert_layout(const tensor& src, llga_layout_id dst_id);


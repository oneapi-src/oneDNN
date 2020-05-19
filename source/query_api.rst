=========
Query API
=========

------------
Layout Query
------------

The tensor layout of DL framework defaults to plain layout (e.g. NCHW, NHWC, NC etc.) which is not always optimal therefore it is very likely an LLGA backend relies on opaque layouts (e.g. NCHW16C) inside the LLGA partitions. But this leads to overhead of layout conversion between plain layout and opaque layout at the boundary of LLGA partitions. LLGA provides a mechanism for LLGA partitions to consume and produce opaque layout directly, provided that DL framework supports tensors with opaque layout. This allows LLGA backend to focus on partitions with smaller fusion patterns without the overhead of layout conversion at the partition boundary. An LLGA partition can support a list of opaque layouts with different computation cost. Depending on how these partitions and framework ops are connected, there would be a choice of input/output layout combinations to get best performance. The layout query APIs serve this purpose. They are supposed to be called after partitions are built and the framework graph is updated with LLGA ops inserted. After preferred layout info is decided, the framework integration code sets the layout ids into the “logical_tensor” of the input and output of each partition for compilation.

LLGA provides two layout query APIs: 1) a basic query API that returns a list of supported layout combinations, ordered by their preferences; 2) an advanced query API that returns a list of “layouts to cost” mapping. The LLGA backend is required to support the basic query API and can optionally support the advanced one if it has a cost model, based on which the framework integration code can do better layout propagation that selects the optimal layouts for each LLGA partition for minimal global cost. For 2), LLGA also provides a conversion cost query API. Simply put, the global cost can be computed as follows:

.. math::

   GlobalCost = \Sigma PartitionCost + \Sigma LayoutConversionCost

LLGA assumes that all backends should support the plain layout. Layouts other than the plain layout are opaque to the frameworks. The framework integration code calls the layout query API to get the preferred layout ids, calls the layout conversion API to convert tensor layout when layout ids of producer and consumer do not match, and sets the layout ids to the input and output tensors of the partition during compilation.

**Discussion:** Lacking concrete tensor shape info would limit the capacity of LLGA backend to report accurate ranking or computation cost of supported layouts. The LLGA backend might even not know what layouts it could support. In the extreme case, the backend can choose to only support plain layouts when the shape info is unavailable or limited.

.. literalinclude:: code_snippets/query_api.cpp
    :language: cpp

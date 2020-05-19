class Engine {
public:
  // wrap a device managed by the framework
  Engine(void* device_handle, bool is_dpcpp);
  // create an Engine that owns a device handle by itself
  Engine(int engine_kind, int device_id);
  // either sycl::device or opaque device handle
  void* get_device_handle();
  // true if the engine wraps a DPCPP device
  bool is_dpcpp();
  // returns id corresponding to this device
  int device_id();
};

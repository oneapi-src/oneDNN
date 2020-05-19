class ThreadPool {
  // Total number of threads in the pool
  virtual int num_threads() = 0;
  // Used by the thread in the pool to know its id
  virtual int thread_id() = 0;
  // Used by the thread in the pool to know if it is running
  // in a parallel region
  virtual bool in_parallel() = 0;
  // Spawn threads and split tasks among threads
  // Parameters:
  //   begin - Begin of the range
  //   end - End of the range
  //   grain_size - Smallest unit of a task
  //   fn - Closure running each task ranging from task_begin and range_end
  virtual void parallel_for(int begin, int end, int grain_size,
                            const std::function<void(int, int)>& fn) = 0;
  // Spawn threads
  // Parameters
  //   fn - Closure run by each thread given task id per thread.
  virtual void parallel(const std::function<void(int)>& fn) = 0;
};

struct StreamAttr {
  void set_threadpool(ThreadPool *threadpool);
  ThreadPool* get_threadpool() const;
};

class Stream {
public:
  // wrap a stream managed by the framework
  Stream(void* stream_handle, Engine* engine);
  Stream(void* stream_handle, Engine* engine, const StreamAttr& attr);
  // either sycl::stream or opaque stream handle
  void* get_stream_handle();
  // true if the stream wraps a DPCPP stream
  bool is_dpcpp();
  // Execute a compiled partition
  // Parameters
  //   c_partition - compiled partition to run
  //   inputs - a list of input tensors of the partition
  //   outputs - a list of outputs tensors of the partition
  //   Return whether the execution succeeds (LLGA_RESULT_SUCCESS) or not.
  llga_result submit(CompiledPartition* c_partition,
                     std::vector<tensor*>* inputs,
                     std::vector<tensor*>* outputs);
  // An async version of the execution API will be added.
};


// Base class for allocator, inherited by FW allocator
struct Allocator {
  void* allocate_persistent(size_t n) = 0;
  void* allocate_temp(size_t n) = 0;
  void* allocate_output(size_t n) = 0;
  void deallocate_persistent(void* buf) = 0;
};
// Pass FW allocator to backend
void set_allocator(Allocator* allocator, Engine* engine);

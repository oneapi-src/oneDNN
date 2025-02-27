/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <atomic>
#include <cstddef>
#include <functional>

#include "async_value_ref.h"
#include "chain.h"

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

// namespace xla::cpu {

// Parallel loop runner uses underlying Eigen ThreadPoolDevice to execute
// parallel loops providing implicit synchronization: the next parallel loop
// starts execution only after all tasks from the previous loop are completed.
//
// Scheduled parallel loops execute asynchronously without blocking the caller
// thread. It is the user's responsibility to ensure that all values captured by
// the task are valid until the task is completed.
//
// Parallel loop runner is an implementation of the `pthreadpool` API adaptor
// for XLA:CPU runtime.
//
// ParallelLoopRunner uses "persistent workers" to execute parallel loops.
// Workers get scheduled into the underlying thread pool and when they start
// executing they pop tasks from the shared work queue. With this approach we
// avoid scheduling closures into the thread pool for each parallel task,
// because fixed thread pool overheads are high and XNNPACK operations tend to
// launch many parallel loops with larget number of very small tasks.
//
// Parallel loop runner can be configured by the `worker_timeslice` parameter,
// that defines the approximate amount of compute (in terms of wall time) that
// each persistent worker will handle. We rely on this parameter to avoid
// scheduling too many workers into the thread pool, because for tiny tasks the
// overheads can be prohibitively expensive.
//
// WARNING: ParallelLoopRunner is not thread-safe, and must be externally
// synchronized by the user.
class ParallelLoopRunner {
 public:
  explicit ParallelLoopRunner(const Eigen::ThreadPoolDevice* device);

  // Takes ownership of the runner and returns a done event. After the done
  // event is transferred to the caller, it is illegal to schedule more parallel
  // loops on the moved-from runner.
  static AsyncValueRef<Chain> TakeDoneEvent(
      ParallelLoopRunner&& runner);

  using Task1D = std::function<void(size_t offset)>;

  // This function implements a parallel version of a following loop:
  //
  //   for (size_t i = 0; i < range; i++)
  //     task(i);
  void Parallelize(size_t range, Task1D task);

  // Resets the parallel loop runner `done_event` and returns the previous one
  // to the caller.
  AsyncValueRef<Chain> ResetDoneEvent();

  AsyncValueRef<Chain> done_event() const { return done_event_; }

  const Eigen::ThreadPoolDevice* device() const { return device_; }
  void set_device(const Eigen::ThreadPoolDevice* device) { device_ = device; }

  // Returns the number of threads in the underlying thread pool.
  size_t num_threads() const;

  // Returns true if the current thread belongs to the underlying thread pool.
  bool is_in_runner() const;

 private:
  // Forward declarations of the parallel tasks.
  struct ParallelTask1D;

  // Schedules `task` as the AndThen callback of the `done_event_`. Updates
  // `done_event_` to the new completion event.
  template <typename Task>
  void ScheduleOne(Task&& task);

  // Schedules `num_tasks` invocation of the `parallel_task` into the Eigen
  // thread pool when the `done_event_` becomes available. Updates `done_event_`
  // to the new completion event.
  template <typename ParallelTask>
  void ScheduleAll(size_t num_tasks, ParallelTask&& parallel_task);

  // Async value that signals completion of the last scheduled parallel loop.
  AsyncValueRef<Chain> done_event_;

  // We keep a pointer to the Eigen thread pool device as an atomic variable
  // because we might update it between concurrent runs of XNNPACK operations
  // and non-atomic access to the `device_` pointer might lead to a data race.
  //
  // In practice PjRt CPU client owns the intra-op thread pool and passes it to
  // XLA via Thunk::ExecuteParams, and PjRt client might have multiple thread
  // pools for different NUMA nodes, and we have to be able to switch between
  // them from run to run.
  std::atomic<const Eigen::ThreadPoolDevice*> device_;
};

// }  // namespace xla::cpu
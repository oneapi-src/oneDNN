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

#include "parallel_loop_runner.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <iostream>

#include "work_queue.h"
#include "async_value_ref.h"
#include "chain.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

// namespace xla::cpu {

using Task = std::function<void(size_t task_index)>;

// Returns non-reference-counted async value ref in constructed state.
//
// Returned async value is a per-process singleton stored in a storage with a
// static duration, and can be safely compared using pointer equality.
static AsyncValueRef<Chain> OkDoneEventSingleton() {
  static AsyncValueOwningRef<Chain>* singleton = [] {
    auto* storage = new internal::AsyncValueStorage<Chain>();
    return new AsyncValueOwningRef<Chain>(
        MakeAvailableAsyncValueRef<Chain>(*storage));
  }();
  return singleton->AsRef();
}

ParallelLoopRunner::ParallelLoopRunner(const Eigen::ThreadPoolDevice* device)
    : done_event_(OkDoneEventSingleton()), device_(device) {}

AsyncValueRef<Chain> ParallelLoopRunner::ResetDoneEvent() {
  auto done_event = std::move(done_event_);
  done_event_ = OkDoneEventSingleton();
  return done_event;
}

size_t ParallelLoopRunner::num_threads() const {
  return device_.load()->numThreads();
}

bool ParallelLoopRunner::is_in_runner() const {
  return device_.load()->currentThreadId() > -1;
}

AsyncValueRef<Chain> ParallelLoopRunner::TakeDoneEvent(
    ParallelLoopRunner&& runner) {
  return std::move(runner.done_event_);
}

template <typename Task>
inline void ParallelLoopRunner::ScheduleOne(Task&& task) {
  auto event = MakeConstructedAsyncValueRef<Chain>();
  done_event_.AndThen([event, task = std::forward<Task>(task)] {
    task();
    event.SetStateConcrete();
  });
  done_event_ = std::move(event);
}

template <typename ParallelTask>
inline void ParallelLoopRunner::ScheduleAll(
    size_t num_tasks, ParallelTask&& parallel_task) {
  assert(num_tasks > 1 && "Expected at least two tasks");

  // Use at most `num_threads()` workers as we can't run more parallel workers
  // than the number of threads in the thread pool.
  size_t num_workers = std::min(std::min(num_tasks, num_threads()),
                                size_t{std::numeric_limits<uint16_t>::max()});

  CountDownAsyncValueRef<Chain> count_down(num_workers);
  auto count_down_done = count_down.AsRef();

  auto parallelize = [this, num_tasks, count_down = std::move(count_down),
                      parallel_task =
                          std::forward<ParallelTask>(parallel_task)] {
    Worker::Parallelize(device_, std::move(count_down), num_tasks,
                        std::move(parallel_task));
  };

  done_event_.AndThen(std::move(parallelize));
  done_event_ = std::move(count_down_done);
}

// In the `Parallelize` implementations below:
//
// (1) If done event is already available, execute the task immediately in the
//     caller thread. In this case we don't need to overwrite the done event,
//     because the existing one will correctly represent the state of the
//     parallel loop runner (all scheduled loops are ready).
//
// (2) If done event is not available, we have to overwrite it with a new one
//     that will be set to concrete state after the task is executed.
//
// We wrap all tasks into structs conforming to the `ParallelTest` API, so that
// in profiles we can see human-readable names of the tasks instead of lambdas.

struct ParallelLoopRunner::ParallelTask1D {
  inline void operator()(size_t task_index) const {
    task(task_index);
  }

  Task1D task;
};

void ParallelLoopRunner::Parallelize(size_t range, Task1D task) {
  assert(done_event_ && "Parallel loop runner is in moved-from state");
  assert(range > 0 && "Expected at least one task");
  // std::cout << "Parallelize(" << range << ")" << std::endl;
  // Fast path for the degenerate parallel loop with single task.
  if (range == 1) {
    // Execute task in the caller thread if done event is already available.
    if (done_event_.IsConcrete()) {
      task(0);
      return;
    }

    // Schedule task when done event becomes available.
    ScheduleOne([task = std::move(task)] { task(0); });
    return;
  }

  ScheduleAll(range, ParallelTask1D{std::move(task)});
}

// }  // namespace xla::cpu

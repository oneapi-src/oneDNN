/* Copyright 2022 Google LLC. All Rights Reserved.

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

#include "async_value.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <utility>
#include <vector>
#include <system_error>
#include <condition_variable>
#include <mutex>
#include <cassert>

#include "async_value_ref.h"
#include "ref_count.h"

// namespace tsl {

// This is a singly linked list of nodes waiting for notification, hanging off
// of AsyncValue.  When the value becomes available or if an error occurs, the
// callbacks are informed.
class NotifierListNode {
 public:
  explicit NotifierListNode(std::function<void()> notification)
      : next_(nullptr), notification_(std::move(notification)) {}

 private:
  friend class AsyncValue;
  // This is the next thing waiting on the AsyncValue.
  NotifierListNode* next_;
  std::function<void()> notification_;
};

uint16_t AsyncValue::CreateTypeInfoAndReturnTypeIdImpl(
    const TypeInfo& type_info) {
  size_t type_id = GetTypeInfoTableSingleton()->emplace_back(type_info) + 1;
  assert(type_id < std::numeric_limits<uint16_t>::max() && "Too many different AsyncValue types.");
  return type_id;
}

AsyncValue::TypeInfoTable* AsyncValue::GetTypeInfoTableSingleton() {
  constexpr int kInitialCapacity = 64;
  static auto* type_info_table = new TypeInfoTable(kInitialCapacity);
  return type_info_table;
}

std::atomic<size_t> AsyncValue::total_allocated_async_values_;

// This is called when the value is set into the ConcreteAsyncValue buffer, or
// when the IndirectAsyncValue is forwarded to an available AsyncValue, and we
// need to change our state and clear out the notifications. The current state
// must be unavailable (i.e. kUnconstructed or kConstructed).
void AsyncValue::NotifyAvailable(State available_state) {
  assert((kind() == Kind::kConcrete || kind() == Kind::kIndirect) && "Should only be used by ConcreteAsyncValue or IndirectAsyncValue");

  assert((available_state == State::kConcrete || available_state == State::kError) && "Invalid state");

  // Mark the value as available, ensuring that new queries for the state see
  // the value that got filled in.
  auto old_value = waiters_and_state_.exchange(
      WaitersAndState(nullptr, available_state), std::memory_order_acq_rel);
  assert((old_value.state() == State::kUnconstructed || old_value.state() == State::kConstructed) && "Invalid state");

  RunWaiters(old_value.waiter());
}

void AsyncValue::RunWaiters(NotifierListNode* list) {
  while (list) {
    NotifierListNode* node = list;
    // TODO(chky): pass state into notification_ so that waiters do not need to
    // check atomic state again.
    node->notification_();
    list = node->next_;
    delete node;
  }
}

// If the value is available or becomes available, this calls the closure
// immediately. Otherwise, the add closure to the waiter list where it will be
// called when the value becomes available.
void AsyncValue::EnqueueWaiter(std::function<void()> waiter,
                               WaitersAndState old_value) {
  // Create the node for our waiter.
  auto* node = new NotifierListNode(std::move(waiter));
  auto old_state = old_value.state();

  // Swap the next link in. old_value.state() must be unavailable when
  // evaluating the loop condition. The acquire barrier on the compare_exchange
  // ensures that prior changes to waiter list are visible here as we may call
  // RunWaiter() on it. The release barrier ensures that prior changes to *node
  // appear to happen before it's added to the list.
  node->next_ = old_value.waiter();
  auto new_value = WaitersAndState(node, old_state);
  while (!waiters_and_state_.compare_exchange_weak(old_value, new_value,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
    // While swapping in our waiter, the value could have become available.  If
    // so, just run the waiter.
    if (old_value.state() == State::kConcrete ||
        old_value.state() == State::kError) {
      assert(old_value.waiter() == nullptr && "Invalid waiter state");
      node->notification_();
      delete node;
      return;
    }
    // Update the waiter list in new_value.
    node->next_ = old_value.waiter();
  }

  // compare_exchange_weak succeeds. The old_value must be in either
  // kUnconstructed or kConstructed state.
  assert((old_value.state() == State::kUnconstructed || old_value.state() == State::kConstructed) && "Invalid state");
}

void AsyncValue::SetError(std::error_code error) {
  assert(error);
  if (kind() == Kind::kConcrete) {
    GetTypeInfo().set_error(this, std::move(error));
  } else {
    assert(kind() == Kind::kIndirect);
    auto error_av = MakeErrorAsyncValueRef(std::move(error));
    static_cast<IndirectAsyncValue*>(this)->ForwardTo(std::move(error_av));
  }
}

// Mark this IndirectAsyncValue as forwarding to the specified value.  This
// gives the IndirectAsyncValue a +1 reference.
void IndirectAsyncValue::ForwardTo(RCReference<AsyncValue> value) {
  assert(IsUnavailable() && "Value is already available");

  auto s = value->state();
  if (s == State::kConcrete || s == State::kError) {
    assert(!value_ && "IndirectAsyncValue::ForwardTo is called more than once");
    auto* concrete_value = value.release();
    if (concrete_value->kind() == Kind::kIndirect) {
      auto* indirect_value = static_cast<IndirectAsyncValue*>(concrete_value);
      concrete_value = indirect_value->value_;
      assert(concrete_value != nullptr);
      assert(concrete_value->kind() == Kind::kConcrete);
      concrete_value->AddRef();
      indirect_value->DropRef();
    }
    // If indirect async value was created for any particular type id, check
    // that forwarded to value has exactly the same type id or an error.
    assert((type_id_ == kUnknownTypeId || type_id_ == concrete_value->type_id_ || concrete_value->IsType<DummyValueForErrorAsyncValue>()) && "IndirectAsyncValue::ForwardTo value has an unexpected type id");
    value_ = concrete_value;
    type_id_ = concrete_value->type_id_;
    NotifyAvailable(s);
  } else {
    AsyncValue* av = value.get();
    av->AndThen([self = FormRef(this), value = std::move(value)]() mutable {
      self->ForwardTo(std::move(value));
    });
  }
}

//===----------------------------------------------------------------------===//
// Functions for awaiting on the async values.
//===----------------------------------------------------------------------===//

void BlockUntilReady(AsyncValue* async_value) {
  if (__builtin_expect(async_value->IsAvailable(), 1)) return;

  std::mutex mtx;
  std::condition_variable cv;
  bool ready = false;

  async_value->AndThen([&] {
    std::lock_guard<std::mutex> lock(mtx);
    ready = true;
    cv.notify_one();
  });

  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [&] { return ready; });
}

// }  // namespace tsl

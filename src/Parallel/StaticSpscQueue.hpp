/*
Copyright (c) 2020 Erik Rigtorp <erik@rigtorp.se>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>  // std::allocator
#include <new>     // std::hardware_destructive_interference_size
#include <stdexcept>
#include <type_traits>  // std::enable_if, std::is_*_constructible

namespace Parallel {

template <typename T, size_t Capacity>
class StaticSpscQueue {
 private:
#ifdef __cpp_lib_hardware_interference_size
  static constexpr size_t cache_line_size_ =
      std::hardware_destructive_interference_size;
#else
  static constexpr size_t cache_line_size_ = 64;
#endif

  // Padding to avoid false sharing between slots_ and adjacent allocations
  static constexpr size_t padding_ = (cache_line_size_ - 1) / sizeof(T) + 1;

 public:
  StaticSpscQueue() = default;
  ~StaticSpscQueue() {
    // Do I need this? Maybe?
    while (front()) {
      pop();
    }
  }

  // non-copyable and non-movable
  StaticSpscQueue(const StaticSpscQueue&) = delete;
  StaticSpscQueue& operator=(const StaticSpscQueue&) = delete;
  StaticSpscQueue(StaticSpscQueue&&) = delete;
  StaticSpscQueue& operator=(StaticSpscQueue&&) = delete;

  template <typename... Args>
  void emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible<T, Args&&...>::value) {
    static_assert(std::is_constructible<T, Args&&...>::value,
                  "T must be constructible with Args&&...");
    auto const write_index = write_index_.load(std::memory_order_relaxed);
    auto next_write_index = write_index + 1;
    if (next_write_index == capacity_) {
      next_write_index = 0;
    }
    while (next_write_index == read_index_cache_) {
      read_index_cache_ = read_index_.load(std::memory_order_acquire);
    }
    new (&data_[write_index + padding_]) T(std::forward<Args>(args)...);
    write_index_.store(next_write_index, std::memory_order_release);
  }

  template <typename... Args>
  [[nodiscard]] bool try_emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible<T, Args&&...>::value) {
    static_assert(std::is_constructible<T, Args&&...>::value,
                  "T must be constructible with Args&&...");
    auto const write_index = write_index_.load(std::memory_order_relaxed);
    auto next_write_index = write_index + 1;
    if (next_write_index == capacity_) {
      next_write_index = 0;
    }
    if (next_write_index == read_index_cache_) {
      read_index_cache_ = read_index_.load(std::memory_order_acquire);
      if (next_write_index == read_index_cache_) {
        return false;
      }
    }
    new (&data_[write_index + padding_]) T(std::forward<Args>(args)...);
    write_index_.store(next_write_index, std::memory_order_release);
    return true;
  }

  void push(const T& v) noexcept(std::is_nothrow_copy_constructible<T>::value) {
    static_assert(std::is_copy_constructible<T>::value,
                  "T must be copy constructible");
    emplace(v);
  }

  template <typename P, typename = typename std::enable_if<
                            std::is_constructible<T, P&&>::value>::type>
  void push(P&& v) noexcept(std::is_nothrow_constructible<T, P&&>::value) {
    emplace(std::forward<P>(v));
  }

  [[nodiscard]] bool try_push(const T& v) noexcept(
      std::is_nothrow_copy_constructible<T>::value) {
    static_assert(std::is_copy_constructible<T>::value,
                  "T must be copy constructible");
    return try_emplace(v);
  }

  template <typename P, typename = typename std::enable_if<
                            std::is_constructible<T, P&&>::value>::type>
  [[nodiscard]] bool try_push(P&& v) noexcept(
      std::is_nothrow_constructible<T, P&&>::value) {
    return try_emplace(std::forward<P>(v));
  }

  [[nodiscard]] T* front() noexcept {
    auto const read_index = read_index_.load(std::memory_order_relaxed);
    if (read_index == write_index_cache_) {
      write_index_cache_ = write_index_.load(std::memory_order_acquire);
      if (write_index_cache_ == read_index) {
        return nullptr;
      }
    }
    return &data_[read_index + padding_];
  }

  void pop() noexcept {
    static_assert(std::is_nothrow_destructible<T>::value,
                  "T must be nothrow destructible");
    auto const read_index = read_index_.load(std::memory_order_relaxed);
    assert(write_index_.load(std::memory_order_acquire) != read_index);
    data_[read_index + padding_].~T();
    auto next_read_index = read_index + 1;
    if (next_read_index == capacity_) {
      next_read_index = 0;
    }
    if (read_index == write_index_cache_) {
      write_index_cache_ = next_read_index;
    }
    read_index_.store(next_read_index, std::memory_order_release);
  }

  [[nodiscard]] size_t size() const noexcept {
    std::ptrdiff_t diff = write_index_.load(std::memory_order_acquire) -
                          read_index_.load(std::memory_order_acquire);
    if (diff < 0) {
      diff += capacity_;
    }
    return static_cast<size_t>(diff);
  }

  [[nodiscard]] bool empty() const noexcept {
    return write_index_.load(std::memory_order_acquire) ==
           read_index_.load(std::memory_order_acquire);
  }

  [[nodiscard]] size_t capacity() const noexcept { return capacity_ - 1; }

 private:
  static constexpr size_t capacity_ = Capacity;
  std::array<T, Capacity + 2 * padding_> data_{};

  // Align to cache line size in order to avoid false sharing
  // readIndexCache_ and writeIndexCache_ is used to reduce the amount of cache
  // coherency traffic
  alignas(cache_line_size_) std::atomic<size_t> write_index_{0};
  alignas(cache_line_size_) size_t read_index_cache_{0};
  alignas(cache_line_size_) std::atomic<size_t> read_index_{0};
  alignas(cache_line_size_) size_t write_index_cache_{0};

  // Padding to avoid adjacent allocations to share cache line with
  // writeIndexCache_
  char padding_data_[cache_line_size_ - sizeof(write_index_cache_)]{};
};
}  // namespace Parallel

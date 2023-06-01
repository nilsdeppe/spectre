// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <converse.h>
#include <memory>
#include <pup.h>

#include <mutex>

#include "Parallel/Spinlock.hpp"

namespace Parallel {

/*!
 * \ingroup ParallelGroup
 * \brief A typesafe wrapper for a lock for synchronization of shared resources
 * on a given node, with safe creation, destruction, and serialization.
 *
 * \details This structure is a thin wrapper around the charm `CmiNodeLock`, in
 * the <a href="https://charm.readthedocs.io/en/latest/converse/manual.html">
 * Converse library</a>. On construction, this class creates a Converse
 * nodelock, and frees the lock on destruction.
 *
 * \note If a locked NodeLock is serialized, it is deserialized as unlocked.
 */
class NodeLock {
 public:
  NodeLock();

  explicit NodeLock(CkMigrateMessage* /*message*/) {}

  NodeLock(const NodeLock&) = delete;
  NodeLock& operator=(const NodeLock&) = delete;
  NodeLock(NodeLock&& moved_lock);
  NodeLock& operator=(NodeLock&& moved_lock);
  ~NodeLock();

  void lock();

  bool try_lock();

  void unlock();

  void destroy();

  bool is_destroyed() { return nullptr == lock_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  // std::unique_ptr<CmiNodeLock> lock_;
  // std::unique_ptr<std::mutex> lock_;
  std::unique_ptr<Spinlock> lock_;
};
}  // namespace Parallel

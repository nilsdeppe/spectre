// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <charm++.h>
#include <iomanip>
#include <sstream>

#include "Utilities/System/ParallelInfo.hpp"

namespace sys {
int number_of_procs() {
#if defined(SPECTRE_USE_CHARM)
  return CkNumPes();
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1;
#else
#error "Must use either Charm++ or findus"
#endif
}

int my_proc() {
#if defined(SPECTRE_USE_CHARM)
  return CkMyPe();
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1;
#else
#error "Must use either Charm++ or findus"
#endif
}

int number_of_nodes() {
#if defined(SPECTRE_USE_CHARM)
  return CkNumNodes();
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1;
#else
#error "Must use either Charm++ or findus"
#endif
}

int my_node() {
#if defined(SPECTRE_USE_CHARM)
  return CkMyNode();
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1;
#else
#error "Must use either Charm++ or findus"
#endif
}

int procs_on_node([[maybe_unused]] const int node_index) {
#if defined(SPECTRE_USE_CHARM)
  return CkNodeSize(node_index);
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1;
#else
#error "Must use either Charm++ or findus"
#endif
}

int my_local_rank() {
#if defined(SPECTRE_USE_CHARM)
  return CkMyRank();
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1;
#else
#error "Must use either Charm++ or findus"
#endif
}

int first_proc_on_node([[maybe_unused]] const int node_index) {
#if defined(SPECTRE_USE_CHARM)
  return CkNodeFirst(node_index);
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1;
#else
#error "Must use either Charm++ or findus"
#endif
}

int node_of([[maybe_unused]] const int proc_index) {
#if defined(SPECTRE_USE_CHARM)
  return CkNodeOf(proc_index);
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1;
#else
#error "Must use either Charm++ or findus"
#endif
}

int local_rank_of([[maybe_unused]] const int proc_index) {
#if defined(SPECTRE_USE_CHARM)
  return CkRankOf(proc_index);
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1;
#else
#error "Must use either Charm++ or findus"
#endif
}

double wall_time() {
#if defined(SPECTRE_USE_CHARM)
  return CkWallTimer();
#elif defined(SPECTRE_USE_FINDUS)
  throw std::runtime_error{"Unimplemented"};
  return -1.0;
#else
#error "Must use either Charm++ or findus"
#endif
}

std::string pretty_wall_time(const double total_seconds) {
  // Subseconds don't really matter so just ignore them. This gives nice round
  // numbers.
  int total = static_cast<int>(total_seconds);
  const int day = total / (24 * 3600);

  total %= (24 * 3600);
  const int hour = total / 3600;

  total %= 3600;
  const int minutes = total / 60;

  total %= 60;
  const int seconds = total;

  std::stringstream ss{};
  ss << std::setfill('0');
  if (day > 0) {
    ss << std::setw(2) << day << "-";
  }

  // std::setw() isn't sticky so it has to be used for every insertion
  ss << std::setw(2) << hour << ":";
  ss << std::setw(2) << minutes << ":";
  ss << std::setw(2) << seconds;
  return ss.str();
}

std::string pretty_wall_time() { return pretty_wall_time(sys::wall_time()); }
}  // namespace sys

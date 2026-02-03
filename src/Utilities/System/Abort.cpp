// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/System/Abort.hpp"

#include <charm++.h>
#include <exception>
#include <string>

namespace sys {
void abort(const std::string& message) {
#if defined(SPECTRE_USE_CHARM)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
  CkPrintf("%s\n", message.c_str());
#pragma GCC diagnostic pop
  CkExit(1);
#else
  std::printf("%s\n", message.c_str());
#endif
  // the following call is never reached, but suppresses the warning that
  // a 'noreturn' function does return
  std::terminate();  // LCOV_EXCL_LINE
}
}  // namespace sys

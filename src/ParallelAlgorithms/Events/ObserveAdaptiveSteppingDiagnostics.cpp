// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveAdaptiveSteppingDiagnostics.hpp"

namespace Events {
#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID ObserveAdaptiveSteppingDiagnostics::my_PUP_ID = 0;  // NOLINT
#endif  // SPECTRE_USE_CHARM
}  // namespace Events

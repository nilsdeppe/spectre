// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/ParallelAlgorithms/EventsAndDenseTriggers/DenseTriggers/TestTrigger.hpp"

namespace TestHelpers::DenseTriggers {
#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID TestTrigger::my_PUP_ID = 0;  // NOLINT
#endif                                         // SPECTRE_USE_CHARM
}  // namespace TestHelpers::DenseTriggers

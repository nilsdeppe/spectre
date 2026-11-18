// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/EventsAndTriggers/LogicalTriggers.hpp"

namespace Triggers {
#if defined(SPECTRE_USE_CHARM)
PUP::able::PUP_ID Always::my_PUP_ID = 0;  // NOLINT
PUP::able::PUP_ID Not::my_PUP_ID = 0;  // NOLINT
PUP::able::PUP_ID And::my_PUP_ID = 0;  // NOLINT
PUP::able::PUP_ID Or::my_PUP_ID = 0;  // NOLINT
#endif                                // SPECTRE_USE_CHARM
}  // namespace Triggers

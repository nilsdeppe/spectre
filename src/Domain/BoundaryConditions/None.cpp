// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/BoundaryConditions/None.hpp"

#include <memory>

namespace domain::BoundaryConditions {
MarkAsNone::~MarkAsNone() = default;

bool is_none(const std::unique_ptr<BoundaryCondition>& boundary_condition) {
  return dynamic_cast<const MarkAsNone* const>(boundary_condition.get()) !=
         nullptr;
}
}  // namespace domain::BoundaryConditions

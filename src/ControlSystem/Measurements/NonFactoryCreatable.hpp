// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"

/*!
 * \brief Simple struct that changes a class from being factory creatable, to
 * non-factory creatable.
 *
 * It does this by defining and setting a static constexpr bool
 * `factory_creatable = false`.
 *
 * \note When using findus the wrapper is not registered separately for
 * serialization, so it is serialized and deserialized as the
 * `FactoryCreatableClass`. This is equivalent because the wrapper adds neither
 * data members nor virtual functions.
 *
 * \tparam FactoryCreatableClass Any class that is already factory creatable.
 */
template <typename FactoryCreatableClass>
struct NonFactoryCreatableWrapper : public FactoryCreatableClass {
  static constexpr bool factory_creatable = false;
  using FactoryCreatableClass::FactoryCreatableClass;

  using factory_creatable_class = FactoryCreatableClass;

  /// \cond
  WRAPPED_PUPable_decl_base_template(  // NOLINT
      NonFactoryCreatableWrapper, NonFactoryCreatableWrapper);
  /// \endcond
};

/// \cond
// NOLINTBEGIN
#if defined(SPECTRE_USE_CHARM)
template <typename FactoryCreatableClass>
PUP::able::PUP_ID NonFactoryCreatableWrapper<FactoryCreatableClass>::my_PUP_ID =
    0;
#endif  // SPECTRE_USE_CHARM
// NOLINTEND
/// \endcond

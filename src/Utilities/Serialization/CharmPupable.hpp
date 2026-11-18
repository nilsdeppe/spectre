// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macros to allow serialization of abstract template base classes

#pragma once

#include <pup.h>

#ifdef __GNUC__
#pragma GCC system_header
#endif

/*!
 * \ingroup ParallelGroup
 * \brief Mark derived classes as serializable
 *
 * Any class that inherits from an abstract base class where the base class is
 * not a template class must contain this macro if it is to be serialized.
 *
 * The macro also defines the constructor `className(CkMigrateMessage*)` that
 * Charm++ uses when unpacking. Therefore `className` must be the name of the
 * class without template arguments, and all base classes and data members
 * must be default constructible.
 */
#define WRAPPED_PUPable_decl_template(className)      \
  explicit className(CkMigrateMessage* /*unused*/) {} \
  PUPable_decl_template(SINGLE_ARG(className))  // NOLINT

/*!
 * \ingroup ParallelGroup
 * \brief Mark derived template classes as serializable
 *
 * Any class that inherits from an abstract base class where the base class is
 * a template class must contain this macro if it is to be serialized.
 *
 * The macro also defines the constructor `className(CkMigrateMessage*)` that
 * Charm++ uses when unpacking. Therefore `className` must be the name of the
 * class without template arguments, and all base classes and data members
 * must be default constructible.
 */
#define WRAPPED_PUPable_decl_base_template(baseClassName, className) \
  explicit className(CkMigrateMessage* /*unused*/) {}                \
  PUPable_decl_base_template(SINGLE_ARG(baseClassName), /* NOLINT */ \
                             SINGLE_ARG(className))     // NOLINT

/// Wraps the Charm++ macro, see the Charm++ documentation
#define WRAPPED_PUPable_abstract(className) \
  PUPable_abstract(SINGLE_ARG(className))  // NOLINT

/// \brief Used for the base class of a serializable class hierarchy to inherit
/// from.
#define SPECTRE_CHARM_PUPable(className) virtual PUP::able

/// \brief Only used for multi-inheritance auto-registration of derived classes
/// for serialization by findus. Empty in Charm++ mode.
///
/// \note findus supplies this macro directly.
#define FINDUS_OVERRIDE_SERIALIZATION_ID(Derived, Base)

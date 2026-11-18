// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macros to allow serialization of abstract template base classes

#pragma once

#include <pup.h>

#ifdef __GNUC__
#pragma GCC system_header
#endif

#if defined(SPECTRE_USE_CHARM)
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

/// \brief Used as the first entry of the member initializer list of
/// user-provided copy constructors in a serializable class hierarchy to
/// initialize the virtual `PUP::able` base class, e.g.
///
/// \code
/// Derived(const Derived& rhs)
///     : SPECTRE_CHARM_PUPable_COPY(rhs) Base(rhs), member_(rhs.member_) {}
/// \endcode
///
/// The trailing comma is part of the macro, so it must be followed by at least
/// one more initializer and must not be followed by a comma.
///
/// Empty in findus mode.
#define SPECTRE_CHARM_PUPable_COPY(rhs) PUP::able(rhs),

/// \brief Used by intermediate abstract classes that derive from a
/// serializable base class but should themselves also be usable as a
/// serializable base class (e.g. `LtsTimeStepper`, which derives from
/// `TimeStepper`). Must directly follow a base class in the base class list,
/// without a comma in between.
///
/// Empty in Charm++ mode.
#define SPECTRE_FINDUS_BASE(className)

/// \brief Used by derived classes to register themselves with the serializable
/// base class `baseName` when building with findus. Must directly follow a
/// base class in the base class list, without a comma in between, e.g.
///
/// \code
/// class Derived : public Base SPECTRE_FINDUS_DERIVED(Derived, Base) {
/// \endcode
///
/// `baseName` need not be a direct base class: it can be several levels up in
/// the hierarchy. A class that is serialized through several base classes uses
/// the macro once for each of them.
///
/// Empty in Charm++ mode.
#define SPECTRE_FINDUS_DERIVED(derivedName, baseName)

/// \brief Only used for multi-inheritance auto-registration of derived classes
/// for serialization by findus. Empty in Charm++ mode.
///
/// \note findus supplies this macro directly.
#define FINDUS_OVERRIDE_SERIALIZATION_ID(Derived, Base)

/// \brief Expands to `virtual` if findus is used. Empty for Charm++.
#define SPECTRE_FINDUS_VIRTUAL()

/// \brief Expands to `override` if Charm++ is used. Empty for findus.
#define SPECTRE_CHARM_OVERRIDE() override

#elif defined(SPECTRE_USE_FINDUS)

/// \brief Used for the base class of a serializable class hierarchy to inherit
/// from.
#define SPECTRE_CHARM_PUPable(className) \
  virtual findus::serialize::SerializableBase<className>

/// \brief Used as the first entry of the member initializer list of
/// user-provided copy constructors in a serializable class hierarchy to
/// initialize the virtual `PUP::able` base class, e.g.
///
/// \code
/// Derived(const Derived& rhs)
///     : SPECTRE_CHARM_PUPable_COPY(rhs) Base(rhs), member_(rhs.member_) {}
/// \endcode
///
/// The trailing comma is part of the macro, so it must be followed by at least
/// one more initializer and must not be followed by a comma.
///
/// Empty in findus mode.
#define SPECTRE_CHARM_PUPable_COPY(rhs)

/// \brief Used by intermediate abstract classes that derive from a
/// serializable base class but should themselves also be usable as a
/// serializable base class (e.g. `LtsTimeStepper`, which derives from
/// `TimeStepper`). Must directly follow a base class in the base class list,
/// without a comma in between.
///
/// Empty in Charm++ mode.
#define SPECTRE_FINDUS_BASE(className) \
  , public virtual findus::serialize::SerializableBase<className>

/// \brief Used by derived classes to register themselves with the serializable
/// base class `baseName` when building with findus. Must directly follow a
/// base class in the base class list, without a comma in between, e.g.
///
/// \code
/// class Derived : public Base SPECTRE_FINDUS_DERIVED(Derived, Base) {
/// \endcode
///
/// `baseName` need not be a direct base class: it can be several levels up in
/// the hierarchy. A class that is serialized through several base classes uses
/// the macro once for each of them.
///
/// Empty in Charm++ mode.
#define SPECTRE_FINDUS_DERIVED(derivedName, baseName) \
  , public virtual findus::serialize::SerializableDerived<derivedName, baseName>

/// \brief Interface with Charm++. Only declares an unused static member
/// function so that the macro can be followed by a semicolon.
#define WRAPPED_PUPable_decl_template(className) \
  static void unused_function_to_silence_compiler_warnings_for_charm_interop()

/// \brief Interface with Charm++. Only declares an unused static member
/// function so that the macro can be followed by a semicolon.
#define WRAPPED_PUPable_decl_base_template(baseClassName, className) \
  static void unused_function_to_silence_compiler_warnings_for_charm_interop()

/// \brief Interface with Charm++. Adds a virtual `pup` function to the base
/// class, so derived classes can call the `pup` function of the base class
/// with both Charm++ and findus.
///
/// A base class that declares its own `pup` function must guard this macro
/// with `#if defined(SPECTRE_USE_CHARM)`. Calls to `PUP::able::pup` must be
/// guarded the same way since `PUP::able` has no `pup` function when using
/// findus.
#define WRAPPED_PUPable_abstract(className) \
  virtual void pup(PUP::er& /*p*/) {}       \
  static void                               \
  unused_function_to_silence_compiler_warnings_for_charm_interop_abstract()

/// \brief Expands to `virtual` if findus is used. Empty for Charm++.
#define SPECTRE_FINDUS_VIRTUAL() virtual

/// \brief Expands to `override` if Charm++ is used. Empty for findus.
#define SPECTRE_CHARM_OVERRIDE()

#else
#error "Either SPECTRE_USE_CHARM or SPECTRE_USE_FINDUS must be defined."
#endif  // SPECTRE_USE_CHARM

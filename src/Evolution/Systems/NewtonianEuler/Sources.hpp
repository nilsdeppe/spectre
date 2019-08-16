// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler {

/*!
 * \brief Compute the source terms for the NewtonianEuler evolution
 * using a problem-specific source of type `SourceTermType`.
 *
 * \details Any source term type used by this class must hold `public` aliases
 * `sourced_variables` and `argument_tags`, which are `tmpl::list`s of the
 * variables whose equations of motion require a source term, and the arguments
 * required to compute those source terms, respectively. `SourceTermType` must
 * also hold a `public` `void` member function `apply` whose arguments are
 * `gsl::not_null` pointers to the variables storing the source terms, followed
 * by the arguments required to compute them.
 * See NewtonianEuler::Sources::UniformAcceleration for an example.
 *
 * While most of physically relevant source terms for the Newtonian Euler
 * equations do not add a source term for the mass density, this class allows
 * for problems that source any set of conserved variables
 * (at least one variable is required).
 */
template <typename SourceTermType>
struct ComputeSources {
 private:
  template <typename SourcedVarsTagList, typename ArgTagsList>
  struct apply_helper;

  template <typename... SourcedVarsTags, typename... ArgsTags>
  struct apply_helper<tmpl::list<SourcedVarsTags...>, tmpl::list<ArgsTags...>> {
    static void function(
        const gsl::not_null<db::item_type<SourcedVarsTags>*>... sourced_vars,
        const db::item_type<Tags::SourceTerm<SourceTermType>>& source,
        const db::item_type<ArgsTags>&... args) noexcept {
      source.apply(sourced_vars..., args...);
    }
  };

 public:
  using return_tags =
      db::wrap_tags_in<::Tags::Source,
                       typename SourceTermType::sourced_variables>;

  using argument_tags = tmpl::push_front<typename SourceTermType::argument_tags,
                                         Tags::SourceTerm<SourceTermType>>;

  template <class... Args>
  static void apply(const Args&... args) noexcept {
    apply_helper<typename SourceTermType::sourced_variables,
                 typename SourceTermType::argument_tags>::function(args...);
  }
};

}  // namespace NewtonianEuler

// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/KeplerianPotential.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace NewtonianEuler {
namespace Solutions {

/*!
 * \brief Cold 2D Keplerian disk
 *
 * This class implements a 2D Keplerian disk orbiting a central object,
 * as studied in \cite Schaal2015 and references therein. The ambient fluid of
 * the disk has negligible mass density and pressure. Every fluid element
 * in the disk rotates on a Keplerian orbit, and the disk itself is confined
 * within two radii. An external central potential is imposed on the flow,
 * so that self-gravity is ignored. Analytically, the initial state should
 * be preserved in time. Numerically, however,
 * the evolution might be prone to numerical instabilities seeded by
 * the lack of pressure support, as well as by shearing flows induced by
 * differential rotation in the disk, which would eventually disrupt the latter.
 * Therefore, the evolution of the initial data implemented here tests
 * the capability of the code to sustain angular motion with differential
 * rotation on a grid that does not necessarily take advantage of the
 * geometry of the problem.
 *
 * Adopting Cartesian coordinates \f$(x, y)\f$, the mass density is initialized
 * as
 *
 * \f{align*}
 * \rho(x', y') &=
 * \begin{cases}
 * \rho_\text{amb}, & r' \leq r_\text{inn, -}\\
 * \rho_\text{amb} + \left(\rho_\text{disk} - \rho_\text{amb}\right)
 * \dfrac{r' - r_\text{inn,-}}{\Delta r}, &
 * r_\text{inn, -} < r' \leq r_\text{inn, +}\\
 * \rho_\text{disk}, & r_\text{inn, +} < r' \leq r_\text{out, -}\\
 * \rho_\text{disk} + \left(\rho_\text{amb} - \rho_\text{disk}\right)
 * \dfrac{r' - r_\text{out,-}}{\Delta r}, &
 * r_\text{out, -} < r' \leq r_\text{out, +}\\
 * \rho_\text{amb}, & r' > r_\text{out, +},
 * \end{cases}
 * \f}
 *
 * where \f$\rho_\text{amb}\f$ is the ambient mass density,
 * \f$\rho_\text{disk}\f$ is the mass density in the disk,
 * \f$(x', y') = (x - x_\text{c}, y - y_\text{c})\f$ are coordinates
 * measured relative to the disk center, which is located at
 * \f$(x_\text{c}, y_\text{c})\f$,
 * \f$r' = \sqrt(x'^2 + y'^2)\f$ is the radial coodinate from the disk center,
 * and \f$r_\text{inn, $\pm$} = r_\text{inn} \pm \Delta r/2\f$ and
 * \f$r_\text{out, $\pm$} = r_\text{out} \pm \Delta r/2\f$, where
 * \f$r_\text{inn}\f$ and \f$r_\text{out}\f$ are the inner and outer radii
 * of the disk, respectively, and \f$\Delta r\f$ is a small number
 * representing the width of transition regions between disk and ambient
 * fluid.
 *
 * The velocity is initialized as
 *
 * \f{align*}
 * v_x(x', y') &=
 * \begin{cases}
 * - \dfrac{y'}{r'^{3/2}}, &
 * r_\text{inn} - 2\,\Delta r \leq r' < r_\text{out} + 2\,\Delta r\\
 * 0, & \text{otherwise}
 * \end{cases}\\
 * v_y(x', y') &=
 * \begin{cases}
 * \dfrac{x'}{r'^{3/2}}, &
 * r_\text{inn} - 2\,\Delta r \leq r' < r_\text{out} + 2\,\Delta r\\
 * 0, & \text{otherwise}.
 * \end{cases}
 * \f}
 *
 * Finally, the pressure is initialized to its ambient value,
 * \f$p = p_\text{amb}\f$, and the system is closed with an ideal equation
 * of state \f$p = (\gamma - 1)\rho\epsilon\f$.
 *
 * \note Although this test is proposed as a 2D problem, here one can also
 * run it in 3D, in which case
 * \f$r' = \sqrt{(x - x_\text{c})^2 + (y - y_\text{c})^2
 * + (z - z_\text{c})^2}\f$, and the \f$z-\f$component of the velocity will be
 * initialized to zero.
 *
 * This test is typically run on a 2D grid with origin at the disk center, using
 * periodic boundary conditions. The source term, a Keplerian potential,
 * accepts a parameter to smooth such potential near the central object
 * (see Sources::KeplerianPotential for details.). Typical parameters
 * for a simulation are:
 *
 * - AdiabaticIndex: 5/3
 * - AmbientMassDensity: \f$10^{-5}\f$
 * - AmbientPressure: \f$10^{-5}\f$
 * - DiskCenter: [3.0, 3.0]
 * - DiskMassDensity: 1.0
 * - DiskInnerRadius: 0.5
 * - DiskOuterRadius: 2.0
 * - SmoothingParameter: 0.25
 * - TransitionWidth: 0.1
 */
template <size_t Dim>
class KeplerianDisk : public MarkAsAnalyticSolution {
  template <typename DataType>
  struct IntermediateVariables;

 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<false>;
  using source_term_type = Sources::KeplerianPotential<Dim>;

  /// The adiabatic index of the fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {"The adiabatic index of the fluid."};
  };

  /// The ambient mass density
  struct AmbientMassDensity {
    using type = double;
    static constexpr OptionString help = {"The ambient mass density."};
  };

  /// The ambient pressure
  struct AmbientPressure {
    using type = double;
    static constexpr OptionString help = {"The ambient pressure."};
  };

  /// The position of the center of the disk.
  struct DiskCenter {
    using type = std::array<double, Dim>;
    static constexpr OptionString help = {
        "The coordinates of the center of the disk."};
  };

  /// The mass density in the disk
  struct DiskMassDensity {
    using type = double;
    static constexpr OptionString help = {"The mass density in the disk."};
  };

  /// The inner radius of the disk
  struct DiskInnerRadius {
    using type = double;
    static constexpr OptionString help = {"The inner radius of the disk."};
  };

  /// The outer radius of the disk
  struct DiskOuterRadius {
    using type = double;
    static constexpr OptionString help = {"The outer radius of the disk."};
  };

  /// Smoothing parameter for the source term
  struct SmoothingParameter {
    using type = double;
    static constexpr OptionString help = {
        "The parameter smoothing the potential."};
  };

  /// The width of the transition regions at both radii of the disk
  struct TransitionWidth {
    using type = double;
    static constexpr OptionString help = {
        "The width of the transition regions."};
  };

  using options =
      tmpl::list<AdiabaticIndex, AmbientMassDensity, AmbientPressure,
                 DiskCenter, DiskMassDensity, DiskInnerRadius, DiskOuterRadius,
                 SmoothingParameter, TransitionWidth>;

  static constexpr OptionString help = {
      "Keplerian disk. Can be used in 2 and 3D."};

  KeplerianDisk() = default;
  KeplerianDisk(const KeplerianDisk& /*rhs*/) = delete;
  KeplerianDisk& operator=(const KeplerianDisk& /*rhs*/) = delete;
  KeplerianDisk(KeplerianDisk&& /*rhs*/) noexcept = default;
  KeplerianDisk& operator=(KeplerianDisk&& /*rhs*/) noexcept = default;
  ~KeplerianDisk() = default;

  KeplerianDisk(double adiabatic_index, double ambient_mass_density,
                double ambient_pressure,
                const std::array<double, Dim>& disk_center,
                double disk_mass_density, double disk_inner_radius,
                double disk_outer_radius, double smoothing_parameter,
                double transition_width);

  /// Retrieve a collection of hydrodynamic variables at position x and time t
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x, double /*t*/,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    IntermediateVariables<DataType> vars(x, disk_center_);
    return {tuples::get<Tags>(variables(tmpl::list<Tags>{}, vars))...};
  }

  const EquationsOfState::IdealFluid<false>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

  const Sources::KeplerianPotential<Dim>& source_term() const noexcept {
    return source_term_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  // @{
  /// Retrieve hydro variable at `(x, t)`
  template <typename DataType>
  auto variables(tmpl::list<Tags::MassDensity<DataType>> /*meta*/,
                 const IntermediateVariables<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<Tags::MassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/,
      const IntermediateVariables<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>;

  template <typename DataType>
  auto variables(tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/,
                 const IntermediateVariables<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(tmpl::list<Tags::Pressure<DataType>> /*meta*/,
                 const IntermediateVariables<DataType>& vars) const noexcept
      -> tuples::TaggedTuple<Tags::Pressure<DataType>>;
  // @}

  // Intermediate variables needed to compute the primitives
  template <typename DataType>
  struct IntermediateVariables {
    IntermediateVariables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                          const std::array<double, Dim>& disk_center) noexcept;
    DataType x_prime{};
    DataType y_prime{};
    DataType r_prime{};
  };

  template <size_t SpatialDim>
  friend bool
  operator==(  // NOLINT (clang-tidy: readability-redundant-declaration)
      const KeplerianDisk<SpatialDim>& lhs,
      const KeplerianDisk<SpatialDim>& rhs) noexcept;

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double ambient_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double ambient_pressure_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, Dim> disk_center_ =
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN());
  double disk_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double disk_inner_radius_ = std::numeric_limits<double>::signaling_NaN();
  double disk_outer_radius_ = std::numeric_limits<double>::signaling_NaN();
  double smoothing_parameter_ = std::numeric_limits<double>::signaling_NaN();
  double transition_width_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::IdealFluid<false> equation_of_state_{};
  Sources::KeplerianPotential<Dim> source_term_{};
};

template <size_t Dim>
bool operator!=(const KeplerianDisk<Dim>& lhs,
                const KeplerianDisk<Dim>& rhs) noexcept;

}  // namespace Solutions
}  // namespace NewtonianEuler
